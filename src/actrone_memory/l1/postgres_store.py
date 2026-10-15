"""Postgres hot-tier (L1) store.

Pairs with :class:`~actrone_memory.l2.pgvector_store.PgVectorStore` so both memory tiers can
live in a Postgres you already run, instead of adding Redis and a vector database. Requires
``pip install actrone-memory[pgvector]``.

Redis gives TTL and list trimming for free; Postgres does not, so this store implements both
explicitly:

* **Expiry** is an ``expires_at`` column. Reads filter on it and writes opportunistically
  delete the session's expired rows, so an abandoned session cannot accumulate forever even
  with no background job. There is no ``pg_cron`` requirement.
* **Retention** is enforced on append by deleting everything older than the newest
  ``max_turns`` rows for that session, mirroring Redis ``LTRIM``.

Ordering uses a ``bigserial`` sequence rather than the timestamp, because two turns appended
in the same clock tick would otherwise have no defined order, and the whole prompt depends on
recent turns coming back oldest-first.
"""

from __future__ import annotations

import json
from datetime import timedelta
from typing import TYPE_CHECKING

from actrone_memory.exceptions import StoreConnectionError
from actrone_memory.logging import bind_logger
from actrone_memory.metrics import track_store_op
from actrone_memory.models import SessionMetadata, Turn

if TYPE_CHECKING:  # pragma: no cover - typing only
    import asyncpg

log = bind_logger(__name__)

_DEFAULT_TABLE = "agent_turns"
# Every `# nosec B608` marker below is on a statement whose only interpolations are identifiers
# validated here (and, in the pgvector store, the validated integer vector width); every value
# is a bound $n parameter. Bandit cannot see that validation, so each one is marked.
_SAFE_IDENTIFIER = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_")


def _validate_identifier(name: str, field: str) -> str:
    if not name or not set(name) <= _SAFE_IDENTIFIER:
        raise ValueError(
            f"{field} must be a plain identifier (letters, digits, underscore), got {name!r}"
        )
    return name


class PostgresStore:
    """Recent conversation turns in Postgres.

    Satisfies the :class:`~actrone_memory.protocols.L1Store` protocol and passes
    :func:`actrone_memory.testing.check_l1_store`. Construct with :meth:`from_dsn`.
    """

    def __init__(
        self,
        pool: asyncpg.Pool,
        table: str,
        session_ttl_hours: int,
        max_turns: int,
        *,
        owns_pool: bool = True,
    ) -> None:
        self._pool = pool
        self._table = _validate_identifier(table, "table")
        self._locks_table = f"{self._table}_locks"
        self._ttl_seconds = int(timedelta(hours=session_ttl_hours).total_seconds())
        self._max_turns = max_turns
        self._owns_pool = owns_pool

    # ------------------------------------------------------------------
    # Schema bootstrap
    # ------------------------------------------------------------------

    async def ensure_schema(self) -> None:
        """Create the turns table, the summary-lock table and their indexes."""
        try:
            async with self._pool.acquire() as conn:
                await conn.execute(
                    f"""
                    CREATE TABLE IF NOT EXISTS {self._table} (
                        seq         bigserial PRIMARY KEY,
                        id          uuid        NOT NULL,
                        agent_id    text        NOT NULL,
                        session_id  text        NOT NULL,
                        turn        jsonb       NOT NULL,
                        created_at  timestamptz NOT NULL DEFAULT now(),
                        expires_at  timestamptz NOT NULL
                    )
                    """
                )
                # Every read is (agent_id, session_id) ordered by seq, so index exactly that.
                await conn.execute(
                    f"CREATE INDEX IF NOT EXISTS {self._table}_session_idx "
                    f"ON {self._table} (agent_id, session_id, seq)"
                )
                await conn.execute(
                    f"CREATE INDEX IF NOT EXISTS {self._table}_expiry_idx "
                    f"ON {self._table} (expires_at)"
                )
                await conn.execute(
                    f"""
                    CREATE TABLE IF NOT EXISTS {self._locks_table} (
                        agent_id   text        NOT NULL,
                        session_id text        NOT NULL,
                        expires_at timestamptz NOT NULL,
                        PRIMARY KEY (agent_id, session_id)
                    )
                    """
                )
        except Exception as exc:
            raise StoreConnectionError("Postgres", exc) from exc

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @track_store_op("l1", "append_turn")
    async def append_turn(self, agent_id: str, session_id: str, turn: Turn) -> None:
        """Append a turn, then enforce expiry and the retention cap for this session.

        All three statements run in one transaction, so a reader never observes a session
        that has been trimmed but not yet appended to.
        """
        try:
            async with self._pool.acquire() as conn, conn.transaction():
                # Expiry is computed by the server, not here. Deriving it from the
                # application clock and then comparing it against Postgres `now()` makes
                # retention depend on the two clocks agreeing, and they routinely do not.
                await conn.execute(
                    f"""
                    INSERT INTO {self._table}
                        (id, agent_id, session_id, turn, created_at, expires_at)
                    VALUES ($1, $2, $3, $4::jsonb, $5, now() + make_interval(secs => $6))
                    """,  # nosec B608
                    turn.id,
                    agent_id,
                    session_id,
                    turn.model_dump_json(),
                    turn.timestamp,
                    self._ttl_seconds,
                )
                # Opportunistic expiry, so no cron job is required for correctness.
                await conn.execute(
                    f"DELETE FROM {self._table} "
                    f"WHERE agent_id = $1 AND session_id = $2 AND expires_at <= now()",  # nosec B608
                    agent_id,
                    session_id,
                )
                # Retention cap, the equivalent of Redis LTRIM.
                await conn.execute(
                    f"""
                    DELETE FROM {self._table}
                    WHERE agent_id = $1 AND session_id = $2 AND seq NOT IN (
                        SELECT seq FROM {self._table}
                        WHERE agent_id = $1 AND session_id = $2
                        ORDER BY seq DESC
                        LIMIT $3
                    )
                    """,  # nosec B608
                    agent_id,
                    session_id,
                    self._max_turns,
                )
        except Exception as exc:
            log.error(
                "postgres.append_turn.failed",
                agent_id=agent_id,
                session_id=session_id,
                error=str(exc),
            )
            raise StoreConnectionError("Postgres", exc) from exc

    @track_store_op("l1", "get_recent_turns")
    async def get_recent_turns(
        self, agent_id: str, session_id: str, n: int | None = None
    ) -> list[Turn]:
        """Return up to ``n`` most recent unexpired turns, oldest first."""
        limit = n if n is not None else self._max_turns
        try:
            async with self._pool.acquire() as conn:
                # Take the newest `limit` rows, then flip to chronological order: the prompt
                # is assembled oldest-first, but "recent" has to window from the end.
                rows = await conn.fetch(
                    f"""
                    SELECT turn FROM (
                        SELECT turn, seq FROM {self._table}
                        WHERE agent_id = $1 AND session_id = $2 AND expires_at > now()
                        ORDER BY seq DESC
                        LIMIT $3
                    ) AS recent
                    ORDER BY seq ASC
                    """,  # nosec B608
                    agent_id,
                    session_id,
                    limit,
                )
        except Exception as exc:
            log.error(
                "postgres.get_turns.failed",
                agent_id=agent_id,
                session_id=session_id,
                error=str(exc),
            )
            raise StoreConnectionError("Postgres", exc) from exc

        return [Turn.model_validate(json.loads(row["turn"])) for row in rows]

    @track_store_op("l1", "turn_count")
    async def turn_count(self, agent_id: str, session_id: str) -> int:
        """Count unexpired turns retained for the session."""
        try:
            async with self._pool.acquire() as conn:
                count = await conn.fetchval(
                    f"SELECT count(*) FROM {self._table} "
                    f"WHERE agent_id = $1 AND session_id = $2 AND expires_at > now()",  # nosec B608
                    agent_id,
                    session_id,
                )
        except Exception as exc:
            raise StoreConnectionError("Postgres", exc) from exc
        return int(count or 0)

    @track_store_op("l1", "clear_session")
    async def clear_session(self, agent_id: str, session_id: str) -> None:
        """Delete the session's turns and release any summary lock it holds."""
        try:
            async with self._pool.acquire() as conn, conn.transaction():
                await conn.execute(
                    f"DELETE FROM {self._table} WHERE agent_id = $1 AND session_id = $2",  # nosec B608
                    agent_id,
                    session_id,
                )
                await conn.execute(
                    f"DELETE FROM {self._locks_table} WHERE agent_id = $1 AND session_id = $2",  # nosec B608
                    agent_id,
                    session_id,
                )
        except Exception as exc:
            raise StoreConnectionError("Postgres", exc) from exc

    @track_store_op("l1", "get_session_metadata")
    async def get_session_metadata(self, agent_id: str, session_id: str) -> SessionMetadata | None:
        """Session stats, or ``None`` when the session has no unexpired turns."""
        try:
            async with self._pool.acquire() as conn:
                row = await conn.fetchrow(
                    f"""
                    SELECT count(*) AS turn_count,
                           min(created_at) AS created_at,
                           max(created_at) AS last_active
                    FROM {self._table}
                    WHERE agent_id = $1 AND session_id = $2 AND expires_at > now()
                    """,  # nosec B608
                    agent_id,
                    session_id,
                )
        except Exception as exc:
            raise StoreConnectionError("Postgres", exc) from exc

        if row is None or not row["turn_count"]:
            return None
        return SessionMetadata(
            agent_id=agent_id,
            session_id=session_id,
            turn_count=int(row["turn_count"]),
            created_at=row["created_at"],
            last_active=row["last_active"],
        )

    @track_store_op("l1", "try_acquire_summary_lock")
    async def try_acquire_summary_lock(
        self, agent_id: str, session_id: str, ttl_seconds: int
    ) -> bool:
        """Claim the right to summarise this session, fleet-wide.

        One statement, so the check and the claim cannot interleave between processes: the
        upsert only overwrites a row whose window has already elapsed, and ``RETURNING`` is
        empty for the losers.
        """
        try:
            async with self._pool.acquire() as conn:
                # Server-side expiry, for the same reason as append_turn: the lock must not
                # depend on the application clock matching the database clock.
                won = await conn.fetchval(
                    f"""
                    INSERT INTO {self._locks_table} (agent_id, session_id, expires_at)
                    VALUES ($1, $2, now() + make_interval(secs => $3))
                    ON CONFLICT (agent_id, session_id) DO UPDATE
                        SET expires_at = EXCLUDED.expires_at
                        WHERE {self._locks_table}.expires_at <= now()
                    RETURNING 1
                    """,  # nosec B608
                    agent_id,
                    session_id,
                    ttl_seconds,
                )
        except Exception as exc:
            raise StoreConnectionError("Postgres", exc) from exc
        return won is not None

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    async def from_dsn(
        cls,
        dsn: str,
        *,
        table: str = _DEFAULT_TABLE,
        session_ttl_hours: int = 24,
        max_turns: int = 50,
        min_pool_size: int = 1,
        max_pool_size: int = 10,
        command_timeout: float = 10.0,
    ) -> PostgresStore:
        """Connect to Postgres, ensure the schema, and return a ready store."""
        try:
            import asyncpg
        except ImportError as exc:
            raise StoreConnectionError(
                "Postgres",
                ImportError(
                    "PostgresStore requires the pgvector extra: "
                    "pip install actrone-memory[pgvector]"
                ),
            ) from exc

        _validate_identifier(table, "table")
        try:
            pool = await asyncpg.create_pool(
                dsn,
                min_size=min_pool_size,
                max_size=max_pool_size,
                command_timeout=command_timeout,
            )
        except Exception as exc:
            raise StoreConnectionError("Postgres", exc) from exc
        if pool is None:  # pragma: no cover - asyncpg returns None only on misuse
            raise StoreConnectionError("Postgres", RuntimeError("asyncpg returned no pool"))

        store = cls(pool, table, session_ttl_hours, max_turns)
        try:
            await store.ensure_schema()
        except BaseException:
            await pool.close()
            raise
        log.info("postgres.l1.ready", table=table, max_turns=max_turns)
        return store

    @classmethod
    def from_pool(
        cls,
        pool: asyncpg.Pool,
        *,
        table: str = _DEFAULT_TABLE,
        session_ttl_hours: int = 24,
        max_turns: int = 50,
    ) -> PostgresStore:
        """Wrap a pool the application already owns.

        :meth:`close` leaves a borrowed pool open, and you must call :meth:`ensure_schema`
        yourself once at startup. Share one pool with
        :meth:`~actrone_memory.l2.pgvector_store.PgVectorStore.from_pool` to run both tiers
        on a single Postgres connection pool.
        """
        return cls(pool, table, session_ttl_hours, max_turns, owns_pool=False)

    async def close(self) -> None:
        """Close the connection pool, unless it was supplied by the application."""
        if self._owns_pool:
            await self._pool.close()
