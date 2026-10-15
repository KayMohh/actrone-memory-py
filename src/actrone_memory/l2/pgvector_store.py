"""Postgres + pgvector L2 store.

The point of this adapter is "no new infrastructure": most teams already run Postgres, so
enabling the durable long-term tier becomes a migration rather than a new service to operate.
Requires the `pgvector <https://github.com/pgvector/pgvector>`_ extension and
``pip install actrone-memory[pgvector]``.

Ranking reuses the same :func:`~actrone_memory.retrieval.hybrid_rank` fusion as the in-process
and Qdrant stores, so recall ordering is identical across all three backends given the same
candidates. Postgres returns the dense-threshold-admitted rows and the shared pure-Python
fusion re-ranks them, rather than reimplementing BM25 in SQL and drifting from the others.

Vector width is fixed at table-creation time by pgvector, so ``dimensions`` must match the
embedder. Changing embedders means a new table or an ``ALTER``, exactly as with Qdrant.
"""

from __future__ import annotations

import json
from datetime import UTC
from typing import TYPE_CHECKING, Any

from actrone_memory.exceptions import MemoryNotFoundError, StoreConnectionError
from actrone_memory.logging import bind_logger
from actrone_memory.metrics import track_store_op
from actrone_memory.models import ContentType, MemoryEntry

if TYPE_CHECKING:  # pragma: no cover - typing only
    import asyncpg

log = bind_logger(__name__)

_DEFAULT_TABLE = "agent_memories"

# Identifiers cannot be bound as parameters, so the table name is interpolated into DDL and
# queries. Restrict it to a conservative charset instead of trusting the caller.
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


def _to_vector_literal(embedding: list[float]) -> str:
    """pgvector accepts its text form, ``[1,2,3]``, which avoids a codec registration."""
    return "[" + ",".join(repr(float(value)) for value in embedding) + "]"


class PgVectorStore:
    """Long-term semantic memory backed by Postgres + pgvector.

    Satisfies the :class:`~actrone_memory.protocols.L2Store` protocol and passes
    :func:`actrone_memory.testing.check_l2_store`. Construct with :meth:`from_dsn`.
    """

    def __init__(
        self,
        pool: asyncpg.Pool,
        table: str,
        dimensions: int,
        relevance_weight: float,
        recency_weight: float,
        *,
        owns_pool: bool = True,
    ) -> None:
        self._pool = pool
        self._table = _validate_identifier(table, "table")
        # Interpolated into the table DDL as vector(<n>), so it must be a real positive integer.
        # bool is excluded explicitly because it is an int subclass.
        if isinstance(dimensions, bool) or not isinstance(dimensions, int) or dimensions <= 0:
            raise ValueError(f"dimensions must be a positive integer, got {dimensions!r}")
        self._dimensions = dimensions
        self._relevance_w = relevance_weight
        self._recency_w = recency_weight
        # A pool handed in by the application must outlive this store, so close() leaves it.
        self._owns_pool = owns_pool

    # ------------------------------------------------------------------
    # Schema bootstrap
    # ------------------------------------------------------------------

    async def ensure_schema(self) -> None:
        """Create the extension, table and indexes if they do not exist.

        Indexes: an HNSW index on the vector for cosine distance, plus a btree on
        ``agent_id``. Every read filters by ``agent_id``, so without the second one Postgres
        scans the whole table per query once the row count grows.
        """
        try:
            async with self._pool.acquire() as conn:
                await conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
                await conn.execute(
                    f"""
                    CREATE TABLE IF NOT EXISTS {self._table} (
                        id              uuid PRIMARY KEY,
                        agent_id        text        NOT NULL,
                        session_id      text        NOT NULL,
                        content         text        NOT NULL,
                        content_type    text        NOT NULL,
                        embedding       vector({self._dimensions}) NOT NULL,
                        importance_score double precision NOT NULL DEFAULT 0.5,
                        topic_tags      jsonb       NOT NULL DEFAULT '[]'::jsonb,
                        token_count     integer     NOT NULL DEFAULT 0,
                        timestamp       timestamptz NOT NULL DEFAULT now(),
                        source_turn_ids jsonb       NOT NULL DEFAULT '[]'::jsonb,
                        source          text        NOT NULL DEFAULT 'unknown',
                        sensitivity     text        NOT NULL DEFAULT 'none'
                    )
                    """
                )
                await conn.execute(
                    f"CREATE INDEX IF NOT EXISTS {self._table}_agent_id_idx "
                    f"ON {self._table} (agent_id)"
                )
                # HNSW needs pgvector >= 0.5. Fall back to a plain scan rather than failing
                # startup on an older extension build: correctness does not depend on it.
                try:
                    await conn.execute(
                        f"CREATE INDEX IF NOT EXISTS {self._table}_embedding_idx "
                        f"ON {self._table} USING hnsw (embedding vector_cosine_ops)"
                    )
                except Exception as exc:  # noqa: BLE001 - index is an optimisation
                    log.warning(
                        "pgvector.hnsw_index.skipped",
                        table=self._table,
                        reason=str(exc),
                    )
        except Exception as exc:
            raise StoreConnectionError("Postgres", exc) from exc

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    @track_store_op("l2", "upsert")
    async def upsert(self, entry: MemoryEntry) -> None:
        """Insert or replace a memory by id. The entry must carry an embedding."""
        if entry.embedding is None:
            raise ValueError(f"MemoryEntry {entry.id} has no embedding, embed before upserting.")
        if len(entry.embedding) != self._dimensions:
            raise ValueError(
                f"MemoryEntry {entry.id} has {len(entry.embedding)} dimensions, but this "
                f"store was created for {self._dimensions}. Recreate the table or use the "
                "embedder it was sized for."
            )

        try:
            async with self._pool.acquire() as conn:
                await conn.execute(
                    f"""
                    INSERT INTO {self._table} (
                        id, agent_id, session_id, content, content_type, embedding,
                        importance_score, topic_tags, token_count, timestamp,
                        source_turn_ids, source, sensitivity
                    ) VALUES ($1, $2, $3, $4, $5, $6::vector, $7, $8::jsonb, $9, $10,
                              $11::jsonb, $12, $13)
                    ON CONFLICT (id) DO UPDATE SET
                        agent_id         = EXCLUDED.agent_id,
                        session_id       = EXCLUDED.session_id,
                        content          = EXCLUDED.content,
                        content_type     = EXCLUDED.content_type,
                        embedding        = EXCLUDED.embedding,
                        importance_score = EXCLUDED.importance_score,
                        topic_tags       = EXCLUDED.topic_tags,
                        token_count      = EXCLUDED.token_count,
                        timestamp        = EXCLUDED.timestamp,
                        source_turn_ids  = EXCLUDED.source_turn_ids,
                        source           = EXCLUDED.source,
                        sensitivity      = EXCLUDED.sensitivity
                    """,  # nosec B608
                    entry.id,
                    entry.agent_id,
                    entry.session_id,
                    entry.content,
                    entry.content_type,
                    _to_vector_literal(entry.embedding),
                    entry.importance_score,
                    json.dumps(entry.topic_tags),
                    entry.token_count,
                    entry.timestamp,
                    json.dumps(entry.source_turn_ids),
                    entry.source,
                    entry.sensitivity,
                )
        except Exception as exc:
            log.error("pgvector.upsert.failed", entry_id=entry.id, error=str(exc))
            raise StoreConnectionError("Postgres", exc) from exc

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    @track_store_op("l2", "search")
    async def search(
        self,
        agent_id: str,
        query_embedding: list[float],
        threshold: float,
        limit: int = 20,
        content_types: list[ContentType] | None = None,
        query_text: str | None = None,
    ) -> list[MemoryEntry]:
        """Hybrid semantic search: dense recall in Postgres, shared fusion in Python.

        Postgres returns candidates whose cosine similarity clears ``threshold`` (over-fetched
        so the re-rank has something to work with), then the shared fusion applies BM25 and
        recency when ``query_text`` is given. Only memories at or above ``threshold`` are ever
        returned, matching the other backends.

        Complexity: one indexed query, then O(n log n) over the over-fetched candidates.
        """
        from actrone_memory.retrieval import hybrid_rank

        # `1 - (embedding <=> query)` is cosine similarity, since <=> is cosine distance.
        sql = f"""
            SELECT id, agent_id, session_id, content, content_type, importance_score,
                   topic_tags, token_count, timestamp, source_turn_ids, source, sensitivity,
                   embedding::text AS embedding_text,
                   1 - (embedding <=> $1::vector) AS similarity
            FROM {self._table}
            WHERE agent_id = $2
              AND 1 - (embedding <=> $1::vector) >= $3
        """  # nosec B608
        params: list[Any] = [_to_vector_literal(query_embedding), agent_id, threshold]
        if content_types:
            sql += " AND content_type = ANY($4::text[])"
            params.append(list(content_types))
            sql += " ORDER BY embedding <=> $1::vector LIMIT $5"
        else:
            sql += " ORDER BY embedding <=> $1::vector LIMIT $4"
        # Over-fetch so the fusion can reorder rather than being handed a pre-truncated set.
        params.append(max(limit * 3, limit))

        try:
            async with self._pool.acquire() as conn:
                rows = await conn.fetch(sql, *params)
        except Exception as exc:
            raise StoreConnectionError("Postgres", exc) from exc

        entries = [self._row_to_entry(row) for row in rows]
        if not entries:
            return []

        return hybrid_rank(
            entries,
            query_embedding,
            query_text,
            threshold=threshold,
            relevance_weight=self._relevance_w,
            recency_weight=self._recency_w,
            limit=limit,
        )

    def _row_to_entry(self, row: Any) -> MemoryEntry:  # noqa: ANN401 - asyncpg Record
        timestamp = row["timestamp"]
        if timestamp.tzinfo is None:  # pragma: no cover - timestamptz is tz-aware
            timestamp = timestamp.replace(tzinfo=UTC)
        return MemoryEntry(
            id=str(row["id"]),
            agent_id=row["agent_id"],
            session_id=row["session_id"],
            content=row["content"],
            content_type=row["content_type"],
            # The fusion recomputes cosine locally, so the vector has to come back with it.
            embedding=json.loads(row["embedding_text"]),
            importance_score=row["importance_score"],
            topic_tags=json.loads(row["topic_tags"]),
            token_count=row["token_count"],
            timestamp=timestamp,
            source_turn_ids=json.loads(row["source_turn_ids"]),
            source=row["source"],
            sensitivity=row["sensitivity"],
        )

    # ------------------------------------------------------------------
    # Delete
    # ------------------------------------------------------------------

    @track_store_op("l2", "delete")
    async def delete(self, memory_id: str) -> None:
        """Delete a memory by id. Raises :class:`MemoryNotFoundError` when absent."""
        try:
            async with self._pool.acquire() as conn:
                status = await conn.execute(
                    f"DELETE FROM {self._table} WHERE id = $1",  # nosec B608
                    memory_id,
                )
        except Exception as exc:
            raise StoreConnectionError("Postgres", exc) from exc

        # asyncpg returns the command tag, e.g. "DELETE 1", so a 0 means nothing matched.
        if status.strip().endswith(" 0"):
            raise MemoryNotFoundError(memory_id)

    @track_store_op("l2", "delete_agent_memories")
    async def delete_agent_memories(self, agent_id: str) -> None:
        """Delete every memory for an agent. Irreversible."""
        try:
            async with self._pool.acquire() as conn:
                await conn.execute(
                    f"DELETE FROM {self._table} WHERE agent_id = $1",  # nosec B608
                    agent_id,
                )
        except Exception as exc:
            raise StoreConnectionError("Postgres", exc) from exc

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    async def from_dsn(
        cls,
        dsn: str,
        *,
        table: str = _DEFAULT_TABLE,
        dimensions: int = 1536,
        relevance_weight: float = 0.7,
        recency_weight: float = 0.3,
        min_pool_size: int = 1,
        max_pool_size: int = 10,
        command_timeout: float = 10.0,
    ) -> PgVectorStore:
        """Connect to Postgres, ensure the schema, and return a ready store.

        Args:
            dsn: ``postgresql://user:pass@host:5432/dbname``.
            table: Table to store memories in. Must be a plain identifier.
            dimensions: Embedding width. Must match your embedder, and is fixed for the
                lifetime of the table.
            relevance_weight: Weight on cosine similarity in the blended rank.
            recency_weight: Weight on recency in the blended rank.
            min_pool_size: Minimum pooled connections.
            max_pool_size: Maximum pooled connections. Keep it conservative.
            command_timeout: Per-command timeout in seconds, so a stalled query cannot hang
                the caller indefinitely.
        """
        try:
            import asyncpg
        except ImportError as exc:
            raise StoreConnectionError(
                "Postgres",
                ImportError(
                    "PgVectorStore requires the pgvector extra: "
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

        store = cls(pool, table, dimensions, relevance_weight, recency_weight)
        try:
            await store.ensure_schema()
        except BaseException:
            # A half-built store has no close(), so release the pool before propagating.
            await pool.close()
            raise
        log.info("pgvector.ready", table=table, dimensions=dimensions)
        return store

    @classmethod
    def from_pool(
        cls,
        pool: asyncpg.Pool,
        *,
        table: str = _DEFAULT_TABLE,
        dimensions: int = 1536,
        relevance_weight: float = 0.7,
        recency_weight: float = 0.3,
    ) -> PgVectorStore:
        """Wrap a pool the application already owns.

        Use this when Postgres is your primary database and you want one pool for everything.
        :meth:`close` will not close a pool it does not own, and you must call
        :meth:`ensure_schema` yourself once at startup.
        """
        return cls(pool, table, dimensions, relevance_weight, recency_weight, owns_pool=False)

    async def close(self) -> None:
        """Close the connection pool, unless it was supplied by the application."""
        if self._owns_pool:
            await self._pool.close()
