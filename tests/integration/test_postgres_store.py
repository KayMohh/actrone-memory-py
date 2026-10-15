"""Integration tests for the Postgres hot-tier (L1) store.

Runs against a real Postgres container. The centrepiece is
:func:`actrone_memory.testing.check_l1_store`, the published conformance suite, plus the
behaviours Postgres has to emulate because it lacks Redis primitives: row expiry, the
retention cap, and a fleet-wide summary lock.

Run with: uv run pytest tests/integration/test_postgres_store.py -m integration
"""

from __future__ import annotations

import asyncio
import re

import pytest
import pytest_asyncio

from actrone_memory.models import Turn
from actrone_memory.testing import check_l1_store

pytestmark = pytest.mark.integration

asyncpg = pytest.importorskip("asyncpg", reason="pgvector extra not installed")
postgres_module = pytest.importorskip(
    "testcontainers.postgres", reason="testcontainers not installed"
)

from actrone_memory.l1.postgres_store import PostgresStore  # noqa: E402

_IMAGE = "pgvector/pgvector:pg16"


def _dsn(container: object) -> str:
    url = container.get_connection_url()  # type: ignore[attr-defined]
    return url.replace("postgresql+psycopg2://", "postgresql://").replace(
        "postgresql+psycopg://", "postgresql://"
    )


# Module-scoped and synchronous on purpose: starting a container per test would dominate
# the runtime, and PostgresContainer is a sync context manager, so this avoids the
# async-fixture scoping dance entirely.
@pytest.fixture(scope="module")
def pg_dsn():
    with postgres_module.PostgresContainer(_IMAGE) as container:
        yield _dsn(container)


@pytest_asyncio.fixture
async def store(pg_dsn: str, request: pytest.FixtureRequest):
    # One container, but a table per test: sharing the container is a speed optimisation,
    # sharing rows between tests is a correctness bug.
    store = await PostgresStore.from_dsn(pg_dsn, table=_table_for(request), max_turns=50)
    yield store
    await store.close()


def _table_for(request: pytest.FixtureRequest) -> str:
    """A unique, SQL-safe table name derived from the test that asked for it."""
    return re.sub(r"[^a-zA-Z0-9_]", "_", request.node.name)[:55]


def _turn(user: str, assistant: str = "ok") -> Turn:
    return Turn(session_id="s1", user_message=user, assistant_message=assistant, token_count=4)


@pytest.mark.asyncio
async def test_passes_the_published_l1_conformance_suite(pg_dsn: str) -> None:
    """The suite we ask community adapters to pass must pass for our own adapter."""
    counter = {"n": 0}

    async def factory() -> PostgresStore:
        counter["n"] += 1
        return await PostgresStore.from_dsn(pg_dsn, table=f"conformance_{counter['n']}")

    await check_l1_store(factory)


@pytest.mark.asyncio
async def test_round_trip_preserves_the_turn_payload(store: PostgresStore) -> None:
    turn = Turn(
        session_id="s1",
        user_message="what is my plan?",
        assistant_message="the pro plan",
        token_count=11,
    )
    await store.append_turn("agent-1", "s1", turn)

    turns = await store.get_recent_turns("agent-1", "s1")

    assert len(turns) == 1
    assert turns[0].id == turn.id
    assert turns[0].user_message == "what is my plan?"
    assert turns[0].assistant_message == "the pro plan"
    assert turns[0].token_count == 11
    assert turns[0].timestamp.tzinfo is not None


@pytest.mark.asyncio
async def test_retention_cap_trims_oldest_turns(pg_dsn: str) -> None:
    store = await PostgresStore.from_dsn(pg_dsn, table="capped", max_turns=3)
    try:
        for i in range(6):
            await store.append_turn("agent-1", "s1", _turn(f"msg-{i}"))

        turns = await store.get_recent_turns("agent-1", "s1")

        # Only the newest 3 survive, still in chronological order.
        assert [t.user_message for t in turns] == ["msg-3", "msg-4", "msg-5"]
        assert await store.turn_count("agent-1", "s1") == 3
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_expired_turns_are_not_returned_or_counted(pg_dsn: str) -> None:
    """A zero-hour TTL expires rows immediately, which is the read filter under test."""
    store = await PostgresStore.from_dsn(pg_dsn, table="expiring", session_ttl_hours=0)
    try:
        await store.append_turn("agent-1", "s1", _turn("should expire"))
        await asyncio.sleep(0.05)

        assert await store.get_recent_turns("agent-1", "s1") == []
        assert await store.turn_count("agent-1", "s1") == 0
        assert await store.get_session_metadata("agent-1", "s1") is None
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_ordering_is_stable_for_turns_in_the_same_clock_tick(store: PostgresStore) -> None:
    """Timestamps can collide, so ordering must come from the sequence, not the clock."""
    same_time = _turn("first").timestamp
    for i in range(5):
        turn = Turn(
            session_id="s1",
            user_message=f"msg-{i}",
            assistant_message="ok",
            token_count=2,
            timestamp=same_time,
        )
        await store.append_turn("agent-1", "s1", turn)

    turns = await store.get_recent_turns("agent-1", "s1")

    assert [t.user_message for t in turns] == ["msg-0", "msg-1", "msg-2", "msg-3", "msg-4"]


@pytest.mark.asyncio
async def test_summary_lock_admits_one_holder_then_expires(store: PostgresStore) -> None:
    assert await store.try_acquire_summary_lock("agent-1", "s1", 60) is True
    assert await store.try_acquire_summary_lock("agent-1", "s1", 60) is False

    # A zero-second window is already elapsed, so the next caller may claim it.
    assert await store.try_acquire_summary_lock("agent-1", "s2", 0) is True
    await asyncio.sleep(0.05)
    assert await store.try_acquire_summary_lock("agent-1", "s2", 60) is True


@pytest.mark.asyncio
async def test_summary_lock_is_exclusive_under_concurrency(store: PostgresStore) -> None:
    """The whole point of the lock: exactly one winner across concurrent workers."""
    results = await asyncio.gather(
        *(store.try_acquire_summary_lock("agent-1", "race", 60) for _ in range(12))
    )

    assert sum(1 for won in results if won) == 1, f"expected exactly one winner, got {results}"


@pytest.mark.asyncio
async def test_concurrent_appends_all_land(store: PostgresStore) -> None:
    await asyncio.gather(
        *(store.append_turn("agent-1", "concurrent", _turn(f"u{i}")) for i in range(20))
    )

    turns = await store.get_recent_turns("agent-1", "concurrent", n=20)

    assert len({t.user_message for t in turns}) == 20


@pytest.mark.asyncio
async def test_clear_session_also_releases_the_summary_lock(store: PostgresStore) -> None:
    await store.append_turn("agent-1", "s1", _turn("hi"))
    assert await store.try_acquire_summary_lock("agent-1", "s1", 600) is True

    await store.clear_session("agent-1", "s1")

    assert await store.get_recent_turns("agent-1", "s1") == []
    # Without releasing the lock, a reused session id could never be summarised again.
    assert await store.try_acquire_summary_lock("agent-1", "s1", 600) is True


@pytest.mark.asyncio
async def test_rejects_an_unsafe_table_name(pg_dsn: str) -> None:
    with pytest.raises(ValueError, match="plain identifier"):
        await PostgresStore.from_dsn(pg_dsn, table='turns"; DROP TABLE users; --')


@pytest.mark.asyncio
async def test_both_tiers_can_share_one_pool(pg_dsn: str) -> None:
    """The "no new infrastructure" path: one Postgres pool serving L1 and L2."""
    pgvector_store = pytest.importorskip("actrone_memory.l2.pgvector_store")

    pool = await asyncpg.create_pool(pg_dsn, min_size=1, max_size=4)
    assert pool is not None
    try:
        l1 = PostgresStore.from_pool(pool, table="shared_turns")
        l2 = pgvector_store.PgVectorStore.from_pool(pool, table="shared_memories", dimensions=4)
        await l1.ensure_schema()
        await l2.ensure_schema()

        await l1.append_turn("agent-1", "s1", _turn("shared pool"))
        assert len(await l1.get_recent_turns("agent-1", "s1")) == 1

        # Closing a borrowed pool is the caller's job, so neither store may close it.
        await l1.close()
        await l2.close()
        async with pool.acquire() as conn:
            assert await conn.fetchval("SELECT 1") == 1
    finally:
        await pool.close()
