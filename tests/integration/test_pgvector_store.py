"""Integration tests for the Postgres + pgvector L2 store.

Runs against a real ``pgvector/pgvector`` container, because the whole point of this adapter
is the SQL and the extension: a mocked asyncpg would test nothing that can actually break.

The centrepiece is :func:`actrone_memory.testing.check_l2_store`, the same published
conformance suite third-party adapters are asked to pass. If a first-party adapter cannot
pass it, the suite is wrong or the adapter is.

Run with: uv run pytest tests/integration/test_pgvector_store.py -m integration
"""

from __future__ import annotations

import re

import pytest
import pytest_asyncio

from actrone_memory.exceptions import MemoryNotFoundError
from actrone_memory.models import MemoryEntry
from actrone_memory.testing import check_l2_store

pytestmark = pytest.mark.integration

asyncpg = pytest.importorskip("asyncpg", reason="pgvector extra not installed")
postgres_module = pytest.importorskip(
    "testcontainers.postgres", reason="testcontainers not installed"
)

from actrone_memory.l2.pgvector_store import PgVectorStore  # noqa: E402

_DIMENSIONS = 8
# pgvector's own image, so the extension is present without a custom Dockerfile.
_IMAGE = "pgvector/pgvector:pg16"


def _dsn(container: object) -> str:
    """asyncpg speaks postgresql://, while testcontainers hands back a SQLAlchemy URL."""
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
    store = await PgVectorStore.from_dsn(pg_dsn, dimensions=_DIMENSIONS, table=_table_for(request))
    yield store
    await store.close()


def _table_for(request: pytest.FixtureRequest) -> str:
    """A unique, SQL-safe table name derived from the test that asked for it."""
    return re.sub(r"[^a-zA-Z0-9_]", "_", request.node.name)[:60]


def _vector(hot: int) -> list[float]:
    vec = [0.0] * _DIMENSIONS
    vec[hot % _DIMENSIONS] = 1.0
    return vec


def _entry(agent_id: str, content: str, hot: int = 0) -> MemoryEntry:
    return MemoryEntry(
        agent_id=agent_id,
        session_id="s1",
        content=content,
        content_type="injected",
        embedding=_vector(hot),
        token_count=4,
    )


@pytest.mark.asyncio
async def test_passes_the_published_l2_conformance_suite(pg_dsn: str) -> None:
    """The suite we ask community adapters to pass must pass for our own adapter."""
    table = {"n": 0}

    async def factory() -> PgVectorStore:
        # A fresh table per call: several checks assume a pristine store.
        table["n"] += 1
        return await PgVectorStore.from_dsn(
            pg_dsn, dimensions=_DIMENSIONS, table=f"conformance_{table['n']}"
        )

    await check_l2_store(factory, dimensions=_DIMENSIONS)


@pytest.mark.asyncio
async def test_upsert_then_search_round_trip(store: PgVectorStore) -> None:
    entry = _entry("agent-1", "the user prefers dark mode")
    await store.upsert(entry)

    hits = await store.search(
        agent_id="agent-1", query_embedding=_vector(0), threshold=0.5, limit=5
    )

    assert [h.id for h in hits] == [entry.id]
    assert hits[0].content == "the user prefers dark mode"
    # Payload fidelity: everything the manager reads back must survive the round trip.
    assert hits[0].content_type == "injected"
    assert hits[0].source == "unknown"
    assert hits[0].sensitivity == "none"
    assert hits[0].token_count == 4
    assert hits[0].timestamp.tzinfo is not None


@pytest.mark.asyncio
async def test_search_never_crosses_agents(store: PgVectorStore) -> None:
    mine = _entry("agent-1", "my memory")
    theirs = _entry("agent-2", "their memory")
    await store.upsert(mine)
    await store.upsert(theirs)

    hits = await store.search(
        agent_id="agent-1", query_embedding=_vector(0), threshold=0.0, limit=10
    )

    assert [h.id for h in hits] == [mine.id]


@pytest.mark.asyncio
async def test_threshold_excludes_dissimilar_vectors(store: PgVectorStore) -> None:
    await store.upsert(_entry("agent-1", "about cats", hot=0))

    # An orthogonal query vector scores 0.0 cosine similarity.
    hits = await store.search(
        agent_id="agent-1", query_embedding=_vector(1), threshold=0.5, limit=10
    )

    assert hits == []


@pytest.mark.asyncio
async def test_upsert_replaces_on_conflicting_id(store: PgVectorStore) -> None:
    entry = _entry("agent-1", "first version")
    await store.upsert(entry)
    entry.content = "second version"
    await store.upsert(entry)

    hits = await store.search(
        agent_id="agent-1", query_embedding=_vector(0), threshold=0.0, limit=10
    )

    assert len(hits) == 1
    assert hits[0].content == "second version"


@pytest.mark.asyncio
async def test_delete_missing_id_raises(store: PgVectorStore) -> None:
    with pytest.raises(MemoryNotFoundError):
        await store.delete("00000000-0000-0000-0000-000000000000")


@pytest.mark.asyncio
async def test_delete_agent_memories_is_scoped(store: PgVectorStore) -> None:
    await store.upsert(_entry("agent-1", "mine"))
    keep = _entry("agent-2", "theirs")
    await store.upsert(keep)

    await store.delete_agent_memories("agent-1")

    assert (
        await store.search(agent_id="agent-1", query_embedding=_vector(0), threshold=0.0, limit=10)
        == []
    )
    survivors = await store.search(
        agent_id="agent-2", query_embedding=_vector(0), threshold=0.0, limit=10
    )
    assert [s.id for s in survivors] == [keep.id]


@pytest.mark.asyncio
async def test_content_types_filter(store: PgVectorStore) -> None:
    injected = _entry("agent-1", "injected fact")
    summary = MemoryEntry(
        agent_id="agent-1",
        session_id="s1",
        content="a summary",
        content_type="summary",
        embedding=_vector(0),
        token_count=3,
    )
    await store.upsert(injected)
    await store.upsert(summary)

    hits = await store.search(
        agent_id="agent-1",
        query_embedding=_vector(0),
        threshold=0.0,
        limit=10,
        content_types=["summary"],
    )

    assert [h.id for h in hits] == [summary.id]


@pytest.mark.asyncio
async def test_wrong_dimension_is_rejected_before_hitting_postgres(
    store: PgVectorStore,
) -> None:
    bad = MemoryEntry(
        agent_id="agent-1",
        session_id="s1",
        content="wrong width",
        content_type="injected",
        embedding=[0.1, 0.2, 0.3],
        token_count=1,
    )

    with pytest.raises(ValueError, match="dimensions"):
        await store.upsert(bad)


@pytest.mark.asyncio
async def test_rejects_an_unsafe_table_name(pg_dsn: str) -> None:
    """The table name is interpolated into SQL, so it must be validated, not trusted."""
    with pytest.raises(ValueError, match="plain identifier"):
        await PgVectorStore.from_dsn(pg_dsn, table='memories"; DROP TABLE users; --')


@pytest.mark.asyncio
async def test_from_pool_does_not_close_a_borrowed_pool(pg_dsn: str) -> None:
    pool = await asyncpg.create_pool(pg_dsn, min_size=1, max_size=2)
    assert pool is not None
    try:
        store = PgVectorStore.from_pool(pool, table="borrowed", dimensions=_DIMENSIONS)
        await store.ensure_schema()
        await store.upsert(_entry("agent-1", "via a borrowed pool"))
        await store.close()

        # The application still owns the pool, so it must remain usable after close().
        async with pool.acquire() as conn:
            count = await conn.fetchval("SELECT count(*) FROM borrowed")
        assert count == 1
    finally:
        await pool.close()
