from __future__ import annotations

import pytest
import pytest_asyncio
from testcontainers.qdrant import QdrantContainer

from actrone_memory.l2.qdrant_store import QdrantStore
from actrone_memory.models import MemoryEntry


@pytest.fixture(scope="module")
def qdrant_url():
    with QdrantContainer("qdrant/qdrant:v1.13.6") as container:
        yield f"http://{container.get_container_host_ip()}:{container.get_exposed_port(6333)}"


@pytest_asyncio.fixture
async def store(qdrant_url: str) -> QdrantStore:
    s = await QdrantStore.from_url(
        qdrant_url,
        collection="test_memories",
        dimensions=4,  # small vectors for tests
        relevance_weight=0.7,
        recency_weight=0.3,
    )
    yield s
    await s.close()


def _entry(agent_id: str = "agent-1", content: str = "Paris is in France.") -> MemoryEntry:
    return MemoryEntry(
        agent_id=agent_id,
        session_id="sess-1",
        content=content,
        content_type="summary",
        embedding=[0.1, 0.2, 0.3, 0.4],
        importance_score=0.8,
        token_count=10,
    )


@pytest.mark.asyncio
async def test_upsert_and_search(store: QdrantStore):
    entry = _entry()
    await store.upsert(entry)

    results = await store.search(
        agent_id="agent-1",
        query_embedding=[0.1, 0.2, 0.3, 0.4],
        threshold=0.0,
        limit=10,
    )
    assert any(r.id == entry.id for r in results)


@pytest.mark.asyncio
async def test_search_respects_agent_id_filter(store: QdrantStore):
    await store.upsert(_entry(agent_id="agent-A", content="A's memory"))
    await store.upsert(_entry(agent_id="agent-B", content="B's memory"))

    results = await store.search(
        agent_id="agent-A",
        query_embedding=[0.1, 0.2, 0.3, 0.4],
        threshold=0.0,
    )
    assert all(r.agent_id == "agent-A" for r in results)


@pytest.mark.asyncio
async def test_delete_memory(store: QdrantStore):
    entry = _entry(content="to be deleted")
    await store.upsert(entry)
    await store.delete(entry.id)

    results = await store.search(
        agent_id="agent-1",
        query_embedding=[0.1, 0.2, 0.3, 0.4],
        threshold=0.0,
    )
    assert all(r.id != entry.id for r in results)


@pytest.mark.asyncio
async def test_upsert_requires_embedding():
    from actrone_memory.l2.qdrant_store import QdrantStore

    entry = MemoryEntry(agent_id="a", session_id="s", content="no embed", content_type="turn")
    store_mock = QdrantStore.__new__(QdrantStore)
    with pytest.raises(ValueError, match="no embedding"):
        await store_mock.upsert(entry)


@pytest.mark.asyncio
async def test_delete_agent_memories(store: QdrantStore):
    for i in range(3):
        await store.upsert(_entry(agent_id="bulk-agent", content=f"memory {i}"))

    await store.delete_agent_memories("bulk-agent")

    results = await store.search(
        agent_id="bulk-agent",
        query_embedding=[0.1, 0.2, 0.3, 0.4],
        threshold=0.0,
    )
    assert results == []
