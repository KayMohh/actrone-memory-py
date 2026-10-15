from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from actrone_memory.exceptions import MemoryNotFoundError, StoreConnectionError
from actrone_memory.l2.qdrant_store import QdrantStore
from actrone_memory.models import MemoryEntry


def _make_store(mock_client: AsyncMock) -> QdrantStore:
    return QdrantStore(
        client=mock_client,
        collection="test_memories",
        dimensions=4,
        relevance_weight=0.7,
        recency_weight=0.3,
    )


def _entry(content: str = "Paris is in France.") -> MemoryEntry:
    return MemoryEntry(
        agent_id="agent-1",
        session_id="sess-1",
        content=content,
        content_type="summary",
        embedding=[0.1, 0.2, 0.3, 0.4],
        importance_score=0.8,
        token_count=10,
    )


# ------------------------------------------------------------------
# upsert
# ------------------------------------------------------------------


@pytest.mark.asyncio
async def test_upsert_calls_client_upsert():
    client = AsyncMock()
    store = _make_store(client)
    await store.upsert(_entry())
    client.upsert.assert_called_once()


@pytest.mark.asyncio
async def test_upsert_raises_without_embedding():
    client = AsyncMock()
    store = _make_store(client)
    entry = MemoryEntry(agent_id="a", session_id="s", content="x", content_type="turn")
    with pytest.raises(ValueError, match="no embedding"):
        await store.upsert(entry)


@pytest.mark.asyncio
async def test_upsert_raises_store_error_on_failure():
    client = AsyncMock()
    client.upsert.side_effect = Exception("network error")
    store = _make_store(client)
    with pytest.raises(StoreConnectionError):
        await store.upsert(_entry())


# ------------------------------------------------------------------
# upsert_batch
# ------------------------------------------------------------------


@pytest.mark.asyncio
async def test_upsert_batch_skips_entries_without_embedding():
    client = AsyncMock()
    store = _make_store(client)

    no_embed = MemoryEntry(agent_id="a", session_id="s", content="x", content_type="turn")
    with_embed = _entry()

    await store.upsert_batch([no_embed, with_embed])
    # Only one point (the one with embedding) should be sent
    call_kwargs = client.upsert.call_args[1]
    assert len(call_kwargs["points"]) == 1


@pytest.mark.asyncio
async def test_upsert_batch_no_op_on_empty_list():
    client = AsyncMock()
    store = _make_store(client)
    await store.upsert_batch([])
    client.upsert.assert_not_called()


# ------------------------------------------------------------------
# search
# ------------------------------------------------------------------


@pytest.mark.asyncio
async def test_search_returns_ranked_entries():
    client = AsyncMock()
    now_iso = datetime.now(UTC).isoformat()

    mock_result = MagicMock()
    mock_result.id = "mem-1"
    mock_result.score = 0.9
    mock_result.payload = {
        "agent_id": "agent-1",
        "session_id": "sess-1",
        "content": "Paris is in France.",
        "content_type": "summary",
        "importance_score": 0.8,
        "topic_tags": [],
        "token_count": 10,
        "timestamp": now_iso,
        "source_turn_ids": [],
    }
    # qdrant-client ≥1.10 returns a QueryResponse with a `.points` list from
    # query_points(); the legacy search() helper was removed.
    client.query_points.return_value = MagicMock(points=[mock_result])

    store = _make_store(client)
    results = await store.search("agent-1", [0.1, 0.2, 0.3, 0.4], threshold=0.72)

    assert len(results) == 1
    assert results[0].content == "Paris is in France."


@pytest.mark.asyncio
async def test_search_raises_on_client_error():
    client = AsyncMock()
    client.query_points.side_effect = Exception("timeout")
    store = _make_store(client)

    with pytest.raises(StoreConnectionError):
        await store.search("agent-1", [0.1, 0.2, 0.3, 0.4], threshold=0.72)


# ------------------------------------------------------------------
# delete
# ------------------------------------------------------------------


@pytest.mark.asyncio
async def test_delete_calls_client_delete():
    client = AsyncMock()
    store = _make_store(client)
    await store.delete("mem-123")
    client.delete.assert_called_once()


@pytest.mark.asyncio
async def test_delete_raises_store_error_on_failure():
    client = AsyncMock()
    client.delete.side_effect = Exception("error")
    store = _make_store(client)
    with pytest.raises(StoreConnectionError):
        await store.delete("mem-123")


# ------------------------------------------------------------------
# delete_agent_memories
# ------------------------------------------------------------------


@pytest.mark.asyncio
async def test_delete_agent_memories_calls_client():
    client = AsyncMock()
    store = _make_store(client)
    await store.delete_agent_memories("agent-1")
    client.delete.assert_called_once()


@pytest.mark.asyncio
async def test_delete_agent_memories_raises_on_failure():
    client = AsyncMock()
    client.delete.side_effect = Exception("error")
    store = _make_store(client)
    with pytest.raises(StoreConnectionError):
        await store.delete_agent_memories("agent-1")


# ------------------------------------------------------------------
# ensure_collection
# ------------------------------------------------------------------


@pytest.mark.asyncio
async def test_ensure_collection_creates_when_missing():
    client = AsyncMock()
    client.collection_exists.return_value = False
    store = _make_store(client)
    await store.ensure_collection()
    client.create_collection.assert_called_once()


@pytest.mark.asyncio
async def test_ensure_collection_skips_when_exists():
    client = AsyncMock()
    client.collection_exists.return_value = True
    store = _make_store(client)
    await store.ensure_collection()
    client.create_collection.assert_not_called()


@pytest.mark.asyncio
async def test_ensure_collection_raises_on_error():
    client = AsyncMock()
    client.collection_exists.side_effect = Exception("network error")
    store = _make_store(client)
    with pytest.raises(StoreConnectionError):
        await store.ensure_collection()


# ------------------------------------------------------------------
# upsert_batch error path
# ------------------------------------------------------------------


@pytest.mark.asyncio
async def test_upsert_batch_raises_on_client_error():
    client = AsyncMock()
    client.upsert.side_effect = Exception("timeout")
    store = _make_store(client)
    with pytest.raises(StoreConnectionError):
        await store.upsert_batch([_entry()])


# ------------------------------------------------------------------
# delete 404 path (MemoryNotFoundError)
# ------------------------------------------------------------------


@pytest.mark.asyncio
async def test_delete_raises_memory_not_found_on_404():
    from qdrant_client.http.exceptions import UnexpectedResponse

    client = AsyncMock()
    not_found = UnexpectedResponse(
        status_code=404, reason_phrase="Not Found", content=b"", headers={}
    )
    client.delete.side_effect = not_found
    store = _make_store(client)

    with pytest.raises(MemoryNotFoundError) as exc_info:
        await store.delete("missing-id")
    assert exc_info.value.memory_id == "missing-id"


# ------------------------------------------------------------------
# close
# ------------------------------------------------------------------


@pytest.mark.asyncio
async def test_close_calls_client_close():
    client = AsyncMock()
    store = _make_store(client)
    await store.close()
    client.close.assert_called_once()


# ------------------------------------------------------------------
# from_url factory (mocked)
# ------------------------------------------------------------------


@pytest.mark.asyncio
async def test_from_url_raises_on_connection_error():
    with patch("actrone_memory.l2.qdrant_store.AsyncQdrantClient") as mock_cls:
        mock_client = AsyncMock()
        mock_client.collection_exists.side_effect = Exception("refused")
        mock_cls.return_value = mock_client

        with pytest.raises(StoreConnectionError):
            await QdrantStore.from_url("http://badhost:6333", collection="test", dimensions=4)
