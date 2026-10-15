from __future__ import annotations

import pytest

from actrone_memory.config import MemoryConfig
from actrone_memory.exceptions import MemoryNotFoundError
from actrone_memory.in_memory import InMemoryStore, cosine_similarity
from actrone_memory.l2.embedder import HashingEmbedder
from actrone_memory.manager import MemoryManager
from actrone_memory.models import MemoryEntry, Turn
from actrone_memory.protocols import L1Store, L2Store


def _turn(session_id: str = "s1", *, tokens: int = 10, msg: str = "hi") -> Turn:
    return Turn(
        session_id=session_id,
        user_message=msg,
        assistant_message="ok",
        token_count=tokens,
    )


def _entry(
    agent_id: str, content: str, embedding: list[float], *, mem_id: str | None = None
) -> MemoryEntry:
    kwargs: dict[str, object] = {}
    if mem_id is not None:
        kwargs["id"] = mem_id
    return MemoryEntry(
        agent_id=agent_id,
        session_id="s1",
        content=content,
        content_type="injected",
        embedding=embedding,
        token_count=5,
        **kwargs,  # type: ignore[arg-type]
    )


# ------------------------------------------------------------------
# Protocol conformance
# ------------------------------------------------------------------


def test_in_memory_store_satisfies_both_protocols():
    store = InMemoryStore()
    assert isinstance(store, L1Store)
    assert isinstance(store, L2Store)


# ------------------------------------------------------------------
# cosine_similarity
# ------------------------------------------------------------------


def test_cosine_identical_vectors_is_one():
    assert cosine_similarity([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]) == pytest.approx(1.0)


def test_cosine_orthogonal_vectors_is_zero():
    assert cosine_similarity([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)


def test_cosine_zero_vector_is_zero():
    assert cosine_similarity([0.0, 0.0], [1.0, 1.0]) == 0.0


def test_cosine_tolerates_length_mismatch():
    # Compares the shared prefix, must not raise.
    assert cosine_similarity([1.0, 0.0, 5.0], [1.0, 0.0]) == pytest.approx(1.0)


# ------------------------------------------------------------------
# L1 behaviour
# ------------------------------------------------------------------


@pytest.mark.asyncio
async def test_append_and_get_recent_turns():
    store = InMemoryStore()
    await store.append_turn("a1", "s1", _turn(msg="one"))
    await store.append_turn("a1", "s1", _turn(msg="two"))
    turns = await store.get_recent_turns("a1", "s1")
    assert [t.user_message for t in turns] == ["one", "two"]
    assert await store.turn_count("a1", "s1") == 2


@pytest.mark.asyncio
async def test_get_recent_turns_limit_returns_newest():
    store = InMemoryStore()
    for i in range(5):
        await store.append_turn("a1", "s1", _turn(msg=f"m{i}"))
    turns = await store.get_recent_turns("a1", "s1", n=2)
    assert [t.user_message for t in turns] == ["m3", "m4"]


@pytest.mark.asyncio
async def test_max_turns_cap_evicts_oldest():
    store = InMemoryStore(max_turns=3)
    for i in range(5):
        await store.append_turn("a1", "s1", _turn(msg=f"m{i}"))
    turns = await store.get_recent_turns("a1", "s1")
    assert [t.user_message for t in turns] == ["m2", "m3", "m4"]


@pytest.mark.asyncio
async def test_sessions_are_isolated_by_agent_and_session():
    store = InMemoryStore()
    await store.append_turn("a1", "s1", _turn())
    assert await store.turn_count("a1", "s2") == 0
    assert await store.turn_count("a2", "s1") == 0


@pytest.mark.asyncio
async def test_clear_session_removes_turns_and_meta():
    store = InMemoryStore()
    await store.append_turn("a1", "s1", _turn())
    await store.clear_session("a1", "s1")
    assert await store.turn_count("a1", "s1") == 0
    assert await store.get_session_metadata("a1", "s1") is None


@pytest.mark.asyncio
async def test_session_metadata_none_for_missing_session():
    store = InMemoryStore()
    assert await store.get_session_metadata("a1", "nope") is None


@pytest.mark.asyncio
async def test_session_metadata_reports_counts_and_timestamps():
    store = InMemoryStore()
    await store.append_turn("a1", "s1", _turn())
    await store.append_turn("a1", "s1", _turn())
    meta = await store.get_session_metadata("a1", "s1")
    assert meta is not None
    assert meta.turn_count == 2
    assert meta.created_at is not None
    assert meta.last_active is not None


@pytest.mark.asyncio
async def test_summary_lock_is_exclusive_within_ttl():
    store = InMemoryStore()
    assert await store.try_acquire_summary_lock("a1", "s1", ttl_seconds=60) is True
    # Second attempt within the window is refused.
    assert await store.try_acquire_summary_lock("a1", "s1", ttl_seconds=60) is False
    # A different session is independent.
    assert await store.try_acquire_summary_lock("a1", "s2", ttl_seconds=60) is True


@pytest.mark.asyncio
async def test_summary_lock_reacquirable_after_expiry(monkeypatch: pytest.MonkeyPatch):
    store = InMemoryStore()
    clock = {"t": 1000.0}
    monkeypatch.setattr("actrone_memory.in_memory.time.monotonic", lambda: clock["t"])
    assert await store.try_acquire_summary_lock("a1", "s1", ttl_seconds=5) is True
    clock["t"] = 1006.0  # past the TTL
    assert await store.try_acquire_summary_lock("a1", "s1", ttl_seconds=5) is True


# ------------------------------------------------------------------
# L2 behaviour
# ------------------------------------------------------------------


@pytest.mark.asyncio
async def test_upsert_requires_embedding():
    store = InMemoryStore()
    entry = MemoryEntry(
        agent_id="a1", session_id="s1", content="x", content_type="injected", token_count=1
    )
    with pytest.raises(ValueError, match="no embedding"):
        await store.upsert(entry)


@pytest.mark.asyncio
async def test_upsert_replaces_same_id():
    store = InMemoryStore()
    await store.upsert(_entry("a1", "first", [1.0, 0.0], mem_id="m1"))
    await store.upsert(_entry("a1", "second", [1.0, 0.0], mem_id="m1"))
    results = await store.search("a1", [1.0, 0.0], threshold=0.0, limit=10)
    assert len(results) == 1
    assert results[0].content == "second"


@pytest.mark.asyncio
async def test_search_filters_by_threshold_and_agent():
    store = InMemoryStore()
    await store.upsert(_entry("a1", "match", [1.0, 0.0]))
    await store.upsert(_entry("a1", "orthogonal", [0.0, 1.0]))
    await store.upsert(_entry("a2", "other agent", [1.0, 0.0]))

    results = await store.search("a1", [1.0, 0.0], threshold=0.5, limit=10)
    contents = {r.content for r in results}
    assert contents == {"match"}  # orthogonal below threshold; a2 excluded


@pytest.mark.asyncio
async def test_search_respects_limit_and_ranks_by_similarity():
    store = InMemoryStore()
    await store.upsert(_entry("a1", "exact", [1.0, 0.0]))
    await store.upsert(_entry("a1", "close", [0.9, 0.1]))
    results = await store.search("a1", [1.0, 0.0], threshold=0.0, limit=1)
    assert len(results) == 1
    assert results[0].content == "exact"


@pytest.mark.asyncio
async def test_search_content_type_filter():
    store = InMemoryStore()
    e = _entry("a1", "summary-only", [1.0, 0.0])
    e_summary = MemoryEntry(
        agent_id="a1",
        session_id="s1",
        content="s",
        content_type="summary",
        embedding=[1.0, 0.0],
        token_count=1,
    )
    await store.upsert(e)
    await store.upsert(e_summary)
    results = await store.search(
        "a1", [1.0, 0.0], threshold=0.0, limit=10, content_types=["summary"]
    )
    assert [r.content_type for r in results] == ["summary"]


@pytest.mark.asyncio
async def test_search_empty_agent_returns_empty():
    store = InMemoryStore()
    assert await store.search("unknown", [1.0, 0.0], threshold=0.0, limit=10) == []


@pytest.mark.asyncio
async def test_delete_removes_and_missing_raises():
    store = InMemoryStore()
    await store.upsert(_entry("a1", "x", [1.0, 0.0], mem_id="m1"))
    await store.delete("m1")
    assert await store.search("a1", [1.0, 0.0], threshold=0.0, limit=10) == []
    with pytest.raises(MemoryNotFoundError):
        await store.delete("m1")


@pytest.mark.asyncio
async def test_delete_agent_memories():
    store = InMemoryStore()
    await store.upsert(_entry("a1", "x", [1.0, 0.0]))
    await store.delete_agent_memories("a1")
    assert await store.search("a1", [1.0, 0.0], threshold=0.0, limit=10) == []


@pytest.mark.asyncio
async def test_close_is_noop():
    store = InMemoryStore()
    await store.close()  # must not raise


# ------------------------------------------------------------------
# End-to-end: zero-service local-first MemoryManager
# ------------------------------------------------------------------


@pytest.mark.asyncio
async def test_local_first_create_needs_no_services_or_key():
    """The headline parity guarantee: create() with all-default config runs with
    no Redis, no Qdrant, and no API key, and completes a full store→retrieve cycle."""
    cfg = MemoryConfig()  # defaults: backend="memory", provider="local"
    assert cfg.backend == "memory"
    # "local" is the default; with no [onnx]/[local] extra installed it degrades gracefully to the
    # dependency-free hashing embedder, so the "no services, no API key" guarantee still holds.
    assert cfg.embedding_provider == "local"

    async with await MemoryManager.create(cfg) as mm:
        await mm.store_turn("agent-1", "sess-1", "My favourite colour is blue.", "Noted.")
        await mm.inject_memory("agent-1", "The user prefers dark mode.", importance=0.9)

        # L1 recall (recent turns) always works with the local backend.
        ctx = await mm.retrieve_context("agent-1", "sess-1", "colour preference", token_budget=4096)
        assert ctx.token_budget == 4096
        assert len(ctx.recent_turns) == 1

        # L2 semantic recall via the hashing embedder is keyword-overlap based, so
        # a near-matching query clears the default threshold and retrieves the fact.
        hits = await mm.search_memories("agent-1", "the user prefers dark mode", limit=5)
        assert any("dark mode" in h.content for h in hits)


@pytest.mark.asyncio
async def test_local_first_uses_in_memory_store_instances():
    cfg = MemoryConfig()
    mm = await MemoryManager.create(cfg)
    try:
        assert isinstance(mm._l1, InMemoryStore)
        assert isinstance(mm._l2, InMemoryStore)
        assert isinstance(mm._embedder, HashingEmbedder)
    finally:
        await mm.close()


# ------------------------------------------------------------------
# Provenance-typing v1 (source + sensitivity) + local erase
# ------------------------------------------------------------------


@pytest.mark.asyncio
async def test_injected_fact_carries_provenance_end_to_end():
    async with await MemoryManager.create(MemoryConfig()) as mm:
        await mm.inject_memory(
            "agent-1",
            "The user's account number is 12345.",
            source="import:crm",
            sensitivity="pii",
        )
        hits = await mm.search_memories("agent-1", "the user account number is 12345", limit=5)
        assert len(hits) == 1
        assert hits[0].source == "import:crm"
        assert hits[0].sensitivity == "pii"


@pytest.mark.asyncio
async def test_provenance_defaults_are_backwards_compatible():
    entry = MemoryEntry(
        agent_id="a",
        session_id="s",
        content="x",
        content_type="injected",
        embedding=[1.0, 0.0],
        token_count=1,
    )
    assert entry.source == "unknown"
    assert entry.sensitivity == "none"


@pytest.mark.asyncio
async def test_erase_agent_memories_wipes_l2():
    async with await MemoryManager.create(MemoryConfig()) as mm:
        await mm.store_turn("agent-1", "s1", "hi", "hello")
        await mm.inject_memory("agent-1", "a durable fact about the user", sensitivity="low")
        await mm.erase_agent_memories("agent-1", session_id="s1")

        # L2 memories gone.
        assert await mm.search_memories("agent-1", "a durable fact about the user", limit=5) == []
        # L1 session cleared too (session_id was supplied).
        assert await mm.get_session_metadata("agent-1", "s1") is None


@pytest.mark.asyncio
async def test_summarised_memory_source_is_summary():
    store = InMemoryStore()
    # Low threshold so the keyword-overlap summary is retrievable via the hashing embedder.
    cfg = MemoryConfig(auto_summarise=False, summarise_after_turns=2, relevance_threshold=0.1)
    mm = MemoryManager(store, store, HashingEmbedder(), cfg)
    try:
        for i in range(2):
            await mm.store_turn("agent-1", "s1", f"question {i}", f"answer {i}")
        await mm._summarise_session("agent-1", "s1")
        hits = await mm.search_memories("agent-1", "user question assistant answer", limit=5)
        summaries = [h for h in hits if h.content_type == "summary"]
        assert summaries
        assert all(h.source == "summary" for h in summaries)
    finally:
        await mm.close()
