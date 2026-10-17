from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, patch

import pytest

from actrone_memory.config import MemoryConfig
from actrone_memory.exceptions import TokenBudgetError, ValidationError
from actrone_memory.in_memory import InMemoryStore
from actrone_memory.manager import MemoryManager, create_memory_manager
from actrone_memory.models import ContentType, MemoryEntry, SessionMetadata, Turn
from tests.conftest import ConstantEmbedder


def _make_manager(
    cfg: MemoryConfig, turns: list[Turn], memories: list[MemoryEntry]
) -> MemoryManager:
    l1 = AsyncMock()
    l1.get_recent_turns.return_value = turns
    l1.turn_count.return_value = len(turns)
    l1.append_turn.return_value = None
    l1.clear_session.return_value = None

    l2 = AsyncMock()
    l2.search.return_value = memories
    l2.upsert.return_value = None
    l2.delete.return_value = None

    return MemoryManager(l1=l1, l2=l2, embedder=ConstantEmbedder(), config=cfg)


@pytest.mark.asyncio
async def test_retrieve_context_invalid_budget(fake_config: MemoryConfig):
    mm = _make_manager(fake_config, [], [])
    with pytest.raises(TokenBudgetError):
        await mm.retrieve_context("a1", "s1", "query", token_budget=0)


@pytest.mark.asyncio
async def test_store_turn_rejects_empty_agent_id(fake_config: MemoryConfig):
    mm = _make_manager(fake_config, [], [])
    with pytest.raises(ValidationError, match="agent_id"):
        await mm.store_turn("", "s1", "hi", "hello")


@pytest.mark.asyncio
async def test_store_turn_rejects_oversized_message(fake_config: MemoryConfig):
    mm = _make_manager(fake_config, [], [])
    with pytest.raises(ValidationError, match="user_message"):
        await mm.store_turn("a1", "s1", "x" * 100_001, "hello")


@pytest.mark.asyncio
async def test_inject_memory_rejects_invalid_importance(fake_config: MemoryConfig):
    mm = _make_manager(fake_config, [], [])
    with pytest.raises(ValidationError, match="importance"):
        await mm.inject_memory("a1", "some content", importance=1.5)


@pytest.mark.asyncio
async def test_inject_memory_rejects_blank_content(fake_config: MemoryConfig):
    mm = _make_manager(fake_config, [], [])
    with pytest.raises(ValidationError, match="content"):
        await mm.inject_memory("a1", "   ")


@pytest.mark.asyncio
async def test_search_memories_rejects_invalid_limit(fake_config: MemoryConfig):
    mm = _make_manager(fake_config, [], [])
    with pytest.raises(ValidationError, match="limit"):
        await mm.search_memories("a1", "query", limit=0)


@pytest.mark.asyncio
async def test_retrieve_context_returns_pruned_turns(fake_config: MemoryConfig):
    turns = [
        Turn(session_id="s1", user_message="q1", assistant_message="a1", token_count=100),
        Turn(session_id="s1", user_message="q2", assistant_message="a2", token_count=100),
        Turn(session_id="s1", user_message="q3", assistant_message="a3", token_count=100),
    ]
    mm = _make_manager(fake_config, turns, [])

    # session budget = 4096 * 0.35 = 1433, all 3 turns fit (300 tokens total)
    ctx = await mm.retrieve_context("a1", "s1", "q", token_budget=4096)
    assert len(ctx.recent_turns) == 3


@pytest.mark.asyncio
async def test_retrieve_context_prunes_when_over_budget(fake_config: MemoryConfig):
    turns = [
        Turn(session_id="s1", user_message="q1", assistant_message="a1", token_count=500),
        Turn(session_id="s1", user_message="q2", assistant_message="a2", token_count=500),
        Turn(session_id="s1", user_message="q3", assistant_message="a3", token_count=500),
    ]
    # Small budget, session budget = 200 * 0.35 = 70, only last turn fits (most recent kept)
    mm = _make_manager(fake_config, turns, [])
    ctx = await mm.retrieve_context("a1", "s1", "q", token_budget=200)
    assert len(ctx.recent_turns) == 0  # 500 > 70, none fit


@pytest.mark.asyncio
async def test_store_turn_calls_l1_append(fake_config: MemoryConfig):
    mm = _make_manager(fake_config, [], [])
    turn_id = await mm.store_turn("a1", "s1", "hello", "world")
    assert turn_id
    mm._l1.append_turn.assert_called_once()


@pytest.mark.asyncio
async def test_inject_memory_upserts_to_l2(fake_config: MemoryConfig):
    mm = _make_manager(fake_config, [], [])
    memory_id = await mm.inject_memory("a1", "Paris is in France.", importance=0.9)
    assert memory_id
    mm._l2.upsert.assert_called_once()
    call_arg: MemoryEntry = mm._l2.upsert.call_args[0][0]
    assert call_arg.content_type == "injected"
    assert call_arg.importance_score == 0.9


@pytest.mark.asyncio
async def test_clear_session_delegates_to_l1(fake_config: MemoryConfig):
    mm = _make_manager(fake_config, [], [])
    await mm.clear_session("a1", "s1")
    mm._l1.clear_session.assert_called_once_with("a1", "s1")


@pytest.mark.asyncio
async def test_delete_memory_delegates_to_l2(fake_config: MemoryConfig):
    mm = _make_manager(fake_config, [], [])
    await mm.delete_memory("a1", "mem-123")
    mm._l2.delete.assert_called_once_with("mem-123")


@pytest.mark.asyncio
async def test_search_memories_delegates_to_l2(fake_config: MemoryConfig):
    memories = [
        MemoryEntry(
            agent_id="a",
            session_id="s",
            content="x",
            content_type="summary",
            embedding=[0.1, 0.2, 0.3, 0.4],
            token_count=10,
        ),
    ]
    mm = _make_manager(fake_config, [], memories)
    results = await mm.search_memories("a", "query", limit=5)
    assert len(results) == 1
    mm._l2.search.assert_called_once()


@pytest.mark.asyncio
async def test_close_calls_both_stores(fake_config: MemoryConfig):
    mm = _make_manager(fake_config, [], [])
    await mm.close()
    mm._l1.close.assert_called_once()
    mm._l2.close.assert_called_once()


@pytest.mark.asyncio
async def test_context_manager_closes_on_exit(fake_config: MemoryConfig):
    mm = _make_manager(fake_config, [], [])
    async with mm:
        pass
    mm._l1.close.assert_called_once()
    mm._l2.close.assert_called_once()


@pytest.mark.asyncio
async def test_auto_summarise_not_triggered_below_threshold(fake_config: MemoryConfig):
    # turn_count returns 5, threshold is 20, no task should be created
    fake_config.auto_summarise = True
    fake_config.summarise_after_turns = 20
    mm = _make_manager(fake_config, [], [])
    mm._l1.turn_count.return_value = 5
    await mm.store_turn("a", "s", "hi", "hello")
    # No summarise task should be scheduled, hard to assert directly,
    # but store_turn must complete without error
    mm._l1.append_turn.assert_called_once()


@pytest.mark.asyncio
async def test_summarise_session_handles_empty_turns(fake_config: MemoryConfig):
    mm = _make_manager(fake_config, [], [])
    mm._l1.get_recent_turns.return_value = []
    # Should return silently without upserting anything
    await mm._summarise_session("agent-1", "s1")
    mm._l2.upsert.assert_not_called()


@pytest.mark.asyncio
async def test_summarise_session_upserts_summary(fake_config: MemoryConfig):
    turns = [
        Turn(session_id="s1", user_message="q1", assistant_message="a1", token_count=10),
        Turn(session_id="s1", user_message="q2", assistant_message="a2", token_count=10),
    ]
    mm = _make_manager(fake_config, turns, [])
    await mm._summarise_session("agent-1", "s1")
    mm._l2.upsert.assert_called_once()
    upserted: MemoryEntry = mm._l2.upsert.call_args[0][0]
    assert upserted.content_type == "summary"
    assert upserted.agent_id == "agent-1"


@pytest.mark.asyncio
async def test_prune_memories_within_budget(fake_config: MemoryConfig):
    memories = [
        MemoryEntry(
            agent_id="a",
            session_id="s",
            content="a",
            content_type="summary",
            embedding=[0.1, 0.2, 0.3, 0.4],
            token_count=50,
        ),
        MemoryEntry(
            agent_id="a",
            session_id="s",
            content="b",
            content_type="summary",
            embedding=[0.1, 0.2, 0.3, 0.4],
            token_count=50,
        ),
    ]
    mm = _make_manager(fake_config, [], memories)
    ctx = await mm.retrieve_context("a", "s", "q", token_budget=4096)
    assert len(ctx.episodic_memories) == 2


# ------------------------------------------------------------------
# auto-summarise trigger (above threshold)
# ------------------------------------------------------------------


@pytest.mark.asyncio
async def test_auto_summarise_triggered_above_threshold(fake_config: MemoryConfig):

    fake_config.auto_summarise = True
    fake_config.summarise_after_turns = 3
    mm = _make_manager(fake_config, [], [])
    mm._l1.turn_count.return_value = 5  # above threshold

    with patch("actrone_memory.manager.asyncio.create_task") as mock_task:
        await mm.store_turn("a", "s", "hi", "hello")
        mock_task.assert_called_once()
    # The summary cooldown lock must have been claimed before spawning.
    mm._l1.try_acquire_summary_lock.assert_called_once()


@pytest.mark.asyncio
async def test_auto_summarise_suppressed_when_lock_not_acquired(fake_config: MemoryConfig):
    """Even above the threshold, no summary task spawns if the cooldown lock is
    already held (concurrent/fleet dedupe + throttle)."""
    fake_config.auto_summarise = True
    fake_config.summarise_after_turns = 3
    mm = _make_manager(fake_config, [], [])
    mm._l1.turn_count.return_value = 5  # above threshold
    mm._l1.try_acquire_summary_lock.return_value = False  # lock held by someone else

    with patch("actrone_memory.manager.asyncio.create_task") as mock_task:
        await mm.store_turn("a", "s", "hi", "hello")
        mock_task.assert_not_called()


# ------------------------------------------------------------------
# get_session_metadata on manager
# ------------------------------------------------------------------


@pytest.mark.asyncio
async def test_get_session_metadata_delegates_to_l1(fake_config: MemoryConfig):
    mm = _make_manager(fake_config, [], [])
    meta = SessionMetadata(agent_id="a", session_id="s", turn_count=3)
    mm._l1.get_session_metadata.return_value = meta

    result = await mm.get_session_metadata("a", "s")
    assert result is meta
    mm._l1.get_session_metadata.assert_called_once_with("a", "s")


# ------------------------------------------------------------------
# _summarise_session error path
# ------------------------------------------------------------------


@pytest.mark.asyncio
async def test_summarise_session_swallows_exceptions(fake_config: MemoryConfig):
    mm = _make_manager(fake_config, [], [])
    mm._l1.get_recent_turns.side_effect = Exception("redis down")
    # Must not propagate, background task failure is logged, not raised
    await mm._summarise_session("a", "s")


# ------------------------------------------------------------------
# create_memory_manager context manager
# ------------------------------------------------------------------


@pytest.mark.asyncio
async def test_create_memory_manager_context_manager(fake_config: MemoryConfig):
    mm = _make_manager(fake_config, [], [])
    with patch("actrone_memory.manager.MemoryManager.create", return_value=mm):
        async with create_memory_manager(fake_config) as manager:
            assert manager is mm
        mm._l1.close.assert_called_once()
        mm._l2.close.assert_called_once()


# ------------------------------------------------------------------
# Custom store injection (the supported path for non-Redis/Qdrant backends)
# ------------------------------------------------------------------


class _RecordingStore:
    """Minimal structural L1Store + L2Store used to prove injection is honoured."""

    def __init__(self) -> None:
        self.turns: list[Turn] = []
        self.entries: list[MemoryEntry] = []
        self.closed = False

    # L1
    async def append_turn(self, agent_id: str, session_id: str, turn: Turn) -> None:
        self.turns.append(turn)

    async def get_recent_turns(
        self, agent_id: str, session_id: str, n: int | None = None
    ) -> list[Turn]:
        return list(self.turns)

    async def turn_count(self, agent_id: str, session_id: str) -> int:
        return len(self.turns)

    async def clear_session(self, agent_id: str, session_id: str) -> None:
        self.turns.clear()

    async def get_session_metadata(self, agent_id: str, session_id: str):
        return None

    async def try_acquire_summary_lock(
        self, agent_id: str, session_id: str, ttl_seconds: int
    ) -> bool:
        return False

    # L2
    async def upsert(self, entry: MemoryEntry) -> None:
        self.entries.append(entry)

    async def search(
        self, agent_id, query_embedding, threshold, limit=20, content_types=None, query_text=None
    ) -> list[MemoryEntry]:
        return list(self.entries)

    async def delete(self, memory_id: str) -> None:
        self.entries = [e for e in self.entries if e.id != memory_id]

    async def delete_agent_memories(self, agent_id: str) -> None:
        self.entries.clear()

    async def close(self) -> None:
        self.closed = True


@pytest.mark.asyncio
async def test_create_uses_injected_stores():
    store = _RecordingStore()
    cfg = MemoryConfig(embedding_provider="hashing")  # type: ignore[call-arg]

    mm = await MemoryManager.create(cfg, l1=store, l2=store)
    await mm.store_turn("a1", "s1", "hello", "hi")
    await mm.inject_memory("a1", "the user prefers concise answers")

    assert len(store.turns) == 1, "injected L1 was not used"
    assert len(store.entries) == 1, "injected L2 was not used"

    await mm.close()
    assert store.closed is True, "close() must propagate to an injected store"


@pytest.mark.asyncio
async def test_injected_stores_bypass_the_redis_qdrant_backend():
    """Injecting both stores must not open a Redis or Qdrant connection.

    The config below points at a port nothing is listening on: if create() still
    tried to build the durable backend, it would raise instead of returning.
    """
    store = _RecordingStore()
    cfg = MemoryConfig(  # type: ignore[call-arg]
        backend="redis_qdrant",
        embedding_provider="hashing",
        redis_url="redis://127.0.0.1:1",
        qdrant_url="http://127.0.0.1:1",
    )

    mm = await MemoryManager.create(cfg, l1=store, l2=store)

    assert await mm.store_turn("a1", "s1", "u", "a")
    await mm.close()


@pytest.mark.asyncio
async def test_create_uses_injected_embedder():
    store = _RecordingStore()
    embedder = ConstantEmbedder()
    cfg = MemoryConfig(embedding_provider="hashing")  # type: ignore[call-arg]

    mm = await MemoryManager.create(cfg, l1=store, l2=store, embedder=embedder)
    await mm.inject_memory("a1", "a fact")

    assert store.entries[0].embedding == [0.1, 0.2, 0.3, 0.4]
    await mm.close()


@pytest.mark.asyncio
async def test_create_uses_an_injected_extractor():
    """Custom extractors must not require dropping to the constructor the docs warn against."""

    class _StubExtractor:
        def __init__(self) -> None:
            self.calls = 0

        async def extract(self, text: str):
            self.calls += 1
            from actrone_memory import ExtractedFact

            return [ExtractedFact(content="the user is called Alex", importance=0.9)]

    store = _RecordingStore()
    extractor = _StubExtractor()
    cfg = MemoryConfig(embedding_provider="hashing")  # type: ignore[call-arg]

    mm = await MemoryManager.create(cfg, l1=store, l2=store, extractor=extractor)
    await mm.store_turn("a1", "s1", "my name is Alex", "hi Alex")
    ids = await mm.extract_memories("a1", "s1")

    assert extractor.calls == 1, "the injected extractor was not used"
    assert len(ids) == 1
    assert store.entries[0].content == "the user is called Alex"
    await mm.close()


@pytest.mark.asyncio
async def test_create_uses_an_injected_reranker():
    class _ReverseReranker:
        def __init__(self) -> None:
            self.calls = 0

        async def rerank(self, query: str, entries, top_k: int):
            self.calls += 1
            return list(reversed(entries))[:top_k]

    store = _RecordingStore()
    reranker = _ReverseReranker()
    cfg = MemoryConfig(embedding_provider="hashing")  # type: ignore[call-arg]

    mm = await MemoryManager.create(cfg, l1=store, l2=store, reranker=reranker)
    await mm.inject_memory("a1", "first fact")
    await mm.inject_memory("a1", "second fact")
    await mm.search_memories("a1", "fact")

    assert reranker.calls == 1, "the injected reranker was not used"
    await mm.close()


class _RendezvousStore(InMemoryStore):
    """Once armed, each tier's read waits until the other tier's read has started.

    A sequential fetch can never satisfy both waits and times out; a concurrent fetch meets in the
    middle straight away. The timeout only bounds a regression, it is not a synchronisation sleep.
    """

    def __init__(self) -> None:
        super().__init__()
        self.armed = False
        self.l1_started = asyncio.Event()
        self.l2_started = asyncio.Event()

    async def get_recent_turns(
        self, agent_id: str, session_id: str, n: int | None = None
    ) -> list[Turn]:
        if self.armed:
            self.l1_started.set()
            await asyncio.wait_for(self.l2_started.wait(), timeout=2)
        return await super().get_recent_turns(agent_id, session_id, n)

    async def search(
        self,
        agent_id: str,
        query_embedding: list[float],
        threshold: float,
        limit: int = 20,
        content_types: list[ContentType] | None = None,
        query_text: str | None = None,
    ) -> list[MemoryEntry]:
        if self.armed:
            self.l2_started.set()
            await asyncio.wait_for(self.l1_started.wait(), timeout=2)
        return await super().search(
            agent_id, query_embedding, threshold, limit, content_types, query_text
        )


@pytest.mark.asyncio
async def test_retrieve_context_reads_l1_while_l2_embeds_and_searches():
    """The L1 read overlaps the whole L2 path, so the fetch waits for the slower side only."""
    store = _RendezvousStore()
    cfg = MemoryConfig(embedding_provider="hashing")  # type: ignore[call-arg]
    mm = await MemoryManager.create(cfg, l1=store, l2=store, embedder=ConstantEmbedder())
    await mm.store_turn("a1", "s1", "hello", "hi")

    store.armed = True
    ctx = await mm.retrieve_context("a1", "s1", "hello", token_budget=1000)

    assert store.l1_started.is_set() and store.l2_started.is_set()
    assert len(ctx.recent_turns) == 1
    await mm.close()
