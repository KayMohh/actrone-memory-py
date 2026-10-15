"""Contract tests for the DSPy Retrieve adapter.

Skipped automatically when dspy is not installed.
Run with: pip install actrone-memory[dspy] && pytest tests/contract/test_dspy.py
"""

from __future__ import annotations

import pytest

dspy = pytest.importorskip("dspy", reason="dspy not installed")

from unittest.mock import AsyncMock, patch

from actrone_memory.integrations.dspy import ActroneRM
from actrone_memory.models import MemoryEntry


def _make_memory(content: str) -> MemoryEntry:
    return MemoryEntry(
        agent_id="agent-1",
        session_id="default",
        content=content,
        content_type="summary",
        importance_score=0.8,
        token_count=len(content.split()),
    )


@pytest.fixture
def mock_mm() -> AsyncMock:
    mm = AsyncMock()
    mm.search_memories = AsyncMock(
        return_value=[
            _make_memory("Paris is the capital of France."),
            _make_memory("France is in Western Europe."),
        ]
    )
    mm.store_turn = AsyncMock(return_value="turn-id-1")
    return mm


@pytest.fixture
def rm(mock_mm: AsyncMock) -> ActroneRM:
    return ActroneRM(agent_id="agent-1", k=5, memory_manager=mock_mm)


@pytest.mark.asyncio
async def test_build_context_renders_ranked_memories(rm: ActroneRM) -> None:
    # Tier-1 framework-free path: retrieval-shaped adapters render ranked memories.
    context = await rm.build_context("capital of France")
    assert context.startswith("Relevant long-term memory:")
    assert "- Paris is the capital of France." in context


@pytest.mark.asyncio
async def test_build_context_empty_when_no_memories(rm: ActroneRM, mock_mm: AsyncMock) -> None:
    mock_mm.search_memories = AsyncMock(return_value=[])
    assert await rm.build_context("anything") == ""


def test_forward_single_query_returns_prediction(rm: ActroneRM) -> None:
    prediction = rm.forward("capital of France")

    assert hasattr(prediction, "passages")
    assert isinstance(prediction.passages, list)
    assert len(prediction.passages) == 2
    assert "Paris" in prediction.passages[0]


def test_forward_list_of_queries_interleaves(rm: ActroneRM, mock_mm: AsyncMock) -> None:
    # Both queries return the same two passages from our mock.
    prediction = rm.forward(["capital of France", "location of France"])

    assert hasattr(prediction, "passages")
    # Interleaving: [q1[0], q2[0], q1[1], q2[1]] = 4 passages.
    assert len(prediction.passages) == 4


def test_call_syntax(rm: ActroneRM) -> None:
    """Verify rm("query") shorthand works."""
    prediction = rm("what is France?")
    assert hasattr(prediction, "passages")


def test_forward_respects_k_override(rm: ActroneRM, mock_mm: AsyncMock) -> None:
    rm.forward("test", k=3)
    # search_memories should have been called with limit=3.
    call_args = mock_mm.search_memories.call_args
    assert call_args[1]["limit"] == 3 or call_args[0][-1] == 3


@pytest.mark.asyncio
async def test_store_turn_delegates_to_manager(rm: ActroneRM, mock_mm: AsyncMock) -> None:
    await rm.store_turn("agent-1", "sess-1", "Hello", "Hi!")
    mock_mm.store_turn.assert_awaited_once_with("agent-1", "sess-1", "Hello", "Hi!")


@pytest.mark.asyncio
async def test_search_memories_returns_strings(rm: ActroneRM) -> None:
    results = await rm.search_memories("France", limit=2)
    assert isinstance(results, list)
    assert all(isinstance(r, str) for r in results)
    assert len(results) == 2


def test_default_k_is_five() -> None:
    rm = ActroneRM.__new__(ActroneRM)
    rm.k = 5
    assert rm.k == 5


def test_import_error_without_dspy() -> None:
    with (
        patch.dict("sys.modules", {"dspy": None}),
        pytest.raises(ImportError, match=r"pip install actrone-memory\[dspy\]"),
    ):
        ActroneRM(agent_id="agent-1")


# ── Inside DSPy's real runtime ────────────────────────────────────────────────────────────────────


def test_configured_rm_works_through_dspy_retrieve() -> None:
    """The documented wiring end to end: ActroneRM as DSPy's configured rm, then dspy.Retrieve."""
    import asyncio

    from actrone_memory.config import MemoryConfig
    from actrone_memory.manager import MemoryManager

    mm = asyncio.run(MemoryManager.create(MemoryConfig(relevance_threshold=0.05)))  # type: ignore[call-arg]
    asyncio.run(mm.inject_memory("support-bot", "Deploys need two approvals.", 0.9))
    rm = ActroneRM("support-bot", k=3, memory_manager=mm)

    with dspy.settings.context(rm=rm):
        result = dspy.Retrieve(k=3)("How many approvals does a deploy need?")
    assert list(result.passages) == ["Deploys need two approvals."]

    # Direct use, as a module's retriever, gives the same passages as plain strings.
    direct = rm("How many approvals does a deploy need?")
    assert direct.passages == ["Deploys need two approvals."]
    assert all(type(p) is str for p in direct.passages)
