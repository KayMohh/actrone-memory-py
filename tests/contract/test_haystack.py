"""Contract tests for the Haystack v2 adapters.

Skipped automatically when haystack-ai is not installed.
Run with: pip install actrone-memory[haystack] && pytest tests/contract/test_haystack.py
"""

from __future__ import annotations

import pytest

haystack = pytest.importorskip("haystack", reason="haystack-ai not installed")

from unittest.mock import AsyncMock, patch

from actrone_memory.integrations.haystack import ActroneRetriever, ActroneWriter
from actrone_memory.models import MemoryEntry


@pytest.fixture
def mock_mm_retriever() -> AsyncMock:
    mm = AsyncMock()
    mm.search_memories = AsyncMock(
        return_value=[
            MemoryEntry(
                agent_id="agent-1",
                session_id="default",
                content="Paris is the capital of France.",
                content_type="summary",
                importance_score=0.85,
                token_count=10,
            ),
            MemoryEntry(
                agent_id="agent-1",
                session_id="default",
                content="France is a country in Western Europe.",
                content_type="summary",
                importance_score=0.70,
                token_count=10,
            ),
        ]
    )
    return mm


@pytest.fixture
def mock_mm_writer() -> AsyncMock:
    mm = AsyncMock()
    mm.store_turn = AsyncMock(return_value="turn-id-1")
    return mm


@pytest.fixture
def retriever(mock_mm_retriever: AsyncMock) -> ActroneRetriever:
    return ActroneRetriever(agent_id="agent-1", top_k=5, memory_manager=mock_mm_retriever)


@pytest.mark.asyncio
async def test_retriever_build_context_renders_ranked_memories(retriever: ActroneRetriever) -> None:
    # Tier-1 framework-free path: retrieval-shaped adapters render ranked memories.
    context = await retriever.build_context("capital of France")
    assert context.startswith("Relevant long-term memory:")
    assert "- Paris is the capital of France." in context


@pytest.fixture
def writer(mock_mm_writer: AsyncMock) -> ActroneWriter:
    return ActroneWriter(agent_id="agent-1", session_id="default", memory_manager=mock_mm_writer)


@pytest.mark.asyncio
async def test_retriever_returns_haystack_documents(
    retriever: ActroneRetriever, mock_mm_retriever: AsyncMock
) -> None:
    from haystack.dataclasses import Document

    result = await retriever.run_async(query="capital of France")

    assert "documents" in result
    docs = result["documents"]
    assert len(docs) == 2
    assert all(isinstance(d, Document) for d in docs)
    assert docs[0].content == "Paris is the capital of France."
    assert docs[0].meta["importance_score"] == pytest.approx(0.85)


@pytest.mark.asyncio
async def test_retriever_respects_top_k_override(
    retriever: ActroneRetriever, mock_mm_retriever: AsyncMock
) -> None:
    await retriever.run_async(query="France", top_k=3)
    mock_mm_retriever.search_memories.assert_awaited_once_with("agent-1", "France", limit=3)


@pytest.mark.asyncio
async def test_writer_stores_turn(writer: ActroneWriter, mock_mm_writer: AsyncMock) -> None:
    result = await writer.run_async(
        user_message="What is the capital of France?",
        assistant_message="Paris is the capital of France.",
    )
    assert result == {"memories_written": 1}
    mock_mm_writer.store_turn.assert_awaited_once_with(
        "agent-1",
        "default",
        "What is the capital of France?",
        "Paris is the capital of France.",
    )


@pytest.mark.asyncio
async def test_retriever_metadata_contains_memory_id(
    retriever: ActroneRetriever,
) -> None:
    result = await retriever.run_async(query="test")
    doc = result["documents"][0]
    assert "memory_id" in doc.meta
    assert "timestamp" in doc.meta
    assert "topic_tags" in doc.meta


def test_import_error_without_haystack() -> None:
    with (
        patch.dict("sys.modules", {"haystack": None}),
        pytest.raises(ImportError, match=r"pip install actrone-memory\[haystack\]"),
    ):
        ActroneRetriever(agent_id="agent-1")

    with (
        patch.dict("sys.modules", {"haystack": None}),
        pytest.raises(ImportError, match=r"pip install actrone-memory\[haystack\]"),
    ):
        ActroneWriter(agent_id="agent-1")


# ── Inside a real Haystack pipeline ───────────────────────────────────────────────────────────────


def test_components_are_real_haystack_components_accepted_by_a_pipeline() -> None:
    from haystack import Pipeline

    retriever = ActroneRetriever(agent_id="agent-1", memory_manager=AsyncMock())
    writer = ActroneWriter(agent_id="agent-1", memory_manager=AsyncMock())
    assert isinstance(retriever, ActroneRetriever) and isinstance(writer, ActroneWriter)
    pipeline = Pipeline()
    pipeline.add_component("memory", retriever)  # rejects anything not built with @component
    pipeline.add_component("writer", writer)


def test_retriever_runs_in_a_pipeline_and_returns_documents() -> None:
    import asyncio

    from haystack import Pipeline
    from haystack.dataclasses import Document

    from actrone_memory.config import MemoryConfig
    from actrone_memory.manager import MemoryManager

    mm = asyncio.run(MemoryManager.create(MemoryConfig(relevance_threshold=0.05)))  # type: ignore[call-arg]
    asyncio.run(mm.inject_memory("agent-1", "The nightly index rebuild runs at 2am.", 0.9))
    pipeline = Pipeline()
    pipeline.add_component("memory", ActroneRetriever(agent_id="agent-1", memory_manager=mm))

    documents = pipeline.run({"memory": {"query": "index rebuild"}})["memory"]["documents"]
    assert documents and all(isinstance(d, Document) for d in documents)
    assert documents[0].content == "The nightly index rebuild runs at 2am."


def test_writer_runs_in_a_pipeline_and_stores_the_turn() -> None:
    import asyncio

    from haystack import Pipeline

    from actrone_memory.config import MemoryConfig
    from actrone_memory.manager import MemoryManager

    mm = asyncio.run(MemoryManager.create(MemoryConfig(relevance_threshold=0.05)))  # type: ignore[call-arg]
    pipeline = Pipeline()
    pipeline.add_component(
        "writer", ActroneWriter(agent_id="agent-1", session_id="s1", memory_manager=mm)
    )

    out = pipeline.run(
        {"writer": {"user_message": "When is the rebuild?", "assistant_message": "Nightly at 2am."}}
    )
    assert out == {"writer": {"memories_written": 1}}
    turns = asyncio.run(mm.get_recent_turns("agent-1", "s1"))
    assert [(t.user_message, t.assistant_message) for t in turns] == [
        ("When is the rebuild?", "Nightly at 2am.")
    ]
