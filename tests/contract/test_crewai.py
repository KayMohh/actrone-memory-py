"""Contract tests for the CrewAI memory adapter.

Skipped automatically when crewai is not installed.
Run with: pip install actrone-memory[crewai] && pytest tests/contract/test_crewai.py
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from typing import Any

import pytest

crewai = pytest.importorskip("crewai", reason="crewai not installed")

from unittest.mock import AsyncMock, patch

from actrone_memory.integrations.crewai import ActroneCrewMemory
from actrone_memory.models import MemoryEntry


@pytest.fixture
def mock_mm() -> AsyncMock:
    mm = AsyncMock()
    mm.store_turn = AsyncMock(return_value="turn-1")
    mm.clear_session = AsyncMock()
    mm.search_memories = AsyncMock(
        return_value=[
            MemoryEntry(
                agent_id="agent-1",
                session_id="default",
                content="Paris is the capital of France.",
                content_type="summary",
                importance_score=0.9,
                token_count=10,
                timestamp=datetime.now(UTC),
            )
        ]
    )
    return mm


@pytest.fixture
def memory(mock_mm: AsyncMock) -> ActroneCrewMemory:
    return ActroneCrewMemory(agent_id="agent-1", session_id="default", memory_manager=mock_mm)


@pytest.mark.asyncio
async def test_build_context_returns_governed_string(mock_mm: AsyncMock) -> None:
    from unittest.mock import MagicMock

    # Tier-1 framework-free path: a single governed system-context block.
    mock_mm.retrieve_context = AsyncMock(
        return_value=MagicMock(
            recent_turns=[],
            episodic_memories=[
                MemoryEntry(
                    agent_id="agent-1",
                    session_id="default",
                    content="Paris is the capital of France.",
                    content_type="summary",
                    importance_score=0.9,
                    token_count=10,
                    timestamp=datetime.now(UTC),
                )
            ],
        )
    )
    memory = ActroneCrewMemory(agent_id="agent-1", session_id="default", memory_manager=mock_mm)
    context = await memory.build_context("capital of France")
    assert "Paris is the capital of France." in context


@pytest.mark.asyncio
async def test_save_persists_value_with_task_input(
    memory: ActroneCrewMemory, mock_mm: AsyncMock
) -> None:
    await memory.save("The answer is 42.", metadata={"task_input": "the question"})
    mock_mm.store_turn.assert_awaited_once()
    kwargs = mock_mm.store_turn.call_args.kwargs
    assert kwargs["user_message"] == "the question"
    assert kwargs["assistant_message"] == "The answer is 42."


@pytest.mark.asyncio
async def test_search_returns_scored_results(memory: ActroneCrewMemory) -> None:
    results = await memory.search("capital of France", limit=5)
    assert len(results) == 1
    assert results[0]["content"] == "Paris is the capital of France."
    assert results[0]["score"] == 0.9
    assert results[0]["metadata"]["content_type"] == "summary"


@pytest.mark.asyncio
async def test_reset_delegates_to_manager(memory: ActroneCrewMemory, mock_mm: AsyncMock) -> None:
    await memory.reset()
    mock_mm.clear_session.assert_awaited_once_with("agent-1", "default")


def test_import_error_without_crewai() -> None:
    with (
        patch.dict("sys.modules", {"crewai": None}),
        pytest.raises(ImportError, match=r"pip install actrone-memory\[crewai\]"),
    ):
        ActroneCrewMemory(agent_id="agent-1", session_id="default")


# ── Inside CrewAI's real runtime ──────────────────────────────────────────────────────────────────


@pytest.fixture
def offline_crewai(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep CrewAI offline: a placeholder key for agent construction, no telemetry."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-offline")
    monkeypatch.setenv("CREWAI_DISABLE_TELEMETRY", "true")
    monkeypatch.setenv("OTEL_SDK_DISABLED", "true")


@pytest.mark.asyncio
@pytest.mark.usefixtures("offline_crewai")
async def test_context_reaches_a_real_task_and_the_result_is_saved() -> None:
    """The documented wiring: Task(description=context + user_input, agent=agent), then save."""
    from crewai import Agent, Task

    from actrone_memory.config import MemoryConfig
    from actrone_memory.manager import MemoryManager

    mm = await MemoryManager.create(MemoryConfig(relevance_threshold=0.05))  # type: ignore[call-arg]
    await mm.inject_memory("support-bot", "Deploys need two approvals.", 0.9)
    memory = ActroneCrewMemory("support-bot", "s1", memory_manager=mm)

    user_input = "How many approvals does a deploy need?"
    context = await memory.build_context("deploy approvals")
    agent = Agent(role="support", goal="Answer deployment questions", backstory="Release manager")
    task = Task(description=f"{context}\n\n{user_input}", expected_output="A number", agent=agent)
    assert "Deploys need two approvals." in task.description and user_input in task.description

    await memory.save("Two approvals.", {"task_input": user_input})
    turns = await mm.get_recent_turns("support-bot", "s1")
    assert [(t.user_message, t.assistant_message) for t in turns] == [
        (user_input, "Two approvals.")
    ]
    await mm.close()


@pytest.mark.asyncio
@pytest.mark.usefixtures("offline_crewai")
async def test_memory_reaches_the_model_in_a_real_crew_kickoff() -> None:
    """A whole crew run with an offline stub LLM (CrewAI's BaseLLM; older CrewAI lacks it)."""
    try:
        from crewai.llms.base_llm import BaseLLM
    except ImportError:
        pytest.skip(
            "this CrewAI version has no BaseLLM for an offline model; construction is tested above"
        )
    from crewai import Agent, Crew, Task

    from actrone_memory.config import MemoryConfig
    from actrone_memory.manager import MemoryManager

    seen: list[str] = []

    class _StubLLM(BaseLLM):  # type: ignore[misc]
        def call(self, messages: Any, *_: Any, **__: Any) -> str:
            seen.append(
                messages
                if isinstance(messages, str)
                else " ".join(str(m.get("content", "")) for m in messages)
            )
            return "Thought: I know the answer.\nFinal Answer: Two approvals."

    mm = await MemoryManager.create(MemoryConfig(relevance_threshold=0.05))  # type: ignore[call-arg]
    await mm.inject_memory("support-bot", "Deploys need two approvals.", 0.9)
    memory = ActroneCrewMemory("support-bot", "s1", memory_manager=mm)

    user_input = "How many approvals does a deploy need?"
    context = await memory.build_context("deploy approvals")
    agent = Agent(
        role="support",
        goal="Answer deployment questions",
        backstory="Release manager",
        llm=_StubLLM(model="stub"),
    )
    task = Task(description=f"{context}\n\n{user_input}", expected_output="A number", agent=agent)
    result = await asyncio.to_thread(Crew(agents=[agent], tasks=[task]).kickoff)

    assert "Two approvals." in str(result)
    assert any("Deploys need two approvals." in prompt for prompt in seen)
    await memory.save(str(result), {"task_input": user_input})
    assert (await mm.get_recent_turns("support-bot", "s1"))[0].user_message == user_input
    await mm.close()
