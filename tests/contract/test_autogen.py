"""Contract tests for the AutoGen 0.4 adapter.

Skipped automatically when autogen-core is not installed.
Run with: pip install actrone-memory[autogen] && pytest tests/contract/test_autogen.py
"""

from __future__ import annotations

import pytest

autogen_core = pytest.importorskip("autogen_core", reason="autogen-core not installed")

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

from actrone_memory.integrations.autogen import ActroneAutoGenMemory
from actrone_memory.models import MemoryEntry


@pytest.fixture
def mock_mm() -> AsyncMock:
    mm = AsyncMock()
    mm.search_memories = AsyncMock(
        return_value=[
            MemoryEntry(
                agent_id="agent-1",
                session_id="default",
                content="Paris is the capital of France.",
                content_type="summary",
                importance_score=0.9,
                token_count=10,
            )
        ]
    )
    mm.retrieve_context = AsyncMock(return_value=MagicMock(recent_turns=[], episodic_memories=[]))
    mm.inject_memory = AsyncMock(return_value="mem-id-1")
    mm.clear_session = AsyncMock()
    return mm


@pytest.fixture
def memory(mock_mm: AsyncMock) -> ActroneAutoGenMemory:
    return ActroneAutoGenMemory(
        agent_id="agent-1",
        session_id="default",
        memory_manager=mock_mm,
    )


@pytest.mark.asyncio
async def test_build_context_returns_governed_string(mock_mm: AsyncMock) -> None:
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
                )
            ],
        )
    )
    memory = ActroneAutoGenMemory(agent_id="agent-1", session_id="default", memory_manager=mock_mm)
    context = await memory.build_context("capital of France")
    assert "Paris is the capital of France." in context


@pytest.mark.asyncio
async def test_add_text_content(memory: ActroneAutoGenMemory, mock_mm: AsyncMock) -> None:
    from autogen_core.memory import MemoryContent, MemoryMimeType

    content = MemoryContent(
        content="The Eiffel Tower is in Paris.",
        mime_type=MemoryMimeType.TEXT,
        metadata={"importance": 0.8},
    )
    await memory.add(content)
    mock_mm.inject_memory.assert_awaited_once()
    call_kwargs = mock_mm.inject_memory.call_args
    assert call_kwargs[0][1] == "The Eiffel Tower is in Paris."


@pytest.mark.asyncio
async def test_query_returns_memory_query_result(
    memory: ActroneAutoGenMemory, mock_mm: AsyncMock
) -> None:
    from autogen_core.memory import MemoryContent, MemoryMimeType, MemoryQueryResult

    # AutoGen's Memory.query takes a plain string or a MemoryContent; both must search the text.
    content = MemoryContent(content="capital of France", mime_type=MemoryMimeType.TEXT)
    for query in ("capital of France", content):
        result = await memory.query(query)
        assert isinstance(result, MemoryQueryResult)
        assert len(result.results) == 1
        assert result.results[0].content == "Paris is the capital of France."
        assert result.results[0].mime_type == MemoryMimeType.TEXT
        assert mock_mm.search_memories.await_args.args[1] == "capital of France"


@pytest.mark.asyncio
async def test_clear_delegates_to_manager(memory: ActroneAutoGenMemory, mock_mm: AsyncMock) -> None:
    await memory.clear()
    mock_mm.clear_session.assert_awaited_once_with("agent-1", "default")


@pytest.mark.asyncio
async def test_close_is_noop(memory: ActroneAutoGenMemory) -> None:
    # close() should not raise and should not call any manager method.
    await memory.close()


@pytest.mark.asyncio
async def test_add_non_text_content_is_noop(
    memory: ActroneAutoGenMemory, mock_mm: AsyncMock
) -> None:
    from autogen_core.memory import MemoryContent, MemoryMimeType

    content = MemoryContent(
        content=b"\x89PNG",  # binary image, not supported
        mime_type=MemoryMimeType.IMAGE,
        metadata={},
    )
    await memory.add(content)
    mock_mm.inject_memory.assert_not_awaited()


def test_import_error_without_autogen() -> None:
    """Verify ImportError is raised with a helpful message when autogen is absent."""
    modules = {"autogen_core": None, "autogen_core.memory": None}
    with (
        patch.dict("sys.modules", modules),
        pytest.raises(ImportError, match=r"pip install actrone-memory\[autogen\]"),
    ):
        ActroneAutoGenMemory(agent_id="agent-1")


# ── Inside AutoGen's real runtime ─────────────────────────────────────────────────────────────────


def _stub_model_client(reply: str, seen: list[list[str]]) -> Any:
    """A ChatCompletionClient that records what the model is sent and answers with ``reply``.

    Offline stand-in for an LLM (AutoGen's replay client lives in the separate autogen-ext package).
    """
    from autogen_core.models import ChatCompletionClient, CreateResult, RequestUsage

    usage = RequestUsage(prompt_tokens=0, completion_tokens=0)
    info = {
        "vision": False,
        "function_calling": False,
        "json_output": False,
        "family": "unknown",
        "structured_output": False,
    }

    class _Stub(ChatCompletionClient):  # type: ignore[misc]
        async def create(self, messages: Any, **_: Any) -> Any:
            seen.append([str(getattr(m, "content", "")) for m in messages])
            return CreateResult(finish_reason="stop", content=reply, usage=usage, cached=False)

        def create_stream(self, messages: Any, **_: Any) -> Any:
            raise NotImplementedError

        async def close(self) -> None:
            return None

        def actual_usage(self) -> Any:
            return usage

        def total_usage(self) -> Any:
            return usage

        def count_tokens(self, messages: Any, **_: Any) -> int:
            return 0

        def remaining_tokens(self, messages: Any, **_: Any) -> int:
            return 100_000

        @property
        def capabilities(self) -> Any:
            return info

        @property
        def model_info(self) -> Any:
            return info

    return _Stub()


@pytest.mark.asyncio
async def test_update_context_returns_autogens_result_shape() -> None:
    """The hook AutoGen calls before every model call must return its UpdateContextResult."""
    from autogen_core.memory import MemoryQueryResult, UpdateContextResult
    from autogen_core.model_context import UnboundedChatCompletionContext
    from autogen_core.models import UserMessage

    from actrone_memory.config import MemoryConfig
    from actrone_memory.manager import MemoryManager

    mm = await MemoryManager.create(MemoryConfig(relevance_threshold=0.05))  # type: ignore[call-arg]
    await mm.inject_memory("support-bot", "Deploys need two approvals.", 0.9)
    memory = ActroneAutoGenMemory("support-bot", "s1", memory_manager=mm)
    context = UnboundedChatCompletionContext()
    await context.add_message(
        UserMessage(content="How many approvals does a deploy need?", source="user")
    )

    result = await memory.update_context(context)
    assert isinstance(result, UpdateContextResult)
    assert isinstance(result.memories, MemoryQueryResult)
    assert [m.content for m in result.memories.results] == ["Deploys need two approvals."]
    assert any(
        "Deploys need two approvals." in str(m.content) for m in await context.get_messages()
    )
    await mm.close()


@pytest.mark.asyncio
async def test_memory_reaches_the_model_in_a_real_assistant_agent() -> None:
    """The documented wiring end to end: AssistantAgent(..., model_client=..., memory=[memory])."""
    from autogen_agentchat.agents import AssistantAgent

    from actrone_memory.config import MemoryConfig
    from actrone_memory.manager import MemoryManager

    mm = await MemoryManager.create(MemoryConfig(relevance_threshold=0.05))  # type: ignore[call-arg]
    await mm.inject_memory("support-bot", "Deploys need two approvals.", 0.9)
    memory = ActroneAutoGenMemory("support-bot", "s1", memory_manager=mm)
    seen: list[list[str]] = []
    agent = AssistantAgent(
        "support", model_client=_stub_model_client("Two approvals.", seen), memory=[memory]
    )

    result = await agent.run(task="How many approvals does a deploy need?")
    assert result.messages[-1].content == "Two approvals."
    # The model call carried the recalled memory alongside the task.
    sent = " ".join(seen[0])
    assert (
        "Deploys need two approvals." in sent and "How many approvals does a deploy need?" in sent
    )
    await mm.close()
