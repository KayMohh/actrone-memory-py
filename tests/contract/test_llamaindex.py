"""Contract tests for the LlamaIndex adapter.

Skipped automatically when llama-index-core is not installed.
Run with: pip install actrone-memory[llamaindex] && pytest tests/contract/test_llamaindex.py
"""

from __future__ import annotations

from typing import Any

import pytest

llama_index = pytest.importorskip("llama_index", reason="llama-index-core not installed")

from unittest.mock import AsyncMock, MagicMock, patch

from actrone_memory.integrations.llamaindex import ActroneLlamaMemory
from actrone_memory.models import MemoryEntry


@pytest.fixture
def mock_mm() -> AsyncMock:
    mm = AsyncMock()
    mm.store_turn = AsyncMock(return_value="turn-id-1")
    mm.clear_session = AsyncMock()
    mm.retrieve_context = AsyncMock(
        return_value=MagicMock(
            recent_turns=[
                MagicMock(user_message="What is 2+2?", assistant_message="4"),
            ],
            episodic_memories=[
                MemoryEntry(
                    agent_id="agent-1",
                    session_id="default",
                    content="Arithmetic is fun.",
                    content_type="summary",
                    importance_score=0.5,
                    token_count=5,
                )
            ],
        )
    )
    return mm


@pytest.fixture
def memory(mock_mm: AsyncMock) -> ActroneLlamaMemory:
    return ActroneLlamaMemory(
        agent_id="agent-1",
        session_id="default",
        memory_manager=mock_mm,
    )


@pytest.mark.asyncio
async def test_build_context_returns_governed_string(memory: ActroneLlamaMemory) -> None:
    # Tier-1 framework-free path: a single governed system-context block.
    context = await memory.build_context("what is 2+2?")
    assert "Arithmetic is fun." in context  # episodic memory
    assert "What is 2+2?" in context  # recent turn


@pytest.mark.asyncio
async def test_aget_returns_chat_messages(memory: ActroneLlamaMemory) -> None:
    from llama_index.core.base.llms.types import MessageRole

    messages = await memory.aget(input="what is 2+2?")

    # Should include system message with episodic memory + 2 turn messages.
    assert len(messages) >= 3
    roles = [m.role for m in messages]
    assert MessageRole.SYSTEM in roles
    assert MessageRole.USER in roles
    assert MessageRole.ASSISTANT in roles


@pytest.mark.asyncio
async def test_aput_user_message_buffered(memory: ActroneLlamaMemory, mock_mm: AsyncMock) -> None:
    from llama_index.core.base.llms.types import ChatMessage, MessageRole

    await memory.aput(ChatMessage(role=MessageRole.USER, content="Hello"))
    # Not yet stored, waiting for assistant response.
    mock_mm.store_turn.assert_not_awaited()
    assert memory._pending_user_msg == "Hello"


@pytest.mark.asyncio
async def test_aput_full_turn_stores(memory: ActroneLlamaMemory, mock_mm: AsyncMock) -> None:
    from llama_index.core.base.llms.types import ChatMessage, MessageRole

    await memory.aput(ChatMessage(role=MessageRole.USER, content="Hello"))
    await memory.aput(ChatMessage(role=MessageRole.ASSISTANT, content="Hi there!"))

    mock_mm.store_turn.assert_awaited_once_with("agent-1", "default", "Hello", "Hi there!")
    assert memory._pending_user_msg == ""


@pytest.mark.asyncio
async def test_areset_clears_session(memory: ActroneLlamaMemory, mock_mm: AsyncMock) -> None:
    await memory.areset()
    mock_mm.clear_session.assert_awaited_once_with("agent-1", "default")
    assert memory._pending_user_msg == ""


@pytest.mark.asyncio
async def test_aget_all_delegates_to_aget(memory: ActroneLlamaMemory) -> None:
    messages = await memory.aget_all()
    assert isinstance(messages, list)


def test_import_error_without_llamaindex() -> None:
    modules = {"llama_index": None, "llama_index.core": None, "llama_index.core.memory": None}
    with (
        patch.dict("sys.modules", modules),
        pytest.raises(ImportError, match=r"pip install actrone-memory\[llamaindex\]"),
    ):
        ActroneLlamaMemory(agent_id="agent-1")


def _mock_llm_class() -> Any:
    """LlamaIndex's echoing test LLM; it moved from ``llms.mock`` to ``llms`` after 0.10."""
    try:
        from llama_index.core.llms import MockLLM
    except ImportError:
        from llama_index.core.llms.mock import MockLLM
    return MockLLM


# ── Inside LlamaIndex's real chat engine ──────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_memory_works_inside_a_real_simple_chat_engine() -> None:
    """The documented wiring end to end. MockLLM echoes its prompt, showing what the model got."""
    from llama_index.core.chat_engine import SimpleChatEngine

    mock_llm = _mock_llm_class()

    from actrone_memory.config import MemoryConfig
    from actrone_memory.manager import MemoryManager

    mm = await MemoryManager.create(MemoryConfig(relevance_threshold=0.05))  # type: ignore[call-arg]
    await mm.inject_memory("support-bot", "Deploys need two approvals.", 0.9)
    memory = ActroneLlamaMemory("support-bot", "s1", memory_manager=mm)
    engine = SimpleChatEngine.from_defaults(llm=mock_llm(), memory=memory)

    first = str(await engine.achat("How many approvals does a deploy need?"))
    # The model received the recalled memory AND the question being asked.
    assert "Deploys need two approvals." in first
    assert "How many approvals does a deploy need?" in first

    second = str(await engine.achat("Even on Fridays?"))
    # The second prompt carries the first exchange back from actrone-memory, then the new question.
    assert "How many approvals does a deploy need?" in second
    assert second.rstrip().endswith("user: Even on Fridays?\nassistant:")

    turns = await mm.get_recent_turns("support-bot", "s1")
    assert [t.user_message for t in turns] == [
        "How many approvals does a deploy need?",
        "Even on Fridays?",
    ]
    await mm.close()


@pytest.mark.asyncio
async def test_chat_history_passed_to_the_engine_replaces_the_session() -> None:
    from llama_index.core.base.llms.types import ChatMessage, MessageRole
    from llama_index.core.chat_engine import SimpleChatEngine

    mock_llm = _mock_llm_class()

    from actrone_memory.config import MemoryConfig
    from actrone_memory.manager import MemoryManager

    mm = await MemoryManager.create(MemoryConfig(relevance_threshold=0.05))  # type: ignore[call-arg]
    memory = ActroneLlamaMemory("support-bot", "s2", memory_manager=mm)
    engine = SimpleChatEngine.from_defaults(llm=mock_llm(), memory=memory)
    history = [
        ChatMessage(role=MessageRole.USER, content="What is the build server called?"),
        ChatMessage(role=MessageRole.ASSISTANT, content="It is called atlas."),
    ]
    reply = str(await engine.achat("Where does it run?", chat_history=history))
    assert "It is called atlas." in reply and "Where does it run?" in reply
    turns = await mm.get_recent_turns("support-bot", "s2")
    assert turns[0].user_message == "What is the build server called?"
    await mm.close()
