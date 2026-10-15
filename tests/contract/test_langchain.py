"""Contract tests for the LangChain adapters.

Two Tier-2 surfaces, gated differently because LangChain changed shape between majors:

* ``ActroneMemory`` needs ``langchain_core.memory.BaseMemory``, removed in LangChain 1.x, so
  its tests are skipped on 1.x and a dedicated test asserts the version-specific error there.
* ``ActroneChatMessageHistory`` targets ``BaseChatMessageHistory``, identical across 0.x and
  1.x, so its tests run on every supported version.

Skipped entirely when langchain-core is not installed.
Run with: pip install actrone-memory[langchain] && pytest tests/contract/test_langchain.py
"""

from __future__ import annotations

import pytest

langchain_core = pytest.importorskip("langchain_core", reason="langchain-core not installed")

import importlib.util
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

from actrone_memory.integrations.langchain import ActroneChatMessageHistory, ActroneMemory
from actrone_memory.models import MemoryEntry, Turn

_HAS_BASE_MEMORY = importlib.util.find_spec("langchain_core.memory") is not None

requires_base_memory = pytest.mark.skipif(
    not _HAS_BASE_MEMORY, reason="langchain_core.memory was removed in LangChain 1.x"
)
requires_no_base_memory = pytest.mark.skipif(
    _HAS_BASE_MEMORY, reason="only meaningful on LangChain 1.x, which dropped BaseMemory"
)


@pytest.fixture
def mock_mm() -> AsyncMock:
    mm = AsyncMock()
    mm.retrieve_context = AsyncMock(
        return_value=MagicMock(
            recent_turns=[MagicMock(user_message="What is 2+2?", assistant_message="4")],
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
    mm.store_turn = AsyncMock(return_value="turn-1")
    mm.clear_session = AsyncMock()
    mm.get_recent_turns = AsyncMock(return_value=[])
    return mm


# ── Tier 2 on LangChain 0.x: BaseMemory ──────────────────────────────────────


@pytest.fixture
def memory(mock_mm: AsyncMock) -> ActroneMemory:
    return ActroneMemory(agent_id="agent-1", session_id="default", memory_manager=mock_mm)


@requires_base_memory
@pytest.mark.asyncio
async def test_build_context_returns_governed_string(memory: ActroneMemory) -> None:
    # Tier-1 framework-free path: a single governed system-context block.
    context = await memory.build_context("what is 2+2?")
    assert "Arithmetic is fun." in context  # episodic memory
    assert "What is 2+2?" in context  # recent turn


@requires_base_memory
@pytest.mark.asyncio
async def test_build_context_empty_when_nothing_relevant(mock_mm: AsyncMock) -> None:
    mock_mm.retrieve_context = AsyncMock(
        return_value=MagicMock(recent_turns=[], episodic_memories=[])
    )
    memory = ActroneMemory(agent_id="agent-1", session_id="default", memory_manager=mock_mm)
    assert await memory.build_context("anything") == ""


@requires_base_memory
def test_memory_variables_is_history(memory: ActroneMemory) -> None:
    assert memory.memory_variables == ["history"]


@requires_base_memory
@pytest.mark.asyncio
async def test_load_memory_variables_returns_history_with_turns_and_memories(
    memory: ActroneMemory,
) -> None:
    out = await memory.load_memory_variables({"input": "what is 2+2?"})
    assert "history" in out
    assert "Human: What is 2+2?" in out["history"]
    assert "AI: 4" in out["history"]
    assert "Arithmetic is fun." in out["history"]


@requires_base_memory
@pytest.mark.asyncio
async def test_save_context_persists_the_turn(memory: ActroneMemory, mock_mm: AsyncMock) -> None:
    await memory.save_context({"input": "hi"}, {"response": "hello"})
    mock_mm.store_turn.assert_awaited_once_with("agent-1", "default", "hi", "hello")


@requires_base_memory
@pytest.mark.asyncio
async def test_clear_delegates_to_manager(memory: ActroneMemory, mock_mm: AsyncMock) -> None:
    await memory.clear()
    mock_mm.clear_session.assert_awaited_once_with("agent-1", "default")


@requires_no_base_memory
def test_base_memory_adapter_explains_itself_on_langchain_1x(mock_mm: AsyncMock) -> None:
    """On 1.x the error must name the alternatives, not just fail to import."""
    with pytest.raises(ImportError) as exc_info:
        ActroneMemory(agent_id="agent-1", session_id="default", memory_manager=mock_mm)

    message = str(exc_info.value)
    assert "ActroneChatMessageHistory" in message
    assert "removed the BaseMemory abstraction" in message
    assert "langchain-core<1" in message


def test_import_error_without_langchain() -> None:
    modules = {"langchain_core": None, "langchain_core.memory": None}
    with (
        patch.dict("sys.modules", modules),
        pytest.raises(ImportError, match=r"pip install actrone-memory\[langchain\]"),
    ):
        ActroneMemory(agent_id="agent-1", session_id="default")


# ── Tier 2 across both majors: BaseChatMessageHistory ────────────────────────


@pytest.fixture
def history(mock_mm: AsyncMock) -> ActroneChatMessageHistory:
    return ActroneChatMessageHistory(
        agent_id="agent-1", session_id="default", memory_manager=mock_mm
    )


def test_chat_history_satisfies_the_base_interface(history: ActroneChatMessageHistory) -> None:
    """Structural check against the real abstract class, which exists in both majors."""
    from langchain_core.chat_history import BaseChatMessageHistory

    for name in ("aget_messages", "aadd_messages", "aclear", "add_messages", "clear"):
        assert hasattr(BaseChatMessageHistory, name), f"upstream dropped {name}"
        assert hasattr(history, name), f"adapter missing {name}"


@pytest.mark.asyncio
async def test_chat_history_pairs_human_then_ai_into_one_turn(
    history: ActroneChatMessageHistory, mock_mm: AsyncMock
) -> None:
    from langchain_core.messages import AIMessage, HumanMessage

    await history.aadd_messages([HumanMessage(content="hi"), AIMessage(content="hello")])

    mock_mm.store_turn.assert_awaited_once_with("agent-1", "default", "hi", "hello")


@pytest.mark.asyncio
async def test_chat_history_flushes_an_unpaired_human_rather_than_dropping_it(
    history: ActroneChatMessageHistory, mock_mm: AsyncMock
) -> None:
    from langchain_core.messages import HumanMessage

    await history.aadd_messages([HumanMessage(content="first"), HumanMessage(content="second")])

    # "first" is stored with an empty response; "second" stays pending for the next AI reply.
    mock_mm.store_turn.assert_awaited_once_with("agent-1", "default", "first", "")


@pytest.mark.asyncio
async def test_chat_history_ignores_system_and_tool_messages(
    history: ActroneChatMessageHistory, mock_mm: AsyncMock
) -> None:
    from langchain_core.messages import SystemMessage

    await history.aadd_messages([SystemMessage(content="you are helpful")])

    mock_mm.store_turn.assert_not_awaited()


@pytest.mark.asyncio
async def test_chat_history_reads_turns_back_as_messages(
    history: ActroneChatMessageHistory, mock_mm: AsyncMock
) -> None:
    mock_mm.get_recent_turns = AsyncMock(
        return_value=[
            Turn(session_id="default", user_message="hi", assistant_message="hello"),
            Turn(session_id="default", user_message="unanswered", assistant_message=""),
        ]
    )

    messages = await history.aget_messages()

    assert [m.type for m in messages] == ["human", "ai", "human"]
    assert [m.content for m in messages] == ["hi", "hello", "unanswered"]


@pytest.mark.asyncio
async def test_chat_history_handles_content_block_lists(
    history: ActroneChatMessageHistory, mock_mm: AsyncMock
) -> None:
    """LangChain 1.x messages may carry a list of content blocks instead of a string."""
    from langchain_core.messages import AIMessage, HumanMessage

    blocks = [{"type": "text", "text": "block "}, {"type": "text", "text": "text"}]
    await history.aadd_messages([HumanMessage(content=blocks), AIMessage(content="ok")])

    mock_mm.store_turn.assert_awaited_once_with("agent-1", "default", "block text", "ok")


@pytest.mark.asyncio
async def test_chat_history_clear_resets_pending_and_session(
    history: ActroneChatMessageHistory, mock_mm: AsyncMock
) -> None:
    from langchain_core.messages import AIMessage, HumanMessage

    await history.aadd_messages([HumanMessage(content="pending")])
    await history.aclear()
    await history.aadd_messages([AIMessage(content="reply")])

    mock_mm.clear_session.assert_awaited_once_with("agent-1", "default")
    # The cleared pending human must not resurface attached to the next AI message.
    mock_mm.store_turn.assert_awaited_once_with("agent-1", "default", "", "reply")


@pytest.mark.parametrize(
    ("member", "args", "alternative"),
    [
        ("add_messages", ([],), "aadd_messages"),
        ("add_message", (None,), "aadd_messages"),
        ("clear", (), "aclear"),
    ],
)
def test_chat_history_sync_members_point_at_the_async_one(
    history: ActroneChatMessageHistory, member: str, args: tuple[object, ...], alternative: str
) -> None:
    with pytest.raises(NotImplementedError, match=alternative):
        getattr(history, member)(*args)


def test_chat_history_sync_messages_property_points_at_async(
    history: ActroneChatMessageHistory,
) -> None:
    with pytest.raises(NotImplementedError, match="aget_messages"):
        _ = history.messages


# ── Inside LangChain's real runtime ───────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_chat_history_works_inside_runnable_with_message_history() -> None:
    """The documented wiring, end to end: history reaches the model and each exchange is stored."""
    from langchain_core.language_models.fake_chat_models import FakeListChatModel
    from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
    from langchain_core.runnables import RunnableLambda
    from langchain_core.runnables.history import RunnableWithMessageHistory

    from actrone_memory.config import MemoryConfig
    from actrone_memory.manager import MemoryManager

    mm = await MemoryManager.create(MemoryConfig(relevance_threshold=0.05))  # type: ignore[call-arg]
    history = ActroneChatMessageHistory("support-bot", "s1", memory_manager=mm)
    seen: list[list[str]] = []

    def capture(prompt: Any) -> Any:
        seen.append([str(m.content) for m in prompt.to_messages()])
        return prompt

    prompt = ChatPromptTemplate.from_messages(
        [MessagesPlaceholder("history"), ("human", "{input}")]
    )
    model = FakeListChatModel(responses=["Two approvals.", "Yes, on Fridays too."])
    chain = RunnableWithMessageHistory(
        prompt | RunnableLambda(capture) | model,
        lambda _session_id: history,  # type: ignore[arg-type,return-value]
        input_messages_key="input",
        history_messages_key="history",
    )
    config = {"configurable": {"session_id": "s1"}}

    first = await chain.ainvoke({"input": "How many approvals does a deploy need?"}, config)
    assert first.content == "Two approvals."
    second = await chain.ainvoke({"input": "Even on Fridays?"}, config)
    assert second.content == "Yes, on Fridays too."

    # The second call's prompt carried the first exchange back from actrone-memory.
    assert seen[1][:2] == ["How many approvals does a deploy need?", "Two approvals."]
    turns = await mm.get_recent_turns("support-bot", "s1")
    assert [(t.user_message, t.assistant_message) for t in turns] == [
        ("How many approvals does a deploy need?", "Two approvals."),
        ("Even on Fridays?", "Yes, on Fridays too."),
    ]
    await mm.close()
