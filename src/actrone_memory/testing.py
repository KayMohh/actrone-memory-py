"""Conformance suite for custom :class:`L1Store` / :class:`L2Store` implementations.

``MemoryManager`` depends on two ``Protocol`` seams rather than on Redis and Qdrant, so any
backend can be plugged in (pgvector, Weaviate, Valkey, Postgres, and so on). The protocols
only describe *shape*, though: they cannot express that recent turns come back oldest-first,
or that a search must never return another agent's memories. This module encodes those
behavioural requirements as runnable checks, so a third-party adapter can prove it satisfies
the same contract the built-in stores do.

It ships in the package (not in the test suite) precisely so you can run it against your own
store without vendoring anything::

    import asyncio
    from actrone_memory.testing import check_l1_store, check_l2_store

    asyncio.run(check_l1_store(lambda: MyRedisLikeStore(...)))
    asyncio.run(check_l2_store(lambda: MyPgVectorStore(...), dimensions=8))

Each check raises :class:`ConformanceError` with a message naming the violated requirement.
Pass a *factory* rather than an instance: several checks need a pristine store, and a factory
lets the suite isolate them. Requires no pytest and no network; bring your own backend.

The checks are written against the same expectations ``MemoryManager`` itself relies on, so a
store that passes here is safe to hand to ``MemoryManager.create(l1=..., l2=...)``.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from contextlib import suppress
from typing import TYPE_CHECKING, Any

from actrone_memory.exceptions import ActroneMemoryError, MemoryNotFoundError
from actrone_memory.models import MemoryEntry, Turn

if TYPE_CHECKING:  # pragma: no cover - typing only
    from actrone_memory.protocols import L1Store, L2Store

__all__ = [
    "ConformanceError",
    "check_l1_store",
    "check_l2_store",
]


class ConformanceError(ActroneMemoryError):
    """Raised when a store violates a documented behavioural requirement."""

    def __init__(self, requirement: str, detail: str) -> None:
        super().__init__(
            f"{requirement}: {detail}",
            code="ERR_STORE_CONFORMANCE",
            details={"requirement": requirement, "detail": detail},
        )
        self.requirement = requirement


def _require(condition: bool, requirement: str, detail: str) -> None:
    if not condition:
        raise ConformanceError(requirement, detail)


def _turn(session_id: str, user: str, assistant: str, tokens: int = 4) -> Turn:
    return Turn(
        session_id=session_id,
        user_message=user,
        assistant_message=assistant,
        token_count=tokens,
    )


def _entry(
    agent_id: str,
    content: str,
    embedding: list[float],
    *,
    session_id: str = "s1",
    content_type: str = "injected",
    entry_id: str | None = None,
) -> MemoryEntry:
    kwargs: dict[str, Any] = {
        "agent_id": agent_id,
        "session_id": session_id,
        "content": content,
        "content_type": content_type,
        "embedding": embedding,
        "token_count": max(1, len(content) // 4),
    }
    if entry_id is not None:
        kwargs["id"] = entry_id
    return MemoryEntry(**kwargs)


def _unit_vector(dimensions: int, hot_index: int) -> list[float]:
    """A one-hot vector, so cosine similarity between two of them is 0.0 or 1.0."""
    vector = [0.0] * dimensions
    vector[hot_index % dimensions] = 1.0
    return vector


async def check_l1_store(factory: Callable[[], L1Store | Awaitable[L1Store]]) -> None:
    """Verify an :class:`L1Store` implementation against the hot-tier contract.

    ``factory`` returns a fresh, empty store (it may be a coroutine function). Anything it
    returns is closed before this function exits.

    Requirements checked:

    * an appended turn is readable back
    * turns are returned oldest first, which is the order prompts are assembled in
    * ``n`` returns the *most recent* ``n`` turns, not the first ``n``
    * ``turn_count`` agrees with what is readable
    * sessions and agents are isolated from each other
    * ``clear_session`` empties only the session it was given
    * ``get_session_metadata`` returns ``None`` for an unknown session
    * ``try_acquire_summary_lock`` admits exactly one holder per window
    """
    store = await _make(factory)
    try:
        # An appended turn must be readable back.
        await store.append_turn("agent-a", "s1", _turn("s1", "first", "reply-1"))
        turns = await store.get_recent_turns("agent-a", "s1")
        _require(
            len(turns) == 1 and turns[0].user_message == "first",
            "append_turn/get_recent_turns round-trip",
            f"expected the appended turn back, got {turns!r}",
        )

        # Chronological order: the manager renders turns in this order into the prompt.
        await store.append_turn("agent-a", "s1", _turn("s1", "second", "reply-2"))
        await store.append_turn("agent-a", "s1", _turn("s1", "third", "reply-3"))
        turns = await store.get_recent_turns("agent-a", "s1")
        _require(
            [t.user_message for t in turns] == ["first", "second", "third"],
            "get_recent_turns ordering",
            f"turns must come back oldest first, got {[t.user_message for t in turns]!r}",
        )

        # `n` must window from the END, otherwise recall silently returns stale context.
        recent = await store.get_recent_turns("agent-a", "s1", n=2)
        _require(
            [t.user_message for t in recent] == ["second", "third"],
            "get_recent_turns(n) windowing",
            f"n=2 must return the 2 most recent turns oldest-first, got "
            f"{[t.user_message for t in recent]!r}",
        )

        count = await store.turn_count("agent-a", "s1")
        _require(
            count == 3,
            "turn_count accuracy",
            f"expected 3 turns, got {count}",
        )

        # Session isolation.
        await store.append_turn("agent-a", "s2", _turn("s2", "other-session", "reply"))
        s1 = await store.get_recent_turns("agent-a", "s1")
        _require(
            len(s1) == 3,
            "session isolation",
            f"writing to another session changed this one, got {len(s1)} turns",
        )

        # Agent isolation: the governance property that matters most.
        await store.append_turn("agent-b", "s1", _turn("s1", "other-agent", "reply"))
        a_turns = await store.get_recent_turns("agent-a", "s1")
        _require(
            all(t.user_message != "other-agent" for t in a_turns),
            "agent isolation",
            "another agent's turn leaked into this agent's session",
        )

        # Metadata for a live session, then for one that does not exist.
        meta = await store.get_session_metadata("agent-a", "s1")
        _require(
            meta is not None and meta.turn_count == 3,
            "get_session_metadata turn_count",
            f"expected metadata reporting 3 turns, got {meta!r}",
        )
        missing = await store.get_session_metadata("agent-a", "no-such-session")
        _require(
            missing is None,
            "get_session_metadata for an unknown session",
            f"must return None, got {missing!r}",
        )

        # clear_session must be scoped.
        await store.clear_session("agent-a", "s1")
        cleared = await store.get_recent_turns("agent-a", "s1")
        survivor = await store.get_recent_turns("agent-a", "s2")
        _require(
            cleared == [],
            "clear_session empties the session",
            f"expected no turns after clear, got {cleared!r}",
        )
        _require(
            len(survivor) == 1,
            "clear_session scope",
            "clearing one session must not clear another",
        )

        # Exactly one holder per cooldown window, or duplicate summaries fan out.
        first = await store.try_acquire_summary_lock("agent-a", "s3", 60)
        second = await store.try_acquire_summary_lock("agent-a", "s3", 60)
        _require(
            first is True,
            "try_acquire_summary_lock admits the first caller",
            f"expected True on an unheld lock, got {first!r}",
        )
        _require(
            second is False,
            "try_acquire_summary_lock excludes the second caller",
            f"expected False while the lock is held, got {second!r}",
        )
    finally:
        await _close(store)


async def check_l2_store(
    factory: Callable[[], L2Store | Awaitable[L2Store]],
    *,
    dimensions: int = 8,
) -> None:
    """Verify an :class:`L2Store` implementation against the long-term-tier contract.

    ``factory`` returns a fresh, empty store (it may be a coroutine function). ``dimensions``
    must match the vector width the store was configured with.

    Requirements checked:

    * an upserted memory is findable by a matching vector
    * an upsert with an existing id replaces rather than duplicates
    * search never returns another agent's memories
    * search honours ``threshold`` and ``limit``
    * passing ``query_text`` is accepted (hybrid ranking is optional)
    * ``content_types`` filters when supplied
    * ``delete`` removes one memory; deleting an unknown id must not corrupt the store
    * ``delete_agent_memories`` removes one agent's memories and only that agent's
    """
    _require(
        dimensions >= 2,
        "check_l2_store arguments",
        f"dimensions must be at least 2 to build distinguishable vectors, got {dimensions}",
    )
    store = await _make(factory)
    try:
        near = _unit_vector(dimensions, 0)
        far = _unit_vector(dimensions, 1)

        entry = _entry("agent-a", "the user prefers dark mode", near)
        await store.upsert(entry)

        found = await store.search(
            agent_id="agent-a", query_embedding=near, threshold=0.5, limit=10
        )
        _require(
            any(m.id == entry.id for m in found),
            "upsert/search round-trip",
            f"a memory matching the query vector was not returned, got {found!r}",
        )

        # Re-upserting the same id must replace, not duplicate.
        updated = _entry("agent-a", "the user prefers light mode", near, entry_id=entry.id)
        await store.upsert(updated)
        found = await store.search(
            agent_id="agent-a", query_embedding=near, threshold=0.5, limit=10
        )
        same_id = [m for m in found if m.id == entry.id]
        _require(
            len(same_id) == 1,
            "upsert idempotency",
            f"re-upserting an id must replace it, found {len(same_id)} copies",
        )
        _require(
            same_id[0].content == "the user prefers light mode",
            "upsert replaces content",
            f"expected the updated content, got {same_id[0].content!r}",
        )

        # Agent isolation: the single most important property of this tier.
        other = _entry("agent-b", "another agent's secret", near)
        await store.upsert(other)
        found = await store.search(
            agent_id="agent-a", query_embedding=near, threshold=0.5, limit=10
        )
        _require(
            all(m.id != other.id for m in found),
            "search agent isolation",
            "another agent's memory leaked into this agent's results",
        )
        _require(
            all(m.agent_id == "agent-a" for m in found),
            "search agent_id fidelity",
            f"every hit must belong to the queried agent, got "
            f"{sorted({m.agent_id for m in found})!r}",
        )

        # Threshold: an orthogonal vector scores 0.0 and must be excluded.
        far_hits = await store.search(
            agent_id="agent-a", query_embedding=far, threshold=0.5, limit=10
        )
        _require(
            all(m.id != entry.id for m in far_hits),
            "search honours threshold",
            "a memory below the similarity threshold was returned",
        )

        # Limit.
        for i in range(4):
            await store.upsert(_entry("agent-a", f"memory {i}", near))
        limited = await store.search(
            agent_id="agent-a", query_embedding=near, threshold=0.0, limit=2
        )
        _require(
            len(limited) <= 2,
            "search honours limit",
            f"limit=2 returned {len(limited)} results",
        )

        # query_text must be accepted. Fusing it into the ranking is optional.
        hybrid = await store.search(
            agent_id="agent-a",
            query_embedding=near,
            threshold=0.0,
            limit=10,
            query_text="dark mode",
        )
        _require(
            isinstance(hybrid, list),
            "search accepts query_text",
            f"expected a list when query_text is supplied, got {type(hybrid).__name__}",
        )

        # content_types filter, when supplied, must exclude other types.
        typed = await store.search(
            agent_id="agent-a",
            query_embedding=near,
            threshold=0.0,
            limit=10,
            content_types=["summary"],
        )
        _require(
            all(m.content_type == "summary" for m in typed),
            "search honours content_types",
            f"expected only 'summary' entries, got {sorted({m.content_type for m in typed})!r}",
        )

        # Delete one.
        await store.delete(entry.id)
        after = await store.search(
            agent_id="agent-a", query_embedding=near, threshold=0.0, limit=10
        )
        _require(
            all(m.id != entry.id for m in after),
            "delete removes the memory",
            "the deleted memory was still returned by search",
        )

        # Deleting an unknown id may raise MemoryNotFoundError or be a no-op, but must
        # leave the store usable either way. The built-in stores differ here: Qdrant
        # treats it as success, InMemoryStore raises.
        with suppress(MemoryNotFoundError):
            await store.delete("00000000-0000-0000-0000-000000000000")
        still_there = await store.search(
            agent_id="agent-a", query_embedding=near, threshold=0.0, limit=10
        )
        _require(
            len(still_there) >= 1,
            "delete of an unknown id is harmless",
            "deleting a missing id must not remove other memories",
        )

        # Erase one agent, leave the other intact.
        await store.delete_agent_memories("agent-a")
        erased = await store.search(
            agent_id="agent-a", query_embedding=near, threshold=0.0, limit=10
        )
        kept = await store.search(agent_id="agent-b", query_embedding=near, threshold=0.0, limit=10)
        _require(
            erased == [],
            "delete_agent_memories erases the agent",
            f"expected no memories for the erased agent, got {erased!r}",
        )
        _require(
            len(kept) == 1,
            "delete_agent_memories scope",
            "erasing one agent must not touch another agent's memories",
        )
    finally:
        await _close(store)


async def _make(factory: Callable[[], Any]) -> Any:  # noqa: ANN401 - generic store factory
    store = factory()
    if hasattr(store, "__await__"):
        store = await store
    return store


async def _close(store: Any) -> None:  # noqa: ANN401 - generic store
    closer = getattr(store, "close", None)
    if callable(closer):
        result = closer()
        if hasattr(result, "__await__"):
            await result
