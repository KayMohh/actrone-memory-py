"""Tests for the published conformance suite.

Two jobs here. First, prove the built-in ``InMemoryStore`` satisfies the contract we ask
third-party adapters to satisfy, so the suite is not holding others to a standard we fail.
Second, prove the suite actually *detects* violations: a conformance suite that passes
everything is worse than none, because it manufactures false confidence.
"""

from __future__ import annotations

import pytest

from actrone_memory.in_memory import InMemoryStore
from actrone_memory.models import MemoryEntry, Turn
from actrone_memory.testing import ConformanceError, check_l1_store, check_l2_store

_DIMENSIONS = 8


@pytest.mark.asyncio
async def test_in_memory_store_satisfies_the_l1_contract() -> None:
    await check_l1_store(InMemoryStore)


@pytest.mark.asyncio
async def test_in_memory_store_satisfies_the_l2_contract() -> None:
    await check_l2_store(InMemoryStore, dimensions=_DIMENSIONS)


@pytest.mark.asyncio
async def test_check_l2_store_rejects_too_few_dimensions() -> None:
    with pytest.raises(ConformanceError, match="at least 2"):
        await check_l2_store(InMemoryStore, dimensions=1)


@pytest.mark.asyncio
async def test_check_l1_store_accepts_an_async_factory() -> None:
    async def build() -> InMemoryStore:
        return InMemoryStore()

    await check_l1_store(build)


# ── The suite must catch real violations ─────────────────────────────────────


class _ReversedOrderStore(InMemoryStore):
    """Returns turns newest-first, the most plausible way to get ordering wrong."""

    async def get_recent_turns(
        self, agent_id: str, session_id: str, n: int | None = None
    ) -> list[Turn]:
        turns = await super().get_recent_turns(agent_id, session_id, n)
        return list(reversed(turns))


class _LeakyL1Store(InMemoryStore):
    """Ignores agent_id, so one agent can read another's session."""

    async def get_recent_turns(
        self, agent_id: str, session_id: str, n: int | None = None
    ) -> list[Turn]:
        merged: list[Turn] = []
        for key, bucket in self._turns.items():
            if key.endswith(f"::{session_id}"):
                merged.extend(bucket)
        return merged if n is None else merged[-n:]


class _LeakyL2Store(InMemoryStore):
    """Ignores agent_id on search, the worst failure this tier can have."""

    async def search(  # type: ignore[override]
        self,
        agent_id: str,
        query_embedding: list[float],
        threshold: float,
        limit: int = 20,
        content_types: list[str] | None = None,
        query_text: str | None = None,
    ) -> list[MemoryEntry]:
        everything: list[MemoryEntry] = []
        for bucket in self._memories.values():
            everything.extend(bucket)
        return everything[:limit]


class _NoLockStore(InMemoryStore):
    """Always grants the summary lock, which would fan out duplicate summaries."""

    async def try_acquire_summary_lock(
        self, agent_id: str, session_id: str, ttl_seconds: int
    ) -> bool:
        return True


class _UnscopedClearStore(InMemoryStore):
    """clear_session wipes every session for the agent."""

    async def clear_session(self, agent_id: str, session_id: str) -> None:
        for key in [k for k in self._turns if k.startswith(f"{agent_id}::")]:
            self._turns.pop(key, None)


@pytest.mark.parametrize(
    ("store_cls", "requirement"),
    [
        (_ReversedOrderStore, "ordering"),
        (_LeakyL1Store, "agent isolation"),
        (_NoLockStore, "try_acquire_summary_lock excludes"),
        (_UnscopedClearStore, "clear_session scope"),
    ],
)
@pytest.mark.asyncio
async def test_l1_suite_detects_violations(
    store_cls: type[InMemoryStore], requirement: str
) -> None:
    with pytest.raises(ConformanceError, match=requirement):
        await check_l1_store(store_cls)


@pytest.mark.asyncio
async def test_l2_suite_detects_cross_agent_leakage() -> None:
    with pytest.raises(ConformanceError, match="agent isolation"):
        await check_l2_store(_LeakyL2Store, dimensions=_DIMENSIONS)


@pytest.mark.asyncio
async def test_l2_suite_detects_a_threshold_that_is_ignored() -> None:
    class _IgnoresThreshold(InMemoryStore):
        async def search(  # type: ignore[override]
            self,
            agent_id: str,
            query_embedding: list[float],
            threshold: float,
            limit: int = 20,
            content_types: list[str] | None = None,
            query_text: str | None = None,
        ) -> list[MemoryEntry]:
            return list(self._memories.get(agent_id, []))[:limit]

    with pytest.raises(ConformanceError, match="threshold"):
        await check_l2_store(_IgnoresThreshold, dimensions=_DIMENSIONS)


@pytest.mark.asyncio
async def test_l2_suite_detects_duplicate_upserts() -> None:
    class _AppendsInsteadOfReplacing(InMemoryStore):
        async def upsert(self, entry: MemoryEntry) -> None:
            self._memories.setdefault(entry.agent_id, []).append(entry)

    with pytest.raises(ConformanceError, match="idempotency"):
        await check_l2_store(_AppendsInsteadOfReplacing, dimensions=_DIMENSIONS)
