"""Unit tests for the opt-in cross-encoder reranker .

``fastembed`` is stubbed so these run in the base venv. They cover the reorder-by-score contract,
the top-K window (tail preserved), the graceful ``build_reranker`` degradation, and that the manager
actually applies the reranker in ``search_memories``.
"""

from __future__ import annotations

import sys
import types
from typing import Any

import pytest

from actrone_memory.config import MemoryConfig
from actrone_memory.manager import MemoryManager
from actrone_memory.models import MemoryEntry
from actrone_memory.rerank import CrossEncoderReranker, build_reranker


def _install_fake_fastembed_reranker(monkeypatch: pytest.MonkeyPatch) -> None:
    """Inject a fake ``TextCrossEncoder`` (in ``fastembed.rerank.cross_encoder``) scoring by
    keyword overlap."""

    class _FakeCrossEncoder:
        def __init__(self, model_name: str = "stub", cache_dir: Any = None) -> None:
            self.model_name = model_name

        def rerank(self, query: str, documents: list[str]) -> Any:
            q = set(query.lower().split())
            for doc in documents:
                yield float(sum(1 for w in doc.lower().split() if w in q))

    fastembed = types.ModuleType("fastembed")
    rerank_pkg = types.ModuleType("fastembed.rerank")
    ce_mod = types.ModuleType("fastembed.rerank.cross_encoder")
    ce_mod.TextCrossEncoder = _FakeCrossEncoder  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "fastembed", fastembed)
    monkeypatch.setitem(sys.modules, "fastembed.rerank", rerank_pkg)
    monkeypatch.setitem(sys.modules, "fastembed.rerank.cross_encoder", ce_mod)


def _entry(mid: str, content: str) -> MemoryEntry:
    return MemoryEntry(
        id=mid,
        agent_id="agent-1",
        session_id="s1",
        content=content,
        content_type="summary",
        importance_score=0.5,
        token_count=len(content.split()),
    )


@pytest.mark.asyncio
async def test_reranker_reorders_by_cross_encoder_score(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_fastembed_reranker(monkeypatch)
    reranker = CrossEncoderReranker()
    entries = [_entry("A", "yellow banana fruit"), _entry("B", "the France Paris capital")]
    ranked = await reranker.rerank("France", entries)
    assert [e.id for e in ranked] == ["B", "A"]  # B wins the keyword overlap


@pytest.mark.asyncio
async def test_reranker_top_k_preserves_tail(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_fastembed_reranker(monkeypatch)
    reranker = CrossEncoderReranker()
    entries = [_entry("A", "no match"), _entry("B", "France"), _entry("C", "tail untouched")]
    ranked = await reranker.rerank("France", entries, top_k=2)
    # Only A,B are rescored (B rises); C stays appended in place.
    assert [e.id for e in ranked] == ["B", "A", "C"]


@pytest.mark.asyncio
async def test_reranker_empty_and_blank_query(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_fastembed_reranker(monkeypatch)
    reranker = CrossEncoderReranker()
    assert await reranker.rerank("France", []) == []
    entries = [_entry("A", "x")]
    assert await reranker.rerank("   ", entries) == entries  # blank query → unchanged


def test_build_reranker_disabled_returns_none() -> None:
    assert build_reranker(enabled=False) is None


def test_build_reranker_degrades_when_dep_absent(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "fastembed.rerank.cross_encoder", None)
    # Enabled but the extra can't import → None (graceful), not a raise.
    assert build_reranker(enabled=True) is None


def test_build_reranker_constructs_when_available(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_fastembed_reranker(monkeypatch)
    reranker = build_reranker(enabled=True, model_name="my-model")
    assert isinstance(reranker, CrossEncoderReranker)
    assert reranker.model_name == "my-model"


@pytest.mark.asyncio
async def test_manager_builds_reranker_when_enabled(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_fastembed_reranker(monkeypatch)
    cfg = MemoryConfig(rerank_enabled=True)  # type: ignore[call-arg]
    async with await MemoryManager.create(cfg) as mm:
        assert isinstance(mm._reranker, CrossEncoderReranker)  # built via build_reranker + the stub


@pytest.mark.asyncio
async def test_manager_applies_reranker_in_search() -> None:
    # A reverse reranker proves the manager routes retrieval through _maybe_rerank: the returned
    # order is the reranker's, not the raw retrieval order.
    cfg = MemoryConfig(relevance_threshold=0.01)  # type: ignore[call-arg]
    async with await MemoryManager.create(cfg) as mm:
        await mm.inject_memory("agent-1", "France Paris capital city", importance=0.9)
        await mm.inject_memory("agent-1", "France Lyon riverside town", importance=0.9)
        baseline = await mm.search_memories("agent-1", "France", limit=2)
        assert len(baseline) == 2

        class _ReverseReranker:
            async def rerank(self, query: str, entries: Any, *, top_k: int | None = None) -> Any:
                return list(reversed(entries))

        mm._reranker = _ReverseReranker()  # type: ignore[assignment]
        reranked = await mm.search_memories("agent-1", "France", limit=2)
        assert [e.id for e in reranked] == [e.id for e in reversed(baseline)]
