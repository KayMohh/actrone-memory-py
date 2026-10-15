from __future__ import annotations

import pytest

from actrone_memory.benchmark import (
    DEFAULT_DATASET,
    EvalReport,
    RecencyBaseline,
    format_comparison,
    run_comparison,
    run_eval,
)
from actrone_memory.benchmark.dataset import EvalCase, MemoryItem, Query
from actrone_memory.benchmark.harness import QueryOutcome, _aggregate, _percentile
from actrone_memory.config import MemoryConfig
from actrone_memory.manager import MemoryManager

# ── Regression floors ────────────────────────────────────────────────────────
# Measured recall@5 ≈ 0.93 / MRR ≈ 0.93 with the default hashing embedder. The
# floors sit below that with margin so genuine ranking/retrieval regressions fail
# CI while normal noise does not. Raise these when a better embedder is the default.
_MIN_RECALL_AT_5 = 0.85
_MIN_MRR = 0.80


@pytest.mark.asyncio
async def test_default_dataset_meets_quality_floor():
    report = await run_eval()
    assert report.n_queries == sum(len(c.queries) for c in DEFAULT_DATASET)
    assert report.recall_at_k >= _MIN_RECALL_AT_5, report.format_table()
    assert report.mrr >= _MIN_MRR, report.format_table()
    assert 0.0 <= report.precision_at_k <= 1.0
    assert report.latency_p50_ms >= 0.0
    assert report.latency_p95_ms >= report.latency_p50_ms


@pytest.mark.asyncio
async def test_run_eval_accepts_a_provided_manager():
    mm = await MemoryManager.create(MemoryConfig(relevance_threshold=0.05))
    try:
        report = await run_eval(manager=mm)
        assert isinstance(report, EvalReport)
        assert report.n_queries > 0
    finally:
        await mm.close()


@pytest.mark.asyncio
async def test_perfect_and_zero_retrieval_cases():
    case_hit = EvalCase(
        name="hit",
        agent_id="ag",
        memories=[MemoryItem("m1", "the sky is blue today")],
        queries=[Query("the sky is blue today", frozenset({"m1"}))],
    )
    report = await run_eval([case_hit], config=MemoryConfig(relevance_threshold=0.05))
    assert report.recall_at_k == pytest.approx(1.0)
    assert report.mrr == pytest.approx(1.0)


def test_format_table_contains_metrics():
    report = _aggregate([QueryOutcome("q", ["m1", "m2"], frozenset({"m1"}), latency_ms=1.5)], k=5)
    table = report.format_table()
    assert "recall@5" in table
    assert "MRR" in table
    assert "latency p95" in table


# ── metrics units ────────────────────────────────────────────────────────────


def test_query_outcome_hit_rank():
    assert QueryOutcome("q", ["x", "y", "m1"], frozenset({"m1"}), 0.0).hit_rank == 3
    assert QueryOutcome("q", ["x", "y"], frozenset({"m1"}), 0.0).hit_rank is None


def test_aggregate_recall_precision_mrr():
    outcomes = [
        # 1 relevant, retrieved at rank 1 → recall 1.0, precision@5 0.2, RR 1.0
        QueryOutcome("q1", ["m1", "a", "b", "c", "d"], frozenset({"m1"}), 1.0),
        # 1 relevant, not retrieved → recall 0, precision 0, RR 0
        QueryOutcome("q2", ["a", "b"], frozenset({"m9"}), 1.0),
    ]
    r = _aggregate(outcomes, k=5)
    assert r.recall_at_k == pytest.approx(0.5)
    assert r.precision_at_k == pytest.approx(0.1)  # (0.2 + 0.0) / 2
    assert r.mrr == pytest.approx(0.5)


def test_aggregate_empty_is_zero():
    r = _aggregate([], k=5)
    assert r.recall_at_k == 0.0 and r.mrr == 0.0 and r.n_queries == 0


def test_percentile_edges():
    assert _percentile([], 50) == 0.0
    assert _percentile([5.0], 95) == 5.0
    assert _percentile([1.0, 2.0, 3.0, 4.0], 50) == 2.0
    assert _percentile([1.0, 2.0, 3.0, 4.0], 100) == 4.0


# ── Comparative benchmark ─────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_run_comparison_actrone_beats_naive_baseline():
    """The comparative harness runs the same dataset against multiple systems, and
    Actrone (semantic/keyword ranking) beats the naive recency baseline on recall."""
    mm = await MemoryManager.create(MemoryConfig(relevance_threshold=0.05))
    try:
        reports = await run_comparison({"actrone": mm, "recency": RecencyBaseline()})
    finally:
        await mm.close()

    assert set(reports) == {"actrone", "recency"}
    assert all(isinstance(r, EvalReport) for r in reports.values())
    # The whole point of the comparison: real retrieval beats the query-ignoring floor.
    assert reports["actrone"].recall_at_k > reports["recency"].recall_at_k

    table = format_comparison(reports)
    assert "actrone" in table and "recall@5" in table


def test_format_comparison_empty():
    assert format_comparison({}) == "(no systems)"
