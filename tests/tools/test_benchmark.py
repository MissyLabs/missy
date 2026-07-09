"""Tests for missy.tools.benchmark (scoring, store, runner)."""

from __future__ import annotations

from pathlib import Path

import pytest

from missy.tools.base import BaseTool, ToolPermissions, ToolResult
from missy.tools.benchmark.benchmark_store import BenchmarkStore
from missy.tools.benchmark.runner import BenchmarkRunner, BenchmarkSuite, BenchmarkTask
from missy.tools.benchmark.scoring import (
    BenchmarkResult,
    BenchmarkScorer,
    ScoredResult,
    ScoreWeights,
)

# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


def _make_result(
    tool_name: str = "calc",
    provider: str = "direct",
    success: bool = True,
    latency_ms: float = 100.0,
    cost_usd: float = 0.0,
    actual: str = "4",
    expected: str = "4",
    tool_call_made: bool = True,
    tool_call_args: dict | None = None,
    required_params: list[str] | None = None,
    safety_violation: bool = False,
    error: str = "",
) -> BenchmarkResult:
    return BenchmarkResult(
        task_id="task1",
        tool_name=tool_name,
        provider=provider,
        success=success,
        latency_ms=latency_ms,
        cost_usd=cost_usd,
        actual_output=actual,
        expected_output=expected,
        tool_call_made=tool_call_made,
        tool_call_args=dict(tool_call_args or {"expression": "2+2"}),
        schema_required_params=list(required_params or ["expression"]),
        safety_violation=safety_violation,
        error=error,
    )


# ---------------------------------------------------------------------------
# BenchmarkScorer
# ---------------------------------------------------------------------------


class TestBenchmarkScorer:
    def test_perfect_result_scores_high(self):
        scorer = BenchmarkScorer()
        r = _make_result(success=True, actual="4", expected="4", latency_ms=10, cost_usd=0)
        sr = scorer.score(r)
        assert sr.composite > 0.8

    def test_failure_has_zero_reliability(self):
        scorer = BenchmarkScorer()
        r = _make_result(success=False, error="exploded")
        sr = scorer.score(r)
        assert sr.reliability == 0.0

    def test_safety_violation_has_zero_safety(self):
        scorer = BenchmarkScorer()
        r = _make_result(safety_violation=True)
        sr = scorer.score(r)
        assert sr.safety == 0.0

    def test_no_tool_call_has_zero_quality(self):
        scorer = BenchmarkScorer()
        r = _make_result(tool_call_made=False)
        sr = scorer.score(r)
        assert sr.tool_call_quality == 0.0

    def test_missing_required_param_reduces_schema_score(self):
        scorer = BenchmarkScorer()
        r = _make_result(
            tool_call_args={"expression": "2+2"},
            required_params=["expression", "precision"],
        )
        sr = scorer.score(r)
        assert sr.schema_score < 1.0

    def test_all_required_params_present_full_schema_score(self):
        scorer = BenchmarkScorer()
        r = _make_result(
            tool_call_args={"expression": "2+2", "precision": 2},
            required_params=["expression", "precision"],
        )
        sr = scorer.score(r)
        assert sr.schema_score == 1.0

    def test_high_latency_reduces_score(self):
        scorer = BenchmarkScorer(max_latency_ms=1000)
        fast = scorer.score(_make_result(latency_ms=10))
        slow = scorer.score(_make_result(latency_ms=990))
        assert fast.latency_score > slow.latency_score

    def test_composite_in_unit_range(self):
        scorer = BenchmarkScorer()
        r = _make_result()
        sr = scorer.score(r)
        assert 0.0 <= sr.composite <= 1.0

    def test_correctness_exact_match(self):
        scorer = BenchmarkScorer()
        r = _make_result(actual="hello", expected="hello")
        sr = scorer.score(r)
        assert sr.correctness == 1.0

    def test_correctness_partial_string_match(self):
        scorer = BenchmarkScorer()
        r = _make_result(actual="The answer is 4 items", expected="4")
        sr = scorer.score(r)
        assert sr.correctness > 0.0

    def test_correctness_numeric_tolerance(self):
        scorer = BenchmarkScorer()
        r = _make_result(actual="3.999", expected="4")
        sr = scorer.score(r)
        assert sr.correctness > 0.9

    def test_no_expected_output_gives_full_correctness_on_success(self):
        scorer = BenchmarkScorer()
        r = _make_result(expected=None)
        sr = scorer.score(r)
        assert sr.correctness == 1.0

    def test_aggregate_empty_returns_empty(self):
        scorer = BenchmarkScorer()
        assert scorer.aggregate([]) == {}

    def test_aggregate_returns_means(self):
        scorer = BenchmarkScorer()
        results = [_make_result() for _ in range(3)]
        scored = scorer.score_batch(results)
        agg = scorer.aggregate(scored)
        assert "composite" in agg
        assert 0.0 <= agg["composite"] <= 1.0


class TestScoreWeights:
    def test_normalised_sums_to_one(self):
        w = ScoreWeights()
        n = w.normalised()
        total = sum(n.values())
        assert abs(total - 1.0) < 1e-9

    def test_custom_weights_normalised(self):
        w = ScoreWeights(
            correctness=10.0,
            latency=0.0,
            cost=0.0,
            reliability=0.0,
            safety=0.0,
            schema_score=0.0,
            tool_call_quality=0.0,
        )
        n = w.normalised()
        assert n["correctness"] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# BenchmarkStore
# ---------------------------------------------------------------------------


@pytest.fixture
def bench_store(tmp_path: Path) -> BenchmarkStore:
    return BenchmarkStore(db_path=tmp_path / "bench.db")


def _make_scored(tool: str = "calc", provider: str = "direct") -> ScoredResult:
    r = _make_result(tool_name=tool, provider=provider)
    scorer = BenchmarkScorer()
    return scorer.score(r)


class TestBenchmarkStore:
    def test_save_and_count(self, bench_store):
        sr = _make_scored()
        bench_store.save(sr)
        assert bench_store.count() == 1

    def test_save_returns_id(self, bench_store):
        sr = _make_scored()
        row_id = bench_store.save(sr)
        assert row_id  # non-empty UUID

    def test_query_by_tool(self, bench_store):
        bench_store.save(_make_scored("tool_a"))
        bench_store.save(_make_scored("tool_b"))
        rows = bench_store.query(tool_name="tool_a")
        assert len(rows) == 1
        assert rows[0]["tool_name"] == "tool_a"

    def test_query_by_provider(self, bench_store):
        bench_store.save(_make_scored(provider="anthropic"))
        bench_store.save(_make_scored(provider="openai"))
        rows = bench_store.query(provider="anthropic")
        assert len(rows) == 1

    def test_query_min_composite(self, bench_store):
        bench_store.save(_make_scored())
        # composite of a perfect run is ~0.999; use 1.1 so rows_high is empty
        rows_high = bench_store.query(min_composite=1.1)
        rows_low = bench_store.query(min_composite=0.0)
        assert len(rows_low) > len(rows_high)

    def test_provider_summary(self, bench_store):
        for _ in range(3):
            bench_store.save(_make_scored("calc", "anthropic"))
        bench_store.save(_make_scored("calc", "openai"))
        summaries = bench_store.provider_summary("calc")
        assert len(summaries) == 2
        providers = {s.provider for s in summaries}
        assert "anthropic" in providers
        assert "openai" in providers

    def test_delete_before(self, bench_store):
        bench_store.save(_make_scored())
        deleted = bench_store.delete_before("2099-01-01T00:00:00")
        assert deleted == 1
        assert bench_store.count() == 0

    def test_save_batch(self, bench_store):
        scored_list = [_make_scored() for _ in range(5)]
        ids = bench_store.save_batch(scored_list)
        assert len(ids) == 5
        assert bench_store.count() == 5

    def test_all_provider_summaries_empty_store(self, bench_store):
        assert bench_store.all_provider_summaries() == []

    def test_all_provider_summaries_covers_every_tool_and_provider(self, bench_store):
        for _ in range(3):
            bench_store.save(_make_scored("calc", "anthropic"))
        bench_store.save(_make_scored("calc", "openai"))
        bench_store.save(_make_scored("search", "anthropic"))

        summaries = bench_store.all_provider_summaries()
        pairs = {(s.tool_name, s.provider) for s in summaries}
        assert pairs == {
            ("calc", "anthropic"),
            ("calc", "openai"),
            ("search", "anthropic"),
        }
        calc_anthropic = next(
            s for s in summaries if s.tool_name == "calc" and s.provider == "anthropic"
        )
        assert calc_anthropic.run_count == 3

    def test_all_provider_summaries_matches_provider_summary(self, bench_store):
        bench_store.save(_make_scored("calc", "anthropic"))
        bench_store.save(_make_scored("calc", "openai"))
        via_all = {s.provider: s.mean_composite for s in bench_store.all_provider_summaries()}
        via_single = {s.provider: s.mean_composite for s in bench_store.provider_summary("calc")}
        assert via_all == via_single


# ---------------------------------------------------------------------------
# BenchmarkTask / BenchmarkSuite
# ---------------------------------------------------------------------------


class TestBenchmarkTask:
    def test_create_assigns_id(self):
        t = BenchmarkTask.create("calc", {"expression": "2+2"})
        assert t.id

    def test_required_params_default_to_input_keys(self):
        t = BenchmarkTask.create("calc", {"expression": "2+2", "precision": 2})
        assert set(t.schema_required_params) == {"expression", "precision"}

    def test_explicit_required_params(self):
        t = BenchmarkTask.create("calc", {"x": 1, "y": 2}, schema_required_params=["x"])
        assert t.schema_required_params == ["x"]


class TestBenchmarkSuite:
    def test_add_task(self):
        suite = BenchmarkSuite(name="test", tool_name="calc")
        t = BenchmarkTask.create("calc", {"expression": "1+1"})
        suite.add_task(t)
        assert suite.task_count() == 1


# ---------------------------------------------------------------------------
# BenchmarkRunner (direct execution against a real tool)
# ---------------------------------------------------------------------------


class EchoTool(BaseTool):
    name = "echo_tool"
    description = "Echo input"
    permissions = ToolPermissions()
    parameters = {"text": {"type": "string", "description": "text", "required": True}}

    def execute(self, *, text: str, **_kw) -> ToolResult:
        return ToolResult(success=True, output=text)


class FailTool(BaseTool):
    name = "fail_tool"
    description = "Always fails"
    permissions = ToolPermissions()
    parameters = {}

    def execute(self, **_kw) -> ToolResult:
        return ToolResult(success=False, output="", error="always fails")


@pytest.fixture
def mini_registry():
    from missy.tools.registry import ToolRegistry

    r = ToolRegistry()
    r.register(EchoTool())
    r.register(FailTool())
    return r


@pytest.fixture
def bench_runner(tmp_path: Path) -> BenchmarkRunner:
    store = BenchmarkStore(db_path=tmp_path / "bench_runner.db")
    return BenchmarkRunner(store=store, provider="direct")


class TestBenchmarkRunner:
    def test_run_task_success(self, bench_runner, mini_registry):
        task = BenchmarkTask.create("echo_tool", {"text": "hello"}, expected_output="hello")
        sr = bench_runner.run_task(task, registry=mini_registry, persist=False)
        assert sr.result.success
        assert sr.correctness == 1.0

    def test_run_task_failure(self, bench_runner, mini_registry):
        task = BenchmarkTask.create("fail_tool", {})
        sr = bench_runner.run_task(task, registry=mini_registry, persist=False)
        assert not sr.result.success
        assert sr.reliability == 0.0

    def test_run_task_missing_tool(self, bench_runner, mini_registry):
        task = BenchmarkTask.create("nonexistent_tool", {})
        sr = bench_runner.run_task(task, registry=mini_registry, persist=False)
        assert not sr.result.success

    def test_run_suite(self, bench_runner, mini_registry):
        suite = BenchmarkSuite(name="echo_suite", tool_name="echo_tool")
        for i in range(3):
            suite.add_task(
                BenchmarkTask.create("echo_tool", {"text": f"msg{i}"}, expected_output=f"msg{i}")
            )
        report = bench_runner.run_suite(suite, registry=mini_registry, persist=False)
        assert report.tool_name == "echo_tool"
        assert len(report.scored_results) == 3
        assert report.error_count == 0
        assert report.aggregate["composite"] > 0.0

    def test_run_suite_persist_saves_results(self, bench_runner, mini_registry):
        suite = BenchmarkSuite(name="persist_test", tool_name="echo_tool")
        suite.add_task(BenchmarkTask.create("echo_tool", {"text": "x"}, expected_output="x"))
        bench_runner.run_suite(suite, registry=mini_registry, persist=True)
        assert bench_runner._store.count() == 1
