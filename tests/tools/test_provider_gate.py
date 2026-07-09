"""Tests for missy.tools.intelligence.provider_gate."""

from __future__ import annotations

from pathlib import Path

import pytest

from missy.tools.benchmark.benchmark_store import BenchmarkStore
from missy.tools.benchmark.scoring import BenchmarkResult, BenchmarkScorer
from missy.tools.intelligence.provider_gate import (
    ProviderGateStore,
    ToolProviderGate,
)


@pytest.fixture
def overrides(tmp_path: Path) -> ProviderGateStore:
    return ProviderGateStore(path=tmp_path / "overrides.json")


@pytest.fixture
def bench_store(tmp_path: Path) -> BenchmarkStore:
    return BenchmarkStore(db_path=tmp_path / "benchmarks.db")


def _save_runs(
    store: BenchmarkStore,
    tool_name: str,
    provider: str,
    good_runs: list[float],
) -> None:
    """Persist synthetic runs that are unambiguously strong (1.0) or weak (0.0).

    Every scoring dimension is pushed the same direction so the resulting
    composite lands clearly above 0.9 (good) or below 0.1 (weak), regardless
    of how the scorer's dimension weights are tuned — the gating tests only
    need "clearly enabled" vs. "clearly disabled" data, not exact composites.
    """
    scorer = BenchmarkScorer()
    for score in good_runs:
        good = score >= 0.5
        result = BenchmarkResult(
            task_id="t",
            tool_name=tool_name,
            provider=provider,
            success=good,
            latency_ms=10.0 if good else 20_000.0,
            cost_usd=0.0 if good else 1.0,
            actual_output="4" if good else "wrong",
            expected_output="4",
            tool_call_made=good,
            tool_call_args={"expression": "2+2"} if good else {},
            schema_required_params=["expression"],
            safety_violation=not good,
            error="" if good else "boom",
        )
        store.save(scorer.score(result))


class TestProviderGateStore:
    def test_get_returns_none_when_unset(self, overrides: ProviderGateStore) -> None:
        assert overrides.get("tool_a", "anthropic") is None

    def test_set_then_get(self, overrides: ProviderGateStore) -> None:
        overrides.set("tool_a", "anthropic", False)
        assert overrides.get("tool_a", "anthropic") is False

    def test_set_true_then_get(self, overrides: ProviderGateStore) -> None:
        overrides.set("tool_a", "ollama", True)
        assert overrides.get("tool_a", "ollama") is True

    def test_clear_removes_override(self, overrides: ProviderGateStore) -> None:
        overrides.set("tool_a", "anthropic", False)
        assert overrides.clear("tool_a", "anthropic") is True
        assert overrides.get("tool_a", "anthropic") is None

    def test_clear_missing_returns_false(self, overrides: ProviderGateStore) -> None:
        assert overrides.clear("tool_a", "anthropic") is False

    def test_list_overrides(self, overrides: ProviderGateStore) -> None:
        overrides.set("tool_a", "anthropic", False)
        overrides.set("tool_a", "ollama", True)
        overrides.set("tool_b", "openai", False)
        listed = overrides.list_overrides()
        assert listed == {
            "tool_a": {"anthropic": False, "ollama": True},
            "tool_b": {"openai": False},
        }

    def test_persists_across_instances(self, tmp_path: Path) -> None:
        path = tmp_path / "overrides.json"
        ProviderGateStore(path=path).set("tool_a", "anthropic", False)
        reloaded = ProviderGateStore(path=path)
        assert reloaded.get("tool_a", "anthropic") is False

    def test_corrupt_file_treated_as_empty(self, tmp_path: Path) -> None:
        path = tmp_path / "overrides.json"
        path.write_text("not json{{{", encoding="utf-8")
        store = ProviderGateStore(path=path)
        assert store.get("tool_a", "anthropic") is None

    def test_file_permissions_restricted(self, overrides: ProviderGateStore) -> None:
        overrides.set("tool_a", "anthropic", False)
        mode = overrides._path.stat().st_mode & 0o777
        assert mode == 0o600


class TestToolProviderGateOverridePrecedence:
    def test_override_disables_regardless_of_benchmark(
        self, overrides: ProviderGateStore, bench_store: BenchmarkStore
    ) -> None:
        _save_runs(bench_store, "tool_a", "anthropic", [1.0, 1.0, 1.0, 1.0])
        overrides.set("tool_a", "anthropic", False)
        gate = ToolProviderGate(overrides=overrides, benchmark_store=bench_store)
        decision = gate.decide("tool_a", "anthropic")
        assert decision.enabled is False
        assert decision.source == "override"

    def test_override_enables_despite_weak_benchmark(
        self, overrides: ProviderGateStore, bench_store: BenchmarkStore
    ) -> None:
        _save_runs(bench_store, "tool_a", "anthropic", [0.0, 0.0, 0.0, 0.0])
        overrides.set("tool_a", "anthropic", True)
        gate = ToolProviderGate(
            overrides=overrides, benchmark_store=bench_store, min_composite=0.9
        )
        decision = gate.decide("tool_a", "anthropic")
        assert decision.enabled is True
        assert decision.source == "override"


class TestToolProviderGateBenchmarkDriven:
    def test_no_data_defaults_to_enabled(
        self, overrides: ProviderGateStore, bench_store: BenchmarkStore
    ) -> None:
        gate = ToolProviderGate(overrides=overrides, benchmark_store=bench_store)
        decision = gate.decide("unknown_tool", "anthropic")
        assert decision.enabled is True
        assert decision.source == "default"
        assert "no benchmark data" in decision.reason

    def test_insufficient_samples_defaults_to_enabled(
        self, overrides: ProviderGateStore, bench_store: BenchmarkStore
    ) -> None:
        _save_runs(bench_store, "tool_a", "anthropic", [0.0])
        gate = ToolProviderGate(
            overrides=overrides, benchmark_store=bench_store, min_samples=3
        )
        decision = gate.decide("tool_a", "anthropic")
        assert decision.enabled is True
        assert decision.source == "default"
        assert "below min_samples" in decision.reason

    def test_weak_provider_is_disabled(
        self, overrides: ProviderGateStore, bench_store: BenchmarkStore
    ) -> None:
        _save_runs(bench_store, "tool_a", "ollama", [0.0, 0.0, 0.0, 0.0])
        gate = ToolProviderGate(
            overrides=overrides, benchmark_store=bench_store, min_samples=3, min_composite=0.5
        )
        decision = gate.decide("tool_a", "ollama")
        assert decision.enabled is False
        assert decision.source == "benchmark"
        assert "below threshold" not in decision.reason or "<" in decision.reason

    def test_strong_provider_is_enabled(
        self, overrides: ProviderGateStore, bench_store: BenchmarkStore
    ) -> None:
        _save_runs(bench_store, "tool_a", "anthropic", [1.0, 1.0, 1.0, 1.0])
        gate = ToolProviderGate(
            overrides=overrides, benchmark_store=bench_store, min_samples=3, min_composite=0.5
        )
        decision = gate.decide("tool_a", "anthropic")
        assert decision.enabled is True
        assert decision.source == "benchmark"


class TestFilterTools:
    def test_splits_allowed_and_denied(
        self, overrides: ProviderGateStore, bench_store: BenchmarkStore
    ) -> None:
        _save_runs(bench_store, "weak_tool", "ollama", [0.0, 0.0, 0.0])
        _save_runs(bench_store, "strong_tool", "ollama", [1.0, 1.0, 1.0])
        gate = ToolProviderGate(
            overrides=overrides, benchmark_store=bench_store, min_samples=3, min_composite=0.5
        )
        allowed, denied = gate.filter_tools(["weak_tool", "strong_tool", "no_data_tool"], "ollama")
        assert set(allowed) == {"strong_tool", "no_data_tool"}
        assert "weak_tool" in denied

    def test_empty_input_returns_empty(
        self, overrides: ProviderGateStore, bench_store: BenchmarkStore
    ) -> None:
        gate = ToolProviderGate(overrides=overrides, benchmark_store=bench_store)
        allowed, denied = gate.filter_tools([], "anthropic")
        assert allowed == []
        assert denied == {}


class TestRecommendProvider:
    def test_recommends_highest_scoring_enabled_provider(
        self, overrides: ProviderGateStore, bench_store: BenchmarkStore
    ) -> None:
        _save_runs(bench_store, "tool_a", "anthropic", [1.0, 1.0, 1.0])
        _save_runs(bench_store, "tool_a", "ollama", [0.0, 0.0, 0.0])
        gate = ToolProviderGate(
            overrides=overrides, benchmark_store=bench_store, min_samples=3, min_composite=0.5
        )
        best = gate.recommend_provider("tool_a", ["anthropic", "ollama"])
        assert best == "anthropic"

    def test_returns_none_when_all_disabled(
        self, overrides: ProviderGateStore, bench_store: BenchmarkStore
    ) -> None:
        _save_runs(bench_store, "tool_a", "ollama", [0.0, 0.0, 0.0])
        gate = ToolProviderGate(
            overrides=overrides, benchmark_store=bench_store, min_samples=3, min_composite=0.5
        )
        best = gate.recommend_provider("tool_a", ["ollama"])
        assert best is None

    def test_returns_none_for_empty_candidates(
        self, overrides: ProviderGateStore, bench_store: BenchmarkStore
    ) -> None:
        gate = ToolProviderGate(overrides=overrides, benchmark_store=bench_store)
        assert gate.recommend_provider("tool_a", []) is None

