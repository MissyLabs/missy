"""Tests for benchmark-to-candidate reconciliation."""

from __future__ import annotations

from pathlib import Path

import pytest

from missy.tools.benchmark.benchmark_store import BenchmarkStore
from missy.tools.benchmark.scoring import BenchmarkResult, BenchmarkScorer
from missy.tools.intelligence import (
    CandidateBenchmarkReconciler,
    CandidateStore,
    ToolCandidate,
    ToolLifecycleState,
)


def _candidate_store(tmp_path: Path) -> CandidateStore:
    return CandidateStore(db_path=tmp_path / "candidates.db")


def _benchmark_store(tmp_path: Path) -> BenchmarkStore:
    return BenchmarkStore(db_path=tmp_path / "benchmarks.db")


def _candidate(name: str = "calc_candidate") -> ToolCandidate:
    return ToolCandidate.create(
        name=name,
        description="calculator candidate",
        schema={
            "type": "object",
            "properties": {"expression": {"type": "string"}},
            "required": ["expression"],
        },
        permissions={"network": False, "shell": False},
        provenance="unit test",
    )


def _save_result(
    store: BenchmarkStore,
    tool_name: str,
    provider: str,
    *,
    success: bool = True,
    safety_violation: bool = False,
    tool_call_args: dict[str, str] | None = None,
    required_params: list[str] | None = None,
) -> None:
    scorer = BenchmarkScorer()
    result = BenchmarkResult(
        task_id=f"{tool_name}-{provider}",
        tool_name=tool_name,
        provider=provider,
        success=success,
        latency_ms=10.0 if success else 20_000.0,
        cost_usd=0.0,
        actual_output="4" if success else "wrong",
        expected_output="4",
        tool_call_made=success,
        tool_call_args=tool_call_args if tool_call_args is not None else {"expression": "2+2"},
        schema_required_params=required_params if required_params is not None else ["expression"],
        safety_violation=safety_violation,
        error="" if success else "boom",
    )
    store.save(scorer.score(result))


def test_reconcile_imports_provider_summaries_and_marks_benchmarked(tmp_path: Path) -> None:
    candidates = _candidate_store(tmp_path)
    benchmarks = _benchmark_store(tmp_path)
    candidate = candidates.add(_candidate("calc_candidate"))

    for _ in range(3):
        _save_result(benchmarks, "calc_candidate", "anthropic")
        _save_result(benchmarks, "calc_candidate", "ollama", success=False)

    reconciler = CandidateBenchmarkReconciler(candidates, benchmarks)
    result = reconciler.reconcile_candidate(candidate.id)

    assert result is not None
    assert result.candidate.state == ToolLifecycleState.BENCHMARKED
    assert result.candidate.provider_enabled["anthropic"] is True
    assert result.candidate.provider_enabled["ollama"] is False
    assert {s.provider for s in result.candidate.benchmark_scores} == {"anthropic", "ollama"}


def test_reconcile_can_import_from_tool_name_override(tmp_path: Path) -> None:
    candidates = _candidate_store(tmp_path)
    benchmarks = _benchmark_store(tmp_path)
    candidate = candidates.add(_candidate("candidate_name"))
    for _ in range(3):
        _save_result(benchmarks, "registered_tool", "mock")

    reconciler = CandidateBenchmarkReconciler(candidates, benchmarks)
    result = reconciler.reconcile_candidate(candidate.id, tool_name="registered_tool")

    assert result is not None
    assert result.candidate.benchmark_scores[0].provider == "mock"
    assert result.candidate.provider_enabled["mock"] is True


def test_reconcile_requires_benchmark_data(tmp_path: Path) -> None:
    candidates = _candidate_store(tmp_path)
    benchmarks = _benchmark_store(tmp_path)
    candidate = candidates.add(_candidate())
    reconciler = CandidateBenchmarkReconciler(candidates, benchmarks)

    with pytest.raises(ValueError, match="no benchmark summaries"):
        reconciler.reconcile_candidate(candidate.id)


def test_reconcile_rejects_insufficient_samples_for_provider_flag(tmp_path: Path) -> None:
    candidates = _candidate_store(tmp_path)
    benchmarks = _benchmark_store(tmp_path)
    candidate = candidates.add(_candidate("calc_candidate"))
    _save_result(benchmarks, "calc_candidate", "anthropic")

    reconciler = CandidateBenchmarkReconciler(candidates, benchmarks, min_samples=3)
    result = reconciler.reconcile_candidate(candidate.id)

    assert result is not None
    decision = result.decisions[0]
    assert decision.enabled is False
    assert "only 1 run" in decision.reason
    assert result.candidate.provider_enabled["anthropic"] is False


def test_reconcile_keeps_missing_candidate_as_none(tmp_path: Path) -> None:
    reconciler = CandidateBenchmarkReconciler(
        _candidate_store(tmp_path),
        _benchmark_store(tmp_path),
    )

    assert reconciler.reconcile_candidate("missing") is None
