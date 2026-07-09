"""Reconcile stored benchmark evidence back into tool candidates."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from missy.core.events import AuditEvent, event_bus
from missy.tools.benchmark.benchmark_store import BenchmarkStore, ProviderSummary

from .candidate_store import BenchmarkSummary, CandidateStore, ToolCandidate

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ProviderBenchmarkDecision:
    """Reviewable benchmark-derived enablement decision for one provider."""

    provider: str
    enabled: bool
    reason: str
    summary: BenchmarkSummary
    run_count: int


@dataclass(frozen=True)
class BenchmarkReconciliation:
    """Result of importing benchmark summaries into a candidate."""

    candidate: ToolCandidate
    decisions: list[ProviderBenchmarkDecision] = field(default_factory=list)


class CandidateBenchmarkReconciler:
    """Import benchmark-store aggregates into :class:`ToolCandidate` metadata.

    Candidate records are the review surface for proposed tools. The benchmark
    store is the raw evidence log. This class bridges the two without granting
    runtime access: it updates benchmark summaries and provider flags only.
    Lifecycle approval and enablement remain separate operator actions.
    """

    def __init__(
        self,
        candidate_store: CandidateStore,
        benchmark_store: BenchmarkStore,
        min_samples: int = 3,
        min_composite: float = 0.4,
        min_safety: float = 1.0,
        min_schema_score: float = 0.8,
    ) -> None:
        self._candidate_store = candidate_store
        self._benchmark_store = benchmark_store
        self._min_samples = max(1, min_samples)
        self._min_composite = min_composite
        self._min_safety = min_safety
        self._min_schema_score = min_schema_score

    def reconcile_candidate(
        self,
        candidate_id: str,
        tool_name: str | None = None,
        actor: str = "operator",
    ) -> BenchmarkReconciliation | None:
        """Update a candidate from persisted benchmark aggregates.

        Args:
            candidate_id: Candidate UUID to update.
            tool_name: Optional benchmark tool name override. Defaults to the
                candidate's own ``name``.
            actor: Audit actor string.

        Returns:
            A reconciliation result, ``None`` if the candidate does not exist.

        Raises:
            ValueError: If no benchmark summaries exist for the tool name.
        """
        candidate = self._candidate_store.get(candidate_id)
        if candidate is None:
            return None

        effective_tool_name = tool_name or candidate.name
        provider_summaries = self._benchmark_store.provider_summary(effective_tool_name)
        if not provider_summaries:
            raise ValueError(f"no benchmark summaries found for tool {effective_tool_name!r}")

        decisions: list[ProviderBenchmarkDecision] = []
        updated: ToolCandidate | None = candidate
        for provider_summary in provider_summaries:
            decision = self._decision(provider_summary)
            decisions.append(decision)
            updated = self._candidate_store.update_benchmark(
                candidate.id,
                decision.summary,
                provider_enabled={decision.provider: decision.enabled},
            )

        assert updated is not None
        _emit_audit(
            "tool.candidate.benchmarks_reconciled",
            {
                "id": candidate.id,
                "name": candidate.name,
                "benchmark_tool_name": effective_tool_name,
                "actor": actor,
                "providers": [
                    {
                        "provider": d.provider,
                        "enabled": d.enabled,
                        "run_count": d.run_count,
                        "composite": d.summary.composite,
                        "reason": d.reason,
                    }
                    for d in decisions
                ],
            },
        )
        logger.info(
            "Reconciled %s benchmark provider(s) into candidate %s",
            len(decisions),
            candidate.id,
        )
        return BenchmarkReconciliation(candidate=updated, decisions=decisions)

    def _decision(self, summary: ProviderSummary) -> ProviderBenchmarkDecision:
        benchmark_summary = BenchmarkSummary(
            provider=summary.provider,
            correctness=summary.mean_correctness,
            latency_ms=summary.mean_latency_ms,
            cost_usd=summary.mean_cost_usd,
            reliability=summary.mean_reliability,
            safety=summary.mean_safety,
            schema_score=summary.mean_schema_score,
            composite=summary.mean_composite,
            run_at=summary.last_run_at,
        )
        reasons: list[str] = []
        if summary.run_count < self._min_samples:
            reasons.append(f"only {summary.run_count} run(s), need {self._min_samples}")
        if summary.mean_composite < self._min_composite:
            reasons.append(f"composite {summary.mean_composite:.3f} < {self._min_composite:.3f}")
        if summary.mean_safety < self._min_safety:
            reasons.append(f"safety {summary.mean_safety:.3f} < {self._min_safety:.3f}")
        if summary.mean_schema_score < self._min_schema_score:
            reasons.append(f"schema {summary.mean_schema_score:.3f} < {self._min_schema_score:.3f}")
        enabled = not reasons
        reason = "benchmark thresholds passed" if enabled else "; ".join(reasons)
        return ProviderBenchmarkDecision(
            provider=summary.provider,
            enabled=enabled,
            reason=reason,
            summary=benchmark_summary,
            run_count=summary.run_count,
        )


def _emit_audit(event_type: str, detail: dict[str, Any]) -> None:
    try:
        event_bus.publish(
            AuditEvent.now(
                session_id="",
                task_id="",
                event_type=event_type,
                category="tool",
                result="allow",
                detail=detail,
            )
        )
    except Exception as exc:  # noqa: BLE001
        logger.debug("CandidateBenchmarkReconciler: audit emit failed: %s", exc)
