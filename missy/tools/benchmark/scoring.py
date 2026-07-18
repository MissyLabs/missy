"""Benchmark scoring — compute composite quality scores for tool runs.

A :class:`BenchmarkResult` captures the raw measurements from a single
benchmark task execution.  :class:`BenchmarkScorer` converts those
measurements into a normalised composite score in ``[0, 1]``.

Scoring dimensions
------------------

* **correctness** — binary or graded match between actual and expected output.
* **latency** — inverse latency score (faster is better), capped at a ceiling.
* **cost** — inverse cost score (cheaper is better), capped at a ceiling.
* **reliability** — fraction of non-error / non-timeout runs in the suite.
* **safety** — flagged ``False`` when the result contains a safety violation.
* **schema_score** — fraction of required parameters present in the tool call.
* **tool_call_quality** — penalises hallucinated or absent tool calls.

The composite is a weighted average controlled by
:attr:`BenchmarkScorer.weights`.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class BenchmarkResult:
    """Raw measurements from a single benchmark task run.

    Attributes:
        task_id: Identifier of the :class:`~.runner.BenchmarkTask` that
            produced this result.
        tool_name: Name of the tool being benchmarked.
        provider: Provider identifier that ran the task.
        success: ``True`` when the tool call completed without an exception.
        latency_ms: Wall-clock latency in milliseconds.
        cost_usd: Estimated cost in US dollars (``0.0`` if unknown).
        actual_output: The value returned by the tool (or an error string).
        expected_output: The expected value from the task definition.
        tool_call_made: ``True`` when the provider actually emitted a
            tool-call event (rather than answering in plain text).
        tool_call_args: The arguments the provider supplied to the tool.
        schema_required_params: Parameter names that are marked ``required``
            in the tool schema (used to compute ``schema_score``).
        safety_violation: ``True`` when a safety policy was triggered.
        error: Error message when ``success`` is ``False``.
        metadata: Arbitrary extra context.
    """

    task_id: str
    tool_name: str
    provider: str
    success: bool
    latency_ms: float = 0.0
    cost_usd: float = 0.0
    actual_output: Any = None
    expected_output: Any = None
    tool_call_made: bool = False
    tool_call_args: dict[str, Any] = field(default_factory=dict)
    schema_required_params: list[str] = field(default_factory=list)
    safety_violation: bool = False
    error: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "tool_name": self.tool_name,
            "provider": self.provider,
            "success": self.success,
            "latency_ms": self.latency_ms,
            "cost_usd": self.cost_usd,
            "actual_output": str(self.actual_output) if self.actual_output is not None else None,
            "expected_output": (
                str(self.expected_output) if self.expected_output is not None else None
            ),
            "tool_call_made": self.tool_call_made,
            "tool_call_args": self.tool_call_args,
            "schema_required_params": self.schema_required_params,
            "safety_violation": self.safety_violation,
            "error": self.error,
            "metadata": self.metadata,
        }


@dataclass
class ScoredResult:
    """A :class:`BenchmarkResult` enriched with dimension scores.

    Attributes:
        result: The raw measurement.
        correctness: ``[0, 1]`` match quality.
        latency_score: ``[0, 1]`` inverse-latency score.
        cost_score: ``[0, 1]`` inverse-cost score.
        reliability: ``[0, 1]`` (1.0 when no error, 0.0 on timeout/exception).
        safety: ``1.0`` when no safety violation, ``0.0`` otherwise.
        schema_score: Fraction of required params present in ``tool_call_args``.
        tool_call_quality: ``1.0`` when a tool call was made, ``0.0`` otherwise.
        composite: Weighted average of all dimension scores.
    """

    result: BenchmarkResult
    correctness: float = 0.0
    latency_score: float = 0.0
    cost_score: float = 0.0
    reliability: float = 0.0
    safety: float = 0.0
    schema_score: float = 0.0
    tool_call_quality: float = 0.0
    composite: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        d = self.result.to_dict()
        d.update(
            {
                "correctness": round(self.correctness, 4),
                "latency_score": round(self.latency_score, 4),
                "cost_score": round(self.cost_score, 4),
                "reliability": round(self.reliability, 4),
                "safety": round(self.safety, 4),
                "schema_score": round(self.schema_score, 4),
                "tool_call_quality": round(self.tool_call_quality, 4),
                "composite": round(self.composite, 4),
            }
        )
        return d


@dataclass
class ScoreWeights:
    """Relative weights for the composite score.  Automatically normalised."""

    correctness: float = 3.0
    latency: float = 1.0
    cost: float = 0.5
    reliability: float = 2.0
    safety: float = 2.0
    schema_score: float = 1.5
    tool_call_quality: float = 2.0

    def normalised(self) -> dict[str, float]:
        total = sum(
            [
                self.correctness,
                self.latency,
                self.cost,
                self.reliability,
                self.safety,
                self.schema_score,
                self.tool_call_quality,
            ]
        )
        if total == 0:
            total = 1.0
        return {
            "correctness": self.correctness / total,
            "latency": self.latency / total,
            "cost": self.cost / total,
            "reliability": self.reliability / total,
            "safety": self.safety / total,
            "schema_score": self.schema_score / total,
            "tool_call_quality": self.tool_call_quality / total,
        }


class BenchmarkScorer:
    """Convert raw :class:`BenchmarkResult` measurements into :class:`ScoredResult`.

    Args:
        weights: :class:`ScoreWeights` controlling the composite.
        max_latency_ms: Latency at which the latency score hits zero (default 10 s).
        max_cost_usd: Cost at which the cost score hits zero (default $0.01).
    """

    def __init__(
        self,
        weights: ScoreWeights | None = None,
        max_latency_ms: float = 10_000.0,
        max_cost_usd: float = 0.01,
        judge_fn: Callable[[Any, Any], float] | None = None,
    ) -> None:
        self.weights = weights or ScoreWeights()
        self.max_latency_ms = max(max_latency_ms, 1.0)
        self.max_cost_usd = max(max_cost_usd, 1e-9)
        # F21: optional LLM-judge for the correctness dimension. When set, an
        # open-ended result (one with a ground-truth ``expected_output`` that
        # the heuristic can't confidently match) is scored semantically by the
        # judge instead. Any judge error falls back to the heuristic, so a flaky
        # judge never corrupts a benchmark run.
        self._judge_fn = judge_fn

    def score(self, result: BenchmarkResult) -> ScoredResult:
        """Compute all dimension scores and composite for *result*."""
        correctness = self._correctness(result)
        latency_score = self._latency_score(result.latency_ms)
        cost_score = self._cost_score(result.cost_usd)
        reliability = 0.0 if (not result.success or bool(result.error)) else 1.0
        safety = 0.0 if result.safety_violation else 1.0
        schema_score = self._schema_score(result)
        tool_call_quality = 1.0 if result.tool_call_made else 0.0

        w = self.weights.normalised()
        composite = (
            w["correctness"] * correctness
            + w["latency"] * latency_score
            + w["cost"] * cost_score
            + w["reliability"] * reliability
            + w["safety"] * safety
            + w["schema_score"] * schema_score
            + w["tool_call_quality"] * tool_call_quality
        )

        return ScoredResult(
            result=result,
            correctness=correctness,
            latency_score=latency_score,
            cost_score=cost_score,
            reliability=reliability,
            safety=safety,
            schema_score=schema_score,
            tool_call_quality=tool_call_quality,
            composite=min(1.0, max(0.0, composite)),
        )

    def score_batch(self, results: list[BenchmarkResult]) -> list[ScoredResult]:
        """Score all results in *results*."""
        return [self.score(r) for r in results]

    def aggregate(self, scored: list[ScoredResult]) -> dict[str, float]:
        """Compute mean scores across *scored*.

        Returns:
            Dict with mean for each dimension plus ``composite``.
        """
        if not scored:
            return {}
        dims = [
            "correctness",
            "latency_score",
            "cost_score",
            "reliability",
            "safety",
            "schema_score",
            "tool_call_quality",
            "composite",
        ]
        return {dim: round(sum(getattr(s, dim) for s in scored) / len(scored), 4) for dim in dims}

    # ------------------------------------------------------------------
    # Dimension scorers
    # ------------------------------------------------------------------

    def _correctness(self, result: BenchmarkResult) -> float:
        if not result.success:
            return 0.0
        actual = result.actual_output
        expected = result.expected_output
        if expected is None:
            # No ground truth — full credit if tool succeeded.
            return 1.0 if result.success else 0.0
        if actual == expected:
            return 1.0
        # F21: when a judge is configured, defer semantic scoring to it for the
        # inexact case (an exact match above is trivially correct and needs no
        # judge). A judge failure falls through to the heuristics below.
        if self._judge_fn is not None:
            score = self._judge_correctness(expected, actual)
            if score is not None:
                return score
        # Partial string match.
        a_str = str(actual).lower().strip()
        e_str = str(expected).lower().strip()
        if e_str in a_str:
            return 0.8
        # Numeric tolerance.
        try:
            a_num = float(_strip_non_numeric(a_str))
            e_num = float(_strip_non_numeric(e_str))
            if e_num == 0:
                return 1.0 if a_num == 0 else 0.0
            rel_err = abs(a_num - e_num) / abs(e_num)
            return max(0.0, 1.0 - rel_err)
        except (ValueError, ZeroDivisionError):
            pass
        # Token overlap.
        a_tokens = set(a_str.split())
        e_tokens = set(e_str.split())
        if not e_tokens:
            return 0.0
        return len(a_tokens & e_tokens) / len(e_tokens)

    def _judge_correctness(self, expected: Any, actual: Any) -> float | None:
        """Score correctness via the configured judge, clamped to [0, 1].

        Returns ``None`` (so the caller falls back to the heuristic) when no
        judge is set, the judge raises, or it returns a non-numeric value — a
        judge must never be able to crash or corrupt a benchmark run.
        """
        if self._judge_fn is None:
            return None
        try:
            raw = self._judge_fn(expected, actual)
            score = float(raw)
        except Exception:  # noqa: BLE001 - an untrusted judge must never crash a run
            logger.debug("LLM judge failed; falling back to heuristic correctness.", exc_info=True)
            return None
        if score != score:  # NaN guard
            return None
        return max(0.0, min(1.0, score))

    def _latency_score(self, latency_ms: float) -> float:
        if latency_ms <= 0:
            return 1.0
        return max(0.0, 1.0 - latency_ms / self.max_latency_ms)

    def _cost_score(self, cost_usd: float) -> float:
        if cost_usd <= 0:
            return 1.0
        return max(0.0, 1.0 - cost_usd / self.max_cost_usd)

    def _schema_score(self, result: BenchmarkResult) -> float:
        required = result.schema_required_params
        if not required:
            return 1.0
        supplied = set(result.tool_call_args.keys())
        present = sum(1 for p in required if p in supplied)
        return present / len(required)


def _strip_non_numeric(s: str) -> str:
    return re.sub(r"[^0-9.eE+-]", "", s)
