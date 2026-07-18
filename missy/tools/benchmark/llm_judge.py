"""LLM-judge correctness scoring for benchmarks (F21).

The default benchmark ``correctness`` dimension is heuristic (exact / substring /
numeric-tolerance / token-overlap). That undersells open-ended tool output whose
*meaning* matches the expected answer without matching its characters. F21 adds
an optional judge: a model rates how well the actual output satisfies the
expected one, feeding the same composite/provider-gating pipeline.

``make_llm_judge`` turns any ``complete_fn(prompt) -> str`` into the
``judge_fn(expected, actual) -> float`` that
:class:`~missy.tools.benchmark.scoring.BenchmarkScorer` accepts. Keeping the
seam a plain callable makes it trivially testable (pass a lambda) and provider-
agnostic; ``provider_complete_fn`` adapts a real ``BaseProvider``.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Callable
from typing import Any

logger = logging.getLogger(__name__)

_JUDGE_SYSTEM = (
    "You are a strict evaluator scoring how well an ACTUAL answer matches an "
    "EXPECTED answer for a tool benchmark. Reply with ONLY an integer 0-100, "
    "where 100 means semantically equivalent and 0 means entirely wrong. No prose."
)

_SCORE_RE = re.compile(r"-?\d+(?:\.\d+)?")


def _build_prompt(expected: Any, actual: Any, task_description: str = "") -> str:
    parts = [_JUDGE_SYSTEM, ""]
    if task_description:
        parts.append(f"TASK: {task_description}")
    parts.append(f"EXPECTED: {expected!r}")
    parts.append(f"ACTUAL: {actual!r}")
    parts.append("SCORE (0-100):")
    return "\n".join(parts)


def _parse_score(text: str) -> float | None:
    """Parse the first number in *text* as a 0..1 correctness score."""
    m = _SCORE_RE.search(text or "")
    if not m:
        return None
    try:
        val = float(m.group(0))
    except ValueError:
        return None
    # Accept either a 0-100 scale (the prompt) or an already-0..1 value.
    if val > 1.0:
        val /= 100.0
    return max(0.0, min(1.0, val))


def make_llm_judge(
    complete_fn: Callable[[str], str],
    *,
    task_description: str = "",
) -> Callable[[Any, Any], float]:
    """Return a ``judge_fn(expected, actual) -> float`` backed by *complete_fn*.

    Args:
        complete_fn: Sends a prompt string to a model and returns its text.
        task_description: Optional context included in the judge prompt.

    Returns:
        A callable suitable for ``BenchmarkScorer(judge_fn=...)``. It never
        raises: on any error it returns 0.0, and the scorer's own guard treats
        a non-numeric/failed judge as "fall back to heuristic".
    """

    def _judge(expected: Any, actual: Any) -> float:
        prompt = _build_prompt(expected, actual, task_description)
        text = complete_fn(prompt)
        score = _parse_score(str(text))
        if score is None:
            raise ValueError(f"judge returned an unparseable score: {text!r}")
        return score

    return _judge


def provider_complete_fn(provider: Any, model: str | None = None) -> Callable[[str], str]:
    """Adapt a ``BaseProvider`` into a ``complete_fn(prompt) -> str`` for judging."""
    from missy.providers.base import Message

    def _complete(prompt: str) -> str:
        kwargs: dict = {"temperature": 0.0}
        if model:
            kwargs["model"] = model
        resp = provider.complete([Message(role="user", content=prompt)], **kwargs)
        return str(getattr(resp, "content", "") or "")

    return _complete


def make_structured_llm_judge(
    provider: Any,
    *,
    model: str | None = None,
    task_description: str = "",
    max_retries: int = 1,
) -> Callable[[Any, Any], float]:
    """Return a schema-enforced ``judge_fn`` backed by ``StructuredOutputRunner``.

    This is the robust variant of :func:`make_llm_judge` and a real adopter of
    the previously-unwired ``StructuredOutputRunner`` (F09): the judge model must
    return JSON validating against a ``{score: float, reason: str}`` schema, with
    automatic retry-on-invalid-JSON (error fed back to the model). A number is no
    longer scraped from free text — an invalid verdict is retried, then raises so
    the scorer falls back to its heuristic.

    Args:
        provider: A ``BaseProvider`` used for the judgment call.
        model: Optional model override for the judge.
        task_description: Optional context for the judge prompt.
        max_retries: Extra attempts on schema-validation failure.

    Returns:
        A ``judge_fn(expected, actual) -> float`` for ``BenchmarkScorer``.
    """
    from pydantic import BaseModel, Field

    from missy.agent.structured_output import OutputSchema, StructuredOutputRunner
    from missy.providers.base import Message

    class JudgeVerdict(BaseModel):
        score: float = Field(description="Correctness 0-100 (100 = equivalent).")
        reason: str = Field(default="", description="Brief justification.")

    schema = OutputSchema(JudgeVerdict, max_retries=max_retries)
    runner = StructuredOutputRunner(provider)

    def _judge(expected: Any, actual: Any) -> float:
        prompt = _build_prompt(expected, actual, task_description)
        kwargs: dict = {}
        if model:
            kwargs["model"] = model
        result = runner.complete_structured(
            [Message(role="user", content=prompt)], schema, **kwargs
        )
        if not result.success or result.data is None:
            raise ValueError(
                f"structured judge failed after {result.attempts} attempt(s): "
                f"{result.validation_errors}"
            )
        raw = float(result.data.score)
        if raw > 1.0:
            raw /= 100.0
        return max(0.0, min(1.0, raw))

    return _judge
