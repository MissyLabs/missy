"""Tests for F21 — LLM-judge correctness dimension."""

from __future__ import annotations

from unittest.mock import MagicMock

from missy.tools.benchmark.llm_judge import (
    _parse_score,
    make_llm_judge,
    make_structured_llm_judge,
    provider_complete_fn,
)
from missy.tools.benchmark.scoring import BenchmarkResult, BenchmarkScorer


def _result(*, actual, expected, success=True) -> BenchmarkResult:
    return BenchmarkResult(
        task_id="t1",
        tool_name="tool",
        provider="p",
        success=success,
        actual_output=actual,
        expected_output=expected,
    )


class TestParseScore:
    def test_parses_0_100_scale(self) -> None:
        assert _parse_score("85") == 0.85
        assert _parse_score("100") == 1.0
        assert _parse_score("0") == 0.0

    def test_parses_0_1_scale(self) -> None:
        assert _parse_score("0.7") == 0.7

    def test_extracts_from_prose(self) -> None:
        assert _parse_score("I would say 90 out of 100") == 0.9

    def test_clamps_out_of_range(self) -> None:
        assert _parse_score("150") == 1.0

    def test_none_when_no_number(self) -> None:
        assert _parse_score("no idea") is None
        assert _parse_score("") is None


class TestMakeLlmJudge:
    def test_judge_returns_normalized_score(self) -> None:
        judge = make_llm_judge(lambda prompt: "80")
        assert judge("expected", "actual") == 0.8

    def test_judge_raises_on_unparseable(self) -> None:
        judge = make_llm_judge(lambda prompt: "banana")
        try:
            judge("e", "a")
        except ValueError:
            pass
        else:
            raise AssertionError("expected ValueError on unparseable score")

    def test_prompt_includes_expected_actual_and_task(self) -> None:
        seen: dict = {}

        def _complete(prompt: str) -> str:
            seen["prompt"] = prompt
            return "50"

        judge = make_llm_judge(_complete, task_description="add two numbers")
        judge("4", "four")
        assert "add two numbers" in seen["prompt"]
        assert "EXPECTED" in seen["prompt"] and "ACTUAL" in seen["prompt"]


class TestScorerWithJudge:
    def test_judge_used_for_inexact_match(self) -> None:
        # Heuristic would give this a low token-overlap score; the judge
        # overrides with semantic equivalence.
        scorer = BenchmarkScorer(judge_fn=lambda e, a: 0.95)
        r = _result(actual="the capital is Paris", expected="Paris")
        assert scorer.score(r).correctness == 0.95

    def test_exact_match_skips_judge(self) -> None:
        judge = MagicMock(return_value=0.1)
        scorer = BenchmarkScorer(judge_fn=judge)
        r = _result(actual="Paris", expected="Paris")
        assert scorer.score(r).correctness == 1.0
        judge.assert_not_called()  # exact match short-circuits before the judge

    def test_judge_failure_falls_back_to_heuristic(self) -> None:
        def _boom(e, a):
            raise RuntimeError("judge down")

        scorer = BenchmarkScorer(judge_fn=_boom)
        # Heuristic substring match -> 0.8 (expected 'Paris' is inside actual).
        r = _result(actual="Paris, France", expected="Paris")
        assert scorer.score(r).correctness == 0.8

    def test_judge_nan_falls_back(self) -> None:
        scorer = BenchmarkScorer(judge_fn=lambda e, a: float("nan"))
        r = _result(actual="Paris, France", expected="Paris")
        assert scorer.score(r).correctness == 0.8

    def test_no_judge_is_pure_heuristic(self) -> None:
        scorer = BenchmarkScorer()  # default, no judge
        r = _result(actual="Paris, France", expected="Paris")
        assert scorer.score(r).correctness == 0.8

    def test_judge_score_clamped(self) -> None:
        scorer = BenchmarkScorer(judge_fn=lambda e, a: 5.0)  # out of range
        r = _result(actual="x", expected="y")
        assert scorer.score(r).correctness == 1.0

    def test_judge_not_called_when_no_expected(self) -> None:
        judge = MagicMock(return_value=0.0)
        scorer = BenchmarkScorer(judge_fn=judge)
        r = _result(actual="anything", expected=None)
        # No ground truth -> full credit, judge not consulted.
        assert scorer.score(r).correctness == 1.0
        judge.assert_not_called()


class TestProviderCompleteFn:
    def test_wraps_provider(self) -> None:
        from missy.providers.base import CompletionResponse

        provider = MagicMock()
        provider.complete.return_value = CompletionResponse(
            content="88", model="m", provider="p", usage={}, raw={}, finish_reason="stop"
        )
        fn = provider_complete_fn(provider, model="judge-model")
        assert fn("rate this") == "88"
        # The judge model override was forwarded.
        _, kwargs = provider.complete.call_args
        assert kwargs.get("model") == "judge-model"

    def test_end_to_end_provider_judge(self) -> None:
        from missy.providers.base import CompletionResponse

        provider = MagicMock()
        provider.complete.return_value = CompletionResponse(
            content="92", model="m", provider="p", usage={}, raw={}, finish_reason="stop"
        )
        judge = make_llm_judge(provider_complete_fn(provider))
        scorer = BenchmarkScorer(judge_fn=judge)
        r = _result(actual="Paris is the capital of France", expected="Paris")
        assert scorer.score(r).correctness == 0.92


class TestStructuredJudge:
    """F09 — the schema-enforced judge backed by StructuredOutputRunner."""

    def _provider(self, content: str) -> MagicMock:
        from missy.providers.base import CompletionResponse

        p = MagicMock()
        p.complete.return_value = CompletionResponse(
            content=content, model="m", provider="p", usage={}, raw={}, finish_reason="stop"
        )
        return p

    def test_valid_structured_verdict(self) -> None:
        judge = make_structured_llm_judge(self._provider('{"score": 92, "reason": "equivalent"}'))
        assert judge("Paris", "the capital is Paris") == 0.92

    def test_accepts_0_1_scale(self) -> None:
        judge = make_structured_llm_judge(self._provider('{"score": 0.5, "reason": "half"}'))
        assert judge("a", "b") == 0.5

    def test_clamps(self) -> None:
        judge = make_structured_llm_judge(self._provider('{"score": 250, "reason": "x"}'))
        assert judge("a", "b") == 1.0

    def test_invalid_json_raises_after_retries(self) -> None:
        # Never returns valid JSON -> StructuredOutputRunner exhausts retries ->
        # our judge raises so the scorer falls back to the heuristic.
        judge = make_structured_llm_judge(self._provider("not json at all"), max_retries=0)
        try:
            judge("a", "b")
        except ValueError:
            pass
        else:
            raise AssertionError("expected ValueError when structured judge can't validate")

    def test_integrates_with_scorer(self) -> None:
        judge = make_structured_llm_judge(self._provider('{"score": 88, "reason": "close"}'))
        scorer = BenchmarkScorer(judge_fn=judge)
        r = _result(actual="The capital of France is Paris", expected="Paris")
        assert scorer.score(r).correctness == 0.88

    def test_scorer_falls_back_when_structured_judge_fails(self) -> None:
        judge = make_structured_llm_judge(self._provider("garbage"), max_retries=0)
        scorer = BenchmarkScorer(judge_fn=judge)
        # Heuristic substring -> 0.8 despite the judge failing.
        r = _result(actual="Paris, France", expected="Paris")
        assert scorer.score(r).correctness == 0.8
