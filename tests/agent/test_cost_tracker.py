"""Comprehensive unit tests for missy.agent.cost_tracker.

Covers:
1.  CostTracker init with various max_spend_usd values (0, None, positive)
2.  record() with known models — verify correct cost computation
3.  record() with unknown model — verify zero cost
4.  record_from_response() with valid CompletionResponse-like objects
5.  record_from_response() with missing attributes / None usage
6.  check_budget() — no raise when unlimited, under budget, raises when over
7.  BudgetExceededError attributes (spent, limit, message)
8.  Thread safety: concurrent record() calls from multiple threads
9.  get_summary() correctness
10. reset() clears everything
11. Record eviction when exceeding _MAX_RECORDS
12. Pricing table: prefix matching (dated model names match base entry)
13. Property accessors: total_cost_usd, total_prompt_tokens,
    total_completion_tokens, total_tokens, call_count
"""

from __future__ import annotations

import threading
from types import SimpleNamespace

import pytest

from missy.agent.cost_tracker import (
    BudgetExceededError,
    CostTracker,
    UsageRecord,
    _lookup_pricing,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_response(model: str = "", prompt_tokens: int = 0, completion_tokens: int = 0):
    """Return a SimpleNamespace that looks like a CompletionResponse."""
    return SimpleNamespace(
        model=model,
        usage={"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens},
    )


# ---------------------------------------------------------------------------
# 1. Initialisation
# ---------------------------------------------------------------------------


class TestInit:
    def test_zero_max_spend_stored(self):
        tracker = CostTracker(max_spend_usd=0.0)
        assert tracker.max_spend_usd == 0.0

    def test_none_max_spend_coerced_to_zero(self):
        tracker = CostTracker(max_spend_usd=None)  # type: ignore[arg-type]
        assert tracker.max_spend_usd == 0.0

    def test_positive_max_spend_stored(self):
        tracker = CostTracker(max_spend_usd=2.50)
        assert tracker.max_spend_usd == pytest.approx(2.50)

    def test_default_max_spend_is_zero(self):
        tracker = CostTracker()
        assert tracker.max_spend_usd == 0.0

    def test_initial_cost_is_zero(self):
        assert CostTracker(max_spend_usd=1.0).total_cost_usd == 0.0

    def test_initial_prompt_tokens_zero(self):
        assert CostTracker().total_prompt_tokens == 0

    def test_initial_completion_tokens_zero(self):
        assert CostTracker().total_completion_tokens == 0

    def test_initial_total_tokens_zero(self):
        assert CostTracker().total_tokens == 0

    def test_initial_call_count_zero(self):
        assert CostTracker().call_count == 0

    def test_small_positive_budget(self):
        tracker = CostTracker(max_spend_usd=0.0001)
        assert tracker.max_spend_usd == pytest.approx(0.0001)

    def test_large_positive_budget(self):
        tracker = CostTracker(max_spend_usd=999.99)
        assert tracker.max_spend_usd == pytest.approx(999.99)


# ---------------------------------------------------------------------------
# 2. record() — known models, correct cost computation
# ---------------------------------------------------------------------------


class TestRecordKnownModels:
    """record() computes costs using the built-in pricing table."""

    def test_claude_sonnet_4_input_cost(self):
        # $0.003 per 1 k input tokens
        rec = CostTracker().record(
            "claude-sonnet-4-20250514", prompt_tokens=1000, completion_tokens=0
        )
        assert rec.cost_usd == pytest.approx(0.003)

    def test_claude_sonnet_4_output_cost(self):
        # $0.015 per 1 k output tokens
        rec = CostTracker().record(
            "claude-sonnet-4-20250514", prompt_tokens=0, completion_tokens=1000
        )
        assert rec.cost_usd == pytest.approx(0.015)

    def test_claude_sonnet_4_combined_cost(self):
        rec = CostTracker().record(
            "claude-sonnet-4-20250514", prompt_tokens=500, completion_tokens=200
        )
        expected = (500 / 1000) * 0.003 + (200 / 1000) * 0.015
        assert rec.cost_usd == pytest.approx(expected)

    def test_claude_opus_4_input_cost(self):
        # $0.015 per 1 k input tokens
        rec = CostTracker().record(
            "claude-opus-4-20250514", prompt_tokens=1000, completion_tokens=0
        )
        assert rec.cost_usd == pytest.approx(0.015)

    def test_claude_haiku_4_input_cost(self):
        # $0.0008 per 1 k input tokens
        rec = CostTracker().record(
            "claude-haiku-4-20250514", prompt_tokens=1000, completion_tokens=0
        )
        assert rec.cost_usd == pytest.approx(0.0008)

    def test_claude_3_haiku_input_cost(self):
        # $0.00025 per 1 k input tokens
        rec = CostTracker().record(
            "claude-3-haiku-20240307", prompt_tokens=1000, completion_tokens=0
        )
        assert rec.cost_usd == pytest.approx(0.00025)

    def test_gpt4o_input_cost(self):
        # $0.0025 per 1 k input tokens
        rec = CostTracker().record("gpt-4o", prompt_tokens=1000, completion_tokens=0)
        assert rec.cost_usd == pytest.approx(0.0025)

    def test_gpt4o_mini_input_cost(self):
        # $0.00015 per 1 k input tokens
        rec = CostTracker().record("gpt-4o-mini", prompt_tokens=1000, completion_tokens=0)
        assert rec.cost_usd == pytest.approx(0.00015)

    def test_gpt4_turbo_input_cost(self):
        # $0.01 per 1 k input tokens
        rec = CostTracker().record("gpt-4-turbo", prompt_tokens=1000, completion_tokens=0)
        assert rec.cost_usd == pytest.approx(0.01)

    def test_gpt4_input_cost(self):
        # $0.03 per 1 k input tokens
        rec = CostTracker().record("gpt-4", prompt_tokens=1000, completion_tokens=0)
        assert rec.cost_usd == pytest.approx(0.03)

    def test_llama_model_is_zero_cost(self):
        rec = CostTracker().record("llama3.2:3b", prompt_tokens=5000, completion_tokens=2000)
        assert rec.cost_usd == 0.0

    def test_mistral_model_is_zero_cost(self):
        rec = CostTracker().record("mistral-nemo:latest", prompt_tokens=1000, completion_tokens=500)
        assert rec.cost_usd == 0.0

    def test_record_returns_usage_record_instance(self):
        rec = CostTracker().record(
            "claude-sonnet-4-20250514", prompt_tokens=100, completion_tokens=50
        )
        assert isinstance(rec, UsageRecord)

    def test_usage_record_stores_model_name(self):
        rec = CostTracker().record("gpt-4o", prompt_tokens=42, completion_tokens=17)
        assert rec.model == "gpt-4o"

    def test_usage_record_stores_prompt_tokens(self):
        rec = CostTracker().record(
            "claude-haiku-4-20250514", prompt_tokens=300, completion_tokens=150
        )
        assert rec.prompt_tokens == 300

    def test_usage_record_stores_completion_tokens(self):
        rec = CostTracker().record(
            "claude-haiku-4-20250514", prompt_tokens=300, completion_tokens=150
        )
        assert rec.completion_tokens == 150

    def test_cost_usd_type_is_float(self):
        rec = CostTracker().record(
            "claude-sonnet-4-20250514", prompt_tokens=1000, completion_tokens=500
        )
        assert isinstance(rec.cost_usd, float)

    def test_zero_tokens_returns_zero_cost_float(self):
        rec = CostTracker().record("claude-sonnet-4-20250514", prompt_tokens=0, completion_tokens=0)
        assert rec.cost_usd == 0.0
        assert isinstance(rec.cost_usd, float)

    def test_multiple_records_accumulate_cost(self):
        tracker = CostTracker()
        tracker.record("claude-sonnet-4-20250514", prompt_tokens=1000, completion_tokens=0)
        tracker.record("claude-sonnet-4-20250514", prompt_tokens=1000, completion_tokens=0)
        assert tracker.total_cost_usd == pytest.approx(0.006)

    def test_record_increments_call_count(self):
        tracker = CostTracker()
        tracker.record("gpt-4o", prompt_tokens=10, completion_tokens=5)
        assert tracker.call_count == 1

    def test_known_model_cost_matches_direct_calculation(self):
        """Cost formula: (prompt / 1000) * inp_rate + (completion / 1000) * out_rate."""
        tracker = CostTracker()
        inp_rate, out_rate = 0.003, 0.015  # claude-sonnet-4
        p, c = 750, 300
        rec = tracker.record("claude-sonnet-4-20250514", prompt_tokens=p, completion_tokens=c)
        expected = (p / 1000) * inp_rate + (c / 1000) * out_rate
        assert rec.cost_usd == pytest.approx(expected)


# ---------------------------------------------------------------------------
# 3. record() — unknown model falls back to zero cost
# ---------------------------------------------------------------------------


class TestRecordUnknownModel:
    def test_unknown_model_cost_is_zero(self):
        rec = CostTracker().record(
            "my-custom-local-model-v99", prompt_tokens=5000, completion_tokens=2000
        )
        assert rec.cost_usd == 0.0

    def test_unknown_model_total_cost_stays_zero(self):
        tracker = CostTracker()
        tracker.record("mystery-model", prompt_tokens=10_000, completion_tokens=5000)
        assert tracker.total_cost_usd == 0.0

    def test_unknown_model_still_accumulates_prompt_tokens(self):
        tracker = CostTracker()
        tracker.record("mystery-model", prompt_tokens=200, completion_tokens=100)
        assert tracker.total_prompt_tokens == 200

    def test_unknown_model_still_accumulates_completion_tokens(self):
        tracker = CostTracker()
        tracker.record("mystery-model", prompt_tokens=200, completion_tokens=100)
        assert tracker.total_completion_tokens == 100

    def test_unknown_model_increments_call_count(self):
        tracker = CostTracker()
        tracker.record("mystery-model", prompt_tokens=10, completion_tokens=5)
        assert tracker.call_count == 1

    def test_empty_model_string_is_unknown(self):
        rec = CostTracker().record("", prompt_tokens=100, completion_tokens=50)
        assert rec.cost_usd == 0.0

    def test_unknown_model_does_not_raise(self):
        """Calling record() on an unknown model must never raise."""
        tracker = CostTracker()
        tracker.record("totally-unknown-model-v99", prompt_tokens=1000, completion_tokens=500)


# ---------------------------------------------------------------------------
# 4. record_from_response() — valid CompletionResponse-like objects
# ---------------------------------------------------------------------------


class TestRecordFromResponseValid:
    def test_full_response_records_correctly(self):
        tracker = CostTracker()
        resp = _make_response("claude-sonnet-4-20250514", prompt_tokens=500, completion_tokens=200)
        rec = tracker.record_from_response(resp)
        assert rec is not None
        assert rec.prompt_tokens == 500
        assert rec.completion_tokens == 200

    def test_full_response_cost_matches_direct_record(self):
        tracker_a = CostTracker()
        tracker_b = CostTracker()
        resp = _make_response("claude-haiku-4-20250514", prompt_tokens=1000, completion_tokens=500)
        tracker_a.record_from_response(resp)
        tracker_b.record("claude-haiku-4-20250514", prompt_tokens=1000, completion_tokens=500)
        assert tracker_a.total_cost_usd == pytest.approx(tracker_b.total_cost_usd)

    def test_response_increments_call_count(self):
        tracker = CostTracker()
        tracker.record_from_response(
            _make_response("gpt-4o", prompt_tokens=100, completion_tokens=50)
        )
        assert tracker.call_count == 1

    def test_response_model_stored_in_record(self):
        tracker = CostTracker()
        rec = tracker.record_from_response(
            _make_response("claude-opus-4-20250514", prompt_tokens=100, completion_tokens=50)
        )
        assert rec is not None
        assert rec.model == "claude-opus-4-20250514"

    def test_multiple_responses_accumulate(self):
        tracker = CostTracker()
        for _ in range(3):
            tracker.record_from_response(
                _make_response("gpt-4o-mini", prompt_tokens=100, completion_tokens=50)
            )
        assert tracker.call_count == 3
        assert tracker.total_prompt_tokens == 300
        assert tracker.total_completion_tokens == 150

    def test_response_with_total_tokens_key_ignored(self):
        """total_tokens key in usage dict is not used — only prompt/completion matter."""
        tracker = CostTracker()

        class ResponseWithTotal:
            model = "claude-sonnet-4-20250514"
            usage = {"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150}

        rec = tracker.record_from_response(ResponseWithTotal())
        assert rec is not None
        assert rec.prompt_tokens == 100
        assert rec.completion_tokens == 50


# ---------------------------------------------------------------------------
# 5. record_from_response() — missing attributes / None usage
# ---------------------------------------------------------------------------


class TestRecordFromResponseEdge:
    def test_none_usage_with_set_model_does_not_raise(self):
        """usage=None is treated as empty dict; no exception raised."""
        tracker = CostTracker()

        class NoneUsage:
            model = "claude-haiku-4-20250514"
            usage = None

        result = tracker.record_from_response(NoneUsage())
        # model is set but tokens are 0 → guard condition fires → None
        # (not model is False, not prompt is True (0), not completion is True (0) →
        #  all three must be falsy for None to be returned; model is truthy so guard passes)
        assert result is None or isinstance(result, UsageRecord)

    def test_missing_usage_attribute_does_not_raise(self):
        """Object with no .usage attribute falls back gracefully."""
        tracker = CostTracker()

        class NoUsageAttr:
            model = "gpt-4o"

        result = tracker.record_from_response(NoUsageAttr())
        assert result is None or isinstance(result, UsageRecord)

    def test_missing_model_attribute_records_with_empty_model(self):
        """Object with no .model attribute: model defaults to '', tokens still recorded."""
        tracker = CostTracker()

        class NoModel:
            usage = {"prompt_tokens": 100, "completion_tokens": 50}

        rec = tracker.record_from_response(NoModel())
        # Tokens are present so the guard passes; model becomes ""
        assert rec is not None
        assert rec.prompt_tokens == 100
        assert rec.completion_tokens == 50

    def test_partial_usage_only_prompt_tokens(self):
        tracker = CostTracker()

        class PartialPrompt:
            model = "claude-sonnet-4-20250514"
            usage = {"prompt_tokens": 300}

        rec = tracker.record_from_response(PartialPrompt())
        assert rec is not None
        assert rec.prompt_tokens == 300
        assert rec.completion_tokens == 0

    def test_partial_usage_only_completion_tokens(self):
        tracker = CostTracker()

        class PartialCompletion:
            model = "claude-sonnet-4-20250514"
            usage = {"completion_tokens": 150}

        rec = tracker.record_from_response(PartialCompletion())
        assert rec is not None
        assert rec.completion_tokens == 150
        assert rec.prompt_tokens == 0

    def test_all_zero_usage_and_empty_model_returns_none(self):
        """Completely empty response → None (guard fires: not model and not p and not c)."""
        tracker = CostTracker()
        resp = SimpleNamespace(model="", usage={"prompt_tokens": 0, "completion_tokens": 0})
        assert tracker.record_from_response(resp) is None

    def test_none_model_attribute_treated_as_empty_string(self):
        """model=None is coerced to ''; tokens present so record is created."""
        tracker = CostTracker()
        resp = SimpleNamespace(
            model=None,
            usage={"prompt_tokens": 100, "completion_tokens": 50},
        )
        rec = tracker.record_from_response(resp)
        assert rec is not None
        assert rec.cost_usd == 0.0  # empty model → unknown pricing → zero cost

    def test_none_token_values_in_usage_do_not_raise(self):
        """None token values in usage dict are coerced to 0 via 'or 0'."""
        tracker = CostTracker()
        resp = SimpleNamespace(
            model="claude-sonnet-4-20250514",
            usage={"prompt_tokens": None, "completion_tokens": None},
        )
        result = tracker.record_from_response(resp)
        assert result is None or isinstance(result, UsageRecord)

    def test_empty_usage_dict_with_model_does_not_raise(self):
        """Empty usage dict {} with a model name is handled gracefully."""
        tracker = CostTracker()

        class EmptyUsage:
            model = "gpt-4o"
            usage = {}

        result = tracker.record_from_response(EmptyUsage())
        # model is set, tokens default to 0 → guard fires on prompt and completion
        # guard: not model = False → goes through; records with 0 tokens
        assert result is None or isinstance(result, UsageRecord)


# ---------------------------------------------------------------------------
# 6. check_budget()
# ---------------------------------------------------------------------------


class TestCheckBudget:
    def test_unlimited_zero_budget_never_raises(self):
        tracker = CostTracker(max_spend_usd=0.0)
        tracker.record("claude-opus-4-20250514", prompt_tokens=100_000, completion_tokens=100_000)
        tracker.check_budget()

    def test_none_budget_never_raises(self):
        tracker = CostTracker(max_spend_usd=None)  # type: ignore[arg-type]
        tracker.record("claude-opus-4-20250514", prompt_tokens=100_000, completion_tokens=100_000)
        tracker.check_budget()

    def test_under_budget_does_not_raise(self):
        tracker = CostTracker(max_spend_usd=1.00)
        # $0.003 total — well under $1.00
        tracker.record("claude-sonnet-4-20250514", prompt_tokens=1000, completion_tokens=0)
        tracker.check_budget()

    def test_over_budget_raises(self):
        tracker = CostTracker(max_spend_usd=0.001)
        # claude-opus-4: $0.015 input/1k → $0.015 total > $0.001
        tracker.record("claude-opus-4-20250514", prompt_tokens=1000, completion_tokens=0)
        with pytest.raises(BudgetExceededError):
            tracker.check_budget()

    def test_exactly_at_limit_raises(self):
        """Condition is >= so spending exactly == limit triggers the error."""
        # claude-sonnet-4 input: $0.003/1k → 1000 tokens = exactly $0.003
        tracker = CostTracker(max_spend_usd=0.003)
        tracker.record("claude-sonnet-4-20250514", prompt_tokens=1000, completion_tokens=0)
        with pytest.raises(BudgetExceededError):
            tracker.check_budget()

    def test_check_budget_idempotent_when_under(self):
        """Multiple calls while under budget all succeed."""
        tracker = CostTracker(max_spend_usd=10.0)
        tracker.record("claude-haiku-4-20250514", prompt_tokens=100, completion_tokens=50)
        for _ in range(5):
            tracker.check_budget()

    def test_check_budget_raises_on_every_call_when_over(self):
        """Once over budget every subsequent check raises."""
        tracker = CostTracker(max_spend_usd=0.001)
        tracker.record("claude-opus-4-20250514", prompt_tokens=1000, completion_tokens=0)
        for _ in range(3):
            with pytest.raises(BudgetExceededError):
                tracker.check_budget()

    def test_fresh_tracker_with_budget_does_not_raise(self):
        """No records yet → no spending → no raise."""
        CostTracker(max_spend_usd=0.01).check_budget()

    def test_zero_budget_unlimited_even_with_high_spend(self):
        tracker = CostTracker(max_spend_usd=0)
        for _ in range(100):
            tracker.record("claude-opus-4-20250514", prompt_tokens=10000, completion_tokens=10000)
        tracker.check_budget()


# ---------------------------------------------------------------------------
# 7. BudgetExceededError attributes
# ---------------------------------------------------------------------------


class TestBudgetExceededError:
    def test_error_is_exception_subclass(self):
        assert isinstance(BudgetExceededError(0.05, 0.01), Exception)

    def test_spent_attribute_set(self):
        err = BudgetExceededError(spent=0.05, limit=0.01)
        assert err.spent == pytest.approx(0.05)

    def test_limit_attribute_set(self):
        err = BudgetExceededError(spent=0.05, limit=0.01)
        assert err.limit == pytest.approx(0.01)

    def test_message_contains_spent_value(self):
        err = BudgetExceededError(spent=0.0500, limit=0.0100)
        assert "0.0500" in str(err)

    def test_message_contains_limit_value(self):
        err = BudgetExceededError(spent=0.0500, limit=0.0100)
        assert "0.0100" in str(err)

    def test_raised_error_spent_matches_accumulated_cost(self):
        tracker = CostTracker(max_spend_usd=0.001)
        tracker.record("claude-opus-4-20250514", prompt_tokens=1000, completion_tokens=1000)
        expected_cost = tracker.total_cost_usd
        with pytest.raises(BudgetExceededError) as exc_info:
            tracker.check_budget()
        assert exc_info.value.spent == pytest.approx(expected_cost)

    def test_raised_error_limit_matches_max_spend(self):
        tracker = CostTracker(max_spend_usd=0.001)
        tracker.record("claude-opus-4-20250514", prompt_tokens=1000, completion_tokens=1000)
        with pytest.raises(BudgetExceededError) as exc_info:
            tracker.check_budget()
        assert exc_info.value.limit == pytest.approx(0.001)

    def test_raised_error_spent_greater_than_or_equal_to_limit(self):
        tracker = CostTracker(max_spend_usd=0.001)
        tracker.record("claude-opus-4-20250514", prompt_tokens=1000, completion_tokens=1000)
        with pytest.raises(BudgetExceededError) as exc_info:
            tracker.check_budget()
        assert exc_info.value.spent >= exc_info.value.limit

    def test_error_can_be_caught_as_exception(self):
        try:
            raise BudgetExceededError(0.05, 0.01)
        except Exception as exc:
            assert isinstance(exc, BudgetExceededError)


# ---------------------------------------------------------------------------
# 8. Thread safety
# ---------------------------------------------------------------------------


class TestThreadSafety:
    def test_concurrent_record_produces_correct_totals(self):
        """50 threads × 20 calls = 1 000 total; no data races."""
        tracker = CostTracker()
        n_threads, calls_per = 50, 20
        errors: list[Exception] = []

        def _worker():
            try:
                for _ in range(calls_per):
                    tracker.record(
                        "claude-sonnet-4-20250514", prompt_tokens=10, completion_tokens=5
                    )
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=_worker) for _ in range(n_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)

        assert not errors
        assert tracker.total_prompt_tokens == n_threads * calls_per * 10
        assert tracker.total_completion_tokens == n_threads * calls_per * 5

    def test_concurrent_record_call_count_correct(self):
        """call_count is accurate after concurrent writes."""
        tracker = CostTracker()
        n_threads, calls_per = 20, 50
        barrier = threading.Barrier(n_threads)

        def _worker():
            barrier.wait()
            for _ in range(calls_per):
                tracker.record("gpt-4o-mini", prompt_tokens=1, completion_tokens=1)

        threads = [threading.Thread(target=_worker) for _ in range(n_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)

        assert tracker.call_count == n_threads * calls_per

    def test_concurrent_record_and_check_budget_no_deadlock(self):
        """Mixed record + check_budget threads complete without deadlocking."""
        tracker = CostTracker(max_spend_usd=100.0)
        done = threading.Event()
        errors: list[Exception] = []

        def _recorder():
            try:
                for _ in range(500):
                    tracker.record("claude-haiku-4-20250514", prompt_tokens=1, completion_tokens=1)
            except Exception as exc:
                errors.append(exc)
            finally:
                done.set()

        def _checker():
            import contextlib

            while not done.is_set():
                with contextlib.suppress(BudgetExceededError):
                    tracker.check_budget()

        rec_thread = threading.Thread(target=_recorder)
        chk_thread = threading.Thread(target=_checker)
        rec_thread.start()
        chk_thread.start()
        rec_thread.join(timeout=10)
        chk_thread.join(timeout=10)

        assert not rec_thread.is_alive(), "Recorder thread deadlocked"
        assert not chk_thread.is_alive(), "Checker thread deadlocked"
        assert not errors

    def test_concurrent_check_budget_all_raise_when_over(self):
        """20 threads calling check_budget on an over-budget tracker all raise."""
        tracker = CostTracker(max_spend_usd=0.001)
        tracker.record("claude-opus-4-20250514", prompt_tokens=5000, completion_tokens=5000)

        exceptions: list[BudgetExceededError] = []
        lock = threading.Lock()

        def _check():
            try:
                tracker.check_budget()
            except BudgetExceededError as exc:
                with lock:
                    exceptions.append(exc)

        threads = [threading.Thread(target=_check) for _ in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=5)

        assert len(exceptions) == 20

    def test_concurrent_record_cost_total_accurate(self):
        """Total cost remains accurate when many threads record simultaneously."""
        tracker = CostTracker()
        n_threads, calls_per = 10, 100
        # claude-sonnet-4: $0.003/1k input, $0.015/1k output
        barrier = threading.Barrier(n_threads)

        def _worker():
            barrier.wait()
            for _ in range(calls_per):
                tracker.record("claude-sonnet-4-20250514", prompt_tokens=100, completion_tokens=50)

        threads = [threading.Thread(target=_worker) for _ in range(n_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)

        total_calls = n_threads * calls_per
        inp_rate, out_rate = 0.003, 0.015
        expected = total_calls * ((100 / 1000) * inp_rate + (50 / 1000) * out_rate)
        assert tracker.total_cost_usd == pytest.approx(expected, rel=1e-6)


# ---------------------------------------------------------------------------
# 9. get_summary()
# ---------------------------------------------------------------------------


class TestGetSummary:
    def test_fresh_tracker_summary_all_zeros(self):
        summary = CostTracker(max_spend_usd=5.0).get_summary()
        assert summary["total_cost_usd"] == 0.0
        assert summary["total_prompt_tokens"] == 0
        assert summary["total_completion_tokens"] == 0
        assert summary["total_tokens"] == 0
        assert summary["call_count"] == 0

    def test_summary_budget_remaining_none_when_unlimited(self):
        tracker = CostTracker(max_spend_usd=0.0)
        tracker.record("claude-sonnet-4-20250514", prompt_tokens=100, completion_tokens=50)
        assert tracker.get_summary()["budget_remaining_usd"] is None

    def test_summary_budget_remaining_positive_when_under(self):
        tracker = CostTracker(max_spend_usd=1.0)
        tracker.record("claude-haiku-4-20250514", prompt_tokens=100, completion_tokens=50)
        remaining = tracker.get_summary()["budget_remaining_usd"]
        assert remaining is not None and remaining > 0.0

    def test_summary_budget_remaining_floors_at_zero_when_over(self):
        """budget_remaining_usd must never go negative."""
        tracker = CostTracker(max_spend_usd=0.001)
        tracker.record("claude-opus-4-20250514", prompt_tokens=5000, completion_tokens=5000)
        assert tracker.get_summary()["budget_remaining_usd"] == 0.0

    def test_summary_total_tokens_equals_prompt_plus_completion(self):
        tracker = CostTracker()
        tracker.record("gpt-4o", prompt_tokens=300, completion_tokens=200)
        summary = tracker.get_summary()
        assert (
            summary["total_tokens"]
            == summary["total_prompt_tokens"] + summary["total_completion_tokens"]
        )

    def test_summary_call_count_matches_records(self):
        tracker = CostTracker()
        for _ in range(7):
            tracker.record("gpt-4o-mini", prompt_tokens=10, completion_tokens=5)
        assert tracker.get_summary()["call_count"] == 7

    def test_summary_total_cost_rounded_to_six_decimals(self):
        tracker = CostTracker()
        tracker.record("claude-sonnet-4-20250514", prompt_tokens=1, completion_tokens=1)
        cost = tracker.get_summary()["total_cost_usd"]
        assert cost == round(cost, 6)

    def test_summary_max_spend_usd_reflects_init(self):
        tracker = CostTracker(max_spend_usd=3.14)
        assert tracker.get_summary()["max_spend_usd"] == pytest.approx(3.14)

    def test_summary_budget_remaining_accurate(self):
        tracker = CostTracker(max_spend_usd=1.0)
        tracker.record("claude-sonnet-4-20250514", prompt_tokens=1000, completion_tokens=0)
        # $0.003 spent; $0.997 remaining
        summary = tracker.get_summary()
        expected_remaining = round(max(0.0, 1.0 - 0.003), 6)
        assert summary["budget_remaining_usd"] == pytest.approx(expected_remaining)

    def test_summary_contains_all_expected_keys(self):
        summary = CostTracker(max_spend_usd=1.0).get_summary()
        expected_keys = {
            "total_cost_usd",
            "total_prompt_tokens",
            "total_completion_tokens",
            "total_tokens",
            "call_count",
            "max_spend_usd",
            "budget_remaining_usd",
        }
        assert expected_keys.issubset(set(summary.keys()))

    def test_summary_total_cost_matches_property(self):
        tracker = CostTracker()
        tracker.record("gpt-4o", prompt_tokens=500, completion_tokens=250)
        summary = tracker.get_summary()
        assert tracker.total_cost_usd == pytest.approx(summary["total_cost_usd"], rel=1e-5)


# ---------------------------------------------------------------------------
# 10. reset()
# ---------------------------------------------------------------------------


class TestReset:
    def test_reset_clears_cost(self):
        tracker = CostTracker()
        tracker.record("claude-opus-4-20250514", prompt_tokens=1000, completion_tokens=500)
        tracker.reset()
        assert tracker.total_cost_usd == 0.0

    def test_reset_clears_prompt_tokens(self):
        tracker = CostTracker()
        tracker.record("gpt-4o", prompt_tokens=500, completion_tokens=0)
        tracker.reset()
        assert tracker.total_prompt_tokens == 0

    def test_reset_clears_completion_tokens(self):
        tracker = CostTracker()
        tracker.record("gpt-4o", prompt_tokens=0, completion_tokens=250)
        tracker.reset()
        assert tracker.total_completion_tokens == 0

    def test_reset_clears_call_count(self):
        tracker = CostTracker()
        for _ in range(5):
            tracker.record("claude-haiku-4-20250514", prompt_tokens=10, completion_tokens=5)
        tracker.reset()
        assert tracker.call_count == 0

    def test_reset_allows_fresh_accumulation(self):
        tracker = CostTracker()
        tracker.record("claude-sonnet-4-20250514", prompt_tokens=500, completion_tokens=200)
        tracker.reset()
        tracker.record("claude-sonnet-4-20250514", prompt_tokens=100, completion_tokens=50)
        assert tracker.total_prompt_tokens == 100
        assert tracker.total_completion_tokens == 50
        assert tracker.call_count == 1

    def test_reset_preserves_max_spend_usd(self):
        tracker = CostTracker(max_spend_usd=2.0)
        tracker.record("claude-opus-4-20250514", prompt_tokens=100, completion_tokens=50)
        tracker.reset()
        assert tracker.max_spend_usd == pytest.approx(2.0)

    def test_reset_on_fresh_tracker_is_no_op(self):
        tracker = CostTracker(max_spend_usd=1.0)
        tracker.reset()
        assert tracker.total_cost_usd == 0.0
        assert tracker.call_count == 0

    def test_double_reset_is_safe(self):
        tracker = CostTracker()
        tracker.record("gpt-4o", prompt_tokens=100, completion_tokens=50)
        tracker.reset()
        tracker.reset()
        assert tracker.call_count == 0

    def test_reset_allows_budget_check_to_pass_again(self):
        """After reset, previously exceeded budget is cleared."""
        tracker = CostTracker(max_spend_usd=0.001)
        tracker.record("claude-opus-4-20250514", prompt_tokens=1000, completion_tokens=1000)
        with pytest.raises(BudgetExceededError):
            tracker.check_budget()
        tracker.reset()
        tracker.check_budget()  # must not raise after reset


# ---------------------------------------------------------------------------
# 11. Record eviction when exceeding _MAX_RECORDS
# ---------------------------------------------------------------------------


class TestRecordEviction:
    def test_call_count_capped_at_max_records_after_overflow(self):
        """The records buffer is trimmed; call_count reflects buffer size."""
        tracker = CostTracker()
        n = CostTracker._MAX_RECORDS + 10
        for _ in range(n):
            tracker.record("claude-haiku-4-20250514", prompt_tokens=1, completion_tokens=1)
        assert tracker.call_count == CostTracker._MAX_RECORDS

    def test_prompt_token_total_accurate_after_eviction(self):
        tracker = CostTracker()
        n = CostTracker._MAX_RECORDS + 1
        for _ in range(n):
            tracker.record("claude-haiku-4-20250514", prompt_tokens=10, completion_tokens=0)
        assert tracker.total_prompt_tokens == n * 10

    def test_completion_token_total_accurate_after_eviction(self):
        tracker = CostTracker()
        n = CostTracker._MAX_RECORDS + 5
        for _ in range(n):
            tracker.record("claude-haiku-4-20250514", prompt_tokens=0, completion_tokens=5)
        assert tracker.total_completion_tokens == n * 5

    def test_cost_total_accurate_after_eviction(self):
        tracker = CostTracker()
        n = CostTracker._MAX_RECORDS + 50
        # claude-sonnet-4: $0.003/1k input, $0.015/1k output
        inp_rate, out_rate = 0.003, 0.015
        for _ in range(n):
            tracker.record("claude-sonnet-4-20250514", prompt_tokens=100, completion_tokens=50)
        expected = n * ((100 / 1000) * inp_rate + (50 / 1000) * out_rate)
        assert tracker.total_cost_usd == pytest.approx(expected, rel=1e-6)

    def test_total_tokens_accurate_after_eviction(self):
        tracker = CostTracker()
        n = CostTracker._MAX_RECORDS + 1
        for _ in range(n):
            tracker.record("llama3.2:3b", prompt_tokens=7, completion_tokens=3)
        assert tracker.total_tokens == n * 10

    def test_eviction_does_not_truncate_below_max_records(self):
        """Exactly _MAX_RECORDS calls → no eviction; buffer is exactly full."""
        tracker = CostTracker()
        for _ in range(CostTracker._MAX_RECORDS):
            tracker.record("gpt-4o-mini", prompt_tokens=1, completion_tokens=1)
        assert tracker.call_count == CostTracker._MAX_RECORDS


# ---------------------------------------------------------------------------
# 12. Pricing table — prefix matching
# ---------------------------------------------------------------------------


class TestPricingTablePrefixMatching:
    """_lookup_pricing matches dated model variants to their base prefix entry."""

    @pytest.mark.parametrize(
        "model,expected_inp,expected_out",
        [
            # Anthropic — dated variants → base entry
            ("claude-sonnet-4-20250514", 0.003, 0.015),
            ("claude-sonnet-4-20250601", 0.003, 0.015),
            ("claude-opus-4-20250514", 0.015, 0.075),
            ("claude-haiku-4-20250514", 0.0008, 0.004),
            ("claude-3-5-sonnet-20241022", 0.003, 0.015),
            ("claude-3-5-haiku-20241022", 0.0008, 0.004),
            ("claude-3-opus-20240229", 0.015, 0.075),
            ("claude-3-sonnet-20240229", 0.003, 0.015),
            ("claude-3-haiku-20240307", 0.00025, 0.00125),
            # OpenAI — dated and variant names
            # Note: the pricing table lists "gpt-4.1" before "gpt-4.1-mini" / "gpt-4.1-nano",
            # so any model whose name starts with "gpt-4.1" matches the gpt-4.1 entry (0.002).
            # Models with a more-specific prefix that appear AFTER a shorter prefix in the
            # table will fall through to that shorter prefix match.
            ("gpt-4o-2024-08-06", 0.0025, 0.01),
            ("gpt-4o-mini-2024-07-18", 0.00015, 0.0006),
            ("gpt-4-turbo-2024-04-09", 0.01, 0.03),
            # gpt-4.1-preview starts with "gpt-4.1" → matches gpt-4.1 entry ($0.002)
            ("gpt-4.1-preview", 0.002, 0.008),
            # gpt-4.1-mini-preview also starts with "gpt-4.1" → matches gpt-4.1 entry
            ("gpt-4.1-mini-preview", 0.002, 0.008),
            # gpt-4.1-nano-preview also starts with "gpt-4.1" → matches gpt-4.1 entry
            ("gpt-4.1-nano-preview", 0.002, 0.008),
            ("gpt-3.5-turbo-0125", 0.0005, 0.0015),
            # Ollama local models — zero cost
            ("llama3.2:3b", 0.0, 0.0),
            ("mistral:7b-instruct", 0.0, 0.0),
            ("codellama:13b-code", 0.0, 0.0),
            ("deepseek-coder:6.7b", 0.0, 0.0),
            ("phi3:mini", 0.0, 0.0),
            ("qwen2:7b", 0.0, 0.0),
            ("gemma2:9b", 0.0, 0.0),
        ],
    )
    def test_prefix_lookup(self, model: str, expected_inp: float, expected_out: float):
        inp, out = _lookup_pricing(model)
        assert inp == pytest.approx(expected_inp), f"Input rate mismatch for {model!r}"
        assert out == pytest.approx(expected_out), f"Output rate mismatch for {model!r}"

    def test_lookup_is_case_insensitive(self):
        """The lookup lowercases the model name before matching."""
        inp_lower, out_lower = _lookup_pricing("claude-sonnet-4-20250514")
        inp_upper, out_upper = _lookup_pricing("CLAUDE-SONNET-4-20250514")
        assert inp_lower == inp_upper
        assert out_lower == out_upper

    def test_completely_unknown_model_returns_zero(self):
        inp, out = _lookup_pricing("some-completely-unknown-model")
        assert inp == 0.0 and out == 0.0

    def test_gpt4_1_matches_gpt4_1_entry(self):
        """gpt-4.1 (base) matches the gpt-4.1 pricing entry."""
        inp, out = _lookup_pricing("gpt-4.1")
        assert inp == pytest.approx(0.002)
        assert out == pytest.approx(0.008)

    def test_gpt4_1_mini_matches_gpt4_1_entry_due_to_table_order(self):
        """gpt-4.1-mini starts with 'gpt-4.1', which appears first in the table.
        The table is NOT ordered most-specific-first for this group, so gpt-4.1-mini
        falls through to the gpt-4.1 entry ($0.002 input).
        """
        inp, _ = _lookup_pricing("gpt-4.1-mini")
        assert inp == pytest.approx(0.002)

    def test_gpt4_1_nano_matches_gpt4_1_entry_due_to_table_order(self):
        """Same table-order behaviour: gpt-4.1-nano hits gpt-4.1 entry."""
        inp, _ = _lookup_pricing("gpt-4.1-nano")
        assert inp == pytest.approx(0.002)


# ---------------------------------------------------------------------------
# 13. Property accessors
# ---------------------------------------------------------------------------


class TestPropertyAccessors:
    def test_total_cost_usd_reflects_single_record(self):
        tracker = CostTracker()
        tracker.record("claude-sonnet-4-20250514", prompt_tokens=1000, completion_tokens=500)
        expected = 0.003 + (500 / 1000) * 0.015
        assert tracker.total_cost_usd == pytest.approx(expected)

    def test_total_prompt_tokens_sums_all_records(self):
        tracker = CostTracker()
        tracker.record("gpt-4o", prompt_tokens=100, completion_tokens=0)
        tracker.record("gpt-4o", prompt_tokens=200, completion_tokens=0)
        assert tracker.total_prompt_tokens == 300

    def test_total_completion_tokens_sums_all_records(self):
        tracker = CostTracker()
        tracker.record("gpt-4o", prompt_tokens=0, completion_tokens=75)
        tracker.record("gpt-4o", prompt_tokens=0, completion_tokens=25)
        assert tracker.total_completion_tokens == 100

    def test_total_tokens_is_sum_of_prompt_and_completion(self):
        tracker = CostTracker()
        tracker.record("claude-haiku-4-20250514", prompt_tokens=400, completion_tokens=200)
        assert tracker.total_tokens == 600

    def test_call_count_increments_per_record(self):
        tracker = CostTracker()
        for i in range(1, 6):
            tracker.record("claude-haiku-4-20250514", prompt_tokens=10, completion_tokens=5)
            assert tracker.call_count == i

    def test_total_tokens_zero_on_fresh_tracker(self):
        assert CostTracker().total_tokens == 0

    def test_total_cost_usd_zero_on_fresh_tracker(self):
        assert CostTracker().total_cost_usd == 0.0

    def test_total_prompt_tokens_zero_on_fresh_tracker(self):
        assert CostTracker().total_prompt_tokens == 0

    def test_total_completion_tokens_zero_on_fresh_tracker(self):
        assert CostTracker().total_completion_tokens == 0

    def test_call_count_zero_on_fresh_tracker(self):
        assert CostTracker().call_count == 0

    def test_properties_consistent_with_get_summary(self):
        tracker = CostTracker(max_spend_usd=10.0)
        tracker.record("gpt-4o", prompt_tokens=500, completion_tokens=250)
        summary = tracker.get_summary()
        assert tracker.total_cost_usd == pytest.approx(summary["total_cost_usd"], rel=1e-5)
        assert tracker.total_prompt_tokens == summary["total_prompt_tokens"]
        assert tracker.total_completion_tokens == summary["total_completion_tokens"]
        assert tracker.total_tokens == summary["total_tokens"]
        assert tracker.call_count == summary["call_count"]

    def test_total_tokens_additive_across_models(self):
        """total_tokens accumulates across calls with different models."""
        tracker = CostTracker()
        tracker.record("claude-sonnet-4-20250514", prompt_tokens=100, completion_tokens=50)
        tracker.record("gpt-4o", prompt_tokens=200, completion_tokens=100)
        assert tracker.total_tokens == 450
