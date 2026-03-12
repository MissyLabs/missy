"""Tests for missy.agent.cost_tracker."""

import threading

import pytest

from missy.agent.cost_tracker import (
    BudgetExceededError,
    CostTracker,
    UsageRecord,
    _lookup_pricing,
)


class TestLookupPricing:
    """Tests for the pricing table lookup function."""

    def test_anthropic_sonnet(self):
        inp, out = _lookup_pricing("claude-sonnet-4-20250514")
        assert inp == 0.003
        assert out == 0.015

    def test_anthropic_opus(self):
        inp, out = _lookup_pricing("claude-opus-4-20250514")
        assert inp == 0.015
        assert out == 0.075

    def test_anthropic_haiku(self):
        inp, out = _lookup_pricing("claude-haiku-4-20250514")
        assert inp == 0.0008
        assert out == 0.004

    def test_openai_gpt4o(self):
        inp, out = _lookup_pricing("gpt-4o-2024-05-13")
        assert inp == 0.0025
        assert out == 0.01

    def test_openai_gpt4o_mini(self):
        inp, out = _lookup_pricing("gpt-4o-mini")
        assert inp == 0.00015
        assert out == 0.0006

    def test_local_model_zero_cost(self):
        inp, out = _lookup_pricing("llama3.1:70b")
        assert inp == 0.0
        assert out == 0.0

    def test_unknown_model_zero_cost(self):
        inp, out = _lookup_pricing("some-unknown-model-xyz")
        assert inp == 0.0
        assert out == 0.0


class TestCostTrackerRecord:
    """Tests for CostTracker.record()."""

    def test_record_returns_usage_record(self):
        tracker = CostTracker()
        rec = tracker.record("claude-sonnet-4-20250514", prompt_tokens=1000, completion_tokens=500)
        assert isinstance(rec, UsageRecord)
        assert rec.model == "claude-sonnet-4-20250514"
        assert rec.prompt_tokens == 1000
        assert rec.completion_tokens == 500

    def test_record_calculates_cost(self):
        tracker = CostTracker()
        rec = tracker.record("claude-sonnet-4-20250514", prompt_tokens=1000, completion_tokens=1000)
        # 1000/1000 * 0.003 + 1000/1000 * 0.015 = 0.018
        assert abs(rec.cost_usd - 0.018) < 1e-9

    def test_accumulates_across_calls(self):
        tracker = CostTracker()
        tracker.record("claude-sonnet-4-20250514", prompt_tokens=500, completion_tokens=200)
        tracker.record("claude-sonnet-4-20250514", prompt_tokens=300, completion_tokens=100)
        assert tracker.total_prompt_tokens == 800
        assert tracker.total_completion_tokens == 300
        assert tracker.total_tokens == 1100
        assert tracker.call_count == 2

    def test_zero_tokens_zero_cost(self):
        tracker = CostTracker()
        rec = tracker.record("claude-sonnet-4-20250514", prompt_tokens=0, completion_tokens=0)
        assert rec.cost_usd == 0.0
        assert tracker.total_cost_usd == 0.0


class TestCostTrackerRecordFromResponse:
    """Tests for CostTracker.record_from_response()."""

    def test_records_from_mock_response(self):
        tracker = CostTracker()

        class FakeResponse:
            model = "claude-sonnet-4-20250514"
            usage = {"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150}

        rec = tracker.record_from_response(FakeResponse())
        assert rec is not None
        assert rec.prompt_tokens == 100
        assert rec.completion_tokens == 50

    def test_returns_none_for_empty_response(self):
        tracker = CostTracker()

        class EmptyResponse:
            model = ""
            usage = {}

        assert tracker.record_from_response(EmptyResponse()) is None

    def test_handles_missing_usage(self):
        tracker = CostTracker()

        class NoUsage:
            model = "claude-sonnet-4-20250514"
            usage = None

        rec = tracker.record_from_response(NoUsage())
        assert rec is not None
        assert rec.prompt_tokens == 0


class TestBudgetEnforcement:
    """Tests for max_spend_usd enforcement."""

    def test_no_limit_does_not_raise(self):
        tracker = CostTracker(max_spend_usd=0)
        tracker.record("claude-opus-4-20250514", prompt_tokens=100000, completion_tokens=100000)
        tracker.check_budget()  # should not raise

    def test_under_budget_does_not_raise(self):
        tracker = CostTracker(max_spend_usd=1.0)
        # Small usage
        tracker.record("claude-sonnet-4-20250514", prompt_tokens=100, completion_tokens=50)
        tracker.check_budget()  # should not raise

    def test_over_budget_raises(self):
        tracker = CostTracker(max_spend_usd=0.001)
        # This should exceed $0.001
        tracker.record("claude-opus-4-20250514", prompt_tokens=1000, completion_tokens=1000)
        with pytest.raises(BudgetExceededError) as exc_info:
            tracker.check_budget()
        assert exc_info.value.spent > 0.001
        assert exc_info.value.limit == 0.001


class TestCostTrackerSummary:
    """Tests for CostTracker.get_summary()."""

    def test_summary_keys(self):
        tracker = CostTracker(max_spend_usd=5.0)
        tracker.record("claude-sonnet-4-20250514", prompt_tokens=100, completion_tokens=50)
        summary = tracker.get_summary()
        assert "total_cost_usd" in summary
        assert "total_prompt_tokens" in summary
        assert "total_completion_tokens" in summary
        assert "total_tokens" in summary
        assert "call_count" in summary
        assert "max_spend_usd" in summary
        assert "budget_remaining_usd" in summary

    def test_summary_budget_remaining(self):
        tracker = CostTracker(max_spend_usd=1.0)
        summary = tracker.get_summary()
        assert summary["budget_remaining_usd"] == 1.0

    def test_summary_no_budget_remaining_is_none(self):
        tracker = CostTracker(max_spend_usd=0)
        summary = tracker.get_summary()
        assert summary["budget_remaining_usd"] is None


class TestCostTrackerReset:
    """Tests for CostTracker.reset()."""

    def test_reset_clears_all(self):
        tracker = CostTracker()
        tracker.record("claude-sonnet-4-20250514", prompt_tokens=1000, completion_tokens=500)
        assert tracker.call_count == 1
        tracker.reset()
        assert tracker.call_count == 0
        assert tracker.total_cost_usd == 0.0
        assert tracker.total_tokens == 0


class TestCostTrackerThreadSafety:
    """Verify thread-safe accumulation."""

    def test_concurrent_records(self):
        tracker = CostTracker()
        barrier = threading.Barrier(10)

        def worker():
            barrier.wait()
            for _ in range(100):
                tracker.record("claude-sonnet-4-20250514", prompt_tokens=10, completion_tokens=5)

        threads = [threading.Thread(target=worker) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert tracker.call_count == 1000
        assert tracker.total_prompt_tokens == 10000
        assert tracker.total_completion_tokens == 5000
