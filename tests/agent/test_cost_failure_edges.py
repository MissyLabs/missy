"""Edge case tests for CostTracker and FailureTracker.


Covers:
- CostTracker: pricing lookup, unknown models, record eviction, budget enforcement,
  concurrent recording, reset, summary, record_from_response edge cases
- FailureTracker: threshold validation, consecutive vs total, strategy prompt,
  reset, stats, success resets consecutive, multi-tool tracking
"""

from __future__ import annotations

import threading
from unittest.mock import MagicMock

import pytest

# ---------------------------------------------------------------------------
# CostTracker tests
# ---------------------------------------------------------------------------


class TestCostTrackerEdgeCases:
    """Edge cases in CostTracker."""

    def test_unknown_model_zero_cost(self):
        """Unknown model names should default to zero cost."""
        from missy.agent.cost_tracker import CostTracker

        tracker = CostTracker()
        rec = tracker.record(model="unknown-model", prompt_tokens=1000, completion_tokens=500)
        assert rec.cost_usd == 0.0

    def test_empty_model_string(self):
        """Empty model string should default to zero cost."""
        from missy.agent.cost_tracker import CostTracker

        tracker = CostTracker()
        rec = tracker.record(model="", prompt_tokens=1000, completion_tokens=500)
        assert rec.cost_usd == 0.0

    def test_claude_sonnet_pricing(self):
        """Claude sonnet model should use correct pricing."""
        from missy.agent.cost_tracker import CostTracker

        tracker = CostTracker()
        rec = tracker.record(model="claude-sonnet-4-20250514", prompt_tokens=1000, completion_tokens=500)
        # $0.003/1k input + $0.015/1k output
        expected = (1000 / 1000) * 0.003 + (500 / 1000) * 0.015
        assert abs(rec.cost_usd - expected) < 1e-8

    def test_gpt4o_pricing(self):
        """GPT-4o model should use correct pricing."""
        from missy.agent.cost_tracker import CostTracker

        tracker = CostTracker()
        rec = tracker.record(model="gpt-4o-2024-05-13", prompt_tokens=1000, completion_tokens=500)
        expected = (1000 / 1000) * 0.0025 + (500 / 1000) * 0.01
        assert abs(rec.cost_usd - expected) < 1e-8

    def test_ollama_model_zero_cost(self):
        """Ollama models should have zero cost."""
        from missy.agent.cost_tracker import CostTracker

        tracker = CostTracker()
        for model in ["llama3.1", "mistral-7b", "deepseek-coder"]:
            rec = tracker.record(model=model, prompt_tokens=10000, completion_tokens=5000)
            assert rec.cost_usd == 0.0

    def test_zero_tokens(self):
        """Zero tokens should produce zero cost."""
        from missy.agent.cost_tracker import CostTracker

        tracker = CostTracker()
        rec = tracker.record(model="claude-sonnet-4", prompt_tokens=0, completion_tokens=0)
        assert rec.cost_usd == 0.0

    def test_negative_tokens(self):
        """Negative tokens should produce negative cost (no validation)."""
        from missy.agent.cost_tracker import CostTracker

        tracker = CostTracker()
        rec = tracker.record(model="claude-sonnet-4", prompt_tokens=-100, completion_tokens=0)
        assert rec.cost_usd < 0

    def test_budget_enforcement_no_limit(self):
        """No budget limit (0) should never raise."""
        from missy.agent.cost_tracker import CostTracker

        tracker = CostTracker(max_spend_usd=0)
        for _ in range(100):
            tracker.record(model="claude-opus-4", prompt_tokens=10000, completion_tokens=5000)
        tracker.check_budget()  # Should not raise

    def test_budget_enforcement_triggers(self):
        """Budget exceeded should raise BudgetExceededError."""
        from missy.agent.cost_tracker import BudgetExceededError, CostTracker

        tracker = CostTracker(max_spend_usd=0.01)
        tracker.record(model="claude-opus-4", prompt_tokens=10000, completion_tokens=5000)
        with pytest.raises(BudgetExceededError) as exc_info:
            tracker.check_budget()
        assert exc_info.value.limit == 0.01
        assert exc_info.value.spent > 0

    def test_budget_exactly_at_limit(self):
        """Cost exactly at limit should raise."""
        from missy.agent.cost_tracker import BudgetExceededError, CostTracker

        tracker = CostTracker(max_spend_usd=0.003)
        # Claude sonnet: $0.003/1k prompt tokens
        tracker.record(model="claude-sonnet-4", prompt_tokens=1000, completion_tokens=0)
        with pytest.raises(BudgetExceededError):
            tracker.check_budget()

    def test_budget_none_equivalent_to_zero(self):
        """max_spend_usd=None should mean unlimited."""
        from missy.agent.cost_tracker import CostTracker

        tracker = CostTracker(max_spend_usd=None)
        tracker.record(model="claude-opus-4", prompt_tokens=100000, completion_tokens=50000)
        tracker.check_budget()  # Should not raise

    def test_total_cost_accumulates(self):
        """Total cost should accumulate across multiple records."""
        from missy.agent.cost_tracker import CostTracker

        tracker = CostTracker()
        tracker.record(model="claude-sonnet-4", prompt_tokens=1000, completion_tokens=0)
        tracker.record(model="claude-sonnet-4", prompt_tokens=1000, completion_tokens=0)
        assert tracker.total_cost_usd == pytest.approx(0.006, rel=1e-6)

    def test_total_tokens(self):
        """total_tokens should sum prompt + completion."""
        from missy.agent.cost_tracker import CostTracker

        tracker = CostTracker()
        tracker.record(model="test", prompt_tokens=100, completion_tokens=50)
        assert tracker.total_prompt_tokens == 100
        assert tracker.total_completion_tokens == 50
        assert tracker.total_tokens == 150

    def test_call_count(self):
        from missy.agent.cost_tracker import CostTracker

        tracker = CostTracker()
        assert tracker.call_count == 0
        tracker.record(model="test", prompt_tokens=1)
        tracker.record(model="test", prompt_tokens=1)
        assert tracker.call_count == 2

    def test_record_eviction(self):
        """Records beyond MAX_RECORDS should be evicted but totals preserved."""
        from missy.agent.cost_tracker import CostTracker

        tracker = CostTracker()
        tracker._MAX_RECORDS = 5  # Reduce for testing
        for _i in range(10):
            tracker.record(model="test", prompt_tokens=100, completion_tokens=50)
        assert tracker.call_count == 5  # Only 5 retained
        assert tracker.total_prompt_tokens == 1000  # All 10 counted
        assert tracker.total_completion_tokens == 500

    def test_get_summary_with_budget(self):
        from missy.agent.cost_tracker import CostTracker

        tracker = CostTracker(max_spend_usd=1.0)
        tracker.record(model="claude-sonnet-4", prompt_tokens=1000, completion_tokens=500)
        summary = tracker.get_summary()
        assert summary["max_spend_usd"] == 1.0
        assert summary["budget_remaining_usd"] is not None
        assert summary["budget_remaining_usd"] > 0
        assert summary["total_tokens"] == 1500

    def test_get_summary_without_budget(self):
        from missy.agent.cost_tracker import CostTracker

        tracker = CostTracker(max_spend_usd=0)
        summary = tracker.get_summary()
        assert summary["budget_remaining_usd"] is None

    def test_reset(self):
        from missy.agent.cost_tracker import CostTracker

        tracker = CostTracker()
        tracker.record(model="test", prompt_tokens=100, completion_tokens=50)
        tracker.reset()
        assert tracker.total_cost_usd == 0.0
        assert tracker.total_tokens == 0
        assert tracker.call_count == 0

    def test_record_from_response_valid(self):
        from missy.agent.cost_tracker import CostTracker

        tracker = CostTracker()
        response = MagicMock()
        response.model = "claude-sonnet-4"
        response.usage = {"prompt_tokens": 100, "completion_tokens": 50}
        rec = tracker.record_from_response(response)
        assert rec is not None
        assert rec.prompt_tokens == 100

    def test_record_from_response_no_usage(self):
        from missy.agent.cost_tracker import CostTracker

        tracker = CostTracker()
        response = MagicMock()
        response.model = ""
        response.usage = None
        rec = tracker.record_from_response(response)
        assert rec is None

    def test_record_from_response_no_model_but_tokens(self):
        from missy.agent.cost_tracker import CostTracker

        tracker = CostTracker()
        response = MagicMock()
        response.model = ""
        response.usage = {"prompt_tokens": 100, "completion_tokens": 50}
        rec = tracker.record_from_response(response)
        assert rec is not None

    def test_concurrent_recording(self):
        """Concurrent recording should be thread-safe."""
        from missy.agent.cost_tracker import CostTracker

        tracker = CostTracker()

        def record_many():
            for _ in range(50):
                tracker.record(model="claude-sonnet-4", prompt_tokens=100, completion_tokens=50)

        threads = [threading.Thread(target=record_many) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)

        assert tracker.total_prompt_tokens == 25000
        assert tracker.total_completion_tokens == 12500

    def test_pricing_case_insensitive(self):
        """Model name lookup should be case-insensitive."""
        from missy.agent.cost_tracker import _lookup_pricing

        inp1, out1 = _lookup_pricing("Claude-Sonnet-4")
        inp2, out2 = _lookup_pricing("claude-sonnet-4")
        assert inp1 == inp2
        assert out1 == out2

    def test_budget_exceeded_error_attrs(self):
        from missy.agent.cost_tracker import BudgetExceededError

        err = BudgetExceededError(0.55, 0.50)
        assert err.spent == 0.55
        assert err.limit == 0.50
        assert "0.55" in str(err)
        assert "0.50" in str(err)


# ---------------------------------------------------------------------------
# FailureTracker tests
# ---------------------------------------------------------------------------


class TestFailureTrackerEdgeCases:
    """Edge cases in FailureTracker."""

    def test_threshold_zero_raises(self):
        from missy.agent.failure_tracker import FailureTracker

        with pytest.raises(ValueError, match="threshold must be >= 1"):
            FailureTracker(threshold=0)

    def test_threshold_negative_raises(self):
        from missy.agent.failure_tracker import FailureTracker

        with pytest.raises(ValueError, match="threshold must be >= 1"):
            FailureTracker(threshold=-5)

    def test_threshold_one(self):
        """Threshold of 1 should trigger on first failure."""
        from missy.agent.failure_tracker import FailureTracker

        tracker = FailureTracker(threshold=1)
        result = tracker.record_failure("shell_exec", "error")
        assert result is True

    def test_below_threshold(self):
        from missy.agent.failure_tracker import FailureTracker

        tracker = FailureTracker(threshold=3)
        assert tracker.record_failure("tool", "err") is False
        assert tracker.record_failure("tool", "err") is False
        assert tracker.record_failure("tool", "err") is True  # Reaches threshold

    def test_above_threshold_keeps_returning_true(self):
        from missy.agent.failure_tracker import FailureTracker

        tracker = FailureTracker(threshold=2)
        tracker.record_failure("tool", "err")
        tracker.record_failure("tool", "err")
        assert tracker.record_failure("tool", "err") is True  # Above threshold

    def test_success_resets_consecutive(self):
        from missy.agent.failure_tracker import FailureTracker

        tracker = FailureTracker(threshold=3)
        tracker.record_failure("tool", "err")
        tracker.record_failure("tool", "err")
        tracker.record_success("tool")
        # After success, consecutive resets
        assert tracker.record_failure("tool", "err") is False

    def test_success_preserves_total(self):
        from missy.agent.failure_tracker import FailureTracker

        tracker = FailureTracker(threshold=3)
        tracker.record_failure("tool", "err")
        tracker.record_failure("tool", "err")
        tracker.record_success("tool")
        stats = tracker.get_stats()
        assert stats["tool"]["total_failures"] == 2
        assert stats["tool"]["failures"] == 0

    def test_should_inject_strategy_unknown_tool(self):
        from missy.agent.failure_tracker import FailureTracker

        tracker = FailureTracker()
        assert tracker.should_inject_strategy("nonexistent") is False

    def test_should_inject_strategy_below_threshold(self):
        from missy.agent.failure_tracker import FailureTracker

        tracker = FailureTracker(threshold=3)
        tracker.record_failure("tool", "err")
        assert tracker.should_inject_strategy("tool") is False

    def test_should_inject_strategy_at_threshold(self):
        from missy.agent.failure_tracker import FailureTracker

        tracker = FailureTracker(threshold=2)
        tracker.record_failure("tool", "err")
        tracker.record_failure("tool", "err")
        assert tracker.should_inject_strategy("tool") is True

    def test_get_strategy_prompt_content(self):
        from missy.agent.failure_tracker import FailureTracker

        tracker = FailureTracker(threshold=2)
        tracker.record_failure("shell_exec", "permission denied")
        tracker.record_failure("shell_exec", "permission denied")
        prompt = tracker.get_strategy_prompt("shell_exec", "permission denied")
        assert "shell_exec" in prompt
        assert "permission denied" in prompt
        assert "2 times" in prompt
        assert "alternative" in prompt.lower()

    def test_get_strategy_prompt_unknown_tool(self):
        """Strategy prompt for unknown tool should still work."""
        from missy.agent.failure_tracker import FailureTracker

        tracker = FailureTracker(threshold=3)
        prompt = tracker.get_strategy_prompt("unknown_tool", "some error")
        assert "unknown_tool" in prompt
        assert "3 times" in prompt  # Falls back to threshold

    def test_multi_tool_tracking(self):
        from missy.agent.failure_tracker import FailureTracker

        tracker = FailureTracker(threshold=2)
        tracker.record_failure("tool_a", "err")
        tracker.record_failure("tool_b", "err")
        tracker.record_failure("tool_b", "err")
        assert tracker.should_inject_strategy("tool_a") is False
        assert tracker.should_inject_strategy("tool_b") is True

    def test_reset_specific_tool(self):
        from missy.agent.failure_tracker import FailureTracker

        tracker = FailureTracker()
        tracker.record_failure("tool_a", "err")
        tracker.record_failure("tool_b", "err")
        tracker.reset("tool_a")
        stats = tracker.get_stats()
        assert "tool_a" not in stats
        assert "tool_b" in stats

    def test_reset_nonexistent_tool(self):
        from missy.agent.failure_tracker import FailureTracker

        tracker = FailureTracker()
        tracker.reset("nonexistent")  # Should not raise

    def test_reset_all(self):
        from missy.agent.failure_tracker import FailureTracker

        tracker = FailureTracker()
        tracker.record_failure("tool_a", "err")
        tracker.record_failure("tool_b", "err")
        tracker.reset_all()
        assert tracker.get_stats() == {}

    def test_get_stats_empty(self):
        from missy.agent.failure_tracker import FailureTracker

        tracker = FailureTracker()
        assert tracker.get_stats() == {}

    def test_get_stats_after_success_only(self):
        from missy.agent.failure_tracker import FailureTracker

        tracker = FailureTracker()
        tracker.record_success("tool_a")
        stats = tracker.get_stats()
        assert stats["tool_a"]["failures"] == 0
        assert stats["tool_a"]["total_failures"] == 0

    def test_record_success_for_new_tool(self):
        """Recording success for a never-seen tool should create state."""
        from missy.agent.failure_tracker import FailureTracker

        tracker = FailureTracker()
        tracker.record_success("new_tool")
        stats = tracker.get_stats()
        assert "new_tool" in stats
