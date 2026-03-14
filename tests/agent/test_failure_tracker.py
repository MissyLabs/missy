"""Tests for missy.agent.failure_tracker.FailureTracker."""

from __future__ import annotations

import pytest

from missy.agent.failure_tracker import FailureTracker

# ---------------------------------------------------------------------------
# Construction / configuration
# ---------------------------------------------------------------------------


class TestFailureTrackerInit:
    def test_default_threshold_is_three(self):
        tracker = FailureTracker()
        assert tracker.threshold == 3

    def test_custom_threshold(self):
        tracker = FailureTracker(threshold=5)
        assert tracker.threshold == 5

    def test_zero_threshold_raises(self):
        with pytest.raises(ValueError, match="threshold must be >= 1"):
            FailureTracker(threshold=0)

    def test_negative_threshold_raises(self):
        with pytest.raises(ValueError, match="threshold must be >= 1"):
            FailureTracker(threshold=-1)

    def test_initial_state_is_empty(self):
        tracker = FailureTracker()
        assert tracker.get_stats() == {}


# ---------------------------------------------------------------------------
# record_failure
# ---------------------------------------------------------------------------


class TestRecordFailure:
    def test_returns_false_before_threshold(self):
        tracker = FailureTracker(threshold=3)
        assert tracker.record_failure("tool_a", "err") is False
        assert tracker.record_failure("tool_a", "err") is False

    def test_returns_true_at_threshold(self):
        tracker = FailureTracker(threshold=3)
        tracker.record_failure("tool_a", "err")
        tracker.record_failure("tool_a", "err")
        result = tracker.record_failure("tool_a", "err")
        assert result is True

    def test_returns_true_beyond_threshold(self):
        tracker = FailureTracker(threshold=2)
        tracker.record_failure("tool_a", "err")
        tracker.record_failure("tool_a", "err")
        # Still True on the 3rd+ call
        assert tracker.record_failure("tool_a", "err") is True

    def test_separate_counters_per_tool(self):
        tracker = FailureTracker(threshold=2)
        assert tracker.record_failure("tool_a", "err") is False
        assert tracker.record_failure("tool_b", "err") is False
        assert tracker.record_failure("tool_a", "err") is True
        # tool_b only has 1 failure still
        assert tracker.should_inject_strategy("tool_b") is False

    def test_increments_total_failures(self):
        tracker = FailureTracker(threshold=3)
        tracker.record_failure("tool_a", "err1")
        tracker.record_success("tool_a")  # resets consecutive but not total
        tracker.record_failure("tool_a", "err2")
        stats = tracker.get_stats()
        assert stats["tool_a"]["total_failures"] == 2
        assert stats["tool_a"]["failures"] == 1

    def test_threshold_one_fires_immediately(self):
        tracker = FailureTracker(threshold=1)
        assert tracker.record_failure("tool_a", "err") is True


# ---------------------------------------------------------------------------
# record_success
# ---------------------------------------------------------------------------


class TestRecordSuccess:
    def test_resets_consecutive_failures(self):
        tracker = FailureTracker(threshold=3)
        tracker.record_failure("tool_a", "err")
        tracker.record_failure("tool_a", "err")
        tracker.record_success("tool_a")
        assert tracker.should_inject_strategy("tool_a") is False

    def test_does_not_affect_other_tools(self):
        tracker = FailureTracker(threshold=2)
        tracker.record_failure("tool_a", "err")
        tracker.record_failure("tool_a", "err")
        tracker.record_success("tool_b")  # tool_b was never seen; shouldn't error
        assert tracker.should_inject_strategy("tool_a") is True

    def test_creates_entry_for_unseen_tool(self):
        tracker = FailureTracker()
        tracker.record_success("new_tool")
        stats = tracker.get_stats()
        assert "new_tool" in stats
        assert stats["new_tool"]["failures"] == 0
        assert stats["new_tool"]["total_failures"] == 0

    def test_consecutive_resets_allow_fresh_count(self):
        tracker = FailureTracker(threshold=2)
        tracker.record_failure("t", "e")
        tracker.record_failure("t", "e")  # threshold hit
        tracker.record_success("t")       # reset
        # After success, needs another 2 failures to trigger again
        assert tracker.record_failure("t", "e") is False
        assert tracker.record_failure("t", "e") is True


# ---------------------------------------------------------------------------
# should_inject_strategy
# ---------------------------------------------------------------------------


class TestShouldInjectStrategy:
    def test_false_for_unknown_tool(self):
        tracker = FailureTracker()
        assert tracker.should_inject_strategy("never_seen") is False

    def test_false_below_threshold(self):
        tracker = FailureTracker(threshold=3)
        tracker.record_failure("t", "e")
        tracker.record_failure("t", "e")
        assert tracker.should_inject_strategy("t") is False

    def test_true_at_threshold(self):
        tracker = FailureTracker(threshold=3)
        for _ in range(3):
            tracker.record_failure("t", "e")
        assert tracker.should_inject_strategy("t") is True

    def test_true_above_threshold(self):
        tracker = FailureTracker(threshold=2)
        for _ in range(5):
            tracker.record_failure("t", "e")
        assert tracker.should_inject_strategy("t") is True


# ---------------------------------------------------------------------------
# get_strategy_prompt
# ---------------------------------------------------------------------------


class TestGetStrategyPrompt:
    def test_prompt_contains_tool_name(self):
        tracker = FailureTracker(threshold=2)
        tracker.record_failure("shell_exec", "permission denied")
        tracker.record_failure("shell_exec", "permission denied")
        prompt = tracker.get_strategy_prompt("shell_exec", "permission denied")
        assert "shell_exec" in prompt

    def test_prompt_contains_last_error(self):
        tracker = FailureTracker(threshold=2)
        tracker.record_failure("shell_exec", "err")
        tracker.record_failure("shell_exec", "err")
        prompt = tracker.get_strategy_prompt("shell_exec", "file not found")
        assert "file not found" in prompt

    def test_prompt_contains_failure_count(self):
        tracker = FailureTracker(threshold=3)
        for _ in range(3):
            tracker.record_failure("t", "e")
        prompt = tracker.get_strategy_prompt("t", "e")
        assert "3" in prompt

    def test_prompt_instructs_three_alternatives(self):
        tracker = FailureTracker(threshold=2)
        tracker.record_failure("t", "e")
        tracker.record_failure("t", "e")
        prompt = tracker.get_strategy_prompt("t", "e")
        assert "3 alternative" in prompt.lower() or "three alternative" in prompt.lower() or "3" in prompt

    def test_prompt_instructs_not_to_retry_tool(self):
        tracker = FailureTracker(threshold=2)
        tracker.record_failure("bad_tool", "e")
        tracker.record_failure("bad_tool", "e")
        prompt = tracker.get_strategy_prompt("bad_tool", "e")
        # The prompt should discourage retrying the same tool
        assert "bad_tool" in prompt
        assert "alternative" in prompt.lower() or "do not" in prompt.lower()

    def test_prompt_for_unseen_tool_uses_threshold_as_count(self):
        tracker = FailureTracker(threshold=5)
        # Should not raise even if tool was never recorded
        prompt = tracker.get_strategy_prompt("mystery_tool", "error")
        assert "mystery_tool" in prompt
        assert "5" in prompt


# ---------------------------------------------------------------------------
# reset / reset_all
# ---------------------------------------------------------------------------


class TestReset:
    def test_reset_clears_specific_tool(self):
        tracker = FailureTracker(threshold=2)
        tracker.record_failure("tool_a", "e")
        tracker.record_failure("tool_a", "e")
        tracker.reset("tool_a")
        assert tracker.should_inject_strategy("tool_a") is False
        assert "tool_a" not in tracker.get_stats()

    def test_reset_does_not_affect_other_tools(self):
        tracker = FailureTracker(threshold=2)
        tracker.record_failure("tool_a", "e")
        tracker.record_failure("tool_a", "e")
        tracker.record_failure("tool_b", "e")
        tracker.reset("tool_a")
        stats = tracker.get_stats()
        assert "tool_b" in stats
        assert stats["tool_b"]["failures"] == 1

    def test_reset_unknown_tool_is_noop(self):
        tracker = FailureTracker()
        tracker.reset("never_seen")  # must not raise

    def test_reset_all_clears_all_state(self):
        tracker = FailureTracker(threshold=2)
        for name in ("a", "b", "c"):
            tracker.record_failure(name, "e")
            tracker.record_failure(name, "e")
        tracker.reset_all()
        assert tracker.get_stats() == {}

    def test_reset_all_on_empty_tracker_is_noop(self):
        tracker = FailureTracker()
        tracker.reset_all()
        assert tracker.get_stats() == {}


# ---------------------------------------------------------------------------
# get_stats
# ---------------------------------------------------------------------------


class TestGetStats:
    def test_returns_empty_for_new_tracker(self):
        assert FailureTracker().get_stats() == {}

    def test_tracks_consecutive_and_total_separately(self):
        tracker = FailureTracker(threshold=10)
        tracker.record_failure("t", "e")   # consecutive=1, total=1
        tracker.record_failure("t", "e")   # consecutive=2, total=2
        tracker.record_success("t")         # consecutive=0, total=2
        tracker.record_failure("t", "e")   # consecutive=1, total=3
        stats = tracker.get_stats()
        assert stats["t"]["failures"] == 1
        assert stats["t"]["total_failures"] == 3

    def test_stats_keys(self):
        tracker = FailureTracker()
        tracker.record_failure("my_tool", "err")
        stats = tracker.get_stats()
        assert set(stats["my_tool"].keys()) == {"failures", "total_failures"}

    def test_multiple_tools_in_stats(self):
        tracker = FailureTracker()
        tracker.record_failure("a", "e")
        tracker.record_failure("b", "e")
        tracker.record_failure("b", "e")
        stats = tracker.get_stats()
        assert stats["a"]["failures"] == 1
        assert stats["b"]["failures"] == 2
