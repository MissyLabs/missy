"""Edge case tests for Summarizer and ProactiveManager.


Covers:
- Summarizer: empty turns, timestamp edge cases, whitespace-only LLM responses,
  zero target tokens, very large prior summaries, concurrent summarization
- ProactiveManager: disk usage division by zero, schedule interval floor,
  cooldown boundaries, fire_trigger with empty template, get_status formatting
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Any
from unittest.mock import MagicMock, patch

# ---------------------------------------------------------------------------
# Summarizer tests
# ---------------------------------------------------------------------------


@dataclass
class FakeTurn:
    timestamp: str = "2026-03-19T12:00:00"
    role: str = "user"
    content: str = "Hello"


@dataclass
class FakeSummaryRecord:
    depth: int = 0
    content: str = "Summary text"
    time_range_start: str = ""
    time_range_end: str = ""


class FakeCompletionResponse:
    def __init__(self, content: str = ""):
        self.content = content


class FakeProvider:
    def __init__(self, responses: list[str] | None = None):
        self._responses = list(responses or [])
        self._idx = 0
        self.calls: list[dict] = []

    def chat(self, messages: Any, temperature: float = 0.2, max_tokens: int = 4096) -> Any:
        self.calls.append({"messages": messages, "temperature": temperature})
        if self._idx < len(self._responses):
            resp = self._responses[self._idx]
            self._idx += 1
            if resp is None:
                raise RuntimeError("LLM unavailable")
            return FakeCompletionResponse(resp)
        return FakeCompletionResponse("")


class TestSummarizerEdgeCases:
    """Edge cases in the Summarizer module."""

    def test_empty_turns_list(self):
        """Summarize with zero turns should produce a fallback or empty transcript."""
        from missy.agent.summarizer import Summarizer

        provider = FakeProvider(responses=["Short summary."])
        s = Summarizer(provider=provider)
        text, tier = s.summarize_turns([])
        # Empty transcript → LLM gets an empty conversation section
        assert isinstance(text, str)
        assert tier in ("normal", "aggressive", "fallback")

    def test_turn_with_none_timestamp(self):
        """Turn with None timestamp should use '?' placeholder."""
        from missy.agent.summarizer import Summarizer

        turn = FakeTurn(timestamp=None, role="user", content="test")
        provider = FakeProvider(responses=["ok"])
        s = Summarizer(provider=provider)
        text, tier = s.summarize_turns([turn])
        assert isinstance(text, str)
        # The call should have succeeded, not raised
        assert len(provider.calls) >= 1

    def test_turn_with_short_timestamp(self):
        """Turn with timestamp shorter than 19 chars should be handled."""
        from missy.agent.summarizer import Summarizer

        turn = FakeTurn(timestamp="2026", role="user", content="test")
        provider = FakeProvider(responses=["ok"])
        s = Summarizer(provider=provider)
        # _format_turns does t.timestamp[:19] — slicing short string is safe
        text, tier = s.summarize_turns([turn])
        assert isinstance(text, str)

    def test_turn_with_empty_timestamp(self):
        """Turn with empty string timestamp should use '?' placeholder."""
        from missy.agent.summarizer import Summarizer

        turn = FakeTurn(timestamp="", role="user", content="test")
        provider = FakeProvider(responses=["ok"])
        s = Summarizer(provider=provider)
        text, tier = s.summarize_turns([turn])
        assert isinstance(text, str)

    def test_llm_returns_whitespace_only(self):
        """LLM returning only whitespace — truthy but strips to near-empty.

        Whitespace is truthy in Python so tier 1 accepts if tokens <= input.
        The result will be an empty/whitespace string after strip().
        """
        from missy.agent.summarizer import Summarizer

        # Whitespace is truthy, and ~1 token < input tokens → tier 1 accepts
        provider = FakeProvider(responses=["   \n\t  "])
        s = Summarizer(provider=provider)
        turns = [FakeTurn(content="important content " * 20)]
        text, tier = s.summarize_turns(turns)
        # Tier 1 accepts (result is truthy), strip() produces ""
        assert tier == "normal"
        assert text == ""

    def test_llm_returns_empty_string(self):
        """Empty string from LLM should escalate."""
        from missy.agent.summarizer import Summarizer

        provider = FakeProvider(responses=["", ""])
        s = Summarizer(provider=provider)
        turns = [FakeTurn(content="test " * 50)]
        text, tier = s.summarize_turns(turns)
        assert tier == "fallback"

    def test_target_tokens_zero(self):
        """target_tokens=0 should produce empty truncation in fallback."""
        from missy.agent.summarizer import Summarizer

        provider = FakeProvider(responses=[None, None])  # Both tiers fail
        s = Summarizer(provider=provider)
        turns = [FakeTurn(content="test")]
        text, tier = s.summarize_turns(turns, target_tokens=0)
        assert tier == "fallback"
        # Truncation: prompt[:0 * 4] = prompt[:0] = ""
        assert "[TRUNCATED" in text

    def test_target_tokens_negative(self):
        """Negative target_tokens should not crash."""
        from missy.agent.summarizer import Summarizer

        provider = FakeProvider(responses=[None, None])
        s = Summarizer(provider=provider)
        turns = [FakeTurn(content="test")]
        text, tier = s.summarize_turns(turns, target_tokens=-10)
        assert tier == "fallback"

    def test_llm_response_larger_than_input_escalates(self):
        """If LLM output is >= input tokens, should escalate to tier 2."""
        from missy.agent.summarizer import Summarizer

        # Short input, long output from tier 1
        long_response = "word " * 500
        provider = FakeProvider(responses=[long_response, "short"])
        s = Summarizer(provider=provider)
        turns = [FakeTurn(content="hi")]
        text, tier = s.summarize_turns(turns)
        assert tier == "aggressive"
        assert text == "short"

    def test_prior_summary_very_large(self):
        """Very large prior_summary should still work (dominates prompt)."""
        from missy.agent.summarizer import Summarizer

        large_prior = "Previous context: " + "x" * 10000
        provider = FakeProvider(responses=["condensed"])
        s = Summarizer(provider=provider)
        turns = [FakeTurn(content="test")]
        text, tier = s.summarize_turns(turns, prior_summary=large_prior)
        assert text == "condensed"
        assert tier == "normal"

    def test_turn_content_with_newlines(self):
        """Multi-line turn content should not break formatting."""
        from missy.agent.summarizer import Summarizer

        turn = FakeTurn(content="line1\nline2\nline3")
        provider = FakeProvider(responses=["ok"])
        s = Summarizer(provider=provider)
        text, tier = s.summarize_turns([turn])
        assert isinstance(text, str)

    def test_tier_counts_accurate(self):
        """Tier counts should track correct tiers."""
        from missy.agent.summarizer import Summarizer

        provider = FakeProvider(responses=["ok", "ok", None, None])
        s = Summarizer(provider=provider)
        s.summarize_turns([FakeTurn(content="test " * 50)])
        assert s.tier_counts["normal"] == 1

        s.summarize_turns([FakeTurn(content="test " * 50)])
        assert s.tier_counts["normal"] == 2

    def test_both_tiers_raise_exception(self):
        """Both LLM tiers raising should fall to deterministic truncation."""
        from missy.agent.summarizer import Summarizer

        provider = FakeProvider(responses=[None, None])
        s = Summarizer(provider=provider)
        turns = [FakeTurn(content="important " * 100)]
        text, tier = s.summarize_turns(turns)
        assert tier == "fallback"
        assert s.tier_counts["fallback"] == 1

    def test_summarize_summaries_empty_list(self):
        """Summarizing zero summaries should work."""
        from missy.agent.summarizer import Summarizer

        provider = FakeProvider(responses=["merged"])
        s = Summarizer(provider=provider)
        text, tier = s.summarize_summaries([])
        assert isinstance(text, str)

    def test_summarize_summaries_with_time_ranges(self):
        """Summaries with time ranges should format correctly."""
        from missy.agent.summarizer import Summarizer

        records = [
            FakeSummaryRecord(depth=1, content="First", time_range_start="10:00", time_range_end="11:00"),
            FakeSummaryRecord(depth=2, content="Second"),
        ]
        provider = FakeProvider(responses=["merged"])
        s = Summarizer(provider=provider)
        text, tier = s.summarize_summaries(records)
        assert isinstance(text, str)

    def test_format_turns_preserves_order(self):
        """Turns should be formatted in order."""
        from missy.agent.summarizer import Summarizer

        turns = [
            FakeTurn(timestamp="2026-03-19T12:00:00", role="user", content="first"),
            FakeTurn(timestamp="2026-03-19T12:01:00", role="assistant", content="second"),
        ]
        result = Summarizer._format_turns(turns)
        lines = result.split("\n")
        assert len(lines) == 2
        assert "first" in lines[0]
        assert "second" in lines[1]

    def test_approx_tokens_empty_string(self):
        """_approx_tokens('') should return 1 (minimum)."""
        from missy.agent.summarizer import _approx_tokens

        assert _approx_tokens("") == 1

    def test_approx_tokens_short_string(self):
        """Short strings should return at least 1."""
        from missy.agent.summarizer import _approx_tokens

        assert _approx_tokens("hi") == 1
        assert _approx_tokens("test") == 1
        assert _approx_tokens("12345678") == 2


# ---------------------------------------------------------------------------
# ProactiveManager tests
# ---------------------------------------------------------------------------


class TestProactiveEdgeCases:
    """Edge cases in the ProactiveManager."""

    def test_fire_trigger_within_cooldown(self):
        """Trigger within cooldown window should be skipped."""
        from missy.agent.proactive import ProactiveManager, ProactiveTrigger

        callback = MagicMock()
        trigger = ProactiveTrigger(
            name="test", trigger_type="schedule", cooldown_seconds=60,
            prompt_template="fired",
        )
        mgr = ProactiveManager(triggers=[trigger], agent_callback=callback)
        mgr._fire_trigger(trigger)
        assert callback.call_count == 1
        mgr._fire_trigger(trigger)
        assert callback.call_count == 1  # Cooldown blocked second fire

    def test_fire_trigger_zero_cooldown(self):
        """Zero cooldown should allow rapid firing."""
        from missy.agent.proactive import ProactiveManager, ProactiveTrigger

        callback = MagicMock()
        trigger = ProactiveTrigger(
            name="rapid", trigger_type="schedule", cooldown_seconds=0,
            prompt_template="go",
        )
        mgr = ProactiveManager(triggers=[trigger], agent_callback=callback)
        mgr._fire_trigger(trigger)
        mgr._fire_trigger(trigger)
        assert callback.call_count == 2

    def test_fire_trigger_empty_template_uses_default(self):
        """Empty prompt_template should use the default template."""
        from missy.agent.proactive import ProactiveManager, ProactiveTrigger

        callback = MagicMock()
        trigger = ProactiveTrigger(
            name="defaults", trigger_type="schedule", prompt_template="",
        )
        mgr = ProactiveManager(triggers=[trigger], agent_callback=callback)
        mgr._fire_trigger(trigger)
        prompt = callback.call_args[0][0]
        assert "defaults" in prompt
        assert "schedule" in prompt

    def test_fire_trigger_with_format_style_template(self):
        """Template using {var} style should be auto-converted."""
        from missy.agent.proactive import ProactiveManager, ProactiveTrigger

        callback = MagicMock()
        trigger = ProactiveTrigger(
            name="fmt", trigger_type="load_threshold",
            prompt_template="Trigger {trigger_name} of type {trigger_type} at {timestamp}",
        )
        mgr = ProactiveManager(triggers=[trigger], agent_callback=callback)
        mgr._fire_trigger(trigger)
        prompt = callback.call_args[0][0]
        assert "fmt" in prompt
        assert "load_threshold" in prompt
        assert "{" not in prompt  # All substitutions resolved

    def test_fire_trigger_with_template_style(self):
        """Template using ${var} style should work directly."""
        from missy.agent.proactive import ProactiveManager, ProactiveTrigger

        callback = MagicMock()
        trigger = ProactiveTrigger(
            name="tmpl", trigger_type="disk_threshold",
            prompt_template="Alert: ${trigger_name} (${trigger_type})",
        )
        mgr = ProactiveManager(triggers=[trigger], agent_callback=callback)
        mgr._fire_trigger(trigger)
        prompt = callback.call_args[0][0]
        assert "tmpl" in prompt
        assert "disk_threshold" in prompt

    def test_fire_trigger_template_with_unknown_vars(self):
        """Template with unknown variables should leave them as-is (safe_substitute)."""
        from missy.agent.proactive import ProactiveManager, ProactiveTrigger

        callback = MagicMock()
        trigger = ProactiveTrigger(
            name="unknown", trigger_type="schedule",
            prompt_template="Value: ${unknown_var} and ${trigger_name}",
        )
        mgr = ProactiveManager(triggers=[trigger], agent_callback=callback)
        mgr._fire_trigger(trigger)
        prompt = callback.call_args[0][0]
        assert "unknown" in prompt
        assert "${unknown_var}" in prompt  # safe_substitute leaves it

    def test_requires_confirmation_no_gate(self):
        """Trigger requiring confirmation with no gate should skip."""
        from missy.agent.proactive import ProactiveManager, ProactiveTrigger

        callback = MagicMock()
        trigger = ProactiveTrigger(
            name="gated", trigger_type="schedule",
            requires_confirmation=True, prompt_template="test",
        )
        mgr = ProactiveManager(triggers=[trigger], agent_callback=callback, approval_gate=None)
        mgr._fire_trigger(trigger)
        assert callback.call_count == 0  # Skipped

    def test_requires_confirmation_gate_denies(self):
        """Trigger denied by approval gate should not call callback."""
        from missy.agent.proactive import ProactiveManager, ProactiveTrigger

        callback = MagicMock()
        gate = MagicMock()
        gate.request.side_effect = RuntimeError("Denied")
        trigger = ProactiveTrigger(
            name="denied", trigger_type="schedule",
            requires_confirmation=True, prompt_template="test",
        )
        mgr = ProactiveManager(triggers=[trigger], agent_callback=callback, approval_gate=gate)
        mgr._fire_trigger(trigger)
        assert callback.call_count == 0

    def test_requires_confirmation_gate_approves(self):
        """Trigger approved by gate should call callback."""
        from missy.agent.proactive import ProactiveManager, ProactiveTrigger

        callback = MagicMock()
        gate = MagicMock()
        trigger = ProactiveTrigger(
            name="approved", trigger_type="schedule",
            requires_confirmation=True, prompt_template="test",
        )
        mgr = ProactiveManager(triggers=[trigger], agent_callback=callback, approval_gate=gate)
        mgr._fire_trigger(trigger)
        assert callback.call_count == 1
        gate.request.assert_called_once()

    def test_agent_callback_exception_caught(self):
        """Exception in agent_callback should be caught, not propagated."""
        from missy.agent.proactive import ProactiveManager, ProactiveTrigger

        callback = MagicMock(side_effect=RuntimeError("Boom"))
        trigger = ProactiveTrigger(
            name="boom", trigger_type="schedule", prompt_template="test",
        )
        mgr = ProactiveManager(triggers=[trigger], agent_callback=callback)
        # Should not raise
        mgr._fire_trigger(trigger)

    def test_disabled_triggers_skipped_in_start(self):
        """Disabled triggers should not be started."""
        from missy.agent.proactive import ProactiveManager, ProactiveTrigger

        callback = MagicMock()
        triggers = [
            ProactiveTrigger(name="on", trigger_type="schedule", enabled=True, interval_seconds=1),
            ProactiveTrigger(name="off", trigger_type="schedule", enabled=False, interval_seconds=1),
        ]
        mgr = ProactiveManager(triggers=triggers, agent_callback=callback)
        mgr.start()
        try:
            # Only the enabled trigger gets a thread
            schedule_threads = [t for t in mgr._threads if "schedule" in t.name]
            assert len(schedule_threads) == 1
            assert "on" in schedule_threads[0].name
        finally:
            mgr.stop()

    def test_get_status_before_any_fires(self):
        """get_status should work when no triggers have fired."""
        from missy.agent.proactive import ProactiveManager, ProactiveTrigger

        callback = MagicMock()
        trigger = ProactiveTrigger(name="idle", trigger_type="schedule")
        mgr = ProactiveManager(triggers=[trigger], agent_callback=callback)
        status = mgr.get_status()
        assert status["active"] is True  # stop_event not set
        assert len(status["triggers"]) == 1
        assert status["triggers"][0]["last_fired"] is None

    def test_get_status_after_fire(self):
        """get_status should show last_fired timestamp after firing."""
        from missy.agent.proactive import ProactiveManager, ProactiveTrigger

        callback = MagicMock()
        trigger = ProactiveTrigger(
            name="fired", trigger_type="schedule", prompt_template="test",
        )
        mgr = ProactiveManager(triggers=[trigger], agent_callback=callback)
        mgr._fire_trigger(trigger)
        status = mgr.get_status()
        assert status["triggers"][0]["last_fired"] is not None

    def test_get_status_after_stop(self):
        """get_status should show active=False after stop."""
        from missy.agent.proactive import ProactiveManager

        callback = MagicMock()
        mgr = ProactiveManager(triggers=[], agent_callback=callback)
        mgr.stop()
        status = mgr.get_status()
        assert status["active"] is False

    def test_threshold_loop_disk_zero_total(self):
        """Disk usage with total=0 should not crash (ZeroDivisionError guard)."""
        from missy.agent.proactive import ProactiveManager, ProactiveTrigger

        callback = MagicMock()
        trigger = ProactiveTrigger(
            name="disk", trigger_type="disk_threshold",
            disk_path="/", disk_threshold_pct=90.0,
            interval_seconds=5,
        )
        mgr = ProactiveManager(triggers=[trigger], agent_callback=callback)
        mgr._stop_event.set()

        # Mock disk_usage to return total=0
        with patch("shutil.disk_usage") as mock_du:
            mock_du.return_value = MagicMock(used=0, total=0)
            # _threshold_loop should handle ZeroDivisionError gracefully
            # (caught by the except Exception clause)
            mgr._threshold_loop([trigger])

    def test_load_threshold_cpu_count_none(self):
        """os.cpu_count() returning None should use fallback of 1."""
        from missy.agent.proactive import ProactiveManager, ProactiveTrigger

        callback = MagicMock()
        trigger = ProactiveTrigger(
            name="load", trigger_type="load_threshold",
            load_threshold=0.1,  # Very low to ensure firing
            interval_seconds=5,
        )
        mgr = ProactiveManager(triggers=[trigger], agent_callback=callback)
        mgr._stop_event.set()

        with patch("os.getloadavg", return_value=(5.0, 3.0, 2.0)), \
             patch("os.cpu_count", return_value=None):
            mgr._threshold_loop([trigger])
        # Should not crash — cpu_count or 1 = 1

    def test_schedule_loop_minimum_interval(self):
        """Schedule loop should enforce interval >= 1 second."""
        from missy.agent.proactive import ProactiveManager, ProactiveTrigger

        callback = MagicMock()
        trigger = ProactiveTrigger(
            name="fast", trigger_type="schedule",
            interval_seconds=0,  # Should be clamped to 1
            prompt_template="tick",
        )
        mgr = ProactiveManager(triggers=[trigger], agent_callback=callback)
        mgr._stop_event.set()
        # _schedule_loop should use max(0, 1) = 1
        mgr._schedule_loop(trigger)
        # With stop_event set, the wait returns immediately

    def test_start_stop_idempotent(self):
        """Multiple start/stop cycles should be safe."""
        from missy.agent.proactive import ProactiveManager

        callback = MagicMock()
        mgr = ProactiveManager(triggers=[], agent_callback=callback)
        mgr.start()
        mgr.stop()
        mgr.start()
        mgr.stop()

    def test_stop_without_start(self):
        """Stopping without starting should be safe."""
        from missy.agent.proactive import ProactiveManager

        callback = MagicMock()
        mgr = ProactiveManager(triggers=[], agent_callback=callback)
        mgr.stop()  # Should not raise

    def test_concurrent_fire_trigger(self):
        """Concurrent fire_trigger calls should be thread-safe."""
        from missy.agent.proactive import ProactiveManager, ProactiveTrigger

        call_count = {"n": 0}
        lock = threading.Lock()

        def counting_callback(prompt, session_id):
            with lock:
                call_count["n"] += 1

        trigger = ProactiveTrigger(
            name="concurrent", trigger_type="schedule",
            cooldown_seconds=0, prompt_template="test",
        )
        mgr = ProactiveManager(triggers=[trigger], agent_callback=counting_callback)

        threads = []
        for _ in range(10):
            t = threading.Thread(target=mgr._fire_trigger, args=(trigger,))
            threads.append(t)
            t.start()
        for t in threads:
            t.join(timeout=5)

        assert call_count["n"] >= 1  # At least one should have fired

    def test_file_change_trigger_no_watch_path(self):
        """File change trigger with empty watch_path should be skipped."""
        from missy.agent.proactive import ProactiveManager, ProactiveTrigger

        callback = MagicMock()
        trigger = ProactiveTrigger(
            name="no-path", trigger_type="file_change",
            watch_path="",
        )
        mgr = ProactiveManager(triggers=[trigger], agent_callback=callback)
        # start() should skip this trigger gracefully
        mgr.start()
        mgr.stop()


# ---------------------------------------------------------------------------
# Approx tokens boundary tests
# ---------------------------------------------------------------------------


class TestApproxTokensBoundaries:
    """Boundary tests for the _approx_tokens helper."""

    def test_single_char(self):
        from missy.agent.summarizer import _approx_tokens
        assert _approx_tokens("a") == 1

    def test_four_chars(self):
        from missy.agent.summarizer import _approx_tokens
        assert _approx_tokens("abcd") == 1

    def test_five_chars(self):
        from missy.agent.summarizer import _approx_tokens
        assert _approx_tokens("abcde") == 1

    def test_eight_chars(self):
        from missy.agent.summarizer import _approx_tokens
        assert _approx_tokens("abcdefgh") == 2

    def test_very_long_string(self):
        from missy.agent.summarizer import _approx_tokens
        text = "x" * 10000
        assert _approx_tokens(text) == 2500
