"""Comprehensive tests for HeartbeatRunner and Watchdog.

Covers lifecycle, scheduling mechanics, active hours boundary conditions,
thread safety, metrics, health check state machine, audit event emission,
report structure, and error resilience — complementing the existing focused
unit tests in test_heartbeat.py and test_watchdog.py.
"""

from __future__ import annotations

import threading
import time
from datetime import datetime
from unittest.mock import MagicMock, patch

import pytest

from missy.agent.heartbeat import HEARTBEAT_FILE, HEARTBEAT_OK_FILE, HeartbeatRunner
from missy.agent.watchdog import SubsystemHealth, Watchdog

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_runner(tmp_path, *, interval=3600, active_hours="", report_fn=None):
    """Return a HeartbeatRunner pointed at tmp_path with a large interval
    so the background thread never fires during unit tests."""
    run_fn = MagicMock(return_value="ok")
    runner = HeartbeatRunner(
        agent_run_fn=run_fn,
        interval_seconds=interval,
        workspace=str(tmp_path),
        active_hours=active_hours,
        report_fn=report_fn,
    )
    return runner, run_fn


def _write_checklist(tmp_path, content="# Daily\n- [ ] Do thing"):
    (tmp_path / HEARTBEAT_FILE).write_text(content, encoding="utf-8")


def _freeze_in_active_hours(runner, hour, minute):
    """Patch _in_active_hours to use a frozen datetime instead of now()."""
    frozen = datetime(2026, 3, 18, hour, minute, 0)

    import re as _re

    def _patched():
        m = _re.match(r"(\d{2}):(\d{2})-(\d{2}):(\d{2})", runner._active_hours)
        if not m:
            return True
        now = frozen
        start = now.replace(hour=int(m[1]), minute=int(m[2]), second=0, microsecond=0)
        end = now.replace(hour=int(m[3]), minute=int(m[4]), second=0, microsecond=0)
        if end < start:
            return now >= start or now <= end
        return start <= now <= end

    runner._in_active_hours = _patched
    return runner


# ===========================================================================
# HeartbeatRunner — configuration
# ===========================================================================


class TestHeartbeatRunnerConfiguration:
    """Constructor arguments are stored correctly."""

    def test_workspace_tilde_expansion(self, tmp_path):
        runner = HeartbeatRunner(agent_run_fn=lambda p: "ok", workspace="~/workspace")
        # Path must be absolute (tilde expanded)
        assert runner._workspace.is_absolute()

    def test_active_hours_stored(self, tmp_path):
        runner, _ = _make_runner(tmp_path, active_hours="09:00-17:00")
        assert runner._active_hours == "09:00-17:00"

    def test_report_fn_stored_as_none_by_default(self, tmp_path):
        runner, _ = _make_runner(tmp_path)
        assert runner._report is None

    def test_report_fn_stored_when_provided(self, tmp_path):
        fn = MagicMock()
        runner, _ = _make_runner(tmp_path, report_fn=fn)
        assert runner._report is fn

    def test_stop_event_initially_clear(self, tmp_path):
        runner, _ = _make_runner(tmp_path)
        assert not runner._stop.is_set()

    def test_thread_initially_none(self, tmp_path):
        runner, _ = _make_runner(tmp_path)
        assert runner._thread is None


# ===========================================================================
# HeartbeatRunner — _fire() prompt construction
# ===========================================================================


class TestHeartbeatFirePromptContent:
    """Verify the prompt passed to the agent function contains the checklist."""

    def test_prompt_contains_heartbeat_header(self, tmp_path):
        checklist = "# Tasks\n- [ ] Check memory\n- [ ] Review logs"
        _write_checklist(tmp_path, checklist)
        runner, run_fn = _make_runner(tmp_path)
        runner._fire()
        prompt = run_fn.call_args[0][0]
        assert "[HEARTBEAT CHECK]" in prompt

    def test_prompt_contains_full_checklist_text(self, tmp_path):
        checklist = "- [ ] Verify disk space\n- [ ] Check error rates"
        _write_checklist(tmp_path, checklist)
        runner, run_fn = _make_runner(tmp_path)
        runner._fire()
        prompt = run_fn.call_args[0][0]
        assert checklist in prompt

    def test_prompt_header_precedes_checklist(self, tmp_path):
        checklist = "# Checks"
        _write_checklist(tmp_path, checklist)
        runner, run_fn = _make_runner(tmp_path)
        runner._fire()
        prompt = run_fn.call_args[0][0]
        header_pos = prompt.index("[HEARTBEAT CHECK]")
        checklist_pos = prompt.index("# Checks")
        assert header_pos < checklist_pos

    def test_fire_passes_single_positional_arg(self, tmp_path):
        """agent_run_fn must be called with exactly one positional arg."""
        _write_checklist(tmp_path)
        runner, run_fn = _make_runner(tmp_path)
        runner._fire()
        args, kwargs = run_fn.call_args
        assert len(args) == 1
        assert kwargs == {}


# ===========================================================================
# HeartbeatRunner — _fire() suppression / skip logic
# ===========================================================================


class TestHeartbeatFireSkipLogic:
    """Suppression precedence: HEARTBEAT_OK > active_hours > missing file."""

    def test_ok_file_takes_priority_over_active_hours(self, tmp_path):
        """HEARTBEAT_OK suppression is checked before active hours."""
        (tmp_path / HEARTBEAT_OK_FILE).write_text("suppress")
        _write_checklist(tmp_path)
        runner, run_fn = _make_runner(tmp_path, active_hours="00:00-23:59")
        runner._fire()
        # Suppressed by OK file — agent must not run
        run_fn.assert_not_called()
        assert runner._skips == 1

    def test_ok_file_removed_after_suppression(self, tmp_path):
        (tmp_path / HEARTBEAT_OK_FILE).write_text("")
        _write_checklist(tmp_path)
        runner, _ = _make_runner(tmp_path)
        runner._fire()
        assert not (tmp_path / HEARTBEAT_OK_FILE).exists()

    def test_missing_heartbeat_file_increments_skips_not_runs(self, tmp_path):
        runner, run_fn = _make_runner(tmp_path)
        runner._fire()
        assert runner._skips == 1
        assert runner._runs == 0
        run_fn.assert_not_called()

    def test_active_hours_skip_does_not_remove_checklist(self, tmp_path):
        _write_checklist(tmp_path)
        runner, _ = _make_runner(tmp_path, active_hours="08:00-22:00")
        with patch.object(runner, "_in_active_hours", return_value=False):
            runner._fire()
        assert (tmp_path / HEARTBEAT_FILE).exists()

    def test_exception_during_run_does_not_increment_skips(self, tmp_path):
        _write_checklist(tmp_path)
        runner, run_fn = _make_runner(tmp_path)
        run_fn.side_effect = ValueError("network error")
        runner._fire()
        assert runner._skips == 0
        assert runner._runs == 0

    def test_report_fn_not_called_when_run_raises(self, tmp_path):
        _write_checklist(tmp_path)
        report_fn = MagicMock()
        runner, run_fn = _make_runner(tmp_path, report_fn=report_fn)
        run_fn.side_effect = RuntimeError("boom")
        runner._fire()
        report_fn.assert_not_called()

    def test_report_fn_not_called_when_no_checklist(self, tmp_path):
        report_fn = MagicMock()
        runner, _ = _make_runner(tmp_path, report_fn=report_fn)
        runner._fire()
        report_fn.assert_not_called()

    def test_mixed_skip_and_run_metrics(self, tmp_path):
        """Three fires: skip (no file), run, run → skips=1, runs=2."""
        runner, run_fn = _make_runner(tmp_path)
        runner._fire()  # skip — no file
        _write_checklist(tmp_path)
        runner._fire()  # run
        runner._fire()  # run
        assert runner.metrics == {"runs": 2, "skips": 1}


# ===========================================================================
# HeartbeatRunner — active hours boundary conditions
# ===========================================================================


class TestHeartbeatActiveHoursBoundaries:
    """Edge cases at the exact start and end of the active window."""

    def test_exactly_at_start_time_is_active(self, tmp_path):
        runner, _ = _make_runner(tmp_path, active_hours="08:00-22:00")
        runner = _freeze_in_active_hours(runner, 8, 0)
        assert runner._in_active_hours() is True

    def test_exactly_at_end_time_is_active(self, tmp_path):
        runner, _ = _make_runner(tmp_path, active_hours="08:00-22:00")
        runner = _freeze_in_active_hours(runner, 22, 0)
        assert runner._in_active_hours() is True

    def test_one_minute_before_start_is_inactive(self, tmp_path):
        runner, _ = _make_runner(tmp_path, active_hours="08:00-22:00")
        runner = _freeze_in_active_hours(runner, 7, 59)
        assert runner._in_active_hours() is False

    def test_one_minute_after_end_is_inactive(self, tmp_path):
        runner, _ = _make_runner(tmp_path, active_hours="08:00-22:00")
        runner = _freeze_in_active_hours(runner, 22, 1)
        assert runner._in_active_hours() is False

    def test_overnight_window_at_midnight_is_active(self, tmp_path):
        """23:00-05:00 window; midnight should be active."""
        runner, _ = _make_runner(tmp_path, active_hours="23:00-05:00")
        runner = _freeze_in_active_hours(runner, 0, 0)
        assert runner._in_active_hours() is True

    def test_overnight_window_at_noon_is_inactive(self, tmp_path):
        """23:00-05:00 window; noon should be outside."""
        runner, _ = _make_runner(tmp_path, active_hours="23:00-05:00")
        runner = _freeze_in_active_hours(runner, 12, 0)
        assert runner._in_active_hours() is False

    def test_partial_match_format_returns_true(self, tmp_path):
        """Garbled active_hours string → default allow."""
        runner, _ = _make_runner(tmp_path, active_hours="8:00-22:00")
        # Non-matching format (single digit hour) — treated as "always active"
        assert runner._in_active_hours() is True

    def test_empty_active_hours_string_is_always_active(self, tmp_path):
        runner, _ = _make_runner(tmp_path, active_hours="")
        assert runner._in_active_hours() is True


# ===========================================================================
# HeartbeatRunner — start / stop / restart lifecycle
# ===========================================================================


class TestHeartbeatLifecycle:
    """Thread lifecycle, double-stop safety, and restart behavior."""

    def test_start_marks_thread_as_daemon(self, tmp_path):
        runner, _ = _make_runner(tmp_path)
        runner.start()
        assert runner._thread.daemon is True
        runner.stop()

    def test_start_names_thread_correctly(self, tmp_path):
        runner, _ = _make_runner(tmp_path)
        runner.start()
        assert runner._thread.name == "missy-heartbeat"
        runner.stop()

    def test_thread_is_alive_after_start(self, tmp_path):
        runner, _ = _make_runner(tmp_path)
        runner.start()
        assert runner._thread.is_alive()
        runner.stop()

    def test_stop_joins_thread(self, tmp_path):
        runner, _ = _make_runner(tmp_path)
        runner.start()
        runner.stop()
        assert not runner._thread.is_alive()

    def test_stop_without_start_is_safe(self, tmp_path):
        runner, _ = _make_runner(tmp_path)
        runner.stop()  # must not raise

    def test_double_stop_is_safe(self, tmp_path):
        runner, _ = _make_runner(tmp_path)
        runner.start()
        runner.stop()
        runner.stop()  # second stop must not raise

    def test_start_after_stop_creates_new_thread(self, tmp_path):
        runner, _ = _make_runner(tmp_path)
        runner.start()
        first_thread = runner._thread
        runner.stop()
        runner.start()
        second_thread = runner._thread
        assert second_thread is not first_thread
        runner.stop()

    def test_stop_clears_stop_event_on_restart(self, tmp_path):
        """After a restart the stop event must be clear so the loop runs."""
        runner, _ = _make_runner(tmp_path)
        runner.start()
        runner.stop()
        runner.start()
        assert not runner._stop.is_set()
        runner.stop()


# ===========================================================================
# HeartbeatRunner — thread safety
# ===========================================================================


class TestHeartbeatThreadSafety:
    """Concurrent _fire() calls do not corrupt metrics counters."""

    def test_concurrent_fire_calls_metrics_consistent(self, tmp_path):
        _write_checklist(tmp_path)
        runner, _ = _make_runner(tmp_path)
        n_threads = 20
        barrier = threading.Barrier(n_threads)

        def fire():
            barrier.wait()
            runner._fire()

        threads = [threading.Thread(target=fire) for _ in range(n_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=5)

        total = runner._runs + runner._skips
        assert total == n_threads


# ===========================================================================
# Watchdog — construction defaults
# ===========================================================================


class TestWatchdogDefaults:
    """Default constructor values."""

    def test_default_check_interval(self):
        wd = Watchdog()
        assert wd._interval == 60.0

    def test_default_failure_threshold(self):
        wd = Watchdog()
        assert wd._threshold == 3

    def test_custom_interval_stored(self):
        wd = Watchdog(check_interval=120.0)
        assert wd._interval == 120.0

    def test_custom_threshold_stored(self):
        wd = Watchdog(failure_threshold=5)
        assert wd._threshold == 5

    def test_checks_dict_initially_empty(self):
        wd = Watchdog()
        assert wd._checks == {}

    def test_health_dict_initially_empty(self):
        wd = Watchdog()
        assert wd._health == {}

    def test_thread_initially_none(self):
        wd = Watchdog()
        assert wd._thread is None

    def test_stop_event_initially_clear(self):
        wd = Watchdog()
        assert not wd._stop.is_set()


# ===========================================================================
# Watchdog — register()
# ===========================================================================


class TestWatchdogRegisterBehavior:
    """register() creates both check and health entries with correct defaults."""

    def test_registered_health_has_correct_name(self):
        wd = Watchdog()
        wd.register("my-svc", lambda: True)
        assert wd._health["my-svc"].name == "my-svc"

    def test_registered_health_starts_healthy(self):
        wd = Watchdog()
        wd.register("svc", lambda: True)
        assert wd._health["svc"].healthy is True

    def test_registered_health_starts_with_zero_failures(self):
        wd = Watchdog()
        wd.register("svc", lambda: True)
        assert wd._health["svc"].consecutive_failures == 0

    def test_registered_health_starts_with_empty_error(self):
        wd = Watchdog()
        wd.register("svc", lambda: True)
        assert wd._health["svc"].last_error == ""

    def test_register_replaces_check_fn(self):
        wd = Watchdog()
        first_fn = MagicMock(return_value=True)
        second_fn = MagicMock(return_value=False)
        wd.register("svc", first_fn)
        wd.register("svc", second_fn)
        assert wd._checks["svc"] is second_fn


# ===========================================================================
# Watchdog — _check_all() health state transitions
# ===========================================================================


class TestWatchdogCheckAllStateTransitions:
    """Health state machine: pass→fail→recover, failure counting, error text."""

    @patch("missy.core.events.event_bus")
    def test_last_checked_updated_on_success(self, mock_bus):
        wd = Watchdog()
        wd.register("svc", lambda: True)
        before = time.monotonic()
        wd._check_all()
        assert wd._health["svc"].last_checked >= before

    @patch("missy.core.events.event_bus")
    def test_last_checked_updated_on_failure(self, mock_bus):
        wd = Watchdog()
        wd.register("svc", lambda: False)
        before = time.monotonic()
        wd._check_all()
        assert wd._health["svc"].last_checked >= before

    @patch("missy.core.events.event_bus")
    def test_error_text_cleared_on_recovery(self, mock_bus):
        results = iter([False, True])
        wd = Watchdog()
        wd.register("svc", lambda: next(results))
        wd._check_all()
        assert wd._health["svc"].last_error != ""
        wd._check_all()
        assert wd._health["svc"].last_error == ""

    @patch("missy.core.events.event_bus")
    def test_false_return_stores_runtime_error_text(self, mock_bus):
        wd = Watchdog()
        wd.register("svc", lambda: False)
        wd._check_all()
        assert "health check returned False" in wd._health["svc"].last_error

    @patch("missy.core.events.event_bus")
    def test_exception_message_stored_in_last_error(self, mock_bus):
        wd = Watchdog()
        wd.register("svc", lambda: (_ for _ in ()).throw(ValueError("disk full")))
        wd._check_all()
        assert "disk full" in wd._health["svc"].last_error

    @patch("missy.core.events.event_bus")
    def test_threshold_not_yet_reached_healthy_remains_true_after_recovery(self, mock_bus):
        results = iter([False, True])
        wd = Watchdog(failure_threshold=3)
        wd.register("svc", lambda: next(results))
        wd._check_all()  # one failure
        wd._check_all()  # recovery
        assert wd._health["svc"].healthy is True
        assert wd._health["svc"].consecutive_failures == 0

    @patch("missy.core.events.event_bus")
    def test_failures_beyond_threshold_keep_incrementing(self, mock_bus):
        wd = Watchdog(failure_threshold=2)
        wd.register("svc", lambda: False)
        for _ in range(6):
            wd._check_all()
        assert wd._health["svc"].consecutive_failures == 6

    @patch("missy.core.events.event_bus")
    def test_check_fn_called_every_check_all_invocation(self, mock_bus):
        fn = MagicMock(return_value=True)
        wd = Watchdog()
        wd.register("svc", fn)
        wd._check_all()
        wd._check_all()
        wd._check_all()
        assert fn.call_count == 3

    @patch("missy.core.events.event_bus")
    def test_all_subsystems_checked_in_single_call(self, mock_bus):
        fn_a = MagicMock(return_value=True)
        fn_b = MagicMock(return_value=True)
        fn_c = MagicMock(return_value=False)
        wd = Watchdog()
        wd.register("a", fn_a)
        wd.register("b", fn_b)
        wd.register("c", fn_c)
        wd._check_all()
        fn_a.assert_called_once()
        fn_b.assert_called_once()
        fn_c.assert_called_once()


# ===========================================================================
# Watchdog — audit event emission
# ===========================================================================


class TestWatchdogAuditEventEmission:
    """AuditEvent content and resilience when the bus is unavailable."""

    @patch("missy.core.events.event_bus")
    def test_event_published_for_each_subsystem(self, mock_bus):
        wd = Watchdog()
        wd.register("alpha", lambda: True)
        wd.register("beta", lambda: True)
        wd._check_all()
        assert mock_bus.publish.call_count == 2

    @patch("missy.core.events.event_bus")
    def test_event_result_allow_when_healthy(self, mock_bus):
        wd = Watchdog()
        wd.register("svc", lambda: True)
        wd._check_all()
        event = mock_bus.publish.call_args[0][0]
        assert event.result == "allow"

    @patch("missy.core.events.event_bus")
    def test_event_result_error_when_unhealthy(self, mock_bus):
        wd = Watchdog()
        wd.register("svc", lambda: False)
        wd._check_all()
        event = mock_bus.publish.call_args[0][0]
        assert event.result == "error"

    @patch("missy.core.events.event_bus")
    def test_event_type_is_watchdog_health_check(self, mock_bus):
        wd = Watchdog()
        wd.register("svc", lambda: True)
        wd._check_all()
        event = mock_bus.publish.call_args[0][0]
        assert event.event_type == "watchdog.health_check"

    @patch("missy.core.events.event_bus")
    def test_event_detail_contains_subsystem_name(self, mock_bus):
        wd = Watchdog()
        wd.register("my-db", lambda: True)
        wd._check_all()
        event = mock_bus.publish.call_args[0][0]
        assert event.detail["subsystem"] == "my-db"

    @patch("missy.core.events.event_bus")
    def test_event_detail_healthy_flag_matches_state(self, mock_bus):
        wd = Watchdog()
        wd.register("svc", lambda: False)
        wd._check_all()
        event = mock_bus.publish.call_args[0][0]
        assert event.detail["healthy"] is False

    @patch("missy.core.events.event_bus")
    def test_event_detail_failures_count(self, mock_bus):
        wd = Watchdog()
        wd.register("svc", lambda: False)
        wd._check_all()
        wd._check_all()
        event = mock_bus.publish.call_args[0][0]
        assert event.detail["failures"] == 2

    @patch("missy.core.events.event_bus")
    def test_bus_publish_exception_does_not_crash_watchdog(self, mock_bus):
        mock_bus.publish.side_effect = Exception("bus exploded")
        wd = Watchdog()
        wd.register("svc", lambda: True)
        wd._check_all()  # must not raise
        assert wd._health["svc"].healthy is True

    @patch("missy.core.events.event_bus")
    def test_check_fn_exception_still_publishes_event(self, mock_bus):
        wd = Watchdog()
        wd.register("svc", lambda: (_ for _ in ()).throw(OSError("pipe broken")))
        wd._check_all()
        mock_bus.publish.assert_called_once()


# ===========================================================================
# Watchdog — get_report()
# ===========================================================================


class TestWatchdogGetReport:
    """Report dict structure and values after various check sequences."""

    @patch("missy.core.events.event_bus")
    def test_report_key_per_registered_subsystem(self, mock_bus):
        wd = Watchdog()
        wd.register("a", lambda: True)
        wd.register("b", lambda: False)
        wd._check_all()
        report = wd.get_report()
        assert set(report.keys()) == {"a", "b"}

    @patch("missy.core.events.event_bus")
    def test_report_healthy_true_for_passing_check(self, mock_bus):
        wd = Watchdog()
        wd.register("svc", lambda: True)
        wd._check_all()
        assert wd.get_report()["svc"]["healthy"] is True

    @patch("missy.core.events.event_bus")
    def test_report_healthy_false_for_failing_check(self, mock_bus):
        wd = Watchdog()
        wd.register("svc", lambda: False)
        wd._check_all()
        assert wd.get_report()["svc"]["healthy"] is False

    @patch("missy.core.events.event_bus")
    def test_report_consecutive_failures_count(self, mock_bus):
        wd = Watchdog()
        wd.register("svc", lambda: False)
        wd._check_all()
        wd._check_all()
        assert wd.get_report()["svc"]["consecutive_failures"] == 2

    @patch("missy.core.events.event_bus")
    def test_report_last_error_populated(self, mock_bus):
        wd = Watchdog()
        wd.register("svc", lambda: False)
        wd._check_all()
        assert wd.get_report()["svc"]["last_error"] != ""

    @patch("missy.core.events.event_bus")
    def test_report_before_any_checks_reflects_initial_state(self, mock_bus):
        wd = Watchdog()
        wd.register("svc", lambda: True)
        report = wd.get_report()
        # Before first check: healthy=True, failures=0, no error
        assert report["svc"]["healthy"] is True
        assert report["svc"]["consecutive_failures"] == 0
        assert report["svc"]["last_error"] == ""

    def test_report_empty_when_no_subsystems_registered(self):
        wd = Watchdog()
        assert wd.get_report() == {}


# ===========================================================================
# Watchdog — lifecycle
# ===========================================================================


class TestWatchdogLifecycle:
    """start/stop thread management."""

    def test_start_thread_is_alive(self):
        wd = Watchdog(check_interval=3600)
        wd.start()
        assert wd._thread.is_alive()
        wd.stop()

    def test_stop_thread_is_dead(self):
        wd = Watchdog(check_interval=3600)
        wd.start()
        wd.stop()
        assert not wd._thread.is_alive()

    def test_double_stop_does_not_raise(self):
        wd = Watchdog(check_interval=3600)
        wd.start()
        wd.stop()
        wd.stop()

    def test_stop_without_start_does_not_raise(self):
        wd = Watchdog()
        wd.stop()

    def test_start_after_stop_creates_fresh_thread(self):
        wd = Watchdog(check_interval=3600)
        wd.start()
        t1 = wd._thread
        wd.stop()
        wd.start()
        t2 = wd._thread
        assert t2 is not t1
        wd.stop()

    def test_thread_is_daemon(self):
        wd = Watchdog(check_interval=3600)
        wd.start()
        assert wd._thread.daemon is True
        wd.stop()

    def test_thread_name_is_missy_watchdog(self):
        wd = Watchdog(check_interval=3600)
        wd.start()
        assert wd._thread.name == "missy-watchdog"
        wd.stop()


# ===========================================================================
# Watchdog — thread safety
# ===========================================================================


class TestWatchdogThreadSafety:
    """Concurrent _check_all() calls do not corrupt the health state."""

    @patch("missy.core.events.event_bus")
    def test_concurrent_check_all_does_not_raise(self, mock_bus):
        wd = Watchdog()
        call_count = [0]
        lock = threading.Lock()

        def counting_check():
            with lock:
                call_count[0] += 1
            return True

        wd.register("svc", counting_check)
        n = 10
        barrier = threading.Barrier(n)

        def run():
            barrier.wait()
            wd._check_all()

        threads = [threading.Thread(target=run) for _ in range(n)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=5)

        assert call_count[0] == n


# ===========================================================================
# SubsystemHealth dataclass
# ===========================================================================


class TestSubsystemHealthDataclass:
    """Field defaults and mutation."""

    def test_name_required(self):
        with pytest.raises(TypeError):
            SubsystemHealth()

    def test_defaults_set_correctly(self):
        h = SubsystemHealth(name="x")
        assert h.healthy is True
        assert h.consecutive_failures == 0
        assert h.last_checked == 0.0
        assert h.last_error == ""

    def test_fields_are_mutable(self):
        h = SubsystemHealth(name="x")
        h.healthy = False
        h.consecutive_failures = 7
        h.last_error = "uh oh"
        assert h.healthy is False
        assert h.consecutive_failures == 7
        assert h.last_error == "uh oh"

    def test_repr_contains_name(self):
        h = SubsystemHealth(name="my-service")
        assert "my-service" in repr(h)
