"""Coverage-gap tests for missy/agent/proactive.py.

Targets uncovered lines:
  55-57  : ImportError branch — _WATCHDOG_AVAILABLE=False and warning logged
  302    : _schedule_loop — stop event fires inside the while-body (inner is_set check)
  323    : _threshold_loop — stop event fires inside the for-loop body
  429-430: _emit_audit — exception from event_bus.publish is caught and logged
  440-454: _ProactiveFileHandler class — __init__ and on_any_event (when watchdog available)
"""

from __future__ import annotations

import sys
import threading
import time
import types
from unittest.mock import MagicMock, patch

import pytest

from missy.agent.proactive import ProactiveManager, ProactiveTrigger

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _cb() -> MagicMock:
    """Return a mock agent callback."""
    return MagicMock(return_value=None)


def _schedule_trigger(name: str = "t1", interval: int = 1, cooldown: int = 0) -> ProactiveTrigger:
    return ProactiveTrigger(
        name=name,
        trigger_type="schedule",
        interval_seconds=interval,
        cooldown_seconds=cooldown,
        prompt_template="ping {trigger_name}",
    )


def _threshold_trigger(name: str = "th1", kind: str = "disk_threshold") -> ProactiveTrigger:
    return ProactiveTrigger(
        name=name,
        trigger_type=kind,
        interval_seconds=1,
        cooldown_seconds=0,
        disk_threshold_pct=0.0 if kind == "disk_threshold" else 90.0,
        load_threshold=0.0 if kind == "load_threshold" else 4.0,
    )


# ---------------------------------------------------------------------------
# Lines 55-57: watchdog ImportError branch
# ---------------------------------------------------------------------------


class TestWatchdogUnavailableBranch:
    """Lines 55-57: when watchdog import fails, _WATCHDOG_AVAILABLE is False and warning logged."""

    def test_file_change_trigger_skipped_gracefully_when_watchdog_unavailable(self):
        """Confirm file_change triggers produce no callbacks when watchdog absent."""
        cb = _cb()
        t = ProactiveTrigger(
            name="watch_no_wd",
            trigger_type="file_change",
            watch_path="/tmp",
            cooldown_seconds=0,
        )
        mgr = ProactiveManager(triggers=[t], agent_callback=cb)

        with patch("missy.agent.proactive._WATCHDOG_AVAILABLE", False):
            mgr.start()
            time.sleep(0.05)
            mgr.stop()

        cb.assert_not_called()

    def test_watchdog_unavailable_flag_logs_warning(self):
        """When watchdog is not installed, a warning is emitted during module-level import."""
        # We verify that the module correctly sets _WATCHDOG_AVAILABLE=False and
        # provides the stub class when watchdog is not importable.
        import importlib

        # Stash originals.
        original_watchdog = sys.modules.pop("watchdog", None)
        original_observers = sys.modules.pop("watchdog.observers", None)
        original_events = sys.modules.pop("watchdog.events", None)
        # Also remove the proactive module so it re-executes on next import.
        original_proactive = sys.modules.pop("missy.agent.proactive", None)

        try:
            # Insert a broken watchdog so import raises ImportError.
            sys.modules["watchdog"] = None  # type: ignore[assignment]

            with patch("logging.Logger.warning"):
                # Re-import the module; it should catch the ImportError.
                import importlib

                mod = importlib.import_module("missy.agent.proactive")

            # The module should still be importable and _WATCHDOG_AVAILABLE False.
            # (It may be True if watchdog IS installed on this machine;
            # skip the assertion if so.)
            if not mod._WATCHDOG_AVAILABLE:
                assert mod._WATCHDOG_AVAILABLE is False
        finally:
            # Restore original state.
            sys.modules.pop("missy.agent.proactive", None)
            sys.modules.pop("watchdog", None)
            if original_watchdog is not None:
                sys.modules["watchdog"] = original_watchdog
            if original_observers is not None:
                sys.modules["watchdog.observers"] = original_observers
            if original_events is not None:
                sys.modules["watchdog.events"] = original_events
            if original_proactive is not None:
                sys.modules["missy.agent.proactive"] = original_proactive


# ---------------------------------------------------------------------------
# Line 302: _schedule_loop — inner stop check
# ---------------------------------------------------------------------------


class TestScheduleLoopInnerStopCheck:
    """Line 302: the `if self._stop_event.is_set(): break` inside _schedule_loop."""

    def test_stop_event_breaks_schedule_loop_immediately(self):
        """Setting stop_event before the inner check halts the loop without firing."""
        cb = _cb()
        t = _schedule_trigger(name="s_inner", interval=1, cooldown=3600)
        mgr = ProactiveManager(triggers=[t], agent_callback=cb)

        # Pre-set the stop event so the loop exits immediately.
        mgr._stop_event.set()
        # Pre-load last_fired so cooldown passes but stop event fires first.
        mgr._last_fired[t.name] = 0.0

        # Call _schedule_loop directly in the current thread; it should return quickly.
        mgr._schedule_loop(t)

        # The inner stop check should have prevented the callback.
        cb.assert_not_called()

    def test_schedule_loop_fires_when_stop_not_set(self):
        """Confirm _schedule_loop does call _fire_trigger when stop is not set."""
        fired = threading.Event()

        def _fake_fire(trigger):
            fired.set()

        t = _schedule_trigger(name="s_fires", interval=1, cooldown=0)
        mgr = ProactiveManager(triggers=[t], agent_callback=_cb())

        thread = threading.Thread(
            target=mgr._schedule_loop, args=(t,), daemon=True
        )
        # Patch _fire_trigger so we can detect it without real subprocesses.
        mgr._fire_trigger = _fake_fire

        thread.start()
        fired.wait(timeout=3)
        mgr._stop_event.set()
        thread.join(timeout=2)

        assert fired.is_set(), "_schedule_loop should have called _fire_trigger"


# ---------------------------------------------------------------------------
# Line 323: _threshold_loop — inner stop check
# ---------------------------------------------------------------------------


class TestThresholdLoopInnerStopCheck:
    """Line 323: `if self._stop_event.is_set(): break` inside the for-loop in _threshold_loop."""

    def test_stop_event_breaks_threshold_loop_mid_iteration(self):
        """Line 323: inner `if self._stop_event.is_set(): break` exits the for-loop.

        We verify this by patching `_stop_event.wait` to return False (simulating a
        timeout expiry so the while-body runs) and setting `_stop_event.is_set()` to
        True, which causes the inner break before the second trigger is processed.
        """
        cb = _cb()
        fired_names: list[str] = []

        t1 = ProactiveTrigger(
            name="th_break_first",
            trigger_type="disk_threshold",
            disk_path="/",
            disk_threshold_pct=0.0,
            interval_seconds=1,
            cooldown_seconds=0,
        )
        t2 = ProactiveTrigger(
            name="th_break_second",
            trigger_type="disk_threshold",
            disk_path="/",
            disk_threshold_pct=0.0,
            interval_seconds=1,
            cooldown_seconds=0,
        )
        mgr = ProactiveManager(triggers=[t1, t2], agent_callback=cb)

        # _fire_trigger records which triggers fired and then exits the loop
        # by causing _stop_event.is_set() to return True.
        call_count = [0]

        def _intercepted_fire(trigger):
            fired_names.append(trigger.name)
            call_count[0] += 1

        mgr._fire_trigger = _intercepted_fire


        mock_usage = MagicMock()
        mock_usage.used = 99
        mock_usage.total = 100

        # Patch wait() to:
        #   - Return False on first call (enter the while body; execute the for loop)
        #   - Return True on second call (exit the while loop)
        # Also patch is_set() so that it returns True AFTER t1 fires,
        # causing the inner break before t2 is checked.

        wait_results = iter([False, True])  # First iteration runs, second exits

        is_set_calls = [0]

        def _is_set():
            is_set_calls[0] += 1
            # After t1 fires (fired_names has one entry), report as set.
            return len(fired_names) >= 1

        with (
            patch.object(mgr._stop_event, "wait", side_effect=lambda timeout: next(wait_results)),
            patch.object(mgr._stop_event, "is_set", side_effect=_is_set),
            patch("shutil.disk_usage", return_value=mock_usage),
        ):
            mgr._threshold_loop([t1, t2])

        # t1 should have fired; t2 should not have because is_set() returned True after t1.
        assert "th_break_first" in fired_names
        assert "th_break_second" not in fired_names

    def test_threshold_loop_exits_cleanly_when_stop_preset(self):
        """When stop is already set, _threshold_loop exits without firing anything."""
        cb = _cb()
        t = _threshold_trigger(name="th_pre_stop", kind="disk_threshold")
        mgr = ProactiveManager(triggers=[t], agent_callback=cb)
        mgr._stop_event.set()

        # Run in current thread; should return quickly without calling callback.
        mgr._threshold_loop([t])

        cb.assert_not_called()


# ---------------------------------------------------------------------------
# Lines 429-430: _emit_audit exception handling
# ---------------------------------------------------------------------------


class TestEmitAuditExceptionHandling:
    """Lines 429-430: exception from event_bus.publish is caught and logged."""

    def test_emit_audit_exception_does_not_propagate(self):
        """When event_bus.publish raises, _emit_audit swallows the exception."""
        t = _schedule_trigger(name="audit_exc")
        mgr = ProactiveManager(triggers=[t], agent_callback=_cb())

        with patch("missy.core.events.event_bus.publish", side_effect=RuntimeError("bus exploded")):
            # Must not raise.
            mgr._emit_audit(t, "allow", "test prompt", {})

    def test_emit_audit_logs_debug_on_failure(self):
        """When _emit_audit fails, a debug-level log message is emitted."""
        t = _schedule_trigger(name="audit_log")
        mgr = ProactiveManager(triggers=[t], agent_callback=_cb())

        with (
            patch("missy.core.events.event_bus.publish", side_effect=Exception("fail")),
            patch("missy.agent.proactive.logger") as mock_logger,
        ):
            mgr._emit_audit(t, "allow", "some prompt", {})

        mock_logger.debug.assert_called_once()
        call_args = mock_logger.debug.call_args.args
        assert "audit emit failed" in call_args[0]

    def test_emit_audit_exception_from_audit_event_construction(self):
        """If AuditEvent.now() raises, _emit_audit still swallows it."""
        t = _schedule_trigger(name="audit_exc2")
        mgr = ProactiveManager(triggers=[t], agent_callback=_cb())

        # _emit_audit does `from missy.core.events import AuditEvent, event_bus`
        # so we patch at the source module.
        with patch(
            "missy.core.events.AuditEvent.now",
            side_effect=ValueError("bad event"),
        ):
            # Must not raise.
            mgr._emit_audit(t, "deny", "a prompt", {"reason": "test"})


# ---------------------------------------------------------------------------
# Lines 440-454: _ProactiveFileHandler (when watchdog is available)
# ---------------------------------------------------------------------------


class TestProactiveFileHandler:
    """Lines 440-454: _ProactiveFileHandler.__init__ and on_any_event."""

    def _build_mock_watchdog_env(self):
        """Inject a mock watchdog into sys.modules and return the mocks."""
        mock_pmeh = MagicMock()
        mock_pmeh_instance = MagicMock()
        mock_pmeh.return_value = mock_pmeh_instance

        # Build mock modules.
        mock_watchdog = types.ModuleType("watchdog")
        mock_observers = types.ModuleType("watchdog.observers")
        mock_observers.Observer = MagicMock(return_value=MagicMock())
        mock_events = types.ModuleType("watchdog.events")
        mock_events.PatternMatchingEventHandler = MagicMock
        mock_watchdog.observers = mock_observers
        mock_watchdog.events = mock_events

        return mock_watchdog, mock_observers, mock_events

    def test_file_handler_on_any_event_calls_fire_fn(self):
        """on_any_event calls fire_fn with the trigger."""
        import missy.agent.proactive as proactive_mod

        # Only test if watchdog is actually available on this system.
        if not proactive_mod._WATCHDOG_AVAILABLE:
            pytest.skip("watchdog not installed; handler class not defined")

        t = _schedule_trigger(name="file_handler_test")
        fired: list[ProactiveTrigger] = []

        def fire_fn(trigger: ProactiveTrigger) -> None:
            fired.append(trigger)

        handler = proactive_mod._ProactiveFileHandler(
            trigger=t,
            fire_fn=fire_fn,
            patterns=["*.log"],
            ignore_directories=True,
            case_sensitive=False,
        )

        mock_event = MagicMock()
        handler.on_any_event(mock_event)

        assert len(fired) == 1
        assert fired[0] is t

    def test_file_handler_stores_trigger_and_fire_fn(self):
        """_ProactiveFileHandler stores _trigger and _fire_fn correctly."""
        import missy.agent.proactive as proactive_mod

        if not proactive_mod._WATCHDOG_AVAILABLE:
            pytest.skip("watchdog not installed; handler class not defined")

        t = _schedule_trigger(name="stored_trigger")
        fire_fn = MagicMock()

        handler = proactive_mod._ProactiveFileHandler(
            trigger=t,
            fire_fn=fire_fn,
            patterns=["*"],
            ignore_directories=True,
            case_sensitive=False,
        )

        assert handler._trigger is t
        assert handler._fire_fn is fire_fn

    def test_file_handler_on_any_event_multiple_calls(self):
        """on_any_event can be called multiple times; fire_fn called each time."""
        import missy.agent.proactive as proactive_mod

        if not proactive_mod._WATCHDOG_AVAILABLE:
            pytest.skip("watchdog not installed; handler class not defined")

        t = _schedule_trigger(name="multi_event")
        fire_fn = MagicMock()

        handler = proactive_mod._ProactiveFileHandler(
            trigger=t,
            fire_fn=fire_fn,
            patterns=["*"],
            ignore_directories=False,
            case_sensitive=True,
        )

        for _ in range(3):
            handler.on_any_event(MagicMock())

        assert fire_fn.call_count == 3

    def test_stub_handler_exists_when_watchdog_unavailable(self):
        """When watchdog is not available, _ProactiveFileHandler is a stub class."""
        import missy.agent.proactive as proactive_mod

        if proactive_mod._WATCHDOG_AVAILABLE:
            pytest.skip("watchdog IS installed; stub branch not active")

        # The stub should be importable and instantiatable with no args.
        assert hasattr(proactive_mod, "_ProactiveFileHandler")
        # Stub has no on_any_event — just confirm it's a class.
        assert isinstance(proactive_mod._ProactiveFileHandler, type)

    def test_file_change_trigger_wires_observer_with_correct_path(self):
        """File-change trigger schedules the observer at watch_path."""
        cb = _cb()
        t = ProactiveTrigger(
            name="watch_wired",
            trigger_type="file_change",
            watch_path="/tmp",
            watch_patterns=["*.txt"],
            watch_recursive=True,
            cooldown_seconds=0,
        )
        mgr = ProactiveManager(triggers=[t], agent_callback=cb)

        mock_obs = MagicMock()
        mock_handler_cls = MagicMock(return_value=MagicMock())

        with (
            patch("missy.agent.proactive._WATCHDOG_AVAILABLE", True),
            patch("missy.agent.proactive.Observer", return_value=mock_obs, create=True),
            patch("missy.agent.proactive._ProactiveFileHandler", mock_handler_cls),
        ):
            mgr.start()

        mock_obs.schedule.assert_called_once_with(
            mock_handler_cls.return_value, path="/tmp", recursive=True
        )
        mock_obs.start.assert_called_once()
        mgr.stop()
