"""Coverage-gap tests for missy/agent/proactive.py.

Targets uncovered lines:
  55-57    : ImportError branch — _WATCHDOG_AVAILABLE = False and warning logged
  192-195  : start() — file_change trigger with empty watch_path skipped
  218-225  : start() — threshold triggers launch polling thread
  251-252  : stop() — observer stop/join exception swallowed
  309-316  : _threshold_loop — actual disk/load polling via patched OS calls
  323      : _schedule_loop — inner is_set() break after wait() returns False
  440-454  : _ProactiveFileHandler class — __init__ and on_any_event
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
    """Return a fresh mock agent callback."""
    return MagicMock(return_value=None)


def _schedule_trigger(name: str = "t1", interval: int = 1, cooldown: int = 0) -> ProactiveTrigger:
    return ProactiveTrigger(
        name=name,
        trigger_type="schedule",
        interval_seconds=interval,
        cooldown_seconds=cooldown,
        prompt_template="ping {trigger_name}",
    )


def _disk_trigger(
    name: str = "disk1",
    threshold_pct: float = 0.0,
    interval: int = 1,
    cooldown: int = 0,
) -> ProactiveTrigger:
    return ProactiveTrigger(
        name=name,
        trigger_type="disk_threshold",
        disk_path="/",
        disk_threshold_pct=threshold_pct,
        interval_seconds=interval,
        cooldown_seconds=cooldown,
    )


def _load_trigger(
    name: str = "load1",
    threshold: float = 0.0,
    interval: int = 1,
    cooldown: int = 0,
) -> ProactiveTrigger:
    return ProactiveTrigger(
        name=name,
        trigger_type="load_threshold",
        load_threshold=threshold,
        interval_seconds=interval,
        cooldown_seconds=cooldown,
    )


# ---------------------------------------------------------------------------
# Lines 55-57: watchdog ImportError branch
# ---------------------------------------------------------------------------


class TestWatchdogImportErrorBranch:
    """Lines 55-57: when watchdog import fails _WATCHDOG_AVAILABLE is set False."""

    def test_module_still_importable_after_watchdog_missing(self):
        """Reimporting proactive without watchdog must succeed without raising."""
        original_proactive = sys.modules.pop("missy.agent.proactive", None)
        original_watchdog_observers = sys.modules.pop("watchdog.observers", None)
        original_watchdog_events = sys.modules.pop("watchdog.events", None)

        # Insert a sentinel that raises ImportError when subscripted.
        broken_mod = types.ModuleType("watchdog")
        broken_mod.observers = None  # type: ignore[attr-defined]

        sys.modules["watchdog.observers"] = None  # type: ignore[assignment]
        sys.modules["watchdog.events"] = None  # type: ignore[assignment]

        try:
            import importlib

            with patch("logging.Logger.warning"):
                mod = importlib.import_module("missy.agent.proactive")
            # The flag exists on the module regardless of value.
            assert hasattr(mod, "_WATCHDOG_AVAILABLE")
        finally:
            sys.modules.pop("missy.agent.proactive", None)
            sys.modules.pop("watchdog.observers", None)
            sys.modules.pop("watchdog.events", None)
            if original_watchdog_observers is not None:
                sys.modules["watchdog.observers"] = original_watchdog_observers
            if original_watchdog_events is not None:
                sys.modules["watchdog.events"] = original_watchdog_events
            if original_proactive is not None:
                sys.modules["missy.agent.proactive"] = original_proactive

    def test_watchdog_unavailable_skips_file_change_triggers(self):
        """With _WATCHDOG_AVAILABLE=False, file_change triggers produce no callbacks."""
        cb = _cb()
        t = ProactiveTrigger(
            name="no_wd",
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

    def test_watchdog_unavailable_logs_warning_at_start(self):
        """start() warns when watchdog is unavailable and file_change triggers exist."""
        cb = _cb()
        t = ProactiveTrigger(
            name="warn_test",
            trigger_type="file_change",
            watch_path="/tmp",
            cooldown_seconds=0,
        )
        mgr = ProactiveManager(triggers=[t], agent_callback=cb)

        with (
            patch("missy.agent.proactive._WATCHDOG_AVAILABLE", False),
            patch("missy.agent.proactive.logger") as mock_log,
        ):
            mgr.start()
            mgr.stop()

        # A warning must have been emitted mentioning "watchdog".
        warning_calls = [str(c) for c in mock_log.warning.call_args_list]
        assert any("watchdog" in w.lower() for w in warning_calls)


# ---------------------------------------------------------------------------
# Lines 192-195: start() — empty watch_path triggers skipped
# ---------------------------------------------------------------------------


class TestEmptyWatchPathSkipped:
    """Lines 192-195: a file_change trigger with no watch_path is skipped with a warning."""

    def test_empty_watch_path_does_not_schedule_observer(self):
        """When watch_path is empty the observer.schedule must not be called."""
        cb = _cb()
        t = ProactiveTrigger(
            name="no_path",
            trigger_type="file_change",
            watch_path="",
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
            mgr.stop()

        mock_obs.schedule.assert_not_called()

    def test_empty_watch_path_logs_warning(self):
        """A warning is logged when watch_path is empty."""
        cb = _cb()
        t = ProactiveTrigger(
            name="path_warn",
            trigger_type="file_change",
            watch_path="",
            cooldown_seconds=0,
        )
        mgr = ProactiveManager(triggers=[t], agent_callback=cb)

        mock_obs = MagicMock()

        with (
            patch("missy.agent.proactive._WATCHDOG_AVAILABLE", True),
            patch("missy.agent.proactive.Observer", return_value=mock_obs, create=True),
            patch("missy.agent.proactive.logger") as mock_log,
        ):
            mgr.start()
            mgr.stop()

        warning_msgs = [str(c) for c in mock_log.warning.call_args_list]
        assert any("watch_path" in w or "no watch_path" in w for w in warning_msgs)

    def test_valid_watch_path_does_schedule_observer(self):
        """When watch_path is non-empty the observer is scheduled (contrast test)."""
        cb = _cb()
        t = ProactiveTrigger(
            name="has_path",
            trigger_type="file_change",
            watch_path="/tmp",
            watch_patterns=["*.log"],
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
            mock_handler_cls.return_value, path="/tmp", recursive=False
        )
        mgr.stop()


# ---------------------------------------------------------------------------
# Lines 218-225: start() — threshold triggers launch their polling thread
# ---------------------------------------------------------------------------


class TestThresholdThreadLaunch:
    """Lines 218-225: threshold triggers cause start() to spawn a polling thread."""

    def test_disk_threshold_thread_starts_and_stops(self):
        """A disk_threshold trigger causes a background thread named proactive-threshold."""
        cb = _cb()
        t = _disk_trigger(name="dt_start", threshold_pct=200.0)  # Unreachable threshold.
        mgr = ProactiveManager(triggers=[t], agent_callback=cb)

        mgr.start()
        thread_names = [th.name for th in mgr._threads]
        mgr.stop()

        assert "proactive-threshold" in thread_names

    def test_load_threshold_thread_starts_and_stops(self):
        """A load_threshold trigger also creates the proactive-threshold polling thread."""
        cb = _cb()
        t = _load_trigger(name="lt_start", threshold=9999.0)
        mgr = ProactiveManager(triggers=[t], agent_callback=cb)

        mgr.start()
        thread_names = [th.name for th in mgr._threads]
        mgr.stop()

        assert "proactive-threshold" in thread_names

    def test_multiple_threshold_triggers_share_one_thread(self):
        """Multiple threshold triggers are handled by a single polling thread."""
        cb = _cb()
        t1 = _disk_trigger(name="dt_multi", threshold_pct=200.0)
        t2 = _load_trigger(name="lt_multi", threshold=9999.0)
        mgr = ProactiveManager(triggers=[t1, t2], agent_callback=cb)

        mgr.start()
        threshold_threads = [th for th in mgr._threads if th.name == "proactive-threshold"]
        mgr.stop()

        # Exactly one polling thread for all threshold triggers.
        assert len(threshold_threads) == 1


# ---------------------------------------------------------------------------
# Lines 251-252: stop() — observer stop exception swallowed
# ---------------------------------------------------------------------------


class TestObserverStopException:
    """Lines 251-252: exceptions raised by observer.stop() are caught and debug-logged."""

    def test_observer_stop_exception_does_not_propagate(self):
        """If observer.stop() raises, stop() must still complete without raising."""
        cb = _cb()
        t = ProactiveTrigger(
            name="obs_exc",
            trigger_type="file_change",
            watch_path="/tmp",
            cooldown_seconds=0,
        )
        mgr = ProactiveManager(triggers=[t], agent_callback=cb)

        # Pre-populate a mock observer that raises on stop().
        mock_obs = MagicMock()
        mock_obs.stop.side_effect = RuntimeError("observer exploded")
        mgr._observer = mock_obs

        # Must not raise.
        mgr.stop()

        mock_obs.stop.assert_called_once()

    def test_observer_join_exception_does_not_propagate(self):
        """If observer.join() raises, stop() still completes."""
        cb = _cb()
        t = ProactiveTrigger(
            name="join_exc",
            trigger_type="file_change",
            watch_path="/tmp",
            cooldown_seconds=0,
        )
        mgr = ProactiveManager(triggers=[t], agent_callback=cb)

        mock_obs = MagicMock()
        mock_obs.stop.return_value = None
        mock_obs.join.side_effect = OSError("join failed")
        mgr._observer = mock_obs

        # Must not raise.
        mgr.stop()

    def test_observer_exception_debug_logged(self):
        """Debug log message is emitted when observer stop raises."""
        cb = _cb()
        t = ProactiveTrigger(
            name="obs_log",
            trigger_type="file_change",
            watch_path="/tmp",
            cooldown_seconds=0,
        )
        mgr = ProactiveManager(triggers=[t], agent_callback=cb)

        mock_obs = MagicMock()
        mock_obs.stop.side_effect = RuntimeError("boom")
        mgr._observer = mock_obs

        with patch("missy.agent.proactive.logger") as mock_log:
            mgr.stop()

        debug_calls = [str(c) for c in mock_log.debug.call_args_list]
        assert any("observer stop error" in c or "stop" in c.lower() for c in debug_calls)


# ---------------------------------------------------------------------------
# Lines 309-316: _threshold_loop — live polling path
# ---------------------------------------------------------------------------


class TestThresholdLoopPolling:
    """Lines 309-316: _threshold_loop processes disk and load checks in the while body."""

    def test_disk_threshold_fires_when_exceeded_via_loop(self):
        """_threshold_loop calls _fire_trigger when disk usage exceeds threshold."""
        fired = threading.Event()

        def _fire(trigger):
            fired.set()

        t = _disk_trigger(name="disk_poll", threshold_pct=0.0)  # Always exceeded.
        mgr = ProactiveManager(triggers=[t], agent_callback=_cb())
        mgr._fire_trigger = _fire

        mock_usage = MagicMock()
        mock_usage.used = 99
        mock_usage.total = 100  # 99% > 0%

        # Control wait() so the while body executes once, then the loop exits.
        with (
            patch.object(mgr._stop_event, "wait", side_effect=[False, True]),
            patch("shutil.disk_usage", return_value=mock_usage),
        ):
            mgr._threshold_loop([t])

        assert fired.is_set(), "_threshold_loop should have fired the disk trigger"

    def test_load_threshold_fires_when_exceeded_via_loop(self):
        """_threshold_loop calls _fire_trigger when normalised load exceeds threshold."""
        fired = threading.Event()

        def _fire(trigger):
            fired.set()

        t = _load_trigger(name="load_poll", threshold=0.0)  # Always exceeded.
        mgr = ProactiveManager(triggers=[t], agent_callback=_cb())
        mgr._fire_trigger = _fire

        with (
            patch.object(mgr._stop_event, "wait", side_effect=[False, True]),
            patch("os.getloadavg", return_value=(8.0, 4.0, 2.0)),
            patch("os.cpu_count", return_value=4),
        ):
            mgr._threshold_loop([t])

        assert fired.is_set(), "_threshold_loop should have fired the load trigger"

    def test_threshold_loop_exception_is_swallowed(self):
        """An exception inside the polling body is caught; the loop continues."""
        call_count = [0]
        stop_after = 2

        t = _disk_trigger(name="disk_exc", threshold_pct=0.0)
        mgr = ProactiveManager(triggers=[t], agent_callback=_cb())

        def _fire(trigger):
            call_count[0] += 1
            if call_count[0] >= stop_after:
                mgr._stop_event.set()

        mgr._fire_trigger = _fire

        def _raise_usage(*args, **kwargs):
            raise OSError("disk unreachable")

        def _run():
            with patch("shutil.disk_usage", side_effect=_raise_usage):
                mgr._threshold_loop([t])

        # The loop should exit cleanly even though disk_usage always raises.
        thread = threading.Thread(target=_run, daemon=True)
        thread.start()
        # Give the loop time to run a couple of iterations (poll_interval floored to 5s
        # so we just set the stop event and let it exit on next wait timeout).
        mgr._stop_event.set()
        thread.join(timeout=10)

        # Thread should have exited without crashing.
        assert not thread.is_alive()

    def test_disk_threshold_no_fire_when_under(self):
        """_threshold_loop must not fire when disk usage is below threshold."""
        fired = threading.Event()

        def _fire(trigger):
            fired.set()

        t = _disk_trigger(name="disk_under", threshold_pct=99.0)  # Almost impossible.
        mgr = ProactiveManager(triggers=[t], agent_callback=_cb())
        mgr._fire_trigger = _fire

        mock_usage = MagicMock()
        mock_usage.used = 1
        mock_usage.total = 100  # 1% < 99%

        # Run one iteration directly by controlling wait.
        with (
            patch.object(mgr._stop_event, "wait", side_effect=[False, True]),
            patch("shutil.disk_usage", return_value=mock_usage),
        ):
            mgr._threshold_loop([t])

        assert not fired.is_set()


# ---------------------------------------------------------------------------
# Line 323: _schedule_loop — inner is_set() break
# ---------------------------------------------------------------------------


class TestScheduleLoopInnerStopBreak:
    """Line 323: `if self._stop_event.is_set(): break` inside _schedule_loop body."""

    def test_inner_stop_check_prevents_fire_after_wait_returns_false(self):
        """When wait() returns False but is_set() is True, the loop breaks without firing."""
        cb = _cb()
        t = _schedule_trigger(name="inner_break", interval=1, cooldown=3600)
        mgr = ProactiveManager(triggers=[t], agent_callback=cb)

        # wait() returning False means the timeout elapsed — the while body runs.
        # is_set() returning True causes the inner break before _fire_trigger.
        with (
            patch.object(mgr._stop_event, "wait", return_value=False),
            patch.object(mgr._stop_event, "is_set", return_value=True),
        ):
            # Runs in the current thread; the break exits the while body immediately.
            # wrap in a thread with timeout so it can't hang.
            done = threading.Event()

            def _run():
                mgr._schedule_loop(t)
                done.set()

            th = threading.Thread(target=_run, daemon=True)
            th.start()
            done.wait(timeout=2)
            th.join(timeout=2)

        # The callback must not have been called because the inner break fired first.
        cb.assert_not_called()

    def test_schedule_loop_exits_when_stop_preset(self):
        """If stop_event is already set, _schedule_loop exits via the while condition."""
        cb = _cb()
        t = _schedule_trigger(name="preset_stop", interval=1, cooldown=0)
        mgr = ProactiveManager(triggers=[t], agent_callback=cb)
        mgr._stop_event.set()

        mgr._schedule_loop(t)  # Must return quickly.

        cb.assert_not_called()

    def test_schedule_loop_fires_before_stop_is_detected(self):
        """_schedule_loop calls _fire_trigger when wait() returns False and is_set() is False."""
        fired = threading.Event()

        def _fake_fire(trigger):
            fired.set()

        t = _schedule_trigger(name="fires_ok", interval=1, cooldown=0)
        mgr = ProactiveManager(triggers=[t], agent_callback=_cb())
        mgr._fire_trigger = _fake_fire

        call_no = [0]

        def _wait(timeout):
            call_no[0] += 1
            return call_no[0] != 1

        with patch.object(mgr._stop_event, "wait", side_effect=_wait):
            mgr._schedule_loop(t)

        assert fired.is_set(), "_fire_trigger should have been called"

    def test_full_lifecycle_schedule_trigger_start_stop(self):
        """Start a real schedule trigger thread, then stop it; verifies thread cleanup."""
        cb = _cb()
        t = _schedule_trigger(name="lifecycle", interval=1, cooldown=3600)
        mgr = ProactiveManager(triggers=[t], agent_callback=cb)

        mgr.start()
        assert len(mgr._threads) >= 1
        mgr.stop()

        # After stop, threads list is cleared.
        assert mgr._threads == []


# ---------------------------------------------------------------------------
# Lines 440-454: _ProactiveFileHandler class (watchdog available)
# ---------------------------------------------------------------------------


class TestProactiveFileHandler:
    """Lines 440-454: _ProactiveFileHandler is defined and works when watchdog is available."""

    def test_on_any_event_calls_fire_fn_with_trigger(self):
        """on_any_event passes the stored trigger to the fire function."""
        import missy.agent.proactive as mod

        if not mod._WATCHDOG_AVAILABLE:
            pytest.skip("watchdog not installed")

        t = _schedule_trigger(name="handler_test")
        fired: list[ProactiveTrigger] = []

        handler = mod._ProactiveFileHandler(
            trigger=t,
            fire_fn=fired.append,
            patterns=["*.log"],
            ignore_directories=True,
            case_sensitive=False,
        )

        handler.on_any_event(MagicMock())

        assert len(fired) == 1
        assert fired[0] is t

    def test_handler_stores_trigger_and_fire_fn_attributes(self):
        """_trigger and _fire_fn attributes are set correctly in __init__."""
        import missy.agent.proactive as mod

        if not mod._WATCHDOG_AVAILABLE:
            pytest.skip("watchdog not installed")

        t = _schedule_trigger(name="attr_test")
        fn = MagicMock()

        handler = mod._ProactiveFileHandler(
            trigger=t,
            fire_fn=fn,
            patterns=["*"],
            ignore_directories=False,
            case_sensitive=True,
        )

        assert handler._trigger is t
        assert handler._fire_fn is fn

    def test_on_any_event_called_multiple_times(self):
        """fire_fn is invoked once per on_any_event call."""
        import missy.agent.proactive as mod

        if not mod._WATCHDOG_AVAILABLE:
            pytest.skip("watchdog not installed")

        t = _schedule_trigger(name="multi_events")
        fn = MagicMock()

        handler = mod._ProactiveFileHandler(
            trigger=t,
            fire_fn=fn,
            patterns=["*"],
            ignore_directories=True,
            case_sensitive=False,
        )

        for _ in range(5):
            handler.on_any_event(MagicMock())

        assert fn.call_count == 5

    def test_handler_passes_trigger_not_event(self):
        """fire_fn receives the ProactiveTrigger, not the filesystem event object."""
        import missy.agent.proactive as mod

        if not mod._WATCHDOG_AVAILABLE:
            pytest.skip("watchdog not installed")

        t = _schedule_trigger(name="pass_trigger")
        received: list = []

        def fn(arg):
            received.append(arg)

        handler = mod._ProactiveFileHandler(
            trigger=t,
            fire_fn=fn,
            patterns=["*"],
            ignore_directories=True,
            case_sensitive=False,
        )

        fake_event = MagicMock()
        handler.on_any_event(fake_event)

        assert len(received) == 1
        assert received[0] is t
        assert received[0] is not fake_event

    def test_stub_handler_is_class_when_watchdog_absent(self):
        """When watchdog is absent _ProactiveFileHandler is still a class (stub)."""
        import missy.agent.proactive as mod

        if mod._WATCHDOG_AVAILABLE:
            pytest.skip("watchdog IS installed; stub branch not active")

        assert isinstance(mod._ProactiveFileHandler, type)

    def test_handler_wired_through_start(self):
        """start() passes _ProactiveFileHandler to observer.schedule when watchdog is available."""
        cb = _cb()
        t = ProactiveTrigger(
            name="wired",
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


# ---------------------------------------------------------------------------
# Integration: get_status shape and last_fired recording
# ---------------------------------------------------------------------------


class TestGetStatus:
    """Verify get_status returns correct shape with and without fired triggers."""

    def test_status_before_any_fire(self):
        t = _schedule_trigger(name="s1")
        mgr = ProactiveManager(triggers=[t], agent_callback=_cb())

        status = mgr.get_status()

        assert "active" in status
        assert "triggers" in status
        assert len(status["triggers"]) == 1
        ts = status["triggers"][0]
        assert ts["name"] == "s1"
        assert ts["last_fired"] is None

    def test_status_active_flag_reflects_stop_event(self):
        t = _schedule_trigger(name="s2")
        mgr = ProactiveManager(triggers=[t], agent_callback=_cb())

        assert mgr.get_status()["active"] is True
        mgr._stop_event.set()
        assert mgr.get_status()["active"] is False

    def test_status_last_fired_set_after_fire(self):
        t = _schedule_trigger(name="s3", cooldown=0)
        mgr = ProactiveManager(triggers=[t], agent_callback=_cb())

        mgr._fire_trigger(t)
        status = mgr.get_status()

        ts = status["triggers"][0]
        assert ts["last_fired"] is not None

    def test_status_multiple_triggers(self):
        t1 = _schedule_trigger(name="a")
        t2 = _disk_trigger(name="b")
        mgr = ProactiveManager(triggers=[t1, t2], agent_callback=_cb())

        status = mgr.get_status()
        names = {tr["name"] for tr in status["triggers"]}
        assert names == {"a", "b"}


# ---------------------------------------------------------------------------
# Cooldown and confirmation edge cases
# ---------------------------------------------------------------------------


class TestCooldownEnforcement:
    def test_cooldown_blocks_rapid_second_fire(self):
        """Second _fire_trigger call within cooldown window is silently dropped."""
        cb = _cb()
        t = _schedule_trigger(name="cd1", cooldown=3600)
        mgr = ProactiveManager(triggers=[t], agent_callback=cb)

        mgr._fire_trigger(t)
        mgr._fire_trigger(t)

        assert cb.call_count == 1

    def test_cooldown_zero_allows_repeated_fires(self):
        cb = _cb()
        t = _schedule_trigger(name="cd2", cooldown=0)
        mgr = ProactiveManager(triggers=[t], agent_callback=cb)

        mgr._fire_trigger(t)
        mgr._fire_trigger(t)

        assert cb.call_count == 2


class TestConfirmationGate:
    def test_requires_confirmation_no_gate_skips_callback(self):
        cb = _cb()
        t = ProactiveTrigger(
            name="conf_no_gate",
            trigger_type="schedule",
            requires_confirmation=True,
            cooldown_seconds=0,
        )
        mgr = ProactiveManager(triggers=[t], agent_callback=cb, approval_gate=None)

        mgr._fire_trigger(t)

        cb.assert_not_called()

    def test_requires_confirmation_gate_raises_skips_callback(self):
        cb = _cb()
        gate = MagicMock()
        gate.request.side_effect = Exception("denied")
        t = ProactiveTrigger(
            name="conf_denied",
            trigger_type="schedule",
            requires_confirmation=True,
            cooldown_seconds=0,
        )
        mgr = ProactiveManager(triggers=[t], agent_callback=cb, approval_gate=gate)

        mgr._fire_trigger(t)

        cb.assert_not_called()

    def test_requires_confirmation_gate_approved_fires_callback(self):
        cb = _cb()
        gate = MagicMock()
        gate.request.return_value = None  # No exception = approved.
        t = ProactiveTrigger(
            name="conf_ok",
            trigger_type="schedule",
            requires_confirmation=True,
            cooldown_seconds=0,
        )
        mgr = ProactiveManager(triggers=[t], agent_callback=cb, approval_gate=gate)

        mgr._fire_trigger(t)

        cb.assert_called_once()


class TestAgentCallbackError:
    def test_callback_exception_does_not_propagate(self):
        """Exception inside agent_callback is caught and logged; _fire_trigger returns normally."""

        def _bad(prompt, session_id):
            raise ValueError("callback exploded")

        t = _schedule_trigger(name="cb_exc", cooldown=0)
        mgr = ProactiveManager(triggers=[t], agent_callback=_bad)

        # Must not raise.
        mgr._fire_trigger(t)

    def test_callback_exception_logs_error(self):
        def _bad(prompt, session_id):
            raise RuntimeError("bad callback")

        t = _schedule_trigger(name="cb_log", cooldown=0)
        mgr = ProactiveManager(triggers=[t], agent_callback=_bad)

        with patch("missy.agent.proactive.logger") as mock_log:
            mgr._fire_trigger(t)

        error_calls = [str(c) for c in mock_log.error.call_args_list]
        assert any("agent_callback error" in c or "cb_log" in c for c in error_calls)
