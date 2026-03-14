"""Tests targeting specific missed lines in missy/agent/proactive.py.

Coverage report gaps addressed here:
  55-57   : watchdog import SUCCESS path (_WATCHDOG_AVAILABLE = True)
  302     : inner ``if self._stop_event.is_set(): break`` in _threshold_loop
            when stop fires between triggers in the same iteration
  429-430 : ``event_bus.publish(event)`` executes AND the except branch when
            publish raises
  440-454 : ``_ProactiveFileHandler`` class body (only reachable when
            _WATCHDOG_AVAILABLE is True — requires mocking the module-level
            flag and the class itself)
"""

from __future__ import annotations

import importlib
import sys
import types
from unittest.mock import MagicMock, patch

from missy.agent.proactive import ProactiveManager, ProactiveTrigger

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _cb() -> MagicMock:
    return MagicMock(return_value=None)


def _disk_trigger(name: str = "disk1", threshold_pct: float = 0.0) -> ProactiveTrigger:
    return ProactiveTrigger(
        name=name,
        trigger_type="disk_threshold",
        disk_path="/",
        disk_threshold_pct=threshold_pct,
        interval_seconds=1,
        cooldown_seconds=0,
    )


def _load_trigger(name: str = "load1", threshold: float = 0.0) -> ProactiveTrigger:
    return ProactiveTrigger(
        name=name,
        trigger_type="load_threshold",
        load_threshold=threshold,
        interval_seconds=1,
        cooldown_seconds=0,
    )


def _schedule_trigger(name: str = "s1", cooldown: int = 0) -> ProactiveTrigger:
    return ProactiveTrigger(
        name=name,
        trigger_type="schedule",
        interval_seconds=1,
        cooldown_seconds=cooldown,
        prompt_template="tick {trigger_name}",
    )


# ---------------------------------------------------------------------------
# Lines 55-57: watchdog import SUCCESS path
# ---------------------------------------------------------------------------


class TestWatchdogImportSuccessPath:
    """Lines 55-57 are only executed when watchdog imports successfully.

    We force a reimport of the module with a fully-mocked watchdog in
    sys.modules so that the ``try`` branch completes and
    ``_WATCHDOG_AVAILABLE`` is set to ``True``.
    """

    def _inject_mock_watchdog(self) -> tuple[types.ModuleType, types.ModuleType]:
        """Return (mock_observers_mod, mock_events_mod) after injecting into sys.modules."""
        mock_pmeh = MagicMock()
        mock_pmeh.__name__ = "PatternMatchingEventHandler"

        mock_observer_cls = MagicMock()
        mock_observer_cls.__name__ = "Observer"

        mock_events_mod = types.ModuleType("watchdog.events")
        mock_events_mod.PatternMatchingEventHandler = mock_pmeh

        mock_observers_mod = types.ModuleType("watchdog.observers")
        mock_observers_mod.Observer = mock_observer_cls

        mock_watchdog_mod = types.ModuleType("watchdog")
        mock_watchdog_mod.events = mock_events_mod
        mock_watchdog_mod.observers = mock_observers_mod

        sys.modules["watchdog"] = mock_watchdog_mod
        sys.modules["watchdog.events"] = mock_events_mod
        sys.modules["watchdog.observers"] = mock_observers_mod

        return mock_observers_mod, mock_events_mod

    def test_watchdog_available_flag_is_true_when_import_succeeds(self) -> None:
        """When watchdog is importable, _WATCHDOG_AVAILABLE must be True."""
        saved = {
            k: sys.modules.pop(k)
            for k in ("missy.agent.proactive", "watchdog", "watchdog.events", "watchdog.observers")
            if k in sys.modules
        }
        try:
            self._inject_mock_watchdog()
            mod = importlib.import_module("missy.agent.proactive")
            assert mod._WATCHDOG_AVAILABLE is True
        finally:
            for k in ("missy.agent.proactive", "watchdog", "watchdog.events", "watchdog.observers"):
                sys.modules.pop(k, None)
            sys.modules.update(saved)

    def test_observer_name_imported_into_module_namespace(self) -> None:
        """After a successful watchdog import, Observer is accessible in the module."""
        saved = {
            k: sys.modules.pop(k)
            for k in ("missy.agent.proactive", "watchdog", "watchdog.events", "watchdog.observers")
            if k in sys.modules
        }
        try:
            self._inject_mock_watchdog()
            mod = importlib.import_module("missy.agent.proactive")
            # Observer must be a name in the module's namespace.
            assert hasattr(mod, "Observer")
        finally:
            for k in ("missy.agent.proactive", "watchdog", "watchdog.events", "watchdog.observers"):
                sys.modules.pop(k, None)
            sys.modules.update(saved)

    def test_pattern_matching_event_handler_imported(self) -> None:
        """PatternMatchingEventHandler is present in the module after successful import."""
        saved = {
            k: sys.modules.pop(k)
            for k in ("missy.agent.proactive", "watchdog", "watchdog.events", "watchdog.observers")
            if k in sys.modules
        }
        try:
            self._inject_mock_watchdog()
            mod = importlib.import_module("missy.agent.proactive")
            assert hasattr(mod, "PatternMatchingEventHandler")
        finally:
            for k in ("missy.agent.proactive", "watchdog", "watchdog.events", "watchdog.observers"):
                sys.modules.pop(k, None)
            sys.modules.update(saved)


# ---------------------------------------------------------------------------
# Line 302: inner is_set() break in _threshold_loop with multiple triggers
# ---------------------------------------------------------------------------


class TestThresholdLoopInnerBreak:
    """Line 302: ``if self._stop_event.is_set(): break`` fires when stop is
    detected between triggers in the same polling iteration.

    The existing tests cover single-trigger iterations but do not exercise the
    path where the inner ``is_set()`` check breaks out of the *for* loop with
    multiple triggers queued.
    """

    def test_inner_break_exits_for_loop_when_stop_set_between_triggers(self) -> None:
        """Second trigger must NOT be processed when stop fires after the first."""
        fired_names: list[str] = []

        t1 = _disk_trigger(name="first", threshold_pct=0.0)
        t2 = _disk_trigger(name="second", threshold_pct=0.0)
        mgr = ProactiveManager(triggers=[t1, t2], agent_callback=_cb())

        original_fire = mgr._fire_trigger

        def _fire_and_stop(trigger: ProactiveTrigger) -> None:
            fired_names.append(trigger.name)
            # After the first trigger fires, set the stop event so that the
            # inner is_set() check on the next iteration of the for loop
            # triggers the break on line 302.
            mgr._stop_event.set()
            original_fire(trigger)

        mgr._fire_trigger = _fire_and_stop

        mock_usage = MagicMock()
        mock_usage.used = 99
        mock_usage.total = 100  # 99% > 0% — always exceeds threshold

        # wait() returns False once (body executes), then True (loop exits).
        # After the first trigger sets stop_event, is_set() returns True and
        # the inner break fires before the second trigger is reached.
        with (
            patch.object(mgr._stop_event, "wait", side_effect=[False, True]),
            patch("shutil.disk_usage", return_value=mock_usage),
        ):
            mgr._threshold_loop([t1, t2])

        assert "first" in fired_names
        assert "second" not in fired_names, (
            "Second trigger should be skipped due to inner is_set() break"
        )

    def test_inner_break_does_not_fire_when_stop_not_set(self) -> None:
        """Both triggers are processed when stop_event is not set during iteration."""
        fired_names: list[str] = []

        t1 = _disk_trigger(name="first2", threshold_pct=0.0)
        t2 = _disk_trigger(name="second2", threshold_pct=0.0)
        mgr = ProactiveManager(triggers=[t1, t2], agent_callback=_cb())

        original_fire = mgr._fire_trigger

        def _record_fire(trigger: ProactiveTrigger) -> None:
            fired_names.append(trigger.name)
            original_fire(trigger)

        mgr._fire_trigger = _record_fire

        mock_usage = MagicMock()
        mock_usage.used = 99
        mock_usage.total = 100

        with (
            patch.object(mgr._stop_event, "wait", side_effect=[False, True]),
            patch("shutil.disk_usage", return_value=mock_usage),
        ):
            mgr._threshold_loop([t1, t2])

        assert "first2" in fired_names
        assert "second2" in fired_names


# ---------------------------------------------------------------------------
# Lines 429-430: event_bus.publish() call and the except fallback in _emit_audit
# ---------------------------------------------------------------------------


class TestEmitAuditPublish:
    """Lines 429-430: event_bus.publish() is called (line 428/429) and the
    except clause that swallows publish errors (line 430) is exercised.
    """

    def test_emit_audit_calls_event_bus_publish(self) -> None:
        """_emit_audit must call event_bus.publish with an AuditEvent."""
        t = _schedule_trigger(name="audit_pub")
        mgr = ProactiveManager(triggers=[t], agent_callback=_cb())

        with patch("missy.core.events.event_bus") as mock_bus:
            mgr._emit_audit(t, "allow", "test prompt", {})

        mock_bus.publish.assert_called_once()

    def test_emit_audit_publish_exception_is_swallowed(self) -> None:
        """When event_bus.publish raises, _emit_audit must not propagate the error."""
        t = _schedule_trigger(name="audit_exc")
        mgr = ProactiveManager(triggers=[t], agent_callback=_cb())

        with patch("missy.core.events.event_bus") as mock_bus:
            mock_bus.publish.side_effect = RuntimeError("bus exploded")
            # Must not raise.
            mgr._emit_audit(t, "allow", "test prompt", {})

        mock_bus.publish.assert_called_once()

    def test_emit_audit_publish_exception_debug_logged(self) -> None:
        """A debug message is emitted when event_bus.publish raises."""
        t = _schedule_trigger(name="audit_log_exc")
        mgr = ProactiveManager(triggers=[t], agent_callback=_cb())

        with (
            patch("missy.core.events.event_bus") as mock_bus,
            patch("missy.agent.proactive.logger") as mock_log,
        ):
            mock_bus.publish.side_effect = OSError("publish failed")
            mgr._emit_audit(t, "deny", "a prompt", {"reason": "test"})

        debug_calls = [str(c) for c in mock_log.debug.call_args_list]
        assert any("audit emit failed" in c for c in debug_calls)

    def test_emit_audit_audit_event_import_failure_swallowed(self) -> None:
        """If the import of AuditEvent itself fails, _emit_audit swallows it."""
        t = _schedule_trigger(name="audit_import_fail")
        mgr = ProactiveManager(triggers=[t], agent_callback=_cb())

        # Temporarily hide the events module so the import inside _emit_audit raises.
        saved = sys.modules.pop("missy.core.events", None)
        try:
            sys.modules["missy.core.events"] = None  # type: ignore[assignment]
            # Must not raise.
            mgr._emit_audit(t, "allow", "test prompt", {})
        finally:
            sys.modules.pop("missy.core.events", None)
            if saved is not None:
                sys.modules["missy.core.events"] = saved

    def test_fire_trigger_publishes_audit_event_on_allow(self) -> None:
        """_fire_trigger calls event_bus.publish (lines 428-429) for an allow result."""
        t = _schedule_trigger(name="fire_pub", cooldown=0)
        mgr = ProactiveManager(triggers=[t], agent_callback=_cb())

        with patch("missy.core.events.event_bus") as mock_bus:
            mgr._fire_trigger(t)

        mock_bus.publish.assert_called_once()
        published = mock_bus.publish.call_args[0][0]
        assert published.result == "allow"

    def test_fire_trigger_publishes_audit_event_on_deny(self) -> None:
        """_fire_trigger calls event_bus.publish for a deny result (no approval gate)."""
        t = ProactiveTrigger(
            name="fire_deny_pub",
            trigger_type="schedule",
            requires_confirmation=True,
            cooldown_seconds=0,
        )
        mgr = ProactiveManager(triggers=[t], agent_callback=_cb(), approval_gate=None)

        with patch("missy.core.events.event_bus") as mock_bus:
            mgr._fire_trigger(t)

        mock_bus.publish.assert_called_once()
        published = mock_bus.publish.call_args[0][0]
        assert published.result == "deny"


# ---------------------------------------------------------------------------
# Lines 440-454: _ProactiveFileHandler class body with mocked watchdog
# ---------------------------------------------------------------------------


class TestProactiveFileHandlerWithMockedWatchdog:
    """Lines 440-454 are inside ``if _WATCHDOG_AVAILABLE:`` and are therefore
    only evaluated at module parse time when watchdog is importable.

    We reimport the module with a mocked watchdog in sys.modules so that the
    class body is actually executed (not skipped by the else-branch stub).
    """

    def _reimport_with_watchdog(self) -> types.ModuleType:
        """Inject a mock watchdog and reimport proactive, returning the new module."""
        # Build a minimal mock PatternMatchingEventHandler that behaves like the real one.
        class MockPMEH:
            """Minimal stand-in for PatternMatchingEventHandler."""

            def __init__(self, patterns=None, ignore_directories=False, case_sensitive=True):
                self.patterns = patterns
                self.ignore_directories = ignore_directories
                self.case_sensitive = case_sensitive

        mock_observer_cls = MagicMock()
        mock_observer_cls.__name__ = "Observer"

        mock_events_mod = types.ModuleType("watchdog.events")
        mock_events_mod.PatternMatchingEventHandler = MockPMEH

        mock_observers_mod = types.ModuleType("watchdog.observers")
        mock_observers_mod.Observer = mock_observer_cls

        mock_watchdog_mod = types.ModuleType("watchdog")
        mock_watchdog_mod.events = mock_events_mod
        mock_watchdog_mod.observers = mock_observers_mod

        # Evict any cached module so the reimport executes the module body fresh.
        for k in ("missy.agent.proactive", "watchdog", "watchdog.events", "watchdog.observers"):
            sys.modules.pop(k, None)

        sys.modules["watchdog"] = mock_watchdog_mod
        sys.modules["watchdog.events"] = mock_events_mod
        sys.modules["watchdog.observers"] = mock_observers_mod

        mod = importlib.import_module("missy.agent.proactive")
        return mod

    def _cleanup(self, original: dict) -> None:
        for k in ("missy.agent.proactive", "watchdog", "watchdog.events", "watchdog.observers"):
            sys.modules.pop(k, None)
        sys.modules.update(original)

    def test_file_handler_class_is_subclass_of_mock_pmeh(self) -> None:
        """_ProactiveFileHandler inherits from PatternMatchingEventHandler."""
        saved = {
            k: sys.modules.pop(k)
            for k in ("missy.agent.proactive", "watchdog", "watchdog.events", "watchdog.observers")
            if k in sys.modules
        }
        try:
            mod = self._reimport_with_watchdog()
            assert mod._WATCHDOG_AVAILABLE is True
            # The handler must be a class (not the stub pass-through).
            assert isinstance(mod._ProactiveFileHandler, type)
            # It must inherit from whatever PatternMatchingEventHandler was mocked to.
            pmeh = sys.modules["watchdog.events"].PatternMatchingEventHandler
            assert issubclass(mod._ProactiveFileHandler, pmeh)
        finally:
            self._cleanup(saved)

    def test_file_handler_init_stores_trigger_and_fire_fn(self) -> None:
        """__init__ sets _trigger and _fire_fn attributes (lines 450-451)."""
        saved = {
            k: sys.modules.pop(k)
            for k in ("missy.agent.proactive", "watchdog", "watchdog.events", "watchdog.observers")
            if k in sys.modules
        }
        try:
            mod = self._reimport_with_watchdog()
            t = ProactiveTrigger(name="h1", trigger_type="schedule")
            fn = MagicMock()

            handler = mod._ProactiveFileHandler(
                trigger=t,
                fire_fn=fn,
                patterns=["*.log"],
                ignore_directories=True,
                case_sensitive=False,
            )

            assert handler._trigger is t
            assert handler._fire_fn is fn
        finally:
            self._cleanup(saved)

    def test_file_handler_on_any_event_calls_fire_fn(self) -> None:
        """on_any_event calls fire_fn with the stored trigger (line 454)."""
        saved = {
            k: sys.modules.pop(k)
            for k in ("missy.agent.proactive", "watchdog", "watchdog.events", "watchdog.observers")
            if k in sys.modules
        }
        try:
            mod = self._reimport_with_watchdog()
            t = ProactiveTrigger(name="h2", trigger_type="schedule")
            received: list[ProactiveTrigger] = []

            handler = mod._ProactiveFileHandler(
                trigger=t,
                fire_fn=received.append,
                patterns=["*"],
                ignore_directories=True,
                case_sensitive=False,
            )

            fake_event = MagicMock()
            handler.on_any_event(fake_event)

            assert len(received) == 1
            assert received[0] is t
        finally:
            self._cleanup(saved)

    def test_file_handler_on_any_event_does_not_forward_fs_event(self) -> None:
        """fire_fn receives the ProactiveTrigger, NOT the filesystem event object."""
        saved = {
            k: sys.modules.pop(k)
            for k in ("missy.agent.proactive", "watchdog", "watchdog.events", "watchdog.observers")
            if k in sys.modules
        }
        try:
            mod = self._reimport_with_watchdog()
            t = ProactiveTrigger(name="h3", trigger_type="schedule")
            received: list = []

            handler = mod._ProactiveFileHandler(
                trigger=t,
                fire_fn=received.append,
                patterns=["*"],
                ignore_directories=False,
                case_sensitive=True,
            )

            fake_fs_event = MagicMock()
            handler.on_any_event(fake_fs_event)

            assert received[0] is not fake_fs_event
        finally:
            self._cleanup(saved)

    def test_file_handler_on_any_event_called_multiple_times(self) -> None:
        """fire_fn is called once per on_any_event invocation."""
        saved = {
            k: sys.modules.pop(k)
            for k in ("missy.agent.proactive", "watchdog", "watchdog.events", "watchdog.observers")
            if k in sys.modules
        }
        try:
            mod = self._reimport_with_watchdog()
            t = ProactiveTrigger(name="h4", trigger_type="schedule")
            fn = MagicMock()

            handler = mod._ProactiveFileHandler(
                trigger=t,
                fire_fn=fn,
                patterns=["*"],
                ignore_directories=True,
                case_sensitive=False,
            )

            for _ in range(4):
                handler.on_any_event(MagicMock())

            assert fn.call_count == 4
        finally:
            self._cleanup(saved)

    def test_file_handler_wired_through_start_with_watchdog_available(self) -> None:
        """start() instantiates _ProactiveFileHandler and schedules it via observer."""
        saved = {
            k: sys.modules.pop(k)
            for k in ("missy.agent.proactive", "watchdog", "watchdog.events", "watchdog.observers")
            if k in sys.modules
        }
        try:
            mod = self._reimport_with_watchdog()

            trigger = mod.ProactiveTrigger(
                name="wired_h",
                trigger_type="file_change",
                watch_path="/tmp",
                watch_patterns=["*.txt"],
                watch_recursive=False,
                cooldown_seconds=0,
            )

            mock_obs = MagicMock()
            mock_obs_cls = MagicMock(return_value=mock_obs)

            # Patch Observer on the freshly-imported module.
            with patch.object(mod, "Observer", mock_obs_cls):
                mgr = mod.ProactiveManager(triggers=[trigger], agent_callback=_cb())
                mgr.start()

            mock_obs.schedule.assert_called_once()
            _, kwargs = mock_obs.schedule.call_args
            assert kwargs["path"] == "/tmp"
            assert kwargs["recursive"] is False
            mock_obs.start.assert_called_once()
            mgr.stop()
        finally:
            self._cleanup(saved)
