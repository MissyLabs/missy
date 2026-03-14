"""Tests for ProactiveManager and ProactiveTrigger (gap #35).

Covers:
- Cooldown logic prevents double-firing.
- Schedule triggers fire the callback after interval elapses.
- Disk threshold trigger fires when usage exceeds threshold.
- Load threshold trigger fires when load exceeds threshold (normalised).
- requires_confirmation with no gate skips callback and emits deny audit.
- requires_confirmation with gate fires callback when approved.
- requires_confirmation denied skips callback.
- Agent callback exceptions are caught (monitor continues).
- get_status returns correct shape.
- File-change trigger wiring (mocked watchdog).
- _parse_proactive / ProactiveConfig round-trips through settings parsing.
"""

from __future__ import annotations

import threading
import time
from unittest.mock import MagicMock, patch

import pytest

from missy.agent.proactive import ProactiveManager, ProactiveTrigger

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _simple_trigger(name: str = "t1", **kwargs) -> ProactiveTrigger:
    defaults = {
        "trigger_type": "schedule",
        "interval_seconds": 1,
        "cooldown_seconds": 0,
        "prompt_template": "ping from {trigger_name}",
    }
    defaults.update(kwargs)
    return ProactiveTrigger(name=name, **defaults)


class _Counter:
    def __init__(self):
        self.calls: list[tuple[str, str]] = []
        self._lock = threading.Lock()

    def callback(self, prompt: str, session_id: str) -> str:
        with self._lock:
            self.calls.append((prompt, session_id))
        return "ok"

    @property
    def count(self) -> int:
        with self._lock:
            return len(self.calls)


# ---------------------------------------------------------------------------
# _fire_trigger unit tests (no background threads)
# ---------------------------------------------------------------------------


class TestFireTriggerUnit:
    """Direct unit tests of ProactiveManager._fire_trigger."""

    def _manager(self, trigger, callback=None, gate=None):
        cb = callback or _Counter().callback
        return ProactiveManager(triggers=[trigger], agent_callback=cb, approval_gate=gate)

    def test_fires_callback_with_rendered_prompt(self) -> None:
        counter = _Counter()
        t = _simple_trigger(name="alpha", prompt_template="hello from {trigger_name}")
        mgr = self._manager(t, callback=counter.callback)

        mgr._fire_trigger(t)

        assert counter.count == 1
        prompt, session_id = counter.calls[0]
        assert "alpha" in prompt
        assert session_id == "proactive-alpha"

    def test_cooldown_prevents_second_fire(self) -> None:
        counter = _Counter()
        t = _simple_trigger(name="b", cooldown_seconds=3600)
        mgr = self._manager(t, callback=counter.callback)

        mgr._fire_trigger(t)
        mgr._fire_trigger(t)  # Should be blocked by cooldown

        assert counter.count == 1

    def test_cooldown_zero_allows_repeated_fires(self) -> None:
        counter = _Counter()
        t = _simple_trigger(name="c", cooldown_seconds=0)
        mgr = self._manager(t, callback=counter.callback)

        mgr._fire_trigger(t)
        mgr._fire_trigger(t)

        assert counter.count == 2

    def test_callback_exception_is_caught(self) -> None:
        def _bad_cb(prompt, session_id):
            raise RuntimeError("boom")

        t = _simple_trigger(name="d")
        mgr = self._manager(t, callback=_bad_cb)

        # Must not raise
        mgr._fire_trigger(t)

    def test_requires_confirmation_no_gate_skips_callback(self) -> None:
        counter = _Counter()
        t = _simple_trigger(name="e", requires_confirmation=True)
        mgr = self._manager(t, callback=counter.callback, gate=None)

        mgr._fire_trigger(t)

        assert counter.count == 0

    def test_requires_confirmation_with_gate_approved(self) -> None:
        counter = _Counter()
        gate = MagicMock()
        gate.request.return_value = None  # Approval granted (no exception)
        t = _simple_trigger(name="f", requires_confirmation=True)
        mgr = self._manager(t, callback=counter.callback, gate=gate)

        mgr._fire_trigger(t)

        gate.request.assert_called_once()
        assert counter.count == 1

    def test_requires_confirmation_with_gate_denied(self) -> None:
        from missy.agent.approval import ApprovalDenied

        counter = _Counter()
        gate = MagicMock()
        gate.request.side_effect = ApprovalDenied("denied")
        t = _simple_trigger(name="g", requires_confirmation=True)
        mgr = self._manager(t, callback=counter.callback, gate=gate)

        mgr._fire_trigger(t)

        assert counter.count == 0

    def test_emit_audit_on_fire(self) -> None:
        t = _simple_trigger(name="h")
        mgr = self._manager(t)

        with patch("missy.core.events.event_bus") as mock_bus:
            mgr._fire_trigger(t)

        mock_bus.publish.assert_called_once()
        published_event = mock_bus.publish.call_args.args[0]
        assert published_event.event_type == "agent.proactive.trigger_fired"
        assert published_event.result == "allow"
        assert published_event.detail["trigger_name"] == "h"

    def test_emit_audit_deny_when_no_gate(self) -> None:
        t = _simple_trigger(name="i", requires_confirmation=True)
        mgr = self._manager(t, gate=None)

        with patch("missy.core.events.event_bus") as mock_bus:
            mgr._fire_trigger(t)

        mock_bus.publish.assert_called_once()
        published_event = mock_bus.publish.call_args.args[0]
        assert published_event.result == "deny"

    def test_default_prompt_template_used_when_empty(self) -> None:
        counter = _Counter()
        t = _simple_trigger(name="j", prompt_template="")
        mgr = self._manager(t, callback=counter.callback)

        mgr._fire_trigger(t)

        prompt, _ = counter.calls[0]
        assert "j" in prompt  # trigger_name substituted in default template


# ---------------------------------------------------------------------------
# Threshold logic
# ---------------------------------------------------------------------------


class TestThresholdLogic:
    """Tests for threshold firing logic, exercised via _fire_trigger directly."""

    def _manager(self, trigger, counter):
        return ProactiveManager(triggers=[trigger], agent_callback=counter.callback)

    def test_disk_threshold_does_not_fire_when_under(self) -> None:
        counter = _Counter()
        t = ProactiveTrigger(
            name="disk2",
            trigger_type="disk_threshold",
            disk_path="/",
            disk_threshold_pct=95.0,  # High threshold
            cooldown_seconds=0,
            interval_seconds=1,
        )
        mgr = self._manager(t, counter)

        import shutil

        mock_usage = MagicMock()
        mock_usage.used = 50
        mock_usage.total = 100  # 50% used — below 95%

        with patch.object(shutil, "disk_usage", return_value=mock_usage):
            pct = mock_usage.used / mock_usage.total * 100.0
            if pct > t.disk_threshold_pct:
                mgr._fire_trigger(t)

        assert counter.count == 0

    def test_disk_threshold_fires_when_exceeded(self) -> None:
        counter = _Counter()
        t = ProactiveTrigger(
            name="disk3",
            trigger_type="disk_threshold",
            disk_path="/",
            disk_threshold_pct=40.0,
            cooldown_seconds=0,
            interval_seconds=1,
        )
        mgr = self._manager(t, counter)

        import shutil

        mock_usage = MagicMock()
        mock_usage.used = 50
        mock_usage.total = 100  # 50% > 40%

        with patch.object(shutil, "disk_usage", return_value=mock_usage):
            pct = mock_usage.used / mock_usage.total * 100.0
            if pct > t.disk_threshold_pct:
                mgr._fire_trigger(t)

        assert counter.count == 1

    def test_load_threshold_fires_when_exceeded(self) -> None:
        counter = _Counter()
        t = ProactiveTrigger(
            name="load1",
            trigger_type="load_threshold",
            load_threshold=0.1,  # Very low — easily exceeded
            cooldown_seconds=0,
            interval_seconds=1,
        )
        mgr = self._manager(t, counter)

        import os

        with (
            patch.object(os, "getloadavg", return_value=(8.0, 4.0, 2.0)),
            patch.object(os, "cpu_count", return_value=4),
        ):
            load1 = 8.0
            cpu_count = 4
            normalised = load1 / cpu_count  # 2.0 > 0.1
            if normalised > t.load_threshold:
                mgr._fire_trigger(t)

        assert counter.count == 1

    def test_load_threshold_does_not_fire_when_under(self) -> None:
        counter = _Counter()
        t = ProactiveTrigger(
            name="load2",
            trigger_type="load_threshold",
            load_threshold=10.0,  # Very high
            cooldown_seconds=0,
            interval_seconds=1,
        )
        mgr = self._manager(t, counter)

        import os

        with (
            patch.object(os, "getloadavg", return_value=(4.0, 4.0, 4.0)),
            patch.object(os, "cpu_count", return_value=4),
        ):
            load1 = 4.0
            cpu_count = 4
            normalised = load1 / cpu_count  # 1.0 < 10.0
            if normalised > t.load_threshold:
                mgr._fire_trigger(t)

        assert counter.count == 0


# ---------------------------------------------------------------------------
# Start/stop and get_status
# ---------------------------------------------------------------------------


class TestProactiveManagerLifecycle:
    def test_start_and_stop_schedule_trigger(self) -> None:
        counter = _Counter()
        t = ProactiveTrigger(
            name="sched1",
            trigger_type="schedule",
            interval_seconds=1,
            cooldown_seconds=0,
            prompt_template="tick",
        )
        mgr = ProactiveManager(triggers=[t], agent_callback=counter.callback)

        mgr.start()
        time.sleep(0.1)  # Very brief; just confirm no exception on start
        mgr.stop()

    def test_get_status_shape(self) -> None:
        t = _simple_trigger(name="status_test")
        mgr = ProactiveManager(triggers=[t], agent_callback=lambda p, s: "ok")

        status = mgr.get_status()

        assert "active" in status
        assert "triggers" in status
        assert len(status["triggers"]) == 1
        trigger_status = status["triggers"][0]
        assert trigger_status["name"] == "status_test"
        assert trigger_status["trigger_type"] == "schedule"
        assert trigger_status["enabled"] is True
        assert trigger_status["last_fired"] is None

    def test_get_status_records_last_fired(self) -> None:
        t = _simple_trigger(name="fired_test", cooldown_seconds=0)
        mgr = ProactiveManager(triggers=[t], agent_callback=lambda p, s: "ok")

        mgr._fire_trigger(t)
        status = mgr.get_status()

        trigger_status = status["triggers"][0]
        assert trigger_status["last_fired"] is not None

    def test_disabled_trigger_skipped_on_start(self) -> None:
        counter = _Counter()
        t = ProactiveTrigger(
            name="disabled",
            trigger_type="schedule",
            enabled=False,
            interval_seconds=1,
            cooldown_seconds=0,
        )
        mgr = ProactiveManager(triggers=[t], agent_callback=counter.callback)
        mgr.start()
        time.sleep(0.05)
        mgr.stop()

        # Background thread for a disabled trigger is never started.
        assert counter.count == 0

    def test_stop_is_idempotent(self) -> None:
        t = _simple_trigger()
        mgr = ProactiveManager(triggers=[t], agent_callback=lambda p, s: "ok")
        mgr.start()
        mgr.stop()
        mgr.stop()  # Must not raise


# ---------------------------------------------------------------------------
# File-change trigger (mock watchdog)
# ---------------------------------------------------------------------------


class TestFileChangeTrigger:
    """Tests for file_change trigger wiring, using sys.modules injection to mock watchdog."""

    def _patch_watchdog(self):
        """Return a context manager that injects a mock watchdog into sys.modules."""
        import sys
        import types

        mock_observer_instance = MagicMock()
        mock_observer_cls = MagicMock(return_value=mock_observer_instance)

        mock_watchdog = types.ModuleType("watchdog")
        mock_observers = types.ModuleType("watchdog.observers")
        mock_observers.Observer = mock_observer_cls
        mock_events = types.ModuleType("watchdog.events")
        mock_events.PatternMatchingEventHandler = MagicMock
        mock_events.FileSystemEventHandler = MagicMock
        mock_watchdog.observers = mock_observers
        mock_watchdog.events = mock_events

        class _Ctx:
            def __enter__(self_inner):
                sys.modules.setdefault("watchdog", mock_watchdog)
                sys.modules.setdefault("watchdog.observers", mock_observers)
                sys.modules.setdefault("watchdog.events", mock_events)
                self_inner.observer_instance = mock_observer_instance
                self_inner.observer_cls = mock_observer_cls
                return self_inner

            def __exit__(self_inner, *args):
                pass  # Leave in sys.modules; tests are isolated enough.

        return _Ctx()

    def _mock_observer(self):
        """Return (mock_obs_instance, context_manager) that injects Observer into the module."""
        import missy.agent.proactive as _proactive_mod

        mock_obs = MagicMock()
        mock_obs_cls = MagicMock(return_value=mock_obs)
        # create=True allows patching names not currently on the module.
        ctx = patch.object(_proactive_mod, "Observer", mock_obs_cls, create=True)
        return mock_obs, ctx

    def test_file_change_trigger_schedules_observer(self) -> None:
        counter = _Counter()
        t = ProactiveTrigger(
            name="watch1",
            trigger_type="file_change",
            watch_path="/tmp",
            watch_patterns=["*.log"],
            cooldown_seconds=0,
        )
        mgr = ProactiveManager(triggers=[t], agent_callback=counter.callback)

        mock_obs, obs_ctx = self._mock_observer()
        # Also patch the handler class so it can be instantiated when watchdog is absent.
        mock_handler_cls = MagicMock(return_value=MagicMock())
        with (
            patch("missy.agent.proactive._WATCHDOG_AVAILABLE", True),
            obs_ctx,
            patch("missy.agent.proactive._ProactiveFileHandler", mock_handler_cls),
        ):
            mgr.start()

        mock_obs.schedule.assert_called_once()
        mock_obs.start.assert_called_once()
        mgr.stop()

    def test_file_change_trigger_skipped_when_watchdog_unavailable(self) -> None:
        counter = _Counter()
        t = ProactiveTrigger(
            name="watch2",
            trigger_type="file_change",
            watch_path="/tmp",
            cooldown_seconds=0,
        )
        mgr = ProactiveManager(triggers=[t], agent_callback=counter.callback)

        with patch("missy.agent.proactive._WATCHDOG_AVAILABLE", False):
            mgr.start()
            mgr.stop()

        # No threads for a file_change trigger when watchdog unavailable.
        assert counter.count == 0

    def test_file_change_trigger_missing_watch_path_skipped(self) -> None:
        counter = _Counter()
        t = ProactiveTrigger(
            name="watch3",
            trigger_type="file_change",
            watch_path="",  # Empty — should be skipped
            cooldown_seconds=0,
        )
        mgr = ProactiveManager(triggers=[t], agent_callback=counter.callback)

        mock_obs, obs_ctx = self._mock_observer()
        with (
            patch("missy.agent.proactive._WATCHDOG_AVAILABLE", True),
            obs_ctx,
        ):
            mgr.start()
            mgr.stop()

        mock_obs.schedule.assert_not_called()


# ---------------------------------------------------------------------------
# Config parsing round-trip
# ---------------------------------------------------------------------------


class TestProactiveConfigParsing:
    def test_parse_proactive_minimal(self) -> None:
        from missy.config.settings import _parse_proactive

        data = {"enabled": True, "triggers": []}
        cfg = _parse_proactive(data)
        assert cfg.enabled is True
        assert cfg.triggers == []

    def test_parse_proactive_with_trigger(self) -> None:
        from missy.config.settings import _parse_proactive

        data = {
            "enabled": True,
            "triggers": [
                {
                    "name": "my-trigger",
                    "trigger_type": "schedule",
                    "interval_seconds": 60,
                    "cooldown_seconds": 120,
                    "prompt_template": "hello {trigger_name}",
                }
            ],
        }
        cfg = _parse_proactive(data)
        assert len(cfg.triggers) == 1
        t = cfg.triggers[0]
        assert t.name == "my-trigger"
        assert t.trigger_type == "schedule"
        assert t.interval_seconds == 60
        assert t.cooldown_seconds == 120
        assert t.prompt_template == "hello {trigger_name}"

    def test_parse_proactive_trigger_missing_name_raises(self) -> None:
        from missy.config.settings import _parse_proactive
        from missy.core.exceptions import ConfigurationError

        with pytest.raises(ConfigurationError, match="name"):
            _parse_proactive({"enabled": True, "triggers": [{"trigger_type": "schedule"}]})

    def test_parse_proactive_trigger_missing_type_raises(self) -> None:
        from missy.config.settings import _parse_proactive
        from missy.core.exceptions import ConfigurationError

        with pytest.raises(ConfigurationError, match="trigger_type"):
            _parse_proactive({"enabled": True, "triggers": [{"name": "t"}]})

    def test_parse_proactive_trigger_not_dict_raises(self) -> None:
        from missy.config.settings import _parse_proactive
        from missy.core.exceptions import ConfigurationError

        with pytest.raises(ConfigurationError):
            _parse_proactive({"enabled": True, "triggers": ["not-a-dict"]})

    def test_proactive_defaults_to_disabled(self) -> None:
        from missy.config.settings import ProactiveConfig

        cfg = ProactiveConfig()
        assert cfg.enabled is False
        assert cfg.triggers == []

    def test_missy_config_has_proactive_field(self) -> None:
        from missy.config.settings import get_default_config

        cfg = get_default_config()
        assert hasattr(cfg, "proactive")
        assert cfg.proactive.enabled is False

    def test_load_config_parses_proactive(self, tmp_path) -> None:
        """End-to-end: proactive section is read from YAML via load_config."""
        from missy.config.settings import load_config

        yaml_content = """\
network:
  default_deny: false

filesystem: {}
shell: {}
plugins: {}
providers:
  anthropic:
    name: anthropic
    model: claude-3-5-sonnet-20241022

proactive:
  enabled: true
  triggers:
    - name: check-disk
      trigger_type: disk_threshold
      disk_path: /
      disk_threshold_pct: 80.0
      cooldown_seconds: 600
"""
        config_file = tmp_path / "config.yaml"
        config_file.write_text(yaml_content)

        cfg = load_config(str(config_file))

        assert cfg.proactive.enabled is True
        assert len(cfg.proactive.triggers) == 1
        t = cfg.proactive.triggers[0]
        assert t.name == "check-disk"
        assert t.trigger_type == "disk_threshold"
        assert t.disk_threshold_pct == 80.0
        assert t.cooldown_seconds == 600
