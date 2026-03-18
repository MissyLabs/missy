"""Tests for missy.agent.watchdog.Watchdog."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from missy.agent.watchdog import SubsystemHealth, Watchdog


class TestSubsystemHealth:
    """Tests for the SubsystemHealth dataclass."""

    def test_default_values(self):
        h = SubsystemHealth(name="test")
        assert h.healthy is True
        assert h.consecutive_failures == 0
        assert h.last_checked == 0.0
        assert h.last_error == ""

    def test_custom_values(self):
        h = SubsystemHealth(name="db", healthy=False, consecutive_failures=3, last_error="timeout")
        assert h.name == "db"
        assert h.healthy is False
        assert h.consecutive_failures == 3


class TestWatchdogRegister:
    """Tests for registering health checks."""

    def test_register_single(self):
        wd = Watchdog()
        wd.register("memory", lambda: True)
        assert "memory" in wd._checks
        assert "memory" in wd._health
        assert wd._health["memory"].healthy is True

    def test_register_multiple(self):
        wd = Watchdog()
        wd.register("a", lambda: True)
        wd.register("b", lambda: True)
        assert len(wd._checks) == 2

    def test_register_overwrites(self):
        wd = Watchdog()
        wd.register("x", lambda: True)
        wd.register("x", lambda: False)
        assert len(wd._checks) == 1


class TestWatchdogCheckAll:
    """Tests for _check_all() with mocked event bus."""

    @patch("missy.core.events.event_bus")
    def test_healthy_check(self, mock_bus):
        wd = Watchdog()
        wd.register("mem", lambda: True)
        wd._check_all()
        assert wd._health["mem"].healthy is True
        assert wd._health["mem"].consecutive_failures == 0

    @patch("missy.core.events.event_bus")
    def test_unhealthy_check_false(self, mock_bus):
        wd = Watchdog()
        wd.register("mem", lambda: False)
        wd._check_all()
        assert wd._health["mem"].healthy is False
        assert wd._health["mem"].consecutive_failures == 1

    @patch("missy.core.events.event_bus")
    def test_unhealthy_check_exception(self, mock_bus):
        def bad_check():
            raise ConnectionError("db down")

        wd = Watchdog()
        wd.register("db", bad_check)
        wd._check_all()
        assert wd._health["db"].healthy is False
        assert wd._health["db"].last_error == "db down"

    @patch("missy.core.events.event_bus")
    def test_consecutive_failures_increment(self, mock_bus):
        wd = Watchdog(failure_threshold=3)
        wd.register("net", lambda: False)
        for _ in range(5):
            wd._check_all()
        assert wd._health["net"].consecutive_failures == 5

    @patch("missy.core.events.event_bus")
    def test_recovery_resets_failures(self, mock_bus):
        healthy = [False, False, True]
        idx = [0]

        def check_fn():
            result = healthy[idx[0]]
            idx[0] = min(idx[0] + 1, len(healthy) - 1)
            return result

        wd = Watchdog()
        wd.register("svc", check_fn)
        wd._check_all()  # fail
        wd._check_all()  # fail
        assert wd._health["svc"].consecutive_failures == 2
        wd._check_all()  # recover
        assert wd._health["svc"].healthy is True
        assert wd._health["svc"].consecutive_failures == 0
        assert wd._health["svc"].last_error == ""

    @patch("missy.core.events.event_bus")
    def test_audit_event_publish_failure_handled(self, mock_bus):
        """Audit event bus failure should not crash the watchdog."""
        mock_bus.publish.side_effect = RuntimeError("event bus broken")
        wd = Watchdog()
        wd.register("mem", lambda: True)
        # Should not raise
        wd._check_all()

    @patch("missy.core.events.event_bus")
    def test_multiple_subsystems(self, mock_bus):
        wd = Watchdog()
        wd.register("a", lambda: True)
        wd.register("b", lambda: False)
        wd._check_all()
        assert wd._health["a"].healthy is True
        assert wd._health["b"].healthy is False


class TestWatchdogReport:
    """Tests for get_report()."""

    def test_empty_report(self):
        wd = Watchdog()
        assert wd.get_report() == {}

    @patch("missy.core.events.event_bus")
    def test_report_structure(self, mock_bus):
        wd = Watchdog()
        wd.register("mem", lambda: True)
        wd.register("net", lambda: False)
        wd._check_all()
        report = wd.get_report()
        assert "mem" in report
        assert report["mem"]["healthy"] is True
        assert report["net"]["healthy"] is False
        assert "consecutive_failures" in report["net"]
        assert "last_error" in report["net"]


class TestWatchdogStartStop:
    """Thread lifecycle tests."""

    def test_start_creates_thread(self):
        wd = Watchdog(check_interval=3600)
        wd.start()
        assert wd._thread is not None
        assert wd._thread.daemon is True
        assert wd._thread.name == "missy-watchdog"
        wd.stop()

    def test_stop_without_start(self):
        wd = Watchdog()
        wd.stop()  # Should not raise

    def test_stop_sets_event(self):
        wd = Watchdog(check_interval=3600)
        wd.start()
        wd.stop()
        assert wd._stop.is_set()
