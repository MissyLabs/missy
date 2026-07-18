"""Tests for F23 — VisionMonitor continuous monitoring."""

from __future__ import annotations

import pytest

np = pytest.importorskip("numpy")

from missy.vision.monitor import VisionMonitor  # noqa: E402


def _black():
    return np.zeros((32, 32, 3), dtype=np.uint8)


def _white():
    return np.full((32, 32, 3), 255, dtype=np.uint8)


def _monitor(frames, alerts, **kw):
    it = iter(frames)
    kw.setdefault("threshold", 0.15)
    kw.setdefault("min_alert_gap_seconds", 0.0)
    return VisionMonitor(lambda: next(it, None), alerts.append, **kw)


class TestChangeDetection:
    def test_first_frame_no_alert(self) -> None:
        alerts: list = []
        mon = _monitor([_black()], alerts)
        assert mon.check_once() is None
        assert alerts == []

    def test_identical_frames_no_alert(self) -> None:
        alerts: list = []
        mon = _monitor([_black(), _black()], alerts)
        mon.check_once()
        assert mon.check_once() is None
        assert alerts == []

    def test_big_change_alerts(self) -> None:
        alerts: list = []
        mon = _monitor([_black(), _white()], alerts)
        mon.check_once()  # prime
        change = mon.check_once()
        assert change is not None
        assert change.change_score >= 0.15
        assert len(alerts) == 1

    def test_high_threshold_suppresses(self) -> None:
        alerts: list = []
        mon = _monitor([_black(), _white()], alerts, threshold=1.5)  # unreachable
        mon.check_once()
        assert mon.check_once() is None
        assert alerts == []


class TestCooldown:
    def test_cooldown_suppresses_second_alert(self) -> None:
        alerts: list = []
        mon = _monitor([_black(), _white(), _black(), _white()], alerts, min_alert_gap_seconds=999)
        mon.check_once()  # prime
        assert mon.check_once() is not None  # first alert
        mon.check_once()
        assert mon.check_once() is None  # within cooldown -> suppressed
        assert len(alerts) == 1

    def test_zero_cooldown_allows_repeated(self) -> None:
        alerts: list = []
        mon = _monitor([_black(), _white(), _black(), _white()], alerts, min_alert_gap_seconds=0)
        for _ in range(4):
            mon.check_once()
        assert len(alerts) >= 2


class TestRobustness:
    def test_none_capture_skipped(self) -> None:
        alerts: list = []
        mon = _monitor([None, None], alerts)
        assert mon.check_once() is None
        assert mon.checks == 1

    def test_capture_exception_skipped(self) -> None:
        alerts: list = []

        def _boom():
            raise RuntimeError("camera error")

        mon = VisionMonitor(_boom, alerts.append)
        assert mon.check_once() is None  # swallowed
        assert alerts == []

    def test_callback_exception_isolated(self) -> None:
        def _bad_cb(change):
            raise RuntimeError("notify failed")

        it = iter([_black(), _white()])
        mon = VisionMonitor(
            lambda: next(it, None), _bad_cb, threshold=0.15, min_alert_gap_seconds=0
        )
        mon.check_once()
        # The callback raises but check_once must not propagate.
        change = mon.check_once()
        assert change is not None
        assert mon.alerts == 1


class TestStatsAndLifecycle:
    def test_stats(self) -> None:
        alerts: list = []
        mon = _monitor([_black(), _white()], alerts)
        mon.check_once()
        mon.check_once()
        s = mon.stats()
        assert s["checks"] == 2
        assert s["alerts"] == 1
        assert s["threshold"] == 0.15

    def test_start_stop(self) -> None:
        alerts: list = []
        mon = _monitor([_black()], alerts, interval_seconds=9999)
        mon.start()
        assert mon._thread is not None and mon._thread.is_alive()
        mon.stop(timeout=2.0)

    def test_stop_without_start_is_safe(self) -> None:
        VisionMonitor(_black, lambda c: None).stop()
