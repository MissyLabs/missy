"""Tests for missy.vision.health_monitor.VisionHealthMonitor."""

from __future__ import annotations

import threading
import time

import pytest

from missy.vision.health_monitor import (
    CaptureEvent,
    DeviceStats,
    HealthStatus,
    VisionHealthMonitor,
    get_health_monitor,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def monitor():
    return VisionHealthMonitor(max_events=100, recent_window_secs=60)


# ---------------------------------------------------------------------------
# HealthStatus enum
# ---------------------------------------------------------------------------


class TestHealthStatus:
    def test_values(self):
        assert HealthStatus.HEALTHY == "healthy"
        assert HealthStatus.DEGRADED == "degraded"
        assert HealthStatus.UNHEALTHY == "unhealthy"
        assert HealthStatus.UNKNOWN == "unknown"


# ---------------------------------------------------------------------------
# CaptureEvent dataclass
# ---------------------------------------------------------------------------


class TestCaptureEvent:
    def test_defaults(self):
        evt = CaptureEvent(timestamp=1.0, success=True)
        assert evt.device == ""
        assert evt.quality_score == 0.0
        assert evt.error == ""
        assert evt.latency_ms == 0.0
        assert evt.source_type == "webcam"

    def test_all_fields(self):
        evt = CaptureEvent(
            timestamp=1.0,
            success=False,
            device="/dev/video0",
            quality_score=0.85,
            error="timeout",
            latency_ms=150.0,
            source_type="file",
        )
        assert evt.device == "/dev/video0"
        assert evt.error == "timeout"


# ---------------------------------------------------------------------------
# DeviceStats
# ---------------------------------------------------------------------------


class TestDeviceStats:
    def test_success_rate_zero_captures(self):
        stats = DeviceStats(device="/dev/video0")
        assert stats.success_rate == 0.0

    def test_success_rate_all_success(self):
        stats = DeviceStats(device="d", total_captures=10, successful_captures=10)
        assert stats.success_rate == 1.0

    def test_success_rate_mixed(self):
        stats = DeviceStats(
            device="d", total_captures=10, successful_captures=7, failed_captures=3
        )
        assert stats.success_rate == pytest.approx(0.7)

    def test_average_quality_no_success(self):
        stats = DeviceStats(device="d", total_captures=5, successful_captures=0)
        assert stats.average_quality == 0.0

    def test_average_quality_computed(self):
        stats = DeviceStats(
            device="d", successful_captures=4, total_quality=3.2
        )
        assert stats.average_quality == pytest.approx(0.8)

    def test_average_latency_no_captures(self):
        stats = DeviceStats(device="d")
        assert stats.average_latency_ms == 0.0

    def test_average_latency_computed(self):
        stats = DeviceStats(device="d", total_captures=5, total_latency_ms=500.0)
        assert stats.average_latency_ms == pytest.approx(100.0)


# ---------------------------------------------------------------------------
# VisionHealthMonitor: record_capture
# ---------------------------------------------------------------------------


class TestRecordCapture:
    def test_record_success(self, monitor):
        monitor.record_capture(success=True, device="/dev/video0", quality_score=0.9)
        stats = monitor.get_device_stats("/dev/video0")
        assert stats is not None
        assert stats.total_captures == 1
        assert stats.successful_captures == 1
        assert stats.failed_captures == 0

    def test_record_failure(self, monitor):
        monitor.record_capture(success=False, device="/dev/video0", error="blank")
        stats = monitor.get_device_stats("/dev/video0")
        assert stats.failed_captures == 1
        assert stats.last_error == "blank"

    def test_consecutive_failures_tracked(self, monitor):
        for i in range(5):
            monitor.record_capture(success=False, device="d", error=f"err{i}")
        stats = monitor.get_device_stats("d")
        assert stats.consecutive_failures == 5

    def test_success_resets_consecutive_failures(self, monitor):
        for _ in range(3):
            monitor.record_capture(success=False, device="d", error="x")
        monitor.record_capture(success=True, device="d")
        stats = monitor.get_device_stats("d")
        assert stats.consecutive_failures == 0

    def test_quality_accumulated(self, monitor):
        monitor.record_capture(success=True, device="d", quality_score=0.8)
        monitor.record_capture(success=True, device="d", quality_score=0.6)
        stats = monitor.get_device_stats("d")
        assert stats.average_quality == pytest.approx(0.7)

    def test_latency_accumulated(self, monitor):
        monitor.record_capture(success=True, device="d", latency_ms=100)
        monitor.record_capture(success=True, device="d", latency_ms=200)
        stats = monitor.get_device_stats("d")
        assert stats.average_latency_ms == pytest.approx(150.0)

    def test_multiple_devices_independent(self, monitor):
        monitor.record_capture(success=True, device="a")
        monitor.record_capture(success=False, device="b", error="err")
        assert monitor.get_device_stats("a").successful_captures == 1
        assert monitor.get_device_stats("b").failed_captures == 1

    def test_source_type_stored_in_event(self, monitor):
        monitor.record_capture(success=True, device="d", source_type="screenshot")
        report = monitor.get_health_report()
        assert report["total_captures"] == 1

    def test_max_events_respected(self):
        m = VisionHealthMonitor(max_events=5)
        for _ in range(10):
            m.record_capture(success=True, device="d")
        report = m.get_health_report()
        assert report["total_captures"] == 5

    def test_min_max_events_is_one(self):
        m = VisionHealthMonitor(max_events=0)
        m.record_capture(success=True, device="d")
        report = m.get_health_report()
        assert report["total_captures"] == 1


# ---------------------------------------------------------------------------
# record_device_discovery
# ---------------------------------------------------------------------------


class TestRecordDeviceDiscovery:
    def test_creates_device_stats(self, monitor):
        monitor.record_device_discovery("/dev/video2")
        stats = monitor.get_device_stats("/dev/video2")
        assert stats is not None
        assert stats.total_captures == 0
        assert stats.last_seen > 0

    def test_updates_last_seen(self, monitor):
        monitor.record_device_discovery("d")
        t1 = monitor.get_device_stats("d").last_seen
        time.sleep(0.01)
        monitor.record_device_discovery("d")
        t2 = monitor.get_device_stats("d").last_seen
        assert t2 >= t1


# ---------------------------------------------------------------------------
# get_device_health
# ---------------------------------------------------------------------------


class TestGetDeviceHealth:
    def test_unknown_device(self, monitor):
        assert monitor.get_device_health("nonexistent") == HealthStatus.UNKNOWN

    def test_no_captures_is_unknown(self, monitor):
        monitor.record_device_discovery("d")
        assert monitor.get_device_health("d") == HealthStatus.UNKNOWN

    def test_healthy_device(self, monitor):
        for _ in range(10):
            monitor.record_capture(success=True, device="d")
        assert monitor.get_device_health("d") == HealthStatus.HEALTHY

    def test_degraded_device(self, monitor):
        for _ in range(6):
            monitor.record_capture(success=True, device="d")
        for _ in range(4):
            monitor.record_capture(success=False, device="d", error="x")
        # 60% success rate is between 50% (unhealthy) and 80% (degraded)
        assert monitor.get_device_health("d") == HealthStatus.DEGRADED

    def test_unhealthy_device_low_rate(self, monitor):
        for _ in range(3):
            monitor.record_capture(success=True, device="d")
        for _ in range(7):
            monitor.record_capture(success=False, device="d", error="x")
        # 30% success rate
        assert monitor.get_device_health("d") == HealthStatus.UNHEALTHY

    def test_unhealthy_consecutive_failures(self, monitor):
        for _ in range(5):
            monitor.record_capture(success=True, device="d")
        for _ in range(5):
            monitor.record_capture(success=False, device="d", error="x")
        # 50% rate but 5 consecutive failures triggers unhealthy
        assert monitor.get_device_health("d") == HealthStatus.UNHEALTHY


# ---------------------------------------------------------------------------
# get_overall_health
# ---------------------------------------------------------------------------


class TestGetOverallHealth:
    def test_no_devices_is_unknown(self, monitor):
        assert monitor.get_overall_health() == HealthStatus.UNKNOWN

    def test_all_healthy(self, monitor):
        for _ in range(10):
            monitor.record_capture(success=True, device="a")
        for _ in range(10):
            monitor.record_capture(success=True, device="b")
        assert monitor.get_overall_health() == HealthStatus.HEALTHY

    def test_one_unhealthy_makes_overall_unhealthy(self, monitor):
        for _ in range(10):
            monitor.record_capture(success=True, device="a")
        for _ in range(10):
            monitor.record_capture(success=False, device="b", error="x")
        assert monitor.get_overall_health() == HealthStatus.UNHEALTHY

    def test_degraded_propagates(self, monitor):
        for _ in range(10):
            monitor.record_capture(success=True, device="a")
        for _ in range(7):
            monitor.record_capture(success=True, device="b")
        for _ in range(3):
            monitor.record_capture(success=False, device="b", error="x")
        assert monitor.get_overall_health() == HealthStatus.DEGRADED


# ---------------------------------------------------------------------------
# get_health_report
# ---------------------------------------------------------------------------


class TestGetHealthReport:
    def test_empty_report(self, monitor):
        report = monitor.get_health_report()
        assert report["overall_status"] == "unknown"
        assert report["total_captures"] == 0
        assert report["total_failures"] == 0
        assert report["recent_success_rate"] == 0.0
        assert report["devices"] == {}
        assert report["uptime_seconds"] >= 0
        assert report["warnings"] == []

    def test_report_with_captures(self, monitor):
        monitor.record_capture(success=True, device="d", quality_score=0.9, latency_ms=50)
        monitor.record_capture(success=False, device="d", error="blank")
        report = monitor.get_health_report()
        assert report["total_captures"] == 2
        assert report["total_failures"] == 1
        assert "d" in report["devices"]
        assert report["devices"]["d"]["total_captures"] == 2
        assert report["devices"]["d"]["success_rate"] == 0.5

    def test_report_includes_warnings(self, monitor):
        for _ in range(5):
            monitor.record_capture(success=False, device="d", error="timeout")
        report = monitor.get_health_report()
        assert len(report["warnings"]) > 0
        assert "consecutive" in report["warnings"][0].lower()

    def test_low_quality_warning(self, monitor):
        for _ in range(5):
            monitor.record_capture(success=True, device="d", quality_score=0.2)
        report = monitor.get_health_report()
        quality_warnings = [w for w in report["warnings"] if "quality" in w.lower()]
        assert len(quality_warnings) > 0

    def test_degraded_warning(self, monitor):
        for _ in range(4):
            monitor.record_capture(success=True, device="d")
        for _ in range(3):
            monitor.record_capture(success=False, device="d", error="x")
        report = monitor.get_health_report()
        # 4/7 ≈ 57% < 80% threshold
        rate_warnings = [w for w in report["warnings"] if "success rate" in w.lower()]
        assert len(rate_warnings) > 0


# ---------------------------------------------------------------------------
# reset
# ---------------------------------------------------------------------------


class TestReset:
    def test_reset_clears_events(self, monitor):
        monitor.record_capture(success=True, device="d")
        monitor.reset()
        report = monitor.get_health_report()
        assert report["total_captures"] == 0

    def test_reset_clears_devices(self, monitor):
        monitor.record_capture(success=True, device="d")
        monitor.reset()
        assert monitor.get_device_stats("d") is None

    def test_reset_resets_uptime(self, monitor):
        time.sleep(0.02)
        monitor.reset()
        report = monitor.get_health_report()
        assert report["uptime_seconds"] < 1.0


# ---------------------------------------------------------------------------
# get_device_stats returns copy
# ---------------------------------------------------------------------------


class TestDeviceStatsCopy:
    def test_returns_none_for_unknown(self, monitor):
        assert monitor.get_device_stats("nope") is None

    def test_returns_copy(self, monitor):
        monitor.record_capture(success=True, device="d")
        s1 = monitor.get_device_stats("d")
        s1.total_captures = 999
        s2 = monitor.get_device_stats("d")
        assert s2.total_captures == 1  # unaffected by mutation


# ---------------------------------------------------------------------------
# Thread safety
# ---------------------------------------------------------------------------


class TestThreadSafety:
    def test_concurrent_record_capture(self, monitor):
        errors = []

        def worker(device, n):
            try:
                for _ in range(n):
                    monitor.record_capture(success=True, device=device, quality_score=0.5)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker, args=(f"d{i}", 50)) for i in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors
        report = monitor.get_health_report()
        assert report["total_captures"] == 100  # 4 * 50 (with max_events=100)

    def test_concurrent_read_write(self, monitor):
        errors = []
        stop = threading.Event()

        def writer():
            try:
                for i in range(100):
                    monitor.record_capture(success=i % 3 != 0, device="d")
            except Exception as e:
                errors.append(e)
            finally:
                stop.set()

        def reader():
            try:
                while not stop.is_set():
                    monitor.get_health_report()
                    monitor.get_device_health("d")
                    monitor.get_overall_health()
            except Exception as e:
                errors.append(e)

        w = threading.Thread(target=writer)
        r = threading.Thread(target=reader)
        w.start()
        r.start()
        w.join()
        r.join()

        assert not errors


# ---------------------------------------------------------------------------
# Module singleton
# ---------------------------------------------------------------------------


class TestSingleton:
    def test_get_health_monitor_returns_instance(self):
        m = get_health_monitor()
        assert isinstance(m, VisionHealthMonitor)

    def test_get_health_monitor_returns_same_instance(self):
        m1 = get_health_monitor()
        m2 = get_health_monitor()
        assert m1 is m2


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_empty_device_string(self, monitor):
        monitor.record_capture(success=True, device="")
        stats = monitor.get_device_stats("")
        assert stats is not None
        assert stats.total_captures == 1

    def test_large_quality_score(self, monitor):
        monitor.record_capture(success=True, device="d", quality_score=10.0)
        stats = monitor.get_device_stats("d")
        assert stats.average_quality == 10.0

    def test_negative_latency_stored(self, monitor):
        monitor.record_capture(success=True, device="d", latency_ms=-5.0)
        stats = monitor.get_device_stats("d")
        assert stats.average_latency_ms == -5.0

    def test_many_devices(self, monitor):
        for i in range(50):
            monitor.record_capture(success=True, device=f"/dev/video{i}")
        report = monitor.get_health_report()
        assert len(report["devices"]) == 50

    def test_recent_window_excludes_old_events(self):
        m = VisionHealthMonitor(max_events=100, recent_window_secs=0.01)
        m.record_capture(success=True, device="d")
        time.sleep(0.02)  # older than window
        m.record_capture(success=False, device="d", error="x")
        report = m.get_health_report()
        # Only the most recent event is within the window
        assert report["recent_success_rate"] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# get_recommendations
# ---------------------------------------------------------------------------


class TestGetRecommendations:
    def test_no_recommendations_when_empty(self, monitor):
        assert monitor.get_recommendations() == []

    def test_no_recommendations_with_few_captures(self, monitor):
        monitor.record_capture(success=False, device="d", error="timeout")
        assert monitor.get_recommendations() == []

    def test_permission_error_recommends_video_group(self, monitor):
        for _ in range(5):
            monitor.record_capture(success=False, device="/dev/video0", error="permission denied")
        recs = monitor.get_recommendations()
        assert any("video" in r and "usermod" in r for r in recs)

    def test_busy_error_recommends_lsof(self, monitor):
        for _ in range(5):
            monitor.record_capture(success=False, device="/dev/video0", error="device busy")
        recs = monitor.get_recommendations()
        assert any("lsof" in r for r in recs)

    def test_generic_failure_recommends_doctor(self, monitor):
        for _ in range(5):
            monitor.record_capture(success=False, device="/dev/video0", error="unknown error")
        recs = monitor.get_recommendations()
        assert any("doctor" in r for r in recs)

    def test_high_latency_recommends_lower_resolution(self, monitor):
        for _ in range(5):
            monitor.record_capture(success=True, device="d", latency_ms=3000)
        recs = monitor.get_recommendations()
        assert any("resolution" in r for r in recs)

    def test_low_quality_recommends_lighting(self, monitor):
        for _ in range(5):
            monitor.record_capture(success=True, device="d", quality_score=0.2)
        recs = monitor.get_recommendations()
        assert any("lighting" in r for r in recs)

    def test_healthy_device_no_recommendations(self, monitor):
        for _ in range(10):
            monitor.record_capture(success=True, device="d", quality_score=0.9, latency_ms=50)
        assert monitor.get_recommendations() == []

    def test_multiple_devices_independent_recommendations(self, monitor):
        for _ in range(5):
            monitor.record_capture(success=False, device="a", error="permission denied")
        for _ in range(5):
            monitor.record_capture(success=True, device="b", quality_score=0.1)
        recs = monitor.get_recommendations()
        assert len(recs) == 2
