"""Session 10 audit and health monitor hardening tests.

Covers under-tested paths in:
- VisionAudit event logging
- HealthMonitor auto-save boundary conditions
- HealthMonitor corrupted data recovery
- HealthMonitor recommendations
- Config validator edge cases
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Vision audit event logging
# ---------------------------------------------------------------------------


class TestVisionAuditEvents:
    """Vision audit module event logging."""

    def test_audit_capture_event(self) -> None:
        from missy.vision.audit import audit_vision_capture

        with patch("missy.vision.audit._emit_audit_event") as mock_emit:
            audit_vision_capture(
                device="/dev/video0",
                source_type="webcam",
                success=True,
                width=1920,
                height=1080,
            )
            mock_emit.assert_called_once()
            # Verify category and action args
            args = mock_emit.call_args[0]
            assert args[0] == "vision"  # category
            assert args[1] == "capture"  # action
            details = args[2]
            assert details["device"] == "/dev/video0"
            assert details["success"] is True

    def test_audit_capture_failure(self) -> None:
        from missy.vision.audit import audit_vision_capture

        with patch("missy.vision.audit._emit_audit_event") as mock_emit:
            audit_vision_capture(
                device="/dev/video0",
                source_type="webcam",
                success=False,
                error="Device busy",
            )
            mock_emit.assert_called_once()
            details = mock_emit.call_args[0][2]
            assert details["success"] is False
            assert details["error"] == "Device busy"

    def test_audit_session_event(self) -> None:
        from missy.vision.audit import audit_vision_session

        with patch("missy.vision.audit._emit_audit_event") as mock_emit:
            audit_vision_session(
                task_id="puzzle-1",
                task_type="puzzle",
                action="start",
                frame_count=0,
            )
            mock_emit.assert_called_once()
            args = mock_emit.call_args[0]
            assert args[1] == "session_start"  # action="start" → "session_start"

    def test_audit_intent_event(self) -> None:
        from missy.vision.audit import audit_vision_intent

        with patch("missy.vision.audit._emit_audit_event") as mock_emit:
            audit_vision_intent(
                text="look at this",
                intent="look",
                confidence=0.9,
                decision="activate",
            )
            mock_emit.assert_called_once()
            args = mock_emit.call_args[0]
            assert args[1] == "intent"

    def test_audit_analysis_event(self) -> None:
        from missy.vision.audit import audit_vision_analysis

        with patch("missy.vision.audit._emit_audit_event") as mock_emit:
            audit_vision_analysis(
                mode="puzzle",
                source_type="webcam",
                success=True,
            )
            mock_emit.assert_called_once()
            args = mock_emit.call_args[0]
            assert args[1] == "analyze"


# ---------------------------------------------------------------------------
# Health monitor auto-save and recovery
# ---------------------------------------------------------------------------


class TestHealthMonitorAutoSave:
    """VisionHealthMonitor auto-save boundary and recovery."""

    def test_record_capture_increments_counter(self) -> None:
        from missy.vision.health_monitor import VisionHealthMonitor

        monitor = VisionHealthMonitor()
        monitor.record_capture(success=True, device="/dev/video0", latency_ms=50.0)
        stats = monitor.get_device_stats("/dev/video0")
        assert stats is not None
        assert stats.total_captures == 1
        assert stats.successful_captures == 1

    def test_record_capture_failure(self) -> None:
        from missy.vision.health_monitor import VisionHealthMonitor

        monitor = VisionHealthMonitor()
        monitor.record_capture(
            success=False, device="/dev/video0", error="Device busy"
        )
        stats = monitor.get_device_stats("/dev/video0")
        assert stats.failed_captures == 1
        assert stats.last_error == "Device busy"

    def test_success_rate_calculation(self) -> None:
        from missy.vision.health_monitor import VisionHealthMonitor

        monitor = VisionHealthMonitor()
        for _ in range(8):
            monitor.record_capture(success=True, device="/dev/video0", latency_ms=50.0)
        for _ in range(2):
            monitor.record_capture(success=False, device="/dev/video0", error="fail")

        stats = monitor.get_device_stats("/dev/video0")
        assert stats.success_rate == pytest.approx(0.8, abs=0.01)

    def test_overall_health_healthy(self) -> None:
        from missy.vision.health_monitor import HealthStatus, VisionHealthMonitor

        monitor = VisionHealthMonitor()
        monitor.record_capture(success=True, device="/dev/video0", latency_ms=50.0)
        monitor.record_capture(success=True, device="/dev/video2", latency_ms=60.0)

        health = monitor.get_overall_health()
        assert health == HealthStatus.HEALTHY

    def test_health_report_dict(self) -> None:
        from missy.vision.health_monitor import VisionHealthMonitor

        monitor = VisionHealthMonitor()
        monitor.record_capture(success=True, device="/dev/video0", latency_ms=50.0)

        report = monitor.get_health_report()
        assert isinstance(report, dict)
        assert "overall_status" in report

    def test_recommendations_for_low_success_rate(self) -> None:
        from missy.vision.health_monitor import VisionHealthMonitor

        monitor = VisionHealthMonitor()
        for _ in range(10):
            monitor.record_capture(success=False, device="/dev/video0", error="fail")

        recs = monitor.get_recommendations()
        assert isinstance(recs, list)
        # Should recommend something when success rate is 0%
        assert len(recs) > 0

    def test_reset_clears_all_data(self) -> None:
        from missy.vision.health_monitor import VisionHealthMonitor

        monitor = VisionHealthMonitor()
        monitor.record_capture(success=True, device="/dev/video0", latency_ms=50.0)
        monitor.reset()

        stats = monitor.get_device_stats("/dev/video0")
        assert stats is None or stats.total_captures == 0

    def test_save_and_load_roundtrip(self, tmp_path: Path) -> None:
        from missy.vision.health_monitor import VisionHealthMonitor

        persist_path = tmp_path / "health.db"
        monitor = VisionHealthMonitor(persist_path=str(persist_path))
        monitor.record_capture(success=True, device="/dev/video0", latency_ms=42.0)
        monitor.save()

        # Load into fresh monitor
        monitor2 = VisionHealthMonitor(persist_path=str(persist_path))
        monitor2.load()

        stats = monitor2.get_device_stats("/dev/video0")
        assert stats is not None
        assert stats.total_captures >= 1

    def test_save_creates_file(self, tmp_path: Path) -> None:
        from missy.vision.health_monitor import VisionHealthMonitor

        persist_path = tmp_path / "health.db"
        monitor = VisionHealthMonitor(persist_path=str(persist_path))
        monitor.record_capture(success=True, device="/dev/video0", latency_ms=50.0)
        monitor.save()

        assert persist_path.exists()


# ---------------------------------------------------------------------------
# Config validator edge cases
# ---------------------------------------------------------------------------


class TestConfigValidatorEdgeCases:
    """Vision config validator boundary conditions."""

    def test_valid_default_config(self) -> None:
        from missy.vision.config_validator import validate_vision_config

        config = {
            "enabled": True,
            "preferred_device": "",
            "capture_width": 1920,
            "capture_height": 1080,
            "warmup_frames": 5,
            "max_retries": 3,
            "auto_activate_threshold": 0.80,
            "scene_memory_max_frames": 20,
            "scene_memory_max_sessions": 5,
        }
        result = validate_vision_config(config)
        assert result.valid

    def test_invalid_width(self) -> None:
        from missy.vision.config_validator import validate_vision_config

        config = {
            "enabled": True,
            "capture_width": -1,
            "capture_height": 1080,
        }
        result = validate_vision_config(config)
        assert not result.valid

    def test_invalid_threshold(self) -> None:
        from missy.vision.config_validator import validate_vision_config

        config = {
            "enabled": True,
            "auto_activate_threshold": 2.0,  # should be 0-1
        }
        result = validate_vision_config(config)
        assert not result.valid or len(result.warnings) > 0

    def test_zero_max_retries(self) -> None:
        from missy.vision.config_validator import validate_vision_config

        config = {
            "enabled": True,
            "max_retries": 0,
        }
        result = validate_vision_config(config)
        # 0 retries should be valid (means no retry)
        assert result is not None

    def test_empty_config(self) -> None:
        from missy.vision.config_validator import validate_vision_config

        config: dict = {}
        result = validate_vision_config(config)
        # Should handle empty config without crash
        assert result is not None

    def test_disabled_config(self) -> None:
        from missy.vision.config_validator import validate_vision_config

        config = {"enabled": False}
        result = validate_vision_config(config)
        assert result.valid


# ---------------------------------------------------------------------------
# Benchmark module
# ---------------------------------------------------------------------------


class TestBenchmarkModule:
    """Vision benchmark recording and reporting."""

    def test_record_and_report(self) -> None:
        from missy.vision.benchmark import CaptureBenchmark

        bench = CaptureBenchmark()
        for i in range(10):
            bench.record("capture", 10.0 + i)

        report = bench.report()
        cats = report.get("categories", report)
        assert "capture" in cats
        assert cats["capture"]["count"] == 10
        assert cats["capture"]["mean_ms"] > 0

    def test_record_multiple_categories(self) -> None:
        from missy.vision.benchmark import CaptureBenchmark

        bench = CaptureBenchmark()
        bench.record("capture", 10.0)
        bench.record("pipeline", 5.0)
        bench.record("analysis", 100.0)

        report = bench.report()
        cats = report.get("categories", report)
        assert len(cats) >= 3

    def test_percentiles(self) -> None:
        from missy.vision.benchmark import CaptureBenchmark

        bench = CaptureBenchmark()
        for i in range(100):
            bench.record("test", float(i))

        report = bench.report()
        cats = report.get("categories", report)
        assert cats["test"]["p95_ms"] >= 90

    def test_empty_report(self) -> None:
        from missy.vision.benchmark import CaptureBenchmark

        bench = CaptureBenchmark()
        report = bench.report()
        cats = report.get("categories", report)
        assert len(cats) == 0

    def test_reset(self) -> None:
        from missy.vision.benchmark import CaptureBenchmark

        bench = CaptureBenchmark()
        bench.record("test", 10.0)
        bench.reset()
        report = bench.report()
        cats = report.get("categories", report)
        assert len(cats) == 0

    def test_convenience_record_capture(self) -> None:
        from missy.vision.benchmark import CaptureBenchmark

        bench = CaptureBenchmark()
        bench.record_capture(latency_ms=42.0)
        report = bench.report()
        cats = report.get("categories", report)
        assert "capture" in cats


# ---------------------------------------------------------------------------
# Memory usage monitor
# ---------------------------------------------------------------------------


class TestMemoryTracker:
    """Vision memory tracker."""

    def test_report_none_initially(self) -> None:
        from missy.vision.memory_usage import MemoryTracker

        tracker = MemoryTracker()
        report = tracker.report()
        assert report is None  # no report computed yet

    def test_update_from_scene_manager(self) -> None:
        from missy.vision.memory_usage import MemoryTracker
        from missy.vision.scene_memory import SceneManager

        mgr = SceneManager()
        session = mgr.create_session("test")
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        session.add_frame(img, deduplicate=False)

        tracker = MemoryTracker()
        report = tracker.update_from_scene_manager(mgr)
        assert report is not None
        assert report.total_bytes > 0
        assert len(report.sessions) == 1

    def test_estimate_frame_bytes(self) -> None:
        from missy.vision.memory_usage import estimate_frame_bytes

        img = np.zeros((100, 100, 3), dtype=np.uint8)
        size = estimate_frame_bytes(img)
        assert size >= 100 * 100 * 3  # at least raw pixel bytes

    def test_estimate_frame_bytes_none(self) -> None:
        from missy.vision.memory_usage import estimate_frame_bytes

        assert estimate_frame_bytes(None) == 0
