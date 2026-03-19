"""Session 11: Tests for VisionDoctor untested methods and health monitor persistence.

Covers the coverage gaps identified by the exploration agent:
- VisionDoctor.check_camera_discovery() with mocked discovery
- VisionDoctor.check_capture() with mocked camera
- VisionDoctor.check_captures_directory() with mocked filesystem
- VisionDoctor.check_health_monitor() with various states
- VisionHealthMonitor.save() and .load() round-trip
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# VisionDoctor.check_camera_discovery()
# ---------------------------------------------------------------------------


class TestDoctorCheckCameraDiscovery:
    """Tests for VisionDoctor.check_camera_discovery()."""

    def test_no_cameras_found(self) -> None:
        from missy.vision.doctor import VisionDoctor

        doctor = VisionDoctor()
        with patch("missy.vision.discovery.CameraDiscovery") as MockDisc:
            MockDisc.return_value.discover.return_value = []
            result = doctor.check_camera_discovery()

        assert result.name == "camera_discovery"
        assert not result.passed
        assert "no cameras" in result.message.lower()

    def test_cameras_found_all_readable(self) -> None:
        from missy.vision.discovery import CameraDevice
        from missy.vision.doctor import VisionDoctor

        device = CameraDevice("/dev/video0", "Test Cam", "046d", "085c", "usb-1")
        doctor = VisionDoctor()

        with (
            patch("missy.vision.discovery.CameraDiscovery") as MockDisc,
            patch("os.access", return_value=True),
        ):
            MockDisc.return_value.discover.return_value = [device]
            result = doctor.check_camera_discovery()

        assert result.passed
        assert "1 camera" in result.message
        assert "Test Cam" in result.message

    def test_cameras_found_some_unreadable(self) -> None:
        from missy.vision.discovery import CameraDevice
        from missy.vision.doctor import VisionDoctor

        device = CameraDevice("/dev/video0", "Cam", "046d", "085c", "usb-1")
        doctor = VisionDoctor()

        with (
            patch("missy.vision.discovery.CameraDiscovery") as MockDisc,
            patch("os.access", return_value=False),
        ):
            MockDisc.return_value.discover.return_value = [device]
            result = doctor.check_camera_discovery()

        assert not result.passed
        assert "not readable" in result.message.lower()
        assert result.severity == "warning"

    def test_discovery_exception(self) -> None:
        from missy.vision.doctor import VisionDoctor

        doctor = VisionDoctor()
        with patch("missy.vision.discovery.CameraDiscovery", side_effect=RuntimeError("fail")):
            result = doctor.check_camera_discovery()

        assert not result.passed
        assert result.severity == "error"
        assert "fail" in result.message


# ---------------------------------------------------------------------------
# VisionDoctor.check_capture()
# ---------------------------------------------------------------------------


class TestDoctorCheckCapture:
    """Tests for VisionDoctor.check_capture()."""

    def test_no_camera_available(self) -> None:
        from missy.vision.doctor import VisionDoctor

        doctor = VisionDoctor()
        with patch("missy.vision.discovery.find_preferred_camera", return_value=None):
            result = doctor.check_capture()

        assert not result.passed
        assert result.severity == "warning"
        assert "no camera" in result.message.lower()

    def test_capture_succeeds(self) -> None:
        from missy.vision.capture import CaptureResult
        from missy.vision.discovery import CameraDevice
        from missy.vision.doctor import VisionDoctor

        device = CameraDevice("/dev/video0", "Test Cam", "046d", "085c", "usb-1")
        mock_handle = MagicMock()
        mock_handle.capture.return_value = CaptureResult(
            success=True, width=1920, height=1080,
            image=np.zeros((1080, 1920, 3), dtype=np.uint8),
        )

        doctor = VisionDoctor()
        with (
            patch("missy.vision.discovery.find_preferred_camera", return_value=device),
            patch("missy.vision.capture.CameraHandle", return_value=mock_handle),
        ):
            result = doctor.check_capture()

        assert result.passed
        assert "1920x1080" in result.message
        assert result.details["camera"] == "Test Cam"

    def test_capture_fails(self) -> None:
        from missy.vision.capture import CaptureResult
        from missy.vision.discovery import CameraDevice
        from missy.vision.doctor import VisionDoctor

        device = CameraDevice("/dev/video0", "Test Cam", "046d", "085c", "usb-1")
        mock_handle = MagicMock()
        mock_handle.capture.return_value = CaptureResult(
            success=False, error="Blank frame",
        )

        doctor = VisionDoctor()
        with (
            patch("missy.vision.discovery.find_preferred_camera", return_value=device),
            patch("missy.vision.capture.CameraHandle", return_value=mock_handle),
        ):
            result = doctor.check_capture()

        assert not result.passed
        assert result.severity == "error"
        assert "Blank frame" in result.message

    def test_capture_exception(self) -> None:
        from missy.vision.discovery import CameraDevice
        from missy.vision.doctor import VisionDoctor

        device = CameraDevice("/dev/video0", "Test Cam", "046d", "085c", "usb-1")

        doctor = VisionDoctor()
        with (
            patch("missy.vision.discovery.find_preferred_camera", return_value=device),
            patch("missy.vision.capture.CameraHandle", side_effect=RuntimeError("open fail")),
        ):
            result = doctor.check_capture()

        assert not result.passed
        assert "open fail" in result.message


# ---------------------------------------------------------------------------
# VisionDoctor.check_captures_directory()
# ---------------------------------------------------------------------------


class TestDoctorCheckCapturesDirectory:
    """Tests for VisionDoctor.check_captures_directory()."""

    def test_directory_writable_with_space(self, tmp_path: Path) -> None:
        from missy.vision.doctor import VisionDoctor

        doctor = VisionDoctor()
        captures_dir = tmp_path / ".missy" / "captures"
        captures_dir.mkdir(parents=True)

        with (
            patch.object(Path, "home", return_value=tmp_path),
            patch("os.access", return_value=True),
        ):
            result = doctor.check_captures_directory()

        assert result.passed
        assert "ready" in result.message.lower()

    def test_directory_not_writable(self, tmp_path: Path) -> None:
        from missy.vision.doctor import VisionDoctor

        doctor = VisionDoctor()
        captures_dir = tmp_path / ".missy" / "captures"
        captures_dir.mkdir(parents=True)

        with (
            patch.object(Path, "home", return_value=tmp_path),
            patch("os.access", return_value=False),
        ):
            result = doctor.check_captures_directory()

        assert not result.passed
        assert "not writable" in result.message.lower()

    def test_low_disk_space(self, tmp_path: Path) -> None:
        from missy.vision.doctor import VisionDoctor

        doctor = VisionDoctor()
        captures_dir = tmp_path / ".missy" / "captures"
        captures_dir.mkdir(parents=True)

        # Mock low disk space (50 MB free)
        mock_usage = MagicMock()
        mock_usage.free = 50 * 1024 * 1024
        mock_usage.total = 100 * 1024 * 1024

        with (
            patch.object(Path, "home", return_value=tmp_path),
            patch("os.access", return_value=True),
            patch("shutil.disk_usage", return_value=mock_usage),
        ):
            result = doctor.check_captures_directory()

        assert not result.passed
        assert "low disk space" in result.message.lower() or "50" in result.message

    def test_oserror_handled(self, tmp_path: Path) -> None:
        from missy.vision.doctor import VisionDoctor

        doctor = VisionDoctor()
        # Patch Path.home() to return a non-writable path that causes OSError
        bad_path = tmp_path / "readonly"
        bad_path.mkdir()
        with (
            patch.object(Path, "home", return_value=bad_path),
            patch("os.access", side_effect=OSError("nope")),
        ):
            result = doctor.check_captures_directory()

        assert not result.passed


# ---------------------------------------------------------------------------
# VisionDoctor.check_health_monitor()
# ---------------------------------------------------------------------------


class TestDoctorCheckHealthMonitor:
    """Tests for VisionDoctor.check_health_monitor()."""

    def test_no_captures_recorded(self) -> None:
        from missy.vision.doctor import VisionDoctor
        from missy.vision.health_monitor import VisionHealthMonitor

        doctor = VisionDoctor()
        mock_monitor = VisionHealthMonitor()

        with patch("missy.vision.health_monitor.get_health_monitor", return_value=mock_monitor):
            result = doctor.check_health_monitor()

        assert result.passed
        assert "no captures" in result.message.lower()

    def test_healthy_status(self) -> None:
        from missy.vision.doctor import VisionDoctor
        from missy.vision.health_monitor import VisionHealthMonitor

        doctor = VisionDoctor()
        monitor = VisionHealthMonitor()
        for _ in range(10):
            monitor.record_capture(success=True, device="/dev/video0")

        with patch("missy.vision.health_monitor.get_health_monitor", return_value=monitor):
            result = doctor.check_health_monitor()

        assert result.passed
        assert "ok" in result.message.lower()

    def test_unhealthy_status(self) -> None:
        from missy.vision.doctor import VisionDoctor
        from missy.vision.health_monitor import VisionHealthMonitor

        doctor = VisionDoctor()
        monitor = VisionHealthMonitor()
        for _ in range(10):
            monitor.record_capture(success=False, device="/dev/video0", error="fail")

        with patch("missy.vision.health_monitor.get_health_monitor", return_value=monitor):
            result = doctor.check_health_monitor()

        assert not result.passed
        assert result.severity == "error"
        assert "unhealthy" in result.message.lower()

    def test_degraded_status(self) -> None:
        from missy.vision.doctor import VisionDoctor
        from missy.vision.health_monitor import VisionHealthMonitor

        doctor = VisionDoctor()
        monitor = VisionHealthMonitor()
        # 3 failures + 7 successes = 70% success rate (degraded: 50-80% threshold)
        for _ in range(3):
            monitor.record_capture(success=False, device="/dev/video0", error="fail")
        for _ in range(7):
            monitor.record_capture(success=True, device="/dev/video0")

        with patch("missy.vision.health_monitor.get_health_monitor", return_value=monitor):
            result = doctor.check_health_monitor()

        assert not result.passed
        assert "degraded" in result.message.lower()

    def test_exception_handled(self) -> None:
        from missy.vision.doctor import VisionDoctor

        doctor = VisionDoctor()
        with patch("missy.vision.health_monitor.get_health_monitor", side_effect=RuntimeError("boom")):
            result = doctor.check_health_monitor()

        assert result.passed  # gracefully handled
        assert "not available" in result.message.lower()


# ---------------------------------------------------------------------------
# DoctorReport properties
# ---------------------------------------------------------------------------


class TestDoctorReport:
    """DoctorReport aggregate properties."""

    def test_empty_report(self) -> None:
        from missy.vision.doctor import DoctorReport

        report = DoctorReport()
        assert report.passed == 0
        assert report.failed == 0
        assert report.warnings == 0
        assert report.errors == 0
        assert report.overall_healthy is True

    def test_mixed_results(self) -> None:
        from missy.vision.doctor import DiagnosticResult, DoctorReport

        report = DoctorReport()
        report.add(DiagnosticResult("a", True, "ok"))
        report.add(DiagnosticResult("b", False, "warn", severity="warning"))
        report.add(DiagnosticResult("c", False, "err", severity="error"))
        report.add(DiagnosticResult("d", True, "ok"))

        assert report.passed == 2
        assert report.failed == 2
        assert report.warnings == 1
        assert report.errors == 1
        assert report.overall_healthy is False  # has error


# ---------------------------------------------------------------------------
# VisionHealthMonitor save/load round-trip
# ---------------------------------------------------------------------------


class TestHealthMonitorPersistence:
    """Tests for VisionHealthMonitor.save() and .load()."""

    def test_save_and_load_round_trip(self, tmp_path: Path) -> None:
        from missy.vision.health_monitor import VisionHealthMonitor

        db_path = tmp_path / "health.db"
        monitor = VisionHealthMonitor()

        # Record some data
        monitor.record_capture(success=True, device="/dev/video0", quality_score=0.85, latency_ms=50.0)
        monitor.record_capture(success=True, device="/dev/video0", quality_score=0.90, latency_ms=45.0)
        monitor.record_capture(success=False, device="/dev/video0", error="blank frame")
        monitor.record_capture(success=True, device="/dev/video1", quality_score=0.70)

        # Save
        monitor.save(db_path)

        # Load into a new monitor
        monitor2 = VisionHealthMonitor()
        monitor2.load(db_path)

        # Verify device stats were loaded
        stats0 = monitor2.get_device_stats("/dev/video0")
        assert stats0 is not None
        assert stats0.total_captures == 3
        assert stats0.successful_captures == 2
        assert stats0.failed_captures == 1

        stats1 = monitor2.get_device_stats("/dev/video1")
        assert stats1 is not None
        assert stats1.total_captures == 1

    def test_save_creates_tables(self, tmp_path: Path) -> None:
        from missy.vision.health_monitor import VisionHealthMonitor

        db_path = tmp_path / "health.db"
        monitor = VisionHealthMonitor()
        monitor.save(db_path)

        # Verify tables exist
        conn = sqlite3.connect(str(db_path))
        tables = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()
        conn.close()
        table_names = {t[0] for t in tables}
        assert "device_stats" in table_names
        assert "capture_events" in table_names

    def test_save_without_path_raises(self) -> None:
        from missy.vision.health_monitor import VisionHealthMonitor

        monitor = VisionHealthMonitor()
        with pytest.raises(ValueError, match="persist_path"):
            monitor.save()

    def test_load_without_path_raises(self) -> None:
        from missy.vision.health_monitor import VisionHealthMonitor

        monitor = VisionHealthMonitor()
        with pytest.raises(ValueError, match="persist_path"):
            monitor.load()

    def test_load_missing_file_raises(self, tmp_path: Path) -> None:
        from missy.vision.health_monitor import VisionHealthMonitor

        monitor = VisionHealthMonitor()
        with pytest.raises(FileNotFoundError):
            monitor.load(tmp_path / "nonexistent.db")

    def test_auto_persist_on_construct(self, tmp_path: Path) -> None:
        """Monitor with persist_path auto-loads on construction."""
        from missy.vision.health_monitor import VisionHealthMonitor

        db_path = tmp_path / "health.db"
        # First monitor: save some data
        m1 = VisionHealthMonitor()
        m1.record_capture(success=True, device="/dev/video0")
        m1.save(db_path)

        # Second monitor: auto-loads from persist_path
        m2 = VisionHealthMonitor(persist_path=db_path)
        stats = m2.get_device_stats("/dev/video0")
        assert stats is not None
        assert stats.total_captures >= 1

    def test_save_events_restored(self, tmp_path: Path) -> None:
        """Events saved to SQLite are restored on load."""
        from missy.vision.health_monitor import VisionHealthMonitor

        db_path = tmp_path / "health.db"
        monitor = VisionHealthMonitor()
        for i in range(5):
            monitor.record_capture(
                success=(i % 2 == 0),
                device="/dev/video0",
                latency_ms=float(i * 10),
            )
        monitor.save(db_path)

        monitor2 = VisionHealthMonitor()
        monitor2.load(db_path)
        report = monitor2.get_health_report()
        assert report["total_captures"] >= 5

    def test_load_corrupt_json_skipped(self, tmp_path: Path) -> None:
        """Corrupt JSON in device_stats is skipped gracefully."""
        from missy.vision.health_monitor import VisionHealthMonitor

        db_path = tmp_path / "health.db"
        conn = sqlite3.connect(str(db_path))
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute(
            "CREATE TABLE device_stats ("
            "device TEXT PRIMARY KEY, data TEXT NOT NULL, updated_at REAL NOT NULL)"
        )
        conn.execute(
            "CREATE TABLE capture_events ("
            "id INTEGER PRIMARY KEY AUTOINCREMENT, timestamp REAL NOT NULL, "
            "success INTEGER NOT NULL, device TEXT NOT NULL, quality_score REAL, "
            "error TEXT, latency_ms REAL, source_type TEXT)"
        )
        # Insert corrupt data
        conn.execute(
            "INSERT INTO device_stats VALUES (?, ?, ?)",
            ("/dev/video0", "not valid json{{{", 0.0),
        )
        conn.commit()
        conn.close()

        monitor = VisionHealthMonitor()
        monitor.load(db_path)  # should not raise
        # Device was skipped
        stats = monitor.get_device_stats("/dev/video0")
        assert stats is None
