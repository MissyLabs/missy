"""Tests for VisionHealthMonitor SQLite persistence."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from missy.vision.health_monitor import (
    HealthStatus,
    VisionHealthMonitor,
)


class TestHealthMonitorSave:
    """Tests for saving health data to SQLite."""

    def test_save_creates_database(self, tmp_path: Path) -> None:
        db = tmp_path / "health.db"
        monitor = VisionHealthMonitor()
        monitor.record_capture(success=True, device="/dev/video0", quality_score=0.9)
        monitor.save(db)
        assert db.exists()

    def test_save_creates_parent_dirs(self, tmp_path: Path) -> None:
        db = tmp_path / "sub" / "dir" / "health.db"
        monitor = VisionHealthMonitor()
        monitor.record_capture(success=True, device="/dev/video0")
        monitor.save(db)
        assert db.exists()

    def test_save_stores_device_stats(self, tmp_path: Path) -> None:
        db = tmp_path / "health.db"
        monitor = VisionHealthMonitor()
        monitor.record_capture(success=True, device="/dev/video0", quality_score=0.8)
        monitor.record_capture(success=False, device="/dev/video0", error="blank")
        monitor.save(db)

        conn = sqlite3.connect(str(db))
        rows = conn.execute("SELECT * FROM device_stats").fetchall()
        conn.close()
        assert len(rows) == 1
        assert rows[0][0] == "/dev/video0"

    def test_save_stores_events(self, tmp_path: Path) -> None:
        db = tmp_path / "health.db"
        monitor = VisionHealthMonitor()
        for _ in range(5):
            monitor.record_capture(success=True, device="/dev/video0")
        monitor.save(db)

        conn = sqlite3.connect(str(db))
        count = conn.execute("SELECT COUNT(*) FROM capture_events").fetchone()[0]
        conn.close()
        assert count == 5

    def test_save_multiple_devices(self, tmp_path: Path) -> None:
        db = tmp_path / "health.db"
        monitor = VisionHealthMonitor()
        monitor.record_capture(success=True, device="/dev/video0")
        monitor.record_capture(success=True, device="/dev/video2")
        monitor.save(db)

        conn = sqlite3.connect(str(db))
        rows = conn.execute("SELECT device FROM device_stats ORDER BY device").fetchall()
        conn.close()
        assert [r[0] for r in rows] == ["/dev/video0", "/dev/video2"]

    def test_save_overwrites_previous(self, tmp_path: Path) -> None:
        db = tmp_path / "health.db"
        m1 = VisionHealthMonitor()
        m1.record_capture(success=True, device="/dev/video0")
        m1.save(db)

        m2 = VisionHealthMonitor()
        m2.record_capture(success=True, device="/dev/video0")
        m2.record_capture(success=True, device="/dev/video0")
        m2.save(db)

        conn = sqlite3.connect(str(db))
        count = conn.execute("SELECT COUNT(*) FROM capture_events").fetchone()[0]
        conn.close()
        assert count == 2  # not 3

    def test_save_requires_path(self) -> None:
        monitor = VisionHealthMonitor()
        with pytest.raises(ValueError, match="No persist_path"):
            monitor.save()

    def test_save_empty_monitor(self, tmp_path: Path) -> None:
        db = tmp_path / "health.db"
        monitor = VisionHealthMonitor()
        monitor.save(db)
        assert db.exists()

        conn = sqlite3.connect(str(db))
        count = conn.execute("SELECT COUNT(*) FROM device_stats").fetchone()[0]
        conn.close()
        assert count == 0


class TestHealthMonitorLoad:
    """Tests for loading health data from SQLite."""

    def test_load_restores_device_stats(self, tmp_path: Path) -> None:
        db = tmp_path / "health.db"
        m1 = VisionHealthMonitor()
        m1.record_capture(success=True, device="/dev/video0", quality_score=0.9)
        m1.record_capture(success=True, device="/dev/video0", quality_score=0.8)
        m1.record_capture(success=False, device="/dev/video0", error="blank")
        m1.save(db)

        m2 = VisionHealthMonitor()
        m2.load(db)
        stats = m2.get_device_stats("/dev/video0")
        assert stats is not None
        assert stats.total_captures == 3
        assert stats.successful_captures == 2
        assert stats.failed_captures == 1

    def test_load_restores_events(self, tmp_path: Path) -> None:
        db = tmp_path / "health.db"
        m1 = VisionHealthMonitor()
        for _ in range(7):
            m1.record_capture(success=True, device="/dev/video0")
        m1.save(db)

        m2 = VisionHealthMonitor()
        m2.load(db)
        report = m2.get_health_report()
        assert report["total_captures"] == 7

    def test_load_nonexistent_raises(self, tmp_path: Path) -> None:
        monitor = VisionHealthMonitor()
        with pytest.raises(FileNotFoundError):
            monitor.load(tmp_path / "nonexistent.db")

    def test_load_requires_path(self) -> None:
        monitor = VisionHealthMonitor()
        with pytest.raises(ValueError, match="No persist_path"):
            monitor.load()

    def test_load_merges_with_existing(self, tmp_path: Path) -> None:
        db = tmp_path / "health.db"
        m1 = VisionHealthMonitor()
        m1.record_capture(success=True, device="/dev/video0")
        m1.save(db)

        m2 = VisionHealthMonitor()
        m2.record_capture(success=True, device="/dev/video0")
        m2.load(db)
        stats = m2.get_device_stats("/dev/video0")
        assert stats is not None
        assert stats.total_captures == 2  # 1 existing + 1 loaded

    def test_load_preserves_health_status(self, tmp_path: Path) -> None:
        db = tmp_path / "health.db"
        m1 = VisionHealthMonitor()
        for _ in range(10):
            m1.record_capture(success=True, device="/dev/video0", quality_score=0.9)
        m1.save(db)

        m2 = VisionHealthMonitor()
        m2.load(db)
        assert m2.get_device_health("/dev/video0") == HealthStatus.HEALTHY


class TestHealthMonitorPersistPath:
    """Tests for constructor persist_path auto-load."""

    def test_auto_load_on_init(self, tmp_path: Path) -> None:
        db = tmp_path / "health.db"
        m1 = VisionHealthMonitor()
        m1.record_capture(success=True, device="/dev/video0", quality_score=0.85)
        m1.save(db)

        m2 = VisionHealthMonitor(persist_path=db)
        stats = m2.get_device_stats("/dev/video0")
        assert stats is not None
        assert stats.total_captures == 1

    def test_auto_load_missing_file_no_error(self, tmp_path: Path) -> None:
        db = tmp_path / "nonexistent.db"
        # Should not raise — just logs and continues
        monitor = VisionHealthMonitor(persist_path=db)
        assert monitor.get_overall_health() == HealthStatus.UNKNOWN

    def test_save_uses_persist_path(self, tmp_path: Path) -> None:
        db = tmp_path / "health.db"
        monitor = VisionHealthMonitor(persist_path=db)
        monitor.record_capture(success=True, device="/dev/video0")
        monitor.save()  # no explicit path needed
        assert db.exists()


class TestHealthMonitorRoundTrip:
    """End-to-end save/load round-trip tests."""

    def test_full_round_trip(self, tmp_path: Path) -> None:
        db = tmp_path / "health.db"

        m1 = VisionHealthMonitor()
        m1.record_capture(success=True, device="/dev/video0", quality_score=0.9, latency_ms=50.0)
        m1.record_capture(success=True, device="/dev/video0", quality_score=0.85, latency_ms=55.0)
        m1.record_capture(success=False, device="/dev/video0", error="blank frame")
        m1.record_capture(success=True, device="/dev/video2", quality_score=0.7, latency_ms=100.0)
        m1.save(db)

        m2 = VisionHealthMonitor()
        m2.load(db)

        s0 = m2.get_device_stats("/dev/video0")
        assert s0 is not None
        assert s0.total_captures == 3
        assert s0.successful_captures == 2
        assert s0.failed_captures == 1
        assert s0.last_error == "blank frame"

        s2 = m2.get_device_stats("/dev/video2")
        assert s2 is not None
        assert s2.total_captures == 1
        assert s2.successful_captures == 1

    def test_round_trip_preserves_quality(self, tmp_path: Path) -> None:
        db = tmp_path / "health.db"
        m1 = VisionHealthMonitor()
        m1.record_capture(success=True, device="/dev/video0", quality_score=0.75)
        m1.record_capture(success=True, device="/dev/video0", quality_score=0.85)
        m1.save(db)

        m2 = VisionHealthMonitor()
        m2.load(db)
        stats = m2.get_device_stats("/dev/video0")
        assert stats is not None
        assert abs(stats.average_quality - 0.80) < 0.01

    def test_round_trip_preserves_latency(self, tmp_path: Path) -> None:
        db = tmp_path / "health.db"
        m1 = VisionHealthMonitor()
        m1.record_capture(success=True, device="/dev/video0", latency_ms=40.0)
        m1.record_capture(success=True, device="/dev/video0", latency_ms=60.0)
        m1.save(db)

        m2 = VisionHealthMonitor()
        m2.load(db)
        stats = m2.get_device_stats("/dev/video0")
        assert stats is not None
        assert abs(stats.average_latency_ms - 50.0) < 0.01

    def test_round_trip_empty_db(self, tmp_path: Path) -> None:
        db = tmp_path / "health.db"
        m1 = VisionHealthMonitor()
        m1.save(db)

        m2 = VisionHealthMonitor()
        m2.load(db)
        assert m2.get_overall_health() == HealthStatus.UNKNOWN

    def test_corrupt_db_handled(self, tmp_path: Path) -> None:
        db = tmp_path / "health.db"
        db.write_text("not a sqlite database")
        monitor = VisionHealthMonitor()
        with pytest.raises(sqlite3.DatabaseError):
            monitor.load(db)


class TestHealthMonitorAutoSave:
    """Tests for periodic auto-save on record_capture."""

    def test_auto_save_triggers_at_interval(self, tmp_path: Path) -> None:
        db = tmp_path / "health.db"
        monitor = VisionHealthMonitor(persist_path=db, auto_save_interval=5)
        for _ in range(4):
            monitor.record_capture(success=True, device="/dev/video0")
        assert not db.exists()  # not yet reached interval

        monitor.record_capture(success=True, device="/dev/video0")
        assert db.exists()  # interval reached, auto-saved

    def test_auto_save_resets_counter(self, tmp_path: Path) -> None:
        db = tmp_path / "health.db"
        monitor = VisionHealthMonitor(persist_path=db, auto_save_interval=3)
        for _ in range(3):
            monitor.record_capture(success=True, device="/dev/video0")
        assert db.exists()

        # Reset db to check second auto-save
        db.unlink()
        for _ in range(2):
            monitor.record_capture(success=True, device="/dev/video0")
        assert not db.exists()  # only 2 since last save
        monitor.record_capture(success=True, device="/dev/video0")
        assert db.exists()  # 3 since last save

    def test_no_auto_save_without_persist_path(self) -> None:
        monitor = VisionHealthMonitor(auto_save_interval=1)
        # Should not raise
        monitor.record_capture(success=True, device="/dev/video0")
        monitor.record_capture(success=True, device="/dev/video0")

    def test_auto_save_failure_non_fatal(self, tmp_path: Path) -> None:
        # Use a directory as path (can't write SQLite to a directory)
        db = tmp_path / "not_a_file"
        db.mkdir()
        monitor = VisionHealthMonitor(persist_path=db / "health.db", auto_save_interval=1)
        # Remove parent dir to cause save failure
        import shutil
        shutil.rmtree(db)
        db.mkdir()  # recreate as directory so path doesn't exist properly
        # Should not raise even though save fails
        monitor.record_capture(success=True, device="/dev/video0")
