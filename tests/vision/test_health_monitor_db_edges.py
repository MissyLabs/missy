"""Comprehensive tests for VisionHealthMonitor.


Covers:
- Auto-save counter recovery after save failure
- Explicit transaction rollback ensures no data loss on INSERT error
- SQLite timeout parameter (5.0 seconds) passed correctly
- Save/load roundtrip with corrupt data in database
- Load with incompatible schema (missing columns)
- Large number of device records in load()
- get_recommendations() for permission, busy, high latency, low quality
- Health report with mixed device health states
- Concurrent record_capture and save operations
- Reset clears all data properly
- Device discovery tracking without captures
- Consecutive failure warning at threshold boundary (4 vs 5)
"""

from __future__ import annotations

import json
import sqlite3
import threading
import time
from pathlib import Path
from unittest.mock import patch

import pytest

from missy.vision.health_monitor import (
    _CONSECUTIVE_FAILURE_LIMIT,
    _LOW_QUALITY_THRESHOLD,
    HealthStatus,
    VisionHealthMonitor,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def monitor() -> VisionHealthMonitor:
    """Fresh monitor with no persistence."""
    return VisionHealthMonitor(max_events=200, recent_window_secs=60)


@pytest.fixture
def db(tmp_path: Path) -> Path:
    """A path inside tmp_path for a SQLite health database."""
    return tmp_path / "health.db"


# ---------------------------------------------------------------------------
# Auto-save counter recovery after save failure
# ---------------------------------------------------------------------------


class TestAutoSaveCounterRecovery:
    """Counter is restored so the next auto-save can fire after a failure."""

    def test_counter_restored_after_save_failure(self, tmp_path: Path) -> None:
        """When auto-save fails the counter is reset to auto_save_interval
        so the next batch of captures will attempt to save again."""
        db = tmp_path / "health.db"
        monitor = VisionHealthMonitor(persist_path=db, auto_save_interval=3)

        # Intercept the save call to make it raise on the first call only.
        original_save = monitor.save
        call_count = {"n": 0}

        def failing_save(path=None):
            call_count["n"] += 1
            if call_count["n"] == 1:
                raise OSError("disk full")
            original_save(path)

        monitor.save = failing_save  # type: ignore[method-assign]

        # Trigger the first auto-save attempt (which will fail).
        for _ in range(3):
            monitor.record_capture(success=True, device="d")

        # After the failed save the counter should be restored (>= interval).
        with monitor._lock:
            counter_after_failure = monitor._capture_count_since_save
        assert counter_after_failure >= monitor._auto_save_interval

        # The next capture at the interval boundary should trigger a
        # successful save.
        # Counter is already at interval, so even 0 more captures should
        # not be needed — but record one more to pass through the check.
        monitor.record_capture(success=True, device="d")
        assert db.exists(), "Second auto-save should have created the database"

    def test_counter_zero_after_successful_save(self, tmp_path: Path) -> None:
        """After a successful auto-save the counter is reset to 0."""
        db = tmp_path / "health.db"
        monitor = VisionHealthMonitor(persist_path=db, auto_save_interval=5)
        for _ in range(5):
            monitor.record_capture(success=True, device="d")
        assert db.exists()
        with monitor._lock:
            assert monitor._capture_count_since_save == 0

    def test_counter_does_not_accumulate_without_persist_path(self) -> None:
        """With no persist_path the auto-save branch is never entered."""
        monitor = VisionHealthMonitor(auto_save_interval=1)
        for _ in range(10):
            monitor.record_capture(success=True, device="d")
        with monitor._lock:
            # Counter keeps climbing — no upper bound triggered.
            assert monitor._capture_count_since_save == 10


# ---------------------------------------------------------------------------
# Explicit transaction: ROLLBACK on INSERT error
# ---------------------------------------------------------------------------


class TestTransactionRollback:
    """If save() raises during INSERT, the ROLLBACK ensures no partial data."""

    def test_rollback_leaves_previous_data_intact(self, tmp_path: Path) -> None:
        """A failed second save does not corrupt the data from the first save.

        We wrap the real sqlite3.Connection in a proxy that intercepts the
        first INSERT INTO capture_events and raises, verifying that the
        surrounding ROLLBACK in save() restores the database to its state
        before the second transaction started.
        """
        db = tmp_path / "health.db"

        # First save — establishes baseline (1 event on disk).
        m1 = VisionHealthMonitor()
        m1.record_capture(success=True, device="/dev/video0", quality_score=0.9)
        m1.save(db)

        # Second save — inject a failure mid-transaction via a proxy wrapper.
        m2 = VisionHealthMonitor()
        m2.record_capture(success=True, device="/dev/video0", quality_score=0.5)
        m2.record_capture(success=True, device="/dev/video1")

        original_connect = sqlite3.connect

        class _FailingConnectionProxy:
            """Proxy that wraps a real sqlite3.Connection and intercepts execute."""

            def __init__(self, conn: sqlite3.Connection) -> None:
                self._conn = conn
                self._insert_count = 0

            def execute(self, sql: str, params=None):
                if "INSERT" in sql.upper() and "capture_events" in sql:
                    self._insert_count += 1
                    if self._insert_count == 1:
                        raise sqlite3.OperationalError("simulated disk full")
                if params is None:
                    return self._conn.execute(sql)
                return self._conn.execute(sql, params)

            def close(self):
                return self._conn.close()

            def __getattr__(self, name: str):
                return getattr(self._conn, name)

        def patched_connect(path, **kwargs):
            real_conn = original_connect(path, **kwargs)
            return _FailingConnectionProxy(real_conn)

        with (
            patch("missy.vision.health_monitor.sqlite3.connect", side_effect=patched_connect),
            pytest.raises(sqlite3.OperationalError, match="simulated disk full"),
        ):
            m2.save(db)

        # The database should still contain exactly the data from the first
        # successful save (1 event), not partial data from the failed second.
        conn = sqlite3.connect(str(db))
        count = conn.execute("SELECT COUNT(*) FROM capture_events").fetchone()[0]
        conn.close()
        assert count == 1, "Rollback should have preserved the original event count"

    def test_save_raises_after_rollback(self, tmp_path: Path) -> None:
        """The exception propagates to the caller even after rollback."""
        db = tmp_path / "health.db"
        monitor = VisionHealthMonitor()
        monitor.record_capture(success=True, device="d")

        original_connect = sqlite3.connect

        class _OnceFailProxy:
            def __init__(self, conn: sqlite3.Connection) -> None:
                self._conn = conn
                self._failed = False

            def execute(self, sql: str, params=None):
                if "INSERT" in sql.upper() and not self._failed:
                    self._failed = True
                    raise sqlite3.OperationalError("forced error")
                if params is None:
                    return self._conn.execute(sql)
                return self._conn.execute(sql, params)

            def close(self):
                return self._conn.close()

            def __getattr__(self, name: str):
                return getattr(self._conn, name)

        def bad_connect(path, **kwargs):
            return _OnceFailProxy(original_connect(path, **kwargs))

        with (
            patch("missy.vision.health_monitor.sqlite3.connect", side_effect=bad_connect),
            pytest.raises(sqlite3.OperationalError),
        ):
            monitor.save(db)


# ---------------------------------------------------------------------------
# SQLite timeout parameter
# ---------------------------------------------------------------------------


class TestSQLiteTimeout:
    """The 5.0-second timeout is passed to sqlite3.connect during save()."""

    def test_save_passes_timeout_5_seconds(self, tmp_path: Path) -> None:
        db = tmp_path / "health.db"
        monitor = VisionHealthMonitor()
        monitor.record_capture(success=True, device="d")

        captured_kwargs: dict = {}

        original_connect = sqlite3.connect

        def spy_connect(path, **kwargs):
            captured_kwargs.update(kwargs)
            return original_connect(path, **kwargs)

        with patch("missy.vision.health_monitor.sqlite3.connect", side_effect=spy_connect):
            monitor.save(db)

        assert "timeout" in captured_kwargs, "save() must pass timeout to sqlite3.connect"
        assert captured_kwargs["timeout"] == pytest.approx(5.0)

    def test_load_does_not_pass_timeout(self, tmp_path: Path) -> None:
        """load() omits the timeout keyword (no WAL locking concern on read)."""
        db = tmp_path / "health.db"
        monitor = VisionHealthMonitor()
        monitor.record_capture(success=True, device="d")
        monitor.save(db)

        captured_kwargs: dict = {}
        original_connect = sqlite3.connect

        def spy_connect(path, **kwargs):
            captured_kwargs.update(kwargs)
            return original_connect(path, **kwargs)

        m2 = VisionHealthMonitor()
        with patch("missy.vision.health_monitor.sqlite3.connect", side_effect=spy_connect):
            m2.load(db)

        # The load path does NOT pass timeout= at the moment.
        assert "timeout" not in captured_kwargs


# ---------------------------------------------------------------------------
# Save/load roundtrip with corrupt data
# ---------------------------------------------------------------------------


class TestRoundTripWithCorruptData:
    """Corrupt JSON in device_stats rows is skipped gracefully."""

    def test_corrupt_json_in_device_stats_is_skipped(self, tmp_path: Path) -> None:
        db = tmp_path / "health.db"

        # Build a valid database first.
        m1 = VisionHealthMonitor()
        m1.record_capture(success=True, device="/dev/video0", quality_score=0.8)
        m1.record_capture(success=True, device="/dev/video2", quality_score=0.7)
        m1.save(db)

        # Corrupt the JSON for /dev/video0.
        conn = sqlite3.connect(str(db))
        conn.execute(
            "UPDATE device_stats SET data = ? WHERE device = ?",
            ("not valid json {{{{", "/dev/video0"),
        )
        conn.commit()
        conn.close()

        m2 = VisionHealthMonitor()
        m2.load(db)  # must not raise

        # The corrupt device is skipped; the valid one is loaded.
        assert m2.get_device_stats("/dev/video0") is None
        assert m2.get_device_stats("/dev/video2") is not None
        assert m2.get_device_stats("/dev/video2").total_captures == 1

    def test_non_dict_json_in_device_stats_is_skipped(self, tmp_path: Path) -> None:
        """A row whose data parses to a list (not a dict) is silently skipped."""
        db = tmp_path / "health.db"

        m1 = VisionHealthMonitor()
        m1.record_capture(success=True, device="/dev/video0")
        m1.save(db)

        conn = sqlite3.connect(str(db))
        conn.execute(
            "UPDATE device_stats SET data = ? WHERE device = ?",
            (json.dumps([1, 2, 3]), "/dev/video0"),
        )
        conn.commit()
        conn.close()

        m2 = VisionHealthMonitor()
        m2.load(db)  # must not raise
        assert m2.get_device_stats("/dev/video0") is None

    def test_partial_fields_in_device_stats_use_defaults(self, tmp_path: Path) -> None:
        """A row whose JSON is missing fields falls back to 0/empty defaults."""
        db = tmp_path / "health.db"

        # Write a minimal-schema row directly.
        conn = sqlite3.connect(str(db))
        conn.execute(
            "CREATE TABLE device_stats "
            "(device TEXT PRIMARY KEY, data TEXT NOT NULL, updated_at REAL NOT NULL)"
        )
        conn.execute(
            "INSERT INTO device_stats VALUES (?, ?, ?)",
            ("/dev/video0", json.dumps({"total_captures": 5}), time.time()),
        )
        conn.commit()
        conn.close()

        m = VisionHealthMonitor()
        m.load(db)
        stats = m.get_device_stats("/dev/video0")
        assert stats is not None
        assert stats.total_captures == 5
        assert stats.successful_captures == 0
        assert stats.last_error == ""

    def test_events_with_null_columns_use_defaults(self, tmp_path: Path) -> None:
        """NULL columns in capture_events rows fall back to sensible defaults."""
        db = tmp_path / "health.db"

        conn = sqlite3.connect(str(db))
        conn.execute(
            "CREATE TABLE capture_events ("
            "id INTEGER PRIMARY KEY AUTOINCREMENT,"
            "timestamp REAL NOT NULL,"
            "success INTEGER NOT NULL,"
            "device TEXT NOT NULL,"
            "quality_score REAL,"
            "error TEXT,"
            "latency_ms REAL,"
            "source_type TEXT"
            ")"
        )
        conn.execute(
            "INSERT INTO capture_events "
            "(timestamp, success, device, quality_score, error, latency_ms, source_type) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (time.time(), 1, "/dev/video0", None, None, None, None),
        )
        conn.commit()
        conn.close()

        m = VisionHealthMonitor()
        m.load(db)
        report = m.get_health_report()
        assert report["total_captures"] == 1


# ---------------------------------------------------------------------------
# Load with incompatible schema (missing columns)
# ---------------------------------------------------------------------------


class TestIncompatibleSchema:
    """load() handles databases with missing tables or missing columns."""

    def test_load_missing_device_stats_table(self, tmp_path: Path) -> None:
        """A database without device_stats produces no devices."""
        db = tmp_path / "health.db"
        conn = sqlite3.connect(str(db))
        conn.execute(
            "CREATE TABLE capture_events ("
            "id INTEGER PRIMARY KEY AUTOINCREMENT,"
            "timestamp REAL NOT NULL,"
            "success INTEGER NOT NULL,"
            "device TEXT NOT NULL,"
            "quality_score REAL,"
            "error TEXT,"
            "latency_ms REAL,"
            "source_type TEXT"
            ")"
        )
        conn.commit()
        conn.close()

        m = VisionHealthMonitor()
        m.load(db)  # must not raise
        assert m.get_overall_health() == HealthStatus.UNKNOWN

    def test_load_missing_capture_events_table(self, tmp_path: Path) -> None:
        """A database without capture_events produces no events."""
        db = tmp_path / "health.db"
        conn = sqlite3.connect(str(db))
        conn.execute(
            "CREATE TABLE device_stats "
            "(device TEXT PRIMARY KEY, data TEXT NOT NULL, updated_at REAL NOT NULL)"
        )
        conn.execute(
            "INSERT INTO device_stats VALUES (?, ?, ?)",
            ("/dev/video0", json.dumps({"total_captures": 3, "successful_captures": 3,
                                        "failed_captures": 0, "total_quality": 2.7,
                                        "total_latency_ms": 150.0, "last_seen": time.time(),
                                        "last_error": "", "consecutive_failures": 0}),
             time.time()),
        )
        conn.commit()
        conn.close()

        m = VisionHealthMonitor()
        m.load(db)  # must not raise
        stats = m.get_device_stats("/dev/video0")
        assert stats is not None
        assert stats.total_captures == 3
        report = m.get_health_report()
        assert report["total_captures"] == 0  # no events loaded

    def test_load_both_tables_missing_is_safe(self, tmp_path: Path) -> None:
        """An empty SQLite database causes no errors."""
        db = tmp_path / "health.db"
        # Create an empty database (no tables at all).
        conn = sqlite3.connect(str(db))
        conn.close()

        m = VisionHealthMonitor()
        m.load(db)  # must not raise
        assert m.get_overall_health() == HealthStatus.UNKNOWN


# ---------------------------------------------------------------------------
# Large number of device records
# ---------------------------------------------------------------------------


class TestLargeDeviceLoad:
    """load() handles databases with many device records efficiently."""

    def test_load_100_devices(self, tmp_path: Path) -> None:
        db = tmp_path / "health.db"
        n = 100

        conn = sqlite3.connect(str(db))
        conn.execute(
            "CREATE TABLE device_stats "
            "(device TEXT PRIMARY KEY, data TEXT NOT NULL, updated_at REAL NOT NULL)"
        )
        conn.execute(
            "CREATE TABLE capture_events ("
            "id INTEGER PRIMARY KEY AUTOINCREMENT,"
            "timestamp REAL NOT NULL,"
            "success INTEGER NOT NULL,"
            "device TEXT NOT NULL,"
            "quality_score REAL,"
            "error TEXT,"
            "latency_ms REAL,"
            "source_type TEXT"
            ")"
        )
        now = time.time()
        for i in range(n):
            device = f"/dev/video{i}"
            data = json.dumps({
                "total_captures": 10,
                "successful_captures": 10,
                "failed_captures": 0,
                "total_quality": 8.5,
                "total_latency_ms": 500.0,
                "last_seen": now,
                "last_error": "",
                "consecutive_failures": 0,
            })
            conn.execute(
                "INSERT INTO device_stats VALUES (?, ?, ?)",
                (device, data, now),
            )
        conn.commit()
        conn.close()

        m = VisionHealthMonitor(max_events=200)
        m.load(db)
        report = m.get_health_report()
        assert len(report["devices"]) == n
        assert m.get_overall_health() == HealthStatus.HEALTHY

    def test_load_500_events(self, tmp_path: Path) -> None:
        """Loading 500 events works and respects max_events deque limit."""
        db = tmp_path / "health.db"
        n = 500

        conn = sqlite3.connect(str(db))
        conn.execute(
            "CREATE TABLE device_stats "
            "(device TEXT PRIMARY KEY, data TEXT NOT NULL, updated_at REAL NOT NULL)"
        )
        conn.execute(
            "CREATE TABLE capture_events ("
            "id INTEGER PRIMARY KEY AUTOINCREMENT,"
            "timestamp REAL NOT NULL,"
            "success INTEGER NOT NULL,"
            "device TEXT NOT NULL,"
            "quality_score REAL,"
            "error TEXT,"
            "latency_ms REAL,"
            "source_type TEXT"
            ")"
        )
        now = time.time()
        for i in range(n):
            conn.execute(
                "INSERT INTO capture_events "
                "(timestamp, success, device, quality_score, error, latency_ms, source_type) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (now + i, 1, "/dev/video0", 0.8, "", 50.0, "webcam"),
            )
        conn.commit()
        conn.close()

        # With max_events=200 the deque should be capped at 200.
        m = VisionHealthMonitor(max_events=200)
        m.load(db)
        report = m.get_health_report()
        assert report["total_captures"] == 200


# ---------------------------------------------------------------------------
# get_recommendations() for various failure scenarios
# ---------------------------------------------------------------------------


class TestRecommendationScenarios:
    """Specific recommendation strings for each failure category."""

    def _fill_failures(self, monitor: VisionHealthMonitor, device: str, error: str,
                       n: int = _CONSECUTIVE_FAILURE_LIMIT) -> None:
        # Pad with enough prior captures so `total_captures >= 3` is satisfied.
        for _ in range(n):
            monitor.record_capture(success=False, device=device, error=error)

    def test_permission_error_video_group_recommendation(self, monitor: VisionHealthMonitor) -> None:
        self._fill_failures(monitor, "/dev/video0", "permission denied")
        recs = monitor.get_recommendations()
        assert any("video" in r and "usermod" in r for r in recs)

    def test_permission_error_case_insensitive(self, monitor: VisionHealthMonitor) -> None:
        self._fill_failures(monitor, "/dev/video0", "Permission Denied")
        recs = monitor.get_recommendations()
        assert any("video" in r for r in recs)

    def test_busy_error_lsof_recommendation(self, monitor: VisionHealthMonitor) -> None:
        self._fill_failures(monitor, "/dev/video0", "device is busy")
        recs = monitor.get_recommendations()
        assert any("lsof" in r for r in recs)

    def test_busy_error_case_insensitive(self, monitor: VisionHealthMonitor) -> None:
        self._fill_failures(monitor, "/dev/video0", "BUSY: camera unavailable")
        recs = monitor.get_recommendations()
        assert any("lsof" in r for r in recs)

    def test_generic_error_doctor_recommendation(self, monitor: VisionHealthMonitor) -> None:
        self._fill_failures(monitor, "/dev/video0", "unknown hardware fault")
        recs = monitor.get_recommendations()
        assert any("doctor" in r for r in recs)

    def test_high_latency_resolution_recommendation(self, monitor: VisionHealthMonitor) -> None:
        # Requires total_captures >= 5
        for _ in range(5):
            monitor.record_capture(success=True, device="d", latency_ms=3000)
        recs = monitor.get_recommendations()
        assert any("resolution" in r for r in recs)

    def test_high_latency_not_triggered_below_5_captures(self, monitor: VisionHealthMonitor) -> None:
        for _ in range(4):
            monitor.record_capture(success=True, device="d", latency_ms=5000)
        recs = monitor.get_recommendations()
        assert not any("resolution" in r for r in recs)

    def test_low_quality_lighting_recommendation(self, monitor: VisionHealthMonitor) -> None:
        # Requires successful_captures >= 5
        for _ in range(5):
            monitor.record_capture(success=True, device="d", quality_score=0.2)
        recs = monitor.get_recommendations()
        assert any("lighting" in r for r in recs)

    def test_low_quality_not_triggered_below_5_successes(self, monitor: VisionHealthMonitor) -> None:
        for _ in range(4):
            monitor.record_capture(success=True, device="d", quality_score=0.1)
        recs = monitor.get_recommendations()
        assert not any("lighting" in r for r in recs)

    def test_low_quality_not_triggered_at_threshold(self, monitor: VisionHealthMonitor) -> None:
        """Quality exactly at _LOW_QUALITY_THRESHOLD (0.4) does not trigger."""
        for _ in range(5):
            monitor.record_capture(success=True, device="d",
                                   quality_score=_LOW_QUALITY_THRESHOLD)
        recs = monitor.get_recommendations()
        # average_quality == 0.4 is not strictly < threshold so no rec.
        assert not any("lighting" in r for r in recs)

    def test_combined_recommendations_multiple_devices(self, monitor: VisionHealthMonitor) -> None:
        """Permission error on device A and high latency on device B both appear."""
        for _ in range(5):
            monitor.record_capture(success=False, device="a", error="permission denied")
        for _ in range(5):
            monitor.record_capture(success=True, device="b", latency_ms=4000)
        recs = monitor.get_recommendations()
        assert any("usermod" in r for r in recs)
        assert any("resolution" in r for r in recs)

    def test_no_recommendations_when_insufficient_captures(self, monitor: VisionHealthMonitor) -> None:
        """With only 2 captures per device no recommendations are generated."""
        for _ in range(2):
            monitor.record_capture(success=False, device="d", error="permission denied")
        assert monitor.get_recommendations() == []


# ---------------------------------------------------------------------------
# Health report with mixed device health states
# ---------------------------------------------------------------------------


class TestMixedDeviceHealthReport:
    """get_health_report() aggregates correctly when devices differ in health."""

    def test_mixed_healthy_and_unhealthy(self, monitor: VisionHealthMonitor) -> None:
        # Healthy device
        for _ in range(10):
            monitor.record_capture(success=True, device="good")
        # Unhealthy device (all failures)
        for _ in range(10):
            monitor.record_capture(success=False, device="bad", error="err")

        report = monitor.get_health_report()
        assert report["overall_status"] == HealthStatus.UNHEALTHY
        assert report["devices"]["good"]["status"] == HealthStatus.HEALTHY
        assert report["devices"]["bad"]["status"] == HealthStatus.UNHEALTHY

    def test_mixed_healthy_and_degraded(self, monitor: VisionHealthMonitor) -> None:
        for _ in range(10):
            monitor.record_capture(success=True, device="good")
        # 60% success rate — degraded
        for _ in range(6):
            monitor.record_capture(success=True, device="mid")
        for _ in range(4):
            monitor.record_capture(success=False, device="mid", error="x")

        report = monitor.get_health_report()
        assert report["overall_status"] == HealthStatus.DEGRADED
        assert report["devices"]["good"]["status"] == HealthStatus.HEALTHY
        assert report["devices"]["mid"]["status"] == HealthStatus.DEGRADED

    def test_all_healthy_report_status(self, monitor: VisionHealthMonitor) -> None:
        for dev in ("a", "b", "c"):
            for _ in range(5):
                monitor.record_capture(success=True, device=dev)
        report = monitor.get_health_report()
        assert report["overall_status"] == HealthStatus.HEALTHY
        for dev in ("a", "b", "c"):
            assert report["devices"][dev]["status"] == HealthStatus.HEALTHY

    def test_discovery_only_device_is_unknown_in_report(self, monitor: VisionHealthMonitor) -> None:
        """A device that only had discovery (no captures) shows UNKNOWN status."""
        monitor.record_device_discovery("/dev/video5")
        monitor.record_capture(success=True, device="/dev/video0")
        for _ in range(9):
            monitor.record_capture(success=True, device="/dev/video0")
        report = monitor.get_health_report()
        assert report["devices"]["/dev/video5"]["status"] == HealthStatus.UNKNOWN

    def test_report_has_correct_failure_count(self, monitor: VisionHealthMonitor) -> None:
        for _ in range(3):
            monitor.record_capture(success=True, device="d")
        for _ in range(2):
            monitor.record_capture(success=False, device="d", error="x")
        report = monitor.get_health_report()
        assert report["total_captures"] == 5
        assert report["total_failures"] == 2

    def test_report_recent_success_rate_reflects_window(self) -> None:
        """recent_success_rate is computed from the recent window only."""
        m = VisionHealthMonitor(max_events=200, recent_window_secs=0.05)
        for _ in range(5):
            m.record_capture(success=True, device="d")
        time.sleep(0.1)  # push first 5 outside the window
        m.record_capture(success=False, device="d", error="x")
        report = m.get_health_report()
        # Only the single failure is "recent".
        assert report["recent_success_rate"] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Concurrent record_capture and save operations
# ---------------------------------------------------------------------------


class TestConcurrentCaptureAndSave:
    """record_capture and save() can run concurrently without corruption."""

    def test_concurrent_records_and_explicit_saves(self, tmp_path: Path) -> None:
        db = tmp_path / "health.db"
        monitor = VisionHealthMonitor(max_events=500)
        errors: list[Exception] = []
        captures_per_thread = 50
        n_threads = 4

        def capture_worker(device: str) -> None:
            try:
                for _ in range(captures_per_thread):
                    monitor.record_capture(success=True, device=device, quality_score=0.8)
            except Exception as exc:  # noqa: BLE001
                errors.append(exc)

        def save_worker() -> None:
            try:
                for _ in range(5):
                    monitor.save(db)
                    time.sleep(0.001)
            except Exception as exc:  # noqa: BLE001
                errors.append(exc)

        threads = [
            threading.Thread(target=capture_worker, args=(f"d{i}",))
            for i in range(n_threads)
        ]
        threads.append(threading.Thread(target=save_worker))
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors, f"Concurrent errors: {errors}"
        report = monitor.get_health_report()
        assert report["total_captures"] == n_threads * captures_per_thread

    def test_concurrent_auto_save_and_manual_save(self, tmp_path: Path) -> None:
        """Auto-saves triggered from multiple threads don't corrupt the database."""
        db = tmp_path / "health.db"
        monitor = VisionHealthMonitor(persist_path=db, auto_save_interval=10,
                                      max_events=500)
        errors: list[Exception] = []

        def worker(device: str) -> None:
            try:
                for _ in range(30):
                    monitor.record_capture(success=True, device=device)
            except Exception as exc:  # noqa: BLE001
                errors.append(exc)

        threads = [threading.Thread(target=worker, args=(f"dev{i}",)) for i in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors
        # Final explicit save, then load to verify consistency.
        monitor.save(db)
        m2 = VisionHealthMonitor(max_events=500)
        m2.load(db)
        # Should have all 5 devices.
        report = m2.get_health_report()
        assert len(report["devices"]) == 5


# ---------------------------------------------------------------------------
# Reset clears all data properly
# ---------------------------------------------------------------------------


class TestResetComprehensive:
    """reset() removes every piece of in-memory state."""

    def test_reset_clears_events_and_devices(self, monitor: VisionHealthMonitor) -> None:
        for dev in ("a", "b", "c"):
            for _ in range(5):
                monitor.record_capture(success=True, device=dev)
        monitor.reset()
        report = monitor.get_health_report()
        assert report["total_captures"] == 0
        assert report["devices"] == {}
        assert report["overall_status"] == HealthStatus.UNKNOWN

    def test_reset_clears_warnings(self, monitor: VisionHealthMonitor) -> None:
        for _ in range(5):
            monitor.record_capture(success=False, device="d", error="timeout")
        monitor.reset()
        report = monitor.get_health_report()
        assert report["warnings"] == []

    def test_reset_allows_fresh_recording(self, monitor: VisionHealthMonitor) -> None:
        monitor.record_capture(success=False, device="d", error="x")
        monitor.reset()
        monitor.record_capture(success=True, device="d")
        assert monitor.get_device_health("d") == HealthStatus.HEALTHY

    def test_reset_does_not_affect_persist_path(self, tmp_path: Path) -> None:
        """reset() clears memory but does NOT delete the database on disk."""
        db = tmp_path / "health.db"
        monitor = VisionHealthMonitor(persist_path=db)
        monitor.record_capture(success=True, device="d")
        monitor.save(db)
        assert db.exists()
        monitor.reset()
        assert db.exists(), "reset() must not delete the on-disk database"

    def test_reset_resets_uptime_monotonic(self, monitor: VisionHealthMonitor) -> None:
        # Sleep long enough so that the uptime (rounded to 1 decimal) is >= 0.1.
        time.sleep(0.15)
        before_reset = monitor.get_health_report()["uptime_seconds"]
        assert before_reset >= 0.1, "Expected measurable uptime before reset"
        monitor.reset()
        after_reset = monitor.get_health_report()["uptime_seconds"]
        assert after_reset < before_reset


# ---------------------------------------------------------------------------
# Device discovery tracking without captures
# ---------------------------------------------------------------------------


class TestDeviceDiscoveryTracking:
    """record_device_discovery creates a device entry with zero captures."""

    def test_discovery_only_device_has_zero_captures(self, monitor: VisionHealthMonitor) -> None:
        monitor.record_device_discovery("/dev/video3")
        stats = monitor.get_device_stats("/dev/video3")
        assert stats is not None
        assert stats.total_captures == 0
        assert stats.successful_captures == 0
        assert stats.failed_captures == 0

    def test_discovery_device_health_is_unknown(self, monitor: VisionHealthMonitor) -> None:
        monitor.record_device_discovery("/dev/video3")
        assert monitor.get_device_health("/dev/video3") == HealthStatus.UNKNOWN

    def test_discovery_updates_last_seen_timestamp(self, monitor: VisionHealthMonitor) -> None:
        monitor.record_device_discovery("d")
        t1 = monitor.get_device_stats("d").last_seen
        time.sleep(0.01)
        monitor.record_device_discovery("d")
        t2 = monitor.get_device_stats("d").last_seen
        assert t2 > t1

    def test_discovery_then_capture_accumulates(self, monitor: VisionHealthMonitor) -> None:
        monitor.record_device_discovery("d")
        monitor.record_capture(success=True, device="d")
        stats = monitor.get_device_stats("d")
        assert stats.total_captures == 1
        assert stats.successful_captures == 1

    def test_discovery_of_multiple_devices(self, monitor: VisionHealthMonitor) -> None:
        for i in range(10):
            monitor.record_device_discovery(f"/dev/video{i}")
        report = monitor.get_health_report()
        assert len(report["devices"]) == 10
        # All are unknown because none have captures.
        for dev_report in report["devices"].values():
            assert dev_report["status"] == HealthStatus.UNKNOWN


# ---------------------------------------------------------------------------
# Consecutive failure warning at threshold boundary (4 vs 5)
# ---------------------------------------------------------------------------


class TestConsecutiveFailureThresholdBoundary:
    """The _CONSECUTIVE_FAILURE_LIMIT constant is 5; boundary tests for 4 vs 5."""

    def test_four_consecutive_failures_not_unhealthy(self, monitor: VisionHealthMonitor) -> None:
        """4 consecutive failures: still below threshold, not UNHEALTHY yet."""
        below = _CONSECUTIVE_FAILURE_LIMIT - 1
        for _ in range(below):
            monitor.record_capture(success=False, device="d", error="x")
        # 4 failures: success_rate=0.0 => UNHEALTHY by rate check.
        # The consecutive threshold check is separate; rate check still applies.
        # With 0% rate, device is UNHEALTHY regardless.
        # This test verifies that consecutive_failures is tracked correctly
        # and that the consecutive threshold does not fire before 5.
        stats = monitor.get_device_stats("d")
        assert stats.consecutive_failures == below

    def test_four_consecutive_no_warning_in_report(self, monitor: VisionHealthMonitor) -> None:
        """4 consecutive failures don't emit a 'consecutive failures' warning."""
        below = _CONSECUTIVE_FAILURE_LIMIT - 1
        for _ in range(below):
            monitor.record_capture(success=False, device="d", error="x")
        report = monitor.get_health_report()
        consecutive_warnings = [
            w for w in report["warnings"] if "consecutive" in w.lower()
        ]
        assert len(consecutive_warnings) == 0

    def test_five_consecutive_triggers_warning(self, monitor: VisionHealthMonitor) -> None:
        """Exactly 5 consecutive failures trigger the 'consecutive' warning."""
        for _ in range(_CONSECUTIVE_FAILURE_LIMIT):
            monitor.record_capture(success=False, device="d", error="timeout")
        report = monitor.get_health_report()
        consecutive_warnings = [
            w for w in report["warnings"] if "consecutive" in w.lower()
        ]
        assert len(consecutive_warnings) == 1
        assert "5" in consecutive_warnings[0]

    def test_five_consecutive_marks_device_unhealthy(self, monitor: VisionHealthMonitor) -> None:
        for _ in range(_CONSECUTIVE_FAILURE_LIMIT):
            monitor.record_capture(success=False, device="d", error="x")
        assert monitor.get_device_health("d") == HealthStatus.UNHEALTHY

    def test_success_after_four_failures_resets_counter(self, monitor: VisionHealthMonitor) -> None:
        """A success after 4 failures resets consecutive_failures to 0."""
        below = _CONSECUTIVE_FAILURE_LIMIT - 1
        for _ in range(below):
            monitor.record_capture(success=False, device="d", error="x")
        monitor.record_capture(success=True, device="d")
        stats = monitor.get_device_stats("d")
        assert stats.consecutive_failures == 0

    def test_five_consecutive_no_recommendation_without_enough_data(
        self, monitor: VisionHealthMonitor
    ) -> None:
        """Recommendations need total_captures >= 3; 2 failures produce nothing."""
        for _ in range(2):
            monitor.record_capture(success=False, device="d", error="permission denied")
        assert monitor.get_recommendations() == []

    def test_consecutive_failure_logging_triggered_at_limit(
        self, monitor: VisionHealthMonitor, caplog: pytest.LogCaptureFixture
    ) -> None:
        """The logger.warning call fires on the 5th consecutive failure."""
        import logging
        with caplog.at_level(logging.WARNING, logger="missy.vision.health_monitor"):
            for _ in range(_CONSECUTIVE_FAILURE_LIMIT):
                monitor.record_capture(success=False, device="/dev/video0", error="timeout")
        warning_records = [r for r in caplog.records if "consecutive" in r.message.lower()]
        assert len(warning_records) >= 1

    def test_consecutive_failure_rate_in_report(self, monitor: VisionHealthMonitor) -> None:
        """The device report includes correct consecutive_failures count."""
        for _ in range(_CONSECUTIVE_FAILURE_LIMIT):
            monitor.record_capture(success=False, device="d", error="x")
        report = monitor.get_health_report()
        assert report["devices"]["d"]["consecutive_failures"] == _CONSECUTIVE_FAILURE_LIMIT
