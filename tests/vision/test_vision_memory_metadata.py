"""Hardening tests.


Covers the specific fixes made in session 8b:

1. VisionMemoryBridge.store_observation() — metadata with reserved keys
   ("observation_id", "session_id", "timestamp", etc.) must NOT override
   the core fields set by the method itself.

2. VisionHealthMonitor.load() — rows with corrupt/invalid JSON must be
   silently skipped; load() must not raise.

3. VisionHealthMonitor.record_capture() — the auto-save counter is reset
   *inside* the lock so a second concurrent caller cannot trigger a
   duplicate save for the same interval boundary.

All tests run without a real camera or real database by using mocks.
"""

from __future__ import annotations

import json
import sqlite3
import threading
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

from missy.vision.health_monitor import VisionHealthMonitor
from missy.vision.vision_memory import VisionMemoryBridge

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_sqlite_mock() -> MagicMock:
    """Return a SQLiteMemoryStore mock that records calls."""
    store = MagicMock()
    store.add_turn.return_value = None
    store.get_session_turns.return_value = []
    store.search.return_value = []
    store.get_recent.return_value = []
    return store


# ---------------------------------------------------------------------------
# 1. VisionMemoryBridge — reserved-key metadata filtering
# ---------------------------------------------------------------------------


class TestVisionMemoryMetadataFiltering:
    """Verify that caller-supplied metadata cannot overwrite core fields."""

    def _bridge_with_mock(self) -> tuple[VisionMemoryBridge, MagicMock]:
        store = _make_sqlite_mock()
        bridge = VisionMemoryBridge(memory_store=store)
        return bridge, store

    def test_observation_id_not_overridden(self) -> None:
        """A caller-supplied 'observation_id' in metadata must be dropped."""
        bridge, store = self._bridge_with_mock()

        attacker_id = "attacker-controlled-id"
        returned_id = bridge.store_observation(
            session_id="sess-x",
            task_type="general",
            observation="test obs",
            metadata={"observation_id": attacker_id},
        )

        # The returned ID must be a UUID generated internally, not the attacker value.
        assert returned_id != attacker_id
        # The metadata passed to add_turn must also carry the real ID.
        _, kwargs = store.add_turn.call_args
        stored_meta = kwargs["metadata"]
        assert stored_meta["observation_id"] == returned_id
        assert stored_meta["observation_id"] != attacker_id

    def test_session_id_not_overridden(self) -> None:
        """A caller-supplied 'session_id' in metadata must be dropped."""
        bridge, store = self._bridge_with_mock()

        real_session = "real-session-abc"
        bridge.store_observation(
            session_id=real_session,
            task_type="puzzle",
            observation="edge pieces found",
            metadata={"session_id": "injected-session"},
        )

        _, kwargs = store.add_turn.call_args
        stored_meta = kwargs["metadata"]
        assert stored_meta["session_id"] == real_session

    def test_timestamp_not_overridden(self) -> None:
        """A caller-supplied 'timestamp' in metadata must be dropped."""
        bridge, store = self._bridge_with_mock()

        bridge.store_observation(
            session_id="sess-ts",
            task_type="painting",
            observation="blue strokes visible",
            metadata={"timestamp": "1970-01-01T00:00:00+00:00"},
        )

        _, kwargs = store.add_turn.call_args
        stored_meta = kwargs["metadata"]
        # Should not be the epoch sentinel we injected.
        assert stored_meta["timestamp"] != "1970-01-01T00:00:00+00:00"

    def test_all_reserved_keys_are_dropped(self) -> None:
        """All eight reserved keys are stripped from the caller's metadata."""
        bridge, store = self._bridge_with_mock()

        reserved_payload = {
            "observation_id": "bad-oid",
            "session_id": "bad-sid",
            "task_type": "bad-type",
            "observation": "bad-obs",
            "confidence": -999.0,
            "source": "bad-source",
            "frame_id": -1,
            "timestamp": "bad-ts",
        }
        returned_id = bridge.store_observation(
            session_id="sess-all",
            task_type="general",
            observation="real observation",
            confidence=0.9,
            source="webcam:/dev/video0",
            frame_id=3,
            metadata=reserved_payload,
        )

        _, kwargs = store.add_turn.call_args
        m = kwargs["metadata"]

        assert m["observation_id"] == returned_id
        assert m["session_id"] == "sess-all"
        assert m["task_type"] == "general"
        assert m["observation"] == "real observation"
        assert m["confidence"] == 0.9
        assert m["source"] == "webcam:/dev/video0"
        assert m["frame_id"] == 3
        # timestamp must be a real ISO string, not the injected sentinel
        assert m["timestamp"] != "bad-ts"

    def test_non_reserved_metadata_is_preserved(self) -> None:
        """Metadata keys that are not reserved are passed through intact."""
        bridge, store = self._bridge_with_mock()

        bridge.store_observation(
            session_id="sess-extra",
            task_type="inspection",
            observation="crack detected",
            metadata={
                "camera_model": "Logitech C922",
                "operator": "alice",
            },
        )

        _, kwargs = store.add_turn.call_args
        m = kwargs["metadata"]
        assert m["camera_model"] == "Logitech C922"
        assert m["operator"] == "alice"

    def test_none_metadata_is_handled(self) -> None:
        """Passing metadata=None must not raise and core fields are set."""
        bridge, store = self._bridge_with_mock()

        obs_id = bridge.store_observation(
            session_id="sess-none",
            task_type="general",
            observation="no meta",
            metadata=None,
        )

        assert obs_id  # truthy UUID string
        store.add_turn.assert_called_once()

    def test_empty_metadata_is_handled(self) -> None:
        """Passing metadata={} must not raise and core fields are set."""
        bridge, store = self._bridge_with_mock()

        obs_id = bridge.store_observation(
            session_id="sess-empty",
            task_type="general",
            observation="empty meta",
            metadata={},
        )

        assert obs_id
        store.add_turn.assert_called_once()


# ---------------------------------------------------------------------------
# 2. VisionHealthMonitor.load() — corrupt JSON tolerance
# ---------------------------------------------------------------------------


class TestHealthMonitorLoadCorruptJson:
    """load() must not raise when rows contain invalid JSON."""

    def _make_db_with_rows(self, tmp_path: Path, rows: list[tuple[str, str]]) -> Path:
        """Create a real SQLite DB with the given (device, data) rows."""
        db = tmp_path / "health.db"
        conn = sqlite3.connect(str(db))
        conn.execute(
            "CREATE TABLE device_stats ("
            "  device TEXT PRIMARY KEY,"
            "  data TEXT NOT NULL,"
            "  updated_at REAL NOT NULL"
            ")"
        )
        for device, data in rows:
            conn.execute(
                "INSERT INTO device_stats VALUES (?, ?, ?)",
                (device, data, time.time()),
            )
        conn.execute(
            "CREATE TABLE capture_events ("
            "  id INTEGER PRIMARY KEY AUTOINCREMENT,"
            "  timestamp REAL NOT NULL,"
            "  success INTEGER NOT NULL,"
            "  device TEXT NOT NULL,"
            "  quality_score REAL,"
            "  error TEXT,"
            "  latency_ms REAL,"
            "  source_type TEXT"
            ")"
        )
        conn.commit()
        conn.close()
        return db

    def test_all_corrupt_rows_skipped(self, tmp_path: Path) -> None:
        """When every row has unparseable JSON, load() completes without raising."""
        db = self._make_db_with_rows(
            tmp_path,
            [
                ("/dev/video0", "NOT JSON AT ALL"),
                ("/dev/video1", "{broken json"),
            ],
        )
        monitor = VisionHealthMonitor()
        # Must not raise — both rows are caught by the JSONDecodeError handler
        monitor.load(db)
        # No valid device stats are loaded
        assert monitor.get_device_stats("/dev/video0") is None
        assert monitor.get_device_stats("/dev/video1") is None

    def test_corrupt_row_does_not_prevent_valid_rows(self, tmp_path: Path) -> None:
        """A mix of corrupt and valid rows: only valid rows contribute stats."""
        valid_data = json.dumps(
            {
                "total_captures": 10,
                "successful_captures": 8,
                "failed_captures": 2,
                "total_quality": 7.2,
                "total_latency_ms": 1500.0,
                "last_seen": time.time(),
                "last_error": "",
                "consecutive_failures": 0,
            }
        )
        db = self._make_db_with_rows(
            tmp_path,
            [
                ("/dev/video0", "CORRUPT"),
                ("/dev/video1", valid_data),
            ],
        )
        monitor = VisionHealthMonitor()
        monitor.load(db)

        stats = monitor.get_device_stats("/dev/video1")
        assert stats is not None
        assert stats.total_captures == 10
        assert stats.successful_captures == 8

    def test_single_corrupt_row_skipped_silently(self, tmp_path: Path) -> None:
        """A single corrupt row is skipped; no exception surfaces."""
        db = self._make_db_with_rows(
            tmp_path,
            [("/dev/video0", "}{invalid")],
        )
        monitor = VisionHealthMonitor()
        # Patch the logger to confirm a warning is emitted.
        with patch("missy.vision.health_monitor.logger") as mock_log:
            monitor.load(db)
            assert mock_log.warning.called
            warning_text = str(mock_log.warning.call_args)
            assert "corrupt" in warning_text.lower() or "skip" in warning_text.lower()

    def test_empty_table_loads_cleanly(self, tmp_path: Path) -> None:
        """An empty device_stats table must not cause errors."""
        db = self._make_db_with_rows(tmp_path, [])
        monitor = VisionHealthMonitor()
        monitor.load(db)
        assert monitor.get_overall_health().value == "unknown"


# ---------------------------------------------------------------------------
# 3. VisionHealthMonitor.record_capture() — auto-save counter reset in lock
# ---------------------------------------------------------------------------


class TestHealthMonitorAutoSaveCounterReset:
    """The counter is reset inside the lock to prevent duplicate saves."""

    def test_counter_reset_to_zero_after_interval(self, tmp_path: Path) -> None:
        """After reaching auto_save_interval captures, counter resets to 0."""
        db = tmp_path / "health.db"
        monitor = VisionHealthMonitor(
            persist_path=db,
            auto_save_interval=5,
        )

        with patch.object(monitor, "save") as mock_save:
            # Record exactly auto_save_interval captures
            for _ in range(5):
                monitor.record_capture(success=True, device="/dev/video0")

            # save() should have been triggered exactly once
            mock_save.assert_called_once()
            # Counter must be back to 0 so the next interval starts fresh
            assert monitor._capture_count_since_save == 0

    def test_counter_increments_before_interval(self, tmp_path: Path) -> None:
        """Counter increments normally before the interval is reached."""
        db = tmp_path / "health.db"
        monitor = VisionHealthMonitor(
            persist_path=db,
            auto_save_interval=10,
        )

        with patch.object(monitor, "save"):
            for i in range(1, 8):
                monitor.record_capture(success=True, device="/dev/video0")
                assert monitor._capture_count_since_save == i

    def test_save_called_once_not_twice_at_boundary(self, tmp_path: Path) -> None:
        """Exactly one save call happens when the boundary is hit, not two."""
        db = tmp_path / "health.db"
        monitor = VisionHealthMonitor(
            persist_path=db,
            auto_save_interval=3,
        )

        save_calls: list[float] = []

        def _record_save() -> None:
            save_calls.append(time.monotonic())

        with patch.object(monitor, "save", side_effect=_record_save):
            for _ in range(3):
                monitor.record_capture(success=True, device="/dev/video0")

        assert len(save_calls) == 1

    def test_second_interval_triggers_second_save(self, tmp_path: Path) -> None:
        """After the first auto-save, the counter restarts and triggers again."""
        db = tmp_path / "health.db"
        monitor = VisionHealthMonitor(
            persist_path=db,
            auto_save_interval=4,
        )

        with patch.object(monitor, "save") as mock_save:
            for _ in range(8):
                monitor.record_capture(success=True, device="/dev/video0")

            # Two full intervals of 4 each → two saves
            assert mock_save.call_count == 2
            assert monitor._capture_count_since_save == 0

    def test_no_save_without_persist_path(self) -> None:
        """Without a persist_path, auto-save is never triggered."""
        monitor = VisionHealthMonitor(auto_save_interval=2)

        with patch.object(monitor, "save") as mock_save:
            for _ in range(10):
                monitor.record_capture(success=True, device="/dev/video0")

            mock_save.assert_not_called()

    def test_concurrent_captures_save_exactly_once_per_interval(self, tmp_path: Path) -> None:
        """Concurrent threads hitting the boundary call save() exactly once.

        This is the core race-condition test.  The counter reset inside the
        lock means only one thread sees ``should_save=True`` for a given
        interval, regardless of how many threads reach the boundary
        simultaneously.
        """
        db = tmp_path / "health.db"
        interval = 50
        monitor = VisionHealthMonitor(
            persist_path=db,
            auto_save_interval=interval,
        )

        save_count = [0]
        save_lock = threading.Lock()

        def _count_save() -> None:
            with save_lock:
                save_count[0] += 1

        errors: list[Exception] = []

        def _worker(n: int) -> None:
            try:
                for _ in range(n):
                    monitor.record_capture(success=True, device="/dev/video0")
            except Exception as exc:
                errors.append(exc)

        with patch.object(monitor, "save", side_effect=_count_save):
            threads = [threading.Thread(target=_worker, args=(interval,)) for _ in range(4)]
            for t in threads:
                t.start()
            for t in threads:
                t.join()

        assert not errors, f"Thread errors: {errors}"
        # 4 threads × 50 captures = 200 total; interval=50 → expect exactly 4 saves
        assert save_count[0] == 4
