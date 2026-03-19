"""Extended tests for scene_memory.py — eviction, close cleanup, hash fallback."""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import MagicMock, patch

import numpy as np

from missy.vision.scene_memory import SceneFrame, SceneManager, SceneSession, TaskType


class TestSceneSessionClose:
    """Verify close() fully releases memory."""

    def test_close_clears_frames(self):
        session = SceneSession("test", max_frames=10)
        for _ in range(5):
            session.add_frame(np.zeros((50, 50, 3), dtype=np.uint8))
        assert session.frame_count == 5

        session.close()
        assert session.frame_count == 0
        assert session.get_latest_frame() is None

    def test_close_clears_observations(self):
        session = SceneSession("test")
        session.add_observation("Found edge piece")
        session.add_observation("Sky section identified")
        assert len(session.observations) == 2

        session.close()
        assert session.observations == []

    def test_close_clears_state(self):
        session = SceneSession("test")
        session.update_state(pieces_found=10, completion=0.3)
        assert session.state["pieces_found"] == 10

        session.close()
        assert session.state == {}

    def test_close_marks_inactive(self):
        session = SceneSession("test")
        assert session.is_active
        session.close()
        assert not session.is_active

    def test_close_preserves_task_id(self):
        session = SceneSession("puzzle-42")
        session.close()
        assert session.task_id == "puzzle-42"

    def test_summarize_after_close(self):
        session = SceneSession("test", TaskType.PUZZLE)
        session.add_frame(np.zeros((50, 50, 3), dtype=np.uint8))
        session.close()

        summary = session.summarize()
        assert summary["task_id"] == "test"
        assert summary["active"] is False
        assert summary["frame_count"] == 1  # total counter preserved
        assert summary["frames_retained"] == 0  # frames cleared


class TestSceneManagerEviction:
    """Verify eviction uses creation time, not insertion order."""

    def test_evict_inactive_preferred(self):
        """Inactive sessions should be evicted before active ones."""
        mgr = SceneManager(max_sessions=2)
        s1 = mgr.create_session("task-1")
        mgr.create_session("task-2")
        s1.close()  # mark inactive

        # Creating a third should evict s1 (inactive)
        mgr.create_session("task-3")
        assert mgr.get_session("task-1") is None
        assert mgr.get_session("task-2") is not None
        assert mgr.get_session("task-3") is not None

    def test_evict_oldest_when_all_active(self):
        """When all sessions are active, evict the oldest by creation time."""
        mgr = SceneManager(max_sessions=2)
        mgr.create_session("task-1")
        s2 = mgr.create_session("task-2")

        # Manually backdated s2 to be older than s1
        s2._created = datetime(2020, 1, 1, tzinfo=UTC)

        mgr.create_session("task-3")
        # s2 (oldest by creation time) should be evicted
        assert mgr.get_session("task-2") is None
        assert mgr.get_session("task-1") is not None
        assert mgr.get_session("task-3") is not None

    def test_evict_oldest_by_creation_not_insertion(self):
        """Eviction should use _created timestamp, not dict insertion order."""
        mgr = SceneManager(max_sessions=3)
        s1 = mgr.create_session("first")
        s2 = mgr.create_session("second")
        s3 = mgr.create_session("third")

        # Make "second" the oldest
        s2._created = datetime(2019, 1, 1, tzinfo=UTC)
        s1._created = datetime(2025, 1, 1, tzinfo=UTC)
        s3._created = datetime(2025, 6, 1, tzinfo=UTC)

        # Create fourth — should evict "second" (oldest by timestamp)
        mgr.create_session("fourth")
        assert mgr.get_session("second") is None
        assert mgr.get_session("first") is not None

    def test_close_all(self):
        mgr = SceneManager(max_sessions=5)
        for i in range(3):
            mgr.create_session(f"task-{i}")

        mgr.close_all()
        sessions = mgr.list_sessions()
        for s in sessions:
            assert not s["active"]

    def test_list_sessions(self):
        mgr = SceneManager()
        mgr.create_session("a", TaskType.PUZZLE)
        mgr.create_session("b", TaskType.PAINTING)

        sessions = mgr.list_sessions()
        assert len(sessions) == 2
        types = {s["task_type"] for s in sessions}
        assert types == {"puzzle", "painting"}


class TestSceneFrameHash:
    """Tests for hash computation fallback."""

    def test_hash_computed_for_normal_image(self):
        """aHash produces a 16-char hex string for non-uniform images."""
        img = np.random.randint(0, 256, (50, 50, 3), dtype=np.uint8)
        # Use a non-uniform 8x8 grayscale result so aHash is meaningful
        varied = np.arange(64, dtype=np.uint8).reshape((8, 8))
        with patch("cv2.resize", return_value=varied.reshape(8, 8, 1)), \
             patch("cv2.cvtColor", return_value=varied):
            frame = SceneFrame(frame_id=1, image=img)
            assert frame.thumbnail_hash != ""
            assert len(frame.thumbnail_hash) == 16

    def test_hash_fallback_on_cv2_error(self):
        img = np.random.randint(0, 256, (50, 50, 3), dtype=np.uint8)
        with patch("cv2.resize", side_effect=Exception("cv2 broken")):
            frame = SceneFrame(frame_id=1, image=img)
            assert frame.thumbnail_hash != ""
            assert frame.thumbnail_hash != "unknown_hash"

    def test_hash_ultimate_fallback(self):
        """If even tobytes fails, should return unknown_hash."""
        img = MagicMock()
        img.tobytes.side_effect = Exception("corrupted")

        with patch("cv2.resize", side_effect=Exception("cv2 broken")):
            frame = SceneFrame(frame_id=1, image=img)
            assert frame.thumbnail_hash == "unknown_hash"

    def test_hash_for_empty_image(self):
        img = np.array([], dtype=np.uint8)
        with patch("cv2.resize", side_effect=Exception("empty")):
            frame = SceneFrame(frame_id=1, image=img)
            # Should use fallback
            assert frame.thumbnail_hash != ""

    def test_hash_skipped_when_provided(self):
        img = np.zeros((10, 10, 3), dtype=np.uint8)
        frame = SceneFrame(frame_id=1, image=img, thumbnail_hash="custom_hash")
        assert frame.thumbnail_hash == "custom_hash"


class TestSceneSessionFrameEviction:
    def test_frame_eviction_at_limit(self):
        session = SceneSession("test", max_frames=3)
        for _ in range(5):
            session.add_frame(np.zeros((10, 10, 3), dtype=np.uint8))

        assert session.frame_count == 3
        # First two frames should have been evicted
        assert session.get_frame(1) is None
        assert session.get_frame(2) is None
        assert session.get_frame(3) is not None

    def test_get_recent_frames(self):
        session = SceneSession("test", max_frames=10)
        for _ in range(8):
            session.add_frame(np.zeros((10, 10, 3), dtype=np.uint8))

        recent = session.get_recent_frames(3)
        assert len(recent) == 3
        assert recent[-1].frame_id == 8

    def test_detect_latest_change_insufficient_frames(self):
        session = SceneSession("test")
        session.add_frame(np.zeros((10, 10, 3), dtype=np.uint8))
        assert session.detect_latest_change() is None
