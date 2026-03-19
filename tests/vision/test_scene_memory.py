"""Tests for missy.vision.scene_memory — task-scoped scene memory."""

from __future__ import annotations

import numpy as np

from missy.vision.scene_memory import (
    SceneFrame,
    SceneManager,
    SceneSession,
    TaskType,
)

# ---------------------------------------------------------------------------
# SceneFrame tests
# ---------------------------------------------------------------------------


class TestSceneFrame:
    def test_basic_creation(self):
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        frame = SceneFrame(frame_id=1, image=img, source="webcam:/dev/video0")
        assert frame.frame_id == 1
        assert frame.source == "webcam:/dev/video0"
        assert frame.thumbnail_hash != ""

    def test_hash_stability(self):
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        f1 = SceneFrame(frame_id=1, image=img)
        f2 = SceneFrame(frame_id=2, image=img)
        # Same image should produce same hash
        assert f1.thumbnail_hash == f2.thumbnail_hash

    def test_hash_differs_for_different_images(self):
        img1 = np.zeros((100, 100, 3), dtype=np.uint8)
        img2 = np.full((100, 100, 3), 255, dtype=np.uint8)
        f1 = SceneFrame(frame_id=1, image=img1)
        f2 = SceneFrame(frame_id=2, image=img2)
        assert f1.thumbnail_hash != f2.thumbnail_hash


# ---------------------------------------------------------------------------
# SceneSession tests
# ---------------------------------------------------------------------------


class TestSceneSession:
    def test_create_session(self):
        session = SceneSession("task-1", TaskType.PUZZLE)
        assert session.task_id == "task-1"
        assert session.task_type == TaskType.PUZZLE
        assert session.frame_count == 0
        assert session.is_active

    def test_add_frame(self):
        session = SceneSession("task-1")
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        frame = session.add_frame(img, source="webcam:/dev/video0")
        assert frame.frame_id == 1
        assert session.frame_count == 1

    def test_add_multiple_frames(self):
        session = SceneSession("task-1")
        for i in range(5):
            img = np.full((100, 100, 3), i * 50, dtype=np.uint8)
            session.add_frame(img)
        assert session.frame_count == 5

    def test_eviction_on_overflow(self):
        session = SceneSession("task-1", max_frames=3)
        for i in range(5):
            img = np.full((100, 100, 3), i * 50, dtype=np.uint8)
            session.add_frame(img)

        assert session.frame_count == 3
        # Oldest frames should be evicted
        assert session.get_frame(1) is None
        assert session.get_frame(2) is None
        assert session.get_frame(3) is not None

    def test_get_latest_frame(self):
        session = SceneSession("task-1")
        assert session.get_latest_frame() is None

        img = np.zeros((100, 100, 3), dtype=np.uint8)
        session.add_frame(img)
        latest = session.get_latest_frame()
        assert latest is not None
        assert latest.frame_id == 1

    def test_get_recent_frames(self):
        session = SceneSession("task-1")
        for _i in range(10):
            session.add_frame(np.zeros((10, 10, 3), dtype=np.uint8))

        recent = session.get_recent_frames(3)
        assert len(recent) == 3
        assert recent[-1].frame_id == 10

    def test_add_observation(self):
        session = SceneSession("task-1")
        session.add_observation("The puzzle has a blue sky section")
        assert "blue sky" in session.observations[0]

    def test_update_state(self):
        session = SceneSession("task-1", TaskType.PUZZLE)
        session.update_state(completed_sections=["sky"], pieces_remaining=50)
        assert session.state["completed_sections"] == ["sky"]
        assert session.state["pieces_remaining"] == 50

    def test_close_session(self):
        session = SceneSession("task-1")
        session.add_frame(np.zeros((100, 100, 3), dtype=np.uint8))
        session.close()

        assert not session.is_active

    def test_summarize(self):
        session = SceneSession("task-1", TaskType.PUZZLE)
        session.add_frame(np.zeros((10, 10, 3), dtype=np.uint8))
        session.add_observation("Found edge pieces")
        session.update_state(progress="25%")

        summary = session.summarize()
        assert summary["task_id"] == "task-1"
        assert summary["task_type"] == "puzzle"
        assert summary["frame_count"] == 1
        assert summary["active"] is True
        assert "Found edge pieces" in summary["observations"]

    def test_detect_change(self):
        session = SceneSession("task-1")

        img1 = np.zeros((100, 100, 3), dtype=np.uint8)
        img2 = np.full((100, 100, 3), 200, dtype=np.uint8)

        f1 = session.add_frame(img1)
        f2 = session.add_frame(img2)

        change = session.detect_change(f1, f2)
        assert change.change_score > 0.3  # significant change

    def test_detect_no_change(self):
        session = SceneSession("task-1")

        img = np.full((100, 100, 3), 128, dtype=np.uint8)
        f1 = session.add_frame(img.copy())
        f2 = session.add_frame(img.copy())

        change = session.detect_change(f1, f2)
        assert change.change_score < 0.01

    def test_detect_latest_change_insufficient_frames(self):
        session = SceneSession("task-1")
        assert session.detect_latest_change() is None

        session.add_frame(np.zeros((10, 10, 3), dtype=np.uint8))
        assert session.detect_latest_change() is None


# ---------------------------------------------------------------------------
# SceneManager tests
# ---------------------------------------------------------------------------


class TestSceneManager:
    def test_create_session(self):
        mgr = SceneManager()
        session = mgr.create_session("task-1", TaskType.PUZZLE)
        assert session.task_id == "task-1"
        assert session.task_type == TaskType.PUZZLE

    def test_get_session(self):
        mgr = SceneManager()
        mgr.create_session("task-1")
        s = mgr.get_session("task-1")
        assert s is not None
        assert s.task_id == "task-1"

    def test_get_nonexistent(self):
        mgr = SceneManager()
        assert mgr.get_session("nope") is None

    def test_get_active_session(self):
        mgr = SceneManager()
        mgr.create_session("task-1")
        mgr.create_session("task-2")

        active = mgr.get_active_session()
        assert active is not None
        assert active.task_id == "task-2"

    def test_close_session(self):
        mgr = SceneManager()
        mgr.create_session("task-1")
        mgr.close_session("task-1")
        s = mgr.get_session("task-1")
        assert s is not None
        assert not s.is_active

    def test_eviction_at_capacity(self):
        mgr = SceneManager(max_sessions=2)
        mgr.create_session("task-1")
        mgr.create_session("task-2")

        # Close task-1 so it's inactive
        mgr.close_session("task-1")

        # Creating a third should evict the inactive one
        mgr.create_session("task-3")
        assert mgr.get_session("task-1") is None

    def test_eviction_all_active(self):
        mgr = SceneManager(max_sessions=2)
        mgr.create_session("task-1")
        mgr.create_session("task-2")

        # All active — oldest should be evicted
        mgr.create_session("task-3")
        assert mgr.get_session("task-1") is None

    def test_close_all(self):
        mgr = SceneManager()
        mgr.create_session("task-1")
        mgr.create_session("task-2")
        mgr.close_all()

        for s in mgr.list_sessions():
            assert s["active"] is False

    def test_list_sessions(self):
        mgr = SceneManager()
        mgr.create_session("task-1", TaskType.PUZZLE)
        mgr.create_session("task-2", TaskType.PAINTING)

        listings = mgr.list_sessions()
        assert len(listings) == 2
        types = {entry["task_type"] for entry in listings}
        assert types == {"puzzle", "painting"}
