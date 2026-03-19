"""Stress and boundary tests for scene memory and SceneManager."""

from __future__ import annotations

import threading

import numpy as np

from missy.vision.scene_memory import (
    SceneManager,
    SceneSession,
    TaskType,
    compute_phash,
    hamming_distance,
)


class TestSceneSessionStress:
    """Stress tests for SceneSession."""

    def test_large_frame_count(self) -> None:
        """Session handles many frames with eviction."""
        session = SceneSession("stress", max_frames=10)
        for i in range(100):
            img = np.full((10, 10, 3), i % 256, dtype=np.uint8)
            session.add_frame(img)
        assert session.frame_count == 10
        assert session._frame_counter == 100

    def test_frame_ids_monotonic(self) -> None:
        session = SceneSession("ids")
        frames = []
        for _ in range(20):
            f = session.add_frame(np.zeros((10, 10, 3), dtype=np.uint8), deduplicate=False)
            frames.append(f.frame_id)
        assert frames == list(range(1, 21))

    def test_get_frame_after_eviction(self) -> None:
        session = SceneSession("evict", max_frames=3)
        for i in range(5):
            session.add_frame(np.full((10, 10, 3), i, dtype=np.uint8))
        # Frames 1-2 evicted, only 3-5 remain
        assert session.get_frame(1) is None
        assert session.get_frame(2) is None
        assert session.get_frame(3) is not None

    def test_observations_accumulate(self) -> None:
        session = SceneSession("obs")
        for i in range(50):
            session.add_observation(f"Note {i}")
        assert len(session.observations) == 50

    def test_state_updates(self) -> None:
        session = SceneSession("state")
        session.update_state(pieces_found=5)
        session.update_state(pieces_found=10, edge_count=4)
        assert session.state == {"pieces_found": 10, "edge_count": 4}

    def test_close_releases_data(self) -> None:
        session = SceneSession("release")
        for _ in range(10):
            session.add_frame(np.zeros((100, 100, 3), dtype=np.uint8))
        session.add_observation("test")
        session.update_state(x=1)
        session.close()
        assert not session.is_active
        assert session.frame_count == 0
        assert len(session.observations) == 0

    def test_summarize_after_close(self) -> None:
        session = SceneSession("sum", task_type=TaskType.PUZZLE)
        session.add_frame(np.zeros((10, 10, 3), dtype=np.uint8))
        session.close()
        summary = session.summarize()
        assert summary["task_id"] == "sum"
        assert summary["task_type"] == "puzzle"
        assert summary["active"] is False
        assert summary["frame_count"] == 1  # total counter preserved
        assert summary["frames_retained"] == 0  # cleared on close


class TestSceneManagerStress:
    """Stress tests for SceneManager."""

    def test_eviction_at_max_sessions(self) -> None:
        mgr = SceneManager(max_sessions=3)
        s1 = mgr.create_session("t1")
        mgr.create_session("t2")
        mgr.create_session("t3")
        # Close one so it's preferred for eviction
        s1.close()
        mgr.create_session("t4")
        sessions = mgr.list_sessions()
        task_ids = {s["task_id"] for s in sessions}
        assert "t4" in task_ids
        assert len(task_ids) == 3

    def test_eviction_prefers_inactive(self) -> None:
        mgr = SceneManager(max_sessions=2)
        s1 = mgr.create_session("t1")
        mgr.create_session("t2")
        s1.close()
        mgr.create_session("t3")
        assert mgr.get_session("t1") is None  # evicted (was inactive)
        assert mgr.get_session("t2") is not None
        assert mgr.get_session("t3") is not None

    def test_eviction_oldest_active_when_all_active(self) -> None:
        mgr = SceneManager(max_sessions=2)
        mgr.create_session("t1")
        mgr.create_session("t2")
        mgr.create_session("t3")
        assert mgr.get_session("t1") is None  # evicted (oldest)
        assert mgr.get_session("t3") is not None

    def test_close_all(self) -> None:
        mgr = SceneManager()
        for i in range(5):
            mgr.create_session(f"t{i}")
        mgr.close_all()
        sessions = mgr.list_sessions()
        assert all(not s["active"] for s in sessions)

    def test_get_active_session(self) -> None:
        mgr = SceneManager()
        s1 = mgr.create_session("t1")
        s2 = mgr.create_session("t2")
        s1.close()
        active = mgr.get_active_session()
        assert active is s2

    def test_no_active_session(self) -> None:
        mgr = SceneManager()
        s1 = mgr.create_session("t1")
        s1.close()
        assert mgr.get_active_session() is None

    def test_concurrent_session_creation(self) -> None:
        mgr = SceneManager(max_sessions=50)
        errors = []

        def create_sessions(prefix):
            try:
                for i in range(10):
                    mgr.create_session(f"{prefix}_{i}")
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=create_sessions, args=(f"t{j}",)) for j in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)

        assert len(errors) == 0
        sessions = mgr.list_sessions()
        assert len(sessions) <= 50


class TestChangeDetectionEdgeCases:
    """Edge cases for change detection."""

    def test_detect_latest_change_one_frame(self) -> None:
        session = SceneSession("test")
        session.add_frame(np.zeros((10, 10, 3), dtype=np.uint8))
        assert session.detect_latest_change() is None

    def test_detect_latest_change_two_frames(self) -> None:
        session = SceneSession("test")
        session.add_frame(np.zeros((10, 10, 3), dtype=np.uint8))
        session.add_frame(np.full((10, 10, 3), 200, dtype=np.uint8))
        change = session.detect_latest_change()
        assert change is not None
        assert change.change_score > 0

    def test_visualize_change(self) -> None:
        session = SceneSession("test")
        f1 = session.add_frame(np.zeros((64, 64, 3), dtype=np.uint8))
        f2 = session.add_frame(np.full((64, 64, 3), 200, dtype=np.uint8))
        diff_img = session.visualize_change(f1, f2)
        assert diff_img is not None
        assert diff_img.shape == (256, 256, 3)

    def test_visualize_identical_frames(self) -> None:
        session = SceneSession("test")
        img = np.full((64, 64, 3), 100, dtype=np.uint8)
        f1 = session.add_frame(img.copy(), deduplicate=False)
        f2 = session.add_frame(img.copy(), deduplicate=False)
        diff_img = session.visualize_change(f1, f2)
        assert diff_img is not None


class TestPhashBoundary:
    """Boundary tests for perceptual hashing."""

    def test_very_small_image(self) -> None:
        img = np.array([[[128, 64, 32]]], dtype=np.uint8)  # 1x1
        h = compute_phash(img)
        assert len(h) == 16

    def test_tall_narrow_image(self) -> None:
        img = np.random.randint(0, 256, (200, 5, 3), dtype=np.uint8)
        h = compute_phash(img)
        assert len(h) == 16

    def test_wide_short_image(self) -> None:
        img = np.random.randint(0, 256, (5, 200, 3), dtype=np.uint8)
        h = compute_phash(img)
        assert len(h) == 16

    def test_hamming_symmetry(self) -> None:
        assert hamming_distance("abcd", "efgh") == hamming_distance("efgh", "abcd")

    def test_triangle_inequality(self) -> None:
        """Hamming distance obeys triangle inequality."""
        a = "0000000000000000"
        b = "000000000000000f"
        c = "00000000000000ff"
        d_ab = hamming_distance(a, b)
        d_bc = hamming_distance(b, c)
        d_ac = hamming_distance(a, c)
        assert d_ac <= d_ab + d_bc
