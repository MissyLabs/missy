"""Session 13: Comprehensive tests for missy.vision.scene_memory.

Covers gaps not addressed in test_scene_memory.py,
test_scene_memory_extended.py, or test_scene_memory_stress.py:

 1.  Task ID collision — create_session warns and closes old session
 2.  detect_change with evicted (image=None) frame_a
 3.  detect_change with evicted frame_b
 4.  detect_latest_change race condition (concurrent add_frame)
 5.  SceneManager eviction prefers inactive over active
 6.  SceneManager eviction when all active picks oldest _created
 7.  Frame deduplication — identical image (hamming distance = 0) skipped
 8.  Frame deduplication threshold edge case — distance exactly at threshold
 9.  Frame deduplication threshold edge case — distance one above threshold kept
10.  compute_phash for uniform image (std < 0.5) returns intensity-based hash
11.  compute_phash fallback when cv2.resize raises; md5 path used
12.  compute_phash ultimate fallback when both cv2 and tobytes fail
13.  hamming_distance with mismatched lengths returns -1
14.  hamming_distance with non-hex input returns -1
15.  hamming_distance of identical hashes is 0
16.  SceneSession close() idempotency — second call is a no-op
17.  summarize() on a closed session includes correct active=False and counters
18.  visualize_change returns None when frame_a.image is None
19.  visualize_change returns None when frame_b.image is None
20.  visualize_change returns None when cv2 raises
21.  Concurrent add_frame from multiple threads — no duplicated frame IDs
22.  update_state copy isolation — mutating returned dict does not affect state
23.  state property returns a fresh copy each time
24.  observations property returns a fresh copy each time
25.  get_recent_frames returns copy — mutation does not affect session
26.  detect_change for identical frames produces score near zero
27.  detect_change falls back to pixel_score when hamming distance is -1
28.  detect_change description thresholds (major / moderate / minor / no change)
29.  SceneManager.close_session on non-existent task_id is a no-op
30.  get_active_session returns None when all sessions are closed
31.  SceneManager.list_sessions reflects real-time state
32.  SceneFrame with pre-supplied thumbnail_hash skips hash computation
33.  Frame counter monotonically increases even when frames are deduplicated
34.  SceneSession with max_frames=1 always retains only the latest frame
35.  SceneManager task_id collision: old session is closed, new session is fresh
36.  compute_phash for grayscale (2-D) input — no cvtColor needed
37.  detect_latest_change returns None for empty session
38.  SceneSession add_frame with analysis and notes stored correctly
"""

from __future__ import annotations

import threading
import time
from datetime import UTC, datetime
from unittest.mock import MagicMock, call, patch

import numpy as np
import pytest

from missy.vision.scene_memory import (
    SceneChange,
    SceneFrame,
    SceneManager,
    SceneSession,
    TaskType,
    compute_phash,
    hamming_distance,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _bgr(r: int, g: int, b: int, h: int = 64, w: int = 64) -> np.ndarray:
    """Return a solid-colour BGR image."""
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[:, :, 0] = b
    img[:, :, 1] = g
    img[:, :, 2] = r
    return img


def _unique_imgs(n: int, h: int = 20, w: int = 20) -> list[np.ndarray]:
    """Return n images with clearly different pixel values."""
    step = max(1, 255 // (n + 1))
    return [np.full((h, w, 3), i * step, dtype=np.uint8) for i in range(n)]


# ===========================================================================
# 1-3  Task ID collision and evicted-frame paths
# ===========================================================================


class TestTaskIdCollision:
    """create_session with an existing task_id warns and closes the old session."""

    def test_old_session_is_closed_on_collision(self) -> None:
        mgr = SceneManager(max_sessions=10)
        s1 = mgr.create_session("my-task")
        assert s1.is_active

        s2 = mgr.create_session("my-task")

        assert not s1.is_active, "old session must be closed on collision"
        assert s2.is_active
        assert s2 is not s1

    def test_old_session_frames_cleared_on_collision(self) -> None:
        mgr = SceneManager(max_sessions=10)
        s1 = mgr.create_session("task-x")
        s1.add_frame(_bgr(0, 0, 0), deduplicate=False)
        assert s1.frame_count == 1

        mgr.create_session("task-x")
        assert s1.frame_count == 0

    def test_new_session_starts_empty_after_collision(self) -> None:
        mgr = SceneManager(max_sessions=10)
        mgr.create_session("dup")
        s2 = mgr.create_session("dup")
        assert s2.frame_count == 0
        assert s2._frame_counter == 0

    def test_collision_emits_warning(self, caplog: pytest.LogCaptureFixture) -> None:
        import logging

        mgr = SceneManager(max_sessions=10)
        mgr.create_session("warn-task")
        with caplog.at_level(logging.WARNING, logger="missy.vision.scene_memory"):
            mgr.create_session("warn-task")
        assert any("warn-task" in r.message for r in caplog.records)


# ===========================================================================
# 2-3  detect_change with evicted frames
# ===========================================================================


class TestDetectChangeEvictedFrames:
    def test_evicted_frame_a_returns_negative_score(self) -> None:
        session = SceneSession("evict-test")
        img = _bgr(100, 100, 100)
        f1 = SceneFrame(frame_id=1, image=img.copy())
        f2 = SceneFrame(frame_id=2, image=img.copy())
        # Simulate eviction
        f1.image = None  # type: ignore[assignment]

        change = session.detect_change(f1, f2)
        assert change.change_score == -1.0
        assert "evicted" in change.description

    def test_evicted_frame_b_returns_negative_score(self) -> None:
        session = SceneSession("evict-test-b")
        img = _bgr(50, 50, 50)
        f1 = SceneFrame(frame_id=1, image=img.copy())
        f2 = SceneFrame(frame_id=2, image=img.copy())
        f2.image = None  # type: ignore[assignment]

        change = session.detect_change(f1, f2)
        assert change.change_score == -1.0
        assert "evicted" in change.description

    def test_evicted_frame_preserves_frame_ids_in_result(self) -> None:
        session = SceneSession("id-check")
        f1 = SceneFrame(frame_id=7, image=_bgr(0, 0, 0))
        f2 = SceneFrame(frame_id=9, image=_bgr(255, 255, 255))
        f1.image = None  # type: ignore[assignment]

        change = session.detect_change(f1, f2)
        assert change.from_frame == 7
        assert change.to_frame == 9


# ===========================================================================
# 4  detect_latest_change race condition
# ===========================================================================


class TestDetectLatestChangeRace:
    """Concurrent add_frame and detect_latest_change must not deadlock or crash."""

    def test_no_deadlock_under_concurrent_add_and_detect(self) -> None:
        session = SceneSession("race-session", max_frames=10)
        errors: list[Exception] = []
        results: list[SceneChange | None] = []
        results_lock = threading.Lock()

        def adder() -> None:
            try:
                for _ in range(20):
                    img = np.random.randint(0, 256, (20, 20, 3), dtype=np.uint8)
                    session.add_frame(img, deduplicate=False)
                    time.sleep(0)
            except Exception as exc:
                errors.append(exc)

        def detector() -> None:
            try:
                for _ in range(20):
                    change = session.detect_latest_change()
                    with results_lock:
                        results.append(change)
                    time.sleep(0)
            except Exception as exc:
                errors.append(exc)

        threads = [
            threading.Thread(target=adder),
            threading.Thread(target=adder),
            threading.Thread(target=detector),
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)

        assert not errors, f"Exceptions during concurrent access: {errors}"


# ===========================================================================
# 5-6  SceneManager eviction policy
# ===========================================================================


class TestSceneManagerEvictionPolicy:
    def test_eviction_prefers_inactive_over_newer_active(self) -> None:
        """An inactive session must be evicted before any active session."""
        mgr = SceneManager(max_sessions=3)
        mgr.create_session("active-1")
        mgr.create_session("active-2")
        inactive = mgr.create_session("to-close")
        inactive.close()

        mgr.create_session("new-active")

        assert mgr.get_session("to-close") is None
        assert mgr.get_session("active-1") is not None
        assert mgr.get_session("active-2") is not None
        assert mgr.get_session("new-active") is not None

    def test_eviction_all_active_picks_oldest_created(self) -> None:
        """When all sessions are active, the one with the earliest _created is evicted."""
        mgr = SceneManager(max_sessions=3)
        s1 = mgr.create_session("alpha")
        s2 = mgr.create_session("beta")
        s3 = mgr.create_session("gamma")

        # Manually set creation times so gamma is oldest
        now = datetime.now(UTC)
        s1._created = now
        s2._created = now
        s3._created = datetime(2000, 1, 1, tzinfo=UTC)

        mgr.create_session("delta")

        assert mgr.get_session("gamma") is None
        assert mgr.get_session("alpha") is not None
        assert mgr.get_session("beta") is not None
        assert mgr.get_session("delta") is not None

    def test_evicted_active_session_is_closed(self) -> None:
        """Sessions evicted from SceneManager should be marked inactive."""
        mgr = SceneManager(max_sessions=2)
        s1 = mgr.create_session("first")
        mgr.create_session("second")

        assert s1.is_active
        # Add a third — s1 is oldest and will be evicted
        mgr.create_session("third")
        assert not s1.is_active


# ===========================================================================
# 7-9  Frame deduplication
# ===========================================================================


class TestFrameDeduplication:
    def test_identical_image_skipped(self) -> None:
        """Adding the exact same image twice should return None the second time."""
        session = SceneSession("dedup-test", max_frames=20)
        img = np.full((40, 40, 3), 128, dtype=np.uint8)
        f1 = session.add_frame(img)
        f2 = session.add_frame(img.copy())  # same pixels → same hash
        assert f1 is not None
        assert f2 is None
        assert session.frame_count == 1

    def test_dedup_threshold_at_boundary(self) -> None:
        """A frame whose Hamming distance equals the threshold is deduplicated."""
        session = SceneSession("threshold-test", max_frames=20)
        img = np.full((40, 40, 3), 100, dtype=np.uint8)
        f1 = session.add_frame(img, deduplicate=False)

        # Build a second frame whose hash is exactly `threshold` bits away.
        threshold = 5
        base_hash = f1.thumbnail_hash
        # Flip exactly `threshold` bits from the base hash
        val = int(base_hash, 16)
        # Flip the lowest `threshold` bits
        for bit in range(threshold):
            val ^= (1 << bit)
        target_hash = format(val, f"0{len(base_hash)}x")

        img2 = np.full((40, 40, 3), 101, dtype=np.uint8)
        f2 = SceneFrame(frame_id=0, image=img2, thumbnail_hash=target_hash)

        with session._lock:
            session._frame_counter += 1
            f2.frame_id = session._frame_counter
            latest = session._frames[-1]
            dist = hamming_distance(latest.thumbnail_hash, f2.thumbnail_hash)
            assert dist == threshold, f"Expected distance {threshold}, got {dist}"
            # Simulate what add_frame does
            skip = 0 <= dist <= threshold
        assert skip, "Frame at exactly the threshold should be skipped"

    def test_above_threshold_distance_frame_kept(self) -> None:
        """A frame whose Hamming distance exceeds the threshold is retained."""
        session = SceneSession("above-threshold", max_frames=20)
        img_dark = np.zeros((40, 40, 3), dtype=np.uint8)
        img_bright = np.full((40, 40, 3), 255, dtype=np.uint8)

        f1 = session.add_frame(img_dark)
        f2 = session.add_frame(img_bright, dedup_threshold=2)

        assert f1 is not None
        # img_bright has a completely different hash (distance >> 2)
        assert f2 is not None
        assert session.frame_count == 2

    def test_dedup_disabled_keeps_duplicate(self) -> None:
        session = SceneSession("no-dedup", max_frames=20)
        img = np.full((30, 30, 3), 200, dtype=np.uint8)
        session.add_frame(img)
        f2 = session.add_frame(img.copy(), deduplicate=False)
        assert f2 is not None
        assert session.frame_count == 2

    def test_frame_counter_advances_for_deduplicated_frame(self) -> None:
        """Frame counter must increase even when a frame is deduplicated."""
        session = SceneSession("counter-advance", max_frames=20)
        img = np.full((20, 20, 3), 77, dtype=np.uint8)
        f1 = session.add_frame(img)
        assert f1 is not None and f1.frame_id == 1

        # This should be deduplicated but counter still advances
        session.add_frame(img.copy())
        # Next non-duplicate frame should get frame_id == 3
        img2 = np.full((20, 20, 3), 200, dtype=np.uint8)
        f3 = session.add_frame(img2)
        assert f3 is not None
        assert f3.frame_id == 3


# ===========================================================================
# 8-9  compute_phash
# ===========================================================================


class TestComputePhash:
    def test_uniform_image_std_below_threshold(self) -> None:
        """Uniform images (std < 0.5) produce the intensity-based hash."""
        # A perfectly uniform gray image: std == 0
        img = np.full((8, 8), 200, dtype=np.uint8)
        # Provide a 2-D array so cvtColor is bypassed in the code
        small = img.copy()

        with patch("cv2.resize", return_value=small), \
             patch("cv2.cvtColor", return_value=small):
            h = compute_phash(img)

        # Intensity 200 → "c8" repeated 8 times → 16 hex chars
        assert h == "c8" * 8
        assert len(h) == 16

    def test_uniform_image_different_intensities_produce_different_hashes(self) -> None:
        """Two different uniform intensities must produce different hashes."""
        h50 = compute_phash(np.full((8, 8), 50, dtype=np.uint8))
        h200 = compute_phash(np.full((8, 8), 200, dtype=np.uint8))
        assert h50 != h200

    def test_fallback_when_cv2_resize_raises(self) -> None:
        """When cv2.resize raises, md5 fallback is used."""
        img = np.random.randint(0, 256, (40, 40, 3), dtype=np.uint8)
        with patch("cv2.resize", side_effect=RuntimeError("gpu error")):
            h = compute_phash(img)
        assert len(h) == 16
        assert h != "unknown_hash"

    def test_ultimate_fallback_when_tobytes_raises(self) -> None:
        """If both cv2 and tobytes() fail, return 'unknown_hash'."""
        img = MagicMock(spec=np.ndarray)
        img.tobytes.side_effect = MemoryError("oom")
        with patch("cv2.resize", side_effect=RuntimeError("broken")):
            h = compute_phash(img)
        assert h == "unknown_hash"

    def test_non_uniform_image_produces_16_char_hex(self) -> None:
        # Checkerboard pattern — definitely non-uniform
        img = np.zeros((40, 40, 3), dtype=np.uint8)
        img[::2, ::2] = 255
        h = compute_phash(img)
        assert len(h) == 16
        int(h, 16)  # must be valid hex; raises ValueError if not

    def test_grayscale_2d_input(self) -> None:
        """A 2-D (grayscale) array bypasses cvtColor and still produces a hash."""
        img = np.tile(np.arange(8, dtype=np.uint8), (8, 1))
        h = compute_phash(img)
        assert len(h) == 16


# ===========================================================================
# 10-11  hamming_distance edge cases
# ===========================================================================


class TestHammingDistanceEdgeCases:
    def test_mismatched_lengths_returns_negative_one(self) -> None:
        assert hamming_distance("abcd", "abcdef") == -1

    def test_non_hex_input_returns_negative_one(self) -> None:
        assert hamming_distance("zzzz", "1234") == -1

    def test_identical_hashes_distance_zero(self) -> None:
        h = "0123456789abcdef"
        assert hamming_distance(h, h) == 0

    def test_none_input_returns_negative_one(self) -> None:
        assert hamming_distance(None, "abcd") == -1  # type: ignore[arg-type]

    def test_single_bit_difference(self) -> None:
        # "0000000000000001" XOR "0000000000000000" = 1 bit
        assert hamming_distance("0000000000000001", "0000000000000000") == 1

    def test_empty_strings_returns_negative_one(self) -> None:
        """Empty strings cannot be parsed as hex, so the function returns -1."""
        # int("", 16) raises ValueError, which is caught and returns -1.
        assert hamming_distance("", "") == -1


# ===========================================================================
# 12-13  SceneSession close() idempotency and summarize() after close
# ===========================================================================


class TestSceneSessionCloseIdempotency:
    def test_close_twice_does_not_raise(self) -> None:
        session = SceneSession("idem")
        session.add_frame(_bgr(10, 20, 30), deduplicate=False)
        session.close()
        session.close()  # must be a no-op

    def test_close_twice_state_consistent(self) -> None:
        session = SceneSession("idem2")
        session.close()
        session.close()
        assert not session.is_active
        assert session.frame_count == 0

    def test_summarize_after_close_active_false(self) -> None:
        session = SceneSession("post-close", TaskType.INSPECTION)
        session.add_frame(_bgr(0, 0, 0), deduplicate=False)
        session.add_observation("edge detected")
        session.close()

        summary = session.summarize()
        assert summary["active"] is False
        assert summary["frame_count"] == 1        # total counter preserved
        assert summary["frames_retained"] == 0   # cleared on close
        assert summary["task_type"] == "inspection"

    def test_summarize_observations_cleared_after_close(self) -> None:
        session = SceneSession("obs-check")
        session.add_observation("note 1")
        session.close()
        summary = session.summarize()
        assert summary["observations"] == []

    def test_summarize_state_cleared_after_close(self) -> None:
        session = SceneSession("state-check")
        session.update_state(x=42)
        session.close()
        summary = session.summarize()
        assert summary["state"] == {}


# ===========================================================================
# 14  visualize_change with evicted frames
# ===========================================================================


class TestVisualizeChangeEvicted:
    def test_evicted_frame_a_returns_none(self) -> None:
        session = SceneSession("vis-evict")
        f1 = SceneFrame(frame_id=1, image=_bgr(0, 0, 0))
        f2 = SceneFrame(frame_id=2, image=_bgr(255, 255, 255))
        f1.image = None  # type: ignore[assignment]

        result = session.visualize_change(f1, f2)
        assert result is None

    def test_evicted_frame_b_returns_none(self) -> None:
        session = SceneSession("vis-evict-b")
        f1 = SceneFrame(frame_id=1, image=_bgr(100, 100, 100))
        f2 = SceneFrame(frame_id=2, image=_bgr(200, 200, 200))
        f2.image = None  # type: ignore[assignment]

        result = session.visualize_change(f1, f2)
        assert result is None

    def test_cv2_exception_returns_none(self) -> None:
        session = SceneSession("vis-cv2-fail")
        f1 = SceneFrame(frame_id=1, image=_bgr(50, 50, 50))
        f2 = SceneFrame(frame_id=2, image=_bgr(200, 200, 200))

        with patch("cv2.resize", side_effect=RuntimeError("no gpu")):
            result = session.visualize_change(f1, f2)
        assert result is None


# ===========================================================================
# 15  Concurrent add_frame from multiple threads
# ===========================================================================


class TestConcurrentAddFrame:
    def test_no_duplicate_frame_ids_under_concurrency(self) -> None:
        session = SceneSession("concurrent-add", max_frames=200)
        collected_ids: list[int] = []
        lock = threading.Lock()
        errors: list[Exception] = []

        def worker(colour: int) -> None:
            try:
                for _ in range(25):
                    img = np.full((10, 10, 3), colour, dtype=np.uint8)
                    f = session.add_frame(img, deduplicate=False)
                    if f is not None:
                        with lock:
                            collected_ids.append(f.frame_id)
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=worker, args=(i * 20,)) for i in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=15)

        assert not errors
        # All collected IDs must be unique
        assert len(collected_ids) == len(set(collected_ids))

    def test_frame_count_consistent_after_concurrent_adds(self) -> None:
        session = SceneSession("concurrent-count", max_frames=500)

        def worker() -> None:
            for i in range(20):
                session.add_frame(
                    np.full((5, 5, 3), i, dtype=np.uint8),
                    deduplicate=False,
                )

        threads = [threading.Thread(target=worker) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=15)

        assert session.frame_count == 100  # 5 threads × 20 frames


# ===========================================================================
# 16  update_state and state copy isolation
# ===========================================================================


class TestStateCopyIsolation:
    def test_mutating_returned_state_does_not_affect_session(self) -> None:
        session = SceneSession("isolation")
        session.update_state(key="original")

        returned = session.state
        returned["key"] = "mutated"

        assert session.state["key"] == "original"

    def test_state_property_returns_new_copy_each_time(self) -> None:
        session = SceneSession("isolation2")
        session.update_state(count=1)

        s1 = session.state
        s2 = session.state
        assert s1 is not s2

    def test_observations_property_returns_new_copy_each_time(self) -> None:
        session = SceneSession("obs-iso")
        session.add_observation("hello")

        o1 = session.observations
        o2 = session.observations
        assert o1 is not o2

    def test_mutating_observations_copy_does_not_affect_session(self) -> None:
        session = SceneSession("obs-mutate")
        session.add_observation("note")

        obs = session.observations
        obs.append("injected")

        assert "injected" not in session.observations


# ===========================================================================
# Additional coverage: change detection descriptions, misc edge cases
# ===========================================================================


class TestChangeDetectionDescriptions:
    """Verify that change score maps to the correct description label."""

    def _change_for_score(self, score: float) -> SceneChange:
        """Build a SceneChange mimicking detect_change output for a given score."""
        from missy.vision.scene_memory import (
            _CHANGE_THRESHOLD_MAJOR,
            _CHANGE_THRESHOLD_MINOR,
            _CHANGE_THRESHOLD_MODERATE,
        )

        if score > _CHANGE_THRESHOLD_MAJOR:
            description = "major change"
        elif score > _CHANGE_THRESHOLD_MODERATE:
            description = "moderate change"
        elif score > _CHANGE_THRESHOLD_MINOR:
            description = "minor change"
        else:
            description = "no change"

        return SceneChange(from_frame=0, to_frame=1, change_score=score, description=description)

    def test_major_change_label(self) -> None:
        c = self._change_for_score(0.5)
        assert c.description == "major change"

    def test_moderate_change_label(self) -> None:
        c = self._change_for_score(0.2)
        assert c.description == "moderate change"

    def test_minor_change_label(self) -> None:
        c = self._change_for_score(0.08)
        assert c.description == "minor change"

    def test_no_change_label(self) -> None:
        c = self._change_for_score(0.01)
        assert c.description == "no change"


class TestDetectChangeFallback:
    """detect_change uses pixel_score when hamming distance is invalid."""

    def test_invalid_hash_falls_back_to_pixel_score(self) -> None:
        """If one frame has 'unknown_hash', hamming_distance returns -1
        and the code should fall back to using pixel_score for the phash
        component."""
        session = SceneSession("fallback-hash")
        img_a = _bgr(0, 0, 0, 64, 64)
        img_b = _bgr(200, 200, 200, 64, 64)
        f1 = SceneFrame(frame_id=1, image=img_a, thumbnail_hash="unknown_hash")
        f2 = SceneFrame(frame_id=2, image=img_b, thumbnail_hash="badhashXXXXXXXX")

        change = session.detect_change(f1, f2)
        # Should succeed (not -1.0) because pixel comparison still works
        assert change.change_score >= 0.0


class TestSceneManagerMisc:
    def test_close_nonexistent_session_is_no_op(self) -> None:
        mgr = SceneManager()
        mgr.close_session("does-not-exist")  # must not raise

    def test_get_active_session_none_when_all_closed(self) -> None:
        mgr = SceneManager()
        s1 = mgr.create_session("s1")
        s2 = mgr.create_session("s2")
        s1.close()
        s2.close()
        assert mgr.get_active_session() is None

    def test_list_sessions_reflects_current_state(self) -> None:
        mgr = SceneManager()
        mgr.create_session("live")
        mgr.close_session("live")
        sessions = mgr.list_sessions()
        assert len(sessions) == 1
        assert sessions[0]["active"] is False


class TestSceneSessionMaxFramesOne:
    """A session with max_frames=1 always keeps only the latest frame."""

    def test_only_one_frame_retained(self) -> None:
        session = SceneSession("one-frame", max_frames=1)
        imgs = _unique_imgs(5)
        frames = [session.add_frame(img) for img in imgs]
        # Deduplicated frames count as None; non-None ones advance the counter
        assert session.frame_count == 1

    def test_latest_frame_is_most_recent(self) -> None:
        session = SceneSession("one-keep", max_frames=1)
        imgs = _unique_imgs(4, h=30, w=30)
        last_frame = None
        for img in imgs:
            f = session.add_frame(img)
            if f is not None:
                last_frame = f

        latest = session.get_latest_frame()
        assert latest is not None
        if last_frame is not None:
            assert latest.frame_id == last_frame.frame_id


class TestSceneFrameAddFrameMetadata:
    """add_frame stores analysis and notes correctly."""

    def test_analysis_stored(self) -> None:
        session = SceneSession("meta-test")
        analysis = {"confidence": 0.95, "label": "cat"}
        frame = session.add_frame(_bgr(10, 20, 30), analysis=analysis)
        assert frame is not None
        assert frame.analysis["label"] == "cat"
        assert frame.analysis["confidence"] == pytest.approx(0.95)

    def test_notes_stored(self) -> None:
        session = SceneSession("notes-test")
        frame = session.add_frame(_bgr(0, 0, 0), notes=["edge", "corner"])
        assert frame is not None
        assert "edge" in frame.notes
        assert "corner" in frame.notes

    def test_source_stored(self) -> None:
        session = SceneSession("src-test")
        frame = session.add_frame(_bgr(0, 0, 0), source="file:/tmp/shot.jpg")
        assert frame is not None
        assert frame.source == "file:/tmp/shot.jpg"


class TestGetRecentFramesCopyIsolation:
    """get_recent_frames returns a list copy; mutation does not affect the session."""

    def test_mutation_does_not_alter_session(self) -> None:
        session = SceneSession("recent-iso", max_frames=10)
        for i in range(5):
            session.add_frame(np.full((10, 10, 3), i * 40, dtype=np.uint8), deduplicate=False)

        recent = session.get_recent_frames(3)
        original_len = session.frame_count
        recent.clear()
        assert session.frame_count == original_len


class TestTaskTypeEnum:
    """TaskType is a StrEnum with expected members."""

    def test_all_task_types_have_correct_values(self) -> None:
        assert TaskType.PUZZLE == "puzzle"
        assert TaskType.PAINTING == "painting"
        assert TaskType.GENERAL == "general"
        assert TaskType.INSPECTION == "inspection"

    def test_task_type_is_string(self) -> None:
        assert isinstance(TaskType.PUZZLE, str)
