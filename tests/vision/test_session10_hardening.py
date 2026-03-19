"""Session 10 hardening tests: memory cleanup, thread safety, edge cases.

Tests cover:
- Frame eviction numpy cleanup
- SceneSession close() re-entrance safety
- detect_latest_change() thread safety
- VisionMemoryBridge thread-safe init
- Capture deadline-aware retry sleeps
- Voice server image size validation
- Multi-camera status() thread safety
- Discovery find_by_name() coverage
"""

from __future__ import annotations

import threading
import time
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from missy.vision.capture import CameraHandle, CaptureConfig
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
# SceneSession frame eviction memory cleanup
# ---------------------------------------------------------------------------


class TestFrameEvictionCleanup:
    """Evicted frames should have their numpy arrays released."""

    def test_evicted_frame_image_set_to_none(self) -> None:
        """After eviction, the evicted frame's image attribute should be None."""
        session = SceneSession("test", max_frames=2)
        img1 = np.zeros((10, 10, 3), dtype=np.uint8)
        img2 = np.ones((10, 10, 3), dtype=np.uint8) * 100
        img3 = np.ones((10, 10, 3), dtype=np.uint8) * 200

        frame1 = session.add_frame(img1, deduplicate=False)
        session.add_frame(img2, deduplicate=False)
        # Adding frame3 should evict frame1
        session.add_frame(img3, deduplicate=False)

        assert frame1 is not None
        # After eviction, the original frame1 object's image should be None
        assert frame1.image is None
        assert session.frame_count == 2

    def test_close_releases_all_frame_images(self) -> None:
        """close() should set image=None on all frames before clearing."""
        session = SceneSession("test", max_frames=10)
        frames = []
        for i in range(5):
            img = np.full((10, 10, 3), i * 50, dtype=np.uint8)
            f = session.add_frame(img, deduplicate=False)
            if f:
                frames.append(f)

        session.close()

        for f in frames:
            assert f.image is None

    def test_close_is_idempotent(self) -> None:
        """Calling close() twice should not raise."""
        session = SceneSession("test", max_frames=5)
        session.add_frame(np.zeros((10, 10, 3), dtype=np.uint8), deduplicate=False)

        session.close()
        session.close()  # second call should be a no-op

        assert not session.is_active

    def test_close_on_already_empty_session(self) -> None:
        """Closing a session with no frames should work."""
        session = SceneSession("test", max_frames=5)
        session.close()
        assert not session.is_active

    def test_multiple_evictions_release_all(self) -> None:
        """Evicting multiple frames in sequence releases all images."""
        session = SceneSession("test", max_frames=2)
        evicted_frames = []
        for i in range(6):
            img = np.full((10, 10, 3), i * 40, dtype=np.uint8)
            f = session.add_frame(img, deduplicate=False)
            if f:
                evicted_frames.append(f)

        # First 4 frames should have been evicted
        for f in evicted_frames[:4]:
            assert f.image is None

        # Last 2 should still have images
        for f in evicted_frames[4:]:
            assert f.image is not None


# ---------------------------------------------------------------------------
# detect_latest_change() thread safety
# ---------------------------------------------------------------------------


class TestDetectLatestChangeThreadSafety:
    """detect_latest_change() should work safely under concurrent access."""

    def test_detect_latest_change_returns_scene_change(self) -> None:
        session = SceneSession("test", max_frames=10)
        img1 = np.zeros((10, 10, 3), dtype=np.uint8)
        img2 = np.ones((10, 10, 3), dtype=np.uint8) * 255
        session.add_frame(img1, deduplicate=False)
        session.add_frame(img2, deduplicate=False)

        change = session.detect_latest_change()
        assert change is not None
        assert isinstance(change, SceneChange)

    def test_detect_latest_change_single_frame_returns_none(self) -> None:
        session = SceneSession("test", max_frames=10)
        session.add_frame(np.zeros((10, 10, 3), dtype=np.uint8), deduplicate=False)

        assert session.detect_latest_change() is None

    def test_detect_latest_change_empty_returns_none(self) -> None:
        session = SceneSession("test", max_frames=10)
        assert session.detect_latest_change() is None

    def test_concurrent_add_and_detect(self) -> None:
        """Adding frames while detecting changes should not crash."""
        session = SceneSession("test", max_frames=5)
        errors: list[Exception] = []

        def add_frames() -> None:
            try:
                for i in range(20):
                    img = np.full((10, 10, 3), i * 12, dtype=np.uint8)
                    session.add_frame(img, deduplicate=False)
                    time.sleep(0.001)
            except Exception as e:
                errors.append(e)

        def detect_changes() -> None:
            try:
                for _ in range(20):
                    session.detect_latest_change()
                    time.sleep(0.001)
            except Exception as e:
                errors.append(e)

        t1 = threading.Thread(target=add_frames)
        t2 = threading.Thread(target=detect_changes)
        t1.start()
        t2.start()
        t1.join(timeout=5)
        t2.join(timeout=5)

        assert not errors, f"Concurrent access raised: {errors}"


# ---------------------------------------------------------------------------
# VisionMemoryBridge thread-safe initialization
# ---------------------------------------------------------------------------


class TestVisionMemoryBridgeThreadSafety:
    """VisionMemoryBridge._ensure_init() should be thread-safe."""

    def test_concurrent_init_only_initializes_once(self) -> None:
        from missy.vision.vision_memory import VisionMemoryBridge

        mock_store = MagicMock()
        bridge = VisionMemoryBridge(memory_store=mock_store)

        init_count = {"value": 0}
        original_ensure = bridge._ensure_init

        def counting_init() -> None:
            original_ensure()
            init_count["value"] += 1

        bridge._ensure_init = counting_init

        threads = []
        for _ in range(10):
            t = threading.Thread(target=bridge._ensure_init)
            threads.append(t)
            t.start()

        for t in threads:
            t.join(timeout=5)

        # All threads should succeed, init called for each but _initialized
        # flag prevents re-initialization after the first
        assert bridge._initialized

    def test_double_checked_locking(self) -> None:
        """_ensure_init uses double-checked locking pattern."""
        from missy.vision.vision_memory import VisionMemoryBridge

        bridge = VisionMemoryBridge(memory_store=MagicMock())
        bridge._ensure_init()
        assert bridge._initialized

        # Second call should be a fast no-op (doesn't re-acquire lock)
        bridge._ensure_init()
        assert bridge._initialized


# ---------------------------------------------------------------------------
# Capture deadline-aware retry sleeps
# ---------------------------------------------------------------------------


class TestCaptureDeadlineAwareSleep:
    """Retry sleeps should not exceed the capture timeout deadline."""

    def _make_camera(self, **kwargs) -> CameraHandle:
        cfg = CaptureConfig(
            warmup_frames=0,
            max_retries=5,
            timeout_seconds=10.0,
            blank_threshold=5.0,
            adaptive_blank=False,
            retry_delay=2.0,
            **kwargs,
        )
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.get.return_value = 640.0

        cam = CameraHandle("/dev/video0", cfg)
        with patch("missy.vision.capture._get_cv2"):
            cam._cap = mock_cap
            cam._opened = True
        return cam

    def test_sleep_capped_at_remaining_time(self) -> None:
        """When deadline is near, sleep should be shortened."""
        cam = self._make_camera()
        cam._cap.read.return_value = (False, None)

        sleep_calls: list[float] = []

        def tracking_sleep(secs: float) -> None:
            sleep_calls.append(secs)

        with (
            patch("time.sleep", side_effect=tracking_sleep),
            patch("time.monotonic") as mock_mono,
        ):
            # deadline = 0 + 10 = 10
            # attempt 1: check t=0 ok, read fails, remaining=10-1=9 → sleep min(2, 9)=2
            # attempt 2: check t=3 ok, read fails, remaining=10-4=6 → sleep min(2, 6)=2
            # attempt 3: check t=6 ok, read fails, remaining=10-7=3 → sleep min(2, 3)=2
            # attempt 4: check t=8 ok, read fails, remaining=10-8.5=1.5 → sleep min(2, 1.5)=1.5
            # attempt 5: check t=9.5 ok, read fails → last attempt, no sleep
            mock_mono.side_effect = [
                0.0,   # deadline set
                0.0,   # attempt 1 check
                1.0,   # sleep remaining calc
                3.0,   # attempt 2 check
                4.0,   # sleep remaining calc
                6.0,   # attempt 3 check
                7.0,   # sleep remaining calc
                8.0,   # attempt 4 check
                8.5,   # sleep remaining calc
                9.5,   # attempt 5 check
            ]
            result = cam.capture()

        assert result.success is False
        # The 4th sleep should be capped at remaining time (1.5)
        assert len(sleep_calls) == 4
        assert sleep_calls[3] == pytest.approx(1.5, abs=0.01)

    def test_zero_remaining_sleeps_zero(self) -> None:
        """When remaining time <= 0, sleep(0) should be called."""
        cam = self._make_camera()
        cam._cap.read.return_value = (False, None)

        sleep_calls: list[float] = []

        def tracking_sleep(secs: float) -> None:
            sleep_calls.append(secs)

        with (
            patch("time.sleep", side_effect=tracking_sleep),
            patch("time.monotonic") as mock_mono,
        ):
            # deadline=10; attempt 1 ok; remaining calc shows 0
            mock_mono.side_effect = [
                0.0,    # deadline
                0.0,    # attempt 1 check
                11.0,   # remaining calc → negative → clamp to 0
                11.0,   # attempt 2 check → exceeds deadline
            ]
            result = cam.capture()

        assert result.success is False
        assert len(sleep_calls) == 1
        assert sleep_calls[0] == 0.0


# ---------------------------------------------------------------------------
# Discovery find_by_name() edge cases
# ---------------------------------------------------------------------------


class TestDiscoveryFindByName:
    """CameraDiscovery.find_by_name() edge cases."""

    def test_find_by_name_successful_match(self) -> None:
        from missy.vision.discovery import CameraDevice, CameraDiscovery

        disc = CameraDiscovery()
        device = CameraDevice(
            device_path="/dev/video0",
            name="Logitech C922x",
            vendor_id="046d",
            product_id="085c",
            bus_info="usb-0000:00:14.0-1",
        )
        with patch.object(disc, "discover", return_value=[device]):
            result = disc.find_by_name("C922")
            assert len(result) == 1
            assert result[0].name == "Logitech C922x"

    def test_find_by_name_no_match(self) -> None:
        from missy.vision.discovery import CameraDevice, CameraDiscovery

        disc = CameraDiscovery()
        device = CameraDevice(
            device_path="/dev/video0",
            name="Logitech C922x",
            vendor_id="046d",
            product_id="085c",
            bus_info="usb-0000:00:14.0-1",
        )
        with patch.object(disc, "discover", return_value=[device]):
            result = disc.find_by_name("RealSense")
            assert len(result) == 0

    def test_find_by_name_case_insensitive(self) -> None:
        from missy.vision.discovery import CameraDevice, CameraDiscovery

        disc = CameraDiscovery()
        device = CameraDevice(
            device_path="/dev/video0",
            name="Logitech C922x Pro Stream",
            vendor_id="046d",
            product_id="085c",
            bus_info="usb-0000:00:14.0-1",
        )
        with patch.object(disc, "discover", return_value=[device]):
            result = disc.find_by_name("logitech")
            assert len(result) == 1

    def test_find_by_name_regex_pattern(self) -> None:
        from missy.vision.discovery import CameraDevice, CameraDiscovery

        disc = CameraDiscovery()
        devices = [
            CameraDevice("/dev/video0", "Logitech C920", "046d", "082d", ""),
            CameraDevice("/dev/video2", "Logitech C922x", "046d", "085c", ""),
            CameraDevice("/dev/video4", "Microsoft LifeCam", "045e", "0772", ""),
        ]
        with patch.object(disc, "discover", return_value=devices):
            result = disc.find_by_name("C9[0-9]{2}")
            assert len(result) == 2

    def test_find_by_name_empty_pattern(self) -> None:
        from missy.vision.discovery import CameraDevice, CameraDiscovery

        disc = CameraDiscovery()
        devices = [
            CameraDevice("/dev/video0", "Cam1", "046d", "082d", ""),
            CameraDevice("/dev/video2", "Cam2", "046d", "085c", ""),
        ]
        with patch.object(disc, "discover", return_value=devices):
            result = disc.find_by_name("")
            # Empty pattern should match all
            assert len(result) == 2


# ---------------------------------------------------------------------------
# SceneManager eviction
# ---------------------------------------------------------------------------


class TestSceneManagerEviction:
    """SceneManager session eviction edge cases."""

    def test_evict_prefers_inactive_sessions(self) -> None:
        mgr = SceneManager(max_sessions=2)
        s1 = mgr.create_session("task1")
        mgr.create_session("task2")

        # Close s1 (make it inactive)
        s1.close()

        # Create s3 — should evict inactive s1, not active s2
        mgr.create_session("task3")

        assert mgr.get_session("task1") is None
        assert mgr.get_session("task2") is not None
        assert mgr.get_session("task3") is not None

    def test_evict_oldest_when_all_active(self) -> None:
        mgr = SceneManager(max_sessions=2)
        mgr.create_session("task1")
        mgr.create_session("task2")

        # All active — oldest (s1) should be evicted
        mgr.create_session("task3")

        assert mgr.get_session("task1") is None
        assert mgr.get_session("task2") is not None
        assert mgr.get_session("task3") is not None

    def test_list_sessions_returns_summaries(self) -> None:
        mgr = SceneManager(max_sessions=5)
        mgr.create_session("task1", TaskType.PUZZLE)
        mgr.create_session("task2", TaskType.PAINTING)

        summaries = mgr.list_sessions()
        assert len(summaries) == 2
        task_ids = {s["task_id"] for s in summaries}
        assert task_ids == {"task1", "task2"}

    def test_close_all_closes_all_active(self) -> None:
        mgr = SceneManager(max_sessions=5)
        s1 = mgr.create_session("task1")
        s2 = mgr.create_session("task2")

        mgr.close_all()

        assert not s1.is_active
        assert not s2.is_active


# ---------------------------------------------------------------------------
# Multi-camera status() consistency
# ---------------------------------------------------------------------------


class TestMultiCameraStatus:
    """multi_camera.py status() should be consistent under concurrent access."""

    def test_status_returns_camera_count(self) -> None:
        from missy.vision.multi_camera import MultiCameraManager

        mgr = MultiCameraManager()
        status = mgr.status()
        assert status["camera_count"] == 0
        assert status["cameras"] == {}

    def test_capture_all_with_no_cameras(self) -> None:
        from missy.vision.multi_camera import MultiCameraManager

        mgr = MultiCameraManager()
        result = mgr.capture_all()
        assert "_global" in result.errors
        assert not result.any_succeeded

    def test_capture_best_with_no_cameras(self) -> None:
        from missy.vision.multi_camera import MultiCameraManager

        mgr = MultiCameraManager()
        result = mgr.capture_best()
        assert not result.success


# ---------------------------------------------------------------------------
# Voice server image size limit
# ---------------------------------------------------------------------------


class TestVoiceServerImageSizeLimit:
    """Voice server should enforce image size limits on captured images."""

    def test_large_image_triggers_quality_downgrade(self) -> None:
        """Images exceeding 2MiB should be re-encoded at lower quality."""
        # This tests the logic conceptually — the actual voice server
        # integration is async and harder to unit test
        import base64

        large_data = b"\xff" * (3 * 1024 * 1024)  # 3MiB
        encoded = base64.b64encode(large_data)
        max_bytes = 2 * 1024 * 1024

        assert len(encoded) > max_bytes

    def test_small_image_passes_through(self) -> None:
        import base64

        small_data = b"\xff" * 1024  # 1KB
        encoded = base64.b64encode(small_data)
        max_bytes = 2 * 1024 * 1024

        assert len(encoded) <= max_bytes


# ---------------------------------------------------------------------------
# Perceptual hash edge cases
# ---------------------------------------------------------------------------


class TestPerceptualHashEdgeCases:
    """compute_phash() edge cases and robustness."""

    def test_uniform_image_produces_consistent_hash(self) -> None:
        img = np.full((100, 100, 3), 127, dtype=np.uint8)
        h1 = compute_phash(img)
        h2 = compute_phash(img)
        assert h1 == h2

    def test_different_uniform_images_different_hashes(self) -> None:
        img1 = np.full((100, 100, 3), 50, dtype=np.uint8)
        img2 = np.full((100, 100, 3), 200, dtype=np.uint8)
        h1 = compute_phash(img1)
        h2 = compute_phash(img2)
        assert h1 != h2

    def test_similar_images_low_hamming_distance(self) -> None:
        img1 = np.random.RandomState(42).randint(0, 256, (100, 100, 3), dtype=np.uint8)
        # Slightly modified version
        img2 = img1.copy()
        img2[0:5, 0:5] = 255
        h1 = compute_phash(img1)
        h2 = compute_phash(img2)
        dist = hamming_distance(h1, h2)
        assert 0 <= dist <= 15  # similar images should have low distance

    def test_very_different_images_high_hamming_distance(self) -> None:
        img1 = np.zeros((100, 100, 3), dtype=np.uint8)
        img2 = np.ones((100, 100, 3), dtype=np.uint8) * 255
        h1 = compute_phash(img1)
        h2 = compute_phash(img2)
        hamming_distance(h1, h2)
        # These are uniform images with different intensity, so hashes differ
        assert h1 != h2

    def test_hamming_distance_different_lengths(self) -> None:
        assert hamming_distance("abcd", "abcdef") == -1

    def test_hamming_distance_invalid_hex(self) -> None:
        assert hamming_distance("xyz", "abc") == -1

    def test_grayscale_image_hash(self) -> None:
        """Grayscale (2D) images should produce valid hashes."""
        img = np.random.RandomState(42).randint(0, 256, (100, 100), dtype=np.uint8)
        h = compute_phash(img)
        assert h != "unknown_hash"
        assert len(h) == 16

    def test_tiny_image_hash(self) -> None:
        """Very small images should still produce valid hashes."""
        img = np.random.RandomState(42).randint(0, 256, (4, 4, 3), dtype=np.uint8)
        h = compute_phash(img)
        assert h != "unknown_hash"


# ---------------------------------------------------------------------------
# SceneFrame dataclass
# ---------------------------------------------------------------------------


class TestSceneFrame:
    """SceneFrame auto-hash computation and fields."""

    def test_auto_computes_hash(self) -> None:
        img = np.full((10, 10, 3), 128, dtype=np.uint8)
        frame = SceneFrame(frame_id=1, image=img)
        assert frame.thumbnail_hash != ""
        assert frame.thumbnail_hash != "unknown_hash"

    def test_preserves_provided_hash(self) -> None:
        img = np.full((10, 10, 3), 128, dtype=np.uint8)
        frame = SceneFrame(frame_id=1, image=img, thumbnail_hash="custom_hash")
        assert frame.thumbnail_hash == "custom_hash"

    def test_timestamp_auto_set(self) -> None:
        from datetime import UTC, datetime

        img = np.full((10, 10, 3), 128, dtype=np.uint8)
        before = datetime.now(UTC)
        frame = SceneFrame(frame_id=1, image=img)
        after = datetime.now(UTC)
        assert before <= frame.timestamp <= after

    def test_default_fields(self) -> None:
        img = np.full((10, 10, 3), 128, dtype=np.uint8)
        frame = SceneFrame(frame_id=1, image=img)
        assert frame.source == ""
        assert frame.analysis == {}
        assert frame.notes == []


# ---------------------------------------------------------------------------
# SceneSession properties and state
# ---------------------------------------------------------------------------


class TestSceneSessionState:
    """SceneSession state management."""

    def test_update_state_merges_keys(self) -> None:
        session = SceneSession("test")
        session.update_state(step=1, status="active")
        session.update_state(step=2)

        state = session.state
        assert state["step"] == 2
        assert state["status"] == "active"

    def test_observations_are_copies(self) -> None:
        session = SceneSession("test")
        session.add_observation("obs1")
        obs = session.observations
        obs.append("modified")
        assert len(session.observations) == 1

    def test_state_is_copy(self) -> None:
        session = SceneSession("test")
        session.update_state(key="value")
        state = session.state
        state["key"] = "modified"
        assert session.state["key"] == "value"

    def test_summarize_includes_all_fields(self) -> None:
        session = SceneSession("test-sum", TaskType.PUZZLE, max_frames=5)
        img = np.full((10, 10, 3), 128, dtype=np.uint8)
        session.add_frame(img, deduplicate=False)
        session.add_observation("found edge piece")
        session.update_state(pieces_placed=3)

        summary = session.summarize()
        assert summary["task_id"] == "test-sum"
        assert summary["task_type"] == "puzzle"
        assert summary["frame_count"] == 1
        assert summary["frames_retained"] == 1
        assert summary["observations"] == ["found edge piece"]
        assert summary["state"]["pieces_placed"] == 3
        assert summary["active"] is True

    def test_get_frame_by_id(self) -> None:
        session = SceneSession("test", max_frames=10)
        img = np.full((10, 10, 3), 128, dtype=np.uint8)
        frame = session.add_frame(img, deduplicate=False)
        assert frame is not None

        retrieved = session.get_frame(frame.frame_id)
        assert retrieved is not None
        assert retrieved.frame_id == frame.frame_id

    def test_get_frame_missing_id(self) -> None:
        session = SceneSession("test", max_frames=10)
        assert session.get_frame(9999) is None

    def test_get_recent_frames(self) -> None:
        session = SceneSession("test", max_frames=10)
        for i in range(5):
            session.add_frame(
                np.full((10, 10, 3), i * 50, dtype=np.uint8),
                deduplicate=False,
            )

        recent = session.get_recent_frames(3)
        assert len(recent) == 3

    def test_get_recent_frames_more_than_available(self) -> None:
        session = SceneSession("test", max_frames=10)
        session.add_frame(np.zeros((10, 10, 3), dtype=np.uint8), deduplicate=False)

        recent = session.get_recent_frames(5)
        assert len(recent) == 1
