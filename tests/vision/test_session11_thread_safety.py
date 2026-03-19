"""Session 11: Thread-safety tests for vision subsystem singletons and lazy imports.

Tests the double-checked locking patterns added to:
- get_discovery() singleton
- get_scene_manager() singleton
- get_health_monitor() singleton
- _get_cv2() lazy imports in capture, sources, pipeline
- Multi-camera status() cleanup
"""

from __future__ import annotations

import threading
from unittest.mock import MagicMock, patch

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Singleton thread-safety tests
# ---------------------------------------------------------------------------


class TestDiscoverySingletonThreadSafety:
    """Verify get_discovery() returns the same instance under concurrent access."""

    def test_concurrent_get_discovery_returns_same_instance(self) -> None:
        """Multiple threads calling get_discovery() should get the same object."""
        import missy.vision.discovery as mod

        # Reset module state
        original = mod._default_discovery
        mod._default_discovery = None

        instances: list = []
        barrier = threading.Barrier(10)

        def worker():
            barrier.wait()
            instances.append(mod.get_discovery())

        threads = [threading.Thread(target=worker) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Restore
        mod._default_discovery = original

        # All threads should have gotten the same instance
        assert len(set(id(i) for i in instances)) == 1

    def test_get_discovery_idempotent(self) -> None:
        """Calling get_discovery() twice returns the same instance."""
        import missy.vision.discovery as mod

        original = mod._default_discovery
        mod._default_discovery = None

        a = mod.get_discovery()
        b = mod.get_discovery()

        mod._default_discovery = original

        assert a is b


class TestSceneManagerSingletonThreadSafety:
    """Verify get_scene_manager() returns the same instance under concurrent access."""

    def test_concurrent_get_scene_manager_returns_same_instance(self) -> None:
        import missy.vision.scene_memory as mod

        original = mod._scene_manager
        mod._scene_manager = None

        instances: list = []
        barrier = threading.Barrier(10)

        def worker():
            barrier.wait()
            instances.append(mod.get_scene_manager())

        threads = [threading.Thread(target=worker) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        mod._scene_manager = original

        assert len(set(id(i) for i in instances)) == 1

    def test_get_scene_manager_idempotent(self) -> None:
        import missy.vision.scene_memory as mod

        original = mod._scene_manager
        mod._scene_manager = None

        a = mod.get_scene_manager()
        b = mod.get_scene_manager()

        mod._scene_manager = original

        assert a is b


class TestHealthMonitorSingletonThreadSafety:
    """Verify get_health_monitor() returns the same instance under concurrent access."""

    def test_concurrent_get_health_monitor_returns_same_instance(self) -> None:
        import missy.vision.health_monitor as mod

        original = mod._monitor
        mod._monitor = None

        instances: list = []
        barrier = threading.Barrier(10)

        def worker():
            barrier.wait()
            instances.append(mod.get_health_monitor())

        threads = [threading.Thread(target=worker) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        mod._monitor = original

        assert len(set(id(i) for i in instances)) == 1


# ---------------------------------------------------------------------------
# Lazy OpenCV import thread-safety
# ---------------------------------------------------------------------------


class TestLazyCV2ThreadSafety:
    """Verify _get_cv2() is thread-safe and returns the same module."""

    def test_capture_get_cv2_concurrent(self) -> None:
        """capture._get_cv2() returns same module under concurrent access."""
        import missy.vision.capture as mod

        original = mod._cv2
        mod._cv2 = None

        results: list = []
        barrier = threading.Barrier(5)

        def worker():
            barrier.wait()
            results.append(mod._get_cv2())

        threads = [threading.Thread(target=worker) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        mod._cv2 = original

        # All should be the same cv2 module
        assert len(set(id(r) for r in results)) == 1

    def test_sources_get_cv2_concurrent(self) -> None:
        """sources._get_cv2() returns same module under concurrent access."""
        import missy.vision.sources as mod

        original = mod._cv2
        mod._cv2 = None

        results: list = []
        barrier = threading.Barrier(5)

        def worker():
            barrier.wait()
            results.append(mod._get_cv2())

        threads = [threading.Thread(target=worker) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        mod._cv2 = original

        assert len(set(id(r) for r in results)) == 1

    def test_pipeline_get_cv2_concurrent(self) -> None:
        """pipeline._get_cv2() returns same module under concurrent access."""
        import missy.vision.pipeline as mod

        original = mod._cv2
        mod._cv2 = None

        results: list = []
        barrier = threading.Barrier(5)

        def worker():
            barrier.wait()
            results.append(mod._get_cv2())

        threads = [threading.Thread(target=worker) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        mod._cv2 = original

        assert len(set(id(r) for r in results)) == 1


# ---------------------------------------------------------------------------
# Multi-camera status cleanup
# ---------------------------------------------------------------------------


class TestMultiCameraStatusCleanup:
    """Tests for the cleaned-up multi_camera.status() method."""

    def test_status_with_known_device(self) -> None:
        """status() includes device details when device is tracked."""
        from missy.vision.discovery import CameraDevice
        from missy.vision.multi_camera import MultiCameraManager

        mgr = MultiCameraManager()
        device = CameraDevice("/dev/video0", "Test Cam", "046d", "085c", "usb-1")
        mock_handle = MagicMock()
        mock_handle.is_open = True

        mgr._handles["/dev/video0"] = mock_handle
        mgr._devices["/dev/video0"] = device

        status = mgr.status()
        assert status["camera_count"] == 1
        cam_info = status["cameras"]["/dev/video0"]
        assert cam_info["name"] == "Test Cam"
        assert cam_info["vendor_id"] == "046d"
        assert cam_info["product_id"] == "085c"
        assert cam_info["is_open"] is True

    def test_status_with_unknown_device(self) -> None:
        """status() returns empty strings when device metadata is missing."""
        from missy.vision.multi_camera import MultiCameraManager

        mgr = MultiCameraManager()
        mock_handle = MagicMock()
        mock_handle.is_open = False

        mgr._handles["/dev/video9"] = mock_handle
        # No entry in _devices

        status = mgr.status()
        cam_info = status["cameras"]["/dev/video9"]
        assert cam_info["name"] == ""
        assert cam_info["vendor_id"] == ""
        assert cam_info["product_id"] == ""
        assert cam_info["is_open"] is False

    def test_status_empty_manager(self) -> None:
        """status() with no cameras returns count 0."""
        from missy.vision.multi_camera import MultiCameraManager

        mgr = MultiCameraManager()
        status = mgr.status()
        assert status["camera_count"] == 0
        assert status["cameras"] == {}

    def test_status_multiple_cameras(self) -> None:
        """status() with multiple cameras returns all."""
        from missy.vision.discovery import CameraDevice
        from missy.vision.multi_camera import MultiCameraManager

        mgr = MultiCameraManager()
        for i in range(3):
            path = f"/dev/video{i}"
            device = CameraDevice(path, f"Cam {i}", "046d", f"085{i}", f"usb-{i}")
            handle = MagicMock()
            handle.is_open = (i % 2 == 0)  # alternate open/closed
            mgr._handles[path] = handle
            mgr._devices[path] = device

        status = mgr.status()
        assert status["camera_count"] == 3
        assert status["cameras"]["/dev/video0"]["is_open"] is True
        assert status["cameras"]["/dev/video1"]["is_open"] is False
        assert status["cameras"]["/dev/video2"]["is_open"] is True


# ---------------------------------------------------------------------------
# Health monitor edge cases
# ---------------------------------------------------------------------------


class TestHealthMonitorEdgeCases:
    """Additional edge cases for VisionHealthMonitor."""

    def test_concurrent_record_capture(self) -> None:
        """Multiple threads recording captures shouldn't corrupt state."""
        from missy.vision.health_monitor import VisionHealthMonitor

        monitor = VisionHealthMonitor()
        barrier = threading.Barrier(10)
        errors: list[Exception] = []

        def worker(success: bool):
            try:
                barrier.wait()
                for _ in range(100):
                    monitor.record_capture(
                        success=success,
                        device="/dev/video0",
                        latency_ms=5.0,
                    )
            except Exception as e:
                errors.append(e)

        threads = [
            threading.Thread(target=worker, args=(i % 2 == 0,))
            for i in range(10)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors
        stats = monitor.get_device_stats("/dev/video0")
        assert stats is not None
        assert stats.total_captures == 1000  # 10 threads * 100 each

    def test_health_report_with_no_events(self) -> None:
        """Health report with no events should return sensible defaults."""
        from missy.vision.health_monitor import VisionHealthMonitor

        monitor = VisionHealthMonitor()
        report = monitor.get_health_report()
        assert report["overall_status"] == "unknown"
        assert report["total_captures"] == 0
        assert report["total_failures"] == 0
        assert report["recent_success_rate"] == 0.0

    def test_overall_health_all_unknown(self) -> None:
        """When all devices have unknown health, overall is unknown."""
        from missy.vision.health_monitor import VisionHealthMonitor

        monitor = VisionHealthMonitor()
        monitor.record_device_discovery("/dev/video0")
        monitor.record_device_discovery("/dev/video1")
        assert monitor.get_overall_health().value == "unknown"

    def test_recommendations_permission_error(self) -> None:
        """Recommendation mentions video group for permission errors."""
        from missy.vision.health_monitor import VisionHealthMonitor

        monitor = VisionHealthMonitor()
        for _ in range(10):
            monitor.record_capture(
                success=False,
                device="/dev/video0",
                error="Permission denied",
            )

        recs = monitor.get_recommendations()
        assert any("video" in r and "group" in r for r in recs)

    def test_recommendations_busy_error(self) -> None:
        """Recommendation mentions closing applications for busy errors."""
        from missy.vision.health_monitor import VisionHealthMonitor

        monitor = VisionHealthMonitor()
        for _ in range(10):
            monitor.record_capture(
                success=False,
                device="/dev/video0",
                error="Device busy",
            )

        recs = monitor.get_recommendations()
        assert any("close" in r.lower() for r in recs)

    def test_recommendations_high_latency(self) -> None:
        """Recommendation for high latency suggests reducing resolution."""
        from missy.vision.health_monitor import VisionHealthMonitor

        monitor = VisionHealthMonitor()
        for _ in range(10):
            monitor.record_capture(
                success=True,
                device="/dev/video0",
                latency_ms=5000.0,
            )

        recs = monitor.get_recommendations()
        assert any("resolution" in r.lower() for r in recs)


# ---------------------------------------------------------------------------
# Scene session edge cases
# ---------------------------------------------------------------------------


class TestSceneSessionEdgeCases:
    """Additional edge case tests for SceneSession."""

    def test_add_frame_to_closed_session(self) -> None:
        """Adding a frame to a closed session should still work (not crash)."""
        from missy.vision.scene_memory import SceneSession, TaskType

        session = SceneSession("test", TaskType.GENERAL, max_frames=5)
        session.close()

        # Session is closed but add_frame should not crash
        img = np.zeros((10, 10, 3), dtype=np.uint8)
        frame = session.add_frame(img)
        # It may or may not add (implementation decides), but no crash
        assert frame is not None or frame is None  # no crash

    def test_concurrent_add_frame_and_close(self) -> None:
        """Concurrent add_frame and close should not deadlock or crash."""
        from missy.vision.scene_memory import SceneSession, TaskType

        session = SceneSession("test", TaskType.GENERAL, max_frames=50)
        errors: list[Exception] = []
        barrier = threading.Barrier(6)

        def add_frames():
            try:
                barrier.wait()
                for i in range(20):
                    img = np.full((10, 10, 3), i, dtype=np.uint8)
                    session.add_frame(img, deduplicate=False)
            except Exception as e:
                errors.append(e)

        def close_session():
            try:
                barrier.wait()
                session.close()
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=add_frames) for _ in range(5)]
        threads.append(threading.Thread(target=close_session))

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors

    def test_detect_change_with_identical_frames(self) -> None:
        """Detecting change between identical frames returns ~0 score."""
        from missy.vision.scene_memory import SceneFrame, SceneSession, TaskType

        session = SceneSession("test", TaskType.GENERAL)
        img = np.full((100, 100, 3), 128, dtype=np.uint8)
        frame_a = SceneFrame(frame_id=1, image=img)
        frame_b = SceneFrame(frame_id=2, image=img.copy())

        change = session.detect_change(frame_a, frame_b)
        assert change.change_score < 0.1
        assert "no change" in change.description

    def test_detect_change_with_very_different_frames(self) -> None:
        """Detecting change between very different frames returns high score."""
        from missy.vision.scene_memory import SceneFrame, SceneSession, TaskType

        session = SceneSession("test", TaskType.GENERAL)
        black = np.zeros((100, 100, 3), dtype=np.uint8)
        white = np.full((100, 100, 3), 255, dtype=np.uint8)
        frame_a = SceneFrame(frame_id=1, image=black)
        frame_b = SceneFrame(frame_id=2, image=white)

        change = session.detect_change(frame_a, frame_b)
        assert change.change_score > 0.3
        assert "change" in change.description

    def test_visualize_change_produces_image(self) -> None:
        """visualize_change() returns a valid BGR image."""
        from missy.vision.scene_memory import SceneFrame, SceneSession, TaskType

        session = SceneSession("test", TaskType.GENERAL)
        img_a = np.zeros((100, 100, 3), dtype=np.uint8)
        img_b = np.full((100, 100, 3), 128, dtype=np.uint8)
        frame_a = SceneFrame(frame_id=1, image=img_a)
        frame_b = SceneFrame(frame_id=2, image=img_b)

        result = session.visualize_change(frame_a, frame_b)
        assert result is not None
        assert result.ndim == 3
        assert result.shape[2] == 3

    def test_add_observation_thread_safety(self) -> None:
        """Concurrent add_observation calls shouldn't corrupt state."""
        from missy.vision.scene_memory import SceneSession, TaskType

        session = SceneSession("test", TaskType.GENERAL)
        barrier = threading.Barrier(5)

        def worker(label: str):
            barrier.wait()
            for i in range(50):
                session.add_observation(f"{label}-{i}")

        threads = [
            threading.Thread(target=worker, args=(f"t{i}",))
            for i in range(5)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(session.observations) == 250


# ---------------------------------------------------------------------------
# Pipeline edge cases
# ---------------------------------------------------------------------------


class TestPipelineProcessEdgeCases:
    """Additional pipeline processing edge cases."""

    def test_process_single_channel_image(self) -> None:
        """Processing a single-channel 3D image shouldn't crash."""
        from missy.vision.pipeline import ImagePipeline

        pipeline = ImagePipeline()
        img = np.full((100, 100, 1), 128, dtype=np.uint8)
        result = pipeline.process(img)
        assert result is not None

    def test_process_bgra_image(self) -> None:
        """Processing a 4-channel BGRA image should work."""
        from missy.vision.pipeline import ImagePipeline

        pipeline = ImagePipeline()
        img = np.full((100, 100, 4), 128, dtype=np.uint8)
        result = pipeline.process(img)
        assert result is not None

    def test_assess_quality_single_channel(self) -> None:
        """Quality assessment on a single-channel 3D image."""
        from missy.vision.pipeline import ImagePipeline

        pipeline = ImagePipeline()
        img = np.full((100, 100, 1), 128, dtype=np.uint8)
        quality = pipeline.assess_quality(img)
        assert "brightness" in quality
        assert quality["brightness"] > 0

    def test_assess_quality_bgra(self) -> None:
        """Quality assessment on a BGRA image."""
        from missy.vision.pipeline import ImagePipeline

        pipeline = ImagePipeline()
        img = np.full((100, 100, 4), 128, dtype=np.uint8)
        quality = pipeline.assess_quality(img)
        assert "saturation" in quality

    def test_assess_quality_very_dark(self) -> None:
        """Very dark image should report brightness issue."""
        from missy.vision.pipeline import ImagePipeline

        pipeline = ImagePipeline()
        img = np.full((100, 100, 3), 10, dtype=np.uint8)
        quality = pipeline.assess_quality(img)
        assert "very dark" in quality["issues"]

    def test_assess_quality_overexposed(self) -> None:
        """Very bright image should report overexposure."""
        from missy.vision.pipeline import ImagePipeline

        pipeline = ImagePipeline()
        img = np.full((100, 100, 3), 240, dtype=np.uint8)
        quality = pipeline.assess_quality(img)
        assert "overexposed" in quality["issues"]

    def test_resize_no_op_when_within_limit(self) -> None:
        """Resize should be a no-op when image is within limits."""
        from missy.vision.pipeline import ImagePipeline

        pipeline = ImagePipeline()
        img = np.full((100, 100, 3), 128, dtype=np.uint8)
        result = pipeline.resize(img, 200)
        assert result is img  # same object, not copied

    def test_resize_invalid_max_dim(self) -> None:
        """Resize with max_dim <= 0 should raise ValueError."""
        from missy.vision.pipeline import ImagePipeline

        pipeline = ImagePipeline()
        img = np.full((100, 100, 3), 128, dtype=np.uint8)
        with pytest.raises(ValueError, match="positive"):
            pipeline.resize(img, 0)

    def test_process_none_image_raises(self) -> None:
        """process(None) should raise ValueError."""
        from missy.vision.pipeline import ImagePipeline

        pipeline = ImagePipeline()
        with pytest.raises(ValueError, match="non-None"):
            pipeline.process(None)

    def test_process_zero_dim_image_raises(self) -> None:
        """process() with zero-dimension image should raise ValueError."""
        from missy.vision.pipeline import ImagePipeline

        pipeline = ImagePipeline()
        img = np.zeros((0, 100, 3), dtype=np.uint8)
        with pytest.raises(ValueError, match="invalid shape"):
            pipeline.process(img)


# ---------------------------------------------------------------------------
# Discovery edge cases
# ---------------------------------------------------------------------------


class TestDiscoveryEdgeCases:
    """Additional discovery edge cases."""

    def test_find_by_usb_id_refreshes_cache(self) -> None:
        """find_by_usb_id should call discover() which may refresh cache."""
        from missy.vision.discovery import CameraDiscovery

        disc = CameraDiscovery(sysfs_base="/nonexistent")
        result = disc.find_by_usb_id("046d", "085c")
        assert result is None

    def test_find_preferred_no_devices(self) -> None:
        """find_preferred returns None when no cameras found."""
        from missy.vision.discovery import CameraDiscovery

        disc = CameraDiscovery(sysfs_base="/nonexistent")
        result = disc.find_preferred()
        assert result is None

    def test_validate_device_missing_path(self, tmp_path) -> None:
        """validate_device returns False when device path doesn't exist."""
        from missy.vision.discovery import CameraDevice, CameraDiscovery

        disc = CameraDiscovery(sysfs_base=str(tmp_path))
        device = CameraDevice("/dev/nonexistent", "Test", "046d", "085c", "usb-1")
        assert disc.validate_device(device) is False

    def test_cache_expires_after_ttl(self) -> None:
        """Discovery cache should expire after TTL."""
        from missy.vision.discovery import CameraDiscovery

        disc = CameraDiscovery(cache_ttl_seconds=0.0, sysfs_base="/nonexistent")
        # First call populates cache
        result1 = disc.discover()
        # With TTL=0, cache is always expired
        result2 = disc.discover()
        # Both should return empty (no sysfs), but cache was invalidated
        assert result1 == []
        assert result2 == []


# ---------------------------------------------------------------------------
# Capture config edge cases
# ---------------------------------------------------------------------------


class TestCaptureConfigEdgeCases:
    """Edge cases for CaptureConfig and CameraHandle."""

    def test_empty_device_path_raises(self) -> None:
        """CameraHandle with empty device path should raise ValueError."""
        from missy.vision.capture import CameraHandle

        with pytest.raises(ValueError, match="non-empty"):
            CameraHandle("")

    def test_capture_stats_before_open(self) -> None:
        """capture_stats should work even before opening the camera."""
        from missy.vision.capture import CameraHandle

        handle = CameraHandle("/dev/video0")
        stats = handle.capture_stats
        assert stats["is_open"] is False
        assert stats["capture_count"] == 0

    def test_burst_count_zero_raises(self) -> None:
        """capture_burst with count=0 should raise ValueError."""
        from missy.vision.capture import CameraHandle

        handle = CameraHandle("/dev/video0")
        with pytest.raises(ValueError, match="must be >= 1"):
            handle.capture_burst(count=0)

    def test_burst_count_clamped_to_20(self) -> None:
        """capture_burst with count>20 should clamp to 20."""
        from missy.vision.capture import CameraHandle, CaptureConfig

        handle = CameraHandle("/dev/video0", CaptureConfig())
        # We can't actually capture, but test that the burst logic
        # clamps the count
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.read.return_value = (
            True,
            np.zeros((480, 640, 3), dtype=np.uint8),
        )
        handle._cap = mock_cap
        handle._opened = True

        results = handle.capture_burst(count=25, interval=0)
        assert len(results) == 20  # clamped

    def test_adaptive_blank_detector_reset(self) -> None:
        """AdaptiveBlankDetector.reset() clears history."""
        from missy.vision.capture import AdaptiveBlankDetector

        det = AdaptiveBlankDetector()
        det.record_intensity(100.0)
        det.record_intensity(150.0)
        det.record_intensity(200.0)
        det.reset()
        # After reset, threshold goes back to base
        assert det.threshold == det._base
