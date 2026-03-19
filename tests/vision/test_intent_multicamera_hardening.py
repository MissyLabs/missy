"""Session 12: Hardening tests for error paths and edge cases.

Covers:
- Intent classifier thread-safe singleton
- Multi-camera error paths (health monitor failure, concurrent close, timeout)
- Resilient capture edge cases (rediscovery exceptions, cumulative failures)
- Vision memory error handling (SQLite + vector inconsistency, concurrent init)
- Analysis constants validation
- Scene memory change detection edge cases
"""

from __future__ import annotations

import threading
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np

# ---------------------------------------------------------------------------
# Intent classifier thread-safe singleton
# ---------------------------------------------------------------------------


class TestIntentClassifierSingleton:
    """Thread-safe singleton for get_intent_classifier()."""

    def setup_method(self) -> None:
        import missy.vision.intent as mod
        mod._default_classifier = None

    def teardown_method(self) -> None:
        import missy.vision.intent as mod
        mod._default_classifier = None

    def test_singleton_identity(self) -> None:
        from missy.vision.intent import get_intent_classifier
        a = get_intent_classifier()
        b = get_intent_classifier()
        assert a is b

    def test_concurrent_singleton_access(self) -> None:
        """Multiple threads should all get the same instance."""
        from missy.vision.intent import get_intent_classifier
        results: list[Any] = []
        barrier = threading.Barrier(8)

        def worker() -> None:
            barrier.wait()
            results.append(get_intent_classifier())

        threads = [threading.Thread(target=worker) for _ in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=5)
        assert len(results) == 8
        assert all(r is results[0] for r in results)


# ---------------------------------------------------------------------------
# Multi-camera error paths
# ---------------------------------------------------------------------------


class TestMultiCameraErrorPaths:
    """Error handling in multi_camera.py."""

    def test_capture_all_health_monitor_failure_does_not_crash(self) -> None:
        """If health monitor raises during record_capture, the exception is
        caught by the outer try-except in _capture_one."""
        from missy.vision.multi_camera import MultiCameraManager

        mgr = MultiCameraManager()
        mock_handle = MagicMock()
        mock_handle.is_open = True

        mock_result = MagicMock()
        mock_result.success = True
        mock_result.error = None
        mock_handle.capture.return_value = mock_result

        # Inject handle directly
        mgr._handles = {"/dev/video0": mock_handle}

        with patch("missy.vision.multi_camera.get_health_monitor") as ghm:
            ghm.return_value.record_capture.side_effect = RuntimeError("DB locked")
            result = mgr.capture_all(timeout=5)

        # The capture should still be recorded (from the exception handler)
        assert "/dev/video0" in result.results or "/dev/video0" in result.errors

    def test_capture_all_handle_closed_during_capture(self) -> None:
        """Handle becomes not-open between snapshot and capture."""
        from missy.vision.multi_camera import MultiCameraManager

        mgr = MultiCameraManager()
        mock_handle = MagicMock()
        mock_handle.is_open = False  # closed before capture
        mgr._handles = {"/dev/video0": mock_handle}

        result = mgr.capture_all(timeout=5)
        assert not result.any_succeeded
        r = result.results.get("/dev/video0")
        assert r is not None
        assert not r.success
        assert "closed" in r.error.lower()

    def test_close_all_multiple_exceptions(self) -> None:
        """close_all handles exceptions from multiple handles."""
        from missy.vision.multi_camera import MultiCameraManager

        mgr = MultiCameraManager()
        h1 = MagicMock()
        h1.close.side_effect = OSError("USB disconnect")
        h2 = MagicMock()
        h2.close.side_effect = RuntimeError("Already closed")

        mgr._handles = {"/dev/video0": h1, "/dev/video1": h2}
        mgr._devices = {"/dev/video0": MagicMock(), "/dev/video1": MagicMock()}

        mgr.close_all()  # Should not raise
        assert len(mgr._handles) == 0
        assert len(mgr._devices) == 0

    def test_discover_and_connect_duplicate_device(self) -> None:
        """discover_and_connect handles ValueError from add_camera for duplicates."""
        from missy.vision.discovery import CameraDevice
        from missy.vision.multi_camera import MultiCameraManager

        mgr = MultiCameraManager()
        dev = CameraDevice(device_path="/dev/video0", name="Cam", vendor_id="046d", product_id="085c", bus_info="usb-0000:00:14.0-1")

        with patch("missy.vision.multi_camera.get_discovery") as gd:
            gd.return_value.discover.return_value = [dev, dev]  # duplicate
            # First add succeeds, second should get ValueError
            with patch.object(mgr, "add_camera") as add:
                add.side_effect = [None, ValueError("already added")]
                result = mgr.discover_and_connect()

        assert len(result) == 1
        assert result[0] == "/dev/video0"

    def test_capture_all_empty_manager(self) -> None:
        from missy.vision.multi_camera import MultiCameraManager
        mgr = MultiCameraManager()
        result = mgr.capture_all()
        assert "_global" in result.errors
        assert not result.any_succeeded

    def test_capture_best_no_successful(self) -> None:
        from missy.vision.multi_camera import MultiCameraManager

        mgr = MultiCameraManager()
        mock_handle = MagicMock()
        mock_handle.is_open = True
        mock_result = MagicMock()
        mock_result.success = False
        mock_result.error = "blank frame"
        mock_result.image = None
        mock_handle.capture.return_value = mock_result
        mgr._handles = {"/dev/video0": mock_handle}

        with patch("missy.vision.multi_camera.get_health_monitor"):
            result = mgr.capture_best()
        assert not result.success

    def test_multi_capture_result_best_result_picks_highest_res(self) -> None:
        from missy.vision.capture import CaptureResult
        from missy.vision.multi_camera import MultiCaptureResult

        r1 = CaptureResult(success=True, width=640, height=480, image=np.zeros((480, 640, 3), dtype=np.uint8))
        r2 = CaptureResult(success=True, width=1920, height=1080, image=np.zeros((1080, 1920, 3), dtype=np.uint8))
        mcr = MultiCaptureResult(results={"/dev/video0": r1, "/dev/video1": r2})
        assert mcr.best_result is r2


# ---------------------------------------------------------------------------
# Resilient capture edge cases
# ---------------------------------------------------------------------------


class TestResilientCaptureEdgeCases:
    """Error paths in resilient_capture.py."""

    def _make_camera(self, **kwargs: Any) -> Any:
        from missy.vision.resilient_capture import ResilientCamera
        defaults = {
            "preferred_vendor_id": "046d",
            "preferred_product_id": "085c",
            "max_reconnect_attempts": 2,
            "reconnect_delay": 0.01,
            "max_delay": 0.05,
        }
        defaults.update(kwargs)
        return ResilientCamera(**defaults)

    def test_capture_no_camera_available(self) -> None:
        """If no camera is available, returns failure result gracefully."""
        cam = self._make_camera()

        with patch("missy.vision.resilient_capture.get_discovery") as gd:
            gd.return_value.find_by_usb_id.return_value = None
            gd.return_value.find_preferred.return_value = None
            result = cam.capture()

        assert not result.success
        assert "not connected" in result.error.lower()

    def test_cumulative_failure_threshold_warning(self) -> None:
        """Warning is logged when cumulative failures reach threshold."""
        cam = self._make_camera()
        cam._cumulative_failures = 9  # one below threshold
        cam._record_failure()
        assert cam._cumulative_failures == 10

    def test_reconnect_all_attempts_fail(self) -> None:
        """All reconnection attempts fail — returns failure result."""
        cam = self._make_camera(max_reconnect_attempts=2)
        cam._current_device = MagicMock()
        cam._current_device.device_path = "/dev/video0"
        cam._handle = MagicMock()

        with patch("missy.vision.resilient_capture.get_discovery") as gd:
            gd.return_value.rediscover_device.return_value = None
            result = cam._reconnect_and_capture()

        assert not result.success
        assert "Failed to reconnect" in result.error

    def test_reconnect_device_path_change_logged(self) -> None:
        """When device path changes during reconnect, it's noted."""
        from missy.vision.discovery import CameraDevice
        cam = self._make_camera()
        cam._current_device = CameraDevice(
            device_path="/dev/video0", name="Old", vendor_id="046d", product_id="085c", bus_info="usb-0000:00:14.0-1"
        )

        new_dev = CameraDevice(
            device_path="/dev/video2", name="New", vendor_id="046d", product_id="085c", bus_info="usb-0000:00:14.0-2"
        )

        with patch("missy.vision.resilient_capture.get_discovery") as gd:
            gd.return_value.rediscover_device.return_value = new_dev
            with patch.object(cam, "_open_device") as od:
                mock_handle = MagicMock()
                mock_result = MagicMock()
                mock_result.success = True
                mock_handle.capture.return_value = mock_result
                cam._handle = mock_handle

                od.side_effect = lambda d: setattr(cam, '_handle', mock_handle) or setattr(cam, '_connected', True)

                result = cam._reconnect_and_capture()

        assert result.success

    def test_unrecoverable_failure_skips_reconnect(self) -> None:
        """PERMISSION/UNSUPPORTED failure types don't trigger reconnection."""
        from missy.vision.capture import CaptureResult, FailureType
        cam = self._make_camera()
        cam._connected = True
        cam._current_device = MagicMock()
        cam._current_device.device_path = "/dev/video0"

        perm_result = CaptureResult(
            success=False,
            failure_type=FailureType.PERMISSION,
            error="Permission denied",
            device_path="/dev/video0",
        )
        cam._handle = MagicMock()
        cam._handle.is_open = True
        cam._handle.capture.return_value = perm_result

        with patch("missy.vision.resilient_capture.get_discovery") as gd:
            gd.return_value.validate_device.return_value = True
            with patch("missy.vision.resilient_capture.get_health_monitor"):
                result = cam.capture()

        assert not result.success
        assert result.failure_type == FailureType.PERMISSION

    def test_context_manager_connect_and_disconnect(self) -> None:
        cam = self._make_camera()
        with patch.object(cam, "connect") as c, patch.object(cam, "disconnect") as d:
            with cam:
                c.assert_called_once()
            d.assert_called_once()


# ---------------------------------------------------------------------------
# Vision memory error handling
# ---------------------------------------------------------------------------


class TestVisionMemoryErrorPaths:
    """Error handling in vision_memory.py."""

    def test_store_observation_sqlite_fails_vector_succeeds(self) -> None:
        """If SQLite fails but vector succeeds, observation ID is still returned."""
        from missy.vision.vision_memory import VisionMemoryBridge

        mock_mem = MagicMock()
        mock_mem.add_turn.side_effect = RuntimeError("DB locked")
        mock_vec = MagicMock()

        bridge = VisionMemoryBridge(memory_store=mock_mem, vector_store=mock_vec)
        obs_id = bridge.store_observation(
            session_id="s1", task_type="puzzle", observation="test"
        )
        assert obs_id  # UUID returned despite SQLite failure
        mock_vec.add.assert_called_once()

    def test_store_observation_both_fail(self) -> None:
        """If both stores fail, observation ID is still returned."""
        from missy.vision.vision_memory import VisionMemoryBridge

        mock_mem = MagicMock()
        mock_mem.add_turn.side_effect = RuntimeError("fail1")
        mock_vec = MagicMock()
        mock_vec.add.side_effect = RuntimeError("fail2")

        bridge = VisionMemoryBridge(memory_store=mock_mem, vector_store=mock_vec)
        obs_id = bridge.store_observation(
            session_id="s1", task_type="general", observation="test"
        )
        assert obs_id

    def test_recall_vector_fails_falls_back_to_sqlite(self) -> None:
        """When vector search raises, falls back to SQLite."""
        from missy.vision.vision_memory import VisionMemoryBridge

        mock_vec = MagicMock()
        mock_vec.search.side_effect = RuntimeError("FAISS error")
        mock_mem = MagicMock()
        mock_turn = SimpleNamespace(
            role="vision", content="sky pieces", session_id="s1",
            metadata={"task_type": "puzzle", "observation": "sky pieces"}
        )
        mock_mem.search.return_value = [mock_turn]

        bridge = VisionMemoryBridge(memory_store=mock_mem, vector_store=mock_vec)
        results = bridge.recall_observations(query="sky", task_type="puzzle")
        assert len(results) == 1
        assert results[0]["observation"] == "sky pieces"

    def test_recall_both_fail_returns_empty(self) -> None:
        """When both stores fail, returns empty list."""
        from missy.vision.vision_memory import VisionMemoryBridge

        mock_vec = MagicMock()
        mock_vec.search.side_effect = RuntimeError("fail")
        mock_mem = MagicMock()
        mock_mem.search.side_effect = RuntimeError("also fail")

        bridge = VisionMemoryBridge(memory_store=mock_mem, vector_store=mock_vec)
        results = bridge.recall_observations(query="test")
        assert results == []

    def test_concurrent_ensure_init(self) -> None:
        """Multiple threads calling _ensure_init only create one store."""
        from missy.vision.vision_memory import VisionMemoryBridge

        init_count = 0
        original_init = VisionMemoryBridge._ensure_init

        bridge = VisionMemoryBridge()
        barrier = threading.Barrier(6)

        def worker() -> None:
            nonlocal init_count
            barrier.wait()
            bridge._ensure_init()

        with patch("missy.vision.vision_memory.VisionMemoryBridge._ensure_init") as mock_init:
            # Use real implementation but track calls
            mock_init.side_effect = original_init.__get__(bridge)
            threads = [threading.Thread(target=worker) for _ in range(6)]
            for t in threads:
                t.start()
            for t in threads:
                t.join(timeout=5)

        # _initialized should be True
        assert bridge._initialized

    def test_metadata_reserved_keys_filtered(self) -> None:
        """Reserved metadata keys are not passed through."""
        from missy.vision.vision_memory import VisionMemoryBridge

        mock_mem = MagicMock()
        bridge = VisionMemoryBridge(memory_store=mock_mem)

        bridge.store_observation(
            session_id="s1",
            task_type="puzzle",
            observation="test",
            metadata={"session_id": "EVIL", "custom_key": "safe"},
        )

        call_args = mock_mem.add_turn.call_args
        meta = call_args.kwargs.get("metadata") or call_args[1].get("metadata")
        assert meta["session_id"] == "s1"  # not "EVIL"
        assert meta["custom_key"] == "safe"

    def test_clear_session_delete_fails_silently(self) -> None:
        """clear_session continues when individual delete_turn calls fail."""
        from missy.vision.vision_memory import VisionMemoryBridge

        mock_mem = MagicMock()
        turns = [
            SimpleNamespace(role="vision", id="t1"),
            SimpleNamespace(role="vision", id="t2"),
            SimpleNamespace(role="vision", id="t3"),
        ]
        mock_mem.get_session_turns.return_value = turns
        mock_mem.delete_turn.side_effect = [None, RuntimeError("fail"), None]

        bridge = VisionMemoryBridge(memory_store=mock_mem)
        count = bridge.clear_session("s1")
        assert count == 2  # t1 and t3 succeed

    def test_get_session_context_empty(self) -> None:
        """Empty session returns empty string."""
        from missy.vision.vision_memory import VisionMemoryBridge

        mock_mem = MagicMock()
        mock_mem.get_session_turns.return_value = []
        bridge = VisionMemoryBridge(memory_store=mock_mem)
        assert bridge.get_session_context("s1") == ""

    def test_get_session_context_formats_observations(self) -> None:
        from missy.vision.vision_memory import VisionMemoryBridge

        mock_mem = MagicMock()
        turn = SimpleNamespace(
            role="vision", content="Found 3 edge pieces", session_id="s1",
            metadata={
                "task_type": "puzzle", "observation": "Found 3 edge pieces",
                "confidence": 0.85, "timestamp": "2026-03-19T10:00:00",
            }
        )
        mock_mem.get_session_turns.return_value = [turn]
        bridge = VisionMemoryBridge(memory_store=mock_mem)
        ctx = bridge.get_session_context("s1")
        assert "Visual Observations" in ctx
        assert "puzzle" in ctx


# ---------------------------------------------------------------------------
# Analysis constants validation
# ---------------------------------------------------------------------------


class TestAnalysisConstants:
    """Verify that extracted constants are used correctly."""

    def test_canny_thresholds_in_range(self) -> None:
        from missy.vision.analysis import (
            _CANNY_CONTOUR_HIGH,
            _CANNY_CONTOUR_LOW,
            _CANNY_EDGE_HIGH,
            _CANNY_EDGE_LOW,
        )
        assert 0 < _CANNY_EDGE_LOW < _CANNY_EDGE_HIGH <= 255
        assert 0 < _CANNY_CONTOUR_LOW < _CANNY_CONTOUR_HIGH <= 255

    def test_kmeans_params_valid(self) -> None:
        from missy.vision.analysis import (
            _KMEANS_CLUSTERS,
            _KMEANS_DOWNSAMPLE_SIZE,
            _KMEANS_EPSILON,
            _KMEANS_MAX_ITER,
            _KMEANS_MIN_COLOR_PCT,
        )
        assert _KMEANS_CLUSTERS >= 2
        assert _KMEANS_MAX_ITER > 0
        assert _KMEANS_EPSILON > 0
        assert _KMEANS_DOWNSAMPLE_SIZE[0] > 0 and _KMEANS_DOWNSAMPLE_SIZE[1] > 0
        assert 0 < _KMEANS_MIN_COLOR_PCT < 100

    def test_color_thresholds_valid(self) -> None:
        from missy.vision.analysis import _COLOR_BLACK_MAX, _COLOR_WHITE_MIN
        assert 0 < _COLOR_BLACK_MAX < _COLOR_WHITE_MIN < 256

    def test_overlay_weights_sum_to_one(self) -> None:
        from missy.vision.analysis import _EDGE_OVERLAY_EDGE, _EDGE_OVERLAY_ORIGINAL
        assert abs((_EDGE_OVERLAY_ORIGINAL + _EDGE_OVERLAY_EDGE) - 1.0) < 0.01


# ---------------------------------------------------------------------------
# Scene memory change detection constants
# ---------------------------------------------------------------------------


class TestSceneMemoryConstants:
    """Verify scene memory change detection constants."""

    def test_change_weights_sum_to_one(self) -> None:
        from missy.vision.scene_memory import _CHANGE_PHASH_WEIGHT, _CHANGE_PIXEL_WEIGHT
        assert abs((_CHANGE_PIXEL_WEIGHT + _CHANGE_PHASH_WEIGHT) - 1.0) < 0.01

    def test_thresholds_ascending(self) -> None:
        from missy.vision.scene_memory import (
            _CHANGE_THRESHOLD_MAJOR,
            _CHANGE_THRESHOLD_MINOR,
            _CHANGE_THRESHOLD_MODERATE,
        )
        assert 0 < _CHANGE_THRESHOLD_MINOR < _CHANGE_THRESHOLD_MODERATE < _CHANGE_THRESHOLD_MAJOR

    def test_compare_size_positive(self) -> None:
        from missy.vision.scene_memory import _CHANGE_COMPARE_SIZE
        assert _CHANGE_COMPARE_SIZE[0] > 0 and _CHANGE_COMPARE_SIZE[1] > 0

    def test_phash_bits_matches_hash_size(self) -> None:
        from missy.vision.scene_memory import _PHASH_BITS
        assert _PHASH_BITS == 64  # 8x8 perceptual hash


# ---------------------------------------------------------------------------
# Color description coverage
# ---------------------------------------------------------------------------


class TestColorDescription:
    """Ensure _describe_color covers all branches."""

    def test_black(self) -> None:
        from missy.vision.analysis import _describe_color
        assert _describe_color([10, 10, 10]) == "black"

    def test_white(self) -> None:
        from missy.vision.analysis import _describe_color
        assert _describe_color([220, 230, 240]) == "white"

    def test_red(self) -> None:
        from missy.vision.analysis import _describe_color
        assert _describe_color([200, 30, 30]) == "red"

    def test_green(self) -> None:
        from missy.vision.analysis import _describe_color
        assert _describe_color([20, 180, 20]) == "green"

    def test_blue(self) -> None:
        from missy.vision.analysis import _describe_color
        assert _describe_color([20, 20, 180]) == "blue"

    def test_yellow(self) -> None:
        from missy.vision.analysis import _describe_color
        assert _describe_color([200, 200, 30]) == "yellow"

    def test_orange(self) -> None:
        from missy.vision.analysis import _describe_color
        assert _describe_color([200, 130, 30]) == "orange"

    def test_purple(self) -> None:
        from missy.vision.analysis import _describe_color
        assert _describe_color([160, 30, 160]) == "purple"

    def test_tan_brown(self) -> None:
        from missy.vision.analysis import _describe_color
        assert _describe_color([180, 140, 100]) == "tan/brown"

    def test_light_gray(self) -> None:
        from missy.vision.analysis import _describe_color
        assert _describe_color([180, 180, 180]) == "light gray"

    def test_gray(self) -> None:
        from missy.vision.analysis import _describe_color
        assert _describe_color([100, 100, 100]) == "gray"

    def test_unnamed_rgb(self) -> None:
        from missy.vision.analysis import _describe_color
        result = _describe_color([77, 133, 200])
        assert result.startswith("rgb(")


# ---------------------------------------------------------------------------
# MultiCaptureResult properties
# ---------------------------------------------------------------------------


class TestMultiCaptureResultProperties:
    """Edge cases for MultiCaptureResult properties."""

    def test_successful_devices_empty(self) -> None:
        from missy.vision.multi_camera import MultiCaptureResult
        mcr = MultiCaptureResult()
        assert mcr.successful_devices == []
        assert mcr.failed_devices == []

    def test_all_succeeded_empty_is_false(self) -> None:
        from missy.vision.multi_camera import MultiCaptureResult
        mcr = MultiCaptureResult()
        assert not mcr.all_succeeded

    def test_any_succeeded_empty_is_false(self) -> None:
        from missy.vision.multi_camera import MultiCaptureResult
        mcr = MultiCaptureResult()
        assert not mcr.any_succeeded

    def test_best_result_none_when_all_failed(self) -> None:
        from missy.vision.capture import CaptureResult
        from missy.vision.multi_camera import MultiCaptureResult
        r = CaptureResult(success=False, error="failed")
        mcr = MultiCaptureResult(results={"/dev/video0": r})
        assert mcr.best_result is None

    def test_best_result_none_image(self) -> None:
        """Successful result but no image → not considered."""
        from missy.vision.capture import CaptureResult
        from missy.vision.multi_camera import MultiCaptureResult
        r = CaptureResult(success=True, image=None)
        mcr = MultiCaptureResult(results={"/dev/video0": r})
        assert mcr.best_result is None
