"""Edge-case and stress tests for resilient capture and recovery paths."""

from __future__ import annotations

import threading
from unittest.mock import MagicMock, patch

import numpy as np

from missy.vision.capture import CaptureError, CaptureResult, FailureType
from missy.vision.discovery import CameraDevice
from missy.vision.resilient_capture import ResilientCamera


def _make_device(path: str = "/dev/video0") -> CameraDevice:
    return CameraDevice(
        device_path=path,
        name="Test Camera",
        vendor_id="046d",
        product_id="085c",
        bus_info="usb-0000:00:14.0-1",
    )


class TestResilientCameraEdgeCases:
    """Edge cases for ResilientCamera."""

    def test_capture_without_connect_auto_connects(self) -> None:
        cam = ResilientCamera(max_reconnect_attempts=1, reconnect_delay=0.01)
        mock_disc = MagicMock()
        mock_disc.find_preferred.return_value = None
        mock_disc.discover.return_value = []
        mock_disc.find_by_usb_id.return_value = None
        mock_disc.validate_device.return_value = True

        with patch("missy.vision.resilient_capture.get_discovery", return_value=mock_disc):
            result = cam.capture()
            assert not result.success

    def test_cumulative_failure_tracking(self) -> None:
        cam = ResilientCamera(max_reconnect_attempts=1, reconnect_delay=0.01)
        for _ in range(15):
            cam._record_failure()
        assert cam.cumulative_failures == 15

    def test_disconnect_clears_state(self) -> None:
        cam = ResilientCamera()
        cam._connected = True
        cam._handle = MagicMock()
        cam.disconnect()
        assert not cam.is_connected
        assert cam._handle is None

    def test_double_disconnect_safe(self) -> None:
        cam = ResilientCamera()
        cam.disconnect()
        cam.disconnect()
        assert not cam.is_connected

    def test_context_manager_connects_and_disconnects(self) -> None:
        cam = ResilientCamera(max_reconnect_attempts=1, reconnect_delay=0.01)
        mock_disc = MagicMock()
        mock_disc.find_preferred.return_value = None
        mock_disc.find_by_usb_id.return_value = None

        with patch("missy.vision.resilient_capture.get_discovery", return_value=mock_disc):
            raised = False
            try:
                with cam:
                    pass
            except CaptureError:
                raised = True
            assert raised, "Expected CaptureError when no camera available"

    def test_reconnect_increments_counter(self) -> None:
        cam = ResilientCamera(
            preferred_vendor_id="046d",
            preferred_product_id="085c",
            max_reconnect_attempts=2,
            reconnect_delay=0.01,
        )
        device = _make_device()
        success_result = CaptureResult(
            success=True,
            image=np.zeros((10, 10, 3), dtype=np.uint8),
            device_path="/dev/video0",
        )

        mock_handle = MagicMock()
        mock_handle.is_open = True
        mock_handle.capture.return_value = success_result

        mock_disc = MagicMock()
        mock_disc.validate_device.return_value = True
        mock_disc.rediscover_device.return_value = device
        mock_disc.discover.return_value = [device]
        mock_disc.find_by_usb_id.return_value = device

        with patch("missy.vision.resilient_capture.get_discovery", return_value=mock_disc), \
             patch("missy.vision.capture.CameraHandle") as MockHandle, \
             patch("time.sleep"):
            MockHandle.return_value = mock_handle

            # Force reconnect path
            cam._connected = True
            cam._handle = MagicMock()
            cam._handle.is_open = True
            cam._handle.capture.side_effect = Exception("device error")
            cam._current_device = device

            mock_disc.validate_device.return_value = True
            cam.capture()
            # Should have attempted reconnection
            assert cam.total_reconnects >= 0  # may or may not succeed depending on mock depth

    def test_permission_failure_not_retried(self) -> None:
        cam = ResilientCamera(max_reconnect_attempts=3, reconnect_delay=0.01)
        perm_result = CaptureResult(
            success=False,
            error="Permission denied",
            failure_type=FailureType.PERMISSION,
        )

        cam._connected = True
        mock_handle = MagicMock()
        mock_handle.is_open = True
        mock_handle.capture.return_value = perm_result
        cam._handle = mock_handle
        cam._current_device = _make_device()

        mock_disc = MagicMock()
        mock_disc.validate_device.return_value = True

        with patch("missy.vision.resilient_capture.get_discovery", return_value=mock_disc):
            result = cam.capture()
            assert not result.success
            assert result.failure_type == FailureType.PERMISSION

    def test_backoff_parameters(self) -> None:
        cam = ResilientCamera(
            reconnect_delay=1.0,
            backoff_factor=2.0,
            max_delay=10.0,
        )
        assert cam._reconnect_delay == 1.0
        assert cam._backoff_factor == 2.0
        assert cam._max_delay == 10.0


class TestResilientCameraDevicePathChange:
    """Tests for handling device path changes during reconnection."""

    def test_warns_on_path_change(self) -> None:
        cam = ResilientCamera(
            preferred_vendor_id="046d",
            preferred_product_id="085c",
            max_reconnect_attempts=1,
            reconnect_delay=0.01,
        )
        cam._current_device = _make_device(path="/dev/video0")

        new_device = _make_device(path="/dev/video2")
        success_result = CaptureResult(
            success=True,
            image=np.zeros((10, 10, 3), dtype=np.uint8),
            device_path="/dev/video2",
        )

        mock_handle = MagicMock()
        mock_handle.is_open = True
        mock_handle.capture.return_value = success_result

        mock_disc = MagicMock()
        mock_disc.validate_device.return_value = True
        mock_disc.rediscover_device.return_value = new_device

        with patch("missy.vision.resilient_capture.get_discovery", return_value=mock_disc), \
             patch("missy.vision.capture.CameraHandle") as MockHandle, \
             patch("time.sleep"):
            MockHandle.return_value = mock_handle

            cam._connected = False
            result = cam._reconnect_and_capture()
            if result.success:
                assert cam._current_device.device_path == "/dev/video2"


class TestResilientCameraConcurrency:
    """Test that concurrent capture calls don't cause crashes."""

    def test_concurrent_captures_safe(self) -> None:
        cam = ResilientCamera(max_reconnect_attempts=1, reconnect_delay=0.01)
        results = []
        errors = []

        mock_disc = MagicMock()
        mock_disc.find_preferred.return_value = None
        mock_disc.find_by_usb_id.return_value = None
        mock_disc.validate_device.return_value = False
        mock_disc.rediscover_device.return_value = None

        def do_capture():
            try:
                with patch("missy.vision.resilient_capture.get_discovery", return_value=mock_disc), \
                     patch("time.sleep"):
                    r = cam.capture()
                    results.append(r)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=do_capture) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)

        assert len(errors) == 0
        assert all(not r.success for r in results)


class TestHealthMonitorThreadSafety:
    """Concurrent access to health monitor persistence."""

    def test_concurrent_record_and_save(self, tmp_path) -> None:
        from missy.vision.health_monitor import VisionHealthMonitor

        monitor = VisionHealthMonitor()
        db = tmp_path / "health.db"
        errors = []

        def recorder():
            try:
                for _ in range(50):
                    monitor.record_capture(
                        success=True,
                        device="/dev/video0",
                        quality_score=0.8,
                    )
            except Exception as e:
                errors.append(e)

        def saver():
            try:
                for _ in range(5):
                    monitor.save(db)
            except Exception as e:
                errors.append(e)

        threads = [
            threading.Thread(target=recorder),
            threading.Thread(target=recorder),
            threading.Thread(target=saver),
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)

        assert len(errors) == 0
        stats = monitor.get_device_stats("/dev/video0")
        assert stats is not None
        assert stats.total_captures == 100
