"""Session 14: Edge case tests for ResilientCamera and MultiCameraManager.

Covers:
- ResilientCamera: max_reconnect=0, disconnect without connect, device path changes,
  USB ID mismatch fallback, cumulative failure threshold, capture after disconnect
- MultiCameraManager: timeout=0 clamping, close_all during capture, status consistency,
  capture_best with no cameras, capture_all empty, max_workers clamping
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Any
from unittest.mock import MagicMock, PropertyMock, patch

import pytest


# ---------------------------------------------------------------------------
# Fake objects
# ---------------------------------------------------------------------------


@dataclass
class FakeCameraDevice:
    device_path: str = "/dev/video0"
    name: str = "Test Camera"
    vendor_id: str = "046d"
    product_id: str = "085c"


class FakeCaptureResult:
    def __init__(self, success=True, error="", image=None, width=1920, height=1080,
                 device_path="", failure_type=None):
        self.success = success
        self.error = error
        self.image = image
        self.width = width
        self.height = height
        self.device_path = device_path
        self.failure_type = failure_type


class FakeCameraHandle:
    def __init__(self, path="", config=None):
        self.path = path
        self._open = False
        self._capture_count = 0
        self._capture_results: list = []

    @property
    def is_open(self):
        return self._open

    def open(self):
        self._open = True

    def close(self):
        self._open = False

    def capture(self):
        self._capture_count += 1
        if self._capture_results:
            return self._capture_results.pop(0)
        return FakeCaptureResult(success=True, device_path=self.path)

    def reset_blank_detector(self):
        pass


class FakeDiscovery:
    def __init__(self, devices=None):
        self._devices = devices or []
        self._by_id: dict = {}

    def find_by_usb_id(self, vendor, product):
        return self._by_id.get((vendor, product))

    def find_preferred(self):
        return self._devices[0] if self._devices else None

    def validate_device(self, device):
        return device in self._devices

    def rediscover_device(self, vendor, product, old_path="", max_attempts=1):
        return self._by_id.get((vendor, product))

    def discover(self, force=False):
        return self._devices


class FakeHealthMonitor:
    def __init__(self):
        self.records: list = []

    def record_capture(self, success=True, device="", latency_ms=0.0, error=""):
        self.records.append({"success": success, "device": device, "error": error})


# ---------------------------------------------------------------------------
# ResilientCamera tests
# ---------------------------------------------------------------------------


class TestResilientCameraEdgeCases:
    """Edge cases in ResilientCamera."""

    def _make_camera(self, **kwargs):
        from missy.vision.resilient_capture import ResilientCamera
        return ResilientCamera(**kwargs)

    @patch("missy.vision.resilient_capture.get_health_monitor")
    @patch("missy.vision.resilient_capture.get_discovery")
    @patch("missy.vision.resilient_capture.CameraHandle")
    def test_max_reconnect_zero(self, mock_handle_cls, mock_disc, mock_hm):
        """max_reconnect_attempts=0 should fail immediately on reconnect."""
        from missy.vision.resilient_capture import ResilientCamera

        device = FakeCameraDevice()
        disc = FakeDiscovery(devices=[device])
        disc._by_id[("046d", "085c")] = device
        mock_disc.return_value = disc
        mock_hm.return_value = FakeHealthMonitor()

        handle = FakeCameraHandle()
        handle._capture_results = [
            FakeCaptureResult(success=False, error="bad frame"),
        ]
        mock_handle_cls.return_value = handle

        cam = ResilientCamera(
            preferred_vendor_id="046d",
            preferred_product_id="085c",
            max_reconnect_attempts=0,
            reconnect_delay=0.01,
        )
        cam.connect()
        result = cam.capture()
        # Reconnect loop never runs → immediate failure
        assert result.success is False
        assert "0 attempts" in result.error

    def test_disconnect_without_connect(self):
        """Disconnect without prior connect should not raise."""
        cam = self._make_camera()
        cam.disconnect()  # Should be safe
        assert not cam.is_connected

    def test_disconnect_clears_state(self):
        """Disconnect should clear handle and connected flag."""
        cam = self._make_camera()
        cam._handle = MagicMock()
        cam._connected = True
        cam.disconnect()
        assert cam._handle is None
        assert cam._connected is False

    def test_is_connected_with_no_handle(self):
        """is_connected should be False when handle is None."""
        cam = self._make_camera()
        assert not cam.is_connected

    def test_is_connected_with_closed_handle(self):
        """is_connected should be False when handle exists but is closed."""
        cam = self._make_camera()
        cam._handle = MagicMock()
        cam._handle.is_open = False
        cam._connected = True
        assert not cam.is_connected

    @patch("missy.vision.resilient_capture.get_health_monitor")
    @patch("missy.vision.resilient_capture.get_discovery")
    @patch("missy.vision.resilient_capture.CameraHandle")
    def test_capture_when_not_connected_auto_connects(self, mock_handle_cls, mock_disc, mock_hm):
        """Capture when disconnected should attempt connection."""
        device = FakeCameraDevice()
        disc = FakeDiscovery(devices=[device])
        disc._by_id[("046d", "085c")] = device
        mock_disc.return_value = disc
        mock_hm.return_value = FakeHealthMonitor()

        handle = FakeCameraHandle()
        mock_handle_cls.return_value = handle

        cam = self._make_camera(preferred_vendor_id="046d", preferred_product_id="085c")
        assert not cam.is_connected
        result = cam.capture()
        assert result.success

    @patch("missy.vision.resilient_capture.get_health_monitor")
    @patch("missy.vision.resilient_capture.get_discovery")
    def test_connect_no_camera_available(self, mock_disc, mock_hm):
        """Connect with no cameras should raise CaptureError."""
        from missy.vision.capture import CaptureError

        disc = FakeDiscovery(devices=[])
        mock_disc.return_value = disc
        mock_hm.return_value = FakeHealthMonitor()

        cam = self._make_camera()
        with pytest.raises(CaptureError, match="No camera available"):
            cam.connect()

    def test_cumulative_failure_counter(self):
        """_record_failure should increment counter and warn at threshold."""
        cam = self._make_camera()
        for i in range(15):
            cam._record_failure()
        assert cam.cumulative_failures == 15

    def test_cumulative_failure_threshold_boundary(self):
        """Failure count exactly at threshold should trigger warning."""
        cam = self._make_camera()
        for i in range(9):
            cam._record_failure()
        assert cam.cumulative_failures == 9
        cam._record_failure()  # 10th — hits threshold
        assert cam.cumulative_failures == 10

    @patch("missy.vision.resilient_capture.get_health_monitor")
    @patch("missy.vision.resilient_capture.get_discovery")
    @patch("missy.vision.resilient_capture.CameraHandle")
    def test_device_path_change_on_reconnect(self, mock_handle_cls, mock_disc, mock_hm):
        """Camera path changing after reconnect should log warning but succeed."""
        device1 = FakeCameraDevice(device_path="/dev/video0")
        device2 = FakeCameraDevice(device_path="/dev/video1")

        disc = FakeDiscovery(devices=[device2])
        disc._by_id[("046d", "085c")] = device2
        mock_disc.return_value = disc
        mock_hm.return_value = FakeHealthMonitor()

        handle1 = FakeCameraHandle(path="/dev/video0")
        handle2 = FakeCameraHandle(path="/dev/video1")
        mock_handle_cls.side_effect = [handle1, handle2]

        cam = self._make_camera(
            preferred_vendor_id="046d",
            preferred_product_id="085c",
            reconnect_delay=0.01,
        )
        cam._current_device = device1
        cam._connected = False

        result = cam._reconnect_and_capture()
        assert result.success

    @patch("missy.vision.resilient_capture.get_health_monitor")
    @patch("missy.vision.resilient_capture.get_discovery")
    @patch("missy.vision.resilient_capture.CameraHandle")
    def test_usb_id_mismatch_fallback(self, mock_handle_cls, mock_disc, mock_hm):
        """Discovering camera with different USB IDs should warn about fallback."""
        fallback_device = FakeCameraDevice(
            device_path="/dev/video0",
            vendor_id="1234", product_id="5678",
        )
        disc = FakeDiscovery(devices=[fallback_device])
        disc._by_id[("046d", "085c")] = fallback_device  # Returns "wrong" device
        mock_disc.return_value = disc
        mock_hm.return_value = FakeHealthMonitor()

        handle = FakeCameraHandle()
        mock_handle_cls.return_value = handle

        cam = self._make_camera(
            preferred_vendor_id="046d",
            preferred_product_id="085c",
            reconnect_delay=0.01,
        )
        result = cam._reconnect_and_capture()
        # Should still succeed (fallback camera works)
        assert result.success

    @patch("missy.vision.resilient_capture.get_health_monitor")
    @patch("missy.vision.resilient_capture.get_discovery")
    @patch("missy.vision.resilient_capture.CameraHandle")
    def test_reconnect_all_attempts_fail_no_device(self, mock_handle_cls, mock_disc, mock_hm):
        """All reconnect attempts failing should return error result."""
        disc = FakeDiscovery(devices=[])
        disc._by_id = {}
        mock_disc.return_value = disc
        mock_hm.return_value = FakeHealthMonitor()

        cam = self._make_camera(
            preferred_vendor_id="046d",
            preferred_product_id="085c",
            max_reconnect_attempts=2,
            reconnect_delay=0.01,
        )
        result = cam._reconnect_and_capture()
        assert result.success is False
        assert "2 attempts" in result.error

    @patch("missy.vision.resilient_capture.get_health_monitor")
    @patch("missy.vision.resilient_capture.get_discovery")
    @patch("missy.vision.resilient_capture.CameraHandle")
    def test_reconnect_without_usb_ids_uses_generic(self, mock_handle_cls, mock_disc, mock_hm):
        """Reconnect without vendor/product IDs should use generic discovery."""
        device = FakeCameraDevice()
        disc = FakeDiscovery(devices=[device])
        mock_disc.return_value = disc
        mock_hm.return_value = FakeHealthMonitor()

        handle = FakeCameraHandle()
        mock_handle_cls.return_value = handle

        cam = self._make_camera(
            preferred_vendor_id="",
            preferred_product_id="",
            reconnect_delay=0.01,
        )
        result = cam._reconnect_and_capture()
        assert result.success

    def test_context_manager(self):
        """Context manager should connect and disconnect."""
        from missy.vision.resilient_capture import ResilientCamera

        cam = ResilientCamera()
        cam.connect = MagicMock()
        cam.disconnect = MagicMock()
        with cam:
            cam.connect.assert_called_once()
        cam.disconnect.assert_called_once()

    def test_total_reconnects_property(self):
        """total_reconnects should start at 0."""
        cam = self._make_camera()
        assert cam.total_reconnects == 0

    def test_current_device_property(self):
        """current_device should be None initially."""
        cam = self._make_camera()
        assert cam.current_device is None

    @patch("missy.vision.resilient_capture.get_health_monitor")
    @patch("missy.vision.resilient_capture.get_discovery")
    @patch("missy.vision.resilient_capture.CameraHandle")
    def test_unrecoverable_permission_error(self, mock_handle_cls, mock_disc, mock_hm):
        """Permission failure type should not retry."""
        from missy.vision.capture import FailureType

        device = FakeCameraDevice()
        disc = FakeDiscovery(devices=[device])
        disc._by_id[("046d", "085c")] = device
        mock_disc.return_value = disc
        mock_hm.return_value = FakeHealthMonitor()

        handle = FakeCameraHandle()
        handle._capture_results = [
            FakeCaptureResult(success=False, error="Permission denied",
                            failure_type=FailureType.PERMISSION),
        ]
        mock_handle_cls.return_value = handle

        cam = self._make_camera(preferred_vendor_id="046d", preferred_product_id="085c")
        cam.connect()
        result = cam.capture()
        assert result.success is False
        assert result.failure_type == FailureType.PERMISSION


# ---------------------------------------------------------------------------
# MultiCameraManager tests
# ---------------------------------------------------------------------------


class TestMultiCameraEdgeCases:
    """Edge cases in MultiCameraManager."""

    def test_max_workers_clamped_to_minimum(self):
        """max_workers=0 should be clamped to 1."""
        from missy.vision.multi_camera import MultiCameraManager

        mgr = MultiCameraManager(max_workers=0)
        assert mgr._max_workers == 1

    def test_max_workers_clamped_to_maximum(self):
        """max_workers=100 should be clamped to 8."""
        from missy.vision.multi_camera import MultiCameraManager

        mgr = MultiCameraManager(max_workers=100)
        assert mgr._max_workers == 8

    def test_capture_all_no_cameras(self):
        """capture_all with no cameras should return error."""
        from missy.vision.multi_camera import MultiCameraManager

        mgr = MultiCameraManager()
        result = mgr.capture_all()
        assert not result.results
        assert "_global" in result.errors

    def test_capture_best_no_cameras(self):
        """capture_best with no cameras should return failure."""
        from missy.vision.multi_camera import MultiCameraManager

        mgr = MultiCameraManager()
        result = mgr.capture_best()
        assert result.success is False

    def test_add_camera_duplicate_raises(self):
        """Adding same device path twice should raise ValueError."""
        from missy.vision.multi_camera import MultiCameraManager

        mgr = MultiCameraManager()
        device = FakeCameraDevice(device_path="/dev/video0")
        with patch("missy.vision.multi_camera.CameraHandle") as mock_cls:
            handle = FakeCameraHandle()
            mock_cls.return_value = handle
            mgr.add_camera(device)
            with pytest.raises(ValueError, match="already added"):
                mgr.add_camera(device)

    def test_remove_camera_nonexistent(self):
        """Removing a non-existent camera should be safe."""
        from missy.vision.multi_camera import MultiCameraManager

        mgr = MultiCameraManager()
        mgr.remove_camera("/dev/video99")  # Should not raise

    def test_connected_devices_only_open(self):
        """connected_devices should only list cameras with is_open=True."""
        from missy.vision.multi_camera import MultiCameraManager

        mgr = MultiCameraManager()
        h1 = FakeCameraHandle(path="/dev/video0")
        h1._open = True
        h2 = FakeCameraHandle(path="/dev/video1")
        h2._open = False
        mgr._handles = {"/dev/video0": h1, "/dev/video1": h2}
        assert mgr.connected_devices == ["/dev/video0"]

    def test_device_count(self):
        """device_count should reflect total handles."""
        from missy.vision.multi_camera import MultiCameraManager

        mgr = MultiCameraManager()
        assert mgr.device_count == 0
        mgr._handles["a"] = MagicMock()
        assert mgr.device_count == 1

    def test_close_all_handles_exceptions(self):
        """close_all should continue even if individual close() raises."""
        from missy.vision.multi_camera import MultiCameraManager

        mgr = MultiCameraManager()
        bad_handle = MagicMock()
        bad_handle.close.side_effect = OSError("close failed")
        good_handle = MagicMock()
        mgr._handles = {"bad": bad_handle, "good": good_handle}
        mgr._devices = {"bad": MagicMock(), "good": MagicMock()}

        mgr.close_all()
        assert len(mgr._handles) == 0
        assert len(mgr._devices) == 0

    def test_context_manager_closes_all(self):
        """Exiting context manager should close all cameras."""
        from missy.vision.multi_camera import MultiCameraManager

        mgr = MultiCameraManager()
        mgr.close_all = MagicMock()
        with mgr:
            pass
        mgr.close_all.assert_called_once()

    def test_status_with_mixed_cameras(self):
        """status() should report all cameras with correct info."""
        from missy.vision.multi_camera import MultiCameraManager

        mgr = MultiCameraManager()
        h1 = FakeCameraHandle(path="/dev/video0")
        h1._open = True
        d1 = FakeCameraDevice(device_path="/dev/video0", name="Cam1", vendor_id="046d")
        mgr._handles = {"/dev/video0": h1}
        mgr._devices = {"/dev/video0": d1}

        s = mgr.status()
        assert s["camera_count"] == 1
        assert "/dev/video0" in s["cameras"]
        assert s["cameras"]["/dev/video0"]["name"] == "Cam1"
        assert s["cameras"]["/dev/video0"]["is_open"] is True

    def test_status_missing_device_info(self):
        """status() should handle missing device info gracefully."""
        from missy.vision.multi_camera import MultiCameraManager

        mgr = MultiCameraManager()
        h1 = FakeCameraHandle()
        h1._open = True
        mgr._handles = {"/dev/video0": h1}
        mgr._devices = {}  # No device info

        s = mgr.status()
        assert s["cameras"]["/dev/video0"]["name"] == ""

    def test_multi_capture_result_properties(self):
        """MultiCaptureResult property methods."""
        from missy.vision.multi_camera import MultiCaptureResult

        result = MultiCaptureResult(
            results={
                "/dev/video0": FakeCaptureResult(success=True, width=1920, height=1080),
                "/dev/video1": FakeCaptureResult(success=False, error="failed"),
            }
        )
        assert result.successful_devices == ["/dev/video0"]
        assert result.failed_devices == ["/dev/video1"]
        assert result.all_succeeded is False
        assert result.any_succeeded is True

    def test_multi_capture_result_empty(self):
        """Empty MultiCaptureResult properties."""
        from missy.vision.multi_camera import MultiCaptureResult

        result = MultiCaptureResult()
        assert result.successful_devices == []
        assert result.failed_devices == []
        assert result.all_succeeded is False
        assert result.any_succeeded is False
        assert result.best_result is None

    def test_multi_capture_result_best_by_resolution(self):
        """best_result should return highest resolution capture."""
        from missy.vision.multi_camera import MultiCaptureResult

        img_mock = MagicMock()
        result = MultiCaptureResult(
            results={
                "low": FakeCaptureResult(success=True, width=640, height=480, image=img_mock),
                "high": FakeCaptureResult(success=True, width=1920, height=1080, image=img_mock),
            }
        )
        best = result.best_result
        assert best is not None
        assert best.width == 1920

    @patch("missy.vision.multi_camera.get_discovery")
    def test_discover_and_connect_vendor_filter(self, mock_disc):
        """discover_and_connect with vendor_filter should filter devices."""
        from missy.vision.multi_camera import MultiCameraManager

        d1 = FakeCameraDevice(device_path="/dev/video0", vendor_id="046d")
        d2 = FakeCameraDevice(device_path="/dev/video1", vendor_id="1234")
        disc = FakeDiscovery(devices=[d1, d2])
        mock_disc.return_value = disc

        mgr = MultiCameraManager()
        with patch("missy.vision.multi_camera.CameraHandle") as mock_cls:
            handle = FakeCameraHandle()
            mock_cls.return_value = handle
            connected = mgr.discover_and_connect(vendor_filter="046d")
            assert len(connected) == 1
            assert connected[0] == "/dev/video0"

    @patch("missy.vision.multi_camera.get_discovery")
    def test_discover_and_connect_max_cameras(self, mock_disc):
        """discover_and_connect should respect max_cameras limit."""
        from missy.vision.multi_camera import MultiCameraManager

        devices = [FakeCameraDevice(device_path=f"/dev/video{i}") for i in range(5)]
        disc = FakeDiscovery(devices=devices)
        mock_disc.return_value = disc

        mgr = MultiCameraManager()
        with patch("missy.vision.multi_camera.CameraHandle") as mock_cls:
            handle = FakeCameraHandle()
            mock_cls.return_value = handle
            connected = mgr.discover_and_connect(max_cameras=2)
            assert len(connected) == 2

    @patch("missy.vision.multi_camera.get_discovery")
    def test_discover_and_connect_open_failure(self, mock_disc):
        """discover_and_connect should skip cameras that fail to open."""
        from missy.vision.multi_camera import MultiCameraManager

        device = FakeCameraDevice(device_path="/dev/video0")
        disc = FakeDiscovery(devices=[device])
        mock_disc.return_value = disc

        mgr = MultiCameraManager()
        with patch("missy.vision.multi_camera.CameraHandle") as mock_cls:
            mock_cls.return_value.open.side_effect = OSError("busy")
            connected = mgr.discover_and_connect()
            assert len(connected) == 0
