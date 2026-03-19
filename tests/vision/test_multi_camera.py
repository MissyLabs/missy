"""Tests for missy.vision.multi_camera — concurrent multi-camera capture."""

from __future__ import annotations

import threading
import time
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from missy.vision.capture import CaptureConfig, CaptureResult
from missy.vision.discovery import CameraDevice
from missy.vision.multi_camera import MultiCameraManager, MultiCaptureResult

# ---------------------------------------------------------------------------
# Helpers / factories
# ---------------------------------------------------------------------------


def _make_device(
    path: str = "/dev/video0",
    name: str = "Test Camera",
    vendor_id: str = "046d",
    product_id: str = "085c",
    bus_info: str = "usb-0000:00:14.0-1",
) -> CameraDevice:
    return CameraDevice(
        device_path=path,
        name=name,
        vendor_id=vendor_id,
        product_id=product_id,
        bus_info=bus_info,
    )


def _make_success_result(
    path: str = "/dev/video0",
    width: int = 1920,
    height: int = 1080,
) -> CaptureResult:
    image = np.full((height, width, 3), 128, dtype=np.uint8)
    return CaptureResult(
        success=True,
        image=image,
        device_path=path,
        width=width,
        height=height,
    )


def _make_failure_result(
    path: str = "/dev/video0",
    error: str = "capture failed",
) -> CaptureResult:
    return CaptureResult(success=False, device_path=path, error=error)


def _make_mock_handle(is_open: bool = True, capture_result: CaptureResult | None = None) -> MagicMock:
    handle = MagicMock()
    handle.is_open = is_open
    handle.capture.return_value = capture_result or _make_success_result()
    return handle


# ---------------------------------------------------------------------------
# MultiCaptureResult — properties
# ---------------------------------------------------------------------------


class TestMultiCaptureResult:
    def test_defaults(self):
        result = MultiCaptureResult()
        assert result.results == {}
        assert result.elapsed_ms == 0.0
        assert result.errors == {}

    def test_successful_devices_all_success(self):
        r0 = _make_success_result("/dev/video0")
        r1 = _make_success_result("/dev/video1")
        mcr = MultiCaptureResult(results={"/dev/video0": r0, "/dev/video1": r1})
        assert set(mcr.successful_devices) == {"/dev/video0", "/dev/video1"}

    def test_successful_devices_mixed(self):
        r0 = _make_success_result("/dev/video0")
        r1 = _make_failure_result("/dev/video1")
        mcr = MultiCaptureResult(results={"/dev/video0": r0, "/dev/video1": r1})
        assert mcr.successful_devices == ["/dev/video0"]

    def test_successful_devices_all_failed(self):
        r0 = _make_failure_result("/dev/video0")
        mcr = MultiCaptureResult(results={"/dev/video0": r0})
        assert mcr.successful_devices == []

    def test_failed_devices_all_success(self):
        r0 = _make_success_result("/dev/video0")
        mcr = MultiCaptureResult(results={"/dev/video0": r0})
        assert mcr.failed_devices == []

    def test_failed_devices_mixed(self):
        r0 = _make_success_result("/dev/video0")
        r1 = _make_failure_result("/dev/video1")
        mcr = MultiCaptureResult(results={"/dev/video0": r0, "/dev/video1": r1})
        assert mcr.failed_devices == ["/dev/video1"]

    def test_failed_devices_all_failed(self):
        r0 = _make_failure_result("/dev/video0")
        r1 = _make_failure_result("/dev/video1")
        mcr = MultiCaptureResult(results={"/dev/video0": r0, "/dev/video1": r1})
        assert set(mcr.failed_devices) == {"/dev/video0", "/dev/video1"}

    def test_all_succeeded_true(self):
        r0 = _make_success_result("/dev/video0")
        r1 = _make_success_result("/dev/video1")
        mcr = MultiCaptureResult(results={"/dev/video0": r0, "/dev/video1": r1})
        assert mcr.all_succeeded is True

    def test_all_succeeded_false_when_one_fails(self):
        r0 = _make_success_result("/dev/video0")
        r1 = _make_failure_result("/dev/video1")
        mcr = MultiCaptureResult(results={"/dev/video0": r0, "/dev/video1": r1})
        assert mcr.all_succeeded is False

    def test_all_succeeded_false_when_empty(self):
        mcr = MultiCaptureResult(results={})
        assert mcr.all_succeeded is False

    def test_any_succeeded_true(self):
        r0 = _make_success_result("/dev/video0")
        r1 = _make_failure_result("/dev/video1")
        mcr = MultiCaptureResult(results={"/dev/video0": r0, "/dev/video1": r1})
        assert mcr.any_succeeded is True

    def test_any_succeeded_false_when_all_fail(self):
        r0 = _make_failure_result("/dev/video0")
        mcr = MultiCaptureResult(results={"/dev/video0": r0})
        assert mcr.any_succeeded is False

    def test_any_succeeded_false_when_empty(self):
        mcr = MultiCaptureResult(results={})
        assert mcr.any_succeeded is False

    def test_best_result_returns_largest_image(self):
        # 640x480 = 307,200 pixels
        r_small = _make_success_result("/dev/video0", width=640, height=480)
        # 1920x1080 = 2,073,600 pixels
        r_large = _make_success_result("/dev/video1", width=1920, height=1080)
        mcr = MultiCaptureResult(
            results={"/dev/video0": r_small, "/dev/video1": r_large}
        )
        best = mcr.best_result
        assert best is r_large

    def test_best_result_ignores_failures(self):
        r_fail = _make_failure_result("/dev/video0")
        r_success = _make_success_result("/dev/video1", width=640, height=480)
        mcr = MultiCaptureResult(
            results={"/dev/video0": r_fail, "/dev/video1": r_success}
        )
        best = mcr.best_result
        assert best is r_success

    def test_best_result_none_when_all_failed(self):
        r0 = _make_failure_result("/dev/video0")
        r1 = _make_failure_result("/dev/video1")
        mcr = MultiCaptureResult(results={"/dev/video0": r0, "/dev/video1": r1})
        assert mcr.best_result is None

    def test_best_result_none_when_empty(self):
        mcr = MultiCaptureResult()
        assert mcr.best_result is None

    def test_best_result_none_when_success_but_no_image(self):
        # success=True but image is None — should be excluded from best_result
        r = CaptureResult(success=True, image=None, device_path="/dev/video0")
        mcr = MultiCaptureResult(results={"/dev/video0": r})
        assert mcr.best_result is None

    def test_elapsed_ms_stored(self):
        mcr = MultiCaptureResult(elapsed_ms=123.45)
        assert mcr.elapsed_ms == 123.45

    def test_errors_stored(self):
        mcr = MultiCaptureResult(errors={"/dev/video0": "timeout"})
        assert mcr.errors["/dev/video0"] == "timeout"


# ---------------------------------------------------------------------------
# MultiCameraManager — init
# ---------------------------------------------------------------------------


class TestMultiCameraManagerInit:
    def test_defaults(self):
        mgr = MultiCameraManager()
        assert mgr.device_count == 0
        assert mgr.connected_devices == []

    def test_custom_config_stored(self):
        cfg = CaptureConfig(width=640, height=480)
        mgr = MultiCameraManager(config=cfg)
        assert mgr._config is cfg

    def test_max_workers_clamped_low(self):
        mgr = MultiCameraManager(max_workers=0)
        assert mgr._max_workers == 1

    def test_max_workers_clamped_high(self):
        mgr = MultiCameraManager(max_workers=100)
        assert mgr._max_workers == 8

    def test_max_workers_in_range(self):
        mgr = MultiCameraManager(max_workers=3)
        assert mgr._max_workers == 3

    def test_open_timeout_stored(self):
        mgr = MultiCameraManager(open_timeout=5.0)
        assert mgr._open_timeout == 5.0

    def test_default_config_created_when_none(self):
        mgr = MultiCameraManager(config=None)
        assert isinstance(mgr._config, CaptureConfig)


# ---------------------------------------------------------------------------
# MultiCameraManager — add_camera / remove_camera
# ---------------------------------------------------------------------------


class TestAddRemoveCamera:
    @patch("missy.vision.multi_camera.CameraHandle")
    def test_add_camera_success(self, MockHandle):
        mock_handle = _make_mock_handle()
        MockHandle.return_value = mock_handle

        mgr = MultiCameraManager()
        device = _make_device("/dev/video0")
        mgr.add_camera(device)

        assert mgr.device_count == 1
        MockHandle.assert_called_once_with("/dev/video0", mgr._config)
        mock_handle.open.assert_called_once()

    @patch("missy.vision.multi_camera.CameraHandle")
    def test_add_camera_uses_custom_config(self, MockHandle):
        mock_handle = _make_mock_handle()
        MockHandle.return_value = mock_handle

        mgr = MultiCameraManager()
        device = _make_device("/dev/video0")
        custom_cfg = CaptureConfig(width=640, height=480)
        mgr.add_camera(device, config=custom_cfg)

        MockHandle.assert_called_once_with("/dev/video0", custom_cfg)

    @patch("missy.vision.multi_camera.CameraHandle")
    def test_add_camera_duplicate_raises(self, MockHandle):
        mock_handle = _make_mock_handle()
        MockHandle.return_value = mock_handle

        mgr = MultiCameraManager()
        device = _make_device("/dev/video0")
        mgr.add_camera(device)

        with pytest.raises(ValueError, match="/dev/video0 already added"):
            mgr.add_camera(device)

    @patch("missy.vision.multi_camera.CameraHandle")
    def test_add_multiple_cameras(self, MockHandle):
        MockHandle.side_effect = [_make_mock_handle(), _make_mock_handle()]

        mgr = MultiCameraManager()
        mgr.add_camera(_make_device("/dev/video0"))
        mgr.add_camera(_make_device("/dev/video1"))

        assert mgr.device_count == 2

    @patch("missy.vision.multi_camera.CameraHandle")
    def test_remove_camera_closes_handle(self, MockHandle):
        mock_handle = _make_mock_handle()
        MockHandle.return_value = mock_handle

        mgr = MultiCameraManager()
        mgr.add_camera(_make_device("/dev/video0"))
        mgr.remove_camera("/dev/video0")

        mock_handle.close.assert_called_once()
        assert mgr.device_count == 0

    def test_remove_nonexistent_camera_is_noop(self):
        mgr = MultiCameraManager()
        # Should not raise
        mgr.remove_camera("/dev/video99")
        assert mgr.device_count == 0

    @patch("missy.vision.multi_camera.CameraHandle")
    def test_connected_devices_returns_open_only(self, MockHandle):
        open_handle = MagicMock()
        open_handle.is_open = True
        closed_handle = MagicMock()
        closed_handle.is_open = False
        MockHandle.side_effect = [open_handle, closed_handle]

        mgr = MultiCameraManager()
        mgr.add_camera(_make_device("/dev/video0"))
        mgr.add_camera(_make_device("/dev/video1"))

        assert mgr.connected_devices == ["/dev/video0"]


# ---------------------------------------------------------------------------
# MultiCameraManager — discover_and_connect
# ---------------------------------------------------------------------------


class TestDiscoverAndConnect:
    @patch("missy.vision.multi_camera.get_discovery")
    @patch("missy.vision.multi_camera.CameraHandle")
    def test_connects_discovered_cameras(self, MockHandle, mock_get_discovery):
        mock_handle = _make_mock_handle()
        MockHandle.return_value = mock_handle

        mock_discovery = MagicMock()
        mock_discovery.discover.return_value = [
            _make_device("/dev/video0"),
            _make_device("/dev/video1"),
        ]
        mock_get_discovery.return_value = mock_discovery

        mgr = MultiCameraManager()
        connected = mgr.discover_and_connect()

        assert "/dev/video0" in connected
        assert "/dev/video1" in connected
        assert mgr.device_count == 2

    @patch("missy.vision.multi_camera.get_discovery")
    @patch("missy.vision.multi_camera.CameraHandle")
    def test_respects_max_cameras(self, MockHandle, mock_get_discovery):
        MockHandle.side_effect = [_make_mock_handle(), _make_mock_handle()]

        mock_discovery = MagicMock()
        mock_discovery.discover.return_value = [
            _make_device("/dev/video0"),
            _make_device("/dev/video1"),
            _make_device("/dev/video2"),
        ]
        mock_get_discovery.return_value = mock_discovery

        mgr = MultiCameraManager()
        connected = mgr.discover_and_connect(max_cameras=2)

        assert len(connected) == 2
        assert mgr.device_count == 2

    @patch("missy.vision.multi_camera.get_discovery")
    @patch("missy.vision.multi_camera.CameraHandle")
    def test_vendor_filter_applied(self, MockHandle, mock_get_discovery):
        mock_handle = _make_mock_handle()
        MockHandle.return_value = mock_handle

        logitech = _make_device("/dev/video0", vendor_id="046d")
        generic = _make_device("/dev/video1", vendor_id="1234")
        mock_discovery = MagicMock()
        mock_discovery.discover.return_value = [logitech, generic]
        mock_get_discovery.return_value = mock_discovery

        mgr = MultiCameraManager()
        connected = mgr.discover_and_connect(vendor_filter="046d")

        assert connected == ["/dev/video0"]
        assert mgr.device_count == 1

    @patch("missy.vision.multi_camera.get_discovery")
    @patch("missy.vision.multi_camera.CameraHandle")
    def test_vendor_filter_no_match_returns_empty(self, MockHandle, mock_get_discovery):
        mock_discovery = MagicMock()
        mock_discovery.discover.return_value = [
            _make_device("/dev/video0", vendor_id="1234"),
        ]
        mock_get_discovery.return_value = mock_discovery

        mgr = MultiCameraManager()
        connected = mgr.discover_and_connect(vendor_filter="046d")

        assert connected == []
        assert mgr.device_count == 0
        MockHandle.assert_not_called()

    @patch("missy.vision.multi_camera.get_discovery")
    @patch("missy.vision.multi_camera.CameraHandle")
    def test_failed_open_skipped_with_warning(self, MockHandle, mock_get_discovery):
        bad_handle = MagicMock()
        bad_handle.open.side_effect = RuntimeError("camera busy")
        good_handle = _make_mock_handle()
        MockHandle.side_effect = [bad_handle, good_handle]

        mock_discovery = MagicMock()
        mock_discovery.discover.return_value = [
            _make_device("/dev/video0"),
            _make_device("/dev/video1"),
        ]
        mock_get_discovery.return_value = mock_discovery

        mgr = MultiCameraManager()
        connected = mgr.discover_and_connect()

        # Only the successfully opened camera is in the list
        assert connected == ["/dev/video1"]

    @patch("missy.vision.multi_camera.get_discovery")
    def test_no_cameras_discovered_returns_empty(self, mock_get_discovery):
        mock_discovery = MagicMock()
        mock_discovery.discover.return_value = []
        mock_get_discovery.return_value = mock_discovery

        mgr = MultiCameraManager()
        connected = mgr.discover_and_connect()

        assert connected == []

    @patch("missy.vision.multi_camera.get_discovery")
    @patch("missy.vision.multi_camera.CameraHandle")
    def test_discover_calls_force_true(self, MockHandle, mock_get_discovery):
        mock_discovery = MagicMock()
        mock_discovery.discover.return_value = []
        mock_get_discovery.return_value = mock_discovery

        mgr = MultiCameraManager()
        mgr.discover_and_connect()

        mock_discovery.discover.assert_called_once_with(force=True)


# ---------------------------------------------------------------------------
# MultiCameraManager — capture_all
# ---------------------------------------------------------------------------


class TestCaptureAll:
    @patch("missy.vision.multi_camera.get_health_monitor")
    @patch("missy.vision.multi_camera.CameraHandle")
    def test_capture_all_no_cameras(self, MockHandle, mock_get_hm):
        mgr = MultiCameraManager()
        result = mgr.capture_all()

        assert result.any_succeeded is False
        assert "_global" in result.errors
        assert "No cameras" in result.errors["_global"]

    @patch("missy.vision.multi_camera.get_health_monitor")
    @patch("missy.vision.multi_camera.CameraHandle")
    def test_capture_all_single_success(self, MockHandle, mock_get_hm):
        capture_result = _make_success_result("/dev/video0")
        mock_handle = _make_mock_handle(capture_result=capture_result)
        MockHandle.return_value = mock_handle
        mock_get_hm.return_value = MagicMock()

        mgr = MultiCameraManager()
        mgr.add_camera(_make_device("/dev/video0"))

        result = mgr.capture_all()

        assert result.all_succeeded is True
        assert result.any_succeeded is True
        assert "/dev/video0" in result.results
        assert result.results["/dev/video0"].success is True

    @patch("missy.vision.multi_camera.get_health_monitor")
    @patch("missy.vision.multi_camera.CameraHandle")
    def test_capture_all_multiple_cameras(self, MockHandle, mock_get_hm):
        r0 = _make_success_result("/dev/video0", width=1920, height=1080)
        r1 = _make_success_result("/dev/video1", width=640, height=480)
        h0 = _make_mock_handle(capture_result=r0)
        h1 = _make_mock_handle(capture_result=r1)
        MockHandle.side_effect = [h0, h1]
        mock_get_hm.return_value = MagicMock()

        mgr = MultiCameraManager()
        mgr.add_camera(_make_device("/dev/video0"))
        mgr.add_camera(_make_device("/dev/video1"))

        result = mgr.capture_all()

        assert result.all_succeeded is True
        assert len(result.results) == 2

    @patch("missy.vision.multi_camera.get_health_monitor")
    @patch("missy.vision.multi_camera.CameraHandle")
    def test_capture_all_failure_recorded_in_errors(self, MockHandle, mock_get_hm):
        fail_result = _make_failure_result("/dev/video0", error="device gone")
        mock_handle = _make_mock_handle(capture_result=fail_result)
        MockHandle.return_value = mock_handle
        mock_get_hm.return_value = MagicMock()

        mgr = MultiCameraManager()
        mgr.add_camera(_make_device("/dev/video0"))

        result = mgr.capture_all()

        assert result.any_succeeded is False
        assert "/dev/video0" in result.errors

    @patch("missy.vision.multi_camera.get_health_monitor")
    @patch("missy.vision.multi_camera.CameraHandle")
    def test_capture_all_exception_caught_per_device(self, MockHandle, mock_get_hm):
        mock_handle = MagicMock()
        mock_handle.is_open = True
        mock_handle.capture.side_effect = RuntimeError("kernel panic")
        MockHandle.return_value = mock_handle
        mock_get_hm.return_value = MagicMock()

        mgr = MultiCameraManager()
        mgr.add_camera(_make_device("/dev/video0"))

        result = mgr.capture_all()

        # The exception is caught and turned into a failed CaptureResult
        assert result.any_succeeded is False
        assert "/dev/video0" in result.results
        assert result.results["/dev/video0"].success is False

    @patch("missy.vision.multi_camera.get_health_monitor")
    @patch("missy.vision.multi_camera.CameraHandle")
    def test_capture_all_elapsed_ms_positive(self, MockHandle, mock_get_hm):
        mock_handle = _make_mock_handle()
        MockHandle.return_value = mock_handle
        mock_get_hm.return_value = MagicMock()

        mgr = MultiCameraManager()
        mgr.add_camera(_make_device("/dev/video0"))

        result = mgr.capture_all()

        assert result.elapsed_ms >= 0.0

    @patch("missy.vision.multi_camera.get_health_monitor")
    @patch("missy.vision.multi_camera.CameraHandle")
    def test_capture_all_records_health_monitor(self, MockHandle, mock_get_hm):
        capture_result = _make_success_result("/dev/video0")
        mock_handle = _make_mock_handle(capture_result=capture_result)
        MockHandle.return_value = mock_handle
        mock_health = MagicMock()
        mock_get_hm.return_value = mock_health

        mgr = MultiCameraManager()
        mgr.add_camera(_make_device("/dev/video0"))
        mgr.capture_all()

        mock_health.record_capture.assert_called_once()
        call_kwargs = mock_health.record_capture.call_args
        assert call_kwargs.kwargs.get("success") is True or (
            call_kwargs.args and call_kwargs.args[0] is True
        )

    @patch("missy.vision.multi_camera.get_health_monitor")
    @patch("missy.vision.multi_camera.CameraHandle")
    def test_capture_all_mixed_success_failure(self, MockHandle, mock_get_hm):
        r_ok = _make_success_result("/dev/video0")
        r_fail = _make_failure_result("/dev/video1", error="no signal")
        h0 = _make_mock_handle(capture_result=r_ok)
        h1 = _make_mock_handle(capture_result=r_fail)
        MockHandle.side_effect = [h0, h1]
        mock_get_hm.return_value = MagicMock()

        mgr = MultiCameraManager()
        mgr.add_camera(_make_device("/dev/video0"))
        mgr.add_camera(_make_device("/dev/video1"))

        result = mgr.capture_all()

        assert result.any_succeeded is True
        assert result.all_succeeded is False
        assert "/dev/video1" in result.errors


# ---------------------------------------------------------------------------
# MultiCameraManager — capture_best
# ---------------------------------------------------------------------------


class TestCaptureBest:
    @patch("missy.vision.multi_camera.get_health_monitor")
    @patch("missy.vision.multi_camera.CameraHandle")
    def test_capture_best_returns_highest_resolution(self, MockHandle, mock_get_hm):
        r_small = _make_success_result("/dev/video0", width=640, height=480)
        r_large = _make_success_result("/dev/video1", width=1920, height=1080)
        h0 = _make_mock_handle(capture_result=r_small)
        h1 = _make_mock_handle(capture_result=r_large)
        MockHandle.side_effect = [h0, h1]
        mock_get_hm.return_value = MagicMock()

        mgr = MultiCameraManager()
        mgr.add_camera(_make_device("/dev/video0"))
        mgr.add_camera(_make_device("/dev/video1"))

        best = mgr.capture_best()

        assert best.success is True
        assert best.width == 1920
        assert best.height == 1080

    @patch("missy.vision.multi_camera.get_health_monitor")
    @patch("missy.vision.multi_camera.CameraHandle")
    def test_capture_best_single_success(self, MockHandle, mock_get_hm):
        r = _make_success_result("/dev/video0")
        mock_handle = _make_mock_handle(capture_result=r)
        MockHandle.return_value = mock_handle
        mock_get_hm.return_value = MagicMock()

        mgr = MultiCameraManager()
        mgr.add_camera(_make_device("/dev/video0"))

        best = mgr.capture_best()

        assert best.success is True

    def test_capture_best_no_cameras_returns_failure(self):
        mgr = MultiCameraManager()
        best = mgr.capture_best()

        assert best.success is False
        assert "No successful captures" in best.error

    @patch("missy.vision.multi_camera.get_health_monitor")
    @patch("missy.vision.multi_camera.CameraHandle")
    def test_capture_best_all_failed_returns_failure(self, MockHandle, mock_get_hm):
        r = _make_failure_result("/dev/video0")
        mock_handle = _make_mock_handle(capture_result=r)
        MockHandle.return_value = mock_handle
        mock_get_hm.return_value = MagicMock()

        mgr = MultiCameraManager()
        mgr.add_camera(_make_device("/dev/video0"))

        best = mgr.capture_best()

        assert best.success is False
        assert "No successful captures" in best.error


# ---------------------------------------------------------------------------
# MultiCameraManager — close_all
# ---------------------------------------------------------------------------


class TestCloseAll:
    @patch("missy.vision.multi_camera.CameraHandle")
    def test_close_all_calls_close_on_each_handle(self, MockHandle):
        h0 = _make_mock_handle()
        h1 = _make_mock_handle()
        MockHandle.side_effect = [h0, h1]

        mgr = MultiCameraManager()
        mgr.add_camera(_make_device("/dev/video0"))
        mgr.add_camera(_make_device("/dev/video1"))
        mgr.close_all()

        h0.close.assert_called_once()
        h1.close.assert_called_once()
        assert mgr.device_count == 0

    def test_close_all_no_cameras_is_noop(self):
        mgr = MultiCameraManager()
        mgr.close_all()  # Should not raise
        assert mgr.device_count == 0

    @patch("missy.vision.multi_camera.CameraHandle")
    def test_close_all_clears_devices_registry(self, MockHandle):
        MockHandle.return_value = _make_mock_handle()
        mgr = MultiCameraManager()
        mgr.add_camera(_make_device("/dev/video0"))
        mgr.close_all()

        assert mgr._handles == {}
        assert mgr._devices == {}

    @patch("missy.vision.multi_camera.CameraHandle")
    def test_close_all_tolerates_close_exception(self, MockHandle):
        bad_handle = MagicMock()
        bad_handle.is_open = True
        bad_handle.close.side_effect = RuntimeError("driver crashed")
        MockHandle.return_value = bad_handle

        mgr = MultiCameraManager()
        mgr.add_camera(_make_device("/dev/video0"))

        # Should not propagate exception
        mgr.close_all()
        assert mgr.device_count == 0


# ---------------------------------------------------------------------------
# MultiCameraManager — status
# ---------------------------------------------------------------------------


class TestStatus:
    def test_status_empty(self):
        mgr = MultiCameraManager()
        s = mgr.status()

        assert s["camera_count"] == 0
        assert s["cameras"] == {}

    @patch("missy.vision.multi_camera.CameraHandle")
    def test_status_with_cameras(self, MockHandle):
        handle = MagicMock()
        handle.is_open = True
        MockHandle.return_value = handle

        mgr = MultiCameraManager()
        device = _make_device(
            path="/dev/video0",
            name="Logitech C922x",
            vendor_id="046d",
            product_id="085c",
        )
        mgr.add_camera(device)

        s = mgr.status()

        assert s["camera_count"] == 1
        cam_info = s["cameras"]["/dev/video0"]
        assert cam_info["name"] == "Logitech C922x"
        assert cam_info["is_open"] is True
        assert cam_info["vendor_id"] == "046d"
        assert cam_info["product_id"] == "085c"

    @patch("missy.vision.multi_camera.CameraHandle")
    def test_status_multiple_cameras(self, MockHandle):
        MockHandle.side_effect = [_make_mock_handle(), _make_mock_handle()]
        mgr = MultiCameraManager()
        mgr.add_camera(_make_device("/dev/video0"))
        mgr.add_camera(_make_device("/dev/video1"))

        s = mgr.status()

        assert s["camera_count"] == 2
        assert "/dev/video0" in s["cameras"]
        assert "/dev/video1" in s["cameras"]

    @patch("missy.vision.multi_camera.CameraHandle")
    def test_status_after_remove(self, MockHandle):
        MockHandle.return_value = _make_mock_handle()
        mgr = MultiCameraManager()
        mgr.add_camera(_make_device("/dev/video0"))
        mgr.remove_camera("/dev/video0")

        s = mgr.status()
        assert s["camera_count"] == 0


# ---------------------------------------------------------------------------
# Context manager
# ---------------------------------------------------------------------------


class TestContextManager:
    @patch("missy.vision.multi_camera.CameraHandle")
    def test_context_manager_calls_close_all_on_exit(self, MockHandle):
        handle = _make_mock_handle()
        MockHandle.return_value = handle

        with MultiCameraManager() as mgr:
            mgr.add_camera(_make_device("/dev/video0"))
            assert mgr.device_count == 1

        handle.close.assert_called_once()
        assert mgr.device_count == 0

    def test_context_manager_returns_self(self):
        mgr = MultiCameraManager()
        with mgr as m:
            assert m is mgr

    @patch("missy.vision.multi_camera.CameraHandle")
    def test_context_manager_closes_on_exception(self, MockHandle):
        handle = _make_mock_handle()
        MockHandle.return_value = handle

        with pytest.raises(ValueError, match="test error"), MultiCameraManager() as mgr:
            mgr.add_camera(_make_device("/dev/video0"))
            raise ValueError("test error")

        handle.close.assert_called_once()

    def test_context_manager_no_cameras_no_error(self):
        with MultiCameraManager():
            pass  # Should not raise


# ---------------------------------------------------------------------------
# Thread-safety
# ---------------------------------------------------------------------------


class TestThreadSafety:
    @patch("missy.vision.multi_camera.CameraHandle")
    def test_concurrent_add_remove_no_crash(self, MockHandle):
        """Multiple threads adding and removing cameras should not corrupt state."""
        MockHandle.side_effect = lambda path, cfg: _make_mock_handle()

        mgr = MultiCameraManager()
        errors: list[Exception] = []

        def add_and_remove(index: int) -> None:
            path = f"/dev/video{index}"
            device = _make_device(path, name=f"Camera {index}")
            try:
                mgr.add_camera(device)
                time.sleep(0.001)
                mgr.remove_camera(path)
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=add_and_remove, args=(i,)) for i in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=5.0)

        assert errors == [], f"Thread errors: {errors}"

    @patch("missy.vision.multi_camera.get_health_monitor")
    @patch("missy.vision.multi_camera.CameraHandle")
    def test_capture_all_concurrent_from_multiple_threads(self, MockHandle, mock_get_hm):
        """Calling capture_all from multiple threads concurrently should be safe."""
        capture_result = _make_success_result("/dev/video0")
        MockHandle.return_value = _make_mock_handle(capture_result=capture_result)
        mock_get_hm.return_value = MagicMock()

        mgr = MultiCameraManager()
        mgr.add_camera(_make_device("/dev/video0"))

        results: list[MultiCaptureResult] = []
        errors: list[Exception] = []

        def run_capture() -> None:
            try:
                r = mgr.capture_all()
                results.append(r)
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=run_capture) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10.0)

        assert errors == [], f"Thread errors: {errors}"
        assert len(results) == 5
        for r in results:
            assert r.any_succeeded is True

    @patch("missy.vision.multi_camera.CameraHandle")
    def test_connected_devices_thread_safe(self, MockHandle):
        """connected_devices property should not raise under concurrent modification."""
        MockHandle.side_effect = lambda path, cfg: _make_mock_handle()

        mgr = MultiCameraManager()
        for i in range(4):
            mgr.add_camera(_make_device(f"/dev/video{i}"))

        snapshots: list[list[str]] = []
        errors: list[Exception] = []

        def read_connected() -> None:
            try:
                for _ in range(20):
                    snapshots.append(mgr.connected_devices)
            except Exception as exc:
                errors.append(exc)

        def modify_cameras() -> None:
            try:
                for i in range(4, 8):
                    mgr.add_camera(_make_device(f"/dev/video{i}"))
                    time.sleep(0.001)
            except Exception as exc:
                errors.append(exc)

        reader = threading.Thread(target=read_connected)
        writer = threading.Thread(target=modify_cameras)
        reader.start()
        writer.start()
        reader.join(timeout=5.0)
        writer.join(timeout=5.0)

        assert errors == [], f"Thread errors: {errors}"

    @patch("missy.vision.multi_camera.get_health_monitor")
    @patch("missy.vision.multi_camera.CameraHandle")
    def test_close_all_while_capturing(self, MockHandle, mock_get_hm):
        """close_all while capture_all is in progress should not corrupt internal state."""

        def slow_capture() -> CaptureResult:
            time.sleep(0.05)
            return _make_success_result("/dev/video0")

        handle = MagicMock()
        handle.is_open = True
        handle.capture.side_effect = slow_capture
        MockHandle.return_value = handle
        mock_get_hm.return_value = MagicMock()

        mgr = MultiCameraManager()
        mgr.add_camera(_make_device("/dev/video0"))

        capture_errors: list[Exception] = []

        def do_capture() -> None:
            try:
                mgr.capture_all()
            except Exception as exc:
                capture_errors.append(exc)

        t = threading.Thread(target=do_capture)
        t.start()
        time.sleep(0.01)
        mgr.close_all()
        t.join(timeout=5.0)

        # Internal state must be consistent regardless of outcome
        assert isinstance(mgr._handles, dict)
        assert isinstance(mgr._devices, dict)


# ---------------------------------------------------------------------------
# Error handling — edge cases
# ---------------------------------------------------------------------------


class TestErrorHandling:
    @patch("missy.vision.multi_camera.get_health_monitor")
    @patch("missy.vision.multi_camera.CameraHandle")
    def test_capture_all_health_monitor_failure_does_not_propagate(
        self, MockHandle, mock_get_hm
    ):
        """If health monitor.record_capture raises, capture_all should still succeed."""
        capture_result = _make_success_result("/dev/video0")
        MockHandle.return_value = _make_mock_handle(capture_result=capture_result)
        mock_health = MagicMock()
        mock_health.record_capture.side_effect = RuntimeError("otel down")
        mock_get_hm.return_value = mock_health

        mgr = MultiCameraManager()
        mgr.add_camera(_make_device("/dev/video0"))

        # RuntimeError from health monitor should propagate out of the worker,
        # which the code catches and stores as a failed result
        result = mgr.capture_all()
        # Either succeeds or captures the error — either way no unhandled exception
        assert isinstance(result, MultiCaptureResult)

    @patch("missy.vision.multi_camera.CameraHandle")
    def test_add_camera_open_failure_raises(self, MockHandle):
        handle = MagicMock()
        handle.open.side_effect = RuntimeError("device busy")
        MockHandle.return_value = handle

        mgr = MultiCameraManager()
        with pytest.raises(RuntimeError, match="device busy"):
            mgr.add_camera(_make_device("/dev/video0"))

        # Camera should not be registered if open fails
        assert mgr.device_count == 0

    @patch("missy.vision.multi_camera.CameraHandle")
    def test_add_camera_open_failure_does_not_leave_partial_state(self, MockHandle):
        handle = MagicMock()
        handle.open.side_effect = OSError("permission denied")
        MockHandle.return_value = handle

        mgr = MultiCameraManager()
        with pytest.raises(OSError):
            mgr.add_camera(_make_device("/dev/video0"))

        assert "/dev/video0" not in mgr._handles
        assert "/dev/video0" not in mgr._devices

    def test_capture_all_returns_error_in_result_when_no_cameras(self):
        mgr = MultiCameraManager()
        result = mgr.capture_all()

        assert isinstance(result, MultiCaptureResult)
        assert result.any_succeeded is False
        assert result.all_succeeded is False
        assert "_global" in result.errors

    @patch("missy.vision.multi_camera.get_health_monitor")
    @patch("missy.vision.multi_camera.CameraHandle")
    def test_capture_all_result_error_uses_capture_result_error_message(
        self, MockHandle, mock_get_hm
    ):
        fail = _make_failure_result("/dev/video0", error="USB disconnect")
        MockHandle.return_value = _make_mock_handle(capture_result=fail)
        mock_get_hm.return_value = MagicMock()

        mgr = MultiCameraManager()
        mgr.add_camera(_make_device("/dev/video0"))
        result = mgr.capture_all()

        assert result.errors.get("/dev/video0") == "USB disconnect"

    @patch("missy.vision.multi_camera.get_discovery")
    @patch("missy.vision.multi_camera.CameraHandle")
    def test_discover_and_connect_all_open_fail_returns_empty(
        self, MockHandle, mock_get_discovery
    ):
        handle = MagicMock()
        handle.open.side_effect = RuntimeError("cannot open any camera")
        MockHandle.return_value = handle

        mock_discovery = MagicMock()
        mock_discovery.discover.return_value = [
            _make_device("/dev/video0"),
            _make_device("/dev/video1"),
        ]
        mock_get_discovery.return_value = mock_discovery

        mgr = MultiCameraManager()
        connected = mgr.discover_and_connect()

        assert connected == []
        assert mgr.device_count == 0

    @patch("missy.vision.multi_camera.CameraHandle")
    def test_remove_then_re_add_same_device_succeeds(self, MockHandle):
        MockHandle.side_effect = [_make_mock_handle(), _make_mock_handle()]
        mgr = MultiCameraManager()
        device = _make_device("/dev/video0")

        mgr.add_camera(device)
        mgr.remove_camera("/dev/video0")
        mgr.add_camera(device)  # Should not raise

        assert mgr.device_count == 1


# ---------------------------------------------------------------------------
# device_count property
# ---------------------------------------------------------------------------


class TestDeviceCount:
    @patch("missy.vision.multi_camera.CameraHandle")
    def test_device_count_increments_on_add(self, MockHandle):
        MockHandle.side_effect = [_make_mock_handle() for _ in range(3)]
        mgr = MultiCameraManager()
        assert mgr.device_count == 0

        mgr.add_camera(_make_device("/dev/video0"))
        assert mgr.device_count == 1

        mgr.add_camera(_make_device("/dev/video1"))
        assert mgr.device_count == 2

        mgr.add_camera(_make_device("/dev/video2"))
        assert mgr.device_count == 3

    @patch("missy.vision.multi_camera.CameraHandle")
    def test_device_count_decrements_on_remove(self, MockHandle):
        MockHandle.side_effect = [_make_mock_handle(), _make_mock_handle()]
        mgr = MultiCameraManager()
        mgr.add_camera(_make_device("/dev/video0"))
        mgr.add_camera(_make_device("/dev/video1"))

        mgr.remove_camera("/dev/video0")
        assert mgr.device_count == 1

    @patch("missy.vision.multi_camera.CameraHandle")
    def test_device_count_zero_after_close_all(self, MockHandle):
        MockHandle.side_effect = [_make_mock_handle(), _make_mock_handle()]
        mgr = MultiCameraManager()
        mgr.add_camera(_make_device("/dev/video0"))
        mgr.add_camera(_make_device("/dev/video1"))
        mgr.close_all()

        assert mgr.device_count == 0
