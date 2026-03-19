"""Tests for missy.vision.resilient_capture — reconnection resilience."""

from __future__ import annotations

from unittest.mock import MagicMock, patch, PropertyMock

import numpy as np
import pytest

from missy.vision.capture import CaptureConfig, CaptureError, CaptureResult
from missy.vision.discovery import CameraDevice
from missy.vision.resilient_capture import ResilientCamera


def _make_camera(path="/dev/video0", name="Test Cam"):
    return CameraDevice(
        device_path=path,
        name=name,
        vendor_id="046d",
        product_id="085c",
        bus_info="usb-1",
    )


class TestResilientCamera:
    def test_properties_default(self):
        cam = ResilientCamera()
        assert not cam.is_connected
        assert cam.current_device is None
        assert cam.total_reconnects == 0

    @patch("missy.vision.resilient_capture.find_preferred_camera")
    def test_connect_no_camera(self, mock_find):
        mock_find.return_value = None
        cam = ResilientCamera()
        with pytest.raises(CaptureError, match="No camera"):
            cam.connect()

    @patch("missy.vision.resilient_capture.CameraHandle")
    @patch("missy.vision.resilient_capture.get_discovery")
    def test_connect_success(self, mock_disc_fn, mock_handle_cls):
        mock_disc = MagicMock()
        mock_disc.find_preferred.return_value = _make_camera()
        mock_disc_fn.return_value = mock_disc

        mock_handle = MagicMock()
        mock_handle.is_open = True
        mock_handle_cls.return_value = mock_handle

        cam = ResilientCamera()
        cam.connect()

        assert cam.is_connected
        assert cam.current_device is not None

    @patch("missy.vision.resilient_capture.CameraHandle")
    @patch("missy.vision.resilient_capture.get_discovery")
    def test_capture_success(self, mock_disc_fn, mock_handle_cls):
        mock_disc = MagicMock()
        mock_disc.find_preferred.return_value = _make_camera()
        mock_disc_fn.return_value = mock_disc

        good_result = CaptureResult(
            success=True,
            image=np.full((100, 100, 3), 128, dtype=np.uint8),
            width=100,
            height=100,
        )
        mock_handle = MagicMock()
        mock_handle.is_open = True
        mock_handle.capture.return_value = good_result
        mock_handle_cls.return_value = mock_handle

        cam = ResilientCamera()
        cam.connect()
        result = cam.capture()

        assert result.success is True
        assert result.width == 100

    @patch("missy.vision.resilient_capture.CameraHandle")
    @patch("missy.vision.resilient_capture.get_discovery")
    def test_capture_reconnect_on_failure(self, mock_disc_fn, mock_handle_cls):
        mock_disc = MagicMock()
        mock_disc.find_preferred.return_value = _make_camera()
        mock_disc.discover.return_value = [_make_camera()]
        mock_disc.find_by_usb_id.return_value = None
        mock_disc_fn.return_value = mock_disc

        # First capture fails, reconnection succeeds
        fail_result = CaptureResult(success=False, error="read failed")
        good_result = CaptureResult(
            success=True,
            image=np.full((100, 100, 3), 128, dtype=np.uint8),
            width=100,
            height=100,
        )

        mock_handle = MagicMock()
        mock_handle.is_open = True
        mock_handle.capture.side_effect = [fail_result, good_result]
        mock_handle_cls.return_value = mock_handle

        cam = ResilientCamera(
            max_reconnect_attempts=2,
            reconnect_delay=0,
        )
        cam.connect()

        # First capture fails, triggers reconnect, second succeeds
        result = cam.capture()
        assert result.success is True
        assert cam.total_reconnects == 1

    @patch("missy.vision.resilient_capture.get_discovery")
    def test_capture_all_reconnects_fail(self, mock_disc_fn):
        mock_disc = MagicMock()
        mock_disc.find_preferred.return_value = None
        mock_disc.find_by_usb_id.return_value = None
        mock_disc.discover.return_value = []
        mock_disc_fn.return_value = mock_disc

        cam = ResilientCamera(max_reconnect_attempts=2, reconnect_delay=0)

        result = cam.capture()
        assert result.success is False

    @patch("missy.vision.resilient_capture.CameraHandle")
    @patch("missy.vision.resilient_capture.get_discovery")
    def test_disconnect(self, mock_disc_fn, mock_handle_cls):
        mock_disc = MagicMock()
        mock_disc.find_preferred.return_value = _make_camera()
        mock_disc_fn.return_value = mock_disc

        mock_handle = MagicMock()
        mock_handle.is_open = True
        mock_handle_cls.return_value = mock_handle

        cam = ResilientCamera()
        cam.connect()
        assert cam.is_connected

        cam.disconnect()
        assert not cam.is_connected

    @patch("missy.vision.resilient_capture.CameraHandle")
    @patch("missy.vision.resilient_capture.get_discovery")
    def test_context_manager(self, mock_disc_fn, mock_handle_cls):
        mock_disc = MagicMock()
        mock_disc.find_preferred.return_value = _make_camera()
        mock_disc_fn.return_value = mock_disc

        mock_handle = MagicMock()
        mock_handle.is_open = True
        mock_handle_cls.return_value = mock_handle

        with ResilientCamera() as cam:
            assert cam.is_connected
        # After context, should be disconnected
        mock_handle.close.assert_called()

    def test_specific_usb_id_preference(self):
        cam = ResilientCamera(
            preferred_vendor_id="046d",
            preferred_product_id="085c",
        )
        assert cam._vendor_id == "046d"
        assert cam._product_id == "085c"
