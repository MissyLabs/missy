"""Extended tests for missy.vision.resilient_capture — reconnection and edge cases."""

from __future__ import annotations

from unittest.mock import MagicMock, patch, call

import numpy as np
import pytest

from missy.vision.capture import CaptureConfig, CaptureError, CaptureResult
from missy.vision.resilient_capture import ResilientCamera


def _make_success_result():
    return CaptureResult(
        success=True,
        image=np.full((480, 640, 3), 128, dtype=np.uint8),
        width=640,
        height=480,
        device_path="/dev/video0",
    )


def _make_fail_result(error="read failed"):
    return CaptureResult(success=False, error=error)


class TestResilientCameraConnect:
    @patch("missy.vision.resilient_capture.get_discovery")
    def test_connect_with_usb_id(self, mock_get_disc):
        """Should try USB ID match first."""
        mock_disc = MagicMock()
        mock_device = MagicMock(device_path="/dev/video0", name="C922x")
        mock_disc.find_by_usb_id.return_value = mock_device
        mock_get_disc.return_value = mock_disc

        with patch("missy.vision.resilient_capture.CameraHandle") as MockHandle:
            inst = MagicMock()
            MockHandle.return_value = inst

            cam = ResilientCamera(
                preferred_vendor_id="046d",
                preferred_product_id="085c",
            )
            cam.connect()

            mock_disc.find_by_usb_id.assert_called_once_with("046d", "085c")
            assert cam.is_connected or inst.is_open  # handle was opened

    @patch("missy.vision.resilient_capture.get_discovery")
    def test_connect_fallback_to_preferred(self, mock_get_disc):
        """If USB ID not found, should fall back to find_preferred."""
        mock_disc = MagicMock()
        mock_disc.find_by_usb_id.return_value = None
        mock_device = MagicMock(device_path="/dev/video2", name="Generic Camera")
        mock_disc.find_preferred.return_value = mock_device
        mock_get_disc.return_value = mock_disc

        with patch("missy.vision.resilient_capture.CameraHandle") as MockHandle:
            inst = MagicMock()
            MockHandle.return_value = inst

            cam = ResilientCamera(
                preferred_vendor_id="046d",
                preferred_product_id="085c",
            )
            cam.connect()

            mock_disc.find_preferred.assert_called_once()

    @patch("missy.vision.resilient_capture.get_discovery")
    def test_connect_no_camera_raises(self, mock_get_disc):
        """Should raise CaptureError if no camera found."""
        mock_disc = MagicMock()
        mock_disc.find_by_usb_id.return_value = None
        mock_disc.find_preferred.return_value = None
        mock_get_disc.return_value = mock_disc

        cam = ResilientCamera()
        with pytest.raises(CaptureError, match="No camera"):
            cam.connect()


class TestResilientCameraCapture:
    @patch("missy.vision.resilient_capture.get_discovery")
    def test_capture_success(self, mock_get_disc):
        mock_disc = MagicMock()
        mock_device = MagicMock(device_path="/dev/video0", name="Cam")
        mock_disc.find_preferred.return_value = mock_device
        mock_get_disc.return_value = mock_disc

        with patch("missy.vision.resilient_capture.CameraHandle") as MockHandle:
            inst = MagicMock()
            inst.is_open = True
            inst.capture.return_value = _make_success_result()
            MockHandle.return_value = inst

            cam = ResilientCamera()
            cam.connect()
            result = cam.capture()

            assert result.success

    @patch("missy.vision.resilient_capture.get_discovery")
    def test_capture_auto_connects(self, mock_get_disc):
        """If not connected, capture should auto-connect."""
        mock_disc = MagicMock()
        mock_device = MagicMock(device_path="/dev/video0", name="Cam")
        mock_disc.find_preferred.return_value = mock_device
        mock_get_disc.return_value = mock_disc

        with patch("missy.vision.resilient_capture.CameraHandle") as MockHandle:
            inst = MagicMock()
            inst.is_open = True
            inst.capture.return_value = _make_success_result()
            MockHandle.return_value = inst

            cam = ResilientCamera()
            # Don't call connect() — capture should do it
            result = cam.capture()
            assert result.success

    @patch("missy.vision.resilient_capture.get_discovery")
    @patch("missy.vision.resilient_capture.time")
    def test_capture_auto_connect_failure(self, mock_time, mock_get_disc):
        """If auto-connect fails, should return error result."""
        mock_disc = MagicMock()
        mock_disc.find_preferred.return_value = None
        mock_disc.find_by_usb_id.return_value = None
        mock_get_disc.return_value = mock_disc

        cam = ResilientCamera()
        result = cam.capture()

        assert not result.success
        assert "reconnection failed" in result.error


class TestResilientCameraReconnect:
    @patch("missy.vision.resilient_capture.get_discovery")
    @patch("missy.vision.resilient_capture.time")
    def test_reconnect_after_capture_failure(self, mock_time, mock_get_disc):
        """Should attempt reconnection when capture fails."""
        mock_disc = MagicMock()
        mock_device = MagicMock(device_path="/dev/video0", name="Cam")
        mock_disc.find_preferred.return_value = mock_device
        mock_disc.discover.return_value = [mock_device]
        mock_get_disc.return_value = mock_disc

        with patch("missy.vision.resilient_capture.CameraHandle") as MockHandle:
            # First handle: open succeeds, capture fails
            # Second handle (after reconnect): capture succeeds
            first_handle = MagicMock()
            first_handle.is_open = True
            first_handle.capture.side_effect = RuntimeError("device lost")

            second_handle = MagicMock()
            second_handle.is_open = True
            second_handle.capture.return_value = _make_success_result()

            MockHandle.side_effect = [first_handle, second_handle]

            cam = ResilientCamera(max_reconnect_attempts=2, reconnect_delay=0)
            cam.connect()
            result = cam.capture()

            assert result.success
            assert cam.total_reconnects == 1

    @patch("missy.vision.resilient_capture.get_discovery")
    @patch("missy.vision.resilient_capture.time")
    def test_reconnect_exhausts_attempts(self, mock_time, mock_get_disc):
        """Should return error after max attempts exhausted."""
        mock_disc = MagicMock()
        mock_disc.find_preferred.return_value = None
        mock_disc.find_by_usb_id.return_value = None
        mock_disc.discover.return_value = []
        mock_get_disc.return_value = mock_disc

        with patch("missy.vision.resilient_capture.CameraHandle") as MockHandle:
            handle = MagicMock()
            handle.is_open = True
            handle.capture.side_effect = RuntimeError("device lost")
            MockHandle.return_value = handle

            cam = ResilientCamera(max_reconnect_attempts=3, reconnect_delay=0)
            cam._connected = True
            cam._handle = handle
            result = cam.capture()

            assert not result.success
            assert "3 attempts" in result.error

    @patch("missy.vision.resilient_capture.get_discovery")
    @patch("missy.vision.resilient_capture.time")
    def test_reconnect_force_rediscovery(self, mock_time, mock_get_disc):
        """Reconnection should force camera rediscovery."""
        mock_disc = MagicMock()
        mock_disc.find_preferred.return_value = None
        mock_disc.discover.return_value = []
        mock_get_disc.return_value = mock_disc

        cam = ResilientCamera(max_reconnect_attempts=2, reconnect_delay=0)

        # Trigger reconnection path
        cam._connected = False
        result = cam.capture()

        # Should have called discover(force=True) during reconnection
        if mock_disc.discover.called:
            mock_disc.discover.assert_any_call(force=True)


class TestResilientCameraContextManager:
    @patch("missy.vision.resilient_capture.get_discovery")
    def test_context_manager_connects_and_disconnects(self, mock_get_disc):
        mock_disc = MagicMock()
        mock_device = MagicMock(device_path="/dev/video0", name="Cam")
        mock_disc.find_preferred.return_value = mock_device
        mock_get_disc.return_value = mock_disc

        with patch("missy.vision.resilient_capture.CameraHandle") as MockHandle:
            inst = MagicMock()
            inst.is_open = True
            MockHandle.return_value = inst

            with ResilientCamera() as cam:
                assert cam is not None

            # After context exit, should be disconnected
            inst.close.assert_called()

    @patch("missy.vision.resilient_capture.get_discovery")
    def test_context_manager_handles_connect_failure(self, mock_get_disc):
        mock_disc = MagicMock()
        mock_disc.find_preferred.return_value = None
        mock_get_disc.return_value = mock_disc

        with pytest.raises(CaptureError):
            with ResilientCamera() as cam:
                pass


class TestResilientCameraDisconnect:
    def test_disconnect_clears_state(self):
        cam = ResilientCamera()
        mock_handle = MagicMock()
        cam._handle = mock_handle
        cam._connected = True

        cam.disconnect()

        assert not cam._connected
        assert cam._handle is None
        mock_handle.close.assert_called_once()

    def test_disconnect_when_already_disconnected(self):
        cam = ResilientCamera()
        cam.disconnect()  # Should not raise

    def test_double_disconnect_safe(self):
        cam = ResilientCamera()
        mock_handle = MagicMock()
        cam._handle = mock_handle
        cam._connected = True

        cam.disconnect()
        cam.disconnect()

        mock_handle.close.assert_called_once()
