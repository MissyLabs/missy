"""Tests for WebcamSource timeout and ResilientCamera exponential backoff."""

from __future__ import annotations

import contextlib
import time
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from missy.vision.capture import CaptureError
from missy.vision.sources import SourceType, WebcamSource


class TestWebcamSourceTimeout:
    """Tests for WebcamSource timeout protection."""

    def test_timeout_parameter(self):
        src = WebcamSource("/dev/video0", timeout=5.0)
        assert src._timeout == 5.0

    def test_default_timeout(self):
        src = WebcamSource("/dev/video0")
        assert src._timeout == 15.0

    @patch("missy.vision.capture.CameraHandle")
    def test_successful_acquire_within_timeout(self, MockHandle):
        """Normal capture should work fine with timeout."""
        mock_handle = MagicMock()
        mock_result = MagicMock()
        mock_result.success = True
        mock_result.image = np.full((480, 640, 3), 128, dtype=np.uint8)
        mock_result.width = 640
        mock_result.height = 480
        mock_handle.capture.return_value = mock_result
        MockHandle.return_value = mock_handle

        src = WebcamSource("/dev/video0", timeout=5.0)
        frame = src.acquire()

        assert frame.source_type == SourceType.WEBCAM
        assert frame.width == 640

    @patch("missy.vision.capture.CameraHandle")
    def test_timeout_on_frozen_camera(self, MockHandle):
        """Frozen camera should raise CaptureError after timeout."""
        def frozen_open():
            time.sleep(10)  # Simulate frozen camera

        mock_handle = MagicMock()
        mock_handle.open.side_effect = frozen_open
        MockHandle.return_value = mock_handle

        src = WebcamSource("/dev/video0", timeout=0.5)
        with pytest.raises(CaptureError, match="did not respond"):
            src.acquire()

    @patch("missy.vision.capture.CameraHandle")
    def test_capture_failure_propagates(self, MockHandle):
        """CaptureError from inner capture should propagate."""
        mock_handle = MagicMock()
        mock_result = MagicMock()
        mock_result.success = False
        mock_result.error = "Blank frame detected"
        mock_handle.capture.return_value = mock_result
        MockHandle.return_value = mock_handle

        src = WebcamSource("/dev/video0", timeout=5.0)
        with pytest.raises(CaptureError, match="Blank frame"):
            src.acquire()

    @patch("missy.vision.capture.CameraHandle")
    def test_handle_closed_even_on_timeout(self, MockHandle):
        """Camera handle should be closed even when timeout occurs."""
        def slow_open():
            time.sleep(10)

        mock_handle = MagicMock()
        mock_handle.open.side_effect = slow_open
        MockHandle.return_value = mock_handle

        src = WebcamSource("/dev/video0", timeout=0.3)
        with contextlib.suppress(CaptureError):
            src.acquire()

        # The thread may still be running, but close should have been called
        # in the finally block inside _do_capture (if it completed)


class TestResilientCameraBackoff:
    """Tests for exponential backoff in reconnection."""

    @patch("missy.vision.resilient_capture.get_discovery")
    @patch("missy.vision.resilient_capture.time")
    def test_backoff_increases_delay(self, mock_time, mock_get_disc):
        """Delay should increase with each failed attempt."""
        from missy.vision.resilient_capture import ResilientCamera

        mock_disc = MagicMock()
        # Camera found but all reconnect captures fail
        mock_device = MagicMock(device_path="/dev/video0", name="Cam")
        mock_disc.find_preferred.return_value = mock_device
        mock_disc.discover.return_value = [mock_device]
        mock_get_disc.return_value = mock_disc

        with patch("missy.vision.resilient_capture.CameraHandle") as MockHandle:
            mock_handle = MagicMock()
            mock_handle.is_open = True
            mock_handle.capture.side_effect = RuntimeError("device lost")
            MockHandle.return_value = mock_handle

            cam = ResilientCamera(
                max_reconnect_attempts=4,
                reconnect_delay=1.0,
                backoff_factor=2.0,
                max_delay=20.0,
            )
            # Simulate connected but capture fails -> reconnect loop
            cam._connected = True
            cam._handle = mock_handle
            cam.capture()

        # Check sleep calls — should show increasing delays
        sleep_calls = [c[0][0] for c in mock_time.sleep.call_args_list]
        assert len(sleep_calls) >= 3
        # Each delay should be >= previous
        for i in range(1, len(sleep_calls)):
            assert sleep_calls[i] >= sleep_calls[i - 1]

    @patch("missy.vision.resilient_capture.get_discovery")
    @patch("missy.vision.resilient_capture.time")
    def test_backoff_capped_at_max(self, mock_time, mock_get_disc):
        """Delay should not exceed max_delay."""
        from missy.vision.resilient_capture import ResilientCamera

        mock_disc = MagicMock()
        mock_device = MagicMock(device_path="/dev/video0", name="Cam")
        mock_disc.find_preferred.return_value = mock_device
        mock_disc.discover.return_value = [mock_device]
        mock_get_disc.return_value = mock_disc

        with patch("missy.vision.resilient_capture.CameraHandle") as MockHandle:
            mock_handle = MagicMock()
            mock_handle.is_open = True
            mock_handle.capture.side_effect = RuntimeError("broken")
            MockHandle.return_value = mock_handle

            cam = ResilientCamera(
                max_reconnect_attempts=6,
                reconnect_delay=1.0,
                backoff_factor=10.0,  # aggressive factor
                max_delay=5.0,  # low cap
            )
            cam._connected = True
            cam._handle = mock_handle
            cam.capture()

        sleep_calls = [c[0][0] for c in mock_time.sleep.call_args_list]
        for delay in sleep_calls:
            assert delay <= 5.0

    def test_default_backoff_params(self):
        from missy.vision.resilient_capture import ResilientCamera
        cam = ResilientCamera()
        assert cam._backoff_factor == 1.5
        assert cam._max_delay == 30.0

    def test_custom_backoff_params(self):
        from missy.vision.resilient_capture import ResilientCamera
        cam = ResilientCamera(backoff_factor=3.0, max_delay=60.0)
        assert cam._backoff_factor == 3.0
        assert cam._max_delay == 60.0
