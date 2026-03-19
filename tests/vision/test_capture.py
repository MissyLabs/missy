"""Tests for missy.vision.capture — frame capture with resilience."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from missy.vision.capture import (
    CameraHandle,
    CaptureConfig,
    CaptureError,
    CaptureResult,
)


# ---------------------------------------------------------------------------
# CaptureConfig tests
# ---------------------------------------------------------------------------


class TestCaptureConfig:
    def test_defaults(self):
        cfg = CaptureConfig()
        assert cfg.width == 1920
        assert cfg.height == 1080
        assert cfg.warmup_frames == 5
        assert cfg.max_retries == 3
        assert cfg.blank_threshold == 5.0

    def test_custom(self):
        cfg = CaptureConfig(width=640, height=480, warmup_frames=0, max_retries=1)
        assert cfg.width == 640
        assert cfg.max_retries == 1


# ---------------------------------------------------------------------------
# CaptureResult tests
# ---------------------------------------------------------------------------


class TestCaptureResult:
    def test_success_result(self):
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        result = CaptureResult(success=True, image=img, width=640, height=480)
        assert result.success is True
        assert result.shape == (480, 640, 3)

    def test_failure_result(self):
        result = CaptureResult(success=False, error="camera busy")
        assert result.success is False
        assert result.shape == (0, 0, 0)
        assert "busy" in result.error


# ---------------------------------------------------------------------------
# CameraHandle tests (mocked OpenCV)
# ---------------------------------------------------------------------------


class TestCameraHandle:
    @patch("missy.vision.capture._get_cv2")
    def test_open_success(self, mock_cv2_fn):
        mock_cv2 = MagicMock()
        mock_cv2_fn.return_value = mock_cv2
        mock_cv2.CAP_V4L2 = 200

        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.get.return_value = 1920
        mock_cap.read.return_value = (True, np.zeros((10, 10, 3), dtype=np.uint8))
        mock_cv2.VideoCapture.return_value = mock_cap

        config = CaptureConfig(warmup_frames=2)
        handle = CameraHandle("/dev/video0", config)
        handle.open()

        assert handle.is_open
        # Warmup should have called read 2 times
        assert mock_cap.read.call_count == 2

        handle.close()
        assert not handle.is_open

    @patch("missy.vision.capture._get_cv2")
    def test_open_failure(self, mock_cv2_fn):
        mock_cv2 = MagicMock()
        mock_cv2_fn.return_value = mock_cv2
        mock_cv2.CAP_V4L2 = 200

        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = False
        mock_cv2.VideoCapture.return_value = mock_cap

        handle = CameraHandle("/dev/video0")
        with pytest.raises(CaptureError, match="Cannot open camera"):
            handle.open()

    @patch("missy.vision.capture._get_cv2")
    def test_capture_success(self, mock_cv2_fn):
        mock_cv2 = MagicMock()
        mock_cv2_fn.return_value = mock_cv2
        mock_cv2.CAP_V4L2 = 200

        frame = np.full((480, 640, 3), 128, dtype=np.uint8)
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.get.return_value = 640
        mock_cap.read.return_value = (True, frame)
        mock_cv2.VideoCapture.return_value = mock_cap

        config = CaptureConfig(warmup_frames=0)
        handle = CameraHandle("/dev/video0", config)
        handle.open()
        result = handle.capture()

        assert result.success is True
        assert result.width == 640
        assert result.height == 480

        handle.close()

    @patch("missy.vision.capture._get_cv2")
    def test_capture_blank_frame_retry(self, mock_cv2_fn):
        mock_cv2 = MagicMock()
        mock_cv2_fn.return_value = mock_cv2
        mock_cv2.CAP_V4L2 = 200

        blank_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        good_frame = np.full((480, 640, 3), 128, dtype=np.uint8)

        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.get.return_value = 640
        mock_cap.read.side_effect = [
            (True, blank_frame),
            (True, good_frame),
        ]
        mock_cv2.VideoCapture.return_value = mock_cap

        config = CaptureConfig(warmup_frames=0, max_retries=3, retry_delay=0)
        handle = CameraHandle("/dev/video0", config)
        handle.open()
        result = handle.capture()

        assert result.success is True
        assert result.attempt_count == 2

        handle.close()

    @patch("missy.vision.capture._get_cv2")
    def test_capture_all_retries_fail(self, mock_cv2_fn):
        mock_cv2 = MagicMock()
        mock_cv2_fn.return_value = mock_cv2
        mock_cv2.CAP_V4L2 = 200

        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.get.return_value = 640
        mock_cap.read.return_value = (False, None)
        mock_cv2.VideoCapture.return_value = mock_cap

        config = CaptureConfig(warmup_frames=0, max_retries=2, retry_delay=0)
        handle = CameraHandle("/dev/video0", config)
        handle.open()
        result = handle.capture()

        assert result.success is False
        assert "2 attempts" in result.error

        handle.close()

    @patch("missy.vision.capture._get_cv2")
    def test_capture_not_open_raises(self, mock_cv2_fn):
        handle = CameraHandle("/dev/video0")
        with pytest.raises(CaptureError, match="not open"):
            handle.capture()

    @patch("missy.vision.capture._get_cv2")
    def test_context_manager(self, mock_cv2_fn):
        mock_cv2 = MagicMock()
        mock_cv2_fn.return_value = mock_cv2
        mock_cv2.CAP_V4L2 = 200

        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.get.return_value = 640
        mock_cap.read.return_value = (True, np.zeros((10, 10, 3), dtype=np.uint8))
        mock_cv2.VideoCapture.return_value = mock_cap

        config = CaptureConfig(warmup_frames=0)
        with CameraHandle("/dev/video0", config) as handle:
            assert handle.is_open
        assert not handle.is_open

    def test_parse_device_index(self):
        assert CameraHandle._parse_device_index("/dev/video0") == 0
        assert CameraHandle._parse_device_index("/dev/video2") == 2
        assert CameraHandle._parse_device_index("/dev/video10") == 10
        assert CameraHandle._parse_device_index("/dev/sda") is None
        assert CameraHandle._parse_device_index("video0") is None

    @patch("missy.vision.capture._get_cv2")
    def test_capture_to_file(self, mock_cv2_fn, tmp_path):
        mock_cv2 = MagicMock()
        mock_cv2_fn.return_value = mock_cv2
        mock_cv2.CAP_V4L2 = 200
        mock_cv2.IMWRITE_JPEG_QUALITY = 1

        frame = np.full((480, 640, 3), 128, dtype=np.uint8)
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.get.return_value = 640
        mock_cap.read.return_value = (True, frame)
        mock_cv2.VideoCapture.return_value = mock_cap
        mock_cv2.imwrite.return_value = True

        config = CaptureConfig(warmup_frames=0)
        handle = CameraHandle("/dev/video0", config)
        handle.open()

        out_file = tmp_path / "test.jpg"
        result = handle.capture_to_file(str(out_file))

        assert result.success is True
        mock_cv2.imwrite.assert_called_once()

        handle.close()
