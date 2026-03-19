"""Extended tests for missy.vision.capture — edge cases and error paths."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from missy.vision.capture import CameraHandle, CaptureConfig, CaptureError, CaptureResult


class TestBlankFrameDetection:
    """Tests for _is_blank() method edge cases."""

    def test_all_black_is_blank(self):
        handle = CameraHandle("/dev/video0")
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        assert handle._is_blank(frame) is True

    def test_dim_below_threshold_is_blank(self):
        handle = CameraHandle("/dev/video0", CaptureConfig(blank_threshold=10.0))
        frame = np.full((100, 100, 3), 3, dtype=np.uint8)
        assert handle._is_blank(frame) is True

    def test_normal_image_not_blank(self):
        handle = CameraHandle("/dev/video0")
        frame = np.full((100, 100, 3), 128, dtype=np.uint8)
        assert handle._is_blank(frame) is False

    def test_all_white_not_blank(self):
        handle = CameraHandle("/dev/video0")
        frame = np.full((100, 100, 3), 255, dtype=np.uint8)
        assert handle._is_blank(frame) is False

    def test_threshold_boundary(self):
        handle = CameraHandle("/dev/video0", CaptureConfig(blank_threshold=5.0))
        # Exactly at threshold
        frame = np.full((100, 100, 3), 5, dtype=np.uint8)
        # mean == 5.0 which is NOT < 5.0
        assert handle._is_blank(frame) is False

    def test_just_below_threshold(self):
        handle = CameraHandle("/dev/video0", CaptureConfig(blank_threshold=5.0))
        frame = np.full((100, 100, 3), 4, dtype=np.uint8)
        assert handle._is_blank(frame) is True

    def test_single_channel_blank(self):
        handle = CameraHandle("/dev/video0")
        frame = np.zeros((100, 100), dtype=np.uint8)
        assert handle._is_blank(frame) is True


class TestWarmup:
    """Tests for _warmup() behavior."""

    @patch("missy.vision.capture._get_cv2")
    def test_warmup_discards_frames(self, mock_cv2_fn):
        mock_cv2 = MagicMock()
        mock_cv2_fn.return_value = mock_cv2
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.get.return_value = 1920.0
        # _warmup now unpacks ret, frame = self._cap.read()
        import numpy as np

        frame = np.full((100, 100, 3), 128, dtype=np.uint8)
        mock_cap.read.return_value = (True, frame)

        handle = CameraHandle("/dev/video0", CaptureConfig(warmup_frames=5))
        handle._cap = mock_cap
        handle._warmup()

        assert mock_cap.read.call_count == 5

    @patch("missy.vision.capture._get_cv2")
    def test_warmup_zero_frames(self, mock_cv2_fn):
        mock_cv2 = MagicMock()
        mock_cv2_fn.return_value = mock_cv2
        mock_cap = MagicMock()

        handle = CameraHandle("/dev/video0", CaptureConfig(warmup_frames=0))
        handle._cap = mock_cap
        handle._warmup()

        mock_cap.read.assert_not_called()

    @patch("missy.vision.capture._get_cv2")
    def test_warmup_exception_stops_early(self, mock_cv2_fn):
        mock_cv2 = MagicMock()
        mock_cv2_fn.return_value = mock_cv2
        mock_cap = MagicMock()
        import numpy as np

        frame = np.full((100, 100, 3), 128, dtype=np.uint8)
        mock_cap.read.side_effect = [(True, frame), RuntimeError("device lost")]

        handle = CameraHandle("/dev/video0", CaptureConfig(warmup_frames=10))
        handle._cap = mock_cap
        # Should not raise
        handle._warmup()
        assert mock_cap.read.call_count == 2


class TestCaptureToFile:
    """Tests for capture_to_file edge cases."""

    @patch("missy.vision.capture._get_cv2")
    def test_capture_to_file_creates_dirs(self, mock_cv2_fn, tmp_path):
        mock_cv2 = MagicMock()
        mock_cv2_fn.return_value = mock_cv2
        mock_cv2.IMWRITE_JPEG_QUALITY = 1
        mock_cv2.imwrite.return_value = True

        handle = CameraHandle("/dev/video0")
        handle._opened = True
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        frame = np.full((100, 100, 3), 128, dtype=np.uint8)
        mock_cap.read.return_value = (True, frame)
        handle._cap = mock_cap

        nested = tmp_path / "a" / "b" / "c" / "test.jpg"
        result = handle.capture_to_file(nested)

        assert result.success
        assert nested.parent.exists()

    @patch("missy.vision.capture._get_cv2")
    def test_capture_to_file_imwrite_failure(self, mock_cv2_fn, tmp_path):
        mock_cv2 = MagicMock()
        mock_cv2_fn.return_value = mock_cv2
        mock_cv2.IMWRITE_JPEG_QUALITY = 1
        mock_cv2.imwrite.return_value = False

        handle = CameraHandle("/dev/video0")
        handle._opened = True
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        frame = np.full((100, 100, 3), 128, dtype=np.uint8)
        mock_cap.read.return_value = (True, frame)
        handle._cap = mock_cap

        out = tmp_path / "test.jpg"
        result = handle.capture_to_file(out)

        assert not result.success
        assert "imwrite failed" in result.error

    @patch("missy.vision.capture._get_cv2")
    def test_capture_to_file_save_exception(self, mock_cv2_fn, tmp_path):
        mock_cv2 = MagicMock()
        mock_cv2_fn.return_value = mock_cv2
        mock_cv2.IMWRITE_JPEG_QUALITY = 1
        mock_cv2.imwrite.side_effect = OSError("disk full")

        handle = CameraHandle("/dev/video0")
        handle._opened = True
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        frame = np.full((100, 100, 3), 128, dtype=np.uint8)
        mock_cap.read.return_value = (True, frame)
        handle._cap = mock_cap

        out = tmp_path / "test.jpg"
        result = handle.capture_to_file(out)

        assert not result.success
        assert "disk full" in result.error

    @patch("missy.vision.capture._get_cv2")
    def test_capture_to_file_png_params(self, mock_cv2_fn, tmp_path):
        mock_cv2 = MagicMock()
        mock_cv2_fn.return_value = mock_cv2
        mock_cv2.IMWRITE_PNG_COMPRESSION = 16
        mock_cv2.imwrite.return_value = True

        handle = CameraHandle("/dev/video0")
        handle._opened = True
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        frame = np.full((100, 100, 3), 128, dtype=np.uint8)
        mock_cap.read.return_value = (True, frame)
        handle._cap = mock_cap

        out = tmp_path / "test.png"
        result = handle.capture_to_file(out, quality=90)

        assert result.success
        # PNG compression param should be passed
        call_args = mock_cv2.imwrite.call_args
        assert mock_cv2.IMWRITE_PNG_COMPRESSION in call_args[0][2]

    @patch("missy.vision.capture._get_cv2")
    def test_capture_to_file_capture_fails(self, mock_cv2_fn, tmp_path):
        mock_cv2 = MagicMock()
        mock_cv2_fn.return_value = mock_cv2

        handle = CameraHandle("/dev/video0")
        handle._opened = True
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.read.return_value = (False, None)
        handle._cap = mock_cap

        out = tmp_path / "test.jpg"
        result = handle.capture_to_file(out)

        assert not result.success


class TestDeviceIndex:
    """Tests for _parse_device_index."""

    def test_video0(self):
        assert CameraHandle._parse_device_index("/dev/video0") == 0

    def test_video12(self):
        assert CameraHandle._parse_device_index("/dev/video12") == 12

    def test_non_video_path(self):
        assert CameraHandle._parse_device_index("/dev/usb/cam0") is None

    def test_empty_string(self):
        assert CameraHandle._parse_device_index("") is None

    def test_partial_match(self):
        assert CameraHandle._parse_device_index("/dev/video0extra") is None


class TestCaptureValidation:
    def test_empty_device_path_rejected(self):
        with pytest.raises(ValueError, match="non-empty"):
            CameraHandle("")

    def test_capture_when_not_open(self):
        handle = CameraHandle("/dev/video0")
        with pytest.raises(CaptureError, match="not open"):
            handle.capture()

    def test_double_close_safe(self):
        handle = CameraHandle("/dev/video0")
        handle.close()
        handle.close()  # Should not raise


class TestCaptureResult:
    def test_shape_with_image(self):
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        result = CaptureResult(success=True, image=img)
        assert result.shape == (480, 640, 3)

    def test_shape_without_image(self):
        result = CaptureResult(success=False)
        assert result.shape == (0, 0, 0)


class TestBurstCapture:
    @patch("missy.vision.capture._get_cv2")
    def test_burst_count_validation(self, mock_cv2_fn):
        handle = CameraHandle("/dev/video0")
        with pytest.raises(ValueError, match="count must be >= 1"):
            handle.capture_burst(count=0)

    @patch("missy.vision.capture._get_cv2")
    def test_burst_clamps_to_20(self, mock_cv2_fn):
        mock_cv2 = MagicMock()
        mock_cv2_fn.return_value = mock_cv2

        handle = CameraHandle("/dev/video0")
        handle._opened = True
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        frame = np.full((100, 100, 3), 128, dtype=np.uint8)
        mock_cap.read.return_value = (True, frame)
        handle._cap = mock_cap

        results = handle.capture_burst(count=50, interval=0)
        assert len(results) == 20
