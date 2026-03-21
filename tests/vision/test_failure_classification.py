"""Tests for failure classification features in the vision subsystem.

Covers:
- FailureType enum values
- CaptureResult.failure_type default
- Resolution mismatch warning in CameraHandle.open()
- Failure type set on CameraHandle.capture() exceptions
- ResilientCamera cumulative failure tracking
- ResilientCamera permanent failure early return (PERMISSION, UNSUPPORTED)
- ResilientCamera device path change warning
- ImagePipeline.assess_quality() saturation and noise_level metrics
"""

from __future__ import annotations

import logging
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from missy.vision.capture import (
    CameraHandle,
    CaptureConfig,
    CaptureResult,
    FailureType,
)
from missy.vision.discovery import CameraDevice
from missy.vision.pipeline import ImagePipeline, PipelineConfig
from missy.vision.resilient_capture import ResilientCamera

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_camera(path: str = "/dev/video0", name: str = "Test Cam") -> CameraDevice:
    return CameraDevice(
        device_path=path,
        name=name,
        vendor_id="046d",
        product_id="085c",
        bus_info="usb-1",
    )


def _make_cv2_mock(
    *,
    width: int = 1920,
    height: int = 1080,
    warmup_frames: int = 0,
) -> tuple[MagicMock, MagicMock]:
    """Return (mock_cv2_fn_return_value, mock_cap) with sensible defaults."""
    mock_cv2 = MagicMock()
    mock_cv2.CAP_V4L2 = 200
    mock_cv2.CAP_PROP_FRAME_WIDTH = 3
    mock_cv2.CAP_PROP_FRAME_HEIGHT = 4
    mock_cv2.CV_64F = 6

    mock_cap = MagicMock()
    mock_cap.isOpened.return_value = True
    # Return the requested resolution by default
    mock_cap.get.side_effect = lambda prop: (
        float(width) if prop == mock_cv2.CAP_PROP_FRAME_WIDTH else float(height)
    )
    # Warmup reads return a non-blank frame
    warmup_frame = np.full((height, width, 3), 128, dtype=np.uint8)
    mock_cap.read.return_value = (True, warmup_frame)
    mock_cv2.VideoCapture.return_value = mock_cap

    return mock_cv2, mock_cap


# ---------------------------------------------------------------------------
# 1. FailureType enum
# ---------------------------------------------------------------------------


class TestFailureTypeEnum:
    def test_all_values_exist(self):
        assert FailureType.TRANSIENT == "transient"
        assert FailureType.PERMISSION == "permission"
        assert FailureType.DEVICE_GONE == "device_gone"
        assert FailureType.UNSUPPORTED == "unsupported"
        assert FailureType.UNKNOWN == "unknown"

    def test_all_values_are_strings(self):
        for member in FailureType:
            assert isinstance(member, str), f"{member!r} should be a str (StrEnum)"

    def test_enum_has_exactly_five_members(self):
        assert len(list(FailureType)) == 5

    def test_enum_members_are_lowercase(self):
        for member in FailureType:
            assert member == member.lower(), f"{member!r} should be lowercase"


# ---------------------------------------------------------------------------
# 2. CaptureResult.failure_type default
# ---------------------------------------------------------------------------


class TestCaptureResultFailureType:
    def test_default_failure_type_is_empty_string(self):
        result = CaptureResult(success=True)
        assert result.failure_type == ""

    def test_failure_type_can_be_set_to_enum_value(self):
        result = CaptureResult(success=False, failure_type=FailureType.TRANSIENT)
        assert result.failure_type == "transient"

    def test_failure_type_can_be_set_to_plain_string(self):
        result = CaptureResult(success=False, failure_type="permission")
        assert result.failure_type == "permission"

    def test_success_result_keeps_empty_failure_type(self):
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        result = CaptureResult(success=True, image=img, width=100, height=100)
        assert result.failure_type == ""


# ---------------------------------------------------------------------------
# 3. Resolution verification — warning when camera returns different size
# ---------------------------------------------------------------------------


class TestResolutionVerification:
    @patch("missy.vision.capture._get_cv2")
    def test_warning_logged_when_resolution_differs(self, mock_cv2_fn, caplog):
        """Camera delivers 640x480 when 1920x1080 was requested → warning emitted."""
        mock_cv2, mock_cap = _make_cv2_mock(width=1920, height=1080)
        mock_cv2_fn.return_value = mock_cv2

        # Override .get() to return a smaller-than-requested resolution
        mock_cap.get.side_effect = lambda prop: (
            640.0 if prop == mock_cv2.CAP_PROP_FRAME_WIDTH else 480.0
        )

        config = CaptureConfig(width=1920, height=1080, warmup_frames=0)
        handle = CameraHandle("/dev/video0", config)

        with caplog.at_level(logging.WARNING, logger="missy.vision.capture"):
            handle.open()

        assert any("640" in r.message and "480" in r.message for r in caplog.records), (
            "Expected a warning containing the actual resolution (640x480)"
        )
        handle.close()

    @patch("missy.vision.capture._get_cv2")
    def test_no_warning_when_resolution_matches(self, mock_cv2_fn, caplog):
        """No resolution warning when camera honours the requested size."""
        mock_cv2, mock_cap = _make_cv2_mock(width=1920, height=1080)
        mock_cv2_fn.return_value = mock_cv2

        config = CaptureConfig(width=1920, height=1080, warmup_frames=0)
        handle = CameraHandle("/dev/video0", config)

        with caplog.at_level(logging.WARNING, logger="missy.vision.capture"):
            handle.open()

        resolution_warnings = [r for r in caplog.records if "Requested resolution" in r.message]
        assert resolution_warnings == []
        handle.close()


# ---------------------------------------------------------------------------
# 4. Capture failure classification
# ---------------------------------------------------------------------------


class TestCaptureFailureClassification:
    def _open_handle(self, mock_cv2_fn, mock_cv2, mock_cap, config=None):
        mock_cv2_fn.return_value = mock_cv2
        cfg = config or CaptureConfig(warmup_frames=0, max_retries=1, retry_delay=0)
        handle = CameraHandle("/dev/video0", cfg)
        handle.open()
        return handle

    @patch("missy.vision.capture._get_cv2")
    def test_permission_error_sets_permission_failure_type(self, mock_cv2_fn):
        mock_cv2, mock_cap = _make_cv2_mock()
        handle = self._open_handle(mock_cv2_fn, mock_cv2, mock_cap)

        mock_cap.read.side_effect = PermissionError("Permission denied: /dev/video0")

        result = handle.capture()

        assert result.success is False
        assert result.failure_type == FailureType.PERMISSION
        handle.close()

    @patch("missy.vision.capture._get_cv2")
    def test_device_gone_error_sets_device_gone_failure_type(self, mock_cv2_fn):
        mock_cv2, mock_cap = _make_cv2_mock()
        handle = self._open_handle(mock_cv2_fn, mock_cv2, mock_cap)

        mock_cap.read.side_effect = OSError("No such device: /dev/video0")

        result = handle.capture()

        assert result.success is False
        assert result.failure_type == FailureType.DEVICE_GONE
        handle.close()

    @patch("missy.vision.capture._get_cv2")
    def test_no_such_file_sets_device_gone_failure_type(self, mock_cv2_fn):
        mock_cv2, mock_cap = _make_cv2_mock()
        handle = self._open_handle(mock_cv2_fn, mock_cv2, mock_cap)

        mock_cap.read.side_effect = FileNotFoundError("No such file or directory")

        result = handle.capture()

        assert result.success is False
        assert result.failure_type == FailureType.DEVICE_GONE
        handle.close()

    @patch("missy.vision.capture._get_cv2")
    def test_generic_exception_sets_transient_failure_type(self, mock_cv2_fn):
        mock_cv2, mock_cap = _make_cv2_mock()
        handle = self._open_handle(mock_cv2_fn, mock_cv2, mock_cap)

        mock_cap.read.side_effect = RuntimeError("Unexpected camera error")

        result = handle.capture()

        assert result.success is False
        assert result.failure_type == FailureType.TRANSIENT
        handle.close()

    @patch("missy.vision.capture._get_cv2")
    def test_read_false_sets_transient_failure_type(self, mock_cv2_fn):
        """Non-exception read failure (ret=False) also results in TRANSIENT."""
        mock_cv2, mock_cap = _make_cv2_mock()
        handle = self._open_handle(mock_cv2_fn, mock_cv2, mock_cap)

        mock_cap.read.return_value = (False, None)

        result = handle.capture()

        assert result.success is False
        assert result.failure_type == FailureType.TRANSIENT
        handle.close()


# ---------------------------------------------------------------------------
# 5. ResilientCamera cumulative failures
# ---------------------------------------------------------------------------


class TestResilientCameraCumulativeFailures:
    def test_initial_cumulative_failures_is_zero(self):
        cam = ResilientCamera()
        assert cam.cumulative_failures == 0

    @patch("missy.vision.resilient_capture.CameraHandle")
    @patch("missy.vision.resilient_capture.get_discovery")
    def test_failure_increments_cumulative_counter(self, mock_disc_fn, mock_handle_cls):
        mock_disc = MagicMock()
        mock_disc.find_preferred.return_value = _make_camera()
        mock_disc.discover.return_value = [_make_camera()]
        mock_disc.find_by_usb_id.return_value = None
        mock_disc_fn.return_value = mock_disc

        fail_result = CaptureResult(success=False, error="read failed")
        mock_handle = MagicMock()
        mock_handle.is_open = True
        mock_handle.capture.return_value = fail_result
        mock_handle_cls.return_value = mock_handle

        cam = ResilientCamera(max_reconnect_attempts=1, reconnect_delay=0)
        cam.connect()
        cam.capture()

        assert cam.cumulative_failures >= 1

    @patch("missy.vision.resilient_capture.CameraHandle")
    @patch("missy.vision.resilient_capture.get_discovery")
    def test_warning_logged_at_cumulative_failure_threshold(
        self, mock_disc_fn, mock_handle_cls, caplog
    ):
        """A warning must be emitted once cumulative_failures hits the threshold."""
        mock_disc = MagicMock()
        mock_disc.find_preferred.return_value = _make_camera()
        mock_disc.find_by_usb_id.return_value = None
        mock_disc.discover.return_value = []
        mock_disc_fn.return_value = mock_disc

        mock_handle = MagicMock()
        mock_handle.is_open = False
        mock_handle_cls.return_value = mock_handle

        threshold = ResilientCamera._CUMULATIVE_FAILURE_THRESHOLD

        # Build a cam already one below threshold; next _record_failure triggers warning
        cam = ResilientCamera(max_reconnect_attempts=0, reconnect_delay=0)
        cam._cumulative_failures = threshold - 1

        with caplog.at_level(logging.WARNING, logger="missy.vision.resilient_capture"):
            cam._record_failure()

        assert cam.cumulative_failures == threshold
        assert any("cumulative failures" in r.message for r in caplog.records)

    def test_record_failure_increments_by_one_each_call(self):
        cam = ResilientCamera()
        cam._record_failure()
        assert cam.cumulative_failures == 1
        cam._record_failure()
        assert cam.cumulative_failures == 2


# ---------------------------------------------------------------------------
# 6. ResilientCamera permanent failure early return
# ---------------------------------------------------------------------------


class TestResilientCameraPermanentFailureSkip:
    @patch("missy.vision.resilient_capture.CameraHandle")
    @patch("missy.vision.resilient_capture.get_discovery")
    def test_permission_failure_returns_immediately_no_reconnect(
        self, mock_disc_fn, mock_handle_cls
    ):
        """PERMISSION failure_type must cause immediate return without reconnection."""
        mock_disc = MagicMock()
        mock_disc.find_preferred.return_value = _make_camera()
        mock_disc_fn.return_value = mock_disc

        perm_result = CaptureResult(
            success=False,
            error="Permission denied",
            failure_type=FailureType.PERMISSION,
        )
        mock_handle = MagicMock()
        mock_handle.is_open = True
        mock_handle.capture.return_value = perm_result
        mock_handle_cls.return_value = mock_handle

        cam = ResilientCamera(max_reconnect_attempts=5, reconnect_delay=0)
        cam.connect()
        result = cam.capture()

        assert result.success is False
        assert result.failure_type == FailureType.PERMISSION
        # discover() must NOT have been called (no reconnection attempted)
        mock_disc.discover.assert_not_called()

    @patch("missy.vision.resilient_capture.CameraHandle")
    @patch("missy.vision.resilient_capture.get_discovery")
    def test_unsupported_failure_returns_immediately_no_reconnect(
        self, mock_disc_fn, mock_handle_cls
    ):
        """UNSUPPORTED failure_type must cause immediate return without reconnection."""
        mock_disc = MagicMock()
        mock_disc.find_preferred.return_value = _make_camera()
        mock_disc_fn.return_value = mock_disc

        unsup_result = CaptureResult(
            success=False,
            error="Format unsupported",
            failure_type=FailureType.UNSUPPORTED,
        )
        mock_handle = MagicMock()
        mock_handle.is_open = True
        mock_handle.capture.return_value = unsup_result
        mock_handle_cls.return_value = mock_handle

        cam = ResilientCamera(max_reconnect_attempts=5, reconnect_delay=0)
        cam.connect()
        result = cam.capture()

        assert result.success is False
        assert result.failure_type == FailureType.UNSUPPORTED
        mock_disc.discover.assert_not_called()

    @patch("missy.vision.resilient_capture.CameraHandle")
    @patch("missy.vision.resilient_capture.get_discovery")
    def test_transient_failure_does_attempt_reconnect(self, mock_disc_fn, mock_handle_cls):
        """TRANSIENT failures should proceed to reconnection, not short-circuit."""
        mock_disc = MagicMock()
        mock_disc.find_preferred.return_value = _make_camera()
        mock_disc.find_by_usb_id.return_value = None
        mock_disc.discover.return_value = []
        mock_disc_fn.return_value = mock_disc

        transient_result = CaptureResult(
            success=False,
            error="read failed",
            failure_type=FailureType.TRANSIENT,
        )
        mock_handle = MagicMock()
        mock_handle.is_open = True
        mock_handle.capture.return_value = transient_result
        mock_handle_cls.return_value = mock_handle

        cam = ResilientCamera(max_reconnect_attempts=1, reconnect_delay=0)
        cam.connect()
        cam.capture()

        # discover() should have been called as part of reconnection
        mock_disc.discover.assert_called()


# ---------------------------------------------------------------------------
# 7. ResilientCamera device change detection
# ---------------------------------------------------------------------------


class TestResilientCameraDeviceChangeDetection:
    @patch("missy.vision.resilient_capture.CameraHandle")
    @patch("missy.vision.resilient_capture.get_discovery")
    def test_warning_logged_when_device_path_changes(self, mock_disc_fn, mock_handle_cls, caplog):
        """A warning should be logged when the reconnected camera has a different path."""
        old_device = _make_camera(path="/dev/video0")
        new_device = _make_camera(path="/dev/video2", name="Test Cam (re-enumerated)")

        mock_disc = MagicMock()
        # Initial connect returns the old device; rediscovery returns new path
        mock_disc.find_preferred.side_effect = [old_device, new_device]
        mock_disc.find_by_usb_id.return_value = None
        mock_disc.discover.return_value = [new_device]
        mock_disc_fn.return_value = mock_disc

        fail_result = CaptureResult(success=False, error="lost connection")
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

        cam = ResilientCamera(max_reconnect_attempts=2, reconnect_delay=0)
        cam.connect()

        with caplog.at_level(logging.WARNING, logger="missy.vision.resilient_capture"):
            cam.capture()

        path_change_warnings = [
            r
            for r in caplog.records
            if "path changed" in r.message.lower() or "/dev/video0" in r.message
        ]
        assert path_change_warnings, (
            "Expected a warning about the camera path changing from /dev/video0 to /dev/video2"
        )

    @patch("missy.vision.resilient_capture.CameraHandle")
    @patch("missy.vision.resilient_capture.get_discovery")
    def test_no_path_change_warning_when_same_device(self, mock_disc_fn, mock_handle_cls, caplog):
        """No path-change warning when the same device path is used after reconnect."""
        device = _make_camera(path="/dev/video0")

        mock_disc = MagicMock()
        mock_disc.find_preferred.return_value = device
        mock_disc.find_by_usb_id.return_value = None
        mock_disc.discover.return_value = [device]
        mock_disc_fn.return_value = mock_disc

        fail_result = CaptureResult(success=False, error="transient read error")
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

        cam = ResilientCamera(max_reconnect_attempts=2, reconnect_delay=0)
        cam.connect()

        with caplog.at_level(logging.WARNING, logger="missy.vision.resilient_capture"):
            cam.capture()

        path_change_warnings = [r for r in caplog.records if "path changed" in r.message.lower()]
        assert path_change_warnings == []


# ---------------------------------------------------------------------------
# 8 & 9. Pipeline saturation and noise metrics — real numpy arrays
# ---------------------------------------------------------------------------


class TestPipelineSaturationAndNoise:
    """Use real numpy arrays (no cv2 mock) to exercise the actual computation paths.

    The pipeline calls cv2 internally; we patch _get_cv2 to inject a lightweight
    stub for the LAB/HSV conversions while keeping numpy operations real.
    """

    def _saturated_bgr(self) -> np.ndarray:
        """Create a bright-red image that produces high HSV saturation."""
        # Pure red in BGR: B=0, G=0, R=255
        img = np.zeros((50, 50, 3), dtype=np.uint8)
        img[:, :, 2] = 255  # Red channel
        return img

    def _uniform_gray(self) -> np.ndarray:
        """Uniform mid-gray — low HSV saturation, low noise."""
        return np.full((50, 50, 3), 128, dtype=np.uint8)

    @patch("missy.vision.pipeline._get_cv2")
    def test_saturation_present_in_result(self, mock_cv2_fn):
        mock_cv2 = MagicMock()
        mock_cv2_fn.return_value = mock_cv2
        mock_cv2.CV_64F = 6

        # Gray image — saturation channel will be all zeros → mean = 0
        gray_image = self._uniform_gray()
        gray_single = np.full((50, 50), 128, dtype=np.uint8)
        mock_cv2.cvtColor.return_value = gray_single

        # HSV: shape (50, 50, 3) — saturation channel (index 1) all zeros
        hsv_image = np.zeros((50, 50, 3), dtype=np.uint8)
        hsv_image[:, :, 0] = 0  # hue
        hsv_image[:, :, 1] = 0  # saturation = 0
        hsv_image[:, :, 2] = 128  # value

        def cvt_side_effect(img, code):
            # Return HSV when asked for BGR→HSV, otherwise return gray
            if hasattr(mock_cv2, "COLOR_BGR2HSV") and code == mock_cv2.COLOR_BGR2HSV:
                return hsv_image
            return gray_single

        mock_cv2.cvtColor.side_effect = cvt_side_effect

        # Real Laplacian via numpy for noise_level and sharpness
        lap_array = np.zeros((50, 50), dtype=np.float64)
        mock_cv2.Laplacian.return_value = lap_array

        pipeline = ImagePipeline(PipelineConfig())
        result = pipeline.assess_quality(gray_image)

        assert "saturation" in result
        assert isinstance(result["saturation"], float)

    @patch("missy.vision.pipeline._get_cv2")
    def test_saturation_nonzero_for_colorful_image(self, mock_cv2_fn):
        """Saturated HSV input should produce saturation > 0."""
        mock_cv2 = MagicMock()
        mock_cv2_fn.return_value = mock_cv2
        mock_cv2.CV_64F = 6

        # Simulate a colorful image: HSV saturation channel at 200
        gray_single = np.full((50, 50), 128, dtype=np.uint8)
        hsv_image = np.zeros((50, 50, 3), dtype=np.uint8)
        hsv_image[:, :, 1] = 200  # high saturation

        def cvt_side_effect(img, code):
            if code == mock_cv2.COLOR_BGR2HSV:
                return hsv_image
            return gray_single

        mock_cv2.cvtColor.side_effect = cvt_side_effect

        lap_array = np.full((50, 50), 10.0, dtype=np.float64)
        mock_cv2.Laplacian.return_value = lap_array

        pipeline = ImagePipeline(PipelineConfig())
        result = pipeline.assess_quality(np.full((50, 50, 3), 128, dtype=np.uint8))

        assert result["saturation"] == pytest.approx(200.0, abs=1.0)

    @patch("missy.vision.pipeline._get_cv2")
    def test_noise_level_present_in_result(self, mock_cv2_fn):
        mock_cv2 = MagicMock()
        mock_cv2_fn.return_value = mock_cv2
        mock_cv2.CV_64F = 6

        gray_single = np.full((50, 50), 128, dtype=np.uint8)
        hsv_image = np.zeros((50, 50, 3), dtype=np.uint8)

        def cvt_side_effect(img, code):
            if code == mock_cv2.COLOR_BGR2HSV:
                return hsv_image
            return gray_single

        mock_cv2.cvtColor.side_effect = cvt_side_effect

        # Laplacian returns a real numpy array so noise_level computation runs
        lap_array = np.zeros((50, 50), dtype=np.float64)
        mock_cv2.Laplacian.return_value = lap_array

        pipeline = ImagePipeline(PipelineConfig())
        result = pipeline.assess_quality(np.full((50, 50, 3), 128, dtype=np.uint8))

        assert "noise_level" in result
        assert isinstance(result["noise_level"], float)

    @patch("missy.vision.pipeline._get_cv2")
    def test_noise_level_nonzero_for_noisy_laplacian(self, mock_cv2_fn):
        """Non-zero Laplacian values yield a positive noise_level."""
        mock_cv2 = MagicMock()
        mock_cv2_fn.return_value = mock_cv2
        mock_cv2.CV_64F = 6

        gray_single = np.full((50, 50), 128, dtype=np.uint8)
        hsv_image = np.zeros((50, 50, 3), dtype=np.uint8)

        def cvt_side_effect(img, code):
            if code == mock_cv2.COLOR_BGR2HSV:
                return hsv_image
            return gray_single

        mock_cv2.cvtColor.side_effect = cvt_side_effect

        # Simulate noisy Laplacian — uniform non-zero values
        noisy_lap = np.full((50, 50), 25.0, dtype=np.float64)
        mock_cv2.Laplacian.return_value = noisy_lap

        pipeline = ImagePipeline(PipelineConfig())
        result = pipeline.assess_quality(np.full((50, 50, 3), 128, dtype=np.uint8))

        # median(|laplacian|) * 1.4826 = 25.0 * 1.4826 ≈ 37.065
        assert result["noise_level"] == pytest.approx(25.0 * 1.4826, abs=0.1)


# ---------------------------------------------------------------------------
# 10. Pipeline quality assessment — all new fields present
# ---------------------------------------------------------------------------


class TestPipelineAssessQualityNewFields:
    @patch("missy.vision.pipeline._get_cv2")
    def test_saturation_and_noise_level_keys_always_present(self, mock_cv2_fn):
        """assess_quality() must always include saturation and noise_level keys."""
        mock_cv2 = MagicMock()
        mock_cv2_fn.return_value = mock_cv2
        mock_cv2.CV_64F = 6

        gray_single = np.full((80, 80), 150, dtype=np.uint8)
        hsv_image = np.zeros((80, 80, 3), dtype=np.uint8)
        hsv_image[:, :, 1] = 100

        def cvt_side_effect(img, code):
            if code == mock_cv2.COLOR_BGR2HSV:
                return hsv_image
            return gray_single

        mock_cv2.cvtColor.side_effect = cvt_side_effect
        lap_array = np.full((80, 80), 5.0, dtype=np.float64)
        mock_cv2.Laplacian.return_value = lap_array

        pipeline = ImagePipeline(PipelineConfig())
        result = pipeline.assess_quality(np.full((80, 80, 3), 150, dtype=np.uint8))

        required_keys = {
            "width",
            "height",
            "brightness",
            "contrast",
            "sharpness",
            "saturation",
            "noise_level",
            "quality",
            "issues",
        }
        assert required_keys.issubset(result.keys()), (
            f"Missing keys: {required_keys - result.keys()}"
        )

    @patch("missy.vision.pipeline._get_cv2")
    def test_saturation_zero_for_grayscale_image(self, mock_cv2_fn):
        """Grayscale (2-D) input has no colour, so saturation must be 0.0."""
        mock_cv2 = MagicMock()
        mock_cv2_fn.return_value = mock_cv2
        mock_cv2.CV_64F = 6

        lap_array = np.zeros((60, 60), dtype=np.float64)
        mock_cv2.Laplacian.return_value = lap_array

        # No cvtColor call expected for grayscale path
        pipeline = ImagePipeline(PipelineConfig())
        gray_image = np.full((60, 60), 128, dtype=np.uint8)  # 2-D → grayscale
        result = pipeline.assess_quality(gray_image)

        assert result["saturation"] == 0.0

    @patch("missy.vision.pipeline._get_cv2")
    def test_quality_rating_reflects_noise_issue(self, mock_cv2_fn):
        """High noise_level (>30) should add 'noisy' to issues list."""
        mock_cv2 = MagicMock()
        mock_cv2_fn.return_value = mock_cv2
        mock_cv2.CV_64F = 6

        gray_single = np.full((50, 50), 128, dtype=np.uint8)
        hsv_image = np.zeros((50, 50, 3), dtype=np.uint8)
        hsv_image[:, :, 1] = 80

        def cvt_side_effect(img, code):
            if code == mock_cv2.COLOR_BGR2HSV:
                return hsv_image
            return gray_single

        mock_cv2.cvtColor.side_effect = cvt_side_effect

        # noise_level = median(|lap|) * 1.4826; set lap to ~21.6 so result ≈ 32
        high_noise_lap = np.full((50, 50), 21.6, dtype=np.float64)
        mock_cv2.Laplacian.return_value = high_noise_lap

        pipeline = ImagePipeline(PipelineConfig())
        result = pipeline.assess_quality(np.full((50, 50, 3), 128, dtype=np.uint8))

        assert "noisy" in result["issues"]

    @patch("missy.vision.pipeline._get_cv2")
    def test_quality_rating_reflects_oversaturation_issue(self, mock_cv2_fn):
        """Saturation > 230 should add 'oversaturated' to issues list."""
        mock_cv2 = MagicMock()
        mock_cv2_fn.return_value = mock_cv2
        mock_cv2.CV_64F = 6

        gray_single = np.full((50, 50), 128, dtype=np.uint8)
        hsv_image = np.zeros((50, 50, 3), dtype=np.uint8)
        hsv_image[:, :, 1] = 240  # very high saturation

        def cvt_side_effect(img, code):
            if code == mock_cv2.COLOR_BGR2HSV:
                return hsv_image
            return gray_single

        mock_cv2.cvtColor.side_effect = cvt_side_effect

        lap_array = np.full((50, 50), 200.0, dtype=np.float64)
        mock_cv2.Laplacian.return_value = lap_array

        pipeline = ImagePipeline(PipelineConfig())
        result = pipeline.assess_quality(np.full((50, 50, 3), 128, dtype=np.uint8))

        assert "oversaturated" in result["issues"]
        assert result["saturation"] == pytest.approx(240.0, abs=1.0)

    @patch("missy.vision.pipeline._get_cv2")
    def test_quality_rating_reflects_desaturation_issue(self, mock_cv2_fn):
        """Low saturation (0 < saturation < 20) should add 'desaturated' to issues."""
        mock_cv2 = MagicMock()
        mock_cv2_fn.return_value = mock_cv2
        mock_cv2.CV_64F = 6

        gray_single = np.full((50, 50), 128, dtype=np.uint8)
        hsv_image = np.zeros((50, 50, 3), dtype=np.uint8)
        hsv_image[:, :, 1] = 10  # low (but non-zero) saturation

        def cvt_side_effect(img, code):
            if code == mock_cv2.COLOR_BGR2HSV:
                return hsv_image
            return gray_single

        mock_cv2.cvtColor.side_effect = cvt_side_effect

        lap_array = np.full((50, 50), 200.0, dtype=np.float64)
        mock_cv2.Laplacian.return_value = lap_array

        pipeline = ImagePipeline(PipelineConfig())
        result = pipeline.assess_quality(np.full((50, 50, 3), 128, dtype=np.uint8))

        assert "desaturated" in result["issues"]
