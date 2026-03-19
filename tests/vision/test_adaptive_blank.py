"""Tests for adaptive blank frame detection."""

from __future__ import annotations

import numpy as np
import pytest

from missy.vision.capture import AdaptiveBlankDetector, CameraHandle, CaptureConfig


class TestAdaptiveBlankDetector:
    """Tests for AdaptiveBlankDetector."""

    def test_default_threshold_with_no_history(self) -> None:
        det = AdaptiveBlankDetector(base_threshold=5.0)
        assert det.threshold == 5.0

    def test_threshold_adapts_after_recording(self) -> None:
        det = AdaptiveBlankDetector(base_threshold=5.0, adaptation_factor=0.25)
        # Record several dim-but-valid frames
        for _ in range(5):
            det.record_intensity(20.0)
        # Adaptive threshold = min(20) * 0.25 = 5.0
        assert det.threshold == pytest.approx(5.0)

    def test_threshold_lowers_for_dim_environment(self) -> None:
        det = AdaptiveBlankDetector(base_threshold=10.0, adaptation_factor=0.25)
        for _ in range(5):
            det.record_intensity(12.0)
        # Adaptive = 12.0 * 0.25 = 3.0
        assert det.threshold == pytest.approx(3.0)

    def test_threshold_respects_minimum(self) -> None:
        det = AdaptiveBlankDetector(
            base_threshold=10.0, min_threshold=2.0, adaptation_factor=0.1
        )
        for _ in range(5):
            det.record_intensity(5.0)
        # Adaptive = 5.0 * 0.1 = 0.5, but min is 2.0
        assert det.threshold == 2.0

    def test_threshold_capped_at_base(self) -> None:
        det = AdaptiveBlankDetector(base_threshold=5.0, adaptation_factor=0.5)
        for _ in range(5):
            det.record_intensity(200.0)
        # Adaptive = 200 * 0.5 = 100, but capped at base=5.0
        assert det.threshold == 5.0

    def test_is_blank_with_adaptive(self) -> None:
        det = AdaptiveBlankDetector(base_threshold=5.0, adaptation_factor=0.25)
        for _ in range(5):
            det.record_intensity(100.0)
        # Adaptive threshold = 100 * 0.25 = 25, capped at 5.0
        # So a frame with mean=3.0 should be blank
        black = np.full((10, 10, 3), 3, dtype=np.uint8)
        assert det.is_blank(black) is True

    def test_is_not_blank_for_valid_frame(self) -> None:
        det = AdaptiveBlankDetector(base_threshold=5.0)
        frame = np.full((10, 10, 3), 50, dtype=np.uint8)
        assert det.is_blank(frame) is False

    def test_requires_minimum_history(self) -> None:
        det = AdaptiveBlankDetector(base_threshold=5.0, adaptation_factor=0.25)
        # Only 2 recordings (need 3 for adaptation)
        det.record_intensity(100.0)
        det.record_intensity(100.0)
        assert det.threshold == 5.0  # still base

    def test_window_size_limit(self) -> None:
        det = AdaptiveBlankDetector(
            base_threshold=10.0, window_size=3, adaptation_factor=0.5
        )
        det.record_intensity(100.0)
        det.record_intensity(100.0)
        det.record_intensity(100.0)
        # Now record low values that push out the old ones
        det.record_intensity(10.0)
        det.record_intensity(10.0)
        det.record_intensity(10.0)
        # Adaptive = 10 * 0.5 = 5.0
        assert det.threshold == pytest.approx(5.0)

    def test_zero_intensity_ignored(self) -> None:
        det = AdaptiveBlankDetector(base_threshold=5.0)
        det.record_intensity(0.0)  # should be ignored
        assert det.threshold == 5.0  # still base

    def test_reset_clears_history(self) -> None:
        det = AdaptiveBlankDetector(base_threshold=10.0, adaptation_factor=0.25)
        for _ in range(5):
            det.record_intensity(12.0)
        assert det.threshold == pytest.approx(3.0)  # adapted: 12*0.25=3.0
        det.reset()
        assert det.threshold == 10.0  # back to base


class TestCameraHandleAdaptiveBlank:
    """Tests for adaptive blank detection integration in CameraHandle."""

    def test_adaptive_blank_enabled_by_default(self) -> None:
        config = CaptureConfig()
        assert config.adaptive_blank is True

    def test_camera_handle_creates_detector_when_adaptive(self) -> None:
        config = CaptureConfig(adaptive_blank=True, blank_threshold=5.0)
        handle = CameraHandle("/dev/video0", config)
        assert handle._blank_detector is not None
        assert handle._blank_detector.threshold == 5.0

    def test_camera_handle_no_detector_when_disabled(self) -> None:
        config = CaptureConfig(adaptive_blank=False)
        handle = CameraHandle("/dev/video0", config)
        assert handle._blank_detector is None

    def test_is_blank_delegates_to_detector(self) -> None:
        config = CaptureConfig(adaptive_blank=True, blank_threshold=5.0)
        handle = CameraHandle("/dev/video0", config)
        # Feed history to lower threshold via adaptation
        for _ in range(5):
            handle._blank_detector.record_intensity(100.0)
        # Frame with mean=3 should be blank (threshold=5)
        blank = np.full((10, 10, 3), 3, dtype=np.uint8)
        assert handle._is_blank(blank) is True

    def test_is_blank_fixed_when_no_detector(self) -> None:
        config = CaptureConfig(adaptive_blank=False, blank_threshold=5.0)
        handle = CameraHandle("/dev/video0", config)
        # Frame with mean=3 is below threshold
        blank = np.full((10, 10, 3), 3, dtype=np.uint8)
        assert handle._is_blank(blank) is True

    def test_record_successful_frame_updates_detector(self) -> None:
        config = CaptureConfig(adaptive_blank=True, blank_threshold=5.0)
        handle = CameraHandle("/dev/video0", config)
        frame = np.full((10, 10, 3), 120, dtype=np.uint8)
        handle._record_successful_frame(frame)
        assert len(handle._blank_detector._window) == 1

    def test_record_successful_frame_noop_without_detector(self) -> None:
        config = CaptureConfig(adaptive_blank=False)
        handle = CameraHandle("/dev/video0", config)
        frame = np.full((10, 10, 3), 120, dtype=np.uint8)
        # Should not raise
        handle._record_successful_frame(frame)
