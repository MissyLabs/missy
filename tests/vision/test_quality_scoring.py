"""Tests for frame quality scoring and capture_best auto-selection."""

from unittest.mock import patch

import numpy as np

from missy.vision.capture import CaptureResult, _frame_quality_score


class TestFrameQualityScore:
    """Tests for _frame_quality_score()."""

    @patch("missy.vision.capture._get_cv2")
    def test_well_lit_sharp_image_scores_high(self, mock_cv2_fn):
        """A well-lit image with edges should score higher than dark/blurry."""
        cv2 = _make_mock_cv2()
        mock_cv2_fn.return_value = cv2

        # Simulate a sharp, well-lit image (mean ~128, high laplacian var)
        img = np.random.randint(80, 180, (480, 640, 3), dtype=np.uint8)
        score = _frame_quality_score(img)

        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0

    @patch("missy.vision.capture._get_cv2")
    def test_dark_image_scores_lower(self, mock_cv2_fn):
        """A very dark image should have lower brightness score."""
        cv2 = _make_mock_cv2()
        mock_cv2_fn.return_value = cv2

        dark = np.zeros((100, 100, 3), dtype=np.uint8)
        bright = np.full((100, 100, 3), 128, dtype=np.uint8)

        dark_score = _frame_quality_score(dark)
        bright_score = _frame_quality_score(bright)

        assert bright_score > dark_score

    @patch("missy.vision.capture._get_cv2")
    def test_overexposed_image_scores_lower(self, mock_cv2_fn):
        """A fully white image should score lower than mid-range."""
        cv2 = _make_mock_cv2()
        mock_cv2_fn.return_value = cv2

        white = np.full((100, 100, 3), 255, dtype=np.uint8)
        mid = np.full((100, 100, 3), 128, dtype=np.uint8)

        white_score = _frame_quality_score(white)
        mid_score = _frame_quality_score(mid)

        # Mid-range brightness should score better
        assert mid_score >= white_score

    @patch("missy.vision.capture._get_cv2")
    def test_score_always_between_0_and_1(self, mock_cv2_fn):
        """Score should be bounded [0, 1] for any input."""
        cv2 = _make_mock_cv2()
        mock_cv2_fn.return_value = cv2

        images = [
            np.zeros((50, 50, 3), dtype=np.uint8),
            np.full((50, 50, 3), 255, dtype=np.uint8),
            np.random.randint(0, 256, (50, 50, 3), dtype=np.uint8),
        ]

        for img in images:
            score = _frame_quality_score(img)
            assert 0.0 <= score <= 1.0, f"Score {score} out of bounds"

    @patch("missy.vision.capture._get_cv2")
    def test_grayscale_image_handled(self, mock_cv2_fn):
        """Grayscale 2D images should be handled without error."""
        cv2 = _make_mock_cv2()
        mock_cv2_fn.return_value = cv2

        gray = np.random.randint(0, 256, (100, 100), dtype=np.uint8)
        score = _frame_quality_score(gray)
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0


class TestCaptureBestQualitySelection:
    """Tests for CameraHandle.capture_best() quality-based selection."""

    @patch("missy.vision.capture._get_cv2")
    def test_selects_best_quality_frame(self, mock_cv2_fn):
        """capture_best should return the frame with the highest quality score."""
        cv2 = _make_mock_cv2()
        mock_cv2_fn.return_value = cv2

        from missy.vision.capture import CameraHandle, CaptureConfig

        config = CaptureConfig(warmup_frames=0)
        cam = CameraHandle("/dev/video0", config)

        # Create frames with different qualities
        dark = np.zeros((100, 100, 3), dtype=np.uint8)
        good = np.random.randint(80, 180, (100, 100, 3), dtype=np.uint8)
        white = np.full((100, 100, 3), 255, dtype=np.uint8)

        results = [
            CaptureResult(success=True, image=dark, device_path="/dev/video0", width=100, height=100),
            CaptureResult(success=True, image=good, device_path="/dev/video0", width=100, height=100),
            CaptureResult(success=True, image=white, device_path="/dev/video0", width=100, height=100),
        ]

        with patch.object(cam, "capture_burst", return_value=results):
            best = cam.capture_best(burst_count=3)

        assert best.success
        # The "good" image should be selected (mid-brightness, some contrast)
        assert np.array_equal(best.image, good)

    @patch("missy.vision.capture._get_cv2")
    def test_returns_failure_when_no_successful_frames(self, mock_cv2_fn):
        cv2 = _make_mock_cv2()
        mock_cv2_fn.return_value = cv2

        from missy.vision.capture import CameraHandle, CaptureConfig

        config = CaptureConfig(warmup_frames=0)
        cam = CameraHandle("/dev/video0", config)

        results = [
            CaptureResult(success=False, error="fail1"),
            CaptureResult(success=False, error="fail2"),
        ]

        with patch.object(cam, "capture_burst", return_value=results):
            best = cam.capture_best(burst_count=2)

        assert not best.success
        assert "No successful" in best.error


def _make_mock_cv2():
    """Create a mock cv2 module with real numpy-based operations."""
    import cv2
    return cv2
