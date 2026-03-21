"""Tests for camera image orientation detection and correction."""

from unittest.mock import patch

import numpy as np

from missy.vision.orientation import (
    Orientation,
    auto_correct,
    correct_orientation,
    detect_orientation,
)


class TestDetectOrientation:
    """Tests for detect_orientation()."""

    def test_landscape_image_is_normal(self):
        """A 640x480 landscape image should be detected as normal."""
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        result = detect_orientation(img)
        assert result.detected == Orientation.NORMAL
        assert result.confidence > 0.5

    def test_portrait_image_detected_as_rotated(self):
        """A 480x640 portrait image should be detected as rotated 90°."""
        img = np.zeros((640, 480, 3), dtype=np.uint8)
        result = detect_orientation(img)
        assert result.detected == Orientation.ROTATED_90_CW
        assert result.confidence > 0.0

    def test_square_image_normal_low_confidence(self):
        """A square image should be normal with low confidence."""
        img = np.zeros((500, 500, 3), dtype=np.uint8)
        result = detect_orientation(img)
        assert result.detected == Orientation.NORMAL
        assert result.confidence <= 0.5

    def test_none_image_returns_normal(self):
        result = detect_orientation(None)
        assert result.detected == Orientation.NORMAL
        assert result.confidence == 0.0

    def test_1d_array_returns_normal(self):
        result = detect_orientation(np.array([1, 2, 3]))
        assert result.detected == Orientation.NORMAL
        assert result.confidence == 0.0

    def test_zero_dimension_returns_normal(self):
        img = np.zeros((0, 100, 3), dtype=np.uint8)
        result = detect_orientation(img)
        assert result.detected == Orientation.NORMAL

    def test_widescreen_high_confidence(self):
        """16:9 widescreen should be normal with high confidence."""
        img = np.zeros((1080, 1920, 3), dtype=np.uint8)
        result = detect_orientation(img)
        assert result.detected == Orientation.NORMAL
        assert result.confidence >= 0.8

    def test_very_narrow_portrait_high_confidence(self):
        """Very narrow portrait should be high confidence rotated."""
        img = np.zeros((1000, 200, 3), dtype=np.uint8)
        result = detect_orientation(img)
        assert result.detected == Orientation.ROTATED_90_CW
        assert result.confidence > 0.5

    def test_grayscale_image_works(self):
        """Grayscale 2D images should work."""
        img = np.zeros((480, 640), dtype=np.uint8)
        result = detect_orientation(img)
        assert result.detected == Orientation.NORMAL


class TestCorrectOrientation:
    """Tests for correct_orientation()."""

    @patch("missy.vision.orientation._get_cv2")
    def test_normal_returns_unchanged(self, mock_cv2_fn):
        import cv2

        mock_cv2_fn.return_value = cv2
        img = np.zeros((100, 200, 3), dtype=np.uint8)
        result = correct_orientation(img, Orientation.NORMAL)
        assert np.array_equal(result, img)

    @patch("missy.vision.orientation._get_cv2")
    def test_90cw_rotates_counterclockwise(self, mock_cv2_fn):
        import cv2

        mock_cv2_fn.return_value = cv2
        # Portrait 100x200 → should become landscape 200x100
        img = np.zeros((200, 100, 3), dtype=np.uint8)
        result = correct_orientation(img, Orientation.ROTATED_90_CW)
        assert result.shape[0] == 100
        assert result.shape[1] == 200

    @patch("missy.vision.orientation._get_cv2")
    def test_180_rotates_180(self, mock_cv2_fn):
        import cv2

        mock_cv2_fn.return_value = cv2
        img = np.array([[1, 2], [3, 4]], dtype=np.uint8)
        result = correct_orientation(img, Orientation.ROTATED_180)
        assert result.shape == img.shape
        assert result[0, 0] == 4

    @patch("missy.vision.orientation._get_cv2")
    def test_90ccw_rotates_clockwise(self, mock_cv2_fn):
        import cv2

        mock_cv2_fn.return_value = cv2
        img = np.zeros((200, 100, 3), dtype=np.uint8)
        result = correct_orientation(img, Orientation.ROTATED_90_CCW)
        assert result.shape[0] == 100
        assert result.shape[1] == 200


class TestAutoCorrect:
    """Tests for auto_correct()."""

    @patch("missy.vision.orientation._get_cv2")
    def test_corrects_portrait_image(self, mock_cv2_fn):
        import cv2

        mock_cv2_fn.return_value = cv2
        # Strongly portrait image (should be corrected — high confidence)
        img = np.zeros((1080, 200, 3), dtype=np.uint8)
        corrected, result = auto_correct(img)
        assert result.detected == Orientation.ROTATED_90_CW
        assert result.correction_applied

    @patch("missy.vision.orientation._get_cv2")
    def test_does_not_correct_landscape(self, mock_cv2_fn):
        import cv2

        mock_cv2_fn.return_value = cv2
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        corrected, result = auto_correct(img)
        assert result.detected == Orientation.NORMAL
        assert not result.correction_applied
        assert np.array_equal(corrected, img)

    @patch("missy.vision.orientation._get_cv2")
    def test_low_confidence_does_not_correct(self, mock_cv2_fn):
        import cv2

        mock_cv2_fn.return_value = cv2
        # Square image — low confidence
        img = np.zeros((500, 500, 3), dtype=np.uint8)
        corrected, result = auto_correct(img)
        assert not result.correction_applied
        assert np.array_equal(corrected, img)


class TestOrientationEnum:
    """Tests for Orientation enum values."""

    def test_values(self):
        assert Orientation.NORMAL == 0
        assert Orientation.ROTATED_90_CW == 90
        assert Orientation.ROTATED_180 == 180
        assert Orientation.ROTATED_90_CCW == 270

    def test_all_values_unique(self):
        values = [o.value for o in Orientation]
        assert len(values) == len(set(values))
