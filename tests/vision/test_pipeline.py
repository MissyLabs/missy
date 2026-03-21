"""Tests for missy.vision.pipeline — image preprocessing."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np

from missy.vision.pipeline import ImagePipeline, PipelineConfig


class TestPipelineConfig:
    def test_defaults(self):
        cfg = PipelineConfig()
        assert cfg.max_dimension == 1920
        assert cfg.target_dimension == 1280
        assert cfg.normalize_exposure is True


class TestImagePipeline:
    def _make_pipeline(self, **kwargs):
        cfg = PipelineConfig(**kwargs)
        return ImagePipeline(cfg)

    @patch("missy.vision.pipeline._get_cv2")
    def test_resize_small_image_unchanged(self, mock_cv2_fn):
        mock_cv2 = MagicMock()
        mock_cv2_fn.return_value = mock_cv2

        pipeline = self._make_pipeline(normalize_exposure=False, denoise=False, sharpen=False)
        img = np.zeros((100, 100, 3), dtype=np.uint8)

        # For small image, resize should return as-is
        result = pipeline.resize(img, 1280)
        assert result.shape == (100, 100, 3)

    @patch("missy.vision.pipeline._get_cv2")
    def test_resize_large_image(self, mock_cv2_fn):
        mock_cv2 = MagicMock()
        mock_cv2_fn.return_value = mock_cv2

        resized = np.zeros((640, 640, 3), dtype=np.uint8)
        mock_cv2.resize.return_value = resized
        mock_cv2.INTER_AREA = 3

        pipeline = self._make_pipeline()
        img = np.zeros((2000, 2000, 3), dtype=np.uint8)
        result = pipeline.resize(img, 640)

        mock_cv2.resize.assert_called_once()
        assert result.shape == (640, 640, 3)

    @patch("missy.vision.pipeline._get_cv2")
    def test_assess_quality_bright(self, mock_cv2_fn):
        mock_cv2 = MagicMock()
        mock_cv2_fn.return_value = mock_cv2

        # Create a well-lit image
        gray = np.full((100, 100), 128, dtype=np.uint8)
        mock_cv2.cvtColor.return_value = gray

        np.full((100, 100), 100.0)
        mock_lap = MagicMock()
        mock_lap.var.return_value = 100.0
        mock_cv2.Laplacian.return_value = mock_lap
        mock_cv2.CV_64F = 6

        pipeline = self._make_pipeline()
        img = np.full((100, 100, 3), 128, dtype=np.uint8)
        quality = pipeline.assess_quality(img)

        assert quality["width"] == 100
        assert quality["height"] == 100
        assert "brightness" in quality
        assert "quality" in quality

    @patch("missy.vision.pipeline._get_cv2")
    def test_assess_quality_dark(self, mock_cv2_fn):
        mock_cv2 = MagicMock()
        mock_cv2_fn.return_value = mock_cv2

        gray = np.full((100, 100), 10, dtype=np.uint8)
        mock_cv2.cvtColor.return_value = gray

        mock_lap = MagicMock()
        mock_lap.var.return_value = 100.0
        mock_cv2.Laplacian.return_value = mock_lap
        mock_cv2.CV_64F = 6

        pipeline = self._make_pipeline()
        img = np.full((100, 100, 3), 10, dtype=np.uint8)
        quality = pipeline.assess_quality(img)

        assert "very dark" in quality["issues"]

    @patch("missy.vision.pipeline._get_cv2")
    def test_assess_quality_blurry(self, mock_cv2_fn):
        mock_cv2 = MagicMock()
        mock_cv2_fn.return_value = mock_cv2

        gray = np.full((100, 100), 128, dtype=np.uint8)
        mock_cv2.cvtColor.return_value = gray

        mock_lap = MagicMock()
        mock_lap.var.return_value = 10.0  # low = blurry
        mock_cv2.Laplacian.return_value = mock_lap
        mock_cv2.CV_64F = 6

        pipeline = self._make_pipeline()
        img = np.full((100, 100, 3), 128, dtype=np.uint8)
        quality = pipeline.assess_quality(img)

        assert "blurry" in quality["issues"]
