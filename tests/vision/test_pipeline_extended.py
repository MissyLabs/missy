"""Extended tests for missy.vision.pipeline — preprocessing operations."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from missy.vision.pipeline import ImagePipeline, PipelineConfig


class TestSingleChannelImages:
    """Tests for single-channel (H,W,1) image handling — previously a crash bug."""

    @patch("missy.vision.pipeline._get_cv2")
    def test_assess_quality_single_channel_3d(self, mock_cv2_fn):
        """Single-channel 3D image (H,W,1) must not crash assess_quality."""
        mock_cv2 = MagicMock()
        mock_cv2_fn.return_value = mock_cv2
        mock_lap = MagicMock()
        mock_lap.var.return_value = 100.0
        mock_cv2.Laplacian.return_value = mock_lap
        mock_cv2.CV_64F = 6

        pipeline = ImagePipeline()
        img = np.full((100, 100, 1), 128, dtype=np.uint8)
        quality = pipeline.assess_quality(img)

        assert quality["width"] == 100
        assert quality["height"] == 100
        # Key assertion: single-channel didn't crash, and cvtColor was NOT called
        mock_cv2.cvtColor.assert_not_called()

    @patch("missy.vision.pipeline._get_cv2")
    def test_normalize_exposure_single_channel_3d(self, mock_cv2_fn):
        """Single-channel 3D image should be processed via CLAHE directly."""
        mock_cv2 = MagicMock()
        mock_cv2_fn.return_value = mock_cv2

        enhanced = np.full((100, 100), 140, dtype=np.uint8)
        mock_clahe = MagicMock()
        mock_clahe.apply.return_value = enhanced
        mock_cv2.createCLAHE.return_value = mock_clahe

        pipeline = ImagePipeline()
        img = np.full((100, 100, 1), 128, dtype=np.uint8)
        result = pipeline.normalize_exposure(img)

        # Should return (H,W,1) shape
        assert result.ndim == 3
        assert result.shape[2] == 1
        mock_clahe.apply.assert_called_once()
        # Should NOT call cvtColor for single-channel
        mock_cv2.cvtColor.assert_not_called()

    def test_assess_quality_grayscale_2d(self):
        """Pure 2D grayscale images should work without OpenCV cvtColor."""
        with patch("missy.vision.pipeline._get_cv2") as mock_cv2_fn:
            mock_cv2 = MagicMock()
            mock_cv2_fn.return_value = mock_cv2
            mock_lap = MagicMock()
            mock_lap.var.return_value = 80.0
            mock_cv2.Laplacian.return_value = mock_lap
            mock_cv2.CV_64F = 6

            pipeline = ImagePipeline()
            img = np.full((50, 50), 100, dtype=np.uint8)
            quality = pipeline.assess_quality(img)

            assert quality["width"] == 50
            assert quality["height"] == 50
            mock_cv2.cvtColor.assert_not_called()

    @patch("missy.vision.pipeline._get_cv2")
    def test_assess_quality_bgra(self, mock_cv2_fn):
        """4-channel BGRA images should be converted via BGR slice."""
        mock_cv2 = MagicMock()
        mock_cv2_fn.return_value = mock_cv2

        gray = np.full((50, 50), 128, dtype=np.uint8)
        mock_cv2.cvtColor.return_value = gray
        mock_lap = MagicMock()
        mock_lap.var.return_value = 100.0
        mock_cv2.Laplacian.return_value = mock_lap
        mock_cv2.CV_64F = 6

        pipeline = ImagePipeline()
        img = np.full((50, 50, 4), 128, dtype=np.uint8)
        quality = pipeline.assess_quality(img)

        assert quality["width"] == 50
        # Should call cvtColor with BGR slice (first 3 channels)
        mock_cv2.cvtColor.assert_called_once()


class TestNormalizeExposure:
    """Tests for CLAHE exposure normalization."""

    @patch("missy.vision.pipeline._get_cv2")
    def test_normalize_grayscale_2d(self, mock_cv2_fn):
        mock_cv2 = MagicMock()
        mock_cv2_fn.return_value = mock_cv2
        enhanced = np.full((50, 50), 140, dtype=np.uint8)
        mock_clahe = MagicMock()
        mock_clahe.apply.return_value = enhanced
        mock_cv2.createCLAHE.return_value = mock_clahe

        pipeline = ImagePipeline()
        img = np.full((50, 50), 80, dtype=np.uint8)
        result = pipeline.normalize_exposure(img)

        assert result.shape == (50, 50)
        mock_clahe.apply.assert_called_once()

    @patch("missy.vision.pipeline._get_cv2")
    def test_normalize_bgr(self, mock_cv2_fn):
        mock_cv2 = MagicMock()
        mock_cv2_fn.return_value = mock_cv2

        lab = np.zeros((50, 50, 3), dtype=np.uint8)
        mock_cv2.cvtColor.return_value = lab
        mock_cv2.split.return_value = (
            np.zeros((50, 50), dtype=np.uint8),
            np.zeros((50, 50), dtype=np.uint8),
            np.zeros((50, 50), dtype=np.uint8),
        )
        enhanced_l = np.full((50, 50), 140, dtype=np.uint8)
        mock_clahe = MagicMock()
        mock_clahe.apply.return_value = enhanced_l
        mock_cv2.createCLAHE.return_value = mock_clahe
        mock_cv2.merge.return_value = lab
        mock_cv2.COLOR_BGR2LAB = 44
        mock_cv2.COLOR_LAB2BGR = 56

        pipeline = ImagePipeline()
        img = np.full((50, 50, 3), 80, dtype=np.uint8)
        pipeline.normalize_exposure(img)

        # Should convert to LAB and back
        assert mock_cv2.cvtColor.call_count == 2

    @patch("missy.vision.pipeline._get_cv2")
    def test_normalize_bgra_preserves_alpha(self, mock_cv2_fn):
        mock_cv2 = MagicMock()
        mock_cv2_fn.return_value = mock_cv2

        lab = np.zeros((50, 50, 3), dtype=np.uint8)
        mock_cv2.cvtColor.return_value = lab
        mock_cv2.split.return_value = (
            np.zeros((50, 50), dtype=np.uint8),
            np.zeros((50, 50), dtype=np.uint8),
            np.zeros((50, 50), dtype=np.uint8),
        )
        enhanced_l = np.full((50, 50), 140, dtype=np.uint8)
        mock_clahe = MagicMock()
        mock_clahe.apply.return_value = enhanced_l
        mock_cv2.createCLAHE.return_value = mock_clahe
        merged_bgr = np.zeros((50, 50, 3), dtype=np.uint8)
        # merge is called twice: once for LAB, once for BGRA restoration
        mock_cv2.merge.return_value = merged_bgr
        mock_cv2.COLOR_BGR2LAB = 44
        mock_cv2.COLOR_LAB2BGR = 56

        pipeline = ImagePipeline()
        img = np.full((50, 50, 4), 128, dtype=np.uint8)
        pipeline.normalize_exposure(img)

        # merge should be called (at least for LAB reconstruction and alpha restoration)
        assert mock_cv2.merge.call_count >= 2


class TestDenoise:
    @patch("missy.vision.pipeline._get_cv2")
    def test_denoise_calls_fastNl(self, mock_cv2_fn):
        mock_cv2 = MagicMock()
        mock_cv2_fn.return_value = mock_cv2
        denoised = np.zeros((50, 50, 3), dtype=np.uint8)
        mock_cv2.fastNlMeansDenoisingColored.return_value = denoised

        pipeline = ImagePipeline()
        img = np.zeros((50, 50, 3), dtype=np.uint8)
        pipeline.denoise(img)

        mock_cv2.fastNlMeansDenoisingColored.assert_called_once()

    @patch("missy.vision.pipeline._get_cv2")
    def test_denoise_parameters(self, mock_cv2_fn):
        mock_cv2 = MagicMock()
        mock_cv2_fn.return_value = mock_cv2
        mock_cv2.fastNlMeansDenoisingColored.return_value = np.zeros((10, 10, 3), dtype=np.uint8)

        pipeline = ImagePipeline()
        img = np.zeros((10, 10, 3), dtype=np.uint8)
        pipeline.denoise(img)

        args = mock_cv2.fastNlMeansDenoisingColored.call_args
        assert args[0][0] is img
        assert args[0][1] is None  # dst
        assert args[0][2] == 6  # h
        assert args[0][3] == 6  # hForColorComponents


class TestSharpen:
    @patch("missy.vision.pipeline._get_cv2")
    def test_sharpen_uses_unsharp_mask(self, mock_cv2_fn):
        mock_cv2 = MagicMock()
        mock_cv2_fn.return_value = mock_cv2
        blurred = np.zeros((50, 50, 3), dtype=np.uint8)
        mock_cv2.GaussianBlur.return_value = blurred
        sharpened = np.zeros((50, 50, 3), dtype=np.uint8)
        mock_cv2.addWeighted.return_value = sharpened

        pipeline = ImagePipeline()
        img = np.zeros((50, 50, 3), dtype=np.uint8)
        pipeline.sharpen(img)

        mock_cv2.GaussianBlur.assert_called_once()
        mock_cv2.addWeighted.assert_called_once()
        # Verify unsharp mask weights: 1.5 original - 0.5 blurred
        args = mock_cv2.addWeighted.call_args[0]
        assert args[1] == 1.5
        assert args[3] == -0.5


class TestFullProcess:
    @patch("missy.vision.pipeline._get_cv2")
    def test_process_all_steps_enabled(self, mock_cv2_fn):
        """Full pipeline with all steps enabled."""
        mock_cv2 = MagicMock()
        mock_cv2_fn.return_value = mock_cv2

        # Setup all mocks for the full pipeline
        mock_cv2.resize.return_value = np.zeros((640, 640, 3), dtype=np.uint8)
        mock_cv2.INTER_AREA = 3

        lab = np.zeros((640, 640, 3), dtype=np.uint8)
        mock_cv2.cvtColor.return_value = lab
        mock_cv2.split.return_value = (
            np.zeros((640, 640), dtype=np.uint8),
            np.zeros((640, 640), dtype=np.uint8),
            np.zeros((640, 640), dtype=np.uint8),
        )
        mock_clahe = MagicMock()
        mock_clahe.apply.return_value = np.zeros((640, 640), dtype=np.uint8)
        mock_cv2.createCLAHE.return_value = mock_clahe
        mock_cv2.merge.return_value = np.zeros((640, 640, 3), dtype=np.uint8)
        mock_cv2.COLOR_BGR2LAB = 44
        mock_cv2.COLOR_LAB2BGR = 56
        mock_cv2.fastNlMeansDenoisingColored.return_value = np.zeros((640, 640, 3), dtype=np.uint8)
        mock_cv2.GaussianBlur.return_value = np.zeros((640, 640, 3), dtype=np.uint8)
        mock_cv2.addWeighted.return_value = np.zeros((640, 640, 3), dtype=np.uint8)

        cfg = PipelineConfig(
            target_dimension=640,
            normalize_exposure=True,
            denoise=True,
            sharpen=True,
        )
        pipeline = ImagePipeline(cfg)
        img = np.zeros((2000, 2000, 3), dtype=np.uint8)
        pipeline.process(img)

        # All steps should have been called
        mock_cv2.resize.assert_called_once()
        mock_cv2.createCLAHE.assert_called_once()
        mock_cv2.fastNlMeansDenoisingColored.assert_called_once()
        mock_cv2.GaussianBlur.assert_called_once()

    @patch("missy.vision.pipeline._get_cv2")
    def test_process_no_steps(self, mock_cv2_fn):
        """Pipeline with all optional steps disabled."""
        mock_cv2 = MagicMock()
        mock_cv2_fn.return_value = mock_cv2

        cfg = PipelineConfig(
            normalize_exposure=False,
            denoise=False,
            sharpen=False,
        )
        pipeline = ImagePipeline(cfg)
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        pipeline.process(img)

        # Only resize should be attempted (but image is small enough to skip)
        mock_cv2.createCLAHE.assert_not_called()
        mock_cv2.fastNlMeansDenoisingColored.assert_not_called()
        mock_cv2.GaussianBlur.assert_not_called()

    def test_process_rejects_none(self):
        pipeline = ImagePipeline()
        with pytest.raises(ValueError, match="non-None"):
            pipeline.process(None)

    def test_process_rejects_empty(self):
        pipeline = ImagePipeline()
        with pytest.raises(ValueError, match="invalid shape"):
            pipeline.process(np.array([]))

    def test_process_rejects_1d(self):
        pipeline = ImagePipeline()
        with pytest.raises(ValueError, match="invalid shape"):
            pipeline.process(np.zeros((10,), dtype=np.uint8))

    def test_resize_rejects_zero_dim(self):
        pipeline = ImagePipeline()
        with pytest.raises(ValueError, match="positive"):
            pipeline.resize(np.zeros((10, 10, 3), dtype=np.uint8), 0)

    def test_resize_rejects_negative(self):
        pipeline = ImagePipeline()
        with pytest.raises(ValueError, match="positive"):
            pipeline.resize(np.zeros((10, 10, 3), dtype=np.uint8), -100)


class TestQualityClassification:
    """Tests for quality classification boundaries."""

    @patch("missy.vision.pipeline._get_cv2")
    def test_overexposed(self, mock_cv2_fn):
        mock_cv2 = MagicMock()
        mock_cv2_fn.return_value = mock_cv2
        mock_lap = MagicMock()
        mock_lap.var.return_value = 100.0
        mock_cv2.Laplacian.return_value = mock_lap
        mock_cv2.CV_64F = 6

        pipeline = ImagePipeline()
        # Bright image
        img = np.full((50, 50), 230, dtype=np.uint8)
        quality = pipeline.assess_quality(img)
        assert "overexposed" in quality["issues"]

    @patch("missy.vision.pipeline._get_cv2")
    def test_low_contrast(self, mock_cv2_fn):
        mock_cv2 = MagicMock()
        mock_cv2_fn.return_value = mock_cv2
        mock_lap = MagicMock()
        mock_lap.var.return_value = 100.0
        mock_cv2.Laplacian.return_value = mock_lap
        mock_cv2.CV_64F = 6

        pipeline = ImagePipeline()
        # Uniform image = low contrast (std dev ~0)
        img = np.full((50, 50), 128, dtype=np.uint8)
        quality = pipeline.assess_quality(img)
        assert "low contrast" in quality["issues"]

    @patch("missy.vision.pipeline._get_cv2")
    def test_poor_quality_multiple_issues(self, mock_cv2_fn):
        mock_cv2 = MagicMock()
        mock_cv2_fn.return_value = mock_cv2
        mock_lap = MagicMock()
        mock_lap.var.return_value = 5.0  # blurry
        mock_cv2.Laplacian.return_value = mock_lap
        mock_cv2.CV_64F = 6

        pipeline = ImagePipeline()
        # Dark + uniform = dark + low contrast + blurry
        img = np.full((50, 50), 10, dtype=np.uint8)
        quality = pipeline.assess_quality(img)
        assert quality["quality"] == "poor"
        assert len(quality["issues"]) >= 2
