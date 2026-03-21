"""Edge-case tests for the image preprocessing pipeline."""

from __future__ import annotations

import numpy as np
import pytest

from missy.vision.pipeline import ImagePipeline, PipelineConfig


class TestPipelineEdgeCases:
    """Edge cases for ImagePipeline processing."""

    def test_single_pixel_image(self) -> None:
        """Pipeline should handle 1x1 images."""
        pipe = ImagePipeline(PipelineConfig(normalize_exposure=False))
        img = np.array([[[128, 128, 128]]], dtype=np.uint8)
        result = pipe.process(img)
        assert result.shape[0] > 0 and result.shape[1] > 0

    def test_very_large_image_resized(self) -> None:
        """Pipeline should resize images larger than target."""
        pipe = ImagePipeline(PipelineConfig(target_dimension=100))
        img = np.zeros((2000, 3000, 3), dtype=np.uint8)
        result = pipe.process(img)
        assert max(result.shape[:2]) <= 100

    def test_image_already_smaller_than_target(self) -> None:
        """No resizing when image is smaller than target."""
        pipe = ImagePipeline(PipelineConfig(target_dimension=1000))
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        result = pipe.process(img)
        assert result.shape[:2] == (100, 100)

    def test_grayscale_image(self) -> None:
        """Pipeline handles 2D grayscale images in quality assessment."""
        pipe = ImagePipeline()
        img = np.random.randint(0, 256, (100, 100), dtype=np.uint8)
        result = pipe.assess_quality(img)
        assert "brightness" in result
        assert result["saturation"] == 0.0

    def test_bgra_image(self) -> None:
        """Pipeline handles 4-channel BGRA images."""
        pipe = ImagePipeline(PipelineConfig(normalize_exposure=True))
        img = np.random.randint(0, 256, (100, 100, 4), dtype=np.uint8)
        result = pipe.process(img)
        assert result.shape[2] == 4  # alpha preserved

    def test_quality_dark_image(self) -> None:
        pipe = ImagePipeline()
        img = np.full((100, 100, 3), 10, dtype=np.uint8)
        result = pipe.assess_quality(img)
        assert "very dark" in result["issues"]

    def test_quality_overexposed(self) -> None:
        pipe = ImagePipeline()
        img = np.full((100, 100, 3), 240, dtype=np.uint8)
        result = pipe.assess_quality(img)
        assert "overexposed" in result["issues"]

    def test_quality_low_contrast(self) -> None:
        pipe = ImagePipeline()
        img = np.full((100, 100, 3), 128, dtype=np.uint8)
        # Add tiny variation
        img[0, 0] = [127, 127, 127]
        result = pipe.assess_quality(img)
        assert "low contrast" in result["issues"]

    def test_quality_blurry(self) -> None:
        pipe = ImagePipeline()
        # Smooth gradient = very blurry
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        for i in range(100):
            img[i, :] = [int(i * 2.55)] * 3
        result = pipe.assess_quality(img)
        assert "blurry" in result["issues"]

    def test_quality_good_image(self) -> None:
        pipe = ImagePipeline()
        rng = np.random.RandomState(42)
        img = rng.randint(50, 200, (200, 200, 3), dtype=np.uint8)
        result = pipe.assess_quality(img)
        # Random image has good contrast and sharpness
        assert result["quality"] in ("good", "fair")

    def test_none_image_raises(self) -> None:
        pipe = ImagePipeline()
        with pytest.raises(ValueError, match="non-None"):
            pipe.process(None)

    def test_empty_image_raises(self) -> None:
        pipe = ImagePipeline()
        with pytest.raises(ValueError, match="invalid shape"):
            pipe.process(np.array([], dtype=np.uint8).reshape(0, 0))

    def test_resize_negative_max_dim_raises(self) -> None:
        pipe = ImagePipeline()
        with pytest.raises(ValueError, match="positive"):
            pipe.resize(np.zeros((10, 10, 3), dtype=np.uint8), -1)

    def test_denoise(self) -> None:
        pipe = ImagePipeline(
            PipelineConfig(denoise=True, normalize_exposure=False, target_dimension=50)
        )
        rng = np.random.RandomState(7)
        img = rng.randint(50, 200, (50, 50, 3), dtype=np.uint8)
        result = pipe.process(img)
        assert result.shape == img.shape

    def test_sharpen(self) -> None:
        pipe = ImagePipeline(
            PipelineConfig(sharpen=True, normalize_exposure=False, target_dimension=50)
        )
        img = np.full((50, 50, 3), 128, dtype=np.uint8)
        img[20:30, 20:30] = [200, 200, 200]
        result = pipe.process(img)
        assert result.shape == img.shape

    def test_single_channel_3d(self) -> None:
        """Single-channel 3D array (H,W,1) in CLAHE."""
        pipe = ImagePipeline(PipelineConfig(normalize_exposure=True, target_dimension=50))
        img = np.random.randint(0, 256, (50, 50, 1), dtype=np.uint8)
        result = pipe.process(img)
        assert result.ndim == 3


class TestPipelineConfig:
    """Tests for PipelineConfig defaults and overrides."""

    def test_defaults(self) -> None:
        cfg = PipelineConfig()
        assert cfg.max_dimension == 1920
        assert cfg.target_dimension == 1280
        assert cfg.normalize_exposure is True
        assert cfg.denoise is False
        assert cfg.sharpen is False

    def test_overrides(self) -> None:
        cfg = PipelineConfig(
            target_dimension=640,
            denoise=True,
            sharpen=True,
        )
        assert cfg.target_dimension == 640
        assert cfg.denoise is True
        assert cfg.sharpen is True
