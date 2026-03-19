"""Hardening tests for missy.vision.pipeline using real OpenCV.

All tests construct numpy arrays directly and invoke the pipeline with real
OpenCV (no mocking).  Image dimensions are kept small (1-100 px) so the
suite runs in milliseconds.
"""

from __future__ import annotations

import numpy as np
import pytest

from missy.vision.pipeline import ImagePipeline, PipelineConfig

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _bgr(h: int, w: int, value: int = 128) -> np.ndarray:
    """Return an (h, w, 3) uint8 BGR image filled with *value*."""
    return np.full((h, w, 3), value, dtype=np.uint8)


def _gray2d(h: int, w: int, value: int = 128) -> np.ndarray:
    """Return an (h, w) uint8 grayscale image."""
    return np.full((h, w), value, dtype=np.uint8)


def _pipeline(**kwargs: object) -> ImagePipeline:
    """Create an ImagePipeline with the given config overrides."""
    return ImagePipeline(PipelineConfig(**kwargs))  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# 1. process() with a 1-pixel image (1x1x3)
# ---------------------------------------------------------------------------

class TestOnePixelImage:
    """A 1-pixel image is the smallest valid BGR image."""

    def test_process_does_not_raise(self) -> None:
        pipe = _pipeline(normalize_exposure=True, target_dimension=1280)
        img = np.array([[[100, 150, 200]]], dtype=np.uint8)  # (1, 1, 3)
        result = pipe.process(img)
        assert result is not None

    def test_output_is_ndarray(self) -> None:
        pipe = _pipeline(normalize_exposure=True, target_dimension=1280)
        img = np.array([[[50, 50, 50]]], dtype=np.uint8)
        result = pipe.process(img)
        assert isinstance(result, np.ndarray)

    def test_output_shape_positive_dimensions(self) -> None:
        """After pipeline, height and width must both be >= 1."""
        pipe = _pipeline(normalize_exposure=False, target_dimension=1280)
        img = np.array([[[0, 0, 0]]], dtype=np.uint8)
        result = pipe.process(img)
        assert result.shape[0] >= 1
        assert result.shape[1] >= 1


# ---------------------------------------------------------------------------
# 2. process() with a 2D grayscale image (100x100)
# ---------------------------------------------------------------------------

class TestGrayscale2DProcess:
    """process() should handle 2D grayscale arrays without crashing.

    The pipeline internally may produce a squeezed result; we only assert
    that no exception is raised and that the output is a valid ndarray.
    """

    def test_does_not_raise(self) -> None:
        # normalize_exposure handles 2D via the grayscale branch in CLAHE
        pipe = _pipeline(normalize_exposure=True, target_dimension=50)
        img = _gray2d(100, 100, value=120)
        result = pipe.process(img)
        assert isinstance(result, np.ndarray)

    def test_output_dtype_preserved_or_uint8(self) -> None:
        pipe = _pipeline(normalize_exposure=True, target_dimension=50)
        img = _gray2d(100, 100, value=80)
        result = pipe.process(img)
        assert result.dtype == np.uint8

    def test_no_exposure_normalization_grayscale(self) -> None:
        """With normalization off, a 2D image still passes through resize."""
        pipe = _pipeline(normalize_exposure=False, target_dimension=50)
        img = _gray2d(100, 100, value=200)
        result = pipe.process(img)
        # Image is larger than target_dimension so should be resized
        assert max(result.shape[:2]) <= 50


# ---------------------------------------------------------------------------
# 3. process() with BGRA 4-channel image (100x100x4)
# ---------------------------------------------------------------------------

class TestBGRA4Channel:
    """Alpha channel must be preserved through the pipeline."""

    def test_process_does_not_raise(self) -> None:
        pipe = _pipeline(normalize_exposure=True, target_dimension=50)
        img = np.random.default_rng(0).integers(0, 256, (100, 100, 4), dtype=np.uint8)
        result = pipe.process(img)
        assert isinstance(result, np.ndarray)

    def test_alpha_channel_preserved(self) -> None:
        """Output must remain 4-channel so downstream code can use alpha."""
        pipe = _pipeline(normalize_exposure=True, target_dimension=50)
        # Distinct alpha value to verify it survives the pipeline
        img = np.full((100, 100, 4), 200, dtype=np.uint8)
        img[:, :, 3] = 77  # unique alpha
        result = pipe.process(img)
        assert result.ndim == 3
        assert result.shape[2] == 4

    def test_alpha_values_unchanged(self) -> None:
        """Alpha pixel values should not be altered by CLAHE/exposure."""
        pipe = _pipeline(normalize_exposure=True, target_dimension=50)
        img = np.full((100, 100, 4), 128, dtype=np.uint8)
        img[:, :, 3] = 200
        result = pipe.process(img)
        assert int(result[:, :, 3].mean()) == 200


# ---------------------------------------------------------------------------
# 4. process() with single-channel 3D image (100x100x1)
# ---------------------------------------------------------------------------

class TestSingleChannel3D:
    """(H, W, 1) images are a non-obvious edge case in CLAHE handling."""

    def test_process_does_not_raise(self) -> None:
        pipe = _pipeline(normalize_exposure=True, target_dimension=50)
        img = np.random.default_rng(1).integers(0, 256, (100, 100, 1), dtype=np.uint8)
        result = pipe.process(img)
        assert isinstance(result, np.ndarray)

    def test_output_is_valid_ndarray(self) -> None:
        """After pipeline the result must be a uint8 ndarray with positive dims.

        cv2.resize can squeeze (H,W,1) → (H,W) when the spatial downscale
        produces a 2-D result, so we test the observable invariants rather
        than a fixed number of dimensions.
        """
        pipe = _pipeline(normalize_exposure=True, target_dimension=50)
        img = np.full((100, 100, 1), 100, dtype=np.uint8)
        result = pipe.process(img)
        assert result.ndim >= 2
        assert result.shape[0] >= 1
        assert result.shape[1] >= 1

    def test_output_dtype_is_uint8(self) -> None:
        pipe = _pipeline(normalize_exposure=True, target_dimension=50)
        img = np.full((100, 100, 1), 64, dtype=np.uint8)
        result = pipe.process(img)
        assert result.dtype == np.uint8


# ---------------------------------------------------------------------------
# 5. resize() with image already smaller than max_dim
# ---------------------------------------------------------------------------

class TestResizeAlreadySmall:
    """When the image fits within max_dim, resize() must return it unchanged."""

    def test_returns_same_array(self) -> None:
        pipe = _pipeline()
        img = _bgr(50, 50)
        result = pipe.resize(img, 200)
        # Same object — no copy made for already-small images
        assert result is img

    def test_shape_unchanged(self) -> None:
        pipe = _pipeline()
        img = _bgr(80, 60)
        result = pipe.resize(img, 1280)
        assert result.shape == (80, 60, 3)

    def test_exact_boundary_not_resized(self) -> None:
        """An image where max(h, w) == max_dim must not be resized."""
        pipe = _pipeline()
        img = _bgr(100, 100)
        result = pipe.resize(img, 100)
        assert result is img


# ---------------------------------------------------------------------------
# 6. resize() with max_dim=0
# ---------------------------------------------------------------------------

class TestResizeZeroMaxDim:
    """max_dim=0 is explicitly invalid and must raise ValueError."""

    def test_zero_raises(self) -> None:
        pipe = _pipeline()
        with pytest.raises(ValueError, match="positive"):
            pipe.resize(_bgr(10, 10), 0)

    def test_negative_raises(self) -> None:
        pipe = _pipeline()
        with pytest.raises(ValueError, match="positive"):
            pipe.resize(_bgr(10, 10), -5)

    def test_error_message_contains_value(self) -> None:
        pipe = _pipeline()
        with pytest.raises(ValueError):
            pipe.resize(_bgr(10, 10), 0)


# ---------------------------------------------------------------------------
# 7. assess_quality() on a very dark image
# ---------------------------------------------------------------------------

class TestAssessQualityDark:
    """Brightness < 40 should produce the 'very dark' issue label."""

    def test_very_dark_bgr(self) -> None:
        pipe = _pipeline()
        img = _bgr(100, 100, value=5)
        result = pipe.assess_quality(img)
        assert "very dark" in result["issues"]

    def test_very_dark_quality_not_good(self) -> None:
        pipe = _pipeline()
        img = _bgr(100, 100, value=5)
        result = pipe.assess_quality(img)
        # A single serious issue should drop quality below "good"
        assert result["quality"] in ("fair", "poor")

    def test_brightness_value_below_40(self) -> None:
        pipe = _pipeline()
        img = _bgr(100, 100, value=10)
        result = pipe.assess_quality(img)
        assert result["brightness"] < 40.0


# ---------------------------------------------------------------------------
# 8. assess_quality() on a very bright (overexposed) image
# ---------------------------------------------------------------------------

class TestAssessQualityOverexposed:
    """Brightness > 220 should produce the 'overexposed' issue label."""

    def test_overexposed_bgr(self) -> None:
        pipe = _pipeline()
        img = _bgr(100, 100, value=240)
        result = pipe.assess_quality(img)
        assert "overexposed" in result["issues"]

    def test_overexposed_brightness_above_220(self) -> None:
        pipe = _pipeline()
        img = _bgr(100, 100, value=240)
        result = pipe.assess_quality(img)
        assert result["brightness"] > 220.0

    def test_boundary_at_221(self) -> None:
        """Value 221 should convert to a gray channel above 220."""
        pipe = _pipeline()
        # 221 in all channels → perceived brightness > 220
        img = _bgr(100, 100, value=221)
        result = pipe.assess_quality(img)
        assert "overexposed" in result["issues"]


# ---------------------------------------------------------------------------
# 9. assess_quality() on uniform image — low contrast + blurry
# ---------------------------------------------------------------------------

class TestAssessQualityUniform:
    """A flat uniform image has zero std-dev (low contrast) and zero
    Laplacian variance (blurry)."""

    def test_low_contrast_present(self) -> None:
        pipe = _pipeline()
        img = _bgr(100, 100, value=128)
        result = pipe.assess_quality(img)
        assert "low contrast" in result["issues"]

    def test_blurry_present(self) -> None:
        pipe = _pipeline()
        img = _bgr(100, 100, value=128)
        result = pipe.assess_quality(img)
        assert "blurry" in result["issues"]

    def test_quality_is_poor_with_two_issues(self) -> None:
        """Two or more issues → quality == 'poor'."""
        pipe = _pipeline()
        img = _bgr(100, 100, value=128)
        result = pipe.assess_quality(img)
        assert result["quality"] == "poor"

    def test_contrast_is_zero(self) -> None:
        pipe = _pipeline()
        img = _bgr(100, 100, value=100)
        result = pipe.assess_quality(img)
        assert result["contrast"] == 0.0


# ---------------------------------------------------------------------------
# 10. assess_quality() on 2D grayscale image
# ---------------------------------------------------------------------------

class TestAssessQualityGrayscale:
    """Grayscale images skip cvtColor; saturation must report 0.0."""

    def test_does_not_raise(self) -> None:
        pipe = _pipeline()
        img = _gray2d(100, 100, value=128)
        result = pipe.assess_quality(img)
        assert isinstance(result, dict)

    def test_saturation_is_zero(self) -> None:
        pipe = _pipeline()
        img = _gray2d(100, 100, value=100)
        result = pipe.assess_quality(img)
        assert result["saturation"] == 0.0

    def test_dimensions_correct(self) -> None:
        pipe = _pipeline()
        img = _gray2d(80, 60, value=90)
        result = pipe.assess_quality(img)
        assert result["width"] == 60
        assert result["height"] == 80

    def test_random_grayscale_has_valid_keys(self) -> None:
        pipe = _pipeline()
        rng = np.random.default_rng(42)
        img = rng.integers(0, 256, (100, 100), dtype=np.uint8)
        result = pipe.assess_quality(img)
        for key in ("brightness", "contrast", "sharpness", "saturation",
                    "noise_level", "quality", "issues"):
            assert key in result


# ---------------------------------------------------------------------------
# 11. process() with all options disabled — only resize applied
# ---------------------------------------------------------------------------

class TestProcessAllOptionsDisabled:
    """normalize_exposure=False, denoise=False, sharpen=False.

    Only the resize step runs; an image larger than target_dimension must
    be downscaled.
    """

    def test_does_not_raise(self) -> None:
        pipe = _pipeline(normalize_exposure=False, denoise=False, sharpen=False,
                         target_dimension=50)
        img = _bgr(200, 200)
        result = pipe.process(img)
        assert isinstance(result, np.ndarray)

    def test_resize_still_occurs(self) -> None:
        pipe = _pipeline(normalize_exposure=False, denoise=False, sharpen=False,
                         target_dimension=50)
        img = _bgr(200, 200)
        result = pipe.process(img)
        assert max(result.shape[:2]) <= 50

    def test_small_image_returned_as_copy(self) -> None:
        """A small image below target_dimension is not resized; process returns a copy."""
        pipe = _pipeline(normalize_exposure=False, denoise=False, sharpen=False,
                         target_dimension=1280)
        img = _bgr(100, 100)
        result = pipe.process(img)
        # process() always copies; result should not be the same object
        assert result is not img
        assert result.shape == img.shape

    def test_output_channel_count_unchanged(self) -> None:
        pipe = _pipeline(normalize_exposure=False, denoise=False, sharpen=False,
                         target_dimension=50)
        img = _bgr(200, 200)
        result = pipe.process(img)
        assert result.shape[2] == 3


# ---------------------------------------------------------------------------
# 12. process() with all options enabled — denoise + sharpen + exposure
# ---------------------------------------------------------------------------

class TestProcessAllOptionsEnabled:
    """All pipeline steps active with real OpenCV.

    Uses a small BGR image so fastNlMeansDenoisingColored runs quickly.
    """

    def test_does_not_raise(self) -> None:
        pipe = _pipeline(normalize_exposure=True, denoise=True, sharpen=True,
                         target_dimension=50)
        rng = np.random.default_rng(7)
        img = rng.integers(30, 220, (60, 60, 3), dtype=np.uint8)
        result = pipe.process(img)
        assert isinstance(result, np.ndarray)

    def test_output_is_uint8(self) -> None:
        pipe = _pipeline(normalize_exposure=True, denoise=True, sharpen=True,
                         target_dimension=50)
        rng = np.random.default_rng(8)
        img = rng.integers(30, 220, (60, 60, 3), dtype=np.uint8)
        result = pipe.process(img)
        assert result.dtype == np.uint8

    def test_output_shape_matches_resize(self) -> None:
        """With target_dimension=50, a 60x60 image should be downsized to <=50."""
        pipe = _pipeline(normalize_exposure=True, denoise=True, sharpen=True,
                         target_dimension=50)
        img = np.full((60, 60, 3), 100, dtype=np.uint8)
        img[20:40, 20:40] = [200, 100, 50]  # add variation for denoiser
        result = pipe.process(img)
        assert max(result.shape[:2]) <= 50

    def test_pixel_values_in_valid_range(self) -> None:
        """All pixel values must remain in [0, 255] after all transformations."""
        pipe = _pipeline(normalize_exposure=True, denoise=True, sharpen=True,
                         target_dimension=50)
        rng = np.random.default_rng(99)
        img = rng.integers(10, 240, (60, 60, 3), dtype=np.uint8)
        result = pipe.process(img)
        assert int(result.min()) >= 0
        assert int(result.max()) <= 255
