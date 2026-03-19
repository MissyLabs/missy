"""Image normalization and preprocessing pipeline.

Prepares captured images for LLM consumption or local analysis.
Handles resizing, rotation correction, exposure normalization,
and format conversion.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Lazy OpenCV
# ---------------------------------------------------------------------------

_cv2: Any = None


def _get_cv2() -> Any:
    global _cv2
    if _cv2 is None:
        import cv2
        _cv2 = cv2
    return _cv2


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class PipelineConfig:
    """Configuration for the image preprocessing pipeline."""

    max_dimension: int = 1920  # max width or height
    target_dimension: int = 1280  # target for LLM submission
    normalize_exposure: bool = True
    auto_rotate: bool = True
    denoise: bool = False
    sharpen: bool = False
    jpeg_quality: int = 85


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


class ImagePipeline:
    """Preprocesses images for analysis.

    Steps (applied in order):
    1. Resize to target dimensions (preserving aspect ratio)
    2. Optionally normalize exposure via CLAHE
    3. Optionally apply light denoising
    4. Optionally sharpen
    """

    def __init__(self, config: PipelineConfig | None = None) -> None:
        self._config = config or PipelineConfig()

    def process(self, image: np.ndarray) -> np.ndarray:
        """Run the full preprocessing pipeline on an image."""
        result = image.copy()

        # 1. Resize
        result = self.resize(result, self._config.target_dimension)

        # 2. Exposure normalization
        if self._config.normalize_exposure:
            result = self.normalize_exposure(result)

        # 3. Denoise
        if self._config.denoise:
            result = self.denoise(result)

        # 4. Sharpen
        if self._config.sharpen:
            result = self.sharpen(result)

        return result

    def resize(self, image: np.ndarray, max_dim: int) -> np.ndarray:
        """Resize image so largest dimension is at most max_dim."""
        cv2 = _get_cv2()
        h, w = image.shape[:2]

        if max(h, w) <= max_dim:
            return image

        scale = max_dim / max(h, w)
        new_w = int(w * scale)
        new_h = int(h * scale)

        return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)

    def normalize_exposure(self, image: np.ndarray) -> np.ndarray:
        """Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)."""
        cv2 = _get_cv2()

        # Convert to LAB color space
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l_channel, a, b = cv2.split(lab)

        # Apply CLAHE to L channel
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l_enhanced = clahe.apply(l_channel)

        # Merge and convert back
        enhanced = cv2.merge([l_enhanced, a, b])
        return cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)

    def denoise(self, image: np.ndarray) -> np.ndarray:
        """Apply light denoising (fastNlMeansDenoisingColored)."""
        cv2 = _get_cv2()
        return cv2.fastNlMeansDenoisingColored(image, None, 6, 6, 7, 21)

    def sharpen(self, image: np.ndarray) -> np.ndarray:
        """Apply unsharp masking for mild sharpening."""
        cv2 = _get_cv2()
        gaussian = cv2.GaussianBlur(image, (0, 0), 3)
        return cv2.addWeighted(image, 1.5, gaussian, -0.5, 0)

    def assess_quality(self, image: np.ndarray) -> dict[str, Any]:
        """Assess basic image quality metrics."""
        cv2 = _get_cv2()
        h, w = image.shape[:2]

        # Brightness
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        brightness = float(np.mean(gray))

        # Contrast (std dev of grayscale)
        contrast = float(np.std(gray))

        # Blur detection (Laplacian variance)
        laplacian_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())

        # Classify quality
        issues: list[str] = []
        if brightness < 40:
            issues.append("very dark")
        elif brightness < 80:
            issues.append("low light")
        elif brightness > 220:
            issues.append("overexposed")

        if contrast < 20:
            issues.append("low contrast")

        if laplacian_var < 50:
            issues.append("blurry")

        quality = "good"
        if len(issues) >= 2:
            quality = "poor"
        elif issues:
            quality = "fair"

        return {
            "width": w,
            "height": h,
            "brightness": round(brightness, 1),
            "contrast": round(contrast, 1),
            "sharpness": round(laplacian_var, 1),
            "quality": quality,
            "issues": issues,
        }
