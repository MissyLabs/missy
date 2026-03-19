"""Session 14: Edge case tests for orientation detection and provider formatting.

Covers:
- Orientation: aspect ratio boundaries, zero dimensions, None/empty images,
  EXIF parsing, auto-correct logic, correct_orientation for all angles
- Provider format: all providers, validation, unknown provider fallback,
  build_vision_message, empty inputs
"""

from __future__ import annotations

import struct
from unittest.mock import patch

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Orientation detection tests
# ---------------------------------------------------------------------------


class TestOrientationDetection:
    """Edge cases in orientation detection."""

    def test_landscape_image(self):
        from missy.vision.orientation import Orientation, detect_orientation

        img = np.zeros((480, 640, 3), dtype=np.uint8)
        result = detect_orientation(img)
        assert result.detected == Orientation.NORMAL
        assert result.confidence > 0.5

    def test_portrait_image(self):
        from missy.vision.orientation import Orientation, detect_orientation

        img = np.zeros((640, 320, 3), dtype=np.uint8)
        result = detect_orientation(img)
        assert result.detected == Orientation.ROTATED_90_CW
        assert result.confidence > 0.0
        assert result.method == "aspect_ratio"

    def test_square_image(self):
        from missy.vision.orientation import Orientation, detect_orientation

        img = np.zeros((500, 500, 3), dtype=np.uint8)
        result = detect_orientation(img)
        assert result.detected == Orientation.NORMAL
        assert result.confidence == 0.3  # Low confidence for square

    def test_nearly_square_image(self):
        from missy.vision.orientation import Orientation, detect_orientation

        img = np.zeros((500, 550, 3), dtype=np.uint8)  # aspect ~1.1
        result = detect_orientation(img)
        assert result.detected == Orientation.NORMAL
        assert result.confidence == 0.3

    def test_none_image(self):
        from missy.vision.orientation import Orientation, detect_orientation

        result = detect_orientation(None)
        assert result.detected == Orientation.NORMAL
        assert result.confidence == 0.0
        assert result.method == "invalid_input"

    def test_1d_array(self):
        from missy.vision.orientation import Orientation, detect_orientation

        img = np.zeros((100,), dtype=np.uint8)
        result = detect_orientation(img)
        assert result.detected == Orientation.NORMAL
        assert result.confidence == 0.0

    def test_zero_height(self):
        from missy.vision.orientation import Orientation, detect_orientation

        img = np.zeros((0, 640, 3), dtype=np.uint8)
        result = detect_orientation(img)
        assert result.detected == Orientation.NORMAL
        assert result.confidence == 0.0
        assert result.method == "zero_dimension"

    def test_zero_width(self):
        from missy.vision.orientation import Orientation, detect_orientation

        img = np.zeros((480, 0, 3), dtype=np.uint8)
        result = detect_orientation(img)
        assert result.detected == Orientation.NORMAL
        assert result.confidence == 0.0

    def test_very_wide_image(self):
        from missy.vision.orientation import Orientation, detect_orientation

        img = np.zeros((100, 1000, 3), dtype=np.uint8)  # aspect=10
        result = detect_orientation(img)
        assert result.detected == Orientation.NORMAL
        assert result.confidence > 0.8

    def test_very_tall_image(self):
        from missy.vision.orientation import Orientation, detect_orientation

        img = np.zeros((1000, 100, 3), dtype=np.uint8)  # aspect=0.1
        result = detect_orientation(img)
        assert result.detected == Orientation.ROTATED_90_CW
        assert result.confidence > 0.5

    def test_grayscale_image(self):
        from missy.vision.orientation import Orientation, detect_orientation

        img = np.zeros((480, 640), dtype=np.uint8)
        result = detect_orientation(img)
        assert result.detected == Orientation.NORMAL

    def test_16x9_standard(self):
        from missy.vision.orientation import Orientation, detect_orientation

        img = np.zeros((1080, 1920, 3), dtype=np.uint8)
        result = detect_orientation(img)
        assert result.detected == Orientation.NORMAL
        assert result.method == "aspect_ratio"

    def test_portrait_confidence_capped(self):
        """Portrait confidence should not exceed 0.9."""
        from missy.vision.orientation import detect_orientation

        img = np.zeros((10000, 10, 3), dtype=np.uint8)  # Very narrow
        result = detect_orientation(img)
        assert result.confidence <= 0.9

    def test_landscape_confidence_capped(self):
        """Landscape confidence should not exceed 0.95."""
        from missy.vision.orientation import detect_orientation

        img = np.zeros((10, 10000, 3), dtype=np.uint8)  # Very wide
        result = detect_orientation(img)
        assert result.confidence <= 0.95


# ---------------------------------------------------------------------------
# Orientation correction tests
# ---------------------------------------------------------------------------


class TestOrientationCorrection:
    """Tests for correct_orientation."""

    def test_correct_normal(self):
        from missy.vision.orientation import Orientation, correct_orientation

        img = np.zeros((480, 640, 3), dtype=np.uint8)
        result = correct_orientation(img, Orientation.NORMAL)
        assert result.shape == (480, 640, 3)
        assert result is img  # Same object

    def test_correct_90_cw(self):
        from missy.vision.orientation import Orientation, correct_orientation

        img = np.zeros((480, 640, 3), dtype=np.uint8)
        result = correct_orientation(img, Orientation.ROTATED_90_CW)
        # Correcting CW rotation → rotate CCW → transpose
        assert result.shape == (640, 480, 3)

    def test_correct_180(self):
        from missy.vision.orientation import Orientation, correct_orientation

        img = np.zeros((480, 640, 3), dtype=np.uint8)
        result = correct_orientation(img, Orientation.ROTATED_180)
        assert result.shape == (480, 640, 3)

    def test_correct_90_ccw(self):
        from missy.vision.orientation import Orientation, correct_orientation

        img = np.zeros((480, 640, 3), dtype=np.uint8)
        result = correct_orientation(img, Orientation.ROTATED_90_CCW)
        assert result.shape == (640, 480, 3)


# ---------------------------------------------------------------------------
# Auto-correct tests
# ---------------------------------------------------------------------------


class TestAutoCorrect:
    """Tests for auto_correct."""

    def test_auto_correct_normal_image(self):
        from missy.vision.orientation import auto_correct

        img = np.zeros((480, 640, 3), dtype=np.uint8)
        corrected, result = auto_correct(img)
        assert corrected.shape == (480, 640, 3)
        assert result.correction_applied is False

    def test_auto_correct_portrait_image(self):
        from missy.vision.orientation import auto_correct

        img = np.zeros((640, 320, 3), dtype=np.uint8)
        corrected, result = auto_correct(img)
        if result.confidence >= 0.5:
            assert result.correction_applied is True
            assert corrected.shape == (320, 640, 3)

    def test_auto_correct_low_confidence_no_correction(self):
        """Nearly square images should not be auto-corrected (low confidence)."""
        from missy.vision.orientation import auto_correct

        img = np.zeros((500, 500, 3), dtype=np.uint8)
        corrected, result = auto_correct(img)
        assert result.correction_applied is False
        assert corrected is img


# ---------------------------------------------------------------------------
# EXIF parsing tests
# ---------------------------------------------------------------------------


class TestExifParsing:
    """Tests for EXIF orientation parsing."""

    def test_parse_exif_too_short(self):
        from missy.vision.orientation import _parse_exif_orientation

        assert _parse_exif_orientation(b"short") is None

    def test_parse_exif_not_jpeg(self):
        from missy.vision.orientation import _parse_exif_orientation

        data = b"\x89PNG" + b"\x00" * 100
        assert _parse_exif_orientation(data) is None

    def test_parse_exif_jpeg_no_app1(self):
        from missy.vision.orientation import _parse_exif_orientation

        # JPEG SOI + random non-EXIF markers
        data = b"\xff\xd8\xff\xe0" + b"\x00\x10" + b"\x00" * 100
        assert _parse_exif_orientation(data) is None

    def test_parse_exif_empty_bytes(self):
        from missy.vision.orientation import _parse_exif_orientation

        assert _parse_exif_orientation(b"") is None

    def test_find_orientation_in_exif_too_short(self):
        from missy.vision.orientation import _find_orientation_in_exif

        assert _find_orientation_in_exif(b"short") is None

    def test_find_orientation_in_exif_wrong_header(self):
        from missy.vision.orientation import _find_orientation_in_exif

        data = b"NotExif\x00\x00" + b"\x00" * 100
        assert _find_orientation_in_exif(data) is None

    def test_find_orientation_in_exif_little_endian(self):
        """Test EXIF parsing with little-endian byte order (II)."""
        from missy.vision.orientation import _find_orientation_in_exif

        # Build minimal EXIF IFD with orientation tag
        exif_header = b"Exif\x00\x00"
        tiff_header = b"II"  # Little endian
        tiff_magic = struct.pack("<H", 42)
        ifd_offset = struct.pack("<I", 8)  # IFD starts right after TIFF header
        num_entries = struct.pack("<H", 1)
        # Orientation tag: tag=0x0112, type=3 (SHORT), count=1, value=6 (90 CW)
        orientation_entry = struct.pack("<HHI", 0x0112, 3, 1) + struct.pack("<HH", 6, 0)

        data = exif_header + tiff_header + tiff_magic + ifd_offset + num_entries + orientation_entry
        result = _find_orientation_in_exif(data)
        assert result == 6

    def test_find_orientation_in_exif_big_endian(self):
        """Test EXIF parsing with big-endian byte order (MM)."""
        from missy.vision.orientation import _find_orientation_in_exif

        exif_header = b"Exif\x00\x00"
        tiff_header = b"MM"  # Big endian
        tiff_magic = struct.pack(">H", 42)
        ifd_offset = struct.pack(">I", 8)
        num_entries = struct.pack(">H", 1)
        orientation_entry = struct.pack(">HHI", 0x0112, 3, 1) + struct.pack(">HH", 1, 0)

        data = exif_header + tiff_header + tiff_magic + ifd_offset + num_entries + orientation_entry
        result = _find_orientation_in_exif(data)
        assert result == 1  # Normal orientation


# ---------------------------------------------------------------------------
# Provider format tests
# ---------------------------------------------------------------------------


class TestProviderFormat:
    """Tests for provider-specific image formatting."""

    def test_anthropic_format(self):
        from missy.vision.provider_format import format_image_for_anthropic

        result = format_image_for_anthropic("abc123", "image/png")
        assert result["type"] == "image"
        assert result["source"]["type"] == "base64"
        assert result["source"]["media_type"] == "image/png"
        assert result["source"]["data"] == "abc123"

    def test_openai_format(self):
        from missy.vision.provider_format import format_image_for_openai

        result = format_image_for_openai("abc123", "image/jpeg")
        assert result["type"] == "image_url"
        assert "data:image/jpeg;base64,abc123" in result["image_url"]["url"]
        assert result["image_url"]["detail"] == "auto"

    def test_openai_format_custom_detail(self):
        from missy.vision.provider_format import format_image_for_openai

        result = format_image_for_openai("abc", "image/jpeg", detail="high")
        assert result["image_url"]["detail"] == "high"

    def test_format_for_provider_anthropic(self):
        from missy.vision.provider_format import format_image_for_provider

        result = format_image_for_provider("anthropic", "abc", "image/jpeg")
        assert result["type"] == "image"

    def test_format_for_provider_openai(self):
        from missy.vision.provider_format import format_image_for_provider

        result = format_image_for_provider("openai", "abc", "image/jpeg")
        assert result["type"] == "image_url"

    def test_format_for_provider_gpt(self):
        from missy.vision.provider_format import format_image_for_provider

        result = format_image_for_provider("gpt", "abc", "image/jpeg")
        assert result["type"] == "image_url"

    def test_format_for_provider_ollama(self):
        from missy.vision.provider_format import format_image_for_provider

        result = format_image_for_provider("ollama", "abc", "image/jpeg")
        assert result["type"] == "image_url"

    def test_format_for_unknown_provider_defaults_to_anthropic(self):
        from missy.vision.provider_format import format_image_for_provider

        result = format_image_for_provider("unknown_provider", "abc", "image/jpeg")
        assert result["type"] == "image"  # Anthropic format

    def test_format_for_provider_case_insensitive(self):
        from missy.vision.provider_format import format_image_for_provider

        result = format_image_for_provider("ANTHROPIC", "abc", "image/jpeg")
        assert result["type"] == "image"

    def test_format_for_provider_with_whitespace(self):
        from missy.vision.provider_format import format_image_for_provider

        result = format_image_for_provider("  anthropic  ", "abc", "image/jpeg")
        assert result["type"] == "image"

    def test_format_for_provider_empty_name_raises(self):
        from missy.vision.provider_format import format_image_for_provider

        with pytest.raises(ValueError, match="provider_name"):
            format_image_for_provider("", "abc", "image/jpeg")

    def test_format_for_provider_empty_base64_raises(self):
        from missy.vision.provider_format import format_image_for_provider

        with pytest.raises(ValueError, match="image_base64"):
            format_image_for_provider("anthropic", "", "image/jpeg")

    def test_format_for_provider_empty_media_type_raises(self):
        from missy.vision.provider_format import format_image_for_provider

        with pytest.raises(ValueError, match="media_type"):
            format_image_for_provider("anthropic", "abc", "")

    def test_format_for_provider_whitespace_only_name_raises(self):
        from missy.vision.provider_format import format_image_for_provider

        with pytest.raises(ValueError, match="provider_name"):
            format_image_for_provider("   ", "abc", "image/jpeg")

    def test_build_vision_message_anthropic(self):
        from missy.vision.provider_format import build_vision_message

        msg = build_vision_message("anthropic", "abc", "Describe this image")
        assert msg["role"] == "user"
        assert len(msg["content"]) == 2
        assert msg["content"][0]["type"] == "image"
        assert msg["content"][1]["type"] == "text"
        assert msg["content"][1]["text"] == "Describe this image"

    def test_build_vision_message_openai(self):
        from missy.vision.provider_format import build_vision_message

        msg = build_vision_message("openai", "abc", "What is this?")
        assert msg["content"][0]["type"] == "image_url"

    def test_build_vision_message_empty_prompt_raises(self):
        from missy.vision.provider_format import build_vision_message

        with pytest.raises(ValueError, match="prompt"):
            build_vision_message("anthropic", "abc", "")

    def test_build_vision_message_custom_media_type(self):
        from missy.vision.provider_format import build_vision_message

        msg = build_vision_message("anthropic", "abc", "test", media_type="image/png")
        assert msg["content"][0]["source"]["media_type"] == "image/png"


# ---------------------------------------------------------------------------
# OrientationResult dataclass tests
# ---------------------------------------------------------------------------


class TestOrientationResult:
    """Tests for OrientationResult dataclass."""

    def test_defaults(self):
        from missy.vision.orientation import Orientation, OrientationResult

        result = OrientationResult(detected=Orientation.NORMAL, confidence=0.5)
        assert result.correction_applied is False
        assert result.method == ""

    def test_all_orientation_values(self):
        from missy.vision.orientation import Orientation

        assert Orientation.NORMAL == 0
        assert Orientation.ROTATED_90_CW == 90
        assert Orientation.ROTATED_180 == 180
        assert Orientation.ROTATED_90_CCW == 270


# ---------------------------------------------------------------------------
# detect_orientation_from_exif tests
# ---------------------------------------------------------------------------


class TestExifFileOrientation:
    """Tests for file-based EXIF orientation detection."""

    def test_nonexistent_file(self):
        from missy.vision.orientation import detect_orientation_from_exif

        result = detect_orientation_from_exif("/nonexistent/path/image.jpg")
        assert result.confidence == 0.0

    @patch("missy.vision.orientation._get_cv2")
    def test_file_not_found_cv2_fallback(self, mock_cv2):
        from missy.vision.orientation import Orientation, detect_orientation_from_exif

        mock_cv2.return_value.imread.return_value = None
        result = detect_orientation_from_exif("/nonexistent.jpg")
        assert result.detected == Orientation.NORMAL
