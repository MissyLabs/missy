"""Session 12: Orientation detection, EXIF parsing, and pipeline edge cases.

Covers:
- Orientation detection from aspect ratio (landscape, portrait, square)
- Orientation correction (all rotation types)
- auto_correct with confidence thresholds
- EXIF parsing: valid JPEG, non-JPEG, truncated data, endianness
- Image pipeline quality assessment edge cases
- Pipeline preprocessing with various image types
"""

from __future__ import annotations

import struct

import numpy as np

# ---------------------------------------------------------------------------
# Orientation detection from aspect ratio
# ---------------------------------------------------------------------------


class TestOrientationDetection:
    def test_landscape_16_9(self) -> None:
        from missy.vision.orientation import Orientation, detect_orientation
        img = np.zeros((1080, 1920, 3), dtype=np.uint8)
        result = detect_orientation(img)
        assert result.detected == Orientation.NORMAL
        assert result.confidence > 0.5
        assert result.method == "aspect_ratio"

    def test_landscape_4_3(self) -> None:
        from missy.vision.orientation import Orientation, detect_orientation
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        result = detect_orientation(img)
        assert result.detected == Orientation.NORMAL

    def test_portrait_image(self) -> None:
        from missy.vision.orientation import Orientation, detect_orientation
        img = np.zeros((1920, 1080, 3), dtype=np.uint8)
        result = detect_orientation(img)
        # aspect = 1080/1920 ≈ 0.5625 < 0.8 → portrait
        assert result.detected == Orientation.ROTATED_90_CW
        assert result.confidence > 0.0

    def test_very_narrow_portrait(self) -> None:
        from missy.vision.orientation import Orientation, detect_orientation
        img = np.zeros((2000, 200, 3), dtype=np.uint8)
        result = detect_orientation(img)
        assert result.detected == Orientation.ROTATED_90_CW
        assert result.confidence > 0.5

    def test_square_image(self) -> None:
        from missy.vision.orientation import Orientation, detect_orientation
        img = np.zeros((500, 500, 3), dtype=np.uint8)
        result = detect_orientation(img)
        assert result.detected == Orientation.NORMAL
        assert result.confidence < 0.5  # low confidence for ambiguous
        assert result.method == "aspect_ratio"

    def test_near_square_image(self) -> None:
        from missy.vision.orientation import Orientation, detect_orientation
        # aspect = 500/450 ≈ 1.11 → between 0.8 and 1.25
        img = np.zeros((450, 500, 3), dtype=np.uint8)
        result = detect_orientation(img)
        assert result.detected == Orientation.NORMAL
        assert result.confidence == 0.3

    def test_none_image(self) -> None:
        from missy.vision.orientation import Orientation, detect_orientation
        result = detect_orientation(None)
        assert result.detected == Orientation.NORMAL
        assert result.confidence == 0.0
        assert result.method == "invalid_input"

    def test_1d_array(self) -> None:
        from missy.vision.orientation import Orientation, detect_orientation
        result = detect_orientation(np.zeros(100))
        assert result.detected == Orientation.NORMAL
        assert result.method == "invalid_input"

    def test_zero_height(self) -> None:
        from missy.vision.orientation import Orientation, detect_orientation
        img = np.zeros((0, 640, 3), dtype=np.uint8)
        result = detect_orientation(img)
        assert result.detected == Orientation.NORMAL
        assert result.method == "zero_dimension"

    def test_zero_width(self) -> None:
        from missy.vision.orientation import detect_orientation
        img = np.zeros((480, 0, 3), dtype=np.uint8)
        result = detect_orientation(img)
        assert result.method == "zero_dimension"

    def test_grayscale_image(self) -> None:
        from missy.vision.orientation import Orientation, detect_orientation
        img = np.zeros((480, 640), dtype=np.uint8)
        result = detect_orientation(img)
        assert result.detected == Orientation.NORMAL


# ---------------------------------------------------------------------------
# Orientation correction
# ---------------------------------------------------------------------------


class TestOrientationCorrection:
    def test_normal_returns_same(self) -> None:
        from missy.vision.orientation import Orientation, correct_orientation
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        corrected = correct_orientation(img, Orientation.NORMAL)
        assert corrected is img  # same object, not a copy

    def test_90_cw_correction(self) -> None:
        from missy.vision.orientation import Orientation, correct_orientation
        # 1080x1920 portrait → should become 1920x1080 landscape
        img = np.zeros((1920, 1080, 3), dtype=np.uint8)
        corrected = correct_orientation(img, Orientation.ROTATED_90_CW)
        assert corrected.shape[:2] == (1080, 1920)

    def test_180_correction(self) -> None:
        from missy.vision.orientation import Orientation, correct_orientation
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        corrected = correct_orientation(img, Orientation.ROTATED_180)
        assert corrected.shape == img.shape

    def test_90_ccw_correction(self) -> None:
        from missy.vision.orientation import Orientation, correct_orientation
        img = np.zeros((1920, 1080, 3), dtype=np.uint8)
        corrected = correct_orientation(img, Orientation.ROTATED_90_CCW)
        assert corrected.shape[:2] == (1080, 1920)

    def test_unknown_orientation_returns_unchanged(self) -> None:
        from missy.vision.orientation import correct_orientation
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        corrected = correct_orientation(img, 45)  # type: ignore[arg-type]
        assert corrected is img


# ---------------------------------------------------------------------------
# auto_correct
# ---------------------------------------------------------------------------


class TestAutoCorrect:
    def test_corrects_rotated_portrait(self) -> None:
        from missy.vision.orientation import auto_correct
        # Very tall portrait: aspect = 200/2000 = 0.1 → confidence = min(0.9, 1-0.1) = 0.9
        img = np.zeros((2000, 200, 3), dtype=np.uint8)
        corrected, result = auto_correct(img)
        assert result.correction_applied
        assert corrected.shape[:2] == (200, 2000)

    def test_no_correction_for_landscape(self) -> None:
        from missy.vision.orientation import auto_correct
        img = np.zeros((1080, 1920, 3), dtype=np.uint8)
        corrected, result = auto_correct(img)
        assert not result.correction_applied
        assert corrected is img

    def test_no_correction_for_low_confidence(self) -> None:
        from missy.vision.orientation import auto_correct
        # Square → confidence 0.3 < threshold 0.5
        img = np.zeros((500, 500, 3), dtype=np.uint8)
        corrected, result = auto_correct(img)
        assert not result.correction_applied

    def test_none_image(self) -> None:
        from missy.vision.orientation import auto_correct
        corrected, result = auto_correct(None)
        assert not result.correction_applied
        assert result.confidence == 0.0


# ---------------------------------------------------------------------------
# EXIF parsing
# ---------------------------------------------------------------------------


class TestExifParsing:
    def test_non_jpeg_returns_none(self) -> None:
        from missy.vision.orientation import _parse_exif_orientation
        assert _parse_exif_orientation(b"PNG data here") is None

    def test_too_short_returns_none(self) -> None:
        from missy.vision.orientation import _parse_exif_orientation
        assert _parse_exif_orientation(b"\xff\xd8") is None

    def test_empty_returns_none(self) -> None:
        from missy.vision.orientation import _parse_exif_orientation
        assert _parse_exif_orientation(b"") is None

    def test_jpeg_without_exif_returns_none(self) -> None:
        from missy.vision.orientation import _parse_exif_orientation
        # SOI followed by EOI
        data = b"\xff\xd8\xff\xd9"
        assert _parse_exif_orientation(data) is None

    def test_find_orientation_non_exif_returns_none(self) -> None:
        from missy.vision.orientation import _find_orientation_in_exif
        assert _find_orientation_in_exif(b"not exif data") is None

    def test_find_orientation_truncated_returns_none(self) -> None:
        from missy.vision.orientation import _find_orientation_in_exif
        assert _find_orientation_in_exif(b"Exif\x00\x00II") is None

    def _make_exif_data(self, orientation: int, endian: str = "<") -> bytes:
        """Build minimal EXIF data with orientation tag."""
        fmt = endian
        # TIFF header
        if endian == "<":
            tiff = b"II" + struct.pack("<H", 42) + struct.pack("<I", 8)
        else:
            tiff = b"MM" + struct.pack(">H", 42) + struct.pack(">I", 8)

        # IFD with 1 entry: orientation tag (0x0112)
        num_entries = struct.pack(f"{fmt}H", 1)
        tag = struct.pack(f"{fmt}H", 0x0112)
        type_short = struct.pack(f"{fmt}H", 3)  # SHORT
        count = struct.pack(f"{fmt}I", 1)
        value = struct.pack(f"{fmt}H", orientation) + b"\x00\x00"
        ifd = num_entries + tag + type_short + count + value

        return b"Exif\x00\x00" + tiff + ifd

    def test_valid_exif_orientation_1(self) -> None:
        from missy.vision.orientation import _find_orientation_in_exif
        data = self._make_exif_data(1, "<")
        assert _find_orientation_in_exif(data) == 1

    def test_valid_exif_orientation_6(self) -> None:
        from missy.vision.orientation import _find_orientation_in_exif
        data = self._make_exif_data(6, "<")
        assert _find_orientation_in_exif(data) == 6

    def test_valid_exif_orientation_big_endian(self) -> None:
        from missy.vision.orientation import _find_orientation_in_exif
        data = self._make_exif_data(3, ">")
        assert _find_orientation_in_exif(data) == 3

    def test_exif_no_orientation_tag(self) -> None:
        """EXIF data with a non-orientation tag returns None."""
        from missy.vision.orientation import _find_orientation_in_exif
        # Build IFD with tag 0x010F (Make) instead of 0x0112
        tiff = b"II" + struct.pack("<H", 42) + struct.pack("<I", 8)
        num_entries = struct.pack("<H", 1)
        tag = struct.pack("<H", 0x010F)
        type_short = struct.pack("<H", 2)
        count = struct.pack("<I", 1)
        value = b"X\x00\x00\x00"
        ifd = num_entries + tag + type_short + count + value
        data = b"Exif\x00\x00" + tiff + ifd
        assert _find_orientation_in_exif(data) is None


# ---------------------------------------------------------------------------
# detect_orientation_from_exif
# ---------------------------------------------------------------------------


class TestDetectOrientationFromExif:
    def test_nonexistent_file(self, tmp_path) -> None:
        from missy.vision.orientation import detect_orientation_from_exif
        result = detect_orientation_from_exif(str(tmp_path / "nope.jpg"))
        assert result.confidence == 0.0

    def test_file_without_exif_uses_aspect_ratio(self, tmp_path) -> None:
        """A file that exists but has no EXIF falls back to aspect ratio."""
        from missy.vision.orientation import detect_orientation_from_exif

        # Create a minimal file — cv2.imread will handle it
        img_path = tmp_path / "test.jpg"
        img_path.write_bytes(b"not a real image")

        # Even though cv2.imread will return None for garbage, the function
        # should handle it gracefully
        result = detect_orientation_from_exif(str(img_path))
        assert result is not None


# ---------------------------------------------------------------------------
# Pipeline quality assessment
# ---------------------------------------------------------------------------


class TestPipelineQuality:
    def test_quality_assessment_uniform_image(self) -> None:
        """Uniform (solid color) image should get low quality scores."""
        from missy.vision.pipeline import ImagePipeline
        pipeline = ImagePipeline()
        img = np.ones((100, 100, 3), dtype=np.uint8) * 128
        assessment = pipeline.assess_quality(img)
        assert "quality" in assessment
        # Uniform images lack contrast and sharpness
        assert assessment["quality"] in ("poor", "fair")

    def test_quality_assessment_noisy_image(self) -> None:
        """Random noise image should have some contrast but may lack structure."""
        from missy.vision.pipeline import ImagePipeline
        pipeline = ImagePipeline()
        rng = np.random.default_rng(42)
        img = rng.integers(0, 256, (100, 100, 3), dtype=np.uint8)
        assessment = pipeline.assess_quality(img)
        assert "quality" in assessment
        assert assessment["quality"] in ("poor", "fair", "good", "excellent")

    def test_quality_assessment_very_small_image(self) -> None:
        from missy.vision.pipeline import ImagePipeline
        pipeline = ImagePipeline()
        img = np.zeros((5, 5, 3), dtype=np.uint8)
        assessment = pipeline.assess_quality(img)
        assert "quality" in assessment

    def test_process_resizes_large_image(self) -> None:
        from missy.vision.pipeline import ImagePipeline, PipelineConfig
        pipeline = ImagePipeline(PipelineConfig(max_dimension=800, target_dimension=800))
        img = np.zeros((2000, 3000, 3), dtype=np.uint8)
        result = pipeline.process(img)
        h, w = result.shape[:2]
        assert max(h, w) <= 800

    def test_process_does_not_upscale(self) -> None:
        from missy.vision.pipeline import ImagePipeline, PipelineConfig
        pipeline = ImagePipeline(PipelineConfig(max_dimension=2000, target_dimension=2000))
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        result = pipeline.process(img)
        assert result.shape[:2] == (480, 640)

    def test_process_grayscale(self) -> None:
        from missy.vision.pipeline import ImagePipeline
        pipeline = ImagePipeline()
        img = np.zeros((480, 640), dtype=np.uint8)
        result = pipeline.process(img)
        assert result is not None


# ---------------------------------------------------------------------------
# OrientationResult dataclass
# ---------------------------------------------------------------------------


class TestOrientationResult:
    def test_default_values(self) -> None:
        from missy.vision.orientation import Orientation, OrientationResult
        r = OrientationResult(detected=Orientation.NORMAL, confidence=0.5)
        assert not r.correction_applied
        assert r.method == ""

    def test_all_fields(self) -> None:
        from missy.vision.orientation import Orientation, OrientationResult
        r = OrientationResult(
            detected=Orientation.ROTATED_180,
            confidence=0.95,
            correction_applied=True,
            method="exif",
        )
        assert r.detected == 180
        assert r.method == "exif"


# ---------------------------------------------------------------------------
# Orientation enum values
# ---------------------------------------------------------------------------


class TestOrientationEnum:
    def test_enum_values(self) -> None:
        from missy.vision.orientation import Orientation
        assert Orientation.NORMAL == 0
        assert Orientation.ROTATED_90_CW == 90
        assert Orientation.ROTATED_180 == 180
        assert Orientation.ROTATED_90_CCW == 270

    def test_enum_is_int(self) -> None:
        from missy.vision.orientation import Orientation
        assert isinstance(Orientation.NORMAL, int)
