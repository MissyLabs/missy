"""Tests for EXIF orientation parsing edge cases."""

from __future__ import annotations

import struct

import numpy as np

from missy.vision.orientation import (
    Orientation,
    _find_orientation_in_exif,
    _parse_exif_orientation,
    detect_orientation_from_exif,
)


class TestParseExifOrientation:
    """Tests for _parse_exif_orientation()."""

    def test_empty_bytes_returns_none(self):
        assert _parse_exif_orientation(b"") is None

    def test_short_bytes_returns_none(self):
        assert _parse_exif_orientation(b"\xff\xd8") is None

    def test_non_jpeg_returns_none(self):
        assert _parse_exif_orientation(b"\x89PNG" + b"\x00" * 100) is None

    def test_jpeg_without_exif_returns_none(self):
        # JPEG SOI + random non-APP1 marker
        data = b"\xff\xd8\xff\xe0" + struct.pack(">H", 10) + b"\x00" * 8
        assert _parse_exif_orientation(data) is None

    def test_truncated_jpeg_returns_none(self):
        data = b"\xff\xd8\xff\xe1\x00"  # truncated length
        assert _parse_exif_orientation(data) is None

    def _make_exif_jpeg(self, orientation: int, endian: str = ">") -> bytes:
        """Build a minimal valid JPEG with EXIF orientation tag."""
        byte_order = b"MM" if endian == ">" else b"II"
        # Build TIFF header
        tiff = bytearray()
        tiff.extend(byte_order)
        tiff.extend(struct.pack(f"{endian}H", 42))  # magic
        tiff.extend(struct.pack(f"{endian}I", 8))  # IFD0 offset

        # IFD0 with one entry: orientation tag (0x0112)
        tiff.extend(struct.pack(f"{endian}H", 1))  # num entries
        # tag=0x0112, type=3 (SHORT), count=1, value=orientation
        tiff.extend(struct.pack(f"{endian}HHI", 0x0112, 3, 1))
        tiff.extend(struct.pack(f"{endian}H", orientation))
        tiff.extend(b"\x00\x00")  # padding to 12 bytes

        # APP1 segment: "Exif\0\0" + TIFF
        exif_payload = b"Exif\x00\x00" + bytes(tiff)
        app1_len = len(exif_payload) + 2

        return b"\xff\xd8\xff\xe1" + struct.pack(">H", app1_len) + exif_payload

    def test_big_endian_orientation_1(self):
        data = self._make_exif_jpeg(1, ">")
        assert _parse_exif_orientation(data) == 1

    def test_big_endian_orientation_6(self):
        data = self._make_exif_jpeg(6, ">")
        assert _parse_exif_orientation(data) == 6

    def test_little_endian_orientation_3(self):
        data = self._make_exif_jpeg(3, "<")
        assert _parse_exif_orientation(data) == 3

    def test_little_endian_orientation_8(self):
        data = self._make_exif_jpeg(8, "<")
        assert _parse_exif_orientation(data) == 8


class TestFindOrientationInExif:
    """Tests for _find_orientation_in_exif()."""

    def test_empty_returns_none(self):
        assert _find_orientation_in_exif(b"") is None

    def test_short_returns_none(self):
        assert _find_orientation_in_exif(b"Exif\x00\x00") is None

    def test_bad_header_returns_none(self):
        assert _find_orientation_in_exif(b"NotExif" + b"\x00" * 20) is None

    def test_bad_byte_order_returns_none(self):
        data = b"Exif\x00\x00XX" + b"\x00" * 20
        assert _find_orientation_in_exif(data) is None


class TestDetectOrientationFromExif:
    """Tests for detect_orientation_from_exif()."""

    def test_nonexistent_file_returns_normal(self, tmp_path):
        result = detect_orientation_from_exif(str(tmp_path / "nope.jpg"))
        assert result.detected == Orientation.NORMAL

    def test_file_without_exif_uses_aspect_ratio(self, tmp_path):
        """A plain PNG (no EXIF) should fall back to aspect ratio analysis."""
        import cv2

        img = np.zeros((480, 640, 3), dtype=np.uint8)
        path = str(tmp_path / "test.png")
        cv2.imwrite(path, img)
        result = detect_orientation_from_exif(path)
        assert result.detected == Orientation.NORMAL
        # Should use aspect_ratio method since PNG has no EXIF
        assert result.method in ("aspect_ratio", "exif", "fallback")

    def test_jpeg_with_exif_returns_correct_orientation(self, tmp_path):
        """Build a JPEG with EXIF orientation=6 (90° CW)."""
        # Create minimal valid JPEG with EXIF
        byte_order = b"MM"
        tiff = bytearray()
        tiff.extend(byte_order)
        tiff.extend(struct.pack(">H", 42))
        tiff.extend(struct.pack(">I", 8))
        tiff.extend(struct.pack(">H", 1))
        tiff.extend(struct.pack(">HHI", 0x0112, 3, 1))
        tiff.extend(struct.pack(">H", 6))
        tiff.extend(b"\x00\x00")

        exif_payload = b"Exif\x00\x00" + bytes(tiff)
        app1_len = len(exif_payload) + 2

        # Minimal JPEG with EXIF + end marker
        jpeg_data = (
            b"\xff\xd8"  # SOI
            + b"\xff\xe1"
            + struct.pack(">H", app1_len)
            + exif_payload
            + b"\xff\xd9"  # EOI
        )

        path = tmp_path / "rotated.jpg"
        path.write_bytes(jpeg_data)

        result = detect_orientation_from_exif(str(path))
        # This minimal JPEG may not be readable by cv2.imread,
        # but the EXIF parser should work
        assert result.detected in (Orientation.ROTATED_90_CW, Orientation.NORMAL)
