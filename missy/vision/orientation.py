"""Camera image orientation detection and correction.

Detects whether an image is rotated (e.g. from a sideways phone photo
or a rotated webcam mount) and provides correction utilities.

Uses aspect ratio analysis and optional EXIF metadata for file-based
images.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from enum import IntEnum
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

_cv2: Any = None
_cv2_lock = threading.Lock()


def _get_cv2() -> Any:
    """Lazily import OpenCV.  Thread-safe."""
    global _cv2
    if _cv2 is None:
        with _cv2_lock:
            if _cv2 is None:
                import cv2

                _cv2 = cv2
    return _cv2


class Orientation(IntEnum):
    """Image orientation states."""

    NORMAL = 0  # No rotation needed
    ROTATED_90_CW = 90  # Rotated 90° clockwise
    ROTATED_180 = 180  # Upside down
    ROTATED_90_CCW = 270  # Rotated 90° counter-clockwise


@dataclass
class OrientationResult:
    """Result of orientation detection."""

    detected: Orientation
    confidence: float  # 0.0 to 1.0
    correction_applied: bool = False
    method: str = ""  # "exif", "aspect_ratio", "edge_analysis"


def detect_orientation(image: np.ndarray) -> OrientationResult:
    """Detect the likely orientation of an image.

    Uses aspect ratio as the primary signal.  Webcam images are normally
    landscape (wider than tall).  If a webcam image is portrait, it may
    be rotated 90°.

    Parameters
    ----------
    image:
        BGR or grayscale numpy array.

    Returns
    -------
    OrientationResult
        Detected orientation and confidence.
    """
    if image is None or image.ndim < 2:
        return OrientationResult(
            detected=Orientation.NORMAL,
            confidence=0.0,
            method="invalid_input",
        )

    h, w = image.shape[:2]
    if h == 0 or w == 0:
        return OrientationResult(
            detected=Orientation.NORMAL,
            confidence=0.0,
            method="zero_dimension",
        )

    aspect = w / h

    # Standard webcam is 16:9 (1.78) or 4:3 (1.33)
    # If portrait (aspect < 1.0), likely rotated 90°
    if aspect < 0.8:
        # Portrait image from landscape camera — likely rotated 90°
        # Can't determine CW vs CCW from aspect alone
        return OrientationResult(
            detected=Orientation.ROTATED_90_CW,
            confidence=min(0.9, 1.0 - aspect),
            method="aspect_ratio",
        )
    elif 0.8 <= aspect <= 1.25:
        # Nearly square — can't determine orientation
        return OrientationResult(
            detected=Orientation.NORMAL,
            confidence=0.3,
            method="aspect_ratio",
        )
    else:
        # Landscape — normal orientation
        return OrientationResult(
            detected=Orientation.NORMAL,
            confidence=min(0.95, aspect / 2.0),
            method="aspect_ratio",
        )


def detect_orientation_from_exif(file_path: str) -> OrientationResult:
    """Detect orientation from EXIF metadata in a JPEG/TIFF file.

    Falls back to ``detect_orientation`` if EXIF is unavailable.

    Parameters
    ----------
    file_path:
        Path to an image file.

    Returns
    -------
    OrientationResult
        Detected orientation from EXIF or aspect ratio fallback.
    """
    try:
        from pathlib import Path

        path = Path(file_path)
        if not path.exists() or not path.is_file():
            cv2 = _get_cv2()
            img = cv2.imread(str(path))
            if img is not None:
                return detect_orientation(img)
            return OrientationResult(
                detected=Orientation.NORMAL,
                confidence=0.0,
                method="file_not_found",
            )

        # Read EXIF orientation tag from JPEG
        data = path.read_bytes()[:65536]  # only need header
        orientation = _parse_exif_orientation(data)

        if orientation is not None:
            exif_map = {
                1: Orientation.NORMAL,
                3: Orientation.ROTATED_180,
                6: Orientation.ROTATED_90_CW,
                8: Orientation.ROTATED_90_CCW,
            }
            detected = exif_map.get(orientation, Orientation.NORMAL)
            return OrientationResult(
                detected=detected,
                confidence=0.95,
                method="exif",
            )
    except Exception as exc:
        logger.debug("EXIF read failed for %s: %s", file_path, exc)

    # Fallback to aspect ratio analysis
    try:
        cv2 = _get_cv2()
        img = cv2.imread(file_path)
        if img is not None:
            return detect_orientation(img)
    except Exception:
        pass

    return OrientationResult(
        detected=Orientation.NORMAL,
        confidence=0.0,
        method="fallback",
    )


def correct_orientation(
    image: np.ndarray,
    orientation: Orientation,
) -> np.ndarray:
    """Rotate an image to correct the detected orientation.

    Parameters
    ----------
    image:
        BGR or grayscale numpy array.
    orientation:
        The detected (incorrect) orientation to correct.

    Returns
    -------
    np.ndarray
        The corrected image.
    """
    cv2 = _get_cv2()

    if orientation == Orientation.NORMAL:
        return image

    if orientation == Orientation.ROTATED_90_CW:
        return cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    elif orientation == Orientation.ROTATED_180:
        return cv2.rotate(image, cv2.ROTATE_180)
    elif orientation == Orientation.ROTATED_90_CCW:
        return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    else:
        logger.warning("Unknown orientation %s, returning unchanged", orientation)
        return image


def auto_correct(image: np.ndarray) -> tuple[np.ndarray, OrientationResult]:
    """Detect and correct image orientation in one step.

    Returns both the corrected image and the detection result.
    """
    result = detect_orientation(image)
    if result.detected != Orientation.NORMAL and result.confidence >= 0.5:
        corrected = correct_orientation(image, result.detected)
        result.correction_applied = True
        return corrected, result
    return image, result


def _parse_exif_orientation(data: bytes) -> int | None:
    """Parse EXIF orientation tag from raw JPEG bytes.

    Returns the EXIF orientation value (1-8) or None.
    """
    import struct

    if len(data) < 14:
        return None

    # Check for JPEG SOI marker
    if data[:2] != b"\xff\xd8":
        return None

    # Walk through JPEG markers looking for APP1 (EXIF)
    offset = 2
    while offset < len(data) - 4:
        if data[offset] != 0xFF:
            break

        marker = data[offset + 1]
        if marker == 0xE1:  # APP1
            # Found EXIF segment
            seg_len = struct.unpack(">H", data[offset + 2 : offset + 4])[0]
            exif_data = data[offset + 4 : offset + 2 + seg_len]
            return _find_orientation_in_exif(exif_data)

        # Skip to next marker
        if marker in (0xD8, 0xD9):  # SOI, EOI
            offset += 2
        else:
            seg_len = struct.unpack(">H", data[offset + 2 : offset + 4])[0]
            offset += 2 + seg_len

    return None


def _find_orientation_in_exif(exif_data: bytes) -> int | None:
    """Find orientation tag (0x0112) in EXIF data."""
    import struct

    if len(exif_data) < 14:
        return None

    # Check Exif header
    if exif_data[:4] != b"Exif" and exif_data[:6] != b"Exif\x00\x00":
        return None

    # Skip "Exif\0\0"
    tiff_offset = 6
    if len(exif_data) < tiff_offset + 8:
        return None

    tiff_data = exif_data[tiff_offset:]

    # Determine byte order
    if tiff_data[:2] == b"MM":
        endian = ">"
    elif tiff_data[:2] == b"II":
        endian = "<"
    else:
        return None

    # Read IFD0 offset
    ifd_offset = struct.unpack(f"{endian}I", tiff_data[4:8])[0]
    if ifd_offset + 2 > len(tiff_data):
        return None

    # Read number of entries
    num_entries = struct.unpack(f"{endian}H", tiff_data[ifd_offset : ifd_offset + 2])[0]

    # Search for orientation tag (0x0112)
    for i in range(min(num_entries, 50)):  # cap to avoid runaway
        entry_offset = ifd_offset + 2 + i * 12
        if entry_offset + 12 > len(tiff_data):
            break
        tag = struct.unpack(f"{endian}H", tiff_data[entry_offset : entry_offset + 2])[0]
        if tag == 0x0112:  # Orientation
            value = struct.unpack(f"{endian}H", tiff_data[entry_offset + 8 : entry_offset + 10])[0]
            return value

    return None
