"""Unified image source abstraction layer.

Provides a single ``ImageSource`` interface for all vision inputs:
webcam, file, screenshot, and saved photo.  Each source produces an
``ImageFrame`` that downstream analysis consumes uniformly.
"""

from __future__ import annotations

import base64
import logging
import stat
import subprocess
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
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
        try:
            import cv2
            _cv2 = cv2
        except ImportError:
            raise ImportError(
                "opencv-python is required for vision sources. "
                "Install with: pip install opencv-python-headless"
            ) from None
    return _cv2


# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------


class SourceType(StrEnum):
    WEBCAM = "webcam"
    FILE = "file"
    SCREENSHOT = "screenshot"
    PHOTO = "photo"


@dataclass
class ImageFrame:
    """Normalized image frame from any source."""

    image: np.ndarray  # BGR numpy array
    source_type: SourceType
    source_path: str = ""  # file path or device path
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))
    width: int = 0
    height: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.image is not None and self.width == 0:
            h, w = self.image.shape[:2]
            self.width = w
            self.height = h

    def to_jpeg_bytes(self, quality: int = 85) -> bytes:
        """Encode frame as JPEG bytes."""
        cv2 = _get_cv2()
        success, buf = cv2.imencode(".jpg", self.image, [cv2.IMWRITE_JPEG_QUALITY, quality])
        if not success or buf is None:
            raise RuntimeError("Failed to encode image as JPEG")
        return buf.tobytes()

    def to_base64(self, quality: int = 85) -> str:
        """Encode frame as base64 JPEG for LLM consumption."""
        return base64.b64encode(self.to_jpeg_bytes(quality)).decode("ascii")

    def to_png_bytes(self) -> bytes:
        """Encode frame as PNG bytes."""
        cv2 = _get_cv2()
        success, buf = cv2.imencode(".png", self.image)
        if not success or buf is None:
            raise RuntimeError("Failed to encode image as PNG")
        return buf.tobytes()


# ---------------------------------------------------------------------------
# Abstract source
# ---------------------------------------------------------------------------


class ImageSource(ABC):
    """Abstract interface for image sources."""

    @abstractmethod
    def acquire(self) -> ImageFrame:
        """Acquire a single image frame from this source."""
        ...

    @abstractmethod
    def source_type(self) -> SourceType:
        ...

    def is_available(self) -> bool:
        """Check if this source is currently available."""
        return True


# ---------------------------------------------------------------------------
# Webcam source
# ---------------------------------------------------------------------------


class WebcamSource(ImageSource):
    """Acquires frames from a USB webcam via OpenCV.

    Parameters
    ----------
    device_path:
        V4L2 device path (e.g. ``/dev/video0``).
    timeout:
        Maximum seconds to wait for a frame before giving up.
    """

    def __init__(self, device_path: str = "/dev/video0", *, timeout: float = 15.0) -> None:
        # Validate device path to prevent command injection or unexpected access
        import re

        if not re.match(r"^/dev/video\d+$", device_path):
            raise ValueError(
                f"Invalid device path: {device_path!r} "
                "(expected /dev/videoN format)"
            )
        self._device_path = device_path
        self._timeout = timeout

    def source_type(self) -> SourceType:
        return SourceType.WEBCAM

    def is_available(self) -> bool:
        return Path(self._device_path).exists()

    def acquire(self) -> ImageFrame:
        import concurrent.futures

        from missy.vision.capture import CameraHandle, CaptureConfig, CaptureError

        def _do_capture() -> ImageFrame:
            config = CaptureConfig(timeout_seconds=self._timeout)
            handle = CameraHandle(self._device_path, config)
            try:
                handle.open()
                result = handle.capture()
                if not result.success:
                    raise CaptureError(result.error)
                return ImageFrame(
                    image=result.image,
                    source_type=SourceType.WEBCAM,
                    source_path=self._device_path,
                    width=result.width,
                    height=result.height,
                )
            finally:
                handle.close()

        # Run capture in a thread with timeout to defend against frozen cameras
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(_do_capture)
            try:
                return future.result(timeout=self._timeout)
            except concurrent.futures.TimeoutError as err:
                raise CaptureError(
                    f"Camera at {self._device_path} did not respond within "
                    f"{self._timeout}s — device may be frozen or busy"
                ) from err


# ---------------------------------------------------------------------------
# File source
# ---------------------------------------------------------------------------


class FileSource(ImageSource):
    """Loads an image from a file path.

    Security: Resolves the path to detect traversal and verifies it is
    a regular file (not a symlink to a device node, etc.).
    """

    def __init__(self, path: str | Path) -> None:
        self._path = Path(path).resolve()

    def source_type(self) -> SourceType:
        return SourceType.FILE

    def is_available(self) -> bool:
        return self._path.exists() and self._path.is_file()

    # Maximum image dimensions to prevent resource exhaustion
    MAX_DIMENSION = 16384
    # Maximum file size to prevent loading huge files (100 MB)
    MAX_FILE_SIZE = 100 * 1024 * 1024

    def acquire(self) -> ImageFrame:
        cv2 = _get_cv2()
        if not self._path.exists():
            raise FileNotFoundError(f"Image file not found: {self._path}")

        # Verify it's a regular file (not a device node, socket, or pipe)
        file_stat = self._path.stat()
        if not stat.S_ISREG(file_stat.st_mode):
            raise ValueError(
                f"Not a regular file (mode 0o{file_stat.st_mode:o}): {self._path}"
            )

        # Check file size before loading
        file_size = file_stat.st_size  # reuse stat result from above
        if file_size == 0:
            raise ValueError(f"Image file is empty: {self._path}")
        if file_size > self.MAX_FILE_SIZE:
            raise ValueError(
                f"Image file too large ({file_size / 1024 / 1024:.1f} MB, "
                f"max {self.MAX_FILE_SIZE / 1024 / 1024:.0f} MB): {self._path}"
            )

        try:
            img = cv2.imread(str(self._path))
        except Exception as exc:
            raise OSError(f"Error reading image {self._path}: {exc}") from exc
        if img is None:
            raise ValueError(f"Failed to decode image (unsupported format?): {self._path}")

        # Validate dimensions
        h, w = img.shape[:2]
        if h == 0 or w == 0:
            raise ValueError(f"Image has zero dimension ({w}x{h}): {self._path}")
        if h > self.MAX_DIMENSION or w > self.MAX_DIMENSION:
            logger.warning(
                "Image %s has large dimensions (%dx%d), may be slow to process",
                self._path, w, h,
            )

        return ImageFrame(
            image=img,
            source_type=SourceType.FILE,
            source_path=str(self._path),
        )


# ---------------------------------------------------------------------------
# Screenshot source
# ---------------------------------------------------------------------------


class ScreenshotSource(ImageSource):
    """Captures a screenshot of the current display.

    Uses ``scrot`` or ``gnome-screenshot`` as available on the system.
    Falls back to X11 via Pillow if neither CLI tool is found.
    """

    def __init__(self, display: str | None = None) -> None:
        self._display = display

    def source_type(self) -> SourceType:
        return SourceType.SCREENSHOT

    def is_available(self) -> bool:
        # Check if any screenshot tool is available
        for cmd in ("scrot", "gnome-screenshot", "grim"):
            try:
                result = subprocess.run(
                    ["which", cmd], capture_output=True, timeout=5
                )
                if result.returncode == 0:
                    return True
            except Exception:
                continue
        return False

    def acquire(self) -> ImageFrame:
        import tempfile

        cv2 = _get_cv2()

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            tool_used = self._take_screenshot(tmp_path)
            img = cv2.imread(tmp_path)
            if img is None:
                raise RuntimeError(
                    f"Screenshot tool '{tool_used}' created file but image is unreadable"
                )
            return ImageFrame(
                image=img,
                source_type=SourceType.SCREENSHOT,
                source_path=tmp_path,
            )
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    def _take_screenshot(self, output_path: str) -> str:
        """Try available screenshot tools in order.  Returns the tool name used."""
        import os

        env = dict(os.environ)
        if self._display:
            env["DISPLAY"] = self._display

        tools = [
            (["scrot", "-o", output_path], "scrot"),
            (["gnome-screenshot", "-f", output_path], "gnome-screenshot"),
            (["grim", output_path], "grim"),
        ]

        errors: list[str] = []
        for cmd, name in tools:
            try:
                result = subprocess.run(
                    cmd, capture_output=True, timeout=10, env=env
                )
                if result.returncode == 0 and Path(output_path).exists():
                    logger.debug("Screenshot captured with %s", name)
                    return name
                errors.append(f"{name}: exit {result.returncode}")
            except FileNotFoundError:
                errors.append(f"{name}: not found")
            except subprocess.TimeoutExpired:
                errors.append(f"{name}: timed out")
            except Exception as exc:
                errors.append(f"{name}: {exc}")

        raise RuntimeError(
            "No screenshot tool succeeded. Tried: " + "; ".join(errors)
        )


# ---------------------------------------------------------------------------
# Saved photo source
# ---------------------------------------------------------------------------


class PhotoSource(ImageSource):
    """Reviews saved photos from a directory.

    Scans a directory for image files and provides them in order.
    Useful for reviewing a set of saved images.
    """

    SUPPORTED_EXTENSIONS = frozenset({".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"})

    def __init__(self, directory: str | Path, *, pattern: str = "*") -> None:
        self._directory = Path(directory).resolve()
        self._pattern = pattern
        self._files: list[Path] = []
        self._index: int = 0
        self._scanned = False

    def source_type(self) -> SourceType:
        return SourceType.PHOTO

    def is_available(self) -> bool:
        return self._directory.exists() and self._directory.is_dir()

    def scan(self) -> list[Path]:
        """Scan directory for image files and return sorted list.

        Respects the ``pattern`` parameter (glob-style) passed at construction.
        """
        if not self._directory.exists():
            raise FileNotFoundError(f"Photo directory not found: {self._directory}")
        try:
            self._files = sorted(
                p.resolve()
                for p in self._directory.glob(self._pattern)
                if p.is_file()
                and p.suffix.lower() in self.SUPPORTED_EXTENSIONS
                and str(p.resolve()).startswith(str(self._directory))
            )
        except OSError as exc:
            raise OSError(f"Cannot scan directory {self._directory}: {exc}") from exc
        self._scanned = True
        self._index = 0
        return list(self._files)

    @property
    def file_count(self) -> int:
        if not self._scanned:
            self.scan()
        return len(self._files)

    def acquire(self) -> ImageFrame:
        """Acquire the next photo in sequence."""
        if not self._scanned:
            self.scan()
        if not self._files:
            raise FileNotFoundError(f"No images found in {self._directory}")
        if self._index >= len(self._files):
            self._index = 0  # wrap around

        path = self._files[self._index]
        self._index += 1

        source = FileSource(path)
        frame = source.acquire()
        frame.source_type = SourceType.PHOTO
        frame.metadata["photo_index"] = self._index - 1
        frame.metadata["photo_total"] = len(self._files)
        return frame

    def acquire_specific(self, index: int) -> ImageFrame:
        """Acquire a specific photo by index."""
        if not self._scanned:
            self.scan()
        if index < 0 or index >= len(self._files):
            raise IndexError(f"Photo index {index} out of range (0-{len(self._files) - 1})")
        path = self._files[index]
        source = FileSource(path)
        frame = source.acquire()
        frame.source_type = SourceType.PHOTO
        frame.metadata["photo_index"] = index
        frame.metadata["photo_total"] = len(self._files)
        return frame


# ---------------------------------------------------------------------------
# Source factory
# ---------------------------------------------------------------------------


def create_source(
    source_type: SourceType | str,
    *,
    device_path: str = "/dev/video0",
    file_path: str = "",
    directory: str = "",
    display: str | None = None,
) -> ImageSource:
    """Factory function to create the appropriate image source."""
    if isinstance(source_type, str):
        source_type = SourceType(source_type)

    if source_type == SourceType.WEBCAM:
        return WebcamSource(device_path)
    elif source_type == SourceType.FILE:
        if not file_path:
            raise ValueError("file_path required for file source")
        return FileSource(file_path)
    elif source_type == SourceType.SCREENSHOT:
        return ScreenshotSource(display)
    elif source_type == SourceType.PHOTO:
        if not directory:
            raise ValueError("directory required for photo source")
        return PhotoSource(directory)
    else:
        raise ValueError(f"Unknown source type: {source_type}")
