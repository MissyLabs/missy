"""OpenCV-based camera capture with resilience and diagnostics.

Handles device open/close, warm-up, retry on failure, blank frame detection,
and graceful degradation when the camera is unavailable.

Design decisions
----------------
- OpenCV is the primary capture backend.  Raw V4L2 is NOT used because OpenCV
  provides cross-platform abstraction, handles format negotiation, and
  supports the Logitech C922x without additional setup.
- Warm-up frames are discarded to avoid auto-exposure artifacts.
- Blank frame detection prevents processing useless captures.
"""

from __future__ import annotations

import logging
import threading
import time
from collections import deque
from contextlib import contextmanager, suppress
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Lazy OpenCV import
# ---------------------------------------------------------------------------

_cv2: Any = None


def _get_cv2() -> Any:
    """Lazily import OpenCV to avoid hard dependency at module load."""
    global _cv2
    if _cv2 is None:
        try:
            import cv2

            _cv2 = cv2
        except ImportError:
            raise ImportError(
                "opencv-python is required for vision capture. "
                "Install with: pip install opencv-python-headless"
            ) from None
    return _cv2


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


@dataclass
class CaptureResult:
    """Result of a single frame capture."""

    success: bool
    image: np.ndarray | None = None  # BGR numpy array
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))
    device_path: str = ""
    width: int = 0
    height: int = 0
    error: str = ""
    failure_type: str = ""
    attempt_count: int = 1
    warmup_frames_discarded: int = 0

    @property
    def shape(self) -> tuple[int, ...]:
        if self.image is not None:
            return self.image.shape
        return (0, 0, 0)


@dataclass
class CaptureConfig:
    """Configuration for capture operations."""

    width: int = 1920
    height: int = 1080
    warmup_frames: int = 5
    max_retries: int = 3
    retry_delay: float = 0.5
    blank_threshold: float = 5.0  # mean pixel value below this = blank
    adaptive_blank: bool = True  # auto-adjust blank threshold from history
    quality: int = 95  # JPEG quality for saves
    timeout_seconds: float = 10.0


# ---------------------------------------------------------------------------
# Adaptive blank frame detection
# ---------------------------------------------------------------------------


class AdaptiveBlankDetector:
    """Learns ambient light levels from successful captures and adjusts
    the blank-frame threshold to avoid false positives in dim environments.

    Maintains a rolling window of mean pixel intensities from recent
    successful frames.  The adaptive threshold is a fraction of the
    observed minimum intensity, with a floor of ``min_threshold``.
    """

    def __init__(
        self,
        base_threshold: float = 5.0,
        min_threshold: float = 2.0,
        window_size: int = 20,
        adaptation_factor: float = 0.25,
    ) -> None:
        self._base = base_threshold
        self._min = min_threshold
        self._window: deque[float] = deque(maxlen=window_size)
        self._factor = adaptation_factor

    def record_intensity(self, mean_pixel: float) -> None:
        """Record the mean pixel intensity of a successful capture."""
        if mean_pixel > 0:
            self._window.append(mean_pixel)

    @property
    def threshold(self) -> float:
        """Current adaptive blank-frame threshold."""
        if len(self._window) < 3:
            return self._base
        # Use a fraction of the minimum observed intensity
        min_observed = min(self._window)
        adaptive = min_observed * self._factor
        return max(self._min, min(adaptive, self._base))

    def is_blank(self, frame: np.ndarray) -> bool:
        """Check if a frame is blank using the adaptive threshold."""
        mean_val = float(np.mean(frame))
        return mean_val < self.threshold

    def reset(self) -> None:
        """Clear recorded history."""
        self._window.clear()


# ---------------------------------------------------------------------------
# Camera handle
# ---------------------------------------------------------------------------


class CameraHandle:
    """Manages an OpenCV VideoCapture instance with lifecycle controls.

    Use as a context manager for automatic cleanup::

        with CameraHandle("/dev/video0") as cam:
            result = cam.capture()
    """

    def __init__(
        self,
        device_path: str,
        config: CaptureConfig | None = None,
    ) -> None:
        if not device_path:
            raise ValueError("device_path must be a non-empty string")
        self._device_path = device_path
        self._config = config or CaptureConfig()
        self._cap: Any = None  # cv2.VideoCapture
        self._opened = False
        self._open_time: float = 0.0
        self._lock = threading.Lock()
        self._blank_detector = AdaptiveBlankDetector(
            base_threshold=self._config.blank_threshold,
        ) if self._config.adaptive_blank else None

    @property
    def is_open(self) -> bool:
        return self._opened and self._cap is not None and self._cap.isOpened()

    def open(self) -> None:
        """Open the camera device."""
        if self.is_open:
            return

        cv2 = _get_cv2()

        # Try numeric device index first (e.g. /dev/video0 → 0)
        device_index = self._parse_device_index(self._device_path)

        try:
            if device_index is not None:
                self._cap = cv2.VideoCapture(device_index, cv2.CAP_V4L2)
            else:
                self._cap = cv2.VideoCapture(self._device_path, cv2.CAP_V4L2)
        except Exception as exc:
            raise CaptureError(
                f"Failed to create VideoCapture for {self._device_path}: {exc}"
            ) from exc

        if not self._cap.isOpened():
            self._cap.release()
            self._cap = None
            raise CaptureError(
                f"Cannot open camera at {self._device_path}. "
                "Check permissions (user in 'video' group?) and device availability."
            )

        # Set resolution
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self._config.width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._config.height)

        # Verify resolution was accepted
        actual_w = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if actual_w != self._config.width or actual_h != self._config.height:
            logger.warning(
                "Requested resolution %dx%d but camera set %dx%d",
                self._config.width,
                self._config.height,
                actual_w,
                actual_h,
            )

        self._opened = True
        self._open_time = time.monotonic()

        # Warm-up: discard initial frames for auto-exposure/white-balance
        self._warmup()

        logger.info(
            "Camera opened: %s (%dx%d)",
            self._device_path,
            actual_w,
            actual_h,
        )

    def close(self) -> None:
        """Release the camera device.  Thread-safe."""
        with self._lock:
            if self._cap is not None:
                with suppress(Exception):
                    self._cap.release()
                self._cap = None
            self._opened = False

    def capture(self) -> CaptureResult:
        """Capture a single frame with retry logic.  Thread-safe.

        Enforces the ``timeout_seconds`` setting from the capture config.
        If all retries exceed the timeout, returns a failure result.
        """
        with self._lock:
            if not self.is_open:
                raise CaptureError("Camera is not open")

            config = self._config
            last_error = ""
            deadline = time.monotonic() + config.timeout_seconds

            for attempt in range(1, config.max_retries + 1):
                if time.monotonic() > deadline:
                    return CaptureResult(
                        success=False,
                        device_path=self._device_path,
                        error=f"Capture timed out after {config.timeout_seconds}s",
                        failure_type=FailureType.TRANSIENT,
                        attempt_count=attempt - 1,
                    )
                try:
                    ret, frame = self._cap.read()
                    if not ret or frame is None:
                        last_error = f"read() returned {ret} on attempt {attempt}"
                        logger.warning("Frame read failed: %s", last_error)
                        if attempt < config.max_retries:
                            time.sleep(config.retry_delay)
                        else:
                            return CaptureResult(
                                success=False,
                                device_path=self._device_path,
                                error=f"Capture failed after {config.max_retries} attempts: {last_error}",
                                failure_type=FailureType.TRANSIENT,
                                attempt_count=config.max_retries,
                            )
                        continue

                    # Validate frame shape
                    if frame.ndim < 2 or frame.shape[0] == 0 or frame.shape[1] == 0:
                        last_error = f"Invalid frame shape {frame.shape} on attempt {attempt}"
                        logger.warning(last_error)
                        if attempt < config.max_retries:
                            time.sleep(config.retry_delay)
                        continue

                    # Check for blank frame
                    if self._is_blank(frame):
                        last_error = f"Blank frame detected on attempt {attempt}"
                        logger.warning(last_error)
                        if attempt < config.max_retries:
                            time.sleep(config.retry_delay)
                        continue

                    self._record_successful_frame(frame)
                    h, w = frame.shape[:2]
                    return CaptureResult(
                        success=True,
                        image=frame,
                        device_path=self._device_path,
                        width=w,
                        height=h,
                        attempt_count=attempt,
                    )

                except Exception as exc:
                    last_error = str(exc)
                    logger.error("Capture exception on attempt %d: %s", attempt, exc)
                    if attempt < config.max_retries:
                        time.sleep(config.retry_delay)
                    else:
                        exc_msg = last_error.lower()
                        if "permission" in exc_msg:
                            exc_failure_type: str = FailureType.PERMISSION
                        elif "no such file" in exc_msg or "no such device" in exc_msg:
                            exc_failure_type = FailureType.DEVICE_GONE
                        else:
                            exc_failure_type = FailureType.TRANSIENT
                        return CaptureResult(
                            success=False,
                            device_path=self._device_path,
                            error=f"Capture failed after {config.max_retries} attempts: {last_error}",
                            failure_type=exc_failure_type,
                            attempt_count=config.max_retries,
                        )

            return CaptureResult(
                success=False,
                device_path=self._device_path,
                error=f"Capture failed after {config.max_retries} attempts: {last_error}",
                failure_type=FailureType.TRANSIENT,
                attempt_count=config.max_retries,
            )

    def capture_to_file(self, path: str | Path, *, quality: int | None = None) -> CaptureResult:
        """Capture a frame and save it to disk."""
        cv2 = _get_cv2()
        result = self.capture()
        if not result.success:
            return result

        q = quality if quality is not None else self._config.quality
        out_path = Path(path)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        ext = out_path.suffix.lower()
        params: list[int] = []
        if ext in (".jpg", ".jpeg"):
            params = [cv2.IMWRITE_JPEG_QUALITY, q]
        elif ext == ".png":
            params = [cv2.IMWRITE_PNG_COMPRESSION, min(9, max(0, (100 - q) // 10))]

        try:
            success = cv2.imwrite(str(out_path), result.image, params)
            if not success:
                result.success = False
                result.error = f"cv2.imwrite failed for {out_path}"
        except Exception as exc:
            result.success = False
            result.error = f"Save failed: {exc}"

        return result

    def capture_burst(
        self,
        count: int = 3,
        interval: float = 0.5,
    ) -> list[CaptureResult]:
        """Capture multiple frames in rapid succession.

        Useful for motion detection, multi-angle tasks, or selecting the
        sharpest frame from a sequence.

        Parameters
        ----------
        count:
            Number of frames to capture (1-20).
        interval:
            Seconds between captures.

        Returns
        -------
        list[CaptureResult]
            One result per requested frame (some may have ``success=False``).
        """
        if count < 1:
            raise ValueError(f"count must be >= 1, got {count}")
        if count > 20:
            count = 20
            logger.warning("Burst count clamped to 20")

        results: list[CaptureResult] = []
        for i in range(count):
            result = self.capture()
            results.append(result)
            if i < count - 1 and interval > 0:
                time.sleep(interval)
        return results

    def capture_best(self, burst_count: int = 3) -> CaptureResult:
        """Capture a burst and return the sharpest frame.

        Uses Laplacian variance as a sharpness proxy.
        """
        results = self.capture_burst(count=burst_count, interval=0.2)
        successful = [r for r in results if r.success and r.image is not None]
        if not successful:
            return CaptureResult(
                success=False,
                device_path=self._device_path,
                error="No successful frames in burst",
            )

        cv2 = _get_cv2()

        def sharpness(r: CaptureResult) -> float:
            gray = cv2.cvtColor(r.image, cv2.COLOR_BGR2GRAY)
            return float(cv2.Laplacian(gray, cv2.CV_64F).var())

        return max(successful, key=sharpness)

    # -- context manager --

    def __enter__(self) -> CameraHandle:
        self.open()
        return self

    def __exit__(self, *exc_info: Any) -> None:
        self.close()

    # -- internal --

    def _warmup(self) -> None:
        """Discard initial frames for auto-exposure stabilization."""
        n = self._config.warmup_frames
        if n <= 0:
            return
        logger.debug("Warming up camera: discarding %d frames", n)
        for _ in range(n):
            try:
                self._cap.read()
            except Exception:
                break

    def _is_blank(self, frame: np.ndarray) -> bool:
        """Detect blank/black frames using adaptive or fixed threshold."""
        if self._blank_detector is not None:
            return self._blank_detector.is_blank(frame)
        mean_val = float(np.mean(frame))
        return mean_val < self._config.blank_threshold

    def _record_successful_frame(self, frame: np.ndarray) -> None:
        """Record a successful frame's intensity for adaptive blank detection."""
        if self._blank_detector is not None:
            mean_val = float(np.mean(frame))
            self._blank_detector.record_intensity(mean_val)

    @staticmethod
    def _parse_device_index(path: str) -> int | None:
        """Extract numeric index from /dev/videoN path."""
        import re

        m = re.match(r"/dev/video(\d+)$", path)
        if m:
            return int(m.group(1))
        return None


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class CaptureError(Exception):
    """Raised when camera capture fails."""


class FailureType(StrEnum):
    """Classification of capture failures for retry decision-making."""

    TRANSIENT = "transient"       # Device busy, frame read failed — may recover
    PERMISSION = "permission"     # Permission denied — won't recover by retry
    DEVICE_GONE = "device_gone"   # Device removed — needs rediscovery
    UNSUPPORTED = "unsupported"   # Format/resolution unsupported — won't recover
    UNKNOWN = "unknown"           # Unclassified failure


# ---------------------------------------------------------------------------
# Convenience
# ---------------------------------------------------------------------------


@contextmanager
def open_camera(
    device_path: str,
    config: CaptureConfig | None = None,
):
    """Context manager that opens and closes a camera device.

    Example::

        with open_camera("/dev/video0") as cam:
            result = cam.capture()
    """
    handle = CameraHandle(device_path, config)
    handle.open()
    try:
        yield handle
    finally:
        handle.close()
