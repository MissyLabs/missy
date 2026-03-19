"""Resilient camera capture with automatic reconnection.

Wraps :class:`~missy.vision.capture.CameraHandle` with device monitoring
and automatic reconnection when the camera disconnects and reconnects.
Handles:
- Device path changes after re-enumeration
- Temporary disconnects with configurable retry
- Camera warm-up after reconnection
"""

from __future__ import annotations

import logging
import random
import time
from typing import Any

from missy.vision.capture import (
    CameraHandle,
    CaptureConfig,
    CaptureError,
    CaptureResult,
    FailureType,
)
from missy.vision.discovery import CameraDevice, get_discovery
from missy.vision.health_monitor import get_health_monitor

logger = logging.getLogger(__name__)


class ResilientCamera:
    """Camera wrapper with automatic device rediscovery and reconnection.

    Parameters
    ----------
    preferred_vendor_id:
        USB vendor ID to prefer (e.g. "046d" for Logitech).
    preferred_product_id:
        USB product ID to prefer (e.g. "085c" for C922x).
    config:
        Capture configuration.
    max_reconnect_attempts:
        Maximum reconnection attempts before giving up.
    reconnect_delay:
        Seconds between reconnection attempts.
    """

    def __init__(
        self,
        preferred_vendor_id: str = "",
        preferred_product_id: str = "",
        config: CaptureConfig | None = None,
        max_reconnect_attempts: int = 5,
        reconnect_delay: float = 2.0,
        backoff_factor: float = 1.5,
        max_delay: float = 30.0,
    ) -> None:
        self._vendor_id = preferred_vendor_id
        self._product_id = preferred_product_id
        self._config = config or CaptureConfig()
        self._max_reconnect = max_reconnect_attempts
        self._reconnect_delay = reconnect_delay
        self._backoff_factor = backoff_factor
        self._max_delay = max_delay

        self._handle: CameraHandle | None = None
        self._current_device: CameraDevice | None = None
        self._connected = False
        self._total_reconnects = 0
        self._cumulative_failures = 0

    _CUMULATIVE_FAILURE_THRESHOLD = 10

    @property
    def is_connected(self) -> bool:
        return self._connected and self._handle is not None and self._handle.is_open

    @property
    def current_device(self) -> CameraDevice | None:
        return self._current_device

    @property
    def total_reconnects(self) -> int:
        return self._total_reconnects

    @property
    def cumulative_failures(self) -> int:
        return self._cumulative_failures

    def connect(self) -> None:
        """Connect to the preferred camera, discovering it if needed."""
        device = self._discover_camera()
        if device is None:
            raise CaptureError("No camera available")

        self._open_device(device)

    def capture(self) -> CaptureResult:
        """Capture a frame, reconnecting if necessary."""
        # Validate that the device is still present before attempting capture
        if self.is_connected and self._current_device is not None:
            discovery = get_discovery()
            if not discovery.validate_device(self._current_device):
                logger.warning(
                    "Device %s is no longer valid, will reconnect",
                    self._current_device.device_path,
                )
                self.disconnect()

        if not self.is_connected:
            try:
                self.connect()
            except CaptureError:
                return CaptureResult(
                    success=False,
                    error="Camera not connected and reconnection failed",
                )

        # Try capture
        t0 = time.monotonic()
        try:
            if self._handle is None:
                raise CaptureError("Camera handle is None after connect")
            result = self._handle.capture()
            if result.success:
                latency = (time.monotonic() - t0) * 1000
                get_health_monitor().record_capture(
                    success=True,
                    device=result.device_path or (self._current_device.device_path if self._current_device else ""),
                    latency_ms=latency,
                )
                return result
            # Non-exception capture failure — check failure type before reconnecting
            self._record_failure()
            device_path = self._current_device.device_path if self._current_device else ""
            get_health_monitor().record_capture(
                success=False, device=device_path, error=result.error or "capture failed",
            )
            if result.failure_type in (FailureType.PERMISSION, FailureType.UNSUPPORTED):
                logger.error(
                    "Unrecoverable capture failure (%s): %s — not retrying",
                    result.failure_type,
                    result.error,
                )
                return result
        except Exception as exc:
            logger.warning("Capture failed: %s", exc)
            self._record_failure()
            device_path = self._current_device.device_path if self._current_device else ""
            get_health_monitor().record_capture(
                success=False, device=device_path, error=str(exc),
            )

        # Capture failed — attempt reconnection
        return self._reconnect_and_capture()

    def disconnect(self) -> None:
        """Disconnect from the camera."""
        if self._handle is not None:
            self._handle.close()
            self._handle = None
        self._connected = False

    # -- context manager --

    def __enter__(self) -> ResilientCamera:
        self.connect()
        return self

    def __exit__(self, *exc_info: Any) -> None:
        self.disconnect()

    # -- internal --

    def _record_failure(self) -> None:
        """Increment the cumulative failure counter and warn when threshold is crossed."""
        self._cumulative_failures += 1
        if self._cumulative_failures >= self._CUMULATIVE_FAILURE_THRESHOLD:
            logger.warning(
                "Camera may be unreliable: %d cumulative failures",
                self._cumulative_failures,
            )

    def _discover_camera(self) -> CameraDevice | None:
        """Find the target camera."""
        discovery = get_discovery()

        # Try specific USB ID first
        if self._vendor_id and self._product_id:
            device = discovery.find_by_usb_id(self._vendor_id, self._product_id)
            if device:
                return device

        # Fall back to preferred
        return discovery.find_preferred()

    def _open_device(self, device: CameraDevice) -> None:
        """Open a specific camera device.

        Resets the adaptive blank detector when switching to a different
        physical device, since ambient light / sensor characteristics differ.
        """
        old_handle = self._handle
        if old_handle is not None:
            old_handle.close()

        self._handle = CameraHandle(device.device_path, self._config)
        self._handle.open()
        self._current_device = device
        self._connected = True

        # Reset blank detector when switching devices — calibration from the
        # old camera does not apply to the new one.
        self._handle.reset_blank_detector()
        logger.debug("Reset blank detector for new device %s", device.device_path)

        logger.info("Connected to camera: %s at %s", device.name, device.device_path)

    def _reconnect_and_capture(self) -> CaptureResult:
        """Attempt to reconnect to the camera and capture.

        Uses exponential backoff between attempts to avoid hammering
        a device that may be temporarily unavailable.  When vendor/product
        IDs are known, uses ``rediscover_device`` for faster path-change
        detection.
        """
        self.disconnect()
        delay = self._reconnect_delay

        for attempt in range(1, self._max_reconnect + 1):
            logger.info(
                "Reconnection attempt %d/%d (delay %.1fs)",
                attempt,
                self._max_reconnect,
                delay,
            )

            # Use targeted rediscovery when we know the USB IDs
            discovery = get_discovery()
            old_path = self._current_device.device_path if self._current_device else ""
            if self._vendor_id and self._product_id:
                device = discovery.rediscover_device(
                    self._vendor_id,
                    self._product_id,
                    old_path=old_path,
                    max_attempts=1,  # one attempt per outer loop iteration
                )
            else:
                logger.debug("No USB IDs configured; using generic camera discovery")
                discovery.discover(force=True)
                device = self._discover_camera()
            if device is None:
                logger.warning("Camera not found on attempt %d", attempt)
                self._record_failure()
                # Add jitter (±25%) to prevent thundering herd on reconnect
                jittered = delay * (0.75 + random.random() * 0.5)  # noqa: S311
                time.sleep(jittered)
                delay = min(delay * self._backoff_factor, self._max_delay)
                continue

            # Warn when the discovered device differs from what we had before
            if self._current_device is not None:
                old_path = self._current_device.device_path
                if device.device_path != old_path:
                    logger.warning(
                        "Camera path changed: %s → %s",
                        old_path,
                        device.device_path,
                    )

            # Warn when we got a different USB ID than requested
            if (
                self._vendor_id
                and self._product_id
                and (device.vendor_id != self._vendor_id or device.product_id != self._product_id)
            ):
                logger.warning(
                    "Connected to fallback camera: %s (not preferred device)",
                    device.name,
                )

            try:
                self._open_device(device)
                result = self._handle.capture()
                if result.success:
                    self._total_reconnects += 1
                    logger.info(
                        "Reconnected successfully on attempt %d (total reconnects: %d)",
                        attempt,
                        self._total_reconnects,
                    )
                    return result
                # Non-exception failure from the reconnect capture attempt
                self._record_failure()
                if result.failure_type in (FailureType.PERMISSION, FailureType.UNSUPPORTED):
                    logger.error(
                        "Unrecoverable failure during reconnection (%s): %s — aborting",
                        result.failure_type,
                        result.error,
                    )
                    return result
            except Exception as exc:
                logger.warning("Reconnection attempt %d failed: %s", attempt, exc)
                self._record_failure()
                self.disconnect()

            jittered = delay * (0.75 + random.random() * 0.5)  # noqa: S311
            time.sleep(jittered)
            delay = min(delay * self._backoff_factor, self._max_delay)

        return CaptureResult(
            success=False,
            error=f"Failed to reconnect after {self._max_reconnect} attempts",
        )
