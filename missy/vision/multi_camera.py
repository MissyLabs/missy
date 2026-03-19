"""Multi-camera concurrent capture.

Manages simultaneous capture from multiple USB cameras using thread-based
parallelism.  Each camera gets its own :class:`CameraHandle` and capture
thread, with results collected and correlated by timestamp.

Use cases:
- Stereo or multi-angle capture for 3D puzzle assistance
- Monitoring multiple work areas simultaneously
- Comparing views from different camera models
- Fallback capture: try multiple cameras, use best result
"""

from __future__ import annotations

import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any

from missy.vision.capture import CameraHandle, CaptureConfig, CaptureResult
from missy.vision.discovery import CameraDevice, get_discovery
from missy.vision.health_monitor import get_health_monitor

logger = logging.getLogger(__name__)


@dataclass
class MultiCaptureResult:
    """Result of a concurrent multi-camera capture."""

    results: dict[str, CaptureResult] = field(default_factory=dict)
    elapsed_ms: float = 0.0
    errors: dict[str, str] = field(default_factory=dict)

    @property
    def successful_devices(self) -> list[str]:
        """Device paths that produced successful captures."""
        return [d for d, r in self.results.items() if r.success]

    @property
    def failed_devices(self) -> list[str]:
        """Device paths that failed."""
        return [d for d, r in self.results.items() if not r.success]

    @property
    def all_succeeded(self) -> bool:
        return bool(self.results) and all(r.success for r in self.results.values())

    @property
    def any_succeeded(self) -> bool:
        return any(r.success for r in self.results.values())

    @property
    def best_result(self) -> CaptureResult | None:
        """Return the successful result with the largest image, or None."""
        successful = [r for r in self.results.values() if r.success and r.image is not None]
        if not successful:
            return None
        return max(successful, key=lambda r: r.width * r.height)


class MultiCameraManager:
    """Manage and capture from multiple cameras concurrently.

    Parameters
    ----------
    config:
        Default capture configuration applied to all cameras.
    max_workers:
        Maximum parallel capture threads.
    open_timeout:
        Seconds to wait for each camera to open.
    """

    def __init__(
        self,
        config: CaptureConfig | None = None,
        max_workers: int = 4,
        open_timeout: float = 10.0,
    ) -> None:
        self._config = config or CaptureConfig()
        self._max_workers = max(1, min(max_workers, 8))
        self._open_timeout = open_timeout
        self._handles: dict[str, CameraHandle] = {}
        self._devices: dict[str, CameraDevice] = {}
        self._lock = threading.Lock()

    @property
    def connected_devices(self) -> list[str]:
        """Return device paths of currently open cameras."""
        with self._lock:
            return [p for p, h in self._handles.items() if h.is_open]

    @property
    def device_count(self) -> int:
        with self._lock:
            return len(self._handles)

    def discover_and_connect(
        self,
        max_cameras: int = 4,
        vendor_filter: str = "",
    ) -> list[str]:
        """Discover cameras and connect to up to *max_cameras*.

        Parameters
        ----------
        max_cameras:
            Maximum number of cameras to open.
        vendor_filter:
            If non-empty, only connect to cameras with this vendor ID.

        Returns
        -------
        list[str]
            Device paths of successfully opened cameras.
        """
        discovery = get_discovery()
        devices = discovery.discover(force=True)

        if vendor_filter:
            devices = [d for d in devices if d.vendor_id == vendor_filter]

        connected: list[str] = []
        for device in devices[:max_cameras]:
            try:
                self.add_camera(device)
                connected.append(device.device_path)
            except Exception as exc:
                logger.warning(
                    "Failed to open %s: %s", device.device_path, exc
                )

        return connected

    def add_camera(
        self,
        device: CameraDevice,
        config: CaptureConfig | None = None,
    ) -> None:
        """Add and open a camera.

        Raises
        ------
        ValueError
            If the device is already added.
        """
        with self._lock:
            path = device.device_path
            if path in self._handles:
                raise ValueError(f"Camera {path} already added")

            handle = CameraHandle(path, config or self._config)
            handle.open()
            self._handles[path] = handle
            self._devices[path] = device
            logger.info("Multi-camera: added %s (%s)", path, device.name)

    def remove_camera(self, device_path: str) -> None:
        """Close and remove a camera."""
        with self._lock:
            handle = self._handles.pop(device_path, None)
            self._devices.pop(device_path, None)
            if handle:
                handle.close()
                logger.info("Multi-camera: removed %s", device_path)

    def capture_all(self, timeout: float = 15.0) -> MultiCaptureResult:
        """Capture from all connected cameras concurrently.

        Parameters
        ----------
        timeout:
            Maximum seconds to wait for all captures.

        Returns
        -------
        MultiCaptureResult
            Aggregated results from all cameras.
        """
        with self._lock:
            cameras = dict(self._handles)

        if not cameras:
            return MultiCaptureResult(errors={"_global": "No cameras connected"})

        t0 = time.monotonic()
        results: dict[str, CaptureResult] = {}
        errors: dict[str, str] = {}

        def _capture_one(path: str, handle: CameraHandle) -> tuple[str, CaptureResult]:
            try:
                # Guard against handle closed by concurrent remove_camera()
                if not handle.is_open:
                    return path, CaptureResult(
                        success=False,
                        device_path=path,
                        error="Camera handle closed before capture",
                    )
                result = handle.capture()
                latency = (time.monotonic() - t0) * 1000
                get_health_monitor().record_capture(
                    success=result.success,
                    device=path,
                    latency_ms=latency,
                    error=result.error or "",
                )
                return path, result
            except Exception as exc:
                logger.error("Multi-capture error on %s: %s", path, exc)
                return path, CaptureResult(
                    success=False,
                    device_path=path,
                    error=str(exc),
                )

        deadline = time.monotonic() + timeout
        with ThreadPoolExecutor(max_workers=self._max_workers) as pool:
            futures = {
                pool.submit(_capture_one, path, handle): path
                for path, handle in cameras.items()
            }

            for future in as_completed(futures, timeout=timeout):
                path = futures[future]
                try:
                    # Use remaining time as the per-future timeout to
                    # avoid applying the full timeout twice.
                    remaining = max(0.1, deadline - time.monotonic())
                    dev_path, result = future.result(timeout=remaining)
                    results[dev_path] = result
                    if not result.success:
                        errors[dev_path] = result.error
                except Exception as exc:
                    errors[path] = str(exc)
                    results[path] = CaptureResult(
                        success=False,
                        device_path=path,
                        error=str(exc),
                    )

        elapsed = (time.monotonic() - t0) * 1000
        return MultiCaptureResult(
            results=results,
            elapsed_ms=round(elapsed, 2),
            errors=errors,
        )

    def capture_best(self) -> CaptureResult:
        """Capture from all cameras and return the best result.

        "Best" = successful capture with the highest resolution.
        Falls back to the first successful result.
        """
        multi = self.capture_all()
        best = multi.best_result
        if best is not None:
            return best
        return CaptureResult(
            success=False,
            error="No successful captures from any camera",
        )

    def close_all(self) -> None:
        """Close all camera handles."""
        with self._lock:
            for path, handle in self._handles.items():
                try:
                    handle.close()
                except Exception as exc:
                    logger.warning("Error closing %s: %s", path, exc)
            self._handles.clear()
            self._devices.clear()

    def status(self) -> dict[str, Any]:
        """Return status info for all managed cameras."""
        with self._lock:
            cameras: dict[str, dict[str, Any]] = {}
            for path, handle in self._handles.items():
                device = self._devices.get(path)
                cameras[path] = {
                    "name": device.name if device else "",
                    "is_open": handle.is_open,
                    "vendor_id": device.vendor_id if device else "",
                    "product_id": device.product_id if device else "",
                }
            return {
                "camera_count": len(self._handles),
                "cameras": cameras,
            }

    # -- context manager --

    def __enter__(self) -> MultiCameraManager:
        return self

    def __exit__(self, *exc_info: Any) -> None:
        self.close_all()
