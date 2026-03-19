"""Vision subsystem health monitoring and statistics tracking.

Tracks camera capture success/failure rates, quality metrics over time,
and device availability history.  Provides health assessments and
diagnostic summaries for ``missy vision doctor`` and audit logging.

Example::

    from missy.vision.health_monitor import VisionHealthMonitor

    monitor = VisionHealthMonitor()
    monitor.record_capture(success=True, device="/dev/video0", quality=0.85)
    monitor.record_capture(success=False, device="/dev/video0", error="blank frame")

    report = monitor.get_health_report()
    print(report["overall_status"])  # "healthy" / "degraded" / "unhealthy"
"""

from __future__ import annotations

import logging
import threading
import time
from collections import deque
from dataclasses import dataclass
from enum import StrEnum

logger = logging.getLogger(__name__)


class HealthStatus(StrEnum):
    """Overall vision subsystem health status."""

    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


@dataclass
class CaptureEvent:
    """A single capture attempt record."""

    timestamp: float
    success: bool
    device: str = ""
    quality_score: float = 0.0
    error: str = ""
    latency_ms: float = 0.0
    source_type: str = "webcam"  # webcam, file, screenshot


@dataclass
class DeviceStats:
    """Aggregated statistics for a single device."""

    device: str
    total_captures: int = 0
    successful_captures: int = 0
    failed_captures: int = 0
    total_quality: float = 0.0
    total_latency_ms: float = 0.0
    last_seen: float = 0.0
    last_error: str = ""
    consecutive_failures: int = 0

    @property
    def success_rate(self) -> float:
        """Capture success rate as a fraction (0.0 to 1.0)."""
        if self.total_captures == 0:
            return 0.0
        return self.successful_captures / self.total_captures

    @property
    def average_quality(self) -> float:
        """Average quality score across successful captures."""
        if self.successful_captures == 0:
            return 0.0
        return self.total_quality / self.successful_captures

    @property
    def average_latency_ms(self) -> float:
        """Average capture latency in milliseconds."""
        if self.total_captures == 0:
            return 0.0
        return self.total_latency_ms / self.total_captures


# ---------------------------------------------------------------------------
# Thresholds
# ---------------------------------------------------------------------------

#: Success rate below which the device is considered degraded.
_DEGRADED_THRESHOLD = 0.8

#: Success rate below which the device is considered unhealthy.
_UNHEALTHY_THRESHOLD = 0.5

#: Consecutive failures before marking device unhealthy.
_CONSECUTIVE_FAILURE_LIMIT = 5

#: Quality score below which to warn.
_LOW_QUALITY_THRESHOLD = 0.4

#: Maximum events to retain in the rolling window.
_MAX_EVENTS = 1000

#: Time window for recent health assessment (seconds).
_RECENT_WINDOW_SECS = 300  # 5 minutes


class VisionHealthMonitor:
    """Track vision capture health and produce diagnostic reports.

    Thread-safe: all mutations are guarded by an internal lock.

    Args:
        max_events: Maximum capture events to retain.  Oldest are
            dropped when this limit is exceeded.
        recent_window_secs: Time window in seconds for "recent" health
            assessment.
    """

    def __init__(
        self,
        max_events: int = _MAX_EVENTS,
        recent_window_secs: float = _RECENT_WINDOW_SECS,
    ) -> None:
        self._max_events = max(1, max_events)
        self._recent_window = recent_window_secs
        self._events: deque[CaptureEvent] = deque(maxlen=self._max_events)
        self._devices: dict[str, DeviceStats] = {}
        self._lock = threading.Lock()
        self._start_time = time.monotonic()

    # ------------------------------------------------------------------
    # Record events
    # ------------------------------------------------------------------

    def record_capture(
        self,
        success: bool,
        device: str = "",
        quality_score: float = 0.0,
        error: str = "",
        latency_ms: float = 0.0,
        source_type: str = "webcam",
    ) -> None:
        """Record a capture attempt.

        Args:
            success: Whether the capture succeeded.
            device: Device path (e.g. ``"/dev/video0"``).
            quality_score: Quality metric from the pipeline (0.0 to 1.0).
            error: Error description if the capture failed.
            latency_ms: Time taken for the capture in milliseconds.
            source_type: Source type identifier.
        """
        event = CaptureEvent(
            timestamp=time.time(),
            success=success,
            device=device,
            quality_score=quality_score,
            error=error,
            latency_ms=latency_ms,
            source_type=source_type,
        )

        with self._lock:
            self._events.append(event)
            stats = self._get_or_create_device(device)
            stats.total_captures += 1
            stats.last_seen = event.timestamp
            stats.total_latency_ms += latency_ms

            if success:
                stats.successful_captures += 1
                stats.total_quality += quality_score
                stats.consecutive_failures = 0
            else:
                stats.failed_captures += 1
                stats.consecutive_failures += 1
                stats.last_error = error

                if stats.consecutive_failures >= _CONSECUTIVE_FAILURE_LIMIT:
                    logger.warning(
                        "Device %s has %d consecutive failures: %s",
                        device,
                        stats.consecutive_failures,
                        error,
                    )

    def record_device_discovery(self, device: str) -> None:
        """Record that a device was discovered (even without capture).

        Args:
            device: Device path.
        """
        with self._lock:
            stats = self._get_or_create_device(device)
            stats.last_seen = time.time()

    # ------------------------------------------------------------------
    # Health assessment
    # ------------------------------------------------------------------

    def get_device_health(self, device: str) -> HealthStatus:
        """Assess the health of a specific device.

        Args:
            device: Device path.

        Returns:
            Health status enum value.
        """
        with self._lock:
            stats = self._devices.get(device)
            if stats is None:
                return HealthStatus.UNKNOWN
            return self._assess_device(stats)

    def get_overall_health(self) -> HealthStatus:
        """Assess the overall vision subsystem health.

        Returns:
            The worst health status among all known devices, or
            ``UNKNOWN`` if no devices have been recorded.
        """
        with self._lock:
            if not self._devices:
                return HealthStatus.UNKNOWN

            statuses = [self._assess_device(s) for s in self._devices.values()]

            if HealthStatus.UNHEALTHY in statuses:
                return HealthStatus.UNHEALTHY
            if HealthStatus.DEGRADED in statuses:
                return HealthStatus.DEGRADED
            if HealthStatus.UNKNOWN in statuses and len(statuses) == statuses.count(
                HealthStatus.UNKNOWN
            ):
                return HealthStatus.UNKNOWN
            return HealthStatus.HEALTHY

    def get_health_report(self) -> dict:
        """Generate a comprehensive health report.

        Returns:
            A dict suitable for JSON serialization with keys:
            ``overall_status``, ``total_captures``, ``total_failures``,
            ``recent_success_rate``, ``devices``, ``uptime_seconds``,
            ``warnings``.
        """
        with self._lock:
            total = len(self._events)
            failures = sum(1 for e in self._events if not e.success)
            recent = self._get_recent_events()
            recent_success = (
                sum(1 for e in recent if e.success) / len(recent)
                if recent
                else 0.0
            )

            warnings = self._collect_warnings()
            device_reports = {}
            for device, stats in self._devices.items():
                device_reports[device] = {
                    "total_captures": stats.total_captures,
                    "success_rate": round(stats.success_rate, 4),
                    "average_quality": round(stats.average_quality, 4),
                    "average_latency_ms": round(stats.average_latency_ms, 2),
                    "consecutive_failures": stats.consecutive_failures,
                    "last_error": stats.last_error,
                    "status": self._assess_device(stats).value,
                }

            overall = HealthStatus.UNKNOWN
            if self._devices:
                statuses = [self._assess_device(s) for s in self._devices.values()]
                if HealthStatus.UNHEALTHY in statuses:
                    overall = HealthStatus.UNHEALTHY
                elif HealthStatus.DEGRADED in statuses:
                    overall = HealthStatus.DEGRADED
                elif any(s == HealthStatus.HEALTHY for s in statuses):
                    overall = HealthStatus.HEALTHY

            return {
                "overall_status": overall.value,
                "total_captures": total,
                "total_failures": failures,
                "recent_success_rate": round(recent_success, 4),
                "devices": device_reports,
                "uptime_seconds": round(time.monotonic() - self._start_time, 1),
                "warnings": warnings,
            }

    def get_device_stats(self, device: str) -> DeviceStats | None:
        """Return raw stats for a device, or None if unknown."""
        with self._lock:
            stats = self._devices.get(device)
            if stats is None:
                return None
            # Return a copy
            return DeviceStats(
                device=stats.device,
                total_captures=stats.total_captures,
                successful_captures=stats.successful_captures,
                failed_captures=stats.failed_captures,
                total_quality=stats.total_quality,
                total_latency_ms=stats.total_latency_ms,
                last_seen=stats.last_seen,
                last_error=stats.last_error,
                consecutive_failures=stats.consecutive_failures,
            )

    def get_recommendations(self) -> list[str]:
        """Generate actionable recommendations based on collected statistics.

        Returns:
            A list of recommendation strings.  Empty when no issues are
            detected or when insufficient data has been collected.
        """
        with self._lock:
            recs: list[str] = []

            for device, stats in self._devices.items():
                if stats.total_captures < 3:
                    continue  # not enough data

                # Consecutive failure recommendations
                if stats.consecutive_failures >= _CONSECUTIVE_FAILURE_LIMIT:
                    if "permission" in stats.last_error.lower():
                        recs.append(
                            f"Device {device}: add user to 'video' group "
                            f"(sudo usermod -aG video $USER) and re-login"
                        )
                    elif "busy" in stats.last_error.lower():
                        recs.append(
                            f"Device {device}: close other applications "
                            f"using the camera (lsof {device})"
                        )
                    else:
                        recs.append(
                            f"Device {device}: check physical connection "
                            f"and try 'missy vision doctor' for diagnostics"
                        )

                # High latency recommendation
                if stats.average_latency_ms > 2000 and stats.total_captures >= 5:
                    recs.append(
                        f"Device {device}: high average latency "
                        f"({stats.average_latency_ms:.0f}ms) — consider "
                        f"reducing capture resolution"
                    )

                # Low quality recommendation
                if (
                    stats.average_quality > 0
                    and stats.average_quality < _LOW_QUALITY_THRESHOLD
                    and stats.successful_captures >= 5
                ):
                    recs.append(
                        f"Device {device}: low image quality — check "
                        f"lighting conditions and camera lens cleanliness"
                    )

            return recs

    def reset(self) -> None:
        """Clear all recorded data."""
        with self._lock:
            self._events.clear()
            self._devices.clear()
            self._start_time = time.monotonic()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _get_or_create_device(self, device: str) -> DeviceStats:
        """Get or create device stats.  Caller must hold the lock."""
        if device not in self._devices:
            self._devices[device] = DeviceStats(device=device)
        return self._devices[device]

    def _assess_device(self, stats: DeviceStats) -> HealthStatus:
        """Assess a single device's health.  Caller must hold the lock."""
        if stats.total_captures == 0:
            return HealthStatus.UNKNOWN

        if stats.consecutive_failures >= _CONSECUTIVE_FAILURE_LIMIT:
            return HealthStatus.UNHEALTHY

        rate = stats.success_rate
        if rate < _UNHEALTHY_THRESHOLD:
            return HealthStatus.UNHEALTHY
        if rate < _DEGRADED_THRESHOLD:
            return HealthStatus.DEGRADED
        return HealthStatus.HEALTHY

    def _get_recent_events(self) -> list[CaptureEvent]:
        """Return events from the last ``_recent_window`` seconds.

        Caller must hold the lock.
        """
        cutoff = time.time() - self._recent_window
        return [e for e in self._events if e.timestamp >= cutoff]

    def _collect_warnings(self) -> list[str]:
        """Collect diagnostic warnings.  Caller must hold the lock."""
        warnings: list[str] = []

        for device, stats in self._devices.items():
            if stats.consecutive_failures >= _CONSECUTIVE_FAILURE_LIMIT:
                warnings.append(
                    f"Device {device}: {stats.consecutive_failures} consecutive "
                    f"failures (last: {stats.last_error})"
                )
            elif stats.success_rate < _DEGRADED_THRESHOLD and stats.total_captures >= 5:
                warnings.append(
                    f"Device {device}: low success rate "
                    f"({stats.success_rate:.0%}, {stats.total_captures} captures)"
                )

            if (
                stats.average_quality > 0
                and stats.average_quality < _LOW_QUALITY_THRESHOLD
                and stats.successful_captures >= 3
            ):
                warnings.append(
                    f"Device {device}: low average quality "
                    f"({stats.average_quality:.2f})"
                )

        return warnings


# ---------------------------------------------------------------------------
# Module singleton
# ---------------------------------------------------------------------------

_monitor: VisionHealthMonitor | None = None


def get_health_monitor() -> VisionHealthMonitor:
    """Return the process-level singleton VisionHealthMonitor."""
    global _monitor
    if _monitor is None:
        _monitor = VisionHealthMonitor()
    return _monitor
