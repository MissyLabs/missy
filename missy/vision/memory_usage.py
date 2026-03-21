"""Scene memory usage monitoring and limits enforcement.

Tracks how much memory the vision subsystem's scene sessions are consuming
and enforces configurable limits to prevent unbounded growth.

Frame images (numpy arrays) can be large — a single 1920x1080 BGR frame
is ~6 MB.  With 20 frames per session and 5 sessions, that's up to 600 MB.
This module provides visibility and controls.

Example::

    from missy.vision.memory_usage import MemoryTracker

    tracker = MemoryTracker(max_bytes=500_000_000)  # 500 MB
    tracker.update_from_scene_manager()
    print(tracker.report())
"""

from __future__ import annotations

import logging
import sys
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class SessionMemoryInfo:
    """Memory usage information for a single scene session."""

    task_id: str
    frame_count: int
    estimated_bytes: int
    active: bool

    @property
    def estimated_mb(self) -> float:
        return self.estimated_bytes / (1024 * 1024)


@dataclass
class MemoryReport:
    """Aggregated memory usage report for the vision subsystem."""

    total_bytes: int
    total_frames: int
    session_count: int
    active_sessions: int
    sessions: list[SessionMemoryInfo]
    limit_bytes: int
    usage_fraction: float
    over_limit: bool

    @property
    def total_mb(self) -> float:
        return self.total_bytes / (1024 * 1024)

    @property
    def limit_mb(self) -> float:
        return self.limit_bytes / (1024 * 1024)

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_bytes": self.total_bytes,
            "total_mb": round(self.total_mb, 2),
            "total_frames": self.total_frames,
            "session_count": self.session_count,
            "active_sessions": self.active_sessions,
            "limit_bytes": self.limit_bytes,
            "limit_mb": round(self.limit_mb, 2),
            "usage_fraction": round(self.usage_fraction, 4),
            "over_limit": self.over_limit,
            "sessions": [
                {
                    "task_id": s.task_id,
                    "frame_count": s.frame_count,
                    "estimated_mb": round(s.estimated_mb, 2),
                    "active": s.active,
                }
                for s in self.sessions
            ],
        }


def estimate_frame_bytes(frame: Any) -> int:
    """Estimate memory used by a numpy array frame.

    Uses ``sys.getsizeof`` plus the underlying buffer size.
    Returns 0 if the frame is None or not a numpy array.
    """
    if frame is None:
        return 0
    try:
        # numpy ndarray: nbytes gives the raw data size
        return int(frame.nbytes) + sys.getsizeof(frame)
    except (AttributeError, TypeError):
        return sys.getsizeof(frame)


class MemoryTracker:
    """Track and enforce memory limits for the vision scene memory subsystem.

    Parameters
    ----------
    max_bytes:
        Maximum allowed memory for all scene frames combined.
        Default 500 MB.
    warn_fraction:
        Fraction of max_bytes at which to log a warning.
        Default 0.8 (80%).
    """

    def __init__(
        self,
        max_bytes: int = 500_000_000,
        warn_fraction: float = 0.8,
    ) -> None:
        self._max_bytes = max(1, max_bytes)
        self._warn_fraction = warn_fraction
        self._last_report: MemoryReport | None = None

    @property
    def max_bytes(self) -> int:
        return self._max_bytes

    def compute_session_usage(self, session: Any) -> SessionMemoryInfo:
        """Compute memory usage for a single SceneSession.

        Parameters
        ----------
        session:
            A ``SceneSession`` instance (loosely typed to avoid circular imports).
        """
        total = 0
        frame_count = 0
        for frame in getattr(session, "_frames", []):
            img = getattr(frame, "image", None)
            total += estimate_frame_bytes(img)
            frame_count += 1

        return SessionMemoryInfo(
            task_id=getattr(session, "task_id", "unknown"),
            frame_count=frame_count,
            estimated_bytes=total,
            active=getattr(session, "is_active", False),
        )

    def update_from_scene_manager(self, manager: Any = None) -> MemoryReport:
        """Scan the scene manager and produce a memory report.

        Parameters
        ----------
        manager:
            A ``SceneManager`` instance.  If None, uses the module singleton.
        """
        if manager is None:
            from missy.vision.scene_memory import get_scene_manager

            manager = get_scene_manager()

        sessions_info: list[SessionMemoryInfo] = []
        total_bytes = 0
        total_frames = 0
        active_count = 0

        # Access internal sessions dict (under lock if possible)
        lock = getattr(manager, "_lock", None)
        sessions_dict = {}
        if lock:
            with lock:
                sessions_dict = dict(getattr(manager, "_sessions", {}))
        else:
            sessions_dict = dict(getattr(manager, "_sessions", {}))

        for session in sessions_dict.values():
            info = self.compute_session_usage(session)
            sessions_info.append(info)
            total_bytes += info.estimated_bytes
            total_frames += info.frame_count
            if info.active:
                active_count += 1

        usage_fraction = total_bytes / self._max_bytes if self._max_bytes > 0 else 0.0
        over_limit = total_bytes > self._max_bytes

        report = MemoryReport(
            total_bytes=total_bytes,
            total_frames=total_frames,
            session_count=len(sessions_info),
            active_sessions=active_count,
            sessions=sessions_info,
            limit_bytes=self._max_bytes,
            usage_fraction=usage_fraction,
            over_limit=over_limit,
        )

        # Warnings
        if over_limit:
            logger.warning(
                "Vision memory over limit: %.1f MB / %.1f MB (%.0f%%)",
                report.total_mb,
                report.limit_mb,
                usage_fraction * 100,
            )
        elif usage_fraction >= self._warn_fraction:
            logger.warning(
                "Vision memory usage high: %.1f MB / %.1f MB (%.0f%%)",
                report.total_mb,
                report.limit_mb,
                usage_fraction * 100,
            )

        self._last_report = report
        return report

    def should_evict(self) -> bool:
        """Check if memory usage exceeds the limit based on last report."""
        if self._last_report is None:
            return False
        return self._last_report.over_limit

    def report(self) -> MemoryReport | None:
        """Return the last computed report, or None."""
        return self._last_report


# ---------------------------------------------------------------------------
# Module singleton
# ---------------------------------------------------------------------------

_tracker: MemoryTracker | None = None


def get_memory_tracker() -> MemoryTracker:
    """Return the process-level singleton MemoryTracker."""
    global _tracker
    if _tracker is None:
        _tracker = MemoryTracker()
    return _tracker
