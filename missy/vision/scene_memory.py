"""Task-scoped scene memory for multi-step visual tasks.

Maintains a rolling window of captured frames and analysis results during
an active visual task (e.g. puzzle solving, painting review).  Provides
context continuity so the agent can reference earlier observations.

Design
------
- Each ``SceneSession`` tracks frames, analysis notes, and state transitions
  for a single visual task.
- Sessions are identified by a task ID and have a configurable max frame count.
- Scene comparisons detect significant changes between frames.
- Memory is in-process only (not persisted to disk) for privacy; a summary
  can be exported for audit logging.
"""

from __future__ import annotations

import hashlib
import logging
import threading
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------


class TaskType(str, Enum):
    PUZZLE = "puzzle"
    PAINTING = "painting"
    GENERAL = "general"
    INSPECTION = "inspection"


@dataclass
class SceneFrame:
    """A single captured frame within a scene session."""

    frame_id: int
    image: np.ndarray
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))
    source: str = ""  # e.g. "webcam:/dev/video0"
    analysis: dict[str, Any] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)
    thumbnail_hash: str = ""  # for change detection

    def __post_init__(self) -> None:
        if not self.thumbnail_hash and self.image is not None:
            self.thumbnail_hash = self._compute_hash(self.image)

    @staticmethod
    def _compute_hash(image: np.ndarray) -> str:
        """Compute a perceptual hash of a downscaled version for change detection."""
        try:
            import cv2
            small = cv2.resize(image, (16, 16))
            gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
            return hashlib.md5(gray.tobytes()).hexdigest()[:12]
        except Exception:
            try:
                return hashlib.md5(image.tobytes()[:1024]).hexdigest()[:12]
            except Exception:
                return "unknown_hash"


@dataclass
class SceneChange:
    """Describes a detected change between two frames."""

    from_frame: int
    to_frame: int
    change_score: float  # 0.0 = identical, 1.0 = completely different
    description: str = ""
    regions: list[dict[str, Any]] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Scene session
# ---------------------------------------------------------------------------


class SceneSession:
    """Manages a task-scoped sequence of visual observations.

    Parameters
    ----------
    task_id:
        Unique identifier for this visual task.
    task_type:
        Category of task (puzzle, painting, general, inspection).
    max_frames:
        Maximum frames to retain.  Oldest are evicted on overflow.
    """

    def __init__(
        self,
        task_id: str,
        task_type: TaskType = TaskType.GENERAL,
        max_frames: int = 20,
    ) -> None:
        self.task_id = task_id
        self.task_type = task_type
        self.max_frames = max_frames
        self._frames: list[SceneFrame] = []
        self._frame_counter = 0
        self._created = datetime.now(UTC)
        self._state: dict[str, Any] = {}  # task-specific state
        self._observations: list[str] = []  # accumulated observations
        self._active = True

    @property
    def frame_count(self) -> int:
        return len(self._frames)

    @property
    def is_active(self) -> bool:
        return self._active

    @property
    def state(self) -> dict[str, Any]:
        return dict(self._state)

    @property
    def observations(self) -> list[str]:
        return list(self._observations)

    def add_frame(
        self,
        image: np.ndarray,
        source: str = "",
        analysis: dict[str, Any] | None = None,
        notes: list[str] | None = None,
    ) -> SceneFrame:
        """Add a new frame to the session."""
        self._frame_counter += 1
        frame = SceneFrame(
            frame_id=self._frame_counter,
            image=image,
            source=source,
            analysis=analysis or {},
            notes=notes or [],
        )

        self._frames.append(frame)

        # Evict oldest if over limit
        while len(self._frames) > self.max_frames:
            evicted = self._frames.pop(0)
            logger.debug(
                "Evicted frame %d from session %s", evicted.frame_id, self.task_id
            )

        return frame

    def add_observation(self, text: str) -> None:
        """Record a textual observation about the scene."""
        self._observations.append(text)

    def update_state(self, **kwargs: Any) -> None:
        """Update task-specific state (e.g. puzzle board state)."""
        self._state.update(kwargs)

    def get_latest_frame(self) -> SceneFrame | None:
        """Return the most recent frame, or None."""
        return self._frames[-1] if self._frames else None

    def get_frame(self, frame_id: int) -> SceneFrame | None:
        """Retrieve a specific frame by ID."""
        for f in self._frames:
            if f.frame_id == frame_id:
                return f
        return None

    def get_recent_frames(self, n: int = 5) -> list[SceneFrame]:
        """Return the N most recent frames."""
        return list(self._frames[-n:])

    def detect_change(self, frame_a: SceneFrame, frame_b: SceneFrame) -> SceneChange:
        """Detect how much changed between two frames."""
        try:
            import cv2

            # Resize both to same small size for comparison
            size = (64, 64)
            a = cv2.resize(frame_a.image, size)
            b = cv2.resize(frame_b.image, size)

            # Convert to grayscale
            ga = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY).astype(np.float32)
            gb = cv2.cvtColor(b, cv2.COLOR_BGR2GRAY).astype(np.float32)

            # Normalized difference
            diff = np.abs(ga - gb)
            score = float(np.mean(diff) / 255.0)

            description = "no change"
            if score > 0.3:
                description = "major change"
            elif score > 0.15:
                description = "moderate change"
            elif score > 0.05:
                description = "minor change"

            return SceneChange(
                from_frame=frame_a.frame_id,
                to_frame=frame_b.frame_id,
                change_score=round(score, 4),
                description=description,
            )
        except Exception as exc:
            return SceneChange(
                from_frame=frame_a.frame_id,
                to_frame=frame_b.frame_id,
                change_score=-1.0,
                description=f"comparison failed: {exc}",
            )

    def detect_latest_change(self) -> SceneChange | None:
        """Compare the two most recent frames."""
        if len(self._frames) < 2:
            return None
        return self.detect_change(self._frames[-2], self._frames[-1])

    def visualize_change(
        self,
        frame_a: SceneFrame,
        frame_b: SceneFrame,
    ) -> np.ndarray | None:
        """Generate a visual diff image highlighting changes between frames.

        Returns a BGR image where changed regions are highlighted in red,
        or None if comparison fails.
        """
        try:
            import cv2

            size = (256, 256)
            a = cv2.resize(frame_a.image, size)
            b = cv2.resize(frame_b.image, size)

            ga = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY)
            gb = cv2.cvtColor(b, cv2.COLOR_BGR2GRAY)

            diff = cv2.absdiff(ga, gb)
            _, thresh = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)

            # Create overlay: original frame B with red highlights on changes
            overlay = b.copy()
            overlay[thresh > 0] = [0, 0, 255]  # red highlight
            blended = cv2.addWeighted(b, 0.6, overlay, 0.4, 0)

            return blended
        except Exception as exc:
            logger.warning("Failed to visualize change: %s", exc)
            return None

    def close(self) -> None:
        """Mark session as inactive and release frame data."""
        self._active = False
        frame_count = self._frame_counter
        # Fully release frame data and references
        self._frames.clear()
        self._observations.clear()
        self._state.clear()
        logger.info("Scene session %s closed (%d frames)", self.task_id, frame_count)

    def summarize(self) -> dict[str, Any]:
        """Produce a serializable summary for audit logging."""
        return {
            "task_id": self.task_id,
            "task_type": self.task_type.value,
            "created": self._created.isoformat(),
            "frame_count": self._frame_counter,
            "frames_retained": len(self._frames),
            "observations": list(self._observations),
            "state": dict(self._state),
            "active": self._active,
        }


# ---------------------------------------------------------------------------
# Scene manager (process-level singleton)
# ---------------------------------------------------------------------------


class SceneManager:
    """Manages multiple concurrent scene sessions.  Thread-safe."""

    def __init__(self, max_sessions: int = 5) -> None:
        self._sessions: dict[str, SceneSession] = {}
        self._max_sessions = max_sessions
        self._lock = threading.Lock()

    def create_session(
        self,
        task_id: str,
        task_type: TaskType = TaskType.GENERAL,
        max_frames: int = 20,
    ) -> SceneSession:
        """Create a new scene session, evicting oldest if at capacity."""
        with self._lock:
            # Evict oldest inactive session if at capacity
            if len(self._sessions) >= self._max_sessions:
                self._evict_oldest()

            session = SceneSession(task_id, task_type, max_frames)
            self._sessions[task_id] = session
            logger.info("Created scene session: %s (%s)", task_id, task_type.value)
            return session

    def get_session(self, task_id: str) -> SceneSession | None:
        with self._lock:
            return self._sessions.get(task_id)

    def get_active_session(self) -> SceneSession | None:
        """Return the most recently created active session."""
        with self._lock:
            for session in reversed(list(self._sessions.values())):
                if session.is_active:
                    return session
            return None

    def close_session(self, task_id: str) -> None:
        """Close a session by task ID."""
        with self._lock:
            session = self._sessions.get(task_id)
            if session:
                session.close()

    def close_all(self) -> None:
        with self._lock:
            for session in self._sessions.values():
                if session.is_active:
                    session.close()

    def list_sessions(self) -> list[dict[str, Any]]:
        with self._lock:
            return [s.summarize() for s in self._sessions.values()]

    def _evict_oldest(self) -> None:
        """Remove the oldest inactive session, or oldest overall.

        Caller must hold ``self._lock``.
        """
        # Prefer to evict inactive sessions
        for task_id, session in list(self._sessions.items()):
            if not session.is_active:
                del self._sessions[task_id]
                return

        # All active — evict the oldest by creation time
        if self._sessions:
            oldest = min(
                self._sessions,
                key=lambda tid: self._sessions[tid]._created,
            )
            self._sessions[oldest].close()
            del self._sessions[oldest]


# ---------------------------------------------------------------------------
# Module singleton
# ---------------------------------------------------------------------------

_scene_manager: SceneManager | None = None


def get_scene_manager() -> SceneManager:
    global _scene_manager
    if _scene_manager is None:
        _scene_manager = SceneManager()
    return _scene_manager
