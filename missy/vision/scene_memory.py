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
from enum import StrEnum
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Change detection constants
# ---------------------------------------------------------------------------
_CHANGE_COMPARE_SIZE = (64, 64)
_CHANGE_PIXEL_WEIGHT = 0.4
_CHANGE_PHASH_WEIGHT = 0.6
_CHANGE_THRESHOLD_MAJOR = 0.3
_CHANGE_THRESHOLD_MODERATE = 0.15
_CHANGE_THRESHOLD_MINOR = 0.05
_PHASH_BITS = 64


# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------


class TaskType(StrEnum):
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
        """Compute a perceptual average hash (aHash) for change detection.

        aHash is resilient to small changes in zoom, rotation, lighting,
        and compression — much better than raw MD5 for scene comparison.
        The 64-bit hash is returned as a 16-char hex string.
        """
        return compute_phash(image)


def compute_phash(image: np.ndarray) -> str:
    """Compute a perceptual average hash (aHash) of an image.

    Produces a 64-bit hash that is resistant to minor changes in
    scale, rotation, lighting, and compression.  Two similar images
    will have a low Hamming distance between their hashes.

    Returns a 16-character hex string, or ``"unknown_hash"`` on failure.
    """
    try:
        import cv2

        # 1. Shrink to 8x8 (discards high-frequency detail)
        small = cv2.resize(image, (8, 8), interpolation=cv2.INTER_AREA)

        # 2. Convert to grayscale
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY) if small.ndim == 3 else small

        # 3. Compute mean and standard deviation
        mean_val = np.mean(gray)
        std_val = np.std(gray)

        # 4. Build 64-bit hash: each pixel above mean → 1
        # For uniform images (std ≈ 0), use raw intensity as hash seed
        if std_val < 0.5:
            # Uniform image — use the intensity value itself as the hash
            level = int(np.clip(mean_val, 0, 255))
            return f"{level:02x}" * 8  # 16 hex chars, unique per intensity
        bits = (gray.flatten() > mean_val).astype(np.uint8)

        # 5. Pack bits into bytes and hex-encode
        # bits is 64 values of 0/1
        byte_vals = np.packbits(bits)
        return byte_vals.tobytes().hex()
    except Exception:
        try:
            return hashlib.md5(image.tobytes()[:1024]).hexdigest()[:16]
        except Exception:
            return "unknown_hash"


def hamming_distance(hash_a: str, hash_b: str) -> int:
    """Compute Hamming distance between two hex-encoded perceptual hashes.

    Returns the number of differing bits.  Lower values mean more similar
    images.  Typical thresholds: <5 = very similar, 5-10 = similar,
    >10 = different.

    Returns -1 if the hashes are invalid or incomparable.
    """
    try:
        if len(hash_a) != len(hash_b):
            return -1
        a = int(hash_a, 16)
        b = int(hash_b, 16)
        xor = a ^ b
        return bin(xor).count("1")
    except (ValueError, TypeError):
        return -1


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
        self._lock = threading.Lock()

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
        deduplicate: bool = True,
        dedup_threshold: int = 5,
    ) -> SceneFrame | None:
        """Add a new frame to the session.  Thread-safe.

        Parameters
        ----------
        image:
            BGR numpy array to store.
        source:
            Source description (e.g. "webcam:/dev/video0").
        analysis:
            Analysis results dict.
        notes:
            Textual notes about this frame.
        deduplicate:
            If True, skip storing frames that are near-duplicates of the
            most recent frame (based on perceptual hash Hamming distance).
        dedup_threshold:
            Maximum Hamming distance to consider a frame a duplicate.
            Lower values require more similarity.  Default 5 (very similar).

        Returns
        -------
        SceneFrame or None
            The stored frame, or None if deduplicated (skipped).
        """
        # Compute hash outside the lock (CPU-intensive)
        frame_candidate = SceneFrame(
            frame_id=0,  # placeholder, assigned under lock
            image=image,
            source=source,
            analysis=analysis or {},
            notes=notes or [],
        )

        with self._lock:
            self._frame_counter += 1
            frame_candidate.frame_id = self._frame_counter

            # Deduplication: skip near-identical frames
            if deduplicate and self._frames:
                latest = self._frames[-1]
                dist = hamming_distance(latest.thumbnail_hash, frame_candidate.thumbnail_hash)
                if 0 <= dist <= dedup_threshold:
                    logger.debug(
                        "Scene '%s': frame %d deduplicated (distance=%d, threshold=%d)",
                        self.task_id,
                        frame_candidate.frame_id,
                        dist,
                        dedup_threshold,
                    )
                    return None

            self._frames.append(frame_candidate)

            # Evict oldest if over limit, releasing numpy memory eagerly
            while len(self._frames) > self.max_frames:
                evicted = self._frames.pop(0)
                evicted_id = evicted.frame_id
                # Explicitly release the numpy array to free memory
                # immediately rather than waiting for GC
                evicted.image = None  # type: ignore[assignment]
                logger.info(
                    "Scene '%s': evicted frame %d (oldest by timestamp, %d frames remain)",
                    self.task_id,
                    evicted_id,
                    len(self._frames),
                )

            return frame_candidate

    def add_observation(self, text: str) -> None:
        """Record a textual observation about the scene.  Thread-safe."""
        with self._lock:
            self._observations.append(text)

    def update_state(self, **kwargs: Any) -> None:
        """Update task-specific state (e.g. puzzle board state).  Thread-safe."""
        with self._lock:
            self._state.update(kwargs)

    def get_latest_frame(self) -> SceneFrame | None:
        """Return the most recent frame, or None.  Thread-safe."""
        with self._lock:
            return self._frames[-1] if self._frames else None

    def get_frame(self, frame_id: int) -> SceneFrame | None:
        """Retrieve a specific frame by ID.  Thread-safe."""
        with self._lock:
            for f in self._frames:
                if f.frame_id == frame_id:
                    return f
            return None

    def get_recent_frames(self, n: int = 5) -> list[SceneFrame]:
        """Return the N most recent frames.  Thread-safe."""
        with self._lock:
            return list(self._frames[-n:])

    def detect_change(self, frame_a: SceneFrame, frame_b: SceneFrame) -> SceneChange:
        """Detect how much changed between two frames.

        Uses both pixel-level difference and perceptual hash distance
        for robust change detection across lighting/zoom variations.

        Returns a failure result if either frame's image has been evicted
        (set to None).
        """
        if frame_a.image is None or frame_b.image is None:
            return SceneChange(
                from_frame=frame_a.frame_id,
                to_frame=frame_b.frame_id,
                change_score=-1.0,
                description="comparison failed: frame image was evicted",
            )
        try:
            import cv2

            # Resize both to same small size for comparison
            a = cv2.resize(frame_a.image, _CHANGE_COMPARE_SIZE)
            b = cv2.resize(frame_b.image, _CHANGE_COMPARE_SIZE)

            # Convert to grayscale
            ga = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY).astype(np.float32)
            gb = cv2.cvtColor(b, cv2.COLOR_BGR2GRAY).astype(np.float32)

            # Normalized pixel difference
            diff = np.abs(ga - gb)
            pixel_score = float(np.mean(diff) / 255.0)

            # Perceptual hash distance (0-N bits differ → normalize to 0-1)
            hdist = hamming_distance(frame_a.thumbnail_hash, frame_b.thumbnail_hash)
            phash_score = hdist / _PHASH_BITS if hdist >= 0 else pixel_score

            # Blend: weight perceptual hash more (robust to lighting)
            score = _CHANGE_PIXEL_WEIGHT * pixel_score + _CHANGE_PHASH_WEIGHT * phash_score

            description = "no change"
            if score > _CHANGE_THRESHOLD_MAJOR:
                description = "major change"
            elif score > _CHANGE_THRESHOLD_MODERATE:
                description = "moderate change"
            elif score > _CHANGE_THRESHOLD_MINOR:
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
        """Compare the two most recent frames.  Thread-safe."""
        with self._lock:
            if len(self._frames) < 2:
                return None
            # Copy references under lock to prevent eviction during comparison
            frame_a = self._frames[-2]
            frame_b = self._frames[-1]
        return self.detect_change(frame_a, frame_b)

    def visualize_change(
        self,
        frame_a: SceneFrame,
        frame_b: SceneFrame,
    ) -> np.ndarray | None:
        """Generate a visual diff image highlighting changes between frames.

        Returns a BGR image where changed regions are highlighted in red,
        or None if comparison fails or either frame's image was evicted.
        """
        if frame_a.image is None or frame_b.image is None:
            logger.warning("Cannot visualize change: frame image was evicted")
            return None
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
        """Mark session as inactive and release frame data.  Thread-safe."""
        with self._lock:
            if not self._active:
                return  # already closed
            self._active = False
            frame_count = self._frame_counter
            # Eagerly release numpy arrays before clearing the list
            for frame in self._frames:
                frame.image = None  # type: ignore[assignment]
            self._frames.clear()
            self._observations.clear()
            self._state.clear()
        logger.info("Scene session %s closed (%d frames)", self.task_id, frame_count)

    def summarize(self) -> dict[str, Any]:
        """Produce a serializable summary for audit logging.  Thread-safe."""
        with self._lock:
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
                logger.warning(
                    "Evicted %s session '%s' to free resources (%d sessions remain)",
                    "inactive",
                    task_id,
                    len(self._sessions),
                )
                return

        # All active — evict the oldest by creation time
        if self._sessions:
            oldest = min(
                self._sessions,
                key=lambda tid: self._sessions[tid]._created,
            )
            self._sessions[oldest].close()
            del self._sessions[oldest]
            logger.warning(
                "Evicted %s session '%s' to free resources (%d sessions remain)",
                "oldest",
                oldest,
                len(self._sessions),
            )


# ---------------------------------------------------------------------------
# Module singleton
# ---------------------------------------------------------------------------

_scene_manager: SceneManager | None = None
_scene_manager_lock = threading.Lock()


def get_scene_manager() -> SceneManager:
    """Return the process-level singleton SceneManager.  Thread-safe."""
    global _scene_manager
    if _scene_manager is None:
        with _scene_manager_lock:
            if _scene_manager is None:
                _scene_manager = SceneManager()
    return _scene_manager
