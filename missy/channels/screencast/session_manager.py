"""Multi-session state management for the screencast channel.

Tracks per-connection state, provides a bounded asyncio queue for frame
hand-off to the analyzer, and stores analysis results per session.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

# Maximum pending frames in the analysis queue.
_MAX_QUEUE_SIZE = 50

# Maximum stored analysis results per session (oldest evicted).
_MAX_RESULTS_PER_SESSION = 50

# Default maximum concurrent sessions.
_DEFAULT_MAX_SESSIONS = 20

# Maximum number of distinct sessions whose results are retained across
# the process lifetime. unregister_connection() intentionally leaves a
# disconnected session's results in place (they remain briefly queryable
# after disconnect), so without this cap every session that ever streamed
# at least one frame would leave a permanent dict entry forever, growing
# unboundedly over weeks of routine use -- the equivalent protection
# ScreencastTokenRegistry already applies to revoked sessions (auth.py's
# _MAX_TRACKED_SESSIONS) but which was never applied here.
_MAX_TRACKED_RESULT_SESSIONS = 200


@dataclass
class FrameMetadata:
    """Metadata attached to a captured frame."""

    session_id: str
    frame_number: int
    format: str  # "jpeg" or "png"
    width: int = 0
    height: int = 0
    timestamp: float = field(default_factory=time.time)
    size_bytes: int = 0


@dataclass
class AnalysisResult:
    """Result from the vision model for a single frame."""

    session_id: str
    frame_number: int
    timestamp: float = field(default_factory=time.time)
    analysis_text: str = ""
    model: str = ""
    processing_ms: int = 0


@dataclass
class SessionState:
    """Per-connection mutable state (server-side)."""

    session_id: str
    frame_count: int = 0
    capture_interval_ms: int = 10000
    remote_address: str = ""
    connected_at: float = field(default_factory=time.time)


class SessionManager:
    """Manages multi-session state, frame queuing, and analysis results.

    Args:
        max_sessions: Maximum number of concurrent connected sessions.
    """

    def __init__(self, max_sessions: int = _DEFAULT_MAX_SESSIONS) -> None:
        self._max_sessions = max_sessions
        self._connections: dict[str, SessionState] = {}
        self._results: dict[str, deque[AnalysisResult]] = {}
        self._queue: asyncio.Queue[tuple[FrameMetadata, bytes]] | None = None

    def set_queue(self, queue: asyncio.Queue[tuple[FrameMetadata, bytes]]) -> None:
        """Attach the frame queue (created on the server's event loop)."""
        self._queue = queue

    @property
    def queue(self) -> asyncio.Queue[tuple[FrameMetadata, bytes]] | None:
        return self._queue

    @property
    def connection_count(self) -> int:
        return len(self._connections)

    @property
    def at_capacity(self) -> bool:
        return len(self._connections) >= self._max_sessions

    def register_connection(
        self,
        session_id: str,
        remote_address: str = "",
    ) -> SessionState:
        """Register a new authenticated connection."""
        state = SessionState(session_id=session_id, remote_address=remote_address)
        self._connections[session_id] = state
        if session_id not in self._results:
            self._results[session_id] = deque(maxlen=_MAX_RESULTS_PER_SESSION)
        return state

    def unregister_connection(self, session_id: str) -> None:
        """Remove a connection (does not clear results)."""
        self._connections.pop(session_id, None)

    def get_connection(self, session_id: str) -> SessionState | None:
        return self._connections.get(session_id)

    def enqueue_frame(self, metadata: FrameMetadata, data: bytes) -> bool:
        """Put a frame on the analysis queue.  Returns ``False`` if queue is full."""
        if self._queue is None:
            return False
        try:
            self._queue.put_nowait((metadata, data))
            return True
        except asyncio.QueueFull:
            return False

    async def dequeue_frame(self) -> tuple[FrameMetadata, bytes]:
        """Block until a frame is available."""
        if self._queue is None:
            raise RuntimeError("Queue not initialized")
        return await self._queue.get()

    def store_result(self, result: AnalysisResult) -> None:
        """Store an analysis result for the given session."""
        sid = result.session_id
        if sid not in self._results:
            self._results[sid] = deque(maxlen=_MAX_RESULTS_PER_SESSION)
        self._results[sid].append(result)
        self._prune_results()

    def _prune_results(self) -> None:
        """Evict the oldest disconnected sessions' results once the number
        of tracked sessions exceeds ``_MAX_TRACKED_RESULT_SESSIONS``.

        Only disconnected sessions (not in ``self._connections``) are
        eligible for eviction, so an active session's results are never
        dropped out from under it.
        """
        if len(self._results) <= _MAX_TRACKED_RESULT_SESSIONS:
            return
        disconnected = [sid for sid in self._results if sid not in self._connections]
        disconnected.sort(
            key=lambda sid: self._results[sid][-1].timestamp if self._results[sid] else 0.0
        )
        overflow = len(self._results) - _MAX_TRACKED_RESULT_SESSIONS
        for sid in disconnected[:overflow]:
            self._results.pop(sid, None)

    def get_results(self, session_id: str, limit: int = 10) -> list[AnalysisResult]:
        """Return the most recent results for a session."""
        dq = self._results.get(session_id)
        if not dq:
            return []
        items = list(dq)
        return items[-limit:]

    def get_latest_result(self, session_id: str) -> AnalysisResult | None:
        """Return the latest analysis result for a session, or ``None``."""
        dq = self._results.get(session_id)
        if not dq:
            return None
        return dq[-1]

    def get_status(self) -> dict[str, Any]:
        """Return a summary of the session manager state."""
        return {
            "connected_sessions": len(self._connections),
            "max_sessions": self._max_sessions,
            "queue_size": self._queue.qsize() if self._queue else 0,
            "sessions": {
                sid: {
                    "frame_count": st.frame_count,
                    "remote_address": st.remote_address,
                    "connected_at": st.connected_at,
                }
                for sid, st in self._connections.items()
            },
        }
