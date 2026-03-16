"""Screencast token registry — PBKDF2 auth for ephemeral browser sessions.

Each ``!screen share`` command creates a one-time session with a hashed token.
The browser authenticates over WebSocket using the plaintext token (passed via
URL query param).  Tokens are never stored; only PBKDF2-HMAC-SHA256 hashes
are kept in memory.
"""

from __future__ import annotations

import hashlib
import hmac
import logging
import secrets
import threading
import time
from dataclasses import dataclass, field
from typing import Any

from missy.core.events import AuditEvent, event_bus

logger = logging.getLogger(__name__)

_PBKDF2_ITERATIONS = 100_000
_TASK_ID = "screencast-auth"


def _emit(
    session_id: str,
    event_type: str,
    result: str,
    detail: dict[str, Any] | None = None,
) -> None:
    """Publish a screencast audit event."""
    try:
        event_bus.publish(
            AuditEvent.now(
                session_id=session_id,
                task_id=_TASK_ID,
                event_type=event_type,
                category="plugin",
                result=result,  # type: ignore[arg-type]
                detail=detail or {},
            )
        )
    except Exception:
        logger.debug("screencast auth: audit emit failed for %r", event_type, exc_info=True)


@dataclass
class ScreencastSession:
    """State for a single screencast sharing session."""

    session_id: str
    token_hash: str
    created_at: float = field(default_factory=time.time)
    created_by: str = ""  # Discord user ID
    discord_channel_id: str = ""
    label: str = ""
    active: bool = True
    last_frame_at: float = 0.0
    frame_count: int = 0
    analysis_count: int = 0


class ScreencastTokenRegistry:
    """Thread-safe registry of ephemeral screencast sessions with PBKDF2 auth."""

    def __init__(self) -> None:
        self._sessions: dict[str, ScreencastSession] = {}
        self._lock = threading.RLock()

    @staticmethod
    def _hash_token(session_id: str, token: str) -> str:
        """Return the PBKDF2-HMAC-SHA256 hex digest of *token* salted with *session_id*."""
        raw = hashlib.pbkdf2_hmac(
            "sha256",
            token.encode(),
            session_id.encode(),
            iterations=_PBKDF2_ITERATIONS,
        )
        return raw.hex()

    def create_session(
        self,
        *,
        created_by: str = "",
        discord_channel_id: str = "",
        label: str = "",
    ) -> tuple[str, str]:
        """Create a new session and return ``(session_id, plaintext_token)``.

        The plaintext token is returned exactly once and never stored.
        """
        session_id = secrets.token_urlsafe(16)
        token = secrets.token_urlsafe(32)
        token_hash = self._hash_token(session_id, token)

        session = ScreencastSession(
            session_id=session_id,
            token_hash=token_hash,
            created_by=created_by,
            discord_channel_id=discord_channel_id,
            label=label,
        )

        with self._lock:
            self._sessions[session_id] = session

        _emit(
            session_id,
            "screencast.session.created",
            "allow",
            {
                "created_by": created_by,
                "discord_channel_id": discord_channel_id,
                "label": label,
            },
        )
        logger.info(
            "Screencast session created: %s label=%r by=%s",
            session_id,
            label,
            created_by,
        )
        return session_id, token

    def verify_token(self, session_id: str, token: str) -> bool:
        """Return ``True`` if *token* matches the stored hash for *session_id*.

        Uses constant-time comparison to prevent timing attacks.
        """
        with self._lock:
            session = self._sessions.get(session_id)
            if session is None or not session.active:
                return False
            candidate = self._hash_token(session_id, token)
            return hmac.compare_digest(candidate, session.token_hash)

    def get_session(self, session_id: str) -> ScreencastSession | None:
        """Return the session for *session_id*, or ``None``."""
        with self._lock:
            return self._sessions.get(session_id)

    def list_active(self) -> list[ScreencastSession]:
        """Return all active sessions."""
        with self._lock:
            return [s for s in self._sessions.values() if s.active]

    def revoke_session(self, session_id: str) -> bool:
        """Mark a session as inactive.  Returns ``True`` if it existed."""
        with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                return False
            session.active = False

        _emit(session_id, "screencast.session.revoked", "allow")
        logger.info("Screencast session revoked: %s", session_id)
        return True

    def update_frame_stats(
        self,
        session_id: str,
        *,
        frame_count: int | None = None,
        analysis_count: int | None = None,
    ) -> None:
        """Update frame/analysis counters for a session."""
        with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                return
            session.last_frame_at = time.time()
            if frame_count is not None:
                session.frame_count = frame_count
            if analysis_count is not None:
                session.analysis_count = analysis_count
