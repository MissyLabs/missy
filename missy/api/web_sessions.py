"""Browser operator session storage for the Missy API server."""

from __future__ import annotations

import secrets
import threading
import time
from dataclasses import dataclass


@dataclass
class WebSession:
    """Authenticated browser operator session."""

    token: str
    csrf_token: str
    created_at: float
    last_seen: float


class WebSessionStore:
    """Thread-safe in-memory browser session store."""

    def __init__(self, ttl_seconds: int) -> None:
        self._ttl_seconds = max(60, ttl_seconds)
        self._sessions: dict[str, WebSession] = {}
        self._lock = threading.Lock()

    def create(self) -> WebSession:
        now = time.time()
        session = WebSession(
            token=secrets.token_urlsafe(32),
            csrf_token=secrets.token_urlsafe(32),
            created_at=now,
            last_seen=now,
        )
        with self._lock:
            self._sessions[session.token] = session
            self._evict_locked(now)
        return session

    def get(self, token: str | None) -> WebSession | None:
        if not token:
            return None
        now = time.time()
        with self._lock:
            session = self._sessions.get(token)
            if session is None:
                return None
            if now - session.last_seen > self._ttl_seconds:
                self._sessions.pop(token, None)
                return None
            session.last_seen = now
            self._evict_locked(now)
            return session

    def revoke(self, token: str | None) -> None:
        if not token:
            return
        with self._lock:
            self._sessions.pop(token, None)

    def _evict_locked(self, now: float) -> None:
        expired = [
            token
            for token, session in self._sessions.items()
            if now - session.last_seen > self._ttl_seconds
        ]
        for token in expired:
            self._sessions.pop(token, None)
