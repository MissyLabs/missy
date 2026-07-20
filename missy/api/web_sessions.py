"""Browser operator session storage for the Missy API server."""

from __future__ import annotations

import secrets
import threading
import time
from dataclasses import dataclass, field


@dataclass
class WebSession:
    """Authenticated browser operator session."""

    token: str = field(repr=False)
    csrf_token: str = field(repr=False)
    created_at: float
    last_seen: float
    _last_seen_monotonic: float = field(repr=False)


class WebSessionStore:
    """Thread-safe in-memory browser session store."""

    def __init__(self, ttl_seconds: int, max_sessions: int = 1024) -> None:
        self._ttl_seconds = max(60, ttl_seconds)
        self._max_sessions = max(1, max_sessions)
        self._sessions: dict[str, WebSession] = {}
        self._lock = threading.Lock()

    def create(self) -> WebSession:
        now = time.time()
        monotonic_now = time.monotonic()
        session = WebSession(
            token=secrets.token_urlsafe(32),
            csrf_token=secrets.token_urlsafe(32),
            created_at=now,
            last_seen=now,
            _last_seen_monotonic=monotonic_now,
        )
        with self._lock:
            self._evict_locked(monotonic_now)
            while len(self._sessions) >= self._max_sessions:
                oldest = min(self._sessions.values(), key=lambda item: item._last_seen_monotonic)
                self._sessions.pop(oldest.token, None)
            self._sessions[session.token] = session
        return session

    def get(self, token: str | None) -> WebSession | None:
        if not token:
            return None
        now = time.time()
        monotonic_now = time.monotonic()
        with self._lock:
            session = self._sessions.get(token)
            if session is None:
                return None
            if monotonic_now - session._last_seen_monotonic >= self._ttl_seconds:
                self._sessions.pop(token, None)
                return None
            session.last_seen = now
            session._last_seen_monotonic = monotonic_now
            self._evict_locked(monotonic_now)
            return session

    def revoke(self, token: str | None) -> None:
        if not token:
            return
        with self._lock:
            self._sessions.pop(token, None)

    def _evict_locked(self, monotonic_now: float) -> None:
        expired = [
            token
            for token, session in self._sessions.items()
            if monotonic_now - session._last_seen_monotonic >= self._ttl_seconds
        ]
        for token in expired:
            self._sessions.pop(token, None)
