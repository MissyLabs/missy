"""Browser operator session storage for the Missy API server."""

from __future__ import annotations

import contextlib
import json
import logging
import os
import secrets
import tempfile
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class WebSession:
    """Authenticated browser operator session."""

    token: str = field(repr=False)
    csrf_token: str = field(repr=False)
    created_at: float
    last_seen: float
    _last_seen_monotonic: float = field(repr=False)


class WebSessionStore:
    """Thread-safe browser session store, optionally persisted to disk.

    In-memory by default (``store_path=None``), matching the original
    behavior. When *store_path* is given, every session survives a
    ``missy gateway start`` restart -- without this, restarting the
    gateway (a routine, frequent operation during development/config
    changes) silently logged out every open Web TUI tab, even though
    nothing about the operator's actual session should have expired.
    Sessions are re-validated against *ttl_seconds* on load, so a session
    that was already stale before the restart doesn't come back to life.
    """

    def __init__(
        self,
        ttl_seconds: int,
        max_sessions: int = 1024,
        store_path: str | None = None,
    ) -> None:
        self._ttl_seconds = max(60, ttl_seconds)
        self._max_sessions = max(1, max_sessions)
        self._sessions: dict[str, WebSession] = {}
        self._lock = threading.Lock()
        self._store_path = Path(store_path).expanduser() if store_path else None
        if self._store_path is not None:
            self._load_locked()

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
            self._persist_locked()
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
                self._persist_locked()
                return None
            session.last_seen = now
            session._last_seen_monotonic = monotonic_now
            self._evict_locked(monotonic_now)
            # Deliberately not persisted here: get() runs on every
            # authenticated request (far more often than create()/revoke()),
            # and the sliding-window refresh it performs only matters again
            # after a restart, an event rare enough that reusing the
            # last-persisted last_seen (from the most recent create/revoke/
            # eviction) is close enough -- persisting on every single
            # request would turn routine Web TUI browsing into constant
            # disk writes for no real benefit given TTLs are day-scale.
            return session

    def revoke(self, token: str | None) -> None:
        if not token:
            return
        with self._lock:
            self._sessions.pop(token, None)
            self._persist_locked()

    def _evict_locked(self, monotonic_now: float) -> None:
        expired = [
            token
            for token, session in self._sessions.items()
            if monotonic_now - session._last_seen_monotonic >= self._ttl_seconds
        ]
        for token in expired:
            self._sessions.pop(token, None)
        if expired:
            self._persist_locked()

    # ------------------------------------------------------------------
    # Disk persistence
    # ------------------------------------------------------------------

    def _load_locked(self) -> None:
        """Load persisted sessions from *self._store_path*, dropping expired ones.

        ``time.monotonic()`` has no meaning across a process restart (its
        epoch is arbitrary, typically boot time), so each session's
        ``_last_seen_monotonic`` is reconstructed from the wall-clock gap
        between its persisted ``last_seen`` and now, applied against the
        *current* process's monotonic clock -- preserving how much of its
        TTL window is actually left rather than either discarding it or
        silently granting a fresh full TTL.
        """
        path = self._store_path
        if path is None or not path.exists():
            return
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning("WebSessionStore: could not load %s: %s", path, exc)
            return
        if not isinstance(raw, dict):
            return
        now_wall = time.time()
        now_monotonic = time.monotonic()
        loaded = 0
        for token, entry in raw.items():
            if not isinstance(entry, dict):
                continue
            try:
                created_at = float(entry["created_at"])
                last_seen = float(entry["last_seen"])
                csrf_token = str(entry["csrf_token"])
            except (KeyError, TypeError, ValueError):
                continue
            elapsed = max(0.0, now_wall - last_seen)
            if elapsed >= self._ttl_seconds:
                continue  # already expired -- don't resurrect it
            self._sessions[token] = WebSession(
                token=token,
                csrf_token=csrf_token,
                created_at=created_at,
                last_seen=last_seen,
                _last_seen_monotonic=now_monotonic - elapsed,
            )
            loaded += 1
        if loaded:
            logger.info("WebSessionStore: restored %d session(s) from %s", loaded, path)

    def _persist_locked(self) -> None:
        """Atomically write the current session map to *self._store_path*.

        No-op when persistence isn't configured. Must be called with
        ``self._lock`` held. ``_last_seen_monotonic`` is process-local and
        deliberately not persisted -- :meth:`_load_locked` reconstructs it
        from the wall-clock ``last_seen`` it does persist.
        """
        path = self._store_path
        if path is None:
            return
        data = {
            session.token: {
                "csrf_token": session.csrf_token,
                "created_at": session.created_at,
                "last_seen": session.last_seen,
            }
            for session in self._sessions.values()
        }
        try:
            path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
            fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=".web_sessions_")
            try:
                os.fchmod(fd, 0o600)
                with os.fdopen(fd, "w", encoding="utf-8") as fh:
                    json.dump(data, fh)
                os.replace(tmp, str(path))
            except Exception:
                with contextlib.suppress(OSError):
                    os.unlink(tmp)
                raise
        except OSError as exc:
            logger.warning("WebSessionStore: could not persist sessions to %s: %s", path, exc)
