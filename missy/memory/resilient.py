"""Resilient memory store with in-memory fallback on primary failure."""

from __future__ import annotations

import logging
import threading

logger = logging.getLogger(__name__)


class ResilientMemoryStore:
    """Wraps a primary store with an in-memory dict fallback.

    Behaviour:
    - All writes go to both the in-memory cache *and* the primary store.
    - On primary failure, operations transparently fall back to the cache.
    - After *max_failures* consecutive failures the primary is marked
      unhealthy and every subsequent success attempt logs the recovery.
    - On recovery the cached entries are synced back to the primary.

    Args:
        primary: Any store object that implements the MemoryStore interface
            (``add_turn``, ``get_session_turns``, ``get_recent_turns``,
            ``clear_session``, ``search``, ``save_learning``,
            ``get_learnings``, ``cleanup``).
        max_failures: Number of consecutive failures before the store is
            considered unhealthy.

    Example::

        from missy.memory.sqlite_store import SQLiteMemoryStore
        from missy.memory.resilient import ResilientMemoryStore

        store = ResilientMemoryStore(SQLiteMemoryStore())
        store.add_turn(turn)
        print(store.is_healthy)
    """

    def __init__(self, primary, max_failures: int = 3) -> None:
        self._primary = primary
        self._max_failures = max_failures
        self._failures = 0
        self._cache: dict[str, list] = {}  # session_id -> list of turns
        self._lock = threading.Lock()
        self._healthy = True

    # ------------------------------------------------------------------
    # Health tracking
    # ------------------------------------------------------------------

    def _on_success(self) -> None:
        with self._lock:
            if not self._healthy:
                logger.info(
                    "ResilientMemory: primary recovered after %d failure(s), syncing cache",
                    self._failures,
                )
                self._sync_cache_to_primary()
            self._failures = 0
            self._healthy = True

    def _on_failure(self, exc: Exception) -> None:
        with self._lock:
            self._failures += 1
            self._healthy = False
            logger.warning(
                "ResilientMemory: primary failure #%d: %s",
                self._failures,
                exc,
            )
            if self._failures >= self._max_failures:
                logger.error(
                    "ResilientMemory: primary unhealthy after %d consecutive failures",
                    self._failures,
                )

    def _sync_cache_to_primary(self) -> None:
        """Best-effort replay of cached turns to a recovered primary."""
        for turns in self._cache.values():
            for turn in turns:
                try:
                    self._primary.add_turn(turn)
                except Exception as exc:
                    logger.debug("ResilientMemory: sync failed for turn %s: %s", turn.id, exc)

    # ------------------------------------------------------------------
    # Write operations
    # ------------------------------------------------------------------

    def add_turn(self, turn) -> None:
        """Append *turn* to the cache and attempt to write to the primary.

        Args:
            turn: A turn object with at least a ``session_id`` attribute.
        """
        with self._lock:
            sid = turn.session_id
            if sid not in self._cache:
                self._cache[sid] = []
            self._cache[sid].append(turn)

        try:
            self._primary.add_turn(turn)
            self._on_success()
        except Exception as exc:
            self._on_failure(exc)

    def clear_session(self, session_id: str) -> None:
        """Remove all turns for *session_id* from cache and primary.

        Args:
            session_id: The session whose turns should be deleted.
        """
        with self._lock:
            self._cache.pop(session_id, None)
        try:
            self._primary.clear_session(session_id)
            self._on_success()
        except Exception as exc:
            self._on_failure(exc)

    def save_learning(self, learning) -> None:
        """Forward a learning object to the primary store.

        Args:
            learning: An object carrying learning fields (duck-typed).
        """
        try:
            self._primary.save_learning(learning)
            self._on_success()
        except Exception as exc:
            self._on_failure(exc)

    # ------------------------------------------------------------------
    # Read operations
    # ------------------------------------------------------------------

    def get_session_turns(self, session_id: str, limit: int = 100) -> list:
        """Return up to *limit* turns for *session_id*, falling back to cache.

        Args:
            session_id: Session identifier to filter by.
            limit: Maximum number of turns to return.

        Returns:
            A list of turn objects in chronological order.
        """
        try:
            result = self._primary.get_session_turns(session_id, limit)
            self._on_success()
            return result
        except Exception as exc:
            self._on_failure(exc)
            cached = self._cache.get(session_id, [])
            return cached[-limit:]

    def get_recent_turns(self, limit: int = 50) -> list:
        """Return up to *limit* most recent turns, falling back to cache.

        Args:
            limit: Maximum number of turns to return.

        Returns:
            A list of turn objects in chronological order.
        """
        try:
            result = self._primary.get_recent_turns(limit)
            self._on_success()
            return result
        except Exception as exc:
            self._on_failure(exc)
            all_turns: list = []
            for turns in self._cache.values():
                all_turns.extend(turns)
            all_turns.sort(key=lambda t: t.timestamp)
            return all_turns[-limit:]

    def search(
        self,
        query: str,
        limit: int = 10,
        session_id: str | None = None,
    ) -> list:
        """Search conversation history, falling back to keyword scan of cache.

        Args:
            query: Search query string.
            limit: Maximum number of results to return.
            session_id: When given, restrict results to this session.

        Returns:
            A list of matching turn objects.
        """
        try:
            result = self._primary.search(query, limit=limit, session_id=session_id)
            self._on_success()
            return result
        except Exception as exc:
            self._on_failure(exc)
            # Basic case-insensitive keyword scan over the in-memory cache
            matches: list = []
            query_lower = query.lower()
            for sid, turns in self._cache.items():
                if session_id and sid != session_id:
                    continue
                for t in turns:
                    if query_lower in t.content.lower():
                        matches.append(t)
            return matches[:limit]

    def get_learnings(
        self,
        task_type: str | None = None,
        limit: int = 5,
    ) -> list:
        """Retrieve recent learning lessons, returning empty list on failure.

        Args:
            task_type: When given, filter to this task type only.
            limit: Maximum number of lessons to return.

        Returns:
            A list of lesson strings, or an empty list when the primary
            is unavailable.
        """
        try:
            result = self._primary.get_learnings(task_type=task_type, limit=limit)
            self._on_success()
            return result
        except Exception as exc:
            self._on_failure(exc)
            return []

    # ------------------------------------------------------------------
    # Maintenance
    # ------------------------------------------------------------------

    def cleanup(self, older_than_days: int = 30) -> int:
        """Delegate cleanup to the primary store.

        Args:
            older_than_days: Age threshold in days.

        Returns:
            Number of rows deleted, or 0 on failure.
        """
        try:
            result = self._primary.cleanup(older_than_days=older_than_days)
            self._on_success()
            return result
        except Exception as exc:
            self._on_failure(exc)
            return 0

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def is_healthy(self) -> bool:
        """``True`` when the primary store has not exceeded *max_failures*."""
        with self._lock:
            return self._healthy
