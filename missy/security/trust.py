"""Trust scoring system for providers, MCP servers, and tools.

Tracks reliability on a 0-1000 scale.  New entities start at 500.
Successes increase the score; failures and policy violations decrease it.

Optionally persists scores to a JSON file (F11) so they survive a process
restart and can be inspected out-of-band via ``missy tools trust``. When no
``persist_path`` is given the scorer is purely in-memory (its original
behaviour) and touches no disk.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
import threading
from pathlib import Path

logger = logging.getLogger(__name__)

#: Default score assigned to entities seen for the first time.
DEFAULT_SCORE = 500

#: Maximum trust score.
MAX_SCORE = 1000

#: Minimum trust score.
MIN_SCORE = 0

#: Default persistence location used by the production runtime (F11).
DEFAULT_TRUST_PATH = "~/.missy/trust.json"


class TrustScorer:
    """Trust scorer for entity reliability tracking.

    Scores range from 0 (untrusted) to 1000 (fully trusted).

    Args:
        persist_path: When set, scores are loaded from this JSON file on
            construction and re-saved after every mutation, so they survive
            restarts and are readable by a separate process (the ``missy
            tools trust`` CLI). ``None`` keeps the scorer purely in-memory.
    """

    def __init__(self, persist_path: str | os.PathLike[str] | None = None) -> None:
        self._scores: dict[str, int] = {}
        self._lock = threading.Lock()
        self._path: Path | None = (
            Path(persist_path).expanduser() if persist_path is not None else None
        )
        if self._path is not None:
            self._load()

    # ------------------------------------------------------------------
    # Persistence (no-ops when persist_path is None)
    # ------------------------------------------------------------------

    def _load(self) -> None:
        """Load scores from disk. Silently starts empty on any read error."""
        if self._path is None or not self._path.exists():
            return
        try:
            raw = json.loads(self._path.read_text(encoding="utf-8"))
            if isinstance(raw, dict):
                self._scores = {
                    str(k): int(v)
                    for k, v in raw.items()
                    if isinstance(v, (int, float)) and not isinstance(v, bool)
                }
        except (OSError, ValueError, TypeError):
            logger.debug("TrustScorer: could not load %s; starting empty.", self._path)

    def _save_locked(self) -> None:
        """Atomically write scores to disk. Caller must hold ``self._lock``.

        Best-effort: a persistence failure must never break a tool call, so
        write errors are logged at debug and swallowed.
        """
        if self._path is None:
            return
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            fd, tmp = tempfile.mkstemp(dir=str(self._path.parent), prefix=".trust-", suffix=".tmp")
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as fh:
                    json.dump(self._scores, fh)
                os.replace(tmp, self._path)
            finally:
                if os.path.exists(tmp):
                    os.unlink(tmp)
        except OSError:
            logger.debug("TrustScorer: could not persist to %s.", self._path, exc_info=True)

    # ------------------------------------------------------------------
    # Scoring API
    # ------------------------------------------------------------------

    def score(self, entity_id: str) -> int:
        """Return the current trust score for *entity_id* (default 500)."""
        with self._lock:
            return self._scores.get(entity_id, DEFAULT_SCORE)

    def record_success(self, entity_id: str, weight: int = 10) -> None:
        """Increase the score for *entity_id* by *weight* (capped at 1000)."""
        with self._lock:
            current = self._scores.get(entity_id, DEFAULT_SCORE)
            self._scores[entity_id] = min(current + weight, MAX_SCORE)
            self._save_locked()

    def record_failure(self, entity_id: str, weight: int = 50) -> None:
        """Decrease the score for *entity_id* by *weight* (floored at 0)."""
        with self._lock:
            current = self._scores.get(entity_id, DEFAULT_SCORE)
            self._scores[entity_id] = max(current - weight, MIN_SCORE)
            self._save_locked()

    def record_violation(self, entity_id: str, weight: int = 200) -> None:
        """Major decrease for a policy violation (floored at 0)."""
        with self._lock:
            current = self._scores.get(entity_id, DEFAULT_SCORE)
            self._scores[entity_id] = max(current - weight, MIN_SCORE)
            self._save_locked()

    def is_trusted(self, entity_id: str, threshold: int = 200) -> bool:
        """Return ``True`` if the entity's score is above *threshold*."""
        return self.score(entity_id) > threshold

    def get_scores(self) -> dict[str, int]:
        """Return a copy of all current scores."""
        with self._lock:
            return dict(self._scores)

    def reset(self, entity_id: str) -> None:
        """Reset *entity_id* back to the default score (500)."""
        with self._lock:
            self._scores[entity_id] = DEFAULT_SCORE
            self._save_locked()
