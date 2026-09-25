"""Trust scoring system for providers, MCP servers, and tools.

Tracks reliability on a 0-1000 scale.  New entities start at 500.
Successes increase the score; failures and policy violations decrease it.

Optionally persists scores to a JSON file (F11) so they survive a process
restart and can be inspected out-of-band via ``missy tools trust``. When no
``persist_path`` is given the scorer is purely in-memory (its original
behaviour) and touches no disk.
"""

from __future__ import annotations

import atexit
import contextlib
import json
import logging
import os
import tempfile
import threading
import time
from collections.abc import Iterator
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

    Persistence (DATA-02): every mutation is recorded as a *delta* and applied
    to the current on-disk value under an exclusive cross-process ``flock``
    (read-apply-write), so several scorers sharing one file -- the gateway's
    main/Discord/proactive/scheduled-job runtimes, plus ``missy tools trust``
    -- never erase each other's updates the way whole-dict rewrites from
    per-instance caches did. In-process, use :func:`get_trust_scorer` to share
    one instance. Writes are batched to at most one per *flush_interval*;
    policy violations and resets flush immediately.

    Args:
        persist_path: When set, scores are loaded from this JSON file and
            deltas persisted to it. ``None`` keeps the scorer purely in-memory.
        flush_interval: Minimum seconds between routine (success/failure)
            writes. ``0`` writes on every mutation.
    """

    def __init__(
        self,
        persist_path: str | os.PathLike[str] | None = None,
        *,
        flush_interval: float = 1.0,
    ) -> None:
        self._scores: dict[str, int] = {}
        self._lock = threading.Lock()
        self._path: Path | None = (
            Path(persist_path).expanduser() if persist_path is not None else None
        )
        self._flush_interval = max(0.0, float(flush_interval))
        #: Pending operations not yet persisted: (entity, "delta"|"set", value).
        self._pending: list[tuple[str, str, int]] = []
        # None = never flushed: the first write always persists at once.
        # (0.0 compared against time.monotonic() -- seconds since boot --
        # silently deferred the first write on a freshly booted host.)
        self._last_flush: float | None = None
        if self._path is not None:
            self._scores = self._read_disk()

    # ------------------------------------------------------------------
    # Persistence (no-ops when persist_path is None)
    # ------------------------------------------------------------------

    def _read_disk(self) -> dict[str, int]:
        """Return scores from disk; empty on a missing or unreadable file."""
        if self._path is None or not self._path.exists():
            return {}
        try:
            raw = json.loads(self._path.read_text(encoding="utf-8"))
        except (OSError, ValueError, TypeError):
            logger.debug("TrustScorer: could not load %s; starting empty.", self._path)
            return {}
        if not isinstance(raw, dict):
            return {}
        return {
            str(k): int(v)
            for k, v in raw.items()
            if isinstance(v, (int, float)) and not isinstance(v, bool)
        }

    def _load(self) -> None:
        """Reload scores from disk (kept for backward compatibility)."""
        with self._lock:
            self._scores = self._read_disk()

    @contextlib.contextmanager
    def _file_lock(self) -> Iterator[None]:
        try:
            import fcntl
        except ImportError:  # pragma: no cover - non-POSIX
            yield
            return
        assert self._path is not None
        fd = os.open(str(self._path) + ".lock", os.O_CREAT | os.O_RDWR, 0o600)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX)
            yield
        finally:
            with contextlib.suppress(OSError):
                fcntl.flock(fd, fcntl.LOCK_UN)
            os.close(fd)

    def _flush_locked(self) -> None:
        """Apply pending deltas to the on-disk scores. Caller holds ``_lock``.

        Best-effort: a persistence failure must never break a tool call, so
        write errors are logged at debug and swallowed (pending operations
        are kept and retried on the next flush).
        """
        if self._path is None or not self._pending:
            return
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            with self._file_lock():
                merged = self._read_disk()
                for entity, kind, value in self._pending:
                    if kind == "set":
                        merged[entity] = value
                    else:
                        current = merged.get(entity, DEFAULT_SCORE)
                        merged[entity] = max(MIN_SCORE, min(MAX_SCORE, current + value))
                fd, tmp = tempfile.mkstemp(
                    dir=str(self._path.parent), prefix=".trust-", suffix=".tmp"
                )
                try:
                    os.fchmod(fd, 0o600)
                    with os.fdopen(fd, "w", encoding="utf-8") as fh:
                        json.dump(merged, fh)
                    os.replace(tmp, self._path)
                finally:
                    if os.path.exists(tmp):
                        os.unlink(tmp)
            self._scores = merged
            self._pending.clear()
            self._last_flush = time.monotonic()
        except OSError:
            logger.debug("TrustScorer: could not persist to %s.", self._path, exc_info=True)

    def _save_locked(self, *, force: bool = False) -> None:
        """Flush if forced or the batching interval elapsed. Caller holds ``_lock``."""
        if self._path is None:
            self._pending.clear()
            return
        if (
            force
            or self._last_flush is None
            or time.monotonic() - self._last_flush >= self._flush_interval
        ):
            self._flush_locked()

    def flush(self) -> None:
        """Persist any batched updates now."""
        with self._lock:
            self._flush_locked()

    def _apply(self, entity_id: str, delta: int, *, force: bool = False) -> None:
        with self._lock:
            current = self._scores.get(entity_id, DEFAULT_SCORE)
            self._scores[entity_id] = max(MIN_SCORE, min(MAX_SCORE, current + delta))
            self._pending.append((entity_id, "delta", delta))
            self._save_locked(force=force)

    # ------------------------------------------------------------------
    # Scoring API
    # ------------------------------------------------------------------

    def score(self, entity_id: str) -> int:
        """Return the current trust score for *entity_id* (default 500)."""
        with self._lock:
            return self._scores.get(entity_id, DEFAULT_SCORE)

    def record_success(self, entity_id: str, weight: int = 10) -> None:
        """Increase the score for *entity_id* by *weight* (capped at 1000)."""
        self._apply(entity_id, weight)

    def record_failure(self, entity_id: str, weight: int = 50) -> None:
        """Decrease the score for *entity_id* by *weight* (floored at 0)."""
        self._apply(entity_id, -weight)

    def record_violation(self, entity_id: str, weight: int = 200) -> None:
        """Major decrease for a policy violation (floored at 0). Flushed at once."""
        self._apply(entity_id, -weight, force=True)

    def is_trusted(self, entity_id: str, threshold: int = 200) -> bool:
        """Return ``True`` if the entity's score is above *threshold*."""
        return self.score(entity_id) > threshold

    def get_scores(self) -> dict[str, int]:
        """Return a copy of all current scores."""
        with self._lock:
            return dict(self._scores)

    def reset(self, entity_id: str) -> None:
        """Reset *entity_id* back to the default score (500). Flushed at once."""
        with self._lock:
            self._scores[entity_id] = DEFAULT_SCORE
            self._pending.append((entity_id, "set", DEFAULT_SCORE))
            self._save_locked(force=True)


_SHARED: dict[str, TrustScorer] = {}
_SHARED_LOCK = threading.Lock()


def get_trust_scorer(persist_path: str | os.PathLike[str] = DEFAULT_TRUST_PATH) -> TrustScorer:
    """Return the process-wide :class:`TrustScorer` for *persist_path* (DATA-02).

    Every AgentRuntime in a process shares one instance per file, so a
    violation recorded by one runtime is immediately visible to the
    ``is_trusted()`` gate of every other. Pending batched updates are flushed
    at interpreter exit.
    """
    key = str(Path(persist_path).expanduser().resolve(strict=False))
    with _SHARED_LOCK:
        scorer = _SHARED.get(key)
        if scorer is None:
            scorer = TrustScorer(persist_path=persist_path)
            _SHARED[key] = scorer
            atexit.register(scorer.flush)
        return scorer
