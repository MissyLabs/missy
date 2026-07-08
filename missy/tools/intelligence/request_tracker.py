"""Track user requests and detect repeated workflow patterns.

The :class:`RequestTracker` records each agent turn (user message +
tool calls) to a lightweight SQLite database and identifies high-frequency
patterns that are candidates for structured tool extraction.

Pattern detection works by normalising user messages (stripping tokens that
vary turn to turn) and clustering similar messages.  When a pattern exceeds
a configurable frequency threshold the tracker emits an audit event so the
:class:`~missy.tools.intelligence.candidate_generator.CandidateGenerator`
can propose a structured tool.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import sqlite3
import threading
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_DEFAULT_DB = Path("~/.missy/request_tracker.db")

# Minimum occurrences before a pattern is surfaced as a candidate.
DEFAULT_MIN_COUNT = 3
# Time-decay half-life in days — older events score lower.
DECAY_HALF_LIFE_DAYS = 30.0


@dataclass
class RequestEvent:
    """A single recorded user turn.

    Attributes:
        id: Unique event UUID.
        session_id: Owning session identifier.
        timestamp: UTC ISO-8601 string.
        raw_message: Original user message (may contain PII — do not log).
        normalised: Normalised, PII-stripped message used for clustering.
        pattern_key: Hash of the normalised message for fast grouping.
        tool_calls: List of tool names invoked in this turn (may be empty).
        metadata: Arbitrary extra context (channel, provider, etc.).
    """

    id: str
    session_id: str
    timestamp: str
    raw_message: str
    normalised: str
    pattern_key: str
    tool_calls: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def create(
        cls,
        session_id: str,
        raw_message: str,
        tool_calls: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> RequestEvent:
        normalised = _normalise(raw_message)
        return cls(
            id=str(uuid.uuid4()),
            session_id=session_id,
            timestamp=datetime.now(UTC).isoformat(),
            raw_message=raw_message,
            normalised=normalised,
            pattern_key=_hash(normalised),
            tool_calls=list(tool_calls or []),
            metadata=dict(metadata or {}),
        )


@dataclass
class RequestPattern:
    """A detected repeating workflow pattern.

    Attributes:
        pattern_key: Hash identifying this cluster.
        representative: Representative normalised message for the cluster.
        count: Total recorded occurrences.
        recent_count: Occurrences in the last ``DECAY_HALF_LIFE_DAYS`` days.
        frequency_score: Decay-weighted frequency in [0, 1].
        common_tools: Tools used in at least half the events in this pattern.
        first_seen: ISO-8601 timestamp of earliest event.
        last_seen: ISO-8601 timestamp of most recent event.
        example_messages: Up to 3 raw messages for human review (stripped).
    """

    pattern_key: str
    representative: str
    count: int
    recent_count: int
    frequency_score: float
    common_tools: list[str]
    first_seen: str
    last_seen: str
    example_messages: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Normalisation helpers
# ---------------------------------------------------------------------------

_URL_RE = re.compile(r"https?://\S+", re.IGNORECASE)
_EMAIL_RE = re.compile(r"[\w.+-]+@[\w.-]+\.\w{2,}", re.IGNORECASE)
_TOKEN_RE = re.compile(r"\b[A-Za-z0-9+/]{20,}\b")  # long base64-ish tokens
_NUMBER_RE = re.compile(r"\b\d[\d.,]*\b")
_PATH_RE = re.compile(r"/(?:home|tmp|var|etc|usr)/\S*")
_WHITESPACE_RE = re.compile(r"\s+")


def _normalise(text: str) -> str:
    """Return a normalised form of *text* for pattern clustering.

    Strips URLs, emails, long tokens, file paths, and numbers so that
    structurally similar messages collapse to the same key.
    """
    t = text.lower()
    t = _URL_RE.sub(" __url__ ", t)
    t = _EMAIL_RE.sub(" __email__ ", t)
    t = _PATH_RE.sub(" __path__ ", t)
    t = _TOKEN_RE.sub(" __token__ ", t)
    t = _NUMBER_RE.sub(" __num__ ", t)
    t = _WHITESPACE_RE.sub(" ", t).strip()
    return t[:500]  # cap to prevent absurdly large keys


def _hash(normalised: str) -> str:
    return hashlib.sha256(normalised.encode()).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Main tracker
# ---------------------------------------------------------------------------


class RequestTracker:
    """SQLite-backed tracker that records requests and surfaces patterns.

    The store lives at *db_path* (default ``~/.missy/request_tracker.db``).
    All operations are thread-safe via a per-instance lock.

    Args:
        db_path: Path to the SQLite database file.  Expanded with
            :func:`~pathlib.Path.expanduser`.
    """

    def __init__(self, db_path: Path | str | None = None) -> None:
        self._path = Path(db_path or _DEFAULT_DB).expanduser()
        self._lock = threading.Lock()
        self._path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        self._init_schema()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def record(
        self,
        session_id: str,
        user_message: str,
        tool_calls: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> RequestEvent:
        """Record a user turn and return the stored :class:`RequestEvent`.

        Args:
            session_id: Active session identifier.
            user_message: Raw user input text.
            tool_calls: Names of tools invoked during the response (may be
                empty for turns that needed no tools).
            metadata: Optional dict with extra context (channel, provider…).

        Returns:
            The persisted :class:`RequestEvent`.
        """
        event = RequestEvent.create(
            session_id=session_id,
            raw_message=user_message,
            tool_calls=tool_calls,
            metadata=metadata,
        )
        with self._lock:
            conn = self._connect()
            try:
                conn.execute(
                    """
                    INSERT INTO request_events
                        (id, session_id, timestamp, normalised, pattern_key,
                         tool_calls_json, metadata_json)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        event.id,
                        event.session_id,
                        event.timestamp,
                        event.normalised,
                        event.pattern_key,
                        json.dumps(event.tool_calls),
                        json.dumps(event.metadata),
                    ),
                )
                conn.commit()
            finally:
                conn.close()
        logger.debug("RequestTracker: recorded event %s pattern=%s", event.id, event.pattern_key)
        return event

    def get_frequent_patterns(
        self,
        min_count: int = DEFAULT_MIN_COUNT,
        limit: int = 20,
    ) -> list[RequestPattern]:
        """Return patterns that have been observed at least *min_count* times.

        Results are sorted by ``frequency_score`` descending.

        Args:
            min_count: Minimum raw occurrence count to surface a pattern.
            limit: Maximum number of patterns to return.

        Returns:
            List of :class:`RequestPattern` sorted by score descending.
        """
        with self._lock:
            conn = self._connect()
            try:
                rows = conn.execute(
                    """
                    SELECT
                        pattern_key,
                        COUNT(*) AS cnt,
                        MIN(timestamp) AS first_seen,
                        MAX(timestamp) AS last_seen,
                        GROUP_CONCAT(tool_calls_json, '||') AS all_tools,
                        GROUP_CONCAT(normalised, '||') AS all_normalised,
                        SUM(
                            CASE
                                WHEN julianday('now') - julianday(timestamp)
                                     <= ? THEN 1
                                ELSE 0
                            END
                        ) AS recent_cnt
                    FROM request_events
                    GROUP BY pattern_key
                    HAVING cnt >= ?
                    ORDER BY cnt DESC
                    LIMIT ?
                    """,
                    (DECAY_HALF_LIFE_DAYS, min_count, limit),
                ).fetchall()
            finally:
                conn.close()

        patterns: list[RequestPattern] = []
        for row in rows:
            key, cnt, first_seen, last_seen, all_tools_raw, all_norm_raw, recent = row
            all_tools = _flatten_tool_lists(all_tools_raw or "")
            tool_counts: dict[str, int] = {}
            for t in all_tools:
                tool_counts[t] = tool_counts.get(t, 0) + 1
            common_tools = [t for t, c in tool_counts.items() if c >= cnt / 2]

            norm_samples = (all_norm_raw or "").split("||")
            representative = norm_samples[0] if norm_samples else ""

            score = min(1.0, recent / max(cnt, 1))
            # Weight by absolute count so very frequent patterns score higher.
            score = score * min(1.0, cnt / 10.0) + (1.0 - score) * (recent / max(cnt, 1)) * 0.5
            score = min(1.0, score)

            examples = _pick_examples(norm_samples)

            patterns.append(
                RequestPattern(
                    pattern_key=key,
                    representative=representative,
                    count=cnt,
                    recent_count=recent,
                    frequency_score=round(score, 4),
                    common_tools=sorted(set(common_tools)),
                    first_seen=first_seen,
                    last_seen=last_seen,
                    example_messages=examples,
                )
            )

        patterns.sort(key=lambda p: p.frequency_score, reverse=True)
        return patterns

    def pattern_count(self) -> int:
        """Return the number of distinct patterns recorded."""
        with self._lock:
            conn = self._connect()
            try:
                (cnt,) = conn.execute(
                    "SELECT COUNT(DISTINCT pattern_key) FROM request_events"
                ).fetchone()
                return cnt
            finally:
                conn.close()

    def event_count(self) -> int:
        """Return the total number of recorded events."""
        with self._lock:
            conn = self._connect()
            try:
                (cnt,) = conn.execute("SELECT COUNT(*) FROM request_events").fetchone()
                return cnt
            finally:
                conn.close()

    def purge_before(self, before_iso: str) -> int:
        """Delete all events older than *before_iso* (ISO-8601 UTC string).

        Returns the number of rows deleted.
        """
        with self._lock:
            conn = self._connect()
            try:
                cur = conn.execute("DELETE FROM request_events WHERE timestamp < ?", (before_iso,))
                conn.commit()
                return cur.rowcount
            finally:
                conn.close()

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self._path), timeout=10)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_schema(self) -> None:
        with self._lock:
            conn = self._connect()
            try:
                conn.executescript(
                    """
                    PRAGMA journal_mode=WAL;
                    PRAGMA foreign_keys=ON;

                    CREATE TABLE IF NOT EXISTS request_events (
                        id            TEXT PRIMARY KEY,
                        session_id    TEXT NOT NULL,
                        timestamp     TEXT NOT NULL,
                        normalised    TEXT NOT NULL,
                        pattern_key   TEXT NOT NULL,
                        tool_calls_json TEXT NOT NULL DEFAULT '[]',
                        metadata_json   TEXT NOT NULL DEFAULT '{}'
                    );

                    CREATE INDEX IF NOT EXISTS idx_events_pattern
                        ON request_events (pattern_key);
                    CREATE INDEX IF NOT EXISTS idx_events_ts
                        ON request_events (timestamp);
                    """
                )
                conn.commit()
            finally:
                conn.close()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _flatten_tool_lists(raw: str) -> list[str]:
    """Flatten GROUP_CONCAT-ed JSON tool-call arrays into a single list."""
    result: list[str] = []
    for part in raw.split("||"):
        part = part.strip()
        if not part:
            continue
        try:
            names = json.loads(part)
            if isinstance(names, list):
                result.extend(str(n) for n in names)
        except Exception:
            pass
    return result


def _pick_examples(normalised_samples: list[str], n: int = 3) -> list[str]:
    """Return up to *n* representative samples from *normalised_samples*."""
    seen: set[str] = set()
    out: list[str] = []
    for s in normalised_samples:
        if s not in seen:
            seen.add(s)
            out.append(s[:200])
        if len(out) >= n:
            break
    return out


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_tracker: RequestTracker | None = None
_tracker_lock = threading.Lock()


def get_request_tracker(db_path: Path | str | None = None) -> RequestTracker:
    """Return (or lazily create) the module-level :class:`RequestTracker`."""
    global _tracker
    with _tracker_lock:
        if _tracker is None:
            _tracker = RequestTracker(db_path=db_path)
        return _tracker
