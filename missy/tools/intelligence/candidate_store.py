"""Tool candidate lifecycle store.

A *tool candidate* is a structured proposal that arose from pattern
detection.  Every candidate passes through a well-defined lifecycle before
it may be used by the agent:

    proposed → experimental → benchmarked → approved → enabled
                                                    ↘ deprecated → disabled

At any point a candidate may be denied (→ disabled) with a reason.

The store is backed by SQLite at ``~/.missy/tool_candidates.db`` and
emits an audit event on every state transition.
"""

from __future__ import annotations

import enum
import json
import logging
import sqlite3
import threading
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from missy.core.events import AuditEvent, event_bus

logger = logging.getLogger(__name__)

_DEFAULT_DB = Path("~/.missy/tool_candidates.db")


class ToolLifecycleState(enum.StrEnum):
    """Ordered lifecycle states for a :class:`ToolCandidate`.

    Values match the string stored in SQLite so they can be used directly
    in SQL queries.
    """

    PROPOSED = "proposed"
    EXPERIMENTAL = "experimental"
    BENCHMARKED = "benchmarked"
    APPROVED = "approved"
    ENABLED = "enabled"
    DEPRECATED = "deprecated"
    DISABLED = "disabled"

    @classmethod
    def active_states(cls) -> tuple[ToolLifecycleState, ...]:
        """States where the tool is usable by the agent."""
        return (cls.EXPERIMENTAL, cls.APPROVED, cls.ENABLED)

    @classmethod
    def terminal_states(cls) -> tuple[ToolLifecycleState, ...]:
        """States where no further transitions are expected."""
        return (cls.DISABLED,)


_ALLOWED_TRANSITIONS: dict[ToolLifecycleState, frozenset[ToolLifecycleState]] = {
    ToolLifecycleState.PROPOSED: frozenset(
        {
            ToolLifecycleState.EXPERIMENTAL,
            ToolLifecycleState.BENCHMARKED,
            ToolLifecycleState.DISABLED,
        }
    ),
    ToolLifecycleState.EXPERIMENTAL: frozenset(
        {ToolLifecycleState.BENCHMARKED, ToolLifecycleState.DISABLED}
    ),
    ToolLifecycleState.BENCHMARKED: frozenset(
        {ToolLifecycleState.APPROVED, ToolLifecycleState.DISABLED}
    ),
    ToolLifecycleState.APPROVED: frozenset(
        {
            ToolLifecycleState.ENABLED,
            ToolLifecycleState.DEPRECATED,
            ToolLifecycleState.DISABLED,
        }
    ),
    ToolLifecycleState.ENABLED: frozenset(
        {ToolLifecycleState.DEPRECATED, ToolLifecycleState.DISABLED}
    ),
    ToolLifecycleState.DEPRECATED: frozenset(
        {ToolLifecycleState.DISABLED, ToolLifecycleState.ENABLED}
    ),
    ToolLifecycleState.DISABLED: frozenset(),
}


def is_valid_transition(
    current: ToolLifecycleState,
    new_state: ToolLifecycleState,
) -> bool:
    """Return whether a candidate may move from *current* to *new_state*.

    The lifecycle is intentionally monotonic through review and benchmark
    gates. Rollback remains possible via ``deprecated`` or ``disabled``, but
    terminally disabled candidates cannot be resurrected in place.
    """
    if current == new_state:
        return True
    return new_state in _ALLOWED_TRANSITIONS[current]


@dataclass
class BenchmarkSummary:
    """Condensed benchmark outcome stored inline on a candidate."""

    provider: str
    correctness: float = 0.0
    latency_ms: float = 0.0
    cost_usd: float = 0.0
    reliability: float = 0.0
    safety: float = 0.0
    schema_score: float = 0.0
    composite: float = 0.0
    run_at: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "provider": self.provider,
            "correctness": self.correctness,
            "latency_ms": self.latency_ms,
            "cost_usd": self.cost_usd,
            "reliability": self.reliability,
            "safety": self.safety,
            "schema_score": self.schema_score,
            "composite": self.composite,
            "run_at": self.run_at,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> BenchmarkSummary:
        return cls(**{k: d.get(k, v) for k, v in cls.__dataclass_fields__.items()})  # type: ignore[attr-defined]


@dataclass
class ToolCandidate:
    """A proposed or active structured tool.

    Attributes:
        id: Unique UUID string.
        name: Short identifier (``[a-z0-9_-]+``).
        description: One-sentence description shown in tool schemas.
        schema: JSON Schema dict describing the tool's parameters.
        permissions: Dict mapping permission names to bools (network, shell…).
        provenance: Description of how this candidate was generated.
        pattern_key: Pattern hash from :class:`~.request_tracker.RequestTracker`.
        examples: List of example ``{input: …, expected_output: …}`` dicts.
        version: Monotonic integer incremented on schema changes.
        owner: Identity of whoever approved/created the candidate.
        state: Current :class:`ToolLifecycleState`.
        created_at: UTC ISO-8601 creation time.
        updated_at: UTC ISO-8601 last update time.
        notes: Human-readable notes (denial reason, benchmark summary…).
        benchmark_scores: Per-provider :class:`BenchmarkSummary` instances.
        provider_enabled: Per-provider enablement flag based on benchmarks.
        implementation: Explicit runtime binding metadata. Empty candidates
            remain review metadata only and must not be loaded for execution.
        tags: Arbitrary string labels.
    """

    id: str
    name: str
    description: str
    schema: dict[str, Any]
    permissions: dict[str, bool]
    provenance: str
    pattern_key: str = ""
    examples: list[dict[str, Any]] = field(default_factory=list)
    version: int = 1
    owner: str = "agent"
    state: ToolLifecycleState = ToolLifecycleState.PROPOSED
    created_at: str = ""
    updated_at: str = ""
    notes: str = ""
    benchmark_scores: list[BenchmarkSummary] = field(default_factory=list)
    provider_enabled: dict[str, bool] = field(default_factory=dict)
    implementation: dict[str, Any] = field(default_factory=dict)
    tags: list[str] = field(default_factory=list)

    @classmethod
    def create(
        cls,
        name: str,
        description: str,
        schema: dict[str, Any],
        permissions: dict[str, bool] | None = None,
        provenance: str = "",
        pattern_key: str = "",
        examples: list[dict[str, Any]] | None = None,
        owner: str = "agent",
        tags: list[str] | None = None,
        implementation: dict[str, Any] | None = None,
    ) -> ToolCandidate:
        now = datetime.now(UTC).isoformat()
        return cls(
            id=str(uuid.uuid4()),
            name=name,
            description=description,
            schema=schema,
            permissions=dict(permissions or {}),
            provenance=provenance,
            pattern_key=pattern_key,
            examples=list(examples or []),
            owner=owner,
            state=ToolLifecycleState.PROPOSED,
            created_at=now,
            updated_at=now,
            implementation=dict(implementation or {}),
            tags=list(tags or []),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "schema": self.schema,
            "permissions": self.permissions,
            "provenance": self.provenance,
            "pattern_key": self.pattern_key,
            "examples": self.examples,
            "version": self.version,
            "owner": self.owner,
            "state": self.state.value,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "notes": self.notes,
            "benchmark_scores": [b.to_dict() for b in self.benchmark_scores],
            "provider_enabled": self.provider_enabled,
            "implementation": self.implementation,
            "tags": self.tags,
        }

    @classmethod
    def from_row(cls, row: sqlite3.Row) -> ToolCandidate:
        scores_raw = json.loads(row["benchmark_scores_json"] or "[]")
        benchmark_scores = [BenchmarkSummary.from_dict(s) for s in scores_raw]
        return cls(
            id=row["id"],
            name=row["name"],
            description=row["description"],
            schema=json.loads(row["schema_json"] or "{}"),
            permissions=json.loads(row["permissions_json"] or "{}"),
            provenance=row["provenance"] or "",
            pattern_key=row["pattern_key"] or "",
            examples=json.loads(row["examples_json"] or "[]"),
            version=row["version"],
            owner=row["owner"] or "agent",
            state=ToolLifecycleState(row["state"]),
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            notes=row["notes"] or "",
            benchmark_scores=benchmark_scores,
            provider_enabled=json.loads(row["provider_enabled_json"] or "{}"),
            implementation=json.loads(row["implementation_json"] or "{}"),
            tags=json.loads(row["tags_json"] or "[]"),
        )


class CandidateStore:
    """SQLite-backed store for tool candidates with lifecycle tracking.

    Args:
        db_path: SQLite file path (expanded with :func:`~pathlib.Path.expanduser`).
    """

    def __init__(self, db_path: Path | str | None = None) -> None:
        self._path = Path(db_path or _DEFAULT_DB).expanduser()
        self._lock = threading.Lock()
        self._path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        self._init_schema()

    # ------------------------------------------------------------------
    # CRUD
    # ------------------------------------------------------------------

    def add(self, candidate: ToolCandidate) -> ToolCandidate:
        """Persist a new *candidate* and emit a ``tool.candidate.proposed`` audit event."""
        with self._lock:
            conn = self._connect()
            try:
                conn.execute(
                    """
                    INSERT INTO tool_candidates (
                        id, name, description, schema_json, permissions_json,
                        provenance, pattern_key, examples_json, version,
                        owner, state, created_at, updated_at, notes,
                        benchmark_scores_json, provider_enabled_json,
                        implementation_json, tags_json
                    ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                    """,
                    (
                        candidate.id,
                        candidate.name,
                        candidate.description,
                        json.dumps(candidate.schema),
                        json.dumps(candidate.permissions),
                        candidate.provenance,
                        candidate.pattern_key,
                        json.dumps(candidate.examples),
                        candidate.version,
                        candidate.owner,
                        candidate.state.value,
                        candidate.created_at,
                        candidate.updated_at,
                        candidate.notes,
                        json.dumps([b.to_dict() for b in candidate.benchmark_scores]),
                        json.dumps(candidate.provider_enabled),
                        json.dumps(candidate.implementation),
                        json.dumps(candidate.tags),
                    ),
                )
                conn.commit()
            finally:
                conn.close()
        _emit_audit("tool.candidate.proposed", {"id": candidate.id, "name": candidate.name})
        logger.info("CandidateStore: added candidate %s (%s)", candidate.name, candidate.id)
        return candidate

    def get(self, candidate_id: str) -> ToolCandidate | None:
        """Return the candidate with *candidate_id*, or ``None``."""
        with self._lock:
            conn = self._connect()
            try:
                row = conn.execute(
                    "SELECT * FROM tool_candidates WHERE id = ?", (candidate_id,)
                ).fetchone()
                return ToolCandidate.from_row(row) if row else None
            finally:
                conn.close()

    def get_by_name(self, name: str) -> ToolCandidate | None:
        """Return the most-recently-updated candidate with *name*, or ``None``."""
        with self._lock:
            conn = self._connect()
            try:
                row = conn.execute(
                    "SELECT * FROM tool_candidates WHERE name = ? ORDER BY updated_at DESC LIMIT 1",
                    (name,),
                ).fetchone()
                return ToolCandidate.from_row(row) if row else None
            finally:
                conn.close()

    def get_by_pattern_key(self, pattern_key: str) -> ToolCandidate | None:
        """Return the most-recently-updated candidate generated from *pattern_key*.

        Used by automatic candidate generation to avoid proposing duplicate
        candidates for a pattern that has already been synthesized.

        Args:
            pattern_key: Pattern hash from :class:`~.request_tracker.RequestTracker`.

        Returns:
            The matching :class:`ToolCandidate`, or ``None`` if not found.
        """
        if not pattern_key:
            return None
        with self._lock:
            conn = self._connect()
            try:
                row = conn.execute(
                    "SELECT * FROM tool_candidates WHERE pattern_key = ? "
                    "ORDER BY updated_at DESC LIMIT 1",
                    (pattern_key,),
                ).fetchone()
                return ToolCandidate.from_row(row) if row else None
            finally:
                conn.close()

    def list_all(
        self,
        state: ToolLifecycleState | None = None,
        owner: str | None = None,
        limit: int = 100,
    ) -> list[ToolCandidate]:
        """Return candidates filtered by *state* and/or *owner*.

        Args:
            state: If given, only return candidates in this lifecycle state.
            owner: If given, only return candidates with this owner string.
            limit: Maximum number of results.

        Returns:
            List sorted by ``updated_at`` descending.
        """
        clauses: list[str] = []
        params: list[Any] = []
        if state is not None:
            clauses.append("state = ?")
            params.append(state.value)
        if owner is not None:
            clauses.append("owner = ?")
            params.append(owner)
        where = ("WHERE " + " AND ".join(clauses)) if clauses else ""
        params.append(limit)
        with self._lock:
            conn = self._connect()
            try:
                rows = conn.execute(
                    f"SELECT * FROM tool_candidates {where} ORDER BY updated_at DESC LIMIT ?",
                    params,
                ).fetchall()
                return [ToolCandidate.from_row(r) for r in rows]
            finally:
                conn.close()

    def transition(
        self,
        candidate_id: str,
        new_state: ToolLifecycleState,
        notes: str = "",
        actor: str = "agent",
    ) -> ToolCandidate | None:
        """Move *candidate_id* to *new_state*.

        Emits a ``tool.candidate.<new_state>`` audit event.

        Args:
            candidate_id: UUID of the target candidate.
            new_state: Destination lifecycle state.
            notes: Human-readable context for the transition (denial reason…).
            actor: Identity performing the transition.

        Returns:
            Updated :class:`ToolCandidate`, or ``None`` if not found.

        Raises:
            ValueError: If the requested transition would skip a lifecycle
                gate, resurrect a disabled candidate, or otherwise violate
                the documented candidate lifecycle.
        """
        now = datetime.now(UTC).isoformat()
        with self._lock:
            conn = self._connect()
            try:
                existing = conn.execute(
                    "SELECT state FROM tool_candidates WHERE id = ?", (candidate_id,)
                ).fetchone()
                if not existing:
                    return None
                old_state = ToolLifecycleState(existing["state"])
                if not is_valid_transition(old_state, new_state):
                    _emit_audit(
                        "tool.candidate.transition_denied",
                        {
                            "id": candidate_id,
                            "previous_state": old_state.value,
                            "requested_state": new_state.value,
                            "actor": actor,
                            "notes": notes,
                        },
                        result="deny",
                    )
                    raise ValueError(
                        f"invalid tool candidate transition: {old_state.value} -> {new_state.value}"
                    )
                conn.execute(
                    """
                    UPDATE tool_candidates
                       SET state = ?, updated_at = ?, notes = ?
                     WHERE id = ?
                    """,
                    (new_state.value, now, notes, candidate_id),
                )
                conn.commit()
                row = conn.execute(
                    "SELECT * FROM tool_candidates WHERE id = ?", (candidate_id,)
                ).fetchone()
                candidate = ToolCandidate.from_row(row)
            finally:
                conn.close()
        _emit_audit(
            f"tool.candidate.{new_state.value}",
            {
                "id": candidate_id,
                "name": candidate.name,
                "previous_state": old_state.value,
                "actor": actor,
                "notes": notes,
            },
        )
        logger.info(
            "CandidateStore: %s → %s (actor=%s)",
            candidate.name,
            new_state.value,
            actor,
        )
        return candidate

    def update_benchmark(
        self,
        candidate_id: str,
        summary: BenchmarkSummary,
        provider_enabled: dict[str, bool] | None = None,
    ) -> ToolCandidate | None:
        """Attach or update a :class:`BenchmarkSummary` for *candidate_id*.

        If *provider_enabled* is supplied, merges it into the stored map.
        After updating, automatically transitions the candidate to
        :attr:`~ToolLifecycleState.BENCHMARKED` if it is currently
        ``proposed`` or ``experimental``.
        """
        now = datetime.now(UTC).isoformat()
        with self._lock:
            conn = self._connect()
            try:
                row = conn.execute(
                    "SELECT * FROM tool_candidates WHERE id = ?", (candidate_id,)
                ).fetchone()
                if row is None:
                    return None
                existing_scores: list[dict] = json.loads(row["benchmark_scores_json"] or "[]")
                # Replace entry for same provider or append.
                updated = [s for s in existing_scores if s.get("provider") != summary.provider]
                updated.append(summary.to_dict())

                existing_enabled: dict = json.loads(row["provider_enabled_json"] or "{}")
                if provider_enabled:
                    existing_enabled.update(provider_enabled)

                old_state = row["state"]
                new_state = old_state
                if old_state in (
                    ToolLifecycleState.PROPOSED.value,
                    ToolLifecycleState.EXPERIMENTAL.value,
                ):
                    new_state = ToolLifecycleState.BENCHMARKED.value

                conn.execute(
                    """
                    UPDATE tool_candidates
                       SET benchmark_scores_json = ?,
                           provider_enabled_json = ?,
                           state = ?,
                           updated_at = ?
                     WHERE id = ?
                    """,
                    (
                        json.dumps(updated),
                        json.dumps(existing_enabled),
                        new_state,
                        now,
                        candidate_id,
                    ),
                )
                conn.commit()
                row = conn.execute(
                    "SELECT * FROM tool_candidates WHERE id = ?", (candidate_id,)
                ).fetchone()
                candidate = ToolCandidate.from_row(row)
            finally:
                conn.close()
        _emit_audit(
            "tool.candidate.benchmark_updated",
            {"id": candidate_id, "provider": summary.provider, "composite": summary.composite},
        )
        return candidate

    def delete(self, candidate_id: str, actor: str = "agent") -> bool:
        """Permanently delete a candidate.  Returns ``True`` if found."""
        with self._lock:
            conn = self._connect()
            try:
                cur = conn.execute("DELETE FROM tool_candidates WHERE id = ?", (candidate_id,))
                conn.commit()
                deleted = cur.rowcount > 0
            finally:
                conn.close()
        if deleted:
            _emit_audit("tool.candidate.deleted", {"id": candidate_id, "actor": actor})
        return deleted

    def count(self, state: ToolLifecycleState | None = None) -> int:
        """Return count of candidates, optionally filtered by *state*."""
        with self._lock:
            conn = self._connect()
            try:
                if state is not None:
                    (cnt,) = conn.execute(
                        "SELECT COUNT(*) FROM tool_candidates WHERE state = ?", (state.value,)
                    ).fetchone()
                else:
                    (cnt,) = conn.execute("SELECT COUNT(*) FROM tool_candidates").fetchone()
                return cnt
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

                    CREATE TABLE IF NOT EXISTS tool_candidates (
                        id                   TEXT PRIMARY KEY,
                        name                 TEXT NOT NULL,
                        description          TEXT NOT NULL DEFAULT '',
                        schema_json          TEXT NOT NULL DEFAULT '{}',
                        permissions_json     TEXT NOT NULL DEFAULT '{}',
                        provenance           TEXT NOT NULL DEFAULT '',
                        pattern_key          TEXT NOT NULL DEFAULT '',
                        examples_json        TEXT NOT NULL DEFAULT '[]',
                        version              INTEGER NOT NULL DEFAULT 1,
                        owner                TEXT NOT NULL DEFAULT 'agent',
                        state                TEXT NOT NULL DEFAULT 'proposed',
                        created_at           TEXT NOT NULL,
                        updated_at           TEXT NOT NULL,
                        notes                TEXT NOT NULL DEFAULT '',
                        benchmark_scores_json TEXT NOT NULL DEFAULT '[]',
                        provider_enabled_json TEXT NOT NULL DEFAULT '{}',
                        implementation_json  TEXT NOT NULL DEFAULT '{}',
                        tags_json            TEXT NOT NULL DEFAULT '[]'
                    );

                    CREATE INDEX IF NOT EXISTS idx_candidates_state
                        ON tool_candidates (state);
                    CREATE INDEX IF NOT EXISTS idx_candidates_name
                        ON tool_candidates (name);
                    """
                )
                columns = {
                    row["name"]
                    for row in conn.execute("PRAGMA table_info(tool_candidates)").fetchall()
                }
                if "implementation_json" not in columns:
                    conn.execute(
                        "ALTER TABLE tool_candidates "
                        "ADD COLUMN implementation_json TEXT NOT NULL DEFAULT '{}'"
                    )
                conn.commit()
            finally:
                conn.close()


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_store: CandidateStore | None = None
_store_lock = threading.Lock()


def get_candidate_store(db_path: Path | str | None = None) -> CandidateStore:
    """Return (or lazily create) the module-level :class:`CandidateStore`."""
    global _store
    with _store_lock:
        if _store is None:
            _store = CandidateStore(db_path=db_path)
        return _store


# ---------------------------------------------------------------------------
# Internal helper
# ---------------------------------------------------------------------------


def _emit_audit(event_type: str, detail: dict[str, Any], result: str = "allow") -> None:
    try:
        event_bus.publish(
            AuditEvent.now(
                session_id="",
                task_id="",
                event_type=event_type,
                category="tool",
                result=result,
                detail=detail,
            )
        )
    except Exception as exc:  # noqa: BLE001
        logger.debug("CandidateStore: audit emit failed: %s", exc)
