"""Persistent store for benchmark results.

Benchmark runs are written to SQLite at ``~/.missy/benchmark_results.db``.
Each row records one :class:`~.scoring.BenchmarkResult` (raw) alongside
the scores produced by :class:`~.scoring.BenchmarkScorer`.

The store supports:
- inserting individual or batched results
- querying by tool, provider, date, or composite score threshold
- summarising per-provider scores for a given tool
"""

from __future__ import annotations

import json
import logging
import sqlite3
import threading
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .scoring import ScoredResult

logger = logging.getLogger(__name__)

_DEFAULT_DB = Path("~/.missy/benchmark_results.db")


@dataclass
class ProviderSummary:
    """Aggregated benchmark statistics for one (tool, provider) pair.

    Attributes:
        tool_name: Name of the benchmarked tool.
        provider: Provider identifier.
        run_count: Total number of results stored.
        mean_composite: Mean composite score across all runs.
        mean_correctness: Mean correctness score.
        mean_latency_ms: Mean wall-clock latency.
        mean_cost_usd: Mean estimated cost.
        mean_reliability: Mean reliability score.
        mean_safety: Mean safety score.
        last_run_at: ISO-8601 timestamp of the most recent run.
    """

    tool_name: str
    provider: str
    run_count: int
    mean_composite: float
    mean_correctness: float
    mean_latency_ms: float
    mean_cost_usd: float
    mean_reliability: float
    mean_safety: float
    last_run_at: str


class BenchmarkStore:
    """SQLite-backed store for :class:`~.scoring.ScoredResult` instances.

    Args:
        db_path: Path to the SQLite file (expanded with
            :func:`~pathlib.Path.expanduser`).
    """

    def __init__(self, db_path: Path | str | None = None) -> None:
        self._path = Path(db_path or _DEFAULT_DB).expanduser()
        self._lock = threading.Lock()
        self._path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        self._init_schema()

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def save(self, scored: ScoredResult) -> str:
        """Persist *scored* and return its assigned row ID (UUID)."""
        row_id = str(uuid.uuid4())
        r = scored.result
        now = datetime.now(UTC).isoformat()
        with self._lock:
            conn = self._connect()
            try:
                conn.execute(
                    """
                    INSERT INTO benchmark_results (
                        id, task_id, tool_name, provider,
                        success, latency_ms, cost_usd,
                        actual_output, expected_output,
                        tool_call_made, tool_call_args_json,
                        schema_required_params_json,
                        safety_violation, error,
                        correctness, latency_score, cost_score,
                        reliability, safety, schema_score,
                        tool_call_quality, composite,
                        metadata_json, recorded_at
                    ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                    """,
                    (
                        row_id,
                        r.task_id,
                        r.tool_name,
                        r.provider,
                        int(r.success),
                        r.latency_ms,
                        r.cost_usd,
                        str(r.actual_output) if r.actual_output is not None else None,
                        str(r.expected_output) if r.expected_output is not None else None,
                        int(r.tool_call_made),
                        json.dumps(r.tool_call_args),
                        json.dumps(r.schema_required_params),
                        int(r.safety_violation),
                        r.error or "",
                        scored.correctness,
                        scored.latency_score,
                        scored.cost_score,
                        scored.reliability,
                        scored.safety,
                        scored.schema_score,
                        scored.tool_call_quality,
                        scored.composite,
                        json.dumps(r.metadata),
                        now,
                    ),
                )
                conn.commit()
            finally:
                conn.close()
        logger.debug(
            "BenchmarkStore: saved result %s (tool=%s provider=%s)", row_id, r.tool_name, r.provider
        )
        return row_id

    def save_batch(self, scored_list: list[ScoredResult]) -> list[str]:
        """Persist all results in *scored_list* and return their IDs."""
        return [self.save(s) for s in scored_list]

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    def query(
        self,
        tool_name: str | None = None,
        provider: str | None = None,
        min_composite: float | None = None,
        since_iso: str | None = None,
        limit: int = 200,
    ) -> list[dict[str, Any]]:
        """Return stored results as raw dicts, newest first.

        Args:
            tool_name: Filter by tool name (exact match).
            provider: Filter by provider identifier (exact match).
            min_composite: Minimum composite score (inclusive).
            since_iso: Only return rows recorded after this ISO-8601 timestamp.
            limit: Maximum number of rows.

        Returns:
            List of row dicts ordered by ``recorded_at`` descending.
        """
        clauses: list[str] = []
        params: list[Any] = []
        if tool_name is not None:
            clauses.append("tool_name = ?")
            params.append(tool_name)
        if provider is not None:
            clauses.append("provider = ?")
            params.append(provider)
        if min_composite is not None:
            clauses.append("composite >= ?")
            params.append(min_composite)
        if since_iso is not None:
            clauses.append("recorded_at >= ?")
            params.append(since_iso)
        where = ("WHERE " + " AND ".join(clauses)) if clauses else ""
        params.append(limit)
        with self._lock:
            conn = self._connect()
            try:
                rows = conn.execute(
                    f"SELECT * FROM benchmark_results {where} ORDER BY recorded_at DESC LIMIT ?",
                    params,
                ).fetchall()
                return [dict(r) for r in rows]
            finally:
                conn.close()

    def provider_summary(self, tool_name: str) -> list[ProviderSummary]:
        """Return per-provider aggregate statistics for *tool_name*.

        Args:
            tool_name: Tool to aggregate.

        Returns:
            List of :class:`ProviderSummary` sorted by ``mean_composite`` descending.
        """
        with self._lock:
            conn = self._connect()
            try:
                rows = conn.execute(
                    """
                    SELECT
                        tool_name,
                        provider,
                        COUNT(*) AS run_count,
                        AVG(composite) AS mean_composite,
                        AVG(correctness) AS mean_correctness,
                        AVG(latency_ms) AS mean_latency_ms,
                        AVG(cost_usd) AS mean_cost_usd,
                        AVG(reliability) AS mean_reliability,
                        AVG(safety) AS mean_safety,
                        MAX(recorded_at) AS last_run_at
                    FROM benchmark_results
                    WHERE tool_name = ?
                    GROUP BY provider
                    ORDER BY mean_composite DESC
                    """,
                    (tool_name,),
                ).fetchall()
            finally:
                conn.close()
        return [
            ProviderSummary(
                tool_name=r["tool_name"],
                provider=r["provider"],
                run_count=r["run_count"],
                mean_composite=round(r["mean_composite"] or 0.0, 4),
                mean_correctness=round(r["mean_correctness"] or 0.0, 4),
                mean_latency_ms=round(r["mean_latency_ms"] or 0.0, 1),
                mean_cost_usd=round(r["mean_cost_usd"] or 0.0, 6),
                mean_reliability=round(r["mean_reliability"] or 0.0, 4),
                mean_safety=round(r["mean_safety"] or 0.0, 4),
                last_run_at=r["last_run_at"] or "",
            )
            for r in rows
        ]

    def delete_before(self, before_iso: str) -> int:
        """Delete rows recorded before *before_iso*.  Returns deleted count."""
        with self._lock:
            conn = self._connect()
            try:
                cur = conn.execute(
                    "DELETE FROM benchmark_results WHERE recorded_at < ?", (before_iso,)
                )
                conn.commit()
                return cur.rowcount
            finally:
                conn.close()

    def count(self, tool_name: str | None = None, provider: str | None = None) -> int:
        """Return row count, optionally scoped to *tool_name* / *provider*."""
        clauses: list[str] = []
        params: list[Any] = []
        if tool_name:
            clauses.append("tool_name = ?")
            params.append(tool_name)
        if provider:
            clauses.append("provider = ?")
            params.append(provider)
        where = ("WHERE " + " AND ".join(clauses)) if clauses else ""
        with self._lock:
            conn = self._connect()
            try:
                (cnt,) = conn.execute(
                    f"SELECT COUNT(*) FROM benchmark_results {where}", params
                ).fetchone()
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

                    CREATE TABLE IF NOT EXISTS benchmark_results (
                        id                      TEXT PRIMARY KEY,
                        task_id                 TEXT NOT NULL,
                        tool_name               TEXT NOT NULL,
                        provider                TEXT NOT NULL,
                        success                 INTEGER NOT NULL DEFAULT 0,
                        latency_ms              REAL NOT NULL DEFAULT 0,
                        cost_usd                REAL NOT NULL DEFAULT 0,
                        actual_output           TEXT,
                        expected_output         TEXT,
                        tool_call_made          INTEGER NOT NULL DEFAULT 0,
                        tool_call_args_json     TEXT NOT NULL DEFAULT '{}',
                        schema_required_params_json TEXT NOT NULL DEFAULT '[]',
                        safety_violation        INTEGER NOT NULL DEFAULT 0,
                        error                   TEXT NOT NULL DEFAULT '',
                        correctness             REAL NOT NULL DEFAULT 0,
                        latency_score           REAL NOT NULL DEFAULT 0,
                        cost_score              REAL NOT NULL DEFAULT 0,
                        reliability             REAL NOT NULL DEFAULT 0,
                        safety                  REAL NOT NULL DEFAULT 0,
                        schema_score            REAL NOT NULL DEFAULT 0,
                        tool_call_quality       REAL NOT NULL DEFAULT 0,
                        composite               REAL NOT NULL DEFAULT 0,
                        metadata_json           TEXT NOT NULL DEFAULT '{}',
                        recorded_at             TEXT NOT NULL
                    );

                    CREATE INDEX IF NOT EXISTS idx_br_tool
                        ON benchmark_results (tool_name);
                    CREATE INDEX IF NOT EXISTS idx_br_provider
                        ON benchmark_results (provider);
                    CREATE INDEX IF NOT EXISTS idx_br_recorded
                        ON benchmark_results (recorded_at);
                    """
                )
                conn.commit()
            finally:
                conn.close()


# ---------------------------------------------------------------------------
# Module singleton
# ---------------------------------------------------------------------------

_store: BenchmarkStore | None = None
_store_lock = threading.Lock()


def get_benchmark_store(db_path: Path | str | None = None) -> BenchmarkStore:
    """Return (or lazily create) the module-level :class:`BenchmarkStore`."""
    global _store
    with _store_lock:
        if _store is None:
            _store = BenchmarkStore(db_path=db_path)
        return _store
