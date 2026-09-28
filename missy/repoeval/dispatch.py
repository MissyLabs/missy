"""Opt-in, durable, standalone RepoEval dispatch outbox.

No production scheduler is wired here. Each irreversible mutation gets a
committed attempt marker before the external call. Unknown effects are never
replayed without an independently observed exact scheduler job.
"""

from __future__ import annotations

import hashlib
import json
import re
import sqlite3
from collections.abc import Callable, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol


class DispatchError(Exception):
    pass


class DispatchConflict(DispatchError):
    pass


@dataclass(frozen=True)
class JobObservation:
    job: Mapping[str, Any]
    state: str  # pending, running, complete, failed, stopped


class Scheduler(Protocol):
    def submit(self, namespace: str, job: Mapping[str, Any]) -> None: ...
    def lookup(self, namespace: str, job_id: str) -> JobObservation | None: ...
    def stop(self, namespace: str, job_id: str) -> None: ...


_RUN = re.compile(r"^run-[A-Za-z0-9_-]{8,128}$")
_KEY = re.compile(r"^[A-Za-z0-9._:-]{8,128}$")
_SCOPE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,127}$")
_STATES = {"pending", "running", "complete", "failed", "stopped"}


def _json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _hash(value: Any) -> str:
    return hashlib.sha256(_json(value).encode()).hexdigest()


class SQLiteOutbox:
    """Explicitly initialized SQLite reference store, not PostgresStore."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        if str(self.path) == ":memory:":
            raise ValueError("on-disk database required")
        if not self.path.is_file():
            raise DispatchError("database not initialized")

    @classmethod
    def initialize(cls, path: str | Path) -> SQLiteOutbox:
        path = Path(path)
        if str(path) == ":memory:":
            raise ValueError("on-disk database required")
        with sqlite3.connect(path, timeout=30) as db:
            db.execute("PRAGMA journal_mode=WAL")
            db.executescript("""
                CREATE TABLE IF NOT EXISTS dispatch_outbox (
                    project_id TEXT NOT NULL, run_id TEXT NOT NULL,
                    key_hash TEXT NOT NULL, namespace TEXT NOT NULL,
                    job_id TEXT NOT NULL, job_json TEXT NOT NULL,
                    job_hash TEXT NOT NULL, state TEXT NOT NULL,
                    cancel_requested INTEGER NOT NULL DEFAULT 0,
                    stop_attempted INTEGER NOT NULL DEFAULT 0,
                    version INTEGER NOT NULL DEFAULT 0,
                    PRIMARY KEY(project_id,run_id),
                    UNIQUE(project_id,key_hash), UNIQUE(namespace,job_id)
                );
            """)
        return cls(path)

    @contextmanager
    def transaction(self):
        with sqlite3.connect(self.path, timeout=30) as db:
            db.row_factory = sqlite3.Row
            db.execute("BEGIN IMMEDIATE")
            try:
                yield db
            except BaseException:
                db.rollback()
                raise
            else:
                db.commit()

    def get(self, project_id: str, run_id: str) -> dict[str, Any] | None:
        with sqlite3.connect(self.path, timeout=30) as db:
            db.row_factory = sqlite3.Row
            row = db.execute(
                "SELECT * FROM dispatch_outbox WHERE project_id=? AND run_id=?",
                (project_id, run_id),
            ).fetchone()
        return dict(row) if row else None


class DurableDispatcher:
    """Standalone state foundation, not a production authorization adapter.

    authorize must independently verify the exact job against current project
    rights, immutable snapshot, approved image/worker, quota and placement.
    Never pass a caller-controlled predicate as production authorization.
    """

    def __init__(
        self,
        store: SQLiteOutbox,
        scheduler: Scheduler,
        authorize: Callable[[str, str, Mapping[str, Any]], bool],
        *,
        enabled: bool = False,
    ) -> None:
        self.store, self.scheduler, self.authorize, self.enabled = (
            store,
            scheduler,
            authorize,
            enabled,
        )

    def _enabled(self) -> None:
        if not self.enabled:
            raise DispatchError("durable dispatch disabled")

    def status(self, project_id: str, run_id: str) -> dict[str, Any]:
        row = self.store.get(project_id, run_id)
        if row is None:
            raise DispatchError("unknown run")
        # Internal interface. API must enforce project-scoped read permissions.
        return {k: v for k, v in row.items() if k not in {"job_json", "key_hash"}}

    @staticmethod
    def _row(db, project_id, run_id):
        row = db.execute(
            "SELECT * FROM dispatch_outbox WHERE project_id=? AND run_id=?",
            (project_id, run_id),
        ).fetchone()
        if row is None:
            raise DispatchError("unknown run")
        return row

    @staticmethod
    def _update(db, row, state, **fields):
        changes = {"state": state, **fields}
        updated = db.execute(
            f"UPDATE dispatch_outbox SET {','.join(f'{k}=?' for k in changes)},version=version+1 WHERE project_id=? AND run_id=? AND version=?",
            (*changes.values(), row["project_id"], row["run_id"], row["version"]),
        )
        if updated.rowcount != 1:
            raise DispatchConflict("outbox row changed concurrently")

    def reserve(
        self, project_id: str, run_id: str, idempotency_key: str, job: Mapping[str, Any]
    ) -> dict[str, Any]:
        """Commit immutable idempotency/job reservation without external effects."""
        self._enabled()
        if not isinstance(project_id, str) or not _SCOPE.fullmatch(project_id):
            raise ValueError("invalid project")
        if not isinstance(run_id, str) or not _RUN.fullmatch(run_id):
            raise ValueError("invalid run ID")
        if not isinstance(idempotency_key, str) or not _KEY.fullmatch(idempotency_key):
            raise ValueError("invalid idempotency key")
        if not isinstance(job, Mapping):
            raise ValueError("invalid job")
        try:
            normalized = json.loads(_json(job))
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("job must be finite JSON") from exc
        namespace, job_id = normalized.get("Namespace"), normalized.get("ID")
        if (
            not isinstance(namespace, str)
            or not _SCOPE.fullmatch(namespace)
            or job_id != f"foundry-{run_id}"
            or normalized.get("Name") != job_id
            or normalized.get("Type") != "batch"
            or not isinstance(normalized.get("Meta"), dict)
            or normalized["Meta"].get("foundry_run_id") != run_id
        ):
            raise ValueError("job identity must match immutable run")
        if len(_json(normalized)) > 256_000:
            raise ValueError("job too large")
        if self.authorize(project_id, run_id, normalized) is not True:
            raise DispatchError("exact job not authorized")
        key_hash = hashlib.sha256(idempotency_key.encode()).hexdigest()
        with self.store.transaction() as db:
            prior = db.execute(
                "SELECT * FROM dispatch_outbox WHERE project_id=? AND key_hash=?",
                (project_id, key_hash),
            ).fetchone()
            if prior:
                if prior["run_id"] != run_id or prior["job_hash"] != _hash(normalized):
                    raise DispatchConflict("idempotency key reused with different job")
                return self.status(project_id, run_id)
            if db.execute(
                "SELECT 1 FROM dispatch_outbox WHERE project_id=? AND run_id=?",
                (project_id, run_id),
            ).fetchone():
                raise DispatchConflict("run ID already reserved")
            try:
                db.execute(
                    "INSERT INTO dispatch_outbox(project_id,run_id,key_hash,namespace,job_id,job_json,job_hash,state) VALUES(?,?,?,?,?,?,?,'reserved')",
                    (
                        project_id,
                        run_id,
                        key_hash,
                        namespace,
                        job_id,
                        _json(normalized),
                        _hash(normalized),
                    ),
                )
            except sqlite3.IntegrityError as exc:
                raise DispatchConflict("scheduler job ID already reserved") from exc
        return self.status(project_id, run_id)

    def dispatch(self, project_id: str, run_id: str) -> dict[str, Any]:
        """Only reserved runs may submit; an interrupted attempt is uncertain."""
        self._enabled()
        with self.store.transaction() as db:
            row = self._row(db, project_id, run_id)
            if row["state"] != "reserved" or row["cancel_requested"]:
                return self.status(project_id, run_id)
            job = json.loads(row["job_json"])
            if _hash(job) != row["job_hash"]:
                raise DispatchError("stored job integrity mismatch")
            if self.authorize(project_id, run_id, job) is not True:
                raise DispatchError("exact job authorization revoked")
            self._update(db, row, "dispatching")
            namespace = row["namespace"]
        try:
            self.scheduler.submit(namespace, job)
        except Exception:
            self._uncertain(project_id, run_id, "dispatching")
            # A timed-out submission may still have created the job. When a
            # concurrent cancellation exists, only exact lookup evidence can
            # authorize the first stop attempt; never resubmit to find out.
            status = self.status(project_id, run_id)
            return (
                self._reconcile_after_submit(project_id, run_id)
                if status["cancel_requested"] and status["state"] not in {"cancelled", "conflict"}
                else status
            )
        return self._reconcile_after_submit(project_id, run_id)

    def _reconcile_after_submit(self, project_id: str, run_id: str) -> dict[str, Any]:
        status = self.reconcile(project_id, run_id)
        # Cancellation can commit while the first scheduler lookup is in
        # flight. Its own lookup may have run before submission was visible;
        # re-read the exact job once if the competing version made us yield.
        if (
            status["cancel_requested"]
            and not status["stop_attempted"]
            and status["state"] not in {"cancelled", "conflict"}
        ):
            return self.reconcile(project_id, run_id)
        return status

    def _uncertain(self, project_id: str, run_id: str, expected: str) -> None:
        with self.store.transaction() as db:
            row = self._row(db, project_id, run_id)
            if row["state"] == expected:
                self._update(db, row, "uncertain")

    def _lookup(self, row) -> JobObservation | None:
        observed = self.scheduler.lookup(row["namespace"], row["job_id"])
        if observed is None:
            return None
        if not isinstance(observed, JobObservation) or not isinstance(observed.job, Mapping):
            raise DispatchConflict("scheduler returned malformed job evidence")
        try:
            observed_hash = _hash(observed.job)
        except (TypeError, ValueError, OverflowError) as exc:
            raise DispatchConflict("scheduler returned malformed job evidence") from exc
        if (
            observed.state not in _STATES
            or observed_hash != row["job_hash"]
            or observed.job.get("ID") != row["job_id"]
            or observed.job.get("Namespace") != row["namespace"]
        ):
            raise DispatchConflict("scheduler job differs from immutable reserved job")
        return observed

    def reconcile(self, project_id: str, run_id: str) -> dict[str, Any]:
        """Read scheduler evidence. No submit replay and no terminal verification."""
        self._enabled()
        snapshot = self.store.get(project_id, run_id)
        if snapshot is None:
            raise DispatchError("unknown run")
        if snapshot["state"] in {"reserved", "cancelled", "conflict", "failed"}:
            return self.status(project_id, run_id)
        conflict = False
        try:
            observed = self._lookup(snapshot)
        except DispatchConflict:
            observed, conflict = None, True
        except Exception:
            observed = None
        with self.store.transaction() as db:
            row = self._row(db, project_id, run_id)
            if row["version"] != snapshot["version"]:
                return self.status(project_id, run_id)
            if conflict:
                self._update(db, row, "conflict")
            elif observed is None:
                self._update(db, row, "stop_uncertain" if row["stop_attempted"] else "uncertain")
            elif observed.state == "stopped":
                self._update(db, row, "cancelled" if row["cancel_requested"] else "failed")
            elif row["cancel_requested"]:
                self._update(
                    db, row, "cancel_pending" if not row["stop_attempted"] else "stop_uncertain"
                )
            elif observed.state == "failed":
                self._update(db, row, "failed")
            elif observed.state == "complete":
                # Scheduler completion does not verify worker output or artifacts.
                self._update(db, row, "collecting")
            else:
                self._update(db, row, "running" if observed.state == "running" else "submitted")
        if conflict:
            raise DispatchConflict("scheduler job differs from immutable reserved job")
        return self._attempt_pending_stop(project_id, run_id)

    def _attempt_pending_stop(self, project_id: str, run_id: str) -> dict[str, Any]:
        """Claim one stop after exact lookup, before its external effect.

        Both dispatch completion and cancellation reconciliation reach here;
        SQLite serializes their claims across processes. A committed claim is
        never replayed, including when the stop call raises or we crash.
        """
        with self.store.transaction() as db:
            row = self._row(db, project_id, run_id)
            if (
                row["state"] != "cancel_pending"
                or not row["cancel_requested"]
                or row["stop_attempted"]
            ):
                return self.status(project_id, run_id)
            self._update(db, row, "stop_uncertain", stop_attempted=1)
            namespace, job_id = row["namespace"], row["job_id"]
        try:
            self.scheduler.stop(namespace, job_id)
        except Exception:
            return self.status(project_id, run_id)
        return self.reconcile(project_id, run_id)

    def cancel(self, project_id: str, run_id: str) -> dict[str, Any]:
        """Cancel reserved work or stop one exact observed job at most once."""
        self._enabled()
        cancelled_before_submit = False
        with self.store.transaction() as db:
            row = self._row(db, project_id, run_id)
            if row["state"] in {"cancelled", "failed", "conflict"}:
                return self.status(project_id, run_id)
            if row["state"] == "reserved":
                self._update(db, row, "cancelled", cancel_requested=1)
                cancelled_before_submit = True
            else:
                self._update(db, row, row["state"], cancel_requested=1)
        if cancelled_before_submit:
            return self.status(project_id, run_id)
        return self.reconcile(project_id, run_id)
