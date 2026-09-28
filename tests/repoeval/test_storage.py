import hashlib
from datetime import UTC, datetime

import pytest

from missy.repoeval.storage import (
    ConflictError,
    InvalidTransitionError,
    NotFoundError,
    PostgresStore,
    StorageError,
)


class FakeDB:
    """Query-aware small DB-API fake; checks tenant scope and immutable semantics."""

    def __init__(self):
        self.runs = {}
        self.allow = {}
        self.artifacts = {}
        self.tombstones = {}
        self.events = []

    def __call__(self):
        return Connection(self)


class Connection:
    def __init__(self, db):
        self.db = db

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def cursor(self):
        return Cursor(self.db)


class Cursor:
    def __init__(self, db):
        self.db = db
        self.rows = []
        self.description = []

    def execute(self, q, p=()):
        q = " ".join(q.split())
        self.rows = []
        if q.startswith("INSERT INTO runs"):
            project, run, manifest, mh, idem = p
            key = (project, idem)
            if key not in self.db.runs:
                self.db.runs[key] = {
                    "project_id": project,
                    "run_id": run,
                    "manifest": manifest,
                    "manifest_sha256": mh,
                    "idempotency_key_sha256": idem,
                    "state": "planned",
                    "state_detail": "{}",
                }
        elif q.startswith("SELECT project_id,run_id,manifest"):
            self.rows = [
                r.copy()
                for (project, idem), r in self.db.runs.items()
                if (project, idem) == (p[0], p[1])
            ]
        elif q.startswith("SELECT * FROM runs WHERE"):
            self.rows = [
                r.copy()
                for r in self.db.runs.values()
                if r["project_id"] == p[0] and r["run_id"] == p[1]
            ]
        elif q.startswith("SELECT state FROM runs"):
            self.rows = [
                {"state": r["state"]}
                for r in self.db.runs.values()
                if r["project_id"] == p[0] and r["run_id"] == p[1]
            ]
        elif q.startswith("UPDATE runs SET"):
            state, detail, project, run = p
            for r in self.db.runs.values():
                if r["project_id"] == project and r["run_id"] == run:
                    r.update(state=state, state_detail=detail)
        elif q.startswith("INSERT INTO run_state_events"):
            self.db.events.append(p)
        elif q.startswith("SELECT 1 FROM runs"):
            self.rows = (
                [(1,)]
                if any(
                    r["project_id"] == p[0] and r["run_id"] == p[1] for r in self.db.runs.values()
                )
                else []
            )
        elif q.startswith("INSERT INTO artifact_collection_allowlist"):
            self.db.allow.setdefault((p[0], p[1], p[2]), {"kind": p[3], "required": p[4]})
        elif q.startswith("SELECT kind,required"):
            self.rows = (
                [self.db.allow[(p[0], p[1], p[2])]] if (p[0], p[1], p[2]) in self.db.allow else []
            )
        elif q.startswith("SELECT kind FROM artifact_collection_allowlist"):
            self.rows = [
                {"kind": v["kind"]} for k, v in self.db.allow.items() if k == (p[0], p[1], p[2])
            ]
        elif q.startswith("INSERT INTO artifacts"):
            (
                project,
                aid,
                run,
                key,
                kind,
                uri,
                sha,
                size,
                media,
                producer,
                cls,
                ret,
                scan,
                red,
                created,
                expires,
                manifest,
            ) = p
            self.db.artifacts.setdefault(
                (project, aid),
                {
                    "project_id": project,
                    "artifact_id": aid,
                    "run_id": run,
                    "collection_key": key,
                    "kind": kind,
                    "uri": uri,
                    "sha256": sha,
                    "size_bytes": size,
                    "media_type": media,
                    "producer": producer,
                    "classification": cls,
                    "retention_class": ret,
                    "scan_state": scan,
                    "redaction_state": red,
                    "created_at": created,
                    "expires_at": expires,
                    "deleted_at": None,
                    "manifest": __import__("json").loads(manifest),
                },
            )
        elif q.startswith("SELECT * FROM artifacts WHERE project_id=%s AND artifact_id"):
            self.rows = [v.copy() for k, v in self.db.artifacts.items() if k == (p[0], p[1])]
        elif q.startswith("SELECT * FROM artifacts WHERE project_id=%s AND run_id"):
            self.rows = [
                v.copy()
                for k, v in self.db.artifacts.items()
                if k[0] == p[0] and v["run_id"] == p[1] and v["deleted_at"] is None
            ]
        elif q.startswith("SELECT artifact_id,run_id,sha256,expires_at,deleted_at,manifest"):
            self.rows = [v.copy() for k, v in self.db.artifacts.items() if k == (p[0], p[1])]
        elif q.startswith("UPDATE artifacts SET"):
            at, project, aid = p
            self.db.artifacts[(project, aid)]["deleted_at"] = at
        elif q.startswith("INSERT INTO artifact_tombstones"):
            project, aid, run, sha, reason, at = p
            self.db.tombstones.setdefault(
                (project, aid),
                {
                    "project_id": project,
                    "artifact_id": aid,
                    "run_id": run,
                    "sha256": sha,
                    "disposition": "expired",
                    "reason": reason,
                    "deleted_at": at,
                },
            )
        elif q.startswith("SELECT * FROM artifact_tombstones"):
            self.rows = [v.copy() for k, v in self.db.tombstones.items() if k == (p[0], p[1])]
        else:
            raise AssertionError(q)

    def fetchone(self):
        return self.rows.pop(0) if self.rows else None

    def fetchall(self):
        rows, self.rows = self.rows, []
        return rows


MANIFEST = {"run_id": "run-example0001", "workload": "w", "project_id": "p1"}


def artifact():
    return {
        "schema_version": "1.0",
        "artifact_id": "artifact-example0001",
        "run_id": "run-example0001",
        "kind": "report",
        "uri": "s3://x/sha256/" + hashlib.sha256(b"x").hexdigest(),
        "sha256": hashlib.sha256(b"x").hexdigest(),
        "size_bytes": 1,
        "media_type": "application/json",
        "producer": {"component": "test", "version": "1"},
        "classification": "internal",
        "retention_class": "dev",
        "scan_state": "clean",
        "redaction_state": "complete",
        "created_at": "2026-09-28T00:00:00Z",
        "expires_at": "2026-09-29T00:00:00Z",
    }


def verified_store(db, reader=lambda uri: b"x", clearance=lambda manifest, data: True):
    return PostgresStore(db, reader, clearance)


def test_run_idempotency_and_project_scope():
    db = FakeDB()
    store = PostgresStore(db)
    first = store.create_run("p1", MANIFEST, "key-one")
    assert store.create_run("p1", MANIFEST, "key-one")["run_id"] == first["run_id"]
    with pytest.raises(ConflictError):
        store.create_run("p1", {**MANIFEST, "run_id": "run-another0000"}, "key-one")
    with pytest.raises(NotFoundError):
        store.get_run("p2", first["run_id"])


def test_run_transition_enforces_monotonic_state_and_logs_event():
    db = FakeDB()
    store = PostgresStore(db)
    store.create_run("p1", MANIFEST, "key-one")
    store.transition_run("p1", MANIFEST["run_id"], "submitted", expected_state="planned")
    assert len(db.events) == 1
    with pytest.raises(InvalidTransitionError):
        store.transition_run("p1", MANIFEST["run_id"], "verified")


def test_collection_is_allowlisted_and_expiry_tombstone_is_scoped():
    db = FakeDB()
    store = verified_store(db)
    store.create_run("p1", MANIFEST, "key-one")
    store.register_collection_allowlist(
        "p1", MANIFEST["run_id"], [{"key": "report", "kind": "report", "required": True}]
    )
    saved = store.record_artifact("p1", MANIFEST["run_id"], artifact(), "report")
    assert saved["sha256"] == hashlib.sha256(b"x").hexdigest()
    assert len(store.list_artifacts("p1", MANIFEST["run_id"])) == 1
    with pytest.raises(StorageError):
        store.record_artifact(
            "p1", MANIFEST["run_id"], {**artifact(), "artifact_id": "artifact-outside000"}, "other"
        )
    tomb = store.expire_artifact("p1", "artifact-example0001", at=datetime(2026, 9, 30, tzinfo=UTC))
    assert tomb["disposition"] == "expired"
    with pytest.raises(NotFoundError):
        store.expire_artifact("p2", "artifact-example0001")


def test_artifacts_fail_closed_without_reader_or_matching_bytes():
    db = FakeDB()
    store = PostgresStore(db)
    store.create_run("p1", MANIFEST, "key-one")
    store.register_collection_allowlist(
        "p1", MANIFEST["run_id"], [{"key": "report", "kind": "report"}]
    )
    with pytest.raises(StorageError, match="reader"):
        store.record_artifact("p1", MANIFEST["run_id"], artifact(), "report")
    assert not db.artifacts
    bad = verified_store(db, lambda uri: b"z")
    with pytest.raises(StorageError, match="verification"):
        bad.record_artifact("p1", MANIFEST["run_id"], artifact(), "report")
    assert not db.artifacts
    with pytest.raises(StorageError, match="scanning"):
        bad.record_artifact(
            "p1", MANIFEST["run_id"], {**artifact(), "scan_state": "quarantined"}, "report"
        )
    good = verified_store(db)
    good.record_artifact("p1", MANIFEST["run_id"], artifact(), "report")
    assert len(good.list_artifacts("p1", MANIFEST["run_id"])) == 1
    assert good.list_artifacts("p2", MANIFEST["run_id"]) == []
    assert good.list_artifacts("p1", "run-other0000") == []
    with pytest.raises(StorageError, match="verification"):
        bad.list_artifacts("p1", MANIFEST["run_id"])
    db.artifacts[("p1", artifact()["artifact_id"])]["scan_state"] = "quarantined"
    assert good.list_artifacts("p1", MANIFEST["run_id"]) == []


def test_artifact_retry_rejects_changed_immutable_metadata():
    db = FakeDB()
    store = verified_store(db)
    store.create_run("p1", MANIFEST, "key-one")
    store.register_collection_allowlist(
        "p1", MANIFEST["run_id"], [{"key": "report", "kind": "report"}]
    )
    store.record_artifact("p1", MANIFEST["run_id"], artifact(), "report")
    with pytest.raises(ConflictError):
        store.record_artifact(
            "p1", MANIFEST["run_id"], {**artifact(), "classification": "restricted"}, "report"
        )


def test_claimed_clean_state_without_independent_clearance_is_refused():
    db = FakeDB()
    store = PostgresStore(db, lambda uri: b"x")
    store.create_run("p1", MANIFEST, "key-one")
    store.register_collection_allowlist(
        "p1", MANIFEST["run_id"], [{"key": "report", "kind": "report"}]
    )
    with pytest.raises(StorageError, match="clearance"):
        store.record_artifact("p1", MANIFEST["run_id"], artifact(), "report")
    with pytest.raises(StorageError, match="clearance"):
        verified_store(db, clearance=lambda manifest, data: False).record_artifact(
            "p1", MANIFEST["run_id"], artifact(), "report"
        )
    assert not db.artifacts
