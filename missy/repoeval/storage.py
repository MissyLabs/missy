"""PostgreSQL persistence primitives. No connection or migration is opened implicitly."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Callable, Mapping
from datetime import UTC, datetime
from typing import Any


class StorageError(Exception):
    pass


class ConflictError(StorageError):
    pass


class NotFoundError(StorageError):
    pass


class InvalidTransitionError(StorageError):
    pass


_RUN = re.compile(r"^run-[A-Za-z0-9_-]{8,128}$")
_SHA = re.compile(r"^[0-9a-f]{64}$")
_STATES = {
    "planned",
    "submitted",
    "running",
    "collecting",
    "verified",
    "failed",
    "cancelled",
    "incomparable",
}
_TRANSITIONS = {
    "planned": {"submitted", "cancelled", "failed"},
    "submitted": {"running", "cancelled", "failed"},
    "running": {"collecting", "cancelled", "failed"},
    "collecting": {"verified", "incomparable", "failed", "cancelled"},
}


def _json(value: Any) -> str:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    )


def _hash(value: Any) -> str:
    return hashlib.sha256(_json(value).encode()).hexdigest()


def _one(cur: Any) -> dict[str, Any] | None:
    row = cur.fetchone()
    if row is None:
        return None
    if isinstance(row, Mapping):
        return dict(row)
    return dict(zip((c[0] for c in cur.description), row, strict=True))


def _all(cur: Any) -> list[dict[str, Any]]:
    rows = cur.fetchall()
    if not rows:
        return []
    if isinstance(rows[0], Mapping):
        return [dict(x) for x in rows]
    names = [c[0] for c in cur.description]
    return [dict(zip(names, x, strict=True)) for x in rows]


class PostgresStore:
    """DB-API store; caller supplies an approved connection factory/pool."""

    def __init__(
        self,
        connect: Callable[[], Any],
        object_reader: Callable[[str], bytes] | None = None,
        clearance_verifier: Callable[[Mapping[str, Any], bytes], bool] | None = None,
    ):
        self._connect = connect
        self._object_reader = object_reader
        self._clearance_verifier = clearance_verifier

    def _verified_public_artifact(self, item: Mapping[str, Any]) -> dict[str, Any]:
        from .artifacts import ArtifactError, safe_to_expose, validate_manifest, verify_object_bytes

        # A catalog row is a claim, not evidence of object bytes or permission to expose them.
        if item.get("deleted_at") is not None or not safe_to_expose(item):
            raise StorageError("artifact is not cleared for access")
        try:
            manifest = validate_manifest(item["manifest"])
            if not safe_to_expose(manifest) or any(
                item[field] != manifest[field]
                for field in (
                    "artifact_id",
                    "run_id",
                    "kind",
                    "uri",
                    "sha256",
                    "size_bytes",
                    "media_type",
                    "classification",
                    "retention_class",
                    "scan_state",
                    "redaction_state",
                )
            ):
                raise StorageError("artifact catalog and manifest disagree")
            if self._object_reader is None:
                raise StorageError("verified artifact object reader is unavailable")
            data = self._object_reader(manifest["uri"])
            verify_object_bytes(manifest, data)
            # The producer's metadata alone cannot certify scanning or redaction.
            if (
                self._clearance_verifier is None
                or self._clearance_verifier(manifest, data) is not True
            ):
                raise StorageError("independent artifact clearance is unavailable")
        except (ArtifactError, KeyError, TypeError, OSError, ValueError) as exc:
            raise StorageError("artifact object verification failed") from exc
        return dict(item)

    def create_run(
        self, project_id: str, manifest: Mapping[str, Any], idempotency_key: str
    ) -> dict[str, Any]:
        if not project_id or not idempotency_key:
            raise ValueError("project_id and idempotency_key are required")
        data = dict(manifest)
        run_id = data.get("run_id")
        if not isinstance(run_id, str) or not _RUN.fullmatch(run_id):
            raise ValueError("valid run_id required")
        if data.get("project_id", project_id) != project_id:
            raise ValueError("manifest project_id does not match scope")
        data["project_id"] = project_id
        idem = hashlib.sha256(idempotency_key.encode()).hexdigest()
        mh = _hash(data)
        with self._connect() as conn:
            cur = conn.cursor()
            cur.execute(
                "INSERT INTO runs(project_id,run_id,manifest,manifest_sha256,idempotency_key_sha256,state) VALUES(%s,%s,%s::jsonb,%s,%s,'planned') ON CONFLICT(project_id,idempotency_key_sha256) DO NOTHING",
                (project_id, run_id, _json(data), mh, idem),
            )
            cur.execute(
                "SELECT project_id,run_id,manifest,manifest_sha256,idempotency_key_sha256,state,state_detail,created_at,updated_at FROM runs WHERE project_id=%s AND idempotency_key_sha256=%s",
                (project_id, idem),
            )
            result = _one(cur)
            if result is None:
                raise StorageError("run insert did not yield a record")
            if result["run_id"] != run_id or result["manifest_sha256"] != mh:
                raise ConflictError("idempotency key already belongs to a different immutable run")
            return result

    def get_run(self, project_id: str, run_id: str) -> dict[str, Any]:
        with self._connect() as conn:
            cur = conn.cursor()
            cur.execute(
                "SELECT * FROM runs WHERE project_id=%s AND run_id=%s", (project_id, run_id)
            )
            result = _one(cur)
        if result is None:
            raise NotFoundError("run not found in project")
        return result

    def transition_run(
        self,
        project_id: str,
        run_id: str,
        new_state: str,
        expected_state: str | None = None,
        details: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        if new_state not in _STATES:
            raise InvalidTransitionError("unknown run state")
        with self._connect() as conn:
            cur = conn.cursor()
            cur.execute(
                "SELECT state FROM runs WHERE project_id=%s AND run_id=%s FOR UPDATE",
                (project_id, run_id),
            )
            row = _one(cur)
            if row is None:
                raise NotFoundError("run not found in project")
            old = row["state"]
            if expected_state is not None and old != expected_state:
                if old == new_state:
                    return self.get_run(project_id, run_id)
                raise ConflictError("run state changed concurrently")
            if old == new_state:
                return self.get_run(project_id, run_id)
            if new_state not in _TRANSITIONS.get(old, set()):
                raise InvalidTransitionError(f"transition {old} -> {new_state} is not allowed")
            cur.execute(
                "UPDATE runs SET state=%s,state_detail=%s::jsonb,updated_at=now() WHERE project_id=%s AND run_id=%s",
                (new_state, _json(details or {}), project_id, run_id),
            )
            cur.execute(
                "INSERT INTO run_state_events(project_id,run_id,from_state,to_state,details) VALUES(%s,%s,%s,%s,%s::jsonb)",
                (project_id, run_id, old, new_state, _json(details or {})),
            )
        return self.get_run(project_id, run_id)

    def register_collection_allowlist(
        self, project_id: str, run_id: str, entries: list[Mapping[str, Any]]
    ) -> None:
        with self._connect() as conn:
            cur = conn.cursor()
            cur.execute(
                "SELECT 1 FROM runs WHERE project_id=%s AND run_id=%s", (project_id, run_id)
            )
            if cur.fetchone() is None:
                raise NotFoundError("run not found in project")
            for entry in entries:
                key, kind = entry.get("key"), entry.get("kind")
                required = bool(entry.get("required", False))
                if not isinstance(key, str) or not re.fullmatch(r"[a-z0-9][a-z0-9._-]{0,127}", key):
                    raise ValueError("invalid collection key")
                if not isinstance(kind, str) or not re.fullmatch(
                    r"[a-z0-9][a-z0-9._-]{0,63}", kind
                ):
                    raise ValueError("invalid artifact kind")
                cur.execute(
                    "INSERT INTO artifact_collection_allowlist(project_id,run_id,collection_key,kind,required) VALUES(%s,%s,%s,%s,%s) ON CONFLICT(project_id,run_id,collection_key) DO NOTHING",
                    (project_id, run_id, key, kind, required),
                )
                cur.execute(
                    "SELECT kind,required FROM artifact_collection_allowlist WHERE project_id=%s AND run_id=%s AND collection_key=%s",
                    (project_id, run_id, key),
                )
                old = _one(cur)
                if old["kind"] != kind or old["required"] != required:
                    raise ConflictError("collection allowlist is immutable")

    def record_artifact(
        self, project_id: str, run_id: str, manifest: Mapping[str, Any], collection_key: str
    ) -> dict[str, Any]:
        from .artifacts import safe_to_expose, validate_manifest

        item = validate_manifest(manifest, run_id=run_id)
        if not safe_to_expose(item):
            raise StorageError("artifact has not passed scanning and redaction")
        self._verified_public_artifact({**item, "manifest": item})
        with self._connect() as conn:
            cur = conn.cursor()
            cur.execute(
                "SELECT kind FROM artifact_collection_allowlist WHERE project_id=%s AND run_id=%s AND collection_key=%s",
                (project_id, run_id, collection_key),
            )
            allowed = _one(cur)
            if allowed is None or allowed["kind"] != item["kind"]:
                raise StorageError("artifact does not match collection allowlist")
            cur.execute(
                "INSERT INTO artifacts(project_id,artifact_id,run_id,collection_key,kind,uri,sha256,size_bytes,media_type,producer,classification,retention_class,scan_state,redaction_state,created_at,expires_at,manifest) VALUES(%s,%s,%s,%s,%s,%s,%s,%s,%s,%s::jsonb,%s,%s,%s,%s,%s,%s,%s::jsonb) ON CONFLICT(project_id,artifact_id) DO NOTHING",
                (
                    project_id,
                    item["artifact_id"],
                    run_id,
                    collection_key,
                    item["kind"],
                    item["uri"],
                    item["sha256"],
                    item["size_bytes"],
                    item["media_type"],
                    _json(item["producer"]),
                    item["classification"],
                    item["retention_class"],
                    item["scan_state"],
                    item.get("redaction_state", "pending"),
                    item["created_at"],
                    item.get("expires_at"),
                    _json(item),
                ),
            )
            cur.execute(
                "SELECT * FROM artifacts WHERE project_id=%s AND artifact_id=%s",
                (project_id, item["artifact_id"]),
            )
            result = _one(cur)
            if result is None:
                raise StorageError("artifact insert did not yield a record")
            if (
                result["run_id"] != run_id
                or result["collection_key"] != collection_key
                or result["manifest"] != item
            ):
                raise ConflictError(
                    "artifact identity already exists with different immutable manifest"
                )
            return self._verified_public_artifact(result)

    def list_artifacts(self, project_id: str, run_id: str) -> list[dict[str, Any]]:
        with self._connect() as conn:
            cur = conn.cursor()
            cur.execute(
                "SELECT * FROM artifacts WHERE project_id=%s AND run_id=%s AND deleted_at IS NULL ORDER BY created_at,artifact_id",
                (project_id, run_id),
            )
            rows = _all(cur)
        from .artifacts import safe_to_expose

        return [
            self._verified_public_artifact(row)
            for row in rows
            if row.get("deleted_at") is None and safe_to_expose(row)
        ]

    def expire_artifact(
        self,
        project_id: str,
        artifact_id: str,
        at: datetime | None = None,
        reason: str = "retention_expired",
    ) -> dict[str, Any]:
        from .artifacts import due_for_expiry

        when = at or datetime.now(UTC)
        if not reason or len(reason) > 128:
            raise ValueError("bounded tombstone reason required")
        with self._connect() as conn:
            cur = conn.cursor()
            cur.execute(
                "SELECT artifact_id,run_id,sha256,expires_at,deleted_at,manifest FROM artifacts WHERE project_id=%s AND artifact_id=%s FOR UPDATE",
                (project_id, artifact_id),
            )
            item = _one(cur)
            if item is None:
                raise NotFoundError("artifact not found in project")
            if item["deleted_at"] is not None:
                cur.execute(
                    "SELECT * FROM artifact_tombstones WHERE project_id=%s AND artifact_id=%s",
                    (project_id, artifact_id),
                )
                return _one(cur) or item
            manifest = item.get("manifest") or {"expires_at": item["expires_at"]}
            if not due_for_expiry(manifest, when):
                raise StorageError("artifact is not yet eligible for expiry")
            cur.execute(
                "UPDATE artifacts SET deleted_at=%s WHERE project_id=%s AND artifact_id=%s AND deleted_at IS NULL",
                (when, project_id, artifact_id),
            )
            cur.execute(
                "INSERT INTO artifact_tombstones(project_id,artifact_id,run_id,sha256,disposition,reason,deleted_at) VALUES(%s,%s,%s,%s,'expired',%s,%s) ON CONFLICT(project_id,artifact_id) DO NOTHING",
                (project_id, artifact_id, item["run_id"], item["sha256"], reason, when),
            )
            cur.execute(
                "SELECT * FROM artifact_tombstones WHERE project_id=%s AND artifact_id=%s",
                (project_id, artifact_id),
            )
            return _one(cur) or {}
