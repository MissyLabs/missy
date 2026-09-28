"""Artifact manifest validation, collection allowlists, and retention metadata.

This module intentionally does not upload, download, or delete object-store
bytes. It prepares and validates the catalog records consumed by a later adapter.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any

ID = re.compile(r"^artifact-[A-Za-z0-9_-]{8,128}$")
RUN = re.compile(r"^run-[A-Za-z0-9_-]{8,128}$")
SHA = re.compile(r"^[0-9a-f]{64}$")
KIND = re.compile(r"^[a-z0-9][a-z0-9._-]{0,63}$")
RETENTION = re.compile(r"^[a-z0-9][a-z0-9._-]{0,63}$")
CLASSIFICATIONS = {"public", "internal", "repository-sensitive", "restricted"}
SCAN_STATES = {"pending", "clean", "quarantined", "failed"}
REDACTION_STATES = {"not-required", "pending", "complete", "failed"}
MAX_VERIFIED_BYTES = 64_000_000


class ArtifactError(ValueError):
    pass


@dataclass(frozen=True)
class CollectionRule:
    key: str
    kind: str
    required: bool = False

    def __post_init__(self) -> None:
        if not re.fullmatch(r"[a-z0-9][a-z0-9._-]{0,127}", self.key):
            raise ArtifactError("invalid collection key")
        if not KIND.fullmatch(self.kind):
            raise ArtifactError("invalid artifact kind")


class RetentionPolicy:
    """Explicit, project-configured duration mapping; no implicit expiry defaults."""

    def __init__(self, durations: Mapping[str, timedelta | None]):
        self._durations = dict(durations)
        if any(not RETENTION.fullmatch(key) for key in self._durations):
            raise ArtifactError("invalid retention class")
        if any(
            value is not None and value.total_seconds() < 0 for value in self._durations.values()
        ):
            raise ArtifactError("retention durations cannot be negative")

    def expiry(self, retention_class: str, created_at: datetime) -> datetime | None:
        if retention_class not in self._durations:
            raise ArtifactError("unknown retention class")
        duration = self._durations[retention_class]
        return None if duration is None else created_at + duration


def _timestamp(value: Any, name: str) -> datetime:
    if isinstance(value, datetime):
        parsed = value
    elif isinstance(value, str):
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError as exc:
            raise ArtifactError(f"invalid {name}") from exc
    else:
        raise ArtifactError(f"{name} is required")
    if parsed.tzinfo is None:
        raise ArtifactError(f"{name} must include timezone")
    return parsed.astimezone(UTC)


def manifest_digest(manifest: Mapping[str, Any]) -> str:
    canonical = json.dumps(
        manifest, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    )
    return hashlib.sha256(canonical.encode()).hexdigest()


def verify_object_bytes(manifest: Mapping[str, Any], data: bytes) -> None:
    """Verify bytes fetched from a trusted object-store reader, not a worker claim."""
    if not isinstance(data, bytes) or len(data) > MAX_VERIFIED_BYTES:
        raise ArtifactError("artifact bytes unavailable or exceed verification limit")
    if (
        len(data) != manifest["size_bytes"]
        or hashlib.sha256(data).hexdigest() != manifest["sha256"]
    ):
        raise ArtifactError("artifact object checksum or size mismatch")


def safe_to_expose(manifest: Mapping[str, Any]) -> bool:
    """Pending, failed, quarantined and unredacted objects have no read access."""
    return manifest.get("scan_state") == "clean" and manifest.get("redaction_state") in (
        "complete",
        "not-required",
    )


def validate_manifest(
    manifest: Mapping[str, Any], *, run_id: str | None = None, rules: Iterable[CollectionRule] = ()
) -> dict[str, Any]:
    """Validate required fields and ensure manifest matches declared outputs."""
    item = dict(manifest)
    required = {
        "schema_version",
        "artifact_id",
        "run_id",
        "kind",
        "uri",
        "sha256",
        "size_bytes",
        "media_type",
        "producer",
        "classification",
        "retention_class",
        "scan_state",
        "created_at",
    }
    missing = required - item.keys()
    if missing:
        raise ArtifactError("artifact manifest missing required fields")
    if (
        item["schema_version"] != "1.0"
        or not ID.fullmatch(str(item["artifact_id"]))
        or not RUN.fullmatch(str(item["run_id"]))
    ):
        raise ArtifactError("invalid artifact or run identity")
    if run_id is not None and item["run_id"] != run_id:
        raise ArtifactError("artifact belongs to a different run")
    if not KIND.fullmatch(str(item["kind"])) or not SHA.fullmatch(str(item["sha256"])):
        raise ArtifactError("invalid kind or sha256")
    if type(item["size_bytes"]) is not int or item["size_bytes"] < 0:
        raise ArtifactError("size_bytes must be a non-negative integer")
    for name, maxlen in (("uri", 2048), ("media_type", 255)):
        if not isinstance(item[name], str) or not 0 < len(item[name]) <= maxlen:
            raise ArtifactError(f"invalid {name}")
    if (
        not isinstance(item["producer"], dict)
        or set(item["producer"]) != {"component", "version"}
        or any(not isinstance(v, str) or not v or len(v) > 128 for v in item["producer"].values())
    ):
        raise ArtifactError("invalid producer metadata")
    if item["classification"] not in CLASSIFICATIONS or not RETENTION.fullmatch(
        str(item["retention_class"])
    ):
        raise ArtifactError("invalid classification or retention class")
    if (
        item["scan_state"] not in SCAN_STATES
        or item.get("redaction_state", "pending") not in REDACTION_STATES
    ):
        raise ArtifactError("invalid scan or redaction state")
    _timestamp(item["created_at"], "created_at")
    if item.get("expires_at") is not None:
        item["expires_at"] = _timestamp(item["expires_at"], "expires_at").isoformat()
    item["created_at"] = _timestamp(item["created_at"], "created_at").isoformat()
    allowed = {r.kind for r in rules}
    if allowed and item["kind"] not in allowed:
        raise ArtifactError("artifact kind is not allowlisted for collection")
    return item


def make_manifest(
    *,
    artifact_id: str,
    run_id: str,
    kind: str,
    uri: str,
    sha256: str,
    size_bytes: int,
    media_type: str,
    producer: Mapping[str, str],
    classification: str,
    retention_class: str,
    created_at: datetime | None = None,
    expires_at: datetime | None = None,
    scan_state: str = "pending",
    redaction_state: str = "pending",
) -> dict[str, Any]:
    created = created_at or datetime.now(UTC)
    if created.tzinfo is None:
        raise ArtifactError("created_at must include timezone")
    if expires_at is not None and expires_at.tzinfo is None:
        raise ArtifactError("expires_at must include timezone")
    created = created.astimezone(UTC)
    result = {
        "schema_version": "1.0",
        "artifact_id": artifact_id,
        "run_id": run_id,
        "kind": kind,
        "uri": uri,
        "sha256": sha256,
        "size_bytes": size_bytes,
        "media_type": media_type,
        "producer": dict(producer),
        "classification": classification,
        "retention_class": retention_class,
        "scan_state": scan_state,
        "redaction_state": redaction_state,
        "created_at": created.isoformat(),
        "expires_at": expires_at.astimezone(UTC).isoformat() if expires_at else None,
    }
    return validate_manifest(result)


def due_for_expiry(manifest: Mapping[str, Any], now: datetime | None = None) -> bool:
    expiry = manifest.get("expires_at")
    return expiry is not None and _timestamp(expiry, "expires_at") <= (now or datetime.now(UTC))


def tombstone(
    *,
    project_id: str,
    manifest: Mapping[str, Any],
    reason: str,
    deleted_at: datetime | None = None,
    disposition: str = "expired",
) -> dict[str, Any]:
    if (
        not project_id
        or not reason
        or len(reason) > 128
        or disposition not in {"expired", "deleted", "quarantined"}
    ):
        raise ArtifactError("invalid tombstone metadata")
    item = validate_manifest(manifest)
    return {
        "project_id": project_id,
        "artifact_id": item["artifact_id"],
        "run_id": item["run_id"],
        "sha256": item["sha256"],
        "disposition": disposition,
        "reason": reason,
        "deleted_at": (deleted_at or datetime.now(UTC)).astimezone(UTC).isoformat(),
    }
