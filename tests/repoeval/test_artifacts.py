import hashlib
from datetime import UTC, datetime, timedelta

import pytest

from missy.repoeval.artifacts import (
    ArtifactError,
    CollectionRule,
    RetentionPolicy,
    due_for_expiry,
    make_manifest,
    manifest_digest,
    safe_to_expose,
    tombstone,
    validate_manifest,
    verify_object_bytes,
)

NOW = datetime(2026, 9, 28, tzinfo=UTC)


def sample(**overrides):
    data = {
        "artifact_id": "artifact-example0001",
        "run_id": "run-example0001",
        "kind": "validator-report",
        "uri": "s3://bucket/sha256/" + "a" * 64,
        "sha256": "a" * 64,
        "size_bytes": 4,
        "media_type": "application/json",
        "producer": {"component": "test", "version": "1"},
        "classification": "internal",
        "retention_class": "development",
        "created_at": NOW,
        "expires_at": NOW + timedelta(days=1),
        "scan_state": "pending",
    }
    data.update(overrides)
    return make_manifest(**data)


def test_manifest_round_trip_and_digest_stable():
    a = sample()
    assert validate_manifest(a) == a
    assert manifest_digest(a) == manifest_digest(dict(reversed(list(a.items()))))


def test_manifest_rejects_bad_digest_negative_size_and_naive_time():
    with pytest.raises(ArtifactError):
        sample(sha256="A" * 64)
    with pytest.raises(ArtifactError):
        sample(size_bytes=-1)
    with pytest.raises(ArtifactError):
        sample(created_at=datetime(2026, 1, 1))


def test_collection_allowlist_kind_and_run_scope():
    m = sample()
    assert (
        validate_manifest(
            m, run_id=m["run_id"], rules=[CollectionRule("report", "validator-report")]
        )
        == m
    )
    with pytest.raises(ArtifactError):
        validate_manifest(m, rules=[CollectionRule("log", "execution-log")])
    with pytest.raises(ArtifactError):
        validate_manifest(m, run_id="run-other000000")


def test_retention_is_explicit_and_tombstone_preserves_identity():
    policy = RetentionPolicy({"development": timedelta(days=3), "indefinite": None})
    assert policy.expiry("development", NOW) == NOW + timedelta(days=3)
    assert policy.expiry("indefinite", NOW) is None
    with pytest.raises(ArtifactError):
        policy.expiry("unknown", NOW)
    m = sample()
    assert not due_for_expiry(m, NOW)
    assert due_for_expiry(m, NOW + timedelta(days=2))
    t = tombstone(project_id="p1", manifest=m, reason="expired", deleted_at=NOW)
    assert (t["artifact_id"], t["sha256"], t["disposition"]) == (
        m["artifact_id"],
        m["sha256"],
        "expired",
    )


def test_verification_checks_actual_bytes_and_scan_redaction_gate():
    m = sample(sha256=hashlib.sha256(b"test").hexdigest())
    verify_object_bytes(m, b"test")
    for data in (b"bad!", b"test extra", None):
        with pytest.raises(ArtifactError):
            verify_object_bytes(m, data)
    assert not safe_to_expose(m)
    assert not safe_to_expose({**m, "scan_state": "quarantined", "redaction_state": "complete"})
    assert not safe_to_expose({**m, "scan_state": "clean", "redaction_state": "pending"})
    assert safe_to_expose({**m, "scan_state": "clean", "redaction_state": "complete"})
