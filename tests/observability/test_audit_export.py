"""Tests for signed audit export bundles (F22)."""

from __future__ import annotations

import copy
import json
import os
from pathlib import Path

import pytest

from missy.observability.audit_export import export_audit_bundle, verify_audit_bundle
from missy.security.identity import AgentIdentity


@pytest.fixture
def identity() -> AgentIdentity:
    return AgentIdentity.generate()


@pytest.fixture
def log_path(tmp_path: Path) -> str:
    p = tmp_path / "audit.jsonl"
    lines = [
        {
            "timestamp": "2026-07-18T00:00:00+00:00",
            "category": "security",
            "event_type": "e1",
            "detail": {},
        },
        {
            "timestamp": "2026-07-18T01:00:00+00:00",
            "category": "network",
            "event_type": "e2",
            "detail": {},
        },
        {
            "timestamp": "2026-07-18T02:00:00+00:00",
            "category": "security",
            "event_type": "e3",
            "detail": {},
        },
    ]
    p.write_text("\n".join(json.dumps(x) for x in lines) + "\n")
    return str(p)


class TestExport:
    def test_bundle_shape(self, log_path: str, identity: AgentIdentity) -> None:
        bundle = export_audit_bundle(log_path, identity)
        assert set(bundle) == {"manifest", "events", "signature"}
        m = bundle["manifest"]
        assert m["exported_event_count"] == 3
        assert m["identity_jwk"]["kty"] == "OKP"
        assert "events_sha256" in m
        assert isinstance(bundle["signature"], str)

    def test_category_filter(self, log_path: str, identity: AgentIdentity) -> None:
        bundle = export_audit_bundle(log_path, identity, category="security")
        assert bundle["manifest"]["exported_event_count"] == 2
        assert all(e["category"] == "security" for e in bundle["events"])

    def test_since_filter(self, log_path: str, identity: AgentIdentity) -> None:
        bundle = export_audit_bundle(log_path, identity, since="2026-07-18T01:30:00+00:00")
        assert bundle["manifest"]["exported_event_count"] == 1
        assert bundle["events"][0]["event_type"] == "e3"

    def test_limit_keeps_most_recent(self, log_path: str, identity: AgentIdentity) -> None:
        bundle = export_audit_bundle(log_path, identity, limit=2)
        assert bundle["manifest"]["exported_event_count"] == 2
        assert [e["event_type"] for e in bundle["events"]] == ["e2", "e3"]

    def test_missing_log_yields_empty_bundle(self, tmp_path: Path, identity: AgentIdentity) -> None:
        bundle = export_audit_bundle(str(tmp_path / "nope.jsonl"), identity)
        assert bundle["manifest"]["exported_event_count"] == 0
        # An empty bundle still signs + verifies.
        assert verify_audit_bundle(bundle)["signature_valid"] is True


class TestVerify:
    def test_clean_bundle_verifies(self, log_path: str, identity: AgentIdentity) -> None:
        bundle = export_audit_bundle(log_path, identity)
        result = verify_audit_bundle(bundle)
        assert result["signature_valid"] is True
        assert result["events_hash_valid"] is True
        assert result["exported_event_count"] == 3

    def test_tampered_event_fails(self, log_path: str, identity: AgentIdentity) -> None:
        bundle = export_audit_bundle(log_path, identity)
        tampered = copy.deepcopy(bundle)
        tampered["events"][0]["detail"] = {"injected": "evil"}
        result = verify_audit_bundle(tampered)
        assert result["signature_valid"] is False
        assert result["events_hash_valid"] is False

    def test_added_event_fails(self, log_path: str, identity: AgentIdentity) -> None:
        bundle = export_audit_bundle(log_path, identity)
        tampered = copy.deepcopy(bundle)
        tampered["events"].append({"timestamp": "z", "event_type": "forged", "detail": {}})
        result = verify_audit_bundle(tampered)
        assert result["signature_valid"] is False

    def test_swapped_signature_from_other_identity_fails(self, log_path: str) -> None:
        a = AgentIdentity.generate()
        b = AgentIdentity.generate()
        bundle = export_audit_bundle(log_path, a)
        # Replace the signature with one from a different key over the same payload;
        # the embedded JWK is still a's, so verification must fail.
        from missy.observability.audit_export import _canonical

        payload = _canonical({"manifest": bundle["manifest"], "events": bundle["events"]})
        bundle["signature"] = b.sign(payload).hex()
        assert verify_audit_bundle(bundle)["signature_valid"] is False

    def test_verifies_without_local_identity(self, log_path: str, identity: AgentIdentity) -> None:
        # Round-trip through JSON (as the CLI does) and verify with NO identity
        # object available — only the bundle. Proves portability.
        bundle = export_audit_bundle(log_path, identity)
        roundtripped = json.loads(json.dumps(bundle))
        assert verify_audit_bundle(roundtripped)["signature_valid"] is True

    def test_malformed_bundle_is_rejected_cleanly(self) -> None:
        assert verify_audit_bundle({})["signature_valid"] is False
        assert verify_audit_bundle("not a dict")["signature_valid"] is False  # type: ignore[arg-type]
        assert (
            verify_audit_bundle({"manifest": {}, "events": [], "signature": "x"})["signature_valid"]
            is False
        )
