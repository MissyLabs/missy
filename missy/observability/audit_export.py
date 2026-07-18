"""Signed, portable audit export bundles (F22).

``export_audit_bundle`` reads the tamper-evident audit log, optionally filters
it, verifies every selected line's signature + hash chain, and packages the
result into a self-describing JSON bundle that is itself signed by the agent
identity. ``verify_audit_bundle`` re-checks such a bundle using only the public
key embedded in it (reconstructed from the JWK), so a recipient can validate an
exported evidence bundle **without** access to the original host or its private
key — the point of an exportable, court-of-record-friendly artifact.

This complements the existing JSONL log + OTLP span export: the log is the live
record and OTLP is the streaming sink; a bundle is a frozen, signed, portable
slice for hand-off to an auditor or SIEM.
"""

from __future__ import annotations

import base64
import hashlib
import json
import logging
from datetime import UTC, datetime
from typing import Any

from missy.observability.audit_logger import verify_audit_log

logger = logging.getLogger(__name__)

BUNDLE_VERSION = 1


def _b64url_decode(s: str) -> bytes:
    """Decode a base64url string with optional missing padding."""
    return base64.urlsafe_b64decode(s + "=" * (-len(s) % 4))


def _canonical(obj: Any) -> bytes:
    """Deterministic bytes for signing/hashing (sorted keys, compact)."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _read_records(log_path: str) -> list[dict]:
    """Read the audit log into a list of parsed JSON records (skipping blanks)."""
    records: list[dict] = []
    try:
        with open(log_path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(json.loads(line))
                except (ValueError, TypeError):
                    # A malformed line is preserved as an opaque marker so the
                    # bundle's line count still matches the source file.
                    records.append({"_malformed": line})
    except FileNotFoundError:
        pass
    return records


def export_audit_bundle(
    log_path: str,
    identity: Any,
    *,
    since: str | None = None,
    category: str | None = None,
    limit: int = 0,
) -> dict:
    """Build a signed, self-verifiable audit bundle.

    Args:
        log_path: Path to the JSONL audit log.
        identity: An :class:`~missy.security.identity.AgentIdentity` used to
            sign the bundle and whose public JWK is embedded for verification.
        since: Optional ISO-8601 timestamp; only records with
            ``timestamp >= since`` are included (string comparison is valid for
            ISO-8601 in UTC).
        category: Optional exact category filter (e.g. ``"security"``).
        limit: Keep at most this many of the most recent matching records
            (0 = no limit).

    Returns:
        A bundle dict with ``manifest``, ``events``, and ``signature`` (hex over
        the canonical ``{manifest, events}``).
    """
    # Whole-file verification first (chain checks need the full sequence).
    verifications = verify_audit_log(log_path, identity)
    status_counts: dict[str, int] = {}
    chain_broken = 0
    for v in verifications:
        status_counts[v.status] = status_counts.get(v.status, 0) + 1
        if v.chain_ok is False:
            chain_broken += 1

    records = _read_records(log_path)

    def _keep(rec: dict) -> bool:
        if "_malformed" in rec:
            return False
        if since is not None and str(rec.get("timestamp", "")) < since:
            return False
        return not (category is not None and rec.get("category") != category)

    selected = [r for r in records if _keep(r)]
    if limit and limit > 0:
        selected = selected[-limit:]

    events_bytes = _canonical(selected)
    manifest = {
        "bundle_version": BUNDLE_VERSION,
        "generated_at": datetime.now(UTC).isoformat(),
        "source_path": log_path,
        "source_line_count": len(records),
        "exported_event_count": len(selected),
        "filters": {"since": since, "category": category, "limit": limit},
        "whole_log_verification": {
            "status_counts": status_counts,
            "chain_broken_lines": chain_broken,
        },
        "events_sha256": hashlib.sha256(events_bytes).hexdigest(),
        "identity_fingerprint": identity.public_key_fingerprint(),
        "identity_jwk": identity.to_jwk(),
    }

    payload = _canonical({"manifest": manifest, "events": selected})
    signature = identity.sign(payload).hex()

    return {"manifest": manifest, "events": selected, "signature": signature}


def verify_audit_bundle(bundle: dict) -> dict:
    """Verify a bundle produced by :func:`export_audit_bundle`.

    Uses only the public key embedded in the bundle's manifest (reconstructed
    from its JWK), so verification needs nothing but the bundle itself.

    Returns:
        A dict with ``signature_valid`` (bool), ``events_hash_valid`` (bool),
        ``exported_event_count`` (int), and a ``reason`` string on failure.
    """
    from cryptography.exceptions import InvalidSignature
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

    result = {
        "signature_valid": False,
        "events_hash_valid": False,
        "exported_event_count": 0,
        "reason": "",
    }
    if not isinstance(bundle, dict):
        result["reason"] = "bundle is not an object"
        return result
    manifest = bundle.get("manifest")
    events = bundle.get("events")
    sig_hex = bundle.get("signature")
    if (
        not isinstance(manifest, dict)
        or not isinstance(events, list)
        or not isinstance(sig_hex, str)
    ):
        result["reason"] = "bundle missing manifest/events/signature"
        return result

    result["exported_event_count"] = len(events)

    # Recompute the events hash independently of the signature.
    events_bytes = _canonical(events)
    result["events_hash_valid"] = hashlib.sha256(events_bytes).hexdigest() == manifest.get(
        "events_sha256"
    )

    jwk = manifest.get("identity_jwk") or {}
    x = jwk.get("x")
    if not x:
        result["reason"] = "manifest has no public key (jwk.x)"
        return result
    try:
        public_key = Ed25519PublicKey.from_public_bytes(_b64url_decode(x))
        payload = _canonical({"manifest": manifest, "events": events})
        public_key.verify(bytes.fromhex(sig_hex), payload)
        result["signature_valid"] = True
    except (InvalidSignature, ValueError, TypeError) as exc:
        result["reason"] = f"signature verification failed: {exc}"
        return result

    if not result["events_hash_valid"]:
        result["reason"] = "events hash does not match manifest"
    return result
