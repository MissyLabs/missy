"""Canonical identities and versioned comparability rules."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

RULES_VERSION = "1.0"


def canonical_json(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode("utf-8")


def sha256_json(value: Any) -> str:
    return hashlib.sha256(canonical_json(value)).hexdigest()


def canonical_provider_input(
    prompt: str, messages: list[dict], tool_schemas: list[dict], settings: Mapping[str, Any]
) -> dict:
    """One provider-neutral payload shared byte-for-byte by provider adapters."""
    return {
        "prompt": prompt,
        "messages": messages,
        "tool_schemas": tool_schemas,
        "settings": dict(settings),
    }


def definition_hash(workload: Mapping[str, Any]) -> str:
    return sha256_json({k: v for k, v in workload.items() if k not in {"description", "providers"}})


COMPARABILITY_PATHS = (
    "repository.repository_id",
    "repository.commit_sha",
    "repository.snapshot_id",
    "repository.subdirectory",
    "task.class",
    "task.prompt_sha256",
    "task.fixture_digests",
    "task.tool_schema_uris",
    "workload_id",
    "workload_version",
    "definition_sha256",
    "sandbox.image_digest",
    "sandbox.network_policy",
    "sandbox.cpu_mhz",
    "sandbox.memory_mb",
    "sandbox.disk_mb",
    "sandbox.timeout_seconds",
    "sandbox.architecture",
    "evaluator_version",
    "validators",
    "provider_independent_settings_sha256",
    "tool_schemas_sha256",
)


def _get(obj: Mapping[str, Any], path: str):
    cur: Any = obj
    for key in path.split("."):
        if not isinstance(cur, Mapping) or key not in cur:
            return None
        cur = cur[key]
    return cur


def comparability_payload(manifest: Mapping[str, Any], rules_version: str = RULES_VERSION) -> dict:
    return {"rules_version": rules_version, **{p: _get(manifest, p) for p in COMPARABILITY_PATHS}}


def comparability_key(manifest: Mapping[str, Any], rules_version: str = RULES_VERSION) -> str:
    return sha256_json(comparability_payload(manifest, rules_version))


@dataclass(frozen=True)
class ComparabilityResult:
    comparable: bool
    key: str | None
    reasons: tuple[str, ...]


def compare_runs(
    left_manifest: Mapping[str, Any], right_manifest: Mapping[str, Any]
) -> ComparabilityResult:
    lv = left_manifest.get("comparability_rules_version", RULES_VERSION)
    rv = right_manifest.get("comparability_rules_version", RULES_VERSION)
    if lv != rv:
        return ComparabilityResult(False, None, ("comparability_rules_version_mismatch",))
    a, b = (
        comparability_payload(left_manifest, str(lv)),
        comparability_payload(right_manifest, str(rv)),
    )
    reasons = [p for p in COMPARABILITY_PATHS if a[p] != b[p]]
    ak, bk = sha256_json(a), sha256_json(b)
    for m, k in ((left_manifest, ak), (right_manifest, bk)):
        if m.get("comparability_key") not in (None, k):
            reasons.append("comparability_key_invalid")
    if ak != bk and not reasons:
        reasons.append("comparability_key_mismatch")
    return ComparabilityResult(not reasons, ak if not reasons else None, tuple(reasons))
