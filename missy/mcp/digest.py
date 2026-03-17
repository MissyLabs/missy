"""MCP tool manifest digest computation and verification.

Provides SHA-256 digest pinning for MCP server tool manifests so that
changes between connections can be detected and refused.
"""

from __future__ import annotations

import hashlib
import json


def compute_tool_manifest_digest(tools: list[dict]) -> str:
    """Compute a deterministic SHA-256 digest of an MCP tool manifest.

    Tools are sorted by name and serialised as ``[{name, description}]``
    to produce a stable hash regardless of the order the server returns
    them in.

    Args:
        tools: List of tool definition dicts (must have ``"name"`` and
            ``"description"`` keys at minimum).

    Returns:
        A string of the form ``"sha256:<hex>"``.
    """
    canonical = sorted(
        [{"name": t.get("name", ""), "description": t.get("description", "")} for t in tools],
        key=lambda t: t["name"],
    )
    payload = json.dumps(canonical, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return f"sha256:{hashlib.sha256(payload).hexdigest()}"


def verify_digest(expected: str, actual: str) -> bool:
    """Return ``True`` if *expected* and *actual* digests match.

    Args:
        expected: The pinned digest string.
        actual: The freshly computed digest string.

    Returns:
        ``True`` when both strings are identical.
    """
    return expected == actual
