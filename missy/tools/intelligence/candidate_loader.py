"""Controlled runtime loader for enabled tool candidates.

The loader intentionally supports only explicit implementation bindings. It
does not execute generated code and does not infer behavior from a candidate
description. The first supported binding is ``delegated_tool``: a candidate
can become a schema/metadata wrapper around an already-registered tool, while
the normal :class:`missy.tools.registry.ToolRegistry` policy checks still run
for both the candidate wrapper and the delegated tool.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any

from missy.core.events import AuditEvent, event_bus
from missy.tools.base import BaseTool, ToolPermissions, ToolResult
from missy.tools.registry import ToolRegistry

from .candidate_store import CandidateStore, ToolCandidate, ToolLifecycleState

logger = logging.getLogger(__name__)

_SAFE_NAME_RE = re.compile(r"^[a-z][a-z0-9_-]{0,62}$")
_ALLOWED_PERMISSION_KEYS = {
    "network",
    "filesystem_read",
    "filesystem_write",
    "shell",
    "allowed_paths",
    "allowed_hosts",
}
_SUPPORTED_IMPLEMENTATIONS = {"delegated_tool"}


@dataclass(frozen=True)
class CandidateLoadIssue:
    """A candidate skipped by the runtime loader."""

    candidate_id: str
    name: str
    reason: str


@dataclass(frozen=True)
class CandidateLoadReport:
    """Summary of a loader pass."""

    loaded: list[str] = field(default_factory=list)
    skipped: list[CandidateLoadIssue] = field(default_factory=list)


class CandidateDelegatedTool(BaseTool):
    """Runtime wrapper that delegates execution to an existing registered tool."""

    def __init__(self, candidate: ToolCandidate, target_tool: str, registry: ToolRegistry) -> None:
        self.name = candidate.name
        self.description = candidate.description
        self.permissions = _permissions_from_candidate(candidate.permissions)
        self._schema = {
            "name": candidate.name,
            "description": candidate.description,
            "parameters": candidate.schema,
        }
        self._candidate_id = candidate.id
        self._target_tool = target_tool
        self._registry = registry

    @property
    def candidate_id(self) -> str:
        return self._candidate_id

    @property
    def target_tool(self) -> str:
        return self._target_tool

    def get_schema(self) -> dict[str, Any]:
        return dict(self._schema)

    def execute(self, **kwargs: Any) -> ToolResult:
        return self._registry.execute(self._target_tool, **kwargs)


class CandidateRuntimeLoader:
    """Validate and register enabled candidates with explicit implementations."""

    def __init__(self, store: CandidateStore, registry: ToolRegistry) -> None:
        self._store = store
        self._registry = registry

    def load_enabled(self, provider_name: str) -> CandidateLoadReport:
        """Load enabled candidates for *provider_name* into the tool registry.

        Candidates are skipped unless they pass lifecycle, schema,
        provenance, implementation, permission, provider-enable, and conflict
        checks. Every load or skip emits a structured audit event.
        """
        loaded: list[str] = []
        skipped: list[CandidateLoadIssue] = []
        for candidate in self._store.list_all(state=ToolLifecycleState.ENABLED, limit=1000):
            reason = self._validate(candidate, provider_name)
            if reason:
                skipped.append(CandidateLoadIssue(candidate.id, candidate.name, reason))
                _emit_audit("tool.candidate.load_skipped", candidate, provider_name, reason, "deny")
                continue

            target_tool = str(candidate.implementation["tool"])
            self._registry.register(CandidateDelegatedTool(candidate, target_tool, self._registry))
            loaded.append(candidate.name)
            _emit_audit(
                "tool.candidate.loaded", candidate, provider_name, f"delegates:{target_tool}"
            )
        return CandidateLoadReport(loaded=loaded, skipped=skipped)

    def _validate(self, candidate: ToolCandidate, provider_name: str) -> str:
        if candidate.state is not ToolLifecycleState.ENABLED:
            return f"candidate state is {candidate.state.value}, not enabled"
        if not _SAFE_NAME_RE.match(candidate.name):
            return "candidate name is not a safe tool identifier"
        if not candidate.provenance.strip():
            return "candidate provenance is missing"
        schema_error = _validate_schema(candidate.schema)
        if schema_error:
            return schema_error
        permission_error = _validate_permissions(candidate.permissions)
        if permission_error:
            return permission_error
        impl = candidate.implementation
        if not impl:
            return "candidate implementation is missing"
        if not isinstance(impl, dict):
            return "candidate implementation must be an object"
        impl_type = str(impl.get("type") or "")
        if impl_type not in _SUPPORTED_IMPLEMENTATIONS:
            return f"unsupported implementation type: {impl_type or '<missing>'}"
        target_tool = str(impl.get("tool") or "")
        if not _SAFE_NAME_RE.match(target_tool):
            return "delegated tool target is not a safe tool identifier"
        if target_tool == candidate.name:
            return "candidate cannot delegate to itself"
        if self._registry.get(target_tool) is None:
            return f"delegated tool target {target_tool!r} is not registered"
        existing = self._registry.get(candidate.name)
        if existing is not None and not isinstance(existing, CandidateDelegatedTool):
            return f"tool name {candidate.name!r} already exists"
        if provider_name and candidate.provider_enabled.get(provider_name) is not True:
            return f"provider {provider_name!r} is not enabled for candidate"
        return ""


def _permissions_from_candidate(raw: dict[str, Any]) -> ToolPermissions:
    return ToolPermissions(
        network=bool(raw.get("network", False)),
        filesystem_read=bool(raw.get("filesystem_read", False)),
        filesystem_write=bool(raw.get("filesystem_write", False)),
        shell=bool(raw.get("shell", False)),
        allowed_paths=[str(p) for p in raw.get("allowed_paths", []) if isinstance(p, str)],
        allowed_hosts=[str(h) for h in raw.get("allowed_hosts", []) if isinstance(h, str)],
    )


def _validate_permissions(raw: dict[str, Any]) -> str:
    if not isinstance(raw, dict):
        return "candidate permissions must be an object"
    unknown = sorted(set(raw) - _ALLOWED_PERMISSION_KEYS)
    if unknown:
        return f"candidate permissions include unknown keys: {', '.join(unknown)}"
    for key in ("allowed_paths", "allowed_hosts"):
        value = raw.get(key, [])
        if value and (not isinstance(value, list) or not all(isinstance(v, str) for v in value)):
            return f"candidate permission {key} must be a list of strings"
    return ""


def _validate_schema(schema: dict[str, Any]) -> str:
    if not isinstance(schema, dict):
        return "candidate schema must be an object"
    if schema.get("type") != "object":
        return "candidate schema root type must be object"
    properties = schema.get("properties", {})
    if not isinstance(properties, dict):
        return "candidate schema properties must be an object"
    required = schema.get("required", [])
    if not isinstance(required, list) or not all(isinstance(v, str) for v in required):
        return "candidate schema required must be a list of strings"
    missing = sorted(set(required) - set(properties))
    if missing:
        return f"candidate schema requires undefined properties: {', '.join(missing)}"
    for name, definition in properties.items():
        if not re.match(r"^[a-z][a-z0-9_]{0,30}$", str(name)):
            return f"candidate schema property {name!r} is not a safe identifier"
        if not isinstance(definition, dict):
            return f"candidate schema property {name!r} must be an object"
    return ""


def _emit_audit(
    event_type: str,
    candidate: ToolCandidate,
    provider_name: str,
    reason: str,
    result: str = "allow",
) -> None:
    try:
        event_bus.publish(
            AuditEvent.now(
                session_id="",
                task_id="",
                event_type=event_type,
                category="tool",
                result=result,  # type: ignore[arg-type]
                detail={
                    "id": candidate.id,
                    "name": candidate.name,
                    "provider": provider_name,
                    "reason": reason,
                },
            )
        )
    except Exception as exc:  # noqa: BLE001
        logger.debug("CandidateRuntimeLoader: audit emit failed: %s", exc)
