"""Generate tool candidates from observed request patterns.

The :class:`CandidateGenerator` consumes a :class:`~.request_tracker.RequestPattern`
and produces a :class:`~.candidate_store.ToolCandidate`.  Generation is
**always policy-gated**: the caller must hold an explicit
``tool_creation_enabled`` permission in config.

Candidate generation is deliberately conservative:

- Generated tools are ``proposed`` state — they require human or operator
  approval before the agent can use them.
- Schema validation enforces that parameter names are safe identifiers and
  no privileged permissions are granted without explicit config.
- Audit events are emitted at every step.
"""

from __future__ import annotations

import hashlib
import logging
import re
from dataclasses import dataclass
from typing import Any

from missy.core.events import AuditEvent, event_bus

from .candidate_store import ToolCandidate
from .request_tracker import RequestPattern

logger = logging.getLogger(__name__)

# Maximum allowed parameter count in a generated schema.
_MAX_PARAMS = 8
# Allowed permission keys that generation may propose.
_SAFE_PERMISSIONS = {"filesystem_read", "filesystem_write", "network", "shell"}
# Always-denied permissions (require explicit operator override).
_DENIED_PERMISSIONS = {"shell"}


@dataclass
class GenerationResult:
    """Outcome of a generation attempt.

    Attributes:
        ok: ``True`` if a candidate was successfully generated.
        candidate: The produced :class:`~.candidate_store.ToolCandidate`,
            or ``None`` on failure.
        reason: Human-readable explanation when ``ok`` is ``False``.
    """

    ok: bool
    candidate: ToolCandidate | None
    reason: str = ""


class CandidateGenerator:
    """Generate tool candidates from high-frequency request patterns.

    Args:
        tool_creation_enabled: When ``False`` all generation attempts are
            rejected.  Operators must set ``tool_intelligence.candidate_generation:
            enabled: true`` in config to enable.
        allow_shell: If ``True``, shell permission may be proposed (still
            requires explicit approval before enabling).
        owner: Identity tagged on generated candidates.
    """

    def __init__(
        self,
        tool_creation_enabled: bool = False,
        allow_shell: bool = False,
        owner: str = "agent",
    ) -> None:
        self._enabled = tool_creation_enabled
        self._allow_shell = allow_shell
        self._owner = owner

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate_from_pattern(
        self,
        pattern: RequestPattern,
        extra_context: str = "",
    ) -> GenerationResult:
        """Produce a :class:`~.candidate_store.ToolCandidate` from *pattern*.

        Args:
            pattern: High-frequency pattern detected by
                :class:`~.request_tracker.RequestTracker`.
            extra_context: Optional free-text context to embed in the
                provenance and notes fields.

        Returns:
            :class:`GenerationResult` with ``ok=True`` on success.
        """
        if not self._enabled:
            _emit_audit(
                "tool.candidate.generation_denied",
                {
                    "pattern_key": pattern.pattern_key,
                    "reason": "tool_creation_enabled=False",
                },
            )
            return GenerationResult(ok=False, candidate=None, reason="tool creation is disabled")

        name, err = _derive_name(pattern)
        if err:
            return GenerationResult(ok=False, candidate=None, reason=err)

        description = _derive_description(pattern)
        schema = _derive_schema(pattern)
        permissions = _derive_permissions(pattern, allow_shell=self._allow_shell)

        validation_err = _validate(name, schema, permissions)
        if validation_err:
            _emit_audit(
                "tool.candidate.generation_denied",
                {
                    "pattern_key": pattern.pattern_key,
                    "name": name,
                    "reason": validation_err,
                },
            )
            return GenerationResult(ok=False, candidate=None, reason=validation_err)

        provenance = (
            f"Auto-generated from pattern {pattern.pattern_key!r} "
            f"({pattern.count} occurrences, score={pattern.frequency_score:.2f}). "
            f"Common tools: {', '.join(pattern.common_tools) or 'none'}. "
            f"{extra_context}".strip()
        )

        examples = [{"input": ex} for ex in pattern.example_messages[:3]]

        candidate = ToolCandidate.create(
            name=name,
            description=description,
            schema=schema,
            permissions=permissions,
            provenance=provenance,
            pattern_key=pattern.pattern_key,
            examples=examples,
            owner=self._owner,
            tags=["auto_generated", "needs_review"],
        )

        _emit_audit(
            "tool.candidate.generated",
            {
                "id": candidate.id,
                "name": name,
                "pattern_key": pattern.pattern_key,
                "frequency_score": pattern.frequency_score,
            },
        )
        logger.info(
            "CandidateGenerator: proposed %r from pattern %s (score=%.2f)",
            name,
            pattern.pattern_key,
            pattern.frequency_score,
        )
        return GenerationResult(ok=True, candidate=candidate)

    def generate_from_schema(
        self,
        name: str,
        description: str,
        parameters: dict[str, Any],
        permissions: dict[str, bool] | None = None,
        provenance: str = "",
        tags: list[str] | None = None,
    ) -> GenerationResult:
        """Produce a candidate directly from a caller-supplied schema.

        This path is used when the agent has enough context to specify the
        tool precisely without pattern inference.

        Args:
            name: Tool name (``[a-z0-9_-]+``).
            description: One-sentence description.
            parameters: JSON Schema properties dict.
            permissions: Dict of permission flags.
            provenance: Where this specification came from.
            tags: Optional string labels.

        Returns:
            :class:`GenerationResult`.
        """
        if not self._enabled:
            return GenerationResult(ok=False, candidate=None, reason="tool creation is disabled")

        schema = {
            "type": "object",
            "properties": dict(parameters),
            "required": [k for k, v in parameters.items() if v.get("required", False)],
        }
        perms = dict(permissions or {})
        # _derive_permissions() (the generate_from_pattern path) already
        # gates "shell" behind self._allow_shell, but this direct-schema
        # path took caller-supplied permissions verbatim and only ran them
        # through _validate(), which merely checks membership in
        # _SAFE_PERMISSIONS (which itself includes "shell") -- so a caller
        # could request permissions={"shell": True} here even with
        # allow_shell=False, bypassing the class's own documented
        # always-denied-without-override contract for this permission.
        if perms.get("shell") and not self._allow_shell:
            return GenerationResult(
                ok=False,
                candidate=None,
                reason="shell permission requires allow_shell=True on this generator",
            )
        validation_err = _validate(name, schema, perms)
        if validation_err:
            return GenerationResult(ok=False, candidate=None, reason=validation_err)

        candidate = ToolCandidate.create(
            name=name,
            description=description,
            schema=schema,
            permissions=perms,
            provenance=provenance,
            owner=self._owner,
            tags=list(tags or ["manual", "needs_review"]),
        )
        _emit_audit(
            "tool.candidate.generated",
            {"id": candidate.id, "name": name, "source": "direct_schema"},
        )
        return GenerationResult(ok=True, candidate=candidate)


# ---------------------------------------------------------------------------
# Derivation helpers
# ---------------------------------------------------------------------------

_WORD_RE = re.compile(r"[a-z0-9]+")
_SAFE_NAME_RE = re.compile(r"^[a-z][a-z0-9_-]{0,62}$")
_PARAM_NAME_RE = re.compile(r"^[a-z][a-z0-9_]{0,30}$")

# Keywords that imply a specific tool family and its permissions.
_TOOL_HINTS: list[tuple[list[str], dict[str, bool]]] = [
    (["read", "file", "open", "load", "cat"], {"filesystem_read": True}),
    (["write", "save", "create", "update", "edit"], {"filesystem_write": True}),
    (["fetch", "download", "url", "http", "get", "post"], {"network": True}),
    (["run", "exec", "shell", "command", "bash", "script"], {"shell": True}),
]


def _derive_name(pattern: RequestPattern) -> tuple[str, str]:
    """Return (name, error_string).  Name is empty on error."""
    words = _WORD_RE.findall(pattern.representative.lower())
    if not words:
        return "", "pattern representative is empty — cannot derive name"
    # Take first 3 meaningful words, excluding stop words.
    stop = {"a", "an", "the", "and", "or", "is", "it", "to", "for", "of", "in"}
    meaningful = [w for w in words if w not in stop and len(w) > 2][:3]
    if not meaningful:
        meaningful = words[:2]
    # Append a short hash suffix to avoid collisions.
    short_hash = hashlib.sha256(pattern.pattern_key.encode()).hexdigest()[:4]
    name = "_".join(meaningful) + "_" + short_hash
    name = name[:63]
    if not _SAFE_NAME_RE.match(name):
        name = re.sub(r"[^a-z0-9_-]", "_", name)[:63]
    return name, ""


def _derive_description(pattern: RequestPattern) -> str:
    rep = pattern.representative.strip()
    if len(rep) > 120:
        rep = rep[:117] + "..."
    return f"Automated tool extracted from repeated pattern: {rep}"


def _derive_schema(pattern: RequestPattern) -> dict[str, Any]:
    """Produce a minimal JSON Schema from pattern keywords."""
    words = _WORD_RE.findall(pattern.representative.lower())
    stop = {
        "a",
        "an",
        "the",
        "and",
        "or",
        "is",
        "it",
        "to",
        "for",
        "of",
        "in",
        "__num__",
        "__url__",
        "__email__",
        "__path__",
        "__token__",
        "me",
        "my",
        "i",
        "you",
        "can",
        "could",
        "would",
        "please",
        "help",
    }
    tokens = [w for w in words if w not in stop and len(w) > 2]

    # Infer a primary string parameter from the pattern.
    properties: dict[str, Any] = {}
    if tokens:
        primary = tokens[0] if _PARAM_NAME_RE.match(tokens[0]) else "input"
        properties[primary] = {
            "type": "string",
            "description": f"Primary input for the {primary} operation.",
        }

    return {
        "type": "object",
        "properties": properties,
        "required": list(properties.keys()),
    }


def _derive_permissions(pattern: RequestPattern, allow_shell: bool = False) -> dict[str, bool]:
    """Infer minimal required permissions from *pattern*."""
    perms: dict[str, bool] = {}
    text = (pattern.representative + " " + " ".join(pattern.common_tools)).lower()
    for keywords, perm_map in _TOOL_HINTS:
        if any(kw in text for kw in keywords):
            for k, v in perm_map.items():
                if k == "shell" and not allow_shell:
                    continue
                perms[k] = v
    return perms


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def _validate(
    name: str,
    schema: dict[str, Any],
    permissions: dict[str, bool],
) -> str:
    """Return an error string if validation fails, empty string on success."""
    if not name or not _SAFE_NAME_RE.match(name):
        return f"invalid tool name {name!r} — must match [a-z][a-z0-9_-]{{0,62}}"

    props = schema.get("properties", {})
    if len(props) > _MAX_PARAMS:
        return f"schema has {len(props)} parameters, max is {_MAX_PARAMS}"

    for param_name in props:
        if not _PARAM_NAME_RE.match(str(param_name)):
            return f"unsafe parameter name {param_name!r}"

    for perm in permissions:
        if perm not in _SAFE_PERMISSIONS:
            return f"unknown permission {perm!r}"

    return ""


# ---------------------------------------------------------------------------
# Audit helper
# ---------------------------------------------------------------------------


def _emit_audit(event_type: str, detail: dict[str, Any]) -> None:
    try:
        event_bus.publish(
            AuditEvent.now(
                session_id="",
                task_id="",
                event_type=event_type,
                category="tool",
                result="allow",
                detail=detail,
            )
        )
    except Exception as exc:  # noqa: BLE001
        logger.debug("CandidateGenerator: audit emit failed: %s", exc)
