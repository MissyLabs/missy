"""Layered tool availability policy pipeline.

This module resolves the set of tools an agent may see for a turn.  The
pipeline is intentionally separate from tool execution policy: execution still
fails closed in :mod:`missy.tools.registry`, while this layer determines which
tool schemas are exposed to the model.
"""

from __future__ import annotations

import fnmatch
import logging
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Literal

logger = logging.getLogger(__name__)

ToolPolicyProfile = Literal["minimal", "coding", "messaging", "full"]

MISSY_SAFE_CHAT_TOOLS: tuple[str, ...] = (
    "calculator",
    "file_read",
    "list_files",
    "web_fetch",
    "browser_get_content",
    "browser_get_url",
    "browser_screenshot",
    "x11_screenshot",
    "x11_window_list",
    "atspi_get_tree",
    "atspi_get_text",
)

MISSY_DISCORD_TOOLS: tuple[str, ...] = (
    "calculator",
    "file_read",
    "file_write",
    "file_delete",
    "list_files",
    "web_fetch",
    "shell_exec",
    "self_create_tool",
    "code_evolve",
    "discord_upload_file",
    "tts_speak",
    "audio_list_devices",
    "audio_set_volume",
    "incus_list",
    "incus_info",
    "incus_launch",
    "incus_instance_action",
    "incus_exec",
    "incus_file",
    "incus_snapshot",
    "incus_image",
    "incus_network",
    "incus_storage",
    "incus_config",
    "incus_profile",
    "incus_project",
    "incus_device",
    "incus_copy_move",
    "vision_capture",
    "vision_burst",
    "vision_analyze",
    "vision_devices",
    "vision_scene",
)

DEFAULT_TOOL_GROUPS: dict[str, tuple[str, ...]] = {
    # OpenClaw-compatible group name.  Missy keeps these aliases even when the
    # current Python tool registry uses longer names like ``file_read``.
    "fs": ("read", "write", "edit", "apply_patch"),
    "missy_fs": ("file_read", "file_write", "file_delete", "list_files"),
    "read_only": MISSY_SAFE_CHAT_TOOLS,
    "coding": (
        "calculator",
        "file_read",
        "file_write",
        "file_delete",
        "list_files",
        "shell_exec",
        "self_create_tool",
        "code_evolve",
    ),
    "messaging": ("discord_upload_file", "tts_speak"),
    "audio": ("tts_speak", "audio_list_devices", "audio_set_volume"),
    "incus": tuple(name for name in MISSY_DISCORD_TOOLS if name.startswith("incus_")),
    "vision": tuple(name for name in MISSY_DISCORD_TOOLS if name.startswith("vision_")),
    "desktop": (
        "browser_get_content",
        "browser_get_url",
        "browser_screenshot",
        "x11_screenshot",
        "x11_window_list",
        "atspi_get_tree",
        "atspi_get_text",
    ),
}

PROFILE_SPECS: dict[ToolPolicyProfile, tuple[str, ...]] = {
    "minimal": MISSY_SAFE_CHAT_TOOLS,
    "coding": ("group:coding", "browser_*", "x11_*", "atspi_*"),
    "messaging": MISSY_DISCORD_TOOLS,
    "full": ("*",),
}

CAPABILITY_MODE_PROFILES: dict[str, ToolPolicyProfile] = {
    "safe-chat": "minimal",
    "discord": "messaging",
    "full": "full",
}


@dataclass(frozen=True)
class ToolPolicyLayer:
    """One filtering layer in the tool policy pipeline."""

    label: str
    allow: Sequence[str] = ()
    deny: Sequence[str] = ()
    also_allow: Sequence[str] = ()

    @classmethod
    def from_mapping(cls, label: str, data: Mapping[str, object]) -> ToolPolicyLayer:
        """Build a layer from config-style ``allow``/``deny`` keys."""
        return cls(
            label=label,
            allow=_coerce_specs(data.get("allow")),
            deny=_coerce_specs(data.get("deny")),
            also_allow=_coerce_specs(data.get("alsoAllow") or data.get("also_allow")),
        )


@dataclass(frozen=True)
class ToolPolicyTraceStep:
    """Audit trace for a single layer operation."""

    label: str
    operation: str
    patterns: tuple[str, ...]
    matched: tuple[str, ...]
    before: tuple[str, ...]
    after: tuple[str, ...]
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True)
class ToolPolicyDecision:
    """Resolved tool names plus warning and trace metadata."""

    tools: tuple[str, ...]
    trace: tuple[ToolPolicyTraceStep, ...] = ()
    warnings: tuple[str, ...] = ()

    def labels(self) -> tuple[str, ...]:
        """Return the source labels that made filtering decisions."""
        return tuple(step.label for step in self.trace)


def profile_layer(profile: ToolPolicyProfile, *, label: str | None = None) -> ToolPolicyLayer:
    """Return the built-in profile layer for *profile*."""
    return ToolPolicyLayer(label=label or f"profile:{profile}", allow=PROFILE_SPECS[profile])


def layers_for_capability_mode(mode: str) -> tuple[ToolPolicyLayer, ...]:
    """Translate Missy's runtime capability mode into pipeline layers."""
    normalized = (mode or "full").strip().lower()
    if normalized == "no-tools":
        return (ToolPolicyLayer(label="profile:no-tools", deny=("*",)),)
    profile = CAPABILITY_MODE_PROFILES.get(normalized, "full")
    return (profile_layer(profile),)


def build_tool_policy_layers(
    *,
    profile: ToolPolicyProfile = "full",
    provider: Mapping[str, object] | ToolPolicyLayer | None = None,
    global_policy: Mapping[str, object] | ToolPolicyLayer | None = None,
    agent: Mapping[str, object] | ToolPolicyLayer | None = None,
    group: Mapping[str, object] | ToolPolicyLayer | None = None,
    sandbox: Mapping[str, object] | ToolPolicyLayer | None = None,
    subagent: Mapping[str, object] | ToolPolicyLayer | None = None,
) -> tuple[ToolPolicyLayer, ...]:
    """Build the standard profile-to-subagent layer sequence.

    Each optional policy may be a :class:`ToolPolicyLayer` or a mapping with
    ``allow``, ``deny``, and ``alsoAllow``/``also_allow`` keys.
    """
    layers: list[ToolPolicyLayer] = [profile_layer(profile)]
    for label, policy in (
        ("provider", provider),
        ("global", global_policy),
        ("agent", agent),
        ("group", group),
        ("sandbox", sandbox),
        ("subagent", subagent),
    ):
        layer = _coerce_layer(label, policy)
        if layer is not None:
            layers.append(layer)
    return tuple(layers)


def resolve_tool_policy(
    available_tools: Iterable[str],
    layers: Sequence[ToolPolicyLayer] = (),
    *,
    groups: Mapping[str, Sequence[str]] | None = None,
) -> ToolPolicyDecision:
    """Resolve allowed tool names through all policy layers.

    Unknown allow-list entries warn and do not narrow the current tool set when
    they are the only entries in a layer.  That fail-warning behavior prevents a
    plugin-only allow-list from accidentally hiding all core tools.
    """
    all_tools = tuple(dict.fromkeys(str(name) for name in available_tools if str(name)))
    current = all_tools
    trace: list[ToolPolicyTraceStep] = []
    warnings: list[str] = []
    group_map = dict(DEFAULT_TOOL_GROUPS)
    if groups:
        group_map.update({str(k): tuple(str(v) for v in values) for k, values in groups.items()})

    for layer in layers:
        inline_allow, inline_deny = _split_inline_denies(layer.allow)
        inline_also_allow, inline_also_deny = _split_inline_denies(layer.also_allow)
        deny_specs = (*layer.deny, *inline_deny, *inline_also_deny)

        if inline_allow:
            current, step = _apply_allow(
                layer.label,
                "allow",
                current,
                all_tools,
                inline_allow,
                group_map,
            )
            trace.append(step)
            warnings.extend(step.warnings)

        if inline_also_allow:
            current, step = _apply_also_allow(
                layer.label,
                current,
                all_tools,
                inline_also_allow,
                group_map,
            )
            trace.append(step)
            warnings.extend(step.warnings)

        if deny_specs:
            current, step = _apply_deny(layer.label, current, all_tools, deny_specs, group_map)
            trace.append(step)
            warnings.extend(step.warnings)

    for warning in warnings:
        logger.warning("Tool policy warning: %s", warning)
    return ToolPolicyDecision(tools=current, trace=tuple(trace), warnings=tuple(warnings))


def _coerce_specs(value: object) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return (value,)
    if isinstance(value, Iterable):
        return tuple(str(item) for item in value if str(item))
    return (str(value),)


def _coerce_layer(
    label: str,
    policy: Mapping[str, object] | ToolPolicyLayer | None,
) -> ToolPolicyLayer | None:
    if policy is None:
        return None
    if isinstance(policy, ToolPolicyLayer):
        return policy
    return ToolPolicyLayer.from_mapping(label, policy)


def _split_inline_denies(specs: Sequence[str]) -> tuple[tuple[str, ...], tuple[str, ...]]:
    allow: list[str] = []
    deny: list[str] = []
    for spec in specs:
        text = str(spec).strip()
        if not text:
            continue
        if text.startswith("-") and len(text) > 1:
            deny.append(text[1:])
        else:
            allow.append(text)
    return tuple(allow), tuple(deny)


def _apply_allow(
    label: str,
    operation: str,
    current: tuple[str, ...],
    all_tools: tuple[str, ...],
    specs: Sequence[str],
    groups: Mapping[str, Sequence[str]],
) -> tuple[tuple[str, ...], ToolPolicyTraceStep]:
    before = current
    matched, expanded, warnings = _match_specs(specs, all_tools, groups, warn_unknown=True)
    after = tuple(name for name in current if name in matched) if matched else current
    return after, ToolPolicyTraceStep(
        label=label,
        operation=operation,
        patterns=expanded,
        matched=tuple(name for name in all_tools if name in matched),
        before=before,
        after=after,
        warnings=tuple(warnings),
    )


def _apply_also_allow(
    label: str,
    current: tuple[str, ...],
    all_tools: tuple[str, ...],
    specs: Sequence[str],
    groups: Mapping[str, Sequence[str]],
) -> tuple[tuple[str, ...], ToolPolicyTraceStep]:
    before = current
    matched, expanded, warnings = _match_specs(specs, all_tools, groups, warn_unknown=True)
    combined = [*current]
    for name in all_tools:
        if name in matched and name not in combined:
            combined.append(name)
    after = tuple(combined)
    return after, ToolPolicyTraceStep(
        label=label,
        operation="also_allow",
        patterns=expanded,
        matched=tuple(name for name in all_tools if name in matched),
        before=before,
        after=after,
        warnings=tuple(warnings),
    )


def _apply_deny(
    label: str,
    current: tuple[str, ...],
    all_tools: tuple[str, ...],
    specs: Sequence[str],
    groups: Mapping[str, Sequence[str]],
) -> tuple[tuple[str, ...], ToolPolicyTraceStep]:
    before = current
    matched, expanded, warnings = _match_specs(specs, all_tools, groups, warn_unknown=False)
    after = tuple(name for name in current if name not in matched)
    return after, ToolPolicyTraceStep(
        label=label,
        operation="deny",
        patterns=expanded,
        matched=tuple(name for name in all_tools if name in matched),
        before=before,
        after=after,
        warnings=tuple(warnings),
    )


def _match_specs(
    specs: Sequence[str],
    all_tools: tuple[str, ...],
    groups: Mapping[str, Sequence[str]],
    *,
    warn_unknown: bool,
) -> tuple[set[str], tuple[str, ...], list[str]]:
    expanded: list[str] = []
    warnings: list[str] = []

    for raw in specs:
        spec = str(raw).strip()
        if not spec:
            continue
        if spec.startswith("group:"):
            group_name = spec.split(":", 1)[1]
            members = groups.get(group_name)
            if members is None:
                if warn_unknown:
                    warnings.append(f"unknown tool group {spec!r}")
                continue
            expanded.extend(str(member) for member in members)
        else:
            expanded.append(spec)

    matched: set[str] = set()
    for pattern in expanded:
        names = tuple(name for name in all_tools if _matches(pattern, name))
        if not names and warn_unknown:
            warnings.append(f"unknown tool allowlist entry {pattern!r}")
        matched.update(names)

    return matched, tuple(expanded), warnings


def _matches(pattern: str, tool_name: str) -> bool:
    if pattern == "*":
        return True
    if any(char in pattern for char in "*?["):
        return fnmatch.fnmatchcase(tool_name, pattern)
    return tool_name == pattern
