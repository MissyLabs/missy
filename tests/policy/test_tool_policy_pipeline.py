"""Tests for the layered tool policy pipeline."""

from __future__ import annotations

from missy.policy.tool_policy_pipeline import (
    ToolPolicyLayer,
    build_tool_policy_layers,
    layers_for_capability_mode,
    profile_layer,
    resolve_tool_policy,
)


def test_group_reference_expands_openclaw_fs_aliases():
    decision = resolve_tool_policy(
        ["read", "write", "edit", "apply_patch", "shell_exec"],
        [ToolPolicyLayer(label="group", allow=["group:fs"])],
    )

    assert decision.tools == ("read", "write", "edit", "apply_patch")
    assert decision.trace[0].patterns == ("read", "write", "edit", "apply_patch")
    assert decision.trace[0].label == "group"


def test_glob_allow_and_inline_deny_syntax_are_applied_in_one_layer():
    decision = resolve_tool_policy(
        ["file_read", "file_write", "shell_exec", "web_fetch"],
        [ToolPolicyLayer(label="agent", allow=["*", "-shell_*"])],
    )

    assert decision.tools == ("file_read", "file_write", "web_fetch")
    assert [step.operation for step in decision.trace] == ["allow", "deny"]
    assert decision.trace[1].matched == ("shell_exec",)


def test_also_allow_can_restore_tools_after_a_restrictive_layer():
    decision = resolve_tool_policy(
        ["calculator", "file_read", "shell_exec"],
        [
            ToolPolicyLayer(label="profile", allow=["calculator"]),
            ToolPolicyLayer(label="agent", also_allow=["file_*"]),
        ],
    )

    assert decision.tools == ("calculator", "file_read")
    assert decision.trace[-1].operation == "also_allow"


def test_unknown_plugin_only_allowlist_warns_without_hiding_core_tools():
    decision = resolve_tool_policy(
        ["calculator", "file_read"],
        [ToolPolicyLayer(label="plugin", allow=["plugin_special_tool"])],
    )

    assert decision.tools == ("calculator", "file_read")
    assert "unknown tool allowlist entry 'plugin_special_tool'" in decision.warnings
    assert decision.trace[0].before == decision.trace[0].after


def test_profile_layers_track_source_labels():
    decision = resolve_tool_policy(
        ["calculator", "file_read", "shell_exec"],
        [profile_layer("minimal"), ToolPolicyLayer(label="sandbox", deny=["file_*"])],
    )

    assert decision.tools == ("calculator",)
    assert decision.labels() == ("profile:minimal", "sandbox")


def test_standard_layer_builder_preserves_precedence_labels():
    layers = build_tool_policy_layers(
        profile="full",
        provider={"deny": ["browser_*"]},
        global_policy={"allow": ["*", "-shell_exec"]},
        agent={"alsoAllow": ["shell_exec"]},
        group={"deny": ["file_delete"]},
        sandbox={"allow": ["calculator", "file_*", "shell_exec"]},
        subagent={"deny": ["shell_exec"]},
    )

    decision = resolve_tool_policy(
        ["calculator", "browser_open", "file_delete", "file_read", "shell_exec"],
        layers,
    )

    assert decision.tools == ("calculator", "file_read")
    assert decision.labels() == (
        "profile:full",
        "provider",
        "global",
        "global",
        "agent",
        "group",
        "sandbox",
        "subagent",
    )


def test_capability_mode_layers_preserve_existing_runtime_modes():
    safe = resolve_tool_policy(
        ["calculator", "shell_exec", "x11_window_list"],
        layers_for_capability_mode("safe-chat"),
    )
    none = resolve_tool_policy(
        ["calculator", "shell_exec"],
        layers_for_capability_mode("no-tools"),
    )

    assert safe.tools == ("calculator", "x11_window_list")
    assert none.tools == ()
