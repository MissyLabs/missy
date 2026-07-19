"""Tests for the layered tool policy pipeline."""

from __future__ import annotations

from missy.policy.tool_policy_pipeline import (
    ToolPolicyLayer,
    build_configured_tool_policy_layers,
    build_tool_policy_layers,
    collect_tool_policy_groups,
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


def test_discord_capability_mode_includes_voice_tools():
    decision = resolve_tool_policy(
        [
            "calculator",
            "discord_upload_file",
            "discord_voice_join",
            "discord_voice_leave",
            "discord_voice_say",
            "discord_voice_status",
            "browser_navigate",
            "x11_click",
        ],
        layers_for_capability_mode("discord"),
    )

    assert "discord_voice_join" in decision.tools
    assert "discord_voice_leave" in decision.tools
    assert "discord_voice_say" in decision.tools
    assert "discord_voice_status" in decision.tools
    assert "browser_navigate" not in decision.tools
    assert "x11_click" not in decision.tools


def test_configured_layers_apply_provider_global_agent_sandbox_subagent_order():
    layers = build_configured_tool_policy_layers(
        capability_mode="full",
        provider_name="anthropic",
        model_id="claude-haiku-4-5",
        global_policy={
            "profile": "full",
            "allow": ["*", "-shell_exec"],
            "byProvider": {
                "anthropic": {
                    "deny": ["browser_*"],
                    "byModel": {"claude-haiku-*": {"alsoAllow": ["shell_exec"]}},
                },
            },
        },
        agent_policy={"deny": ["file_delete"]},
        sandbox_policy={"allow": ["calculator", "file_*", "shell_exec"]},
        subagent_policy={"deny": ["shell_exec"]},
    )

    decision = resolve_tool_policy(
        ["browser_open", "calculator", "file_delete", "file_read", "shell_exec"],
        layers,
    )

    assert decision.tools == ("calculator", "file_read")
    assert decision.labels() == (
        "profile:full",
        "provider:anthropic",
        "provider:anthropic",
        "global",
        "global",
        "agent",
        "sandbox",
        "subagent",
    )


def test_configured_layers_use_agent_profile_when_mode_is_full():
    layers = build_configured_tool_policy_layers(
        capability_mode="full",
        global_policy={"profile": "coding"},
        agent_policy={"profile": "minimal"},
    )

    decision = resolve_tool_policy(["calculator", "file_read", "shell_exec"], layers)

    assert decision.tools == ("calculator", "file_read")
    assert decision.labels() == ("profile:minimal",)


def test_collect_tool_policy_groups_extends_builtin_groups():
    groups = collect_tool_policy_groups(
        {"groups": {"project": ["calculator", "custom_tool"]}},
        {"groups": {"project": ["file_read"]}},
    )
    decision = resolve_tool_policy(
        ["calculator", "custom_tool", "file_read"],
        [ToolPolicyLayer(label="agent", allow=["group:project"])],
        groups=groups,
    )

    assert groups["project"] == ("file_read",)
    assert decision.tools == ("file_read",)


def test_new_agent_tools_are_reachable_over_discord():
    """Regression: F03/F04/F16 tools must be in the Discord allowlist.

    rag_query (F03), graph_query (F04), and video_storyboard (F16) are
    registered as built-in tools, but the ``discord`` capability_mode only
    exposes tools listed in MISSY_DISCORD_TOOLS. They were originally omitted,
    so the agent correctly reported "I don't have a rag_query tool" over
    Discord and the RAG-*/GRAPH-*/STORY-* validation surface was unreachable.
    Guard the exposure so it can't silently regress.
    """
    from missy.policy.tool_policy_pipeline import MISSY_DISCORD_TOOLS

    for name in ("rag_query", "graph_query", "video_storyboard"):
        assert name in MISSY_DISCORD_TOOLS, f"{name} missing from MISSY_DISCORD_TOOLS"

    # And they actually survive the discord capability-mode policy resolution.
    layers = build_configured_tool_policy_layers(capability_mode="discord")
    decision = resolve_tool_policy(
        ["rag_query", "graph_query", "video_storyboard", "x11_screenshot"], layers
    )
    assert set(decision.tools) == {"rag_query", "graph_query", "video_storyboard"}
    # A GUI tool is still excluded from discord mode (fix didn't widen scope).
    assert "x11_screenshot" not in decision.tools
