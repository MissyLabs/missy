"""Integration tests for Missy policy enforcement.

These tests operate against the real policy engine classes — no mocks for the
engines themselves.  They prove that each enforcement boundary works end-to-end:
network CIDR/domain rules, filesystem path containment, shell command allow-
lists, and plugin gating.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Generator

import pytest

from missy.config.settings import (
    FilesystemPolicy,
    MissyConfig,
    NetworkPolicy,
    PluginPolicy,
    ShellPolicy,
)
from missy.core.events import event_bus
from missy.core.exceptions import PolicyViolationError
from missy.plugins.base import BasePlugin, PluginPermissions
from missy.plugins.loader import PluginLoader
from missy.policy.filesystem import FilesystemPolicyEngine
from missy.policy.network import NetworkPolicyEngine
from missy.policy.shell import ShellPolicyEngine


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def clean_event_bus() -> Generator[None, None, None]:
    """Reset the global event bus before and after each test."""
    event_bus.clear()
    yield
    event_bus.clear()


# ---------------------------------------------------------------------------
# Helper: build a minimal MissyConfig for plugin tests
# ---------------------------------------------------------------------------


def _make_config(
    plugins_enabled: bool = False,
    allowed_plugins: list[str] | None = None,
) -> MissyConfig:
    return MissyConfig(
        network=NetworkPolicy(),
        filesystem=FilesystemPolicy(),
        shell=ShellPolicy(),
        plugins=PluginPolicy(
            enabled=plugins_enabled,
            allowed_plugins=allowed_plugins or [],
        ),
        providers={},
        workspace_path=".",
        audit_log_path="~/.missy/audit.log",
    )


# ---------------------------------------------------------------------------
# Minimal plugin fixtures
# ---------------------------------------------------------------------------


class _OkPlugin(BasePlugin):
    name = "ok_plugin"
    description = "Always initialises successfully."
    permissions = PluginPermissions()

    def initialize(self) -> bool:
        return True

    def execute(self, **kwargs):
        return "ok"


# ---------------------------------------------------------------------------
# Network policy enforcement
# ---------------------------------------------------------------------------


class TestNetworkPolicyEnforcement:
    """Prove that NetworkPolicyEngine blocks and allows hosts as configured."""

    # --- BLOCKED DOMAINS ---------------------------------------------------

    def test_blocked_domain_raises_when_default_deny(self):
        """default_deny=True with no allowlist must raise PolicyViolationError."""
        policy = NetworkPolicy(
            default_deny=True,
            allowed_cidrs=[],
            allowed_domains=[],
            allowed_hosts=[],
        )
        engine = NetworkPolicyEngine(policy)

        with pytest.raises(PolicyViolationError) as exc_info:
            engine.check_host("evil.com")

        assert exc_info.value.category == "network"
        assert "evil.com" in str(exc_info.value)

    def test_blocked_domain_emits_deny_audit_event(self):
        """A denied host must produce exactly one deny audit event."""
        policy = NetworkPolicy(default_deny=True)
        engine = NetworkPolicyEngine(policy)

        with pytest.raises(PolicyViolationError):
            engine.check_host("evil.com")

        deny_events = event_bus.get_events(event_type="network_check", result="deny")
        assert len(deny_events) == 1
        assert deny_events[0].detail["host"] == "evil.com"

    def test_another_blocked_domain(self):
        """Verify blocking is not specific to a single domain name."""
        policy = NetworkPolicy(default_deny=True, allowed_domains=["safe.example.com"])
        engine = NetworkPolicyEngine(policy)

        with pytest.raises(PolicyViolationError):
            engine.check_host("attacker.io")

    # --- BLOCKED CIDRs -----------------------------------------------------

    def test_blocked_cidr_raises_for_ip_outside_allowlist(self):
        """8.8.8.8 is not in 10.0.0.0/8 so must be denied."""
        policy = NetworkPolicy(
            default_deny=True,
            allowed_cidrs=["10.0.0.0/8"],
            allowed_domains=[],
            allowed_hosts=[],
        )
        engine = NetworkPolicyEngine(policy)

        with pytest.raises(PolicyViolationError) as exc_info:
            engine.check_host("8.8.8.8")

        assert exc_info.value.category == "network"

    def test_blocked_cidr_172_not_in_192_block(self):
        """172.16.0.1 is not in 192.168.0.0/16."""
        policy = NetworkPolicy(
            default_deny=True,
            allowed_cidrs=["192.168.0.0/16"],
        )
        engine = NetworkPolicyEngine(policy)

        with pytest.raises(PolicyViolationError):
            engine.check_host("172.16.0.1")

    def test_blocked_cidr_deny_event_carries_ip(self):
        """The audit event for a denied IP must record the IP in its detail."""
        policy = NetworkPolicy(default_deny=True, allowed_cidrs=["10.0.0.0/8"])
        engine = NetworkPolicyEngine(policy)

        with pytest.raises(PolicyViolationError):
            engine.check_host("8.8.8.8")

        events = event_bus.get_events(result="deny")
        assert events[0].detail["host"] == "8.8.8.8"

    # --- ALLOWED CIDRs -----------------------------------------------------

    def test_allowed_cidr_passes_for_ip_inside_block(self):
        """10.0.0.1 is inside 10.0.0.0/8 so must be allowed."""
        policy = NetworkPolicy(
            default_deny=True,
            allowed_cidrs=["10.0.0.0/8"],
            allowed_domains=[],
            allowed_hosts=[],
        )
        engine = NetworkPolicyEngine(policy)

        result = engine.check_host("10.0.0.1")

        assert result is True

    def test_allowed_cidr_emits_allow_event(self):
        """Allowed IP must produce exactly one allow audit event."""
        policy = NetworkPolicy(default_deny=True, allowed_cidrs=["10.0.0.0/8"])
        engine = NetworkPolicyEngine(policy)

        engine.check_host("10.5.100.200")

        allow_events = event_bus.get_events(result="allow")
        assert len(allow_events) == 1
        assert allow_events[0].policy_rule == "cidr:10.0.0.0/8"

    def test_allowed_cidr_last_octet_boundary(self):
        """10.255.255.255 is still inside 10.0.0.0/8."""
        policy = NetworkPolicy(default_deny=True, allowed_cidrs=["10.0.0.0/8"])
        engine = NetworkPolicyEngine(policy)

        assert engine.check_host("10.255.255.255") is True

    def test_allowed_multiple_cidrs_second_matches(self):
        """When the first CIDR does not match, the second must be checked."""
        policy = NetworkPolicy(
            default_deny=True,
            allowed_cidrs=["10.0.0.0/8", "172.16.0.0/12"],
        )
        engine = NetworkPolicyEngine(policy)

        assert engine.check_host("172.20.0.1") is True

    def test_private_localhost_allowed_via_cidr(self):
        """127.0.0.1 is inside 127.0.0.0/8."""
        policy = NetworkPolicy(default_deny=True, allowed_cidrs=["127.0.0.0/8"])
        engine = NetworkPolicyEngine(policy)

        assert engine.check_host("127.0.0.1") is True

    # --- ALLOWED DOMAINS ---------------------------------------------------

    def test_wildcard_domain_allows_subdomain(self):
        """*.github.com must allow api.github.com."""
        policy = NetworkPolicy(
            default_deny=True,
            allowed_cidrs=[],
            allowed_domains=["*.github.com"],
            allowed_hosts=[],
        )
        engine = NetworkPolicyEngine(policy)

        result = engine.check_host("api.github.com")

        assert result is True

    def test_wildcard_domain_allows_root_domain(self):
        """*.github.com must also allow github.com itself."""
        policy = NetworkPolicy(default_deny=True, allowed_domains=["*.github.com"])
        engine = NetworkPolicyEngine(policy)

        assert engine.check_host("github.com") is True

    def test_wildcard_domain_allows_deep_subdomain(self):
        """*.github.com must allow releases.api.github.com."""
        policy = NetworkPolicy(default_deny=True, allowed_domains=["*.github.com"])
        engine = NetworkPolicyEngine(policy)

        assert engine.check_host("releases.api.github.com") is True

    def test_wildcard_domain_blocks_different_root(self):
        """*.github.com must not allow notgithub.com."""
        policy = NetworkPolicy(default_deny=True, allowed_domains=["*.github.com"])
        engine = NetworkPolicyEngine(policy)

        with pytest.raises(PolicyViolationError):
            engine.check_host("notgithub.com")

    def test_exact_domain_match_allowed(self):
        """An exact domain entry must allow that specific host."""
        policy = NetworkPolicy(default_deny=True, allowed_domains=["api.anthropic.com"])
        engine = NetworkPolicyEngine(policy)

        assert engine.check_host("api.anthropic.com") is True

    def test_exact_domain_blocks_subdomains(self):
        """An exact domain entry must not implicitly cover its subdomains."""
        policy = NetworkPolicy(default_deny=True, allowed_domains=["anthropic.com"])
        engine = NetworkPolicyEngine(policy)

        with pytest.raises(PolicyViolationError):
            engine.check_host("evil.anthropic.com")

    def test_wildcard_domain_emits_allow_event_with_rule(self):
        """Allow event policy_rule must reflect the domain pattern."""
        policy = NetworkPolicy(default_deny=True, allowed_domains=["*.github.com"])
        engine = NetworkPolicyEngine(policy)

        engine.check_host("api.github.com")

        event = event_bus.get_events(result="allow")[0]
        assert event.policy_rule == "domain:*.github.com"

    def test_allowed_host_exact_match(self):
        """A host listed in allowed_hosts must be permitted."""
        policy = NetworkPolicy(
            default_deny=True,
            allowed_hosts=["api.openai.com"],
        )
        engine = NetworkPolicyEngine(policy)

        assert engine.check_host("api.openai.com") is True

    # --- DEFAULT ALLOW MODE ------------------------------------------------

    def test_default_allow_mode_permits_anything(self):
        """default_deny=False must pass any host without checking allow-lists."""
        policy = NetworkPolicy(default_deny=False)
        engine = NetworkPolicyEngine(policy)

        assert engine.check_host("evil.com") is True
        assert engine.check_host("8.8.8.8") is True

    def test_default_allow_mode_emits_allow_event(self):
        policy = NetworkPolicy(default_deny=False)
        engine = NetworkPolicyEngine(policy)

        engine.check_host("anything.example.com")

        events = event_bus.get_events(result="allow")
        assert len(events) == 1
        assert events[0].policy_rule == "default_allow"

    # --- INPUT VALIDATION --------------------------------------------------

    def test_empty_host_raises_value_error(self):
        """An empty string must raise ValueError, not PolicyViolationError."""
        policy = NetworkPolicy(default_deny=True)
        engine = NetworkPolicyEngine(policy)

        with pytest.raises(ValueError):
            engine.check_host("")

    def test_session_and_task_ids_recorded_in_event(self):
        """session_id and task_id must appear in the emitted audit event."""
        policy = NetworkPolicy(default_deny=True, allowed_domains=["*.github.com"])
        engine = NetworkPolicyEngine(policy)

        engine.check_host("api.github.com", session_id="s-42", task_id="t-7")

        event = event_bus.get_events()[0]
        assert event.session_id == "s-42"
        assert event.task_id == "t-7"


# ---------------------------------------------------------------------------
# Filesystem policy enforcement
# ---------------------------------------------------------------------------


class TestFilesystemPolicyEnforcement:
    """Prove that FilesystemPolicyEngine enforces workspace boundaries."""

    @pytest.fixture
    def workspace(self, tmp_path: Path) -> Path:
        """Return a temporary directory that acts as the agent workspace."""
        ws = tmp_path / "workspace"
        ws.mkdir()
        return ws

    # --- FORBIDDEN WRITE PATHS -------------------------------------------

    def test_write_to_etc_passwd_is_blocked(self, workspace: Path):
        """Writing to /etc/passwd must be denied when workspace is the only
        allowed write path."""
        policy = FilesystemPolicy(
            allowed_write_paths=[str(workspace)],
            allowed_read_paths=[str(workspace)],
        )
        engine = FilesystemPolicyEngine(policy)

        with pytest.raises(PolicyViolationError) as exc_info:
            engine.check_write("/etc/passwd")

        assert exc_info.value.category == "filesystem"

    def test_write_to_system_binary_is_blocked(self, workspace: Path):
        """/usr/bin/sudo must be denied even when write paths are configured."""
        policy = FilesystemPolicy(allowed_write_paths=[str(workspace)])
        engine = FilesystemPolicyEngine(policy)

        with pytest.raises(PolicyViolationError):
            engine.check_write("/usr/bin/sudo")

    def test_write_outside_workspace_is_blocked(self, workspace: Path):
        """A sibling directory of the workspace must not be writable."""
        policy = FilesystemPolicy(allowed_write_paths=[str(workspace)])
        engine = FilesystemPolicyEngine(policy)

        sibling = workspace.parent / "other_dir"

        with pytest.raises(PolicyViolationError):
            engine.check_write(str(sibling / "file.txt"))

    def test_write_denied_emits_deny_audit_event(self, workspace: Path):
        """A denied write must emit a filesystem_write deny event."""
        policy = FilesystemPolicy(allowed_write_paths=[str(workspace)])
        engine = FilesystemPolicyEngine(policy)

        with pytest.raises(PolicyViolationError):
            engine.check_write("/etc/shadow")

        events = event_bus.get_events(event_type="filesystem_write", result="deny")
        assert len(events) == 1

    def test_path_traversal_attempt_is_blocked(self, workspace: Path):
        """A path that resolves outside the workspace via .. must be denied."""
        policy = FilesystemPolicy(allowed_write_paths=[str(workspace)])
        engine = FilesystemPolicyEngine(policy)

        traversal = str(workspace) + "/../../../etc/passwd"

        with pytest.raises(PolicyViolationError):
            engine.check_write(traversal)

    # --- ALLOWED WRITE PATHS ---------------------------------------------

    def test_write_inside_workspace_is_allowed(self, workspace: Path):
        """A path nested inside the allowed write path must be permitted."""
        policy = FilesystemPolicy(allowed_write_paths=[str(workspace)])
        engine = FilesystemPolicyEngine(policy)

        target = workspace / "output" / "result.txt"

        assert engine.check_write(str(target)) is True

    def test_write_to_workspace_root_is_allowed(self, workspace: Path):
        """The workspace root itself must be a writable location."""
        policy = FilesystemPolicy(allowed_write_paths=[str(workspace)])
        engine = FilesystemPolicyEngine(policy)

        assert engine.check_write(str(workspace / "data.json")) is True

    def test_write_allowed_emits_allow_audit_event(self, workspace: Path):
        """A permitted write must emit a filesystem_write allow event."""
        policy = FilesystemPolicy(allowed_write_paths=[str(workspace)])
        engine = FilesystemPolicyEngine(policy)

        engine.check_write(str(workspace / "notes.txt"))

        events = event_bus.get_events(event_type="filesystem_write", result="allow")
        assert len(events) == 1

    # --- FORBIDDEN READ PATHS --------------------------------------------

    def test_read_from_etc_passwd_is_blocked(self, workspace: Path):
        """/etc/passwd read must be denied when read paths are restricted."""
        policy = FilesystemPolicy(allowed_read_paths=[str(workspace)])
        engine = FilesystemPolicyEngine(policy)

        with pytest.raises(PolicyViolationError) as exc_info:
            engine.check_read("/etc/passwd")

        assert exc_info.value.category == "filesystem"

    def test_read_denied_emits_deny_audit_event(self, workspace: Path):
        """A denied read must emit a filesystem_read deny event."""
        policy = FilesystemPolicy(allowed_read_paths=[str(workspace)])
        engine = FilesystemPolicyEngine(policy)

        with pytest.raises(PolicyViolationError):
            engine.check_read("/root/.ssh/id_rsa")

        events = event_bus.get_events(event_type="filesystem_read", result="deny")
        assert len(events) == 1

    # --- ALLOWED READ PATHS ----------------------------------------------

    def test_read_from_workspace_is_allowed(self, workspace: Path):
        """A path inside the allowed read directory must be permitted."""
        policy = FilesystemPolicy(allowed_read_paths=[str(workspace)])
        engine = FilesystemPolicyEngine(policy)

        assert engine.check_read(str(workspace / "config.yaml")) is True

    def test_multiple_allowed_read_paths(self, tmp_path: Path):
        """A path inside the second allowed read directory must be permitted."""
        dir_a = tmp_path / "a"
        dir_b = tmp_path / "b"
        dir_a.mkdir()
        dir_b.mkdir()

        policy = FilesystemPolicy(allowed_read_paths=[str(dir_a), str(dir_b)])
        engine = FilesystemPolicyEngine(policy)

        assert engine.check_read(str(dir_b / "file.txt")) is True

    def test_empty_allowed_write_paths_denies_all(self, workspace: Path):
        """With an empty allowed_write_paths list every write must be denied."""
        policy = FilesystemPolicy(allowed_write_paths=[])
        engine = FilesystemPolicyEngine(policy)

        with pytest.raises(PolicyViolationError):
            engine.check_write(str(workspace / "anything.txt"))

    def test_empty_allowed_read_paths_denies_all(self, workspace: Path):
        """With an empty allowed_read_paths list every read must be denied."""
        policy = FilesystemPolicy(allowed_read_paths=[])
        engine = FilesystemPolicyEngine(policy)

        with pytest.raises(PolicyViolationError):
            engine.check_read(str(workspace / "anything.txt"))

    def test_session_and_task_ids_in_write_event(self, workspace: Path):
        """session_id and task_id must appear in filesystem write audit events."""
        policy = FilesystemPolicy(allowed_write_paths=[str(workspace)])
        engine = FilesystemPolicyEngine(policy)

        engine.check_write(
            str(workspace / "x.txt"),
            session_id="sess-1",
            task_id="task-2",
        )

        event = event_bus.get_events()[0]
        assert event.session_id == "sess-1"
        assert event.task_id == "task-2"


# ---------------------------------------------------------------------------
# Shell policy enforcement
# ---------------------------------------------------------------------------


class TestShellPolicyEnforcement:
    """Prove that ShellPolicyEngine enforces the shell allow-list."""

    # --- SHELL DISABLED (DEFAULT) ----------------------------------------

    def test_shell_disabled_blocks_all_commands(self):
        """When shell is disabled, every command must raise PolicyViolationError."""
        policy = ShellPolicy(enabled=False, allowed_commands=[])
        engine = ShellPolicyEngine(policy)

        with pytest.raises(PolicyViolationError) as exc_info:
            engine.check_command("ls -la")

        assert exc_info.value.category == "shell"

    def test_shell_disabled_blocks_even_allowed_commands(self):
        """Commands in the allow-list must still be denied when shell is off."""
        policy = ShellPolicy(enabled=False, allowed_commands=["git", "ls"])
        engine = ShellPolicyEngine(policy)

        with pytest.raises(PolicyViolationError):
            engine.check_command("git status")

    def test_shell_disabled_emits_deny_event_with_shell_disabled_rule(self):
        """The deny event must carry the 'shell_disabled' rule string."""
        policy = ShellPolicy(enabled=False)
        engine = ShellPolicyEngine(policy)

        with pytest.raises(PolicyViolationError):
            engine.check_command("whoami")

        events = event_bus.get_events(result="deny")
        assert len(events) == 1
        assert events[0].policy_rule == "shell_disabled"

    def test_shell_disabled_blocks_rm_rf(self):
        """Destructive commands like 'rm -rf /' must be blocked."""
        policy = ShellPolicy(enabled=False)
        engine = ShellPolicyEngine(policy)

        with pytest.raises(PolicyViolationError):
            engine.check_command("rm -rf /")

    def test_shell_disabled_blocks_empty_command(self):
        """An empty command string must be denied when shell is disabled."""
        policy = ShellPolicy(enabled=False)
        engine = ShellPolicyEngine(policy)

        with pytest.raises(PolicyViolationError):
            engine.check_command("")

    def test_default_shell_policy_is_disabled(self):
        """ShellPolicy() with no arguments must have enabled=False."""
        policy = ShellPolicy()
        engine = ShellPolicyEngine(policy)

        with pytest.raises(PolicyViolationError):
            engine.check_command("echo hello")

    # --- SHELL ENABLED WITH ALLOW-LIST -----------------------------------

    def test_allowed_command_git_passes(self):
        """'git' in allowed_commands must permit 'git status'."""
        policy = ShellPolicy(enabled=True, allowed_commands=["git"])
        engine = ShellPolicyEngine(policy)

        result = engine.check_command("git status")

        assert result is True

    def test_allowed_command_with_flags_passes(self):
        """A command with multiple flags must match on the program basename."""
        policy = ShellPolicy(enabled=True, allowed_commands=["ls"])
        engine = ShellPolicyEngine(policy)

        assert engine.check_command("ls -la /tmp") is True

    def test_unlisted_command_is_denied_when_shell_enabled(self):
        """A command not in the allow-list must be denied even if shell is on."""
        policy = ShellPolicy(enabled=True, allowed_commands=["git"])
        engine = ShellPolicyEngine(policy)

        with pytest.raises(PolicyViolationError):
            engine.check_command("curl https://evil.com")

    def test_empty_allowlist_allows_everything_when_enabled(self):
        """An empty allowed_commands list with shell enabled means allow-all."""
        policy = ShellPolicy(enabled=True, allowed_commands=[])
        engine = ShellPolicyEngine(policy)

        assert engine.check_command("ls") is True

    def test_allowed_command_emits_allow_event(self):
        """An allowed command must emit a shell_check allow event."""
        policy = ShellPolicy(enabled=True, allowed_commands=["git"])
        engine = ShellPolicyEngine(policy)

        engine.check_command("git log --oneline")

        events = event_bus.get_events(event_type="shell_check", result="allow")
        assert len(events) == 1
        assert events[0].policy_rule == "cmd:git"

    def test_denied_command_emits_deny_event(self):
        """A denied command must emit a shell_check deny event."""
        policy = ShellPolicy(enabled=True, allowed_commands=["git"])
        engine = ShellPolicyEngine(policy)

        with pytest.raises(PolicyViolationError):
            engine.check_command("wget http://malicious.example")

        events = event_bus.get_events(event_type="shell_check", result="deny")
        assert len(events) == 1

    def test_absolute_path_program_matches_basename(self):
        """'/usr/bin/git status' must match the 'git' allow-list entry."""
        policy = ShellPolicy(enabled=True, allowed_commands=["git"])
        engine = ShellPolicyEngine(policy)

        assert engine.check_command("/usr/bin/git status") is True

    def test_similar_name_not_matched(self):
        """'git' entry must not match 'gitk' — exact basename comparison."""
        policy = ShellPolicy(enabled=True, allowed_commands=["git"])
        engine = ShellPolicyEngine(policy)

        with pytest.raises(PolicyViolationError):
            engine.check_command("gitk")

    def test_malformed_quoted_command_denied(self):
        """An unmatched quote must produce a deny, not a crash."""
        policy = ShellPolicy(enabled=True, allowed_commands=["echo"])
        engine = ShellPolicyEngine(policy)

        with pytest.raises(PolicyViolationError):
            engine.check_command("echo 'unclosed")

    def test_whitespace_only_command_denied(self):
        """A whitespace-only command string must be denied."""
        policy = ShellPolicy(enabled=True, allowed_commands=["ls"])
        engine = ShellPolicyEngine(policy)

        with pytest.raises(PolicyViolationError):
            engine.check_command("   ")

    def test_multiple_allowed_commands(self):
        """Multiple commands in the allow-list must each be individually allowed."""
        policy = ShellPolicy(enabled=True, allowed_commands=["git", "python", "ls"])
        engine = ShellPolicyEngine(policy)

        assert engine.check_command("git status") is True
        assert engine.check_command("python -m pytest") is True
        assert engine.check_command("ls -la") is True


# ---------------------------------------------------------------------------
# Plugin policy enforcement
# ---------------------------------------------------------------------------


class TestPluginPolicyEnforcement:
    """Prove that PluginLoader enforces the plugin allow-list."""

    # --- PLUGINS DISABLED (DEFAULT) --------------------------------------

    def test_plugins_blocked_when_globally_disabled(self):
        """PluginPolicy(enabled=False) must deny any load attempt."""
        config = _make_config(plugins_enabled=False)
        loader = PluginLoader(config)

        with pytest.raises(PolicyViolationError) as exc_info:
            loader.load_plugin(_OkPlugin())

        assert exc_info.value.category == "plugin"

    def test_default_config_blocks_plugins(self):
        """The default MissyConfig must have plugins disabled."""
        from missy.config.settings import get_default_config

        default = get_default_config()
        assert default.plugins.enabled is False

    def test_disabled_plugins_emit_deny_event(self):
        """A denied plugin load must emit a plugin.load deny audit event."""
        config = _make_config(plugins_enabled=False)
        loader = PluginLoader(config)

        with pytest.raises(PolicyViolationError):
            loader.load_plugin(_OkPlugin())

        events = event_bus.get_events(event_type="plugin.load", result="deny")
        assert len(events) == 1
        assert events[0].detail["reason"] == "plugins_disabled"

    def test_plugin_not_in_allowlist_is_blocked(self):
        """A plugin whose name is not in allowed_plugins must be denied."""
        config = _make_config(plugins_enabled=True, allowed_plugins=["other_plugin"])
        loader = PluginLoader(config)

        with pytest.raises(PolicyViolationError) as exc_info:
            loader.load_plugin(_OkPlugin())

        assert "not in allowed_plugins list" in str(exc_info.value)

    def test_plugin_not_in_allowlist_emits_deny_event(self):
        """Deny event must be emitted when plugin name is not allow-listed."""
        config = _make_config(plugins_enabled=True, allowed_plugins=["other_plugin"])
        loader = PluginLoader(config)

        with pytest.raises(PolicyViolationError):
            loader.load_plugin(_OkPlugin())

        events = event_bus.get_events(event_type="plugin.load", result="deny")
        assert len(events) == 1
        assert events[0].detail["reason"] == "not_in_allowed_list"

    # --- PLUGINS ENABLED AND ALLOWED -------------------------------------

    def test_allowed_plugin_loads_successfully(self):
        """A plugin in the allow-list with plugins enabled must load."""
        config = _make_config(plugins_enabled=True, allowed_plugins=["ok_plugin"])
        loader = PluginLoader(config)

        result = loader.load_plugin(_OkPlugin())

        assert result is True

    def test_allowed_plugin_is_enabled_after_load(self):
        """After a successful load the plugin's enabled attribute must be True."""
        config = _make_config(plugins_enabled=True, allowed_plugins=["ok_plugin"])
        loader = PluginLoader(config)
        plugin = _OkPlugin()

        loader.load_plugin(plugin)

        assert plugin.enabled is True

    def test_allowed_plugin_load_emits_allow_event(self):
        """A successful plugin load must emit a plugin.load allow audit event."""
        config = _make_config(plugins_enabled=True, allowed_plugins=["ok_plugin"])
        loader = PluginLoader(config)

        loader.load_plugin(_OkPlugin())

        events = event_bus.get_events(event_type="plugin.load", result="allow")
        assert len(events) == 1
        assert events[0].detail["plugin"] == "ok_plugin"

    def test_execute_allowed_plugin_returns_result(self):
        """A loaded and enabled plugin must execute and return its result."""
        config = _make_config(plugins_enabled=True, allowed_plugins=["ok_plugin"])
        loader = PluginLoader(config)
        loader.load_plugin(_OkPlugin())

        result = loader.execute("ok_plugin")

        assert result == "ok"

    def test_execute_unloaded_plugin_raises_policy_violation(self):
        """Executing a plugin that was never loaded must raise PolicyViolationError."""
        config = _make_config(plugins_enabled=True, allowed_plugins=["ok_plugin"])
        loader = PluginLoader(config)

        with pytest.raises(PolicyViolationError) as exc_info:
            loader.execute("ok_plugin")

        assert "not loaded" in str(exc_info.value)


# ---------------------------------------------------------------------------
# Combined policy engine (PolicyEngine facade)
# ---------------------------------------------------------------------------


class TestPolicyEngineFacade:
    """Verify the PolicyEngine facade delegates correctly to each sub-engine."""

    def _make_full_config(self, workspace: Path) -> MissyConfig:
        return MissyConfig(
            network=NetworkPolicy(
                default_deny=True,
                allowed_cidrs=["10.0.0.0/8"],
                allowed_domains=["*.github.com"],
                allowed_hosts=[],
            ),
            filesystem=FilesystemPolicy(
                allowed_read_paths=[str(workspace)],
                allowed_write_paths=[str(workspace)],
            ),
            shell=ShellPolicy(enabled=True, allowed_commands=["git"]),
            plugins=PluginPolicy(enabled=False),
            providers={},
            workspace_path=str(workspace),
            audit_log_path="~/.missy/audit.log",
        )

    def test_facade_network_allow(self, tmp_path: Path):
        from missy.policy.engine import PolicyEngine

        config = self._make_full_config(tmp_path)
        engine = PolicyEngine(config)

        assert engine.check_network("10.0.0.1") is True

    def test_facade_network_deny(self, tmp_path: Path):
        from missy.policy.engine import PolicyEngine

        config = self._make_full_config(tmp_path)
        engine = PolicyEngine(config)

        with pytest.raises(PolicyViolationError):
            engine.check_network("8.8.8.8")

    def test_facade_filesystem_write_allow(self, tmp_path: Path):
        from missy.policy.engine import PolicyEngine

        config = self._make_full_config(tmp_path)
        engine = PolicyEngine(config)

        assert engine.check_write(str(tmp_path / "out.txt")) is True

    def test_facade_filesystem_write_deny(self, tmp_path: Path):
        from missy.policy.engine import PolicyEngine

        config = self._make_full_config(tmp_path)
        engine = PolicyEngine(config)

        with pytest.raises(PolicyViolationError):
            engine.check_write("/etc/passwd")

    def test_facade_shell_allow(self, tmp_path: Path):
        from missy.policy.engine import PolicyEngine

        config = self._make_full_config(tmp_path)
        engine = PolicyEngine(config)

        assert engine.check_shell("git status") is True

    def test_facade_shell_deny(self, tmp_path: Path):
        from missy.policy.engine import PolicyEngine

        config = self._make_full_config(tmp_path)
        engine = PolicyEngine(config)

        with pytest.raises(PolicyViolationError):
            engine.check_shell("rm -rf /")

    def test_facade_init_and_get_singleton(self, tmp_path: Path):
        """init_policy_engine and get_policy_engine must return the same object."""
        from missy.policy.engine import get_policy_engine, init_policy_engine

        config = self._make_full_config(tmp_path)
        installed = init_policy_engine(config)

        assert get_policy_engine() is installed
