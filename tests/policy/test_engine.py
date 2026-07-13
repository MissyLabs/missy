"""Tests for missy.policy.engine.PolicyEngine and module-level helpers."""

from __future__ import annotations

from collections.abc import Generator
from pathlib import Path

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
from missy.policy import engine as engine_module
from missy.policy.engine import PolicyEngine, get_policy_engine, init_policy_engine

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def reset_engine() -> Generator[None, None, None]:
    """Reset the module-level singleton before and after each test."""
    original = engine_module._engine
    engine_module._engine = None
    event_bus.clear()
    yield
    engine_module._engine = original
    event_bus.clear()


def make_config(
    *,
    default_deny: bool = True,
    allowed_cidrs: list[str] | None = None,
    allowed_domains: list[str] | None = None,
    allowed_hosts: list[str] | None = None,
    read_paths: list[str] | None = None,
    write_paths: list[str] | None = None,
    shell_enabled: bool = False,
    shell_commands: list[str] | None = None,
    shell_unrestricted: bool = False,
) -> MissyConfig:
    return MissyConfig(
        network=NetworkPolicy(
            default_deny=default_deny,
            allowed_cidrs=allowed_cidrs or [],
            allowed_domains=allowed_domains or [],
            allowed_hosts=allowed_hosts or [],
        ),
        filesystem=FilesystemPolicy(
            allowed_read_paths=read_paths or [],
            allowed_write_paths=write_paths or [],
        ),
        shell=ShellPolicy(
            enabled=shell_enabled,
            allowed_commands=shell_commands or [],
            unrestricted=shell_unrestricted,
        ),
        plugins=PluginPolicy(enabled=False, allowed_plugins=[]),
        providers={},
        workspace_path="/tmp/workspace",
        audit_log_path="/tmp/audit.log",
    )


# ---------------------------------------------------------------------------
# PolicyEngine construction
# ---------------------------------------------------------------------------


class TestPolicyEngineConstruction:
    def test_creates_sub_engines(self):
        config = make_config()
        pe = PolicyEngine(config)
        from missy.policy.filesystem import FilesystemPolicyEngine
        from missy.policy.network import NetworkPolicyEngine
        from missy.policy.shell import ShellPolicyEngine

        assert isinstance(pe.network, NetworkPolicyEngine)
        assert isinstance(pe.filesystem, FilesystemPolicyEngine)
        assert isinstance(pe.shell, ShellPolicyEngine)


# ---------------------------------------------------------------------------
# Delegate: check_network
# ---------------------------------------------------------------------------


class TestCheckNetwork:
    def test_allowed_host(self):
        pe = PolicyEngine(make_config(allowed_hosts=["api.example.com"]))
        assert pe.check_network("api.example.com") is True

    def test_denied_host_raises(self):
        pe = PolicyEngine(make_config())
        with pytest.raises(PolicyViolationError):
            pe.check_network("bad.example.com")

    def test_session_task_forwarded(self):
        pe = PolicyEngine(make_config(default_deny=False))
        pe.check_network("anything.com", session_id="S", task_id="T")
        event = event_bus.get_events()[0]
        assert event.session_id == "S"
        assert event.task_id == "T"

    def test_empty_host_raises_value_error(self):
        pe = PolicyEngine(make_config())
        with pytest.raises(ValueError):
            pe.check_network("")


# ---------------------------------------------------------------------------
# Delegate: check_write / check_read
# ---------------------------------------------------------------------------


class TestCheckWrite:
    def test_allowed_write(self, tmp_path: Path):
        pe = PolicyEngine(make_config(write_paths=[str(tmp_path)]))
        assert pe.check_write(str(tmp_path / "out.txt")) is True

    def test_denied_write_raises(self, tmp_path: Path):
        pe = PolicyEngine(make_config())
        with pytest.raises(PolicyViolationError):
            pe.check_write(str(tmp_path / "out.txt"))


class TestCheckRead:
    def test_allowed_read(self, tmp_path: Path):
        pe = PolicyEngine(make_config(read_paths=[str(tmp_path)]))
        assert pe.check_read(str(tmp_path / "file.txt")) is True

    def test_denied_read_raises(self, tmp_path: Path):
        pe = PolicyEngine(make_config())
        with pytest.raises(PolicyViolationError):
            pe.check_read(str(tmp_path / "file.txt"))


# ---------------------------------------------------------------------------
# Delegate: check_shell
# ---------------------------------------------------------------------------


class TestCheckShell:
    def test_allowed_command(self):
        pe = PolicyEngine(make_config(shell_enabled=True, shell_commands=["git"]))
        assert pe.check_shell("git status") is True

    def test_denied_command_raises(self):
        pe = PolicyEngine(make_config(shell_enabled=True, shell_commands=["git"]))
        with pytest.raises(PolicyViolationError):
            pe.check_shell("rm -rf /")

    def test_shell_disabled_always_raises(self):
        pe = PolicyEngine(make_config(shell_enabled=False, shell_commands=["ls"]))
        with pytest.raises(PolicyViolationError):
            pe.check_shell("ls")


class TestCheckShellUnrestricted:
    """The user-reported gap: shell.unrestricted: true must actually make
    `missy` execute shell commands instead of denying with 'allowed_commands
    is empty', while every other, independent policy layer keeps working."""

    def test_empty_allowed_commands_no_longer_denies(self):
        pe = PolicyEngine(make_config(shell_enabled=True, shell_unrestricted=True))
        assert pe.check_shell("curl https://example.com") is True

    def test_still_denied_when_shell_disabled(self):
        pe = PolicyEngine(make_config(shell_enabled=False, shell_unrestricted=True))
        with pytest.raises(PolicyViolationError):
            pe.check_shell("ls")

    def test_filesystem_policy_still_enforced_on_redirect_targets(self):
        """unrestricted only lifts the shell program-name allow-list -- SR-1.7's
        redirect-target-to-filesystem-policy check is a separate layer and
        must still deny writes outside allowed_write_paths."""
        pe = PolicyEngine(make_config(shell_enabled=True, shell_unrestricted=True))
        with pytest.raises(PolicyViolationError, match="Filesystem write denied"):
            pe.check_shell("echo x > /etc/cron.d/pwn")

    def test_filesystem_policy_allows_redirect_within_allowed_write_path(self):
        pe = PolicyEngine(
            make_config(shell_enabled=True, shell_unrestricted=True, write_paths=["/tmp"])
        )
        assert pe.check_shell("echo hi > /tmp/ok.txt") is True

    def test_subshell_command_allowed_end_to_end(self):
        """Follow-up fix: unrestricted must not just skip the allow-list --
        it must also skip the subshell/brace-group parsing-safety
        rejection that runs before the allow-list check, or a command
        this ordinary still gets denied with 'contains subshell'."""
        pe = PolicyEngine(make_config(shell_enabled=True, shell_unrestricted=True))
        assert pe.check_shell("echo $(whoami)") is True

    def test_subshell_command_with_bad_redirect_still_denied(self):
        """Orthogonality check: a subshell-containing command is now
        allowed through the shell layer, but a redirect target outside
        allowed_write_paths must still be caught by the separate
        filesystem policy layer."""
        pe = PolicyEngine(make_config(shell_enabled=True, shell_unrestricted=True))
        with pytest.raises(PolicyViolationError, match="Filesystem write denied"):
            pe.check_shell("echo $(whoami) > /etc/cron.d/pwn")


# ---------------------------------------------------------------------------
# SR-1.7: check_shell() must route redirection targets through the
# filesystem policy engine, not just validate the program name.
# ---------------------------------------------------------------------------
class TestCheckShellRedirectionTargets:
    def test_write_redirect_to_unallowlisted_path_denied(self):
        pe = PolicyEngine(make_config(shell_enabled=True, shell_commands=["echo"]))
        with pytest.raises(PolicyViolationError, match="Filesystem write denied"):
            pe.check_shell("echo x > /etc/cron.d/pwn")

    def test_write_redirect_with_no_spaces_denied(self):
        """An attacker/model could omit whitespace around the operator
        specifically to dodge a naive redirect scanner."""
        pe = PolicyEngine(make_config(shell_enabled=True, shell_commands=["echo"]))
        with pytest.raises(PolicyViolationError, match="Filesystem write denied"):
            pe.check_shell("echo x>/etc/cron.d/pwn")

    def test_write_redirect_to_allowlisted_path_permitted(self):
        pe = PolicyEngine(
            make_config(shell_enabled=True, shell_commands=["echo"], write_paths=["/tmp"])
        )
        assert pe.check_shell("echo hi > /tmp/ok.txt") is True

    def test_read_redirect_from_unallowlisted_path_denied(self):
        pe = PolicyEngine(make_config(shell_enabled=True, shell_commands=["cat"]))
        with pytest.raises(PolicyViolationError, match="Filesystem read denied"):
            pe.check_shell("cat < /etc/shadow")

    def test_read_redirect_from_allowlisted_path_permitted(self):
        pe = PolicyEngine(
            make_config(shell_enabled=True, shell_commands=["cat"], read_paths=["/tmp"])
        )
        assert pe.check_shell("cat < /tmp/data.txt") is True

    def test_fd_duplication_redirect_not_treated_as_file_target(self):
        """ "2>&1" must not be misparsed as a write to a file named "1" --
        it's a file-descriptor duplication, not a filesystem write."""
        pe = PolicyEngine(make_config(shell_enabled=True, shell_commands=["echo"]))
        assert pe.check_shell("echo hi 2>&1") is True

    def test_denied_command_short_circuits_before_redirect_check(self):
        """A denied program name must fail on its own terms -- the error
        should be about the command, not a redirect target that's never
        reached."""
        pe = PolicyEngine(make_config(shell_enabled=True, shell_commands=["git"]))
        with pytest.raises(PolicyViolationError, match="not in the allowed commands list"):
            pe.check_shell("rm x > /tmp/should-not-be-reached")

    def test_multiple_redirects_all_checked(self):
        pe = PolicyEngine(
            make_config(
                shell_enabled=True,
                shell_commands=["cmd"],
                read_paths=["/tmp/in"],
                write_paths=[],
            )
        )
        with pytest.raises(PolicyViolationError, match="Filesystem write denied"):
            pe.check_shell("cmd < /tmp/in/data.txt > /tmp/out/data.txt")


# ---------------------------------------------------------------------------
# Singleton helpers
# ---------------------------------------------------------------------------


class TestSingletonHelpers:
    def test_get_before_init_raises(self):
        with pytest.raises(RuntimeError, match="not been initialised"):
            get_policy_engine()

    def test_init_returns_engine(self):
        config = make_config()
        result = init_policy_engine(config)
        assert isinstance(result, PolicyEngine)

    def test_get_after_init_returns_same_instance(self):
        config = make_config()
        engine = init_policy_engine(config)
        assert get_policy_engine() is engine

    def test_second_init_replaces_engine(self):
        config1 = make_config()
        config2 = make_config(default_deny=False)
        init_policy_engine(config1)
        engine2 = init_policy_engine(config2)
        assert get_policy_engine() is engine2

    def test_init_is_thread_safe(self):
        """Multiple threads calling init_policy_engine should not raise."""
        import threading

        errors: list[Exception] = []
        config = make_config()

        def worker():
            try:
                init_policy_engine(config)
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=worker) for _ in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert errors == []
        # Engine must be set after all threads finish.
        assert get_policy_engine() is not None
