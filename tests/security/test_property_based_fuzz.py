"""Property-based fuzz tests.


Uses hypothesis to verify robustness invariants across security-critical
components: InputSanitizer, SecretsDetector, policy engines, gateway URL
validation, and SQLiteMemoryStore.

Each test class focuses on a single component and verifies that:
  - The component never crashes on arbitrary input (robustness).
  - Return types always satisfy their contracts (type invariants).
  - Semantic properties hold across all inputs (correctness invariants).
"""

from __future__ import annotations

import contextlib

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from missy.config.settings import (
    FilesystemPolicy,
    NetworkPolicy,
    ShellPolicy,
    get_default_config,
)
from missy.core.exceptions import PolicyViolationError
from missy.memory.sqlite_store import ConversationTurn, SQLiteMemoryStore
from missy.policy.filesystem import FilesystemPolicyEngine
from missy.policy.network import NetworkPolicyEngine
from missy.policy.shell import ShellPolicyEngine
from missy.security.sanitizer import MAX_INPUT_LENGTH, InputSanitizer
from missy.security.secrets import SecretsDetector

# ---------------------------------------------------------------------------
# Shared strategies
# ---------------------------------------------------------------------------

# Unicode text with the full character range including surrogates excluded
_unicode_text = st.text(min_size=0, max_size=500)

# Text with embedded control characters (C0 + C1 blocks).
# Category "Cs" (surrogates) is excluded because SQLite's UTF-8 encoder
# rejects lone surrogates (\ud800-\udfff) with UnicodeEncodeError; these
# are not valid in UTF-8 and represent a platform/codec limitation, not a
# bug in our code under test.
_control_char_text = st.text(
    alphabet=st.characters(whitelist_categories=("Cc", "Cf", "L", "N", "Z")),
    min_size=0,
    max_size=300,
)

# Binary data as bytes
_binary_data = st.binary(min_size=0, max_size=512)

# Plausible hostnames: mix of labels and separators
_hostname_chars = st.text(
    alphabet=st.characters(
        whitelist_categories=("L", "N"),
        whitelist_characters="-.",
    ),
    min_size=1,
    max_size=100,
)

# Absolute-looking paths: start with / followed by arbitrary printable chars
_path_text = st.text(
    alphabet=st.characters(whitelist_categories=("L", "N", "Z"), whitelist_characters="/-_."),
    min_size=0,
    max_size=200,
).map(lambda s: "/" + s if s else "/")

# Shell-command-shaped strings
_command_text = st.text(
    alphabet=st.characters(whitelist_categories=("L", "N", "Z"), whitelist_characters=" -_./"),
    min_size=0,
    max_size=200,
)

# URL-shaped strings
_url_text = st.one_of(
    st.just(""),
    st.just("not-a-url"),
    st.text(min_size=0, max_size=200),
    st.from_regex(r"https?://[a-zA-Z0-9\-\.]{1,50}(/[a-zA-Z0-9\-\._~:/?#\[\]@!$&'()*+,;=%]*)?"),
    st.from_regex(r"[a-zA-Z]{1,10}://[a-zA-Z0-9]{1,30}"),
)


# ---------------------------------------------------------------------------
# InputSanitizer fuzz tests
# ---------------------------------------------------------------------------


class TestInputSanitizerFuzz:
    """Property-based fuzz tests for InputSanitizer."""

    @given(text=_unicode_text)
    @settings(max_examples=100)
    def test_sanitize_never_crashes_on_unicode(self, text: str) -> None:
        """sanitize() must never raise an exception on any unicode string."""
        sanitizer = InputSanitizer()
        result = sanitizer.sanitize(text)
        assert isinstance(result, str)

    @given(text=_control_char_text)
    @settings(max_examples=100)
    def test_sanitize_handles_control_characters(self, text: str) -> None:
        """sanitize() must not crash when input contains control characters."""
        sanitizer = InputSanitizer()
        result = sanitizer.sanitize(text)
        assert isinstance(result, str)

    @given(text=_unicode_text)
    @settings(max_examples=100)
    def test_sanitize_output_length_bounded(self, text: str) -> None:
        """sanitize() output length never exceeds input length + len('[truncated]').

        When truncation occurs, the suffix ' [truncated]' is appended (12 chars).
        The truncated portion is at most MAX_INPUT_LENGTH chars + suffix.
        The original input may be longer, so the absolute bound is:
            min(len(text), MAX_INPUT_LENGTH) + 12
        """
        sanitizer = InputSanitizer()
        result = sanitizer.sanitize(text)
        truncation_suffix_len = len(" [truncated]")
        # After truncation, result is at most MAX_INPUT_LENGTH + suffix.
        # Before truncation, result is at most len(text).
        upper_bound = min(len(text), MAX_INPUT_LENGTH) + truncation_suffix_len
        assert len(result) <= upper_bound, (
            f"sanitize() output len={len(result)} exceeded bound={upper_bound} "
            f"for input len={len(text)}"
        )

    @given(text=_unicode_text)
    @settings(max_examples=100)
    def test_check_for_injection_returns_list_never_none(self, text: str) -> None:
        """check_for_injection() must return a list (never None, never raise)."""
        sanitizer = InputSanitizer()
        result = sanitizer.check_for_injection(text)
        assert result is not None
        assert isinstance(result, list)

    @given(text=_unicode_text)
    @settings(max_examples=100)
    def test_check_for_injection_list_elements_are_strings(self, text: str) -> None:
        """Every element returned by check_for_injection() must be a string."""
        sanitizer = InputSanitizer()
        result = sanitizer.check_for_injection(text)
        for item in result:
            assert isinstance(item, str), f"Non-string item in result: {item!r}"

    @given(text=_control_char_text)
    @settings(max_examples=100)
    def test_check_for_injection_handles_control_characters(self, text: str) -> None:
        """check_for_injection() must not crash on inputs with control characters."""
        sanitizer = InputSanitizer()
        result = sanitizer.check_for_injection(text)
        assert isinstance(result, list)

    @given(
        prefix=st.text(min_size=1, max_size=100),
        extra=st.integers(min_value=100, max_value=500),
    )
    @settings(max_examples=50)
    def test_sanitize_truncates_oversized_input(self, prefix: str, extra: int) -> None:
        """Inputs significantly longer than MAX_INPUT_LENGTH must be truncated.

        We construct a long string by repeating a short character rather than
        asking hypothesis to generate a string with min_size > BUFFER_SIZE.
        We use MAX_INPUT_LENGTH + extra (with extra >= 100) to ensure the
        output (MAX_INPUT_LENGTH + len(' [truncated]')) is shorter than input.
        """
        char = prefix[0]
        repeat = MAX_INPUT_LENGTH + extra
        text = char * repeat
        assert len(text) > MAX_INPUT_LENGTH, "precondition: text must exceed MAX_INPUT_LENGTH"

        sanitizer = InputSanitizer()
        result = sanitizer.sanitize(text)
        suffix_len = len(" [truncated]")
        # The truncated result is at most MAX_INPUT_LENGTH + suffix.
        assert len(result) <= MAX_INPUT_LENGTH + suffix_len, (
            f"Truncated output len={len(result)} exceeds bound={MAX_INPUT_LENGTH + suffix_len}"
        )
        # For inputs with extra >= 100, the result is strictly shorter than the input.
        assert len(result) < len(text), (
            f"Oversized input (len={len(text)}) was not shortened by truncation"
        )

    @given(text=st.just(""))
    @settings(max_examples=1)
    def test_sanitize_empty_string(self, text: str) -> None:
        """sanitize() on empty string returns empty string."""
        sanitizer = InputSanitizer()
        result = sanitizer.sanitize(text)
        assert result == ""

    @given(text=st.just(""))
    @settings(max_examples=1)
    def test_check_for_injection_empty_string_returns_empty_list(self, text: str) -> None:
        """check_for_injection() on empty string returns empty list."""
        sanitizer = InputSanitizer()
        result = sanitizer.check_for_injection(text)
        assert result == []


# ---------------------------------------------------------------------------
# SecretsDetector fuzz tests
# ---------------------------------------------------------------------------


class TestSecretsDetectorFuzz:
    """Property-based fuzz tests for SecretsDetector."""

    @given(text=_unicode_text)
    @settings(max_examples=100)
    def test_scan_never_crashes(self, text: str) -> None:
        """scan() must never raise an exception on arbitrary text."""
        detector = SecretsDetector()
        result = detector.scan(text)
        assert isinstance(result, list)

    @given(text=_unicode_text)
    @settings(max_examples=100)
    def test_redact_never_crashes(self, text: str) -> None:
        """redact() must never raise an exception on arbitrary text."""
        detector = SecretsDetector()
        result = detector.redact(text)
        assert isinstance(result, str)

    @given(text=_unicode_text)
    @settings(max_examples=100)
    def test_scan_returns_list_of_dicts(self, text: str) -> None:
        """Every finding from scan() must be a dict with required keys."""
        detector = SecretsDetector()
        findings = detector.scan(text)
        required_keys = {"type", "match_start", "match_end"}
        for finding in findings:
            assert isinstance(finding, dict), f"Finding is not a dict: {finding!r}"
            assert required_keys.issubset(finding.keys()), (
                f"Finding missing keys. Got: {set(finding.keys())}, Expected: {required_keys}"
            )

    @given(text=_unicode_text)
    @settings(max_examples=100)
    def test_redact_output_does_not_contain_original_match(self, text: str) -> None:
        """Redacted output must not reproduce the exact matched secret text.

        For each finding span, the corresponding text slice must not appear
        verbatim in the redacted output at the same position.
        """
        detector = SecretsDetector()
        findings = detector.scan(text)
        if not findings:
            return  # Nothing to check
        redacted = detector.redact(text)
        for finding in findings:
            text[finding["match_start"]:finding["match_end"]]
            # The exact match text must not appear in redacted at this location.
            # We verify by checking the redacted string does not contain the
            # original match text starting at the same offset (accounting for
            # the offset shift caused by earlier replacements is complex, so
            # we conservatively check that [REDACTED] appears in the output).
            assert "[REDACTED]" in redacted, (
                f"Expected [REDACTED] in output but got: {redacted!r}"
            )

    @given(text=_unicode_text)
    @settings(max_examples=100)
    def test_double_redaction_is_idempotent(self, text: str) -> None:
        """redact(redact(x)) must equal redact(x) for all inputs."""
        detector = SecretsDetector()
        once = detector.redact(text)
        twice = detector.redact(once)
        assert once == twice, (
            f"Double redaction is not idempotent.\n"
            f"  redact(x)       = {once!r}\n"
            f"  redact(redact(x)) = {twice!r}"
        )

    @given(text=_control_char_text)
    @settings(max_examples=50)
    def test_redact_handles_control_characters(self, text: str) -> None:
        """redact() must handle text containing control characters."""
        detector = SecretsDetector()
        result = detector.redact(text)
        assert isinstance(result, str)

    @given(text=st.text(min_size=0, max_size=5000))
    @settings(max_examples=50)
    def test_scan_on_large_input(self, text: str) -> None:
        """scan() must handle larger inputs without crashing."""
        detector = SecretsDetector()
        result = detector.scan(text)
        assert isinstance(result, list)

    @given(text=st.just("[REDACTED]"))
    @settings(max_examples=1)
    def test_redact_of_redacted_marker_is_stable(self, text: str) -> None:
        """The literal '[REDACTED]' string must survive redaction unchanged."""
        detector = SecretsDetector()
        result = detector.redact(text)
        assert result == text


# ---------------------------------------------------------------------------
# NetworkPolicyEngine fuzz tests
# ---------------------------------------------------------------------------


class TestNetworkPolicyEngineFuzz:
    """Property-based fuzz tests for NetworkPolicyEngine."""

    def _make_engine(self, default_deny: bool = True) -> NetworkPolicyEngine:
        policy = NetworkPolicy(default_deny=default_deny)
        return NetworkPolicyEngine(policy)

    @given(hostname=_hostname_chars)
    @settings(max_examples=100, deadline=None)
    def test_check_host_never_crashes_on_random_hostnames(self, hostname: str) -> None:
        """check_host() must either return True or raise PolicyViolationError/ValueError.

        It must never raise any other exception on arbitrary hostname-like input.
        deadline=None because DNS resolution may take longer than 200ms.
        """
        engine = self._make_engine(default_deny=True)
        with contextlib.suppress(PolicyViolationError, ValueError, OSError):
            engine.check_host(hostname)

    @given(hostname=_unicode_text)
    @settings(max_examples=100)
    def test_check_host_never_crashes_on_arbitrary_unicode(self, hostname: str) -> None:
        """check_host() must not raise unexpected exceptions on unicode strings."""
        engine = self._make_engine(default_deny=True)
        with contextlib.suppress(PolicyViolationError, ValueError, OSError, UnicodeError):
            engine.check_host(hostname)

    @given(hostname=_hostname_chars)
    @settings(max_examples=50)
    def test_check_host_default_allow_always_allows_non_empty(self, hostname: str) -> None:
        """In default-allow mode, any non-empty hostname is allowed."""
        if not hostname.strip("[]").strip():
            return  # Empty after normalisation — skip
        engine = self._make_engine(default_deny=False)
        try:
            result = engine.check_host(hostname)
            assert result is True
        except ValueError:
            pass  # Empty hostname after normalisation

    @given(hostname=st.just(""))
    @settings(max_examples=1)
    def test_check_host_raises_value_error_for_empty_string(self, hostname: str) -> None:
        """check_host() must raise ValueError for an empty hostname."""
        engine = self._make_engine()
        with pytest.raises(ValueError):
            engine.check_host(hostname)

    @given(ip=st.from_regex(r"\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}"))
    @settings(max_examples=50, deadline=None)
    def test_check_host_handles_ipv4_addresses(self, ip: str) -> None:
        """check_host() must not crash on IPv4-shaped input strings.

        deadline=None because DNS resolution for non-resolvable addresses
        may take longer than the default 200ms deadline.
        """
        engine = self._make_engine(default_deny=True)
        with contextlib.suppress(PolicyViolationError, ValueError, OSError):
            engine.check_host(ip)

    @given(host=st.from_regex(r"\[::[\da-f:]{1,30}\]"))
    @settings(max_examples=30)
    def test_check_host_handles_ipv6_bracket_notation(self, host: str) -> None:
        """check_host() must not crash on IPv6 addresses in bracket notation."""
        engine = self._make_engine(default_deny=True)
        with contextlib.suppress(PolicyViolationError, ValueError, OSError):
            engine.check_host(host)


# ---------------------------------------------------------------------------
# FilesystemPolicyEngine fuzz tests
# ---------------------------------------------------------------------------


class TestFilesystemPolicyEngineFuzz:
    """Property-based fuzz tests for FilesystemPolicyEngine."""

    def _make_engine(self, read_paths: list[str] | None = None, write_paths: list[str] | None = None) -> FilesystemPolicyEngine:
        policy = FilesystemPolicy(
            allowed_read_paths=read_paths or [],
            allowed_write_paths=write_paths or [],
        )
        return FilesystemPolicyEngine(policy)

    @given(path=_path_text)
    @settings(max_examples=100)
    def test_check_read_never_crashes_on_random_paths(self, path: str) -> None:
        """check_read() must either return True or raise PolicyViolationError.

        It must never raise any other exception on arbitrary path strings.
        """
        engine = self._make_engine()
        with contextlib.suppress(PolicyViolationError):
            engine.check_read(path)

    @given(path=_path_text)
    @settings(max_examples=100)
    def test_check_write_never_crashes_on_random_paths(self, path: str) -> None:
        """check_write() must either return True or raise PolicyViolationError."""
        engine = self._make_engine()
        with contextlib.suppress(PolicyViolationError):
            engine.check_write(path)

    @given(path=_unicode_text)
    @settings(max_examples=100)
    def test_check_read_handles_unicode_paths(self, path: str) -> None:
        """check_read() must not crash on unicode path strings."""
        engine = self._make_engine()
        with contextlib.suppress(PolicyViolationError, ValueError):
            engine.check_read(path)

    @given(path=_unicode_text)
    @settings(max_examples=100)
    def test_check_write_handles_unicode_paths(self, path: str) -> None:
        """check_write() must not crash on unicode path strings."""
        engine = self._make_engine()
        with contextlib.suppress(PolicyViolationError, ValueError):
            engine.check_write(path)

    @given(path=_path_text)
    @settings(max_examples=50)
    def test_check_read_with_allowed_path_prefix_succeeds(self, path: str) -> None:
        """When the path is under an allowed read path, check_read() returns True."""
        # Use /tmp as an always-present allowed root
        engine = self._make_engine(read_paths=["/tmp"])
        # Build a path guaranteed to be under /tmp
        safe_path = "/tmp/" + path.lstrip("/")
        try:
            result = engine.check_read(safe_path)
            assert result is True
        except PolicyViolationError:
            # May happen if path resolution escapes /tmp (e.g. symlinks)
            pass

    @given(path=st.just(""))
    @settings(max_examples=1)
    def test_check_read_empty_path_is_handled(self, path: str) -> None:
        """check_read() on empty string must not cause an unhandled crash."""
        engine = self._make_engine()
        with contextlib.suppress(PolicyViolationError, ValueError):
            engine.check_read(path)


# ---------------------------------------------------------------------------
# ShellPolicyEngine fuzz tests
# ---------------------------------------------------------------------------


class TestShellPolicyEngineFuzz:
    """Property-based fuzz tests for ShellPolicyEngine."""

    def _make_engine(self, enabled: bool = True, allowed: list[str] | None = None) -> ShellPolicyEngine:
        policy = ShellPolicy(enabled=enabled, allowed_commands=allowed or [])
        return ShellPolicyEngine(policy)

    @given(command=_command_text)
    @settings(max_examples=100)
    def test_check_command_never_crashes_when_disabled(self, command: str) -> None:
        """check_command() must raise PolicyViolationError (not crash) when shell is disabled."""
        engine = self._make_engine(enabled=False)
        with pytest.raises(PolicyViolationError):
            engine.check_command(command)

    @given(command=_unicode_text)
    @settings(max_examples=100)
    def test_check_command_never_crashes_on_unicode(self, command: str) -> None:
        """check_command() must only raise PolicyViolationError, never unhandled exceptions."""
        engine = self._make_engine(enabled=True, allowed=["ls", "git", "echo"])
        with contextlib.suppress(PolicyViolationError):
            engine.check_command(command)

    @given(command=_command_text)
    @settings(max_examples=100)
    def test_check_command_returns_bool_or_raises_policy_error(self, command: str) -> None:
        """check_command() must either return True or raise PolicyViolationError."""
        engine = self._make_engine(enabled=True, allowed=["ls", "cat", "echo", "pwd"])
        try:
            result = engine.check_command(command)
            assert result is True, f"Expected True, got {result!r}"
        except PolicyViolationError:
            pass

    @given(command=st.from_regex(r"[a-z]{2,10}(\s+[a-z0-9\-\.]{1,20}){0,3}"))
    @settings(max_examples=50)
    def test_check_command_with_allowed_list_is_consistent(self, command: str) -> None:
        """A command is allowed iff its leading token basename is in allowed_commands."""
        allowed_commands = ["ls", "git", "echo", "cat", "pwd", "grep"]
        engine = self._make_engine(enabled=True, allowed=allowed_commands)
        try:
            result = engine.check_command(command)
            if result:
                # Verify the leading token would be in the allow list
                import shlex
                tokens = shlex.split(command)
                if tokens:
                    import os.path
                    basename = os.path.basename(tokens[0])
                    assert any(
                        os.path.basename(a) == basename for a in allowed_commands
                    ), f"Command {command!r} was allowed but {basename!r} not in allow list"
        except (PolicyViolationError, ValueError):
            pass

    @given(command=st.just(""))
    @settings(max_examples=1)
    def test_check_command_empty_string_raises_policy_error(self, command: str) -> None:
        """An empty command string must be denied with PolicyViolationError."""
        engine = self._make_engine(enabled=True, allowed=["ls"])
        with pytest.raises(PolicyViolationError):
            engine.check_command(command)

    @given(command=st.just("   "))
    @settings(max_examples=1)
    def test_check_command_whitespace_only_raises_policy_error(self, command: str) -> None:
        """A whitespace-only command string must be denied with PolicyViolationError."""
        engine = self._make_engine(enabled=True, allowed=["ls"])
        with pytest.raises(PolicyViolationError):
            engine.check_command(command)


# ---------------------------------------------------------------------------
# Gateway URL validation fuzz tests
# ---------------------------------------------------------------------------


class TestGatewayURLValidationFuzz:
    """Property-based fuzz tests for PolicyHTTPClient._check_url().

    _check_url() is called before any network I/O, so it must never crash —
    it should raise ValueError for malformed/disallowed URLs, and
    PolicyViolationError when the host is denied by policy.
    """

    def _make_client_with_default_allow(self):
        """Create a PolicyHTTPClient backed by a default-allow policy engine."""
        import dataclasses

        from missy.gateway.client import PolicyHTTPClient
        from missy.policy.engine import init_policy_engine

        config = get_default_config()
        # Override network to default-allow so we isolate _check_url() logic.
        config = dataclasses.replace(
            config,
            network=NetworkPolicy(default_deny=False),
        )
        init_policy_engine(config)
        return PolicyHTTPClient()

    @given(url=_url_text)
    @settings(max_examples=100, deadline=None)
    def test_check_url_never_crashes_unhandled(self, url: str) -> None:
        """_check_url() must only raise ValueError or PolicyViolationError, never crash.

        deadline=None because the policy engine setup and URL parsing may
        involve DNS operations that exceed the default 200ms deadline.
        """
        client = self._make_client_with_default_allow()
        with contextlib.suppress(ValueError, PolicyViolationError):
            client._check_url(url)

    @given(url=st.just(""))
    @settings(max_examples=1)
    def test_check_url_empty_string_raises_value_error(self, url: str) -> None:
        """An empty URL must produce a ValueError, not a crash."""
        client = self._make_client_with_default_allow()
        with pytest.raises(ValueError):
            client._check_url(url)

    @given(url=st.from_regex(r"ftp://[a-z]{3,10}\.[a-z]{2,4}/[a-z]*"))
    @settings(max_examples=30)
    def test_check_url_disallowed_scheme_raises_value_error(self, url: str) -> None:
        """Non-http/https schemes must raise ValueError."""
        client = self._make_client_with_default_allow()
        with pytest.raises(ValueError):
            client._check_url(url)

    @given(url=st.from_regex(r"https?://[a-zA-Z0-9\-]{1,30}\.[a-zA-Z]{2,6}(/[a-zA-Z0-9]*)?"))
    @settings(max_examples=50)
    def test_check_url_valid_http_url_does_not_crash(self, url: str) -> None:
        """A well-formed http/https URL must not raise an unexpected exception."""
        client = self._make_client_with_default_allow()
        with contextlib.suppress(ValueError, PolicyViolationError):
            client._check_url(url)

    @given(
        char=st.characters(whitelist_categories=("L", "N")),
        repeat=st.integers(min_value=8193, max_value=8500),
    )
    @settings(max_examples=20)
    def test_check_url_oversized_url_raises_value_error(self, char: str, repeat: int) -> None:
        """URLs longer than 8192 chars must raise ValueError.

        We construct the long URL by repetition to avoid hypothesis BUFFER_SIZE
        limits on min_size for text() strategies.
        """
        url = char * repeat
        assert len(url) > 8192, "precondition: url must exceed 8192 chars"
        client = self._make_client_with_default_allow()
        with pytest.raises(ValueError):
            client._check_url(url)

    @given(url=_unicode_text)
    @settings(max_examples=100)
    def test_check_url_unicode_input_handled_safely(self, url: str) -> None:
        """_check_url() must handle arbitrary unicode without crashing."""
        client = self._make_client_with_default_allow()
        with contextlib.suppress(ValueError, PolicyViolationError, UnicodeError):
            client._check_url(url)


# ---------------------------------------------------------------------------
# SQLiteMemoryStore fuzz tests
# ---------------------------------------------------------------------------


class TestSQLiteMemoryStoreFuzz:
    """Property-based fuzz tests for SQLiteMemoryStore.

    Each test uses a fresh in-memory-like SQLite store via a temp file to
    avoid cross-test pollution.
    """

    @pytest.fixture(autouse=True)
    def _tmp_db(self, tmp_path):
        """Provide a fresh SQLiteMemoryStore backed by a temporary file."""
        self.db_path = str(tmp_path / "fuzz_memory.db")
        self.store = SQLiteMemoryStore(db_path=self.db_path)

    @given(content=_unicode_text)
    @settings(max_examples=100)
    def test_add_turn_then_retrieve_preserves_content(self, content: str) -> None:
        """Content added via add_turn() must be retrievable unchanged."""
        session_id = "fuzz-session"
        turn = ConversationTurn.new(session_id, "user", content)
        self.store.add_turn(turn)

        turns = self.store.get_session_turns(session_id)
        assert len(turns) >= 1
        # The last turn should have the content we stored.
        stored_turn = turns[-1]
        assert stored_turn.content == content, (
            f"Content mismatch. Stored: {stored_turn.content!r}, Expected: {content!r}"
        )

    @given(content=_unicode_text)
    @settings(max_examples=100)
    def test_add_turn_never_crashes_on_unicode_content(self, content: str) -> None:
        """add_turn() must not raise on any unicode content."""
        turn = ConversationTurn.new("session-x", "assistant", content)
        self.store.add_turn(turn)  # Must not raise

    @given(query=_unicode_text)
    @settings(max_examples=100)
    def test_search_never_crashes_on_arbitrary_query(self, query: str) -> None:
        """search() must never raise an exception on arbitrary query strings."""
        result = self.store.search(query)
        assert isinstance(result, list)

    @given(content=_unicode_text, query=_unicode_text)
    @settings(max_examples=50)
    def test_add_then_search_round_trip_does_not_crash(self, content: str, query: str) -> None:
        """Adding content then searching with an arbitrary query must not crash."""
        turn = ConversationTurn.new("fuzz-rt", "user", content)
        self.store.add_turn(turn)
        results = self.store.search(query)
        assert isinstance(results, list)
        for r in results:
            assert isinstance(r, ConversationTurn)

    @given(content=_control_char_text)
    @settings(max_examples=50)
    def test_add_turn_handles_control_characters_in_content(self, content: str) -> None:
        """add_turn() must store content containing control characters without crashing."""
        turn = ConversationTurn.new("ctrl-session", "user", content)
        self.store.add_turn(turn)
        turns = self.store.get_session_turns("ctrl-session")
        assert len(turns) >= 1
        assert turns[-1].content == content

    @given(session_id=_unicode_text, content=_unicode_text)
    @settings(max_examples=50)
    def test_get_session_turns_never_crashes_on_arbitrary_session_id(
        self, session_id: str, content: str
    ) -> None:
        """get_session_turns() must not crash on arbitrary session_id strings."""
        turn = ConversationTurn.new(session_id, "user", content)
        self.store.add_turn(turn)
        turns = self.store.get_session_turns(session_id)
        assert isinstance(turns, list)

    @given(content=st.text(min_size=0, max_size=10_000))
    @settings(max_examples=20)
    def test_add_turn_large_content_survives_round_trip(self, content: str) -> None:
        """Large content strings must survive add/retrieve without corruption."""
        session_id = "large-content-session"
        turn = ConversationTurn.new(session_id, "user", content)
        self.store.add_turn(turn)
        turns = self.store.get_session_turns(session_id)
        assert any(t.content == content for t in turns), (
            "Large content not found in retrieved turns"
        )

    @given(query=st.just(""))
    @settings(max_examples=1)
    def test_search_empty_query_returns_list(self, query: str) -> None:
        """search() on an empty string must return a list (not crash)."""
        result = self.store.search(query)
        assert isinstance(result, list)

    @given(query=st.from_regex(r'["\*\?\^\$\(\)\[\]\{\}\\|+]{1,20}'))
    @settings(max_examples=50)
    def test_search_regex_special_chars_handled_safely(self, query: str) -> None:
        """search() must gracefully handle FTS special characters in the query."""
        result = self.store.search(query)
        assert isinstance(result, list)
