"""Tests for missy.providers.health -- SR-4.8 provider error classification.

classify_provider_error() is the trigger for AgentRuntime's automatic key
rotation and cross-provider fallback: an auth failure is worth retrying on
a rotated key, while a rate limit or timeout is not.
"""

from __future__ import annotations

from missy.core.exceptions import ProviderError
from missy.providers.health import ProviderFailureClass, classify_provider_error


class TestClassifyProviderError:
    def test_anthropic_authentication_failure(self):
        exc = ProviderError("Anthropic authentication failed: invalid x-api-key")
        assert classify_provider_error(exc) == ProviderFailureClass.AUTH

    def test_openai_authentication_failure(self):
        exc = ProviderError("OpenAI authentication failed: Incorrect API key provided")
        assert classify_provider_error(exc) == ProviderFailureClass.AUTH

    def test_unauthorized_marker(self):
        exc = ProviderError("request failed: 401 Unauthorized")
        assert classify_provider_error(exc) == ProviderFailureClass.AUTH

    def test_anthropic_rate_limited(self):
        exc = ProviderError("Anthropic rate limited: 429 Too Many Requests")
        assert classify_provider_error(exc) == ProviderFailureClass.RATE_LIMIT

    def test_openai_rate_limited(self):
        exc = ProviderError("OpenAI rate limited: quota exceeded")
        assert classify_provider_error(exc) == ProviderFailureClass.RATE_LIMIT

    def test_bare_429_marker(self):
        exc = ProviderError("upstream returned 429")
        assert classify_provider_error(exc) == ProviderFailureClass.RATE_LIMIT

    def test_timeout_marker(self):
        exc = ProviderError("OpenAI request timed out after 30s: ReadTimeout")
        assert classify_provider_error(exc) == ProviderFailureClass.TIMEOUT

    def test_generic_timeout_word(self):
        exc = ProviderError("connection timeout while dialing host")
        assert classify_provider_error(exc) == ProviderFailureClass.TIMEOUT

    def test_unknown_error_defaults_to_unknown(self):
        exc = ProviderError("Anthropic API error: 500 Internal Server Error")
        assert classify_provider_error(exc) == ProviderFailureClass.UNKNOWN

    def test_case_insensitive(self):
        exc = ProviderError("ANTHROPIC AUTHENTICATION FAILED: BAD KEY")
        assert classify_provider_error(exc) == ProviderFailureClass.AUTH

    def test_auth_marker_takes_precedence_over_unrelated_text(self):
        exc = ProviderError("Something went wrong: authentication failed and also a 429 in the body")
        # Auth is checked first -- both markers are present, auth wins.
        assert classify_provider_error(exc) == ProviderFailureClass.AUTH

    def test_plain_exception_not_just_provider_error(self):
        """Classification is a pure string check -- works on any exception type."""
        assert classify_provider_error(RuntimeError("rate limited")) == ProviderFailureClass.RATE_LIMIT

    def test_provider_failure_class_is_str_enum(self):
        assert ProviderFailureClass.AUTH == "auth"
        assert str(ProviderFailureClass.RATE_LIMIT) == "rate_limit"


class TestClassifyProviderErrorAcpxBlindSpot:
    """Documents a real, verified gap: unlike Anthropic/OpenAI/Codex (which
    catch their SDK's own structured exception types and deliberately
    construct a ProviderError mentioning "authentication failed"/"rate
    limit(ed)"), AcpxProvider wraps an external CLI subprocess and has no
    equivalent structured signal -- its generic nonzero-exit path just
    relays the wrapped CLI's raw stderr text verbatim. Unless that
    external, unowned CLI's own wording happens to contain one of this
    module's marker words, a real acpx auth or rate-limit failure
    classifies as UNKNOWN, silently skipping the rotate_key()/fallback
    response an equivalent Anthropic/OpenAI/Codex failure would trigger.
    """

    def test_real_acpx_nonzero_exit_with_auth_like_stderr_is_not_classified_as_auth(self):
        from unittest.mock import MagicMock, patch

        from missy.providers.acpx_provider import AcpxProvider
        from missy.providers.base import Message

        from tests.providers.test_acpx_provider import _make_config

        with patch("missy.providers.acpx_provider._run_subprocess_with_group_kill") as mock_run:
            # Realistic wording a wrapped CLI might plausibly use for an
            # expired/invalid credential -- deliberately NOT the exact
            # literal "authentication failed"/"unauthorized" markers this
            # module's classifier looks for, since acpx never constructs
            # that vocabulary itself; it only relays whatever the external
            # CLI's stderr actually says.
            mock_run.return_value = MagicMock(
                returncode=1, stdout="", stderr="Error: not logged in. Run `claude login` first."
            )
            provider = AcpxProvider(_make_config())
            try:
                provider.complete([Message(role="user", content="hi")])
                pytest.fail("expected ProviderError")
            except ProviderError as exc:
                real_exc = exc

        # The real, unmodified exception acpx_provider.py actually raises --
        # confirms this is a genuine classification gap, not a hypothetical.
        assert "not logged in" in str(real_exc)
        assert classify_provider_error(real_exc) == ProviderFailureClass.UNKNOWN
