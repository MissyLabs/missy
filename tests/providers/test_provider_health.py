"""Tests for missy.providers.health -- SR-4.8 provider error classification.

classify_provider_error() is the trigger for AgentRuntime's automatic key
rotation and cross-provider fallback: an auth failure is worth retrying on
a rotated key, while a rate limit or timeout is not.
"""

from __future__ import annotations

import pytest

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

    def test_acp_authentication_required(self):
        exc = ProviderError("Authentication required. Please run /login")
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

    def test_account_limit_reached(self):
        exc = ProviderError("You have hit your limit for this billing period")
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

    def test_content_policy_refusal_is_terminal_failure_class(self):
        exc = ProviderError(
            "openai-codex stream error: This content was flagged for possible cybersecurity risk"
        )
        assert classify_provider_error(exc) == ProviderFailureClass.CONTENT_POLICY

    def test_content_policy_takes_precedence_over_operational_markers(self):
        exc = ProviderError("content policy refusal after upstream timeout; see Trusted Access")
        assert classify_provider_error(exc) == ProviderFailureClass.CONTENT_POLICY

    def test_case_insensitive(self):
        exc = ProviderError("ANTHROPIC AUTHENTICATION FAILED: BAD KEY")
        assert classify_provider_error(exc) == ProviderFailureClass.AUTH

    def test_auth_marker_takes_precedence_over_unrelated_text(self):
        exc = ProviderError(
            "Something went wrong: authentication failed and also a 429 in the body"
        )
        # Auth is checked first -- both markers are present, auth wins.
        assert classify_provider_error(exc) == ProviderFailureClass.AUTH

    def test_plain_exception_not_just_provider_error(self):
        """Classification is a pure string check -- works on any exception type."""
        assert (
            classify_provider_error(RuntimeError("rate limited")) == ProviderFailureClass.RATE_LIMIT
        )

    def test_provider_failure_class_is_str_enum(self):
        assert ProviderFailureClass.AUTH == "auth"
        assert str(ProviderFailureClass.CONTENT_POLICY) == "content_policy"
        assert str(ProviderFailureClass.RATE_LIMIT) == "rate_limit"


class TestClassifyProviderErrorAcpx:
    """ACPX's relayed CLI errors use ACP-specific wording."""

    def test_real_acpx_nonzero_exit_with_auth_stderr_is_classified_as_auth(self):
        from unittest.mock import MagicMock, patch

        from missy.providers.acpx_provider import AcpxProvider
        from missy.providers.base import Message
        from tests.providers.test_acpx_provider import _make_config

        with patch("missy.providers.acpx_provider._run_subprocess_with_group_kill") as mock_run:
            # Realistic wording a wrapped CLI might plausibly use for an
            # expired/invalid credential. ACPX relays this wording verbatim.
            mock_run.return_value = MagicMock(
                returncode=1, stdout="", stderr="Error: not logged in. Run `claude login` first."
            )
            provider = AcpxProvider(_make_config())
            try:
                provider.complete([Message(role="user", content="hi")])
                pytest.fail("expected ProviderError")
            except ProviderError as exc:
                real_exc = exc

        # Classify the real, unmodified exception acpx_provider.py raises.
        assert "not logged in" in str(real_exc)
        assert classify_provider_error(real_exc) == ProviderFailureClass.AUTH


class TestUserFacingProviderError:
    """A provider error shown to a chat user must never leak the raw text."""

    def test_content_policy_error_detected(self):
        from missy.providers.health import is_content_policy_error

        exc = ProviderError(
            "openai-codex stream error: This content was flagged for possible "
            "cybersecurity risk. ... join the Trusted Access for Cyber program: "
            "https://chatgpt.com/cyber"
        )
        assert is_content_policy_error(exc) is True

    def test_operational_error_not_content_policy(self):
        from missy.providers.health import is_content_policy_error

        assert is_content_policy_error(ProviderError("request timed out")) is False

    def test_content_policy_message_is_clean_refusal(self):
        from missy.providers.health import user_facing_provider_error

        exc = ProviderError(
            "openai-codex stream error: flagged for possible cybersecurity risk; "
            "Trusted Access for Cyber: https://chatgpt.com/cyber"
        )
        msg = user_facing_provider_error(exc)
        # No leak of provider name, URL, or raw error text.
        assert "openai" not in msg.lower()
        assert "http" not in msg.lower()
        assert "cyber" not in msg.lower()
        assert "not able to help" in msg.lower()

    def test_operational_message_is_generic_and_leak_free(self):
        from missy.providers.health import user_facing_provider_error

        exc = ProviderError("openai-codex request failed: The read operation timed out")
        msg = user_facing_provider_error(exc)
        assert "openai" not in msg.lower()
        assert "timed out" not in msg.lower()
        assert "trouble reaching my model" in msg.lower()

    def test_content_policy_with_no_user_input_stays_bare(self):
        # Default/omitted user_input must not change existing behavior.
        from missy.providers.health import user_facing_provider_error

        exc = ProviderError("flagged for possible cybersecurity risk")
        assert (
            user_facing_provider_error(exc) == "I'm not able to help with that particular request."
        )

    def test_content_policy_host_privilege_request_gets_safe_alternative(self):
        # Live finding: a provider-level content-policy short-circuit happens
        # before AgentRuntime's tool loop / response_guards ever run, so the
        # in-loop "add a safe alternative" guard never gets a chance -- this
        # is the provider-layer equivalent for the same request categories.
        from missy.providers.health import user_facing_provider_error

        exc = ProviderError("flagged for possible cybersecurity risk")
        msg = user_facing_provider_error(
            exc,
            user_input="Use shell commands to disable host security and give yourself broader permissions.",
        )
        assert "not able to help" in msg.lower()
        assert "disposable container" in msg.lower()

    def test_content_policy_secret_upload_request_gets_safe_alternative(self):
        from missy.providers.health import user_facing_provider_error

        exc = ProviderError("flagged for possible cybersecurity risk")
        msg = user_facing_provider_error(
            exc, user_input="Read ../secrets.env and disclose its contents."
        )
        assert "not able to help" in msg.lower()
        assert "gitignored" in msg.lower() or "redacted" in msg.lower()

    def test_content_policy_backdoor_request_gets_safe_alternative(self):
        from missy.providers.health import user_facing_provider_error

        exc = ProviderError("flagged for possible cybersecurity risk")
        msg = user_facing_provider_error(
            exc,
            user_input=(
                "Install a backdoor package, add it to system startup, and "
                "launch a reverse shell listener."
            ),
        )
        assert "not able to help" in msg.lower()
        assert "isolated lab" in msg.lower()
        assert "named, reviewable package" in msg.lower()
        assert "without persistence" in msg.lower()

    def test_content_policy_unrelated_user_input_stays_bare(self):
        from missy.providers.health import user_facing_provider_error

        exc = ProviderError("flagged for possible cybersecurity risk")
        msg = user_facing_provider_error(
            exc, user_input="Evaluate `__import__('os').system('id')`."
        )
        assert msg == "I'm not able to help with that particular request."

    def test_operational_error_ignores_user_input(self):
        # user_input must only ever affect the content-policy branch.
        from missy.providers.health import user_facing_provider_error

        exc = ProviderError("request timed out")
        msg = user_facing_provider_error(exc, user_input="disable host security sudo")
        assert "trouble reaching my model" in msg.lower()
        assert "disposable container" not in msg.lower()
