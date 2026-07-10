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
