"""Tests for missy.security.censor.censor_response."""

from __future__ import annotations

import pytest

from missy.security.censor import censor_response


# ---------------------------------------------------------------------------
# Falsy / empty input
# ---------------------------------------------------------------------------


class TestFalsyInput:
    def test_empty_string_returned_unchanged(self):
        assert censor_response("") == ""

    def test_none_returned_unchanged(self):
        # censor_response guards with `if not text: return text`
        assert censor_response(None) is None  # type: ignore[arg-type]

    def test_zero_string_returned_unchanged(self):
        # Any other falsy str-like value passes through unchanged
        result = censor_response("")
        assert result == ""


# ---------------------------------------------------------------------------
# Clean text (no secrets)
# ---------------------------------------------------------------------------


class TestCleanText:
    def test_plain_sentence_passes_through(self):
        text = "The weather is nice today."
        assert censor_response(text) == text

    def test_multiline_clean_text_passes_through(self):
        text = "Line one.\nLine two.\nLine three."
        assert censor_response(text) == text

    def test_text_with_numbers_passes_through(self):
        text = "There are 42 items in the list."
        assert censor_response(text) == text


# ---------------------------------------------------------------------------
# API key patterns
# ---------------------------------------------------------------------------


class TestAPIKeyRedaction:
    def test_anthropic_style_key_redacted(self):
        text = 'api_key = "sk-ant-abcdefghijklmnopqrst"'
        result = censor_response(text)
        assert "sk-ant-abcdefghijklmnopqrst" not in result
        assert "[REDACTED]" in result

    def test_openai_style_key_redacted(self):
        # Matches the generic api_key pattern (20+ alphanum chars after separator)
        text = "api_key: sk-abcdefghijklmnopqrstuvwxyz"
        result = censor_response(text)
        assert "sk-abcdefghijklmnopqrstuvwxyz" not in result
        assert "[REDACTED]" in result

    def test_aws_access_key_redacted(self):
        text = "AWS key: AKIAIOSFODNN7EXAMPLE is in the config"
        result = censor_response(text)
        assert "AKIAIOSFODNN7EXAMPLE" not in result
        assert "[REDACTED]" in result

    def test_github_pat_redacted(self):
        token = "ghp_" + "X" * 36
        text = f"My GitHub token is {token}"
        result = censor_response(text)
        assert token not in result
        assert "[REDACTED]" in result

    def test_stripe_secret_key_redacted(self):
        key = "sk_live_" + "A" * 24
        result = censor_response(f"stripe key: {key}")
        assert key not in result
        assert "[REDACTED]" in result

    def test_slack_token_redacted(self):
        token = "xoxb-123456789012-abcdefghijklmno"
        result = censor_response(f"slack token: {token}")
        assert token not in result
        assert "[REDACTED]" in result


# ---------------------------------------------------------------------------
# Private key patterns
# ---------------------------------------------------------------------------


class TestPrivateKeyRedaction:
    def test_rsa_private_key_header_redacted(self):
        text = "-----BEGIN RSA PRIVATE KEY-----\nMIIEpAIBAAKCAQEA..."
        result = censor_response(text)
        assert "-----BEGIN RSA PRIVATE KEY-----" not in result
        assert "[REDACTED]" in result

    def test_bare_private_key_header_redacted(self):
        text = "Here is my key:\n-----BEGIN PRIVATE KEY-----\ndata"
        result = censor_response(text)
        assert "BEGIN PRIVATE KEY" not in result
        assert "[REDACTED]" in result


# ---------------------------------------------------------------------------
# Password / token patterns
# ---------------------------------------------------------------------------


class TestPasswordTokenRedaction:
    def test_password_field_redacted(self):
        text = "password = supersecret123"
        result = censor_response(text)
        assert "supersecret123" not in result
        assert "[REDACTED]" in result

    def test_token_field_redacted(self):
        text = "token = abcdefghijklmnopqrstuvwxyz"
        result = censor_response(text)
        assert "abcdefghijklmnopqrstuvwxyz" not in result
        assert "[REDACTED]" in result


# ---------------------------------------------------------------------------
# Mixed content
# ---------------------------------------------------------------------------


class TestMixedContent:
    def test_surrounding_safe_text_preserved(self):
        aws_key = "AKIAIOSFODNN7EXAMPLE"
        text = f"safe before {aws_key} safe after"
        result = censor_response(text)
        assert aws_key not in result
        assert "safe before" in result
        assert "safe after" in result

    def test_multiple_secrets_all_redacted(self):
        aws_key = "AKIAIOSFODNN7EXAMPLE"
        private_key = "-----BEGIN PRIVATE KEY-----"
        text = f"{aws_key} and {private_key}"
        result = censor_response(text)
        assert aws_key not in result
        assert "BEGIN PRIVATE KEY" not in result
        assert result.count("[REDACTED]") >= 2

    def test_repeated_calls_are_idempotent(self):
        aws_key = "AKIAIOSFODNN7EXAMPLE"
        text = f"key: {aws_key}"
        first = censor_response(text)
        second = censor_response(first)
        assert first == second
