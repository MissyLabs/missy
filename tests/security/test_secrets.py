"""Tests for missy.security.secrets.SecretsDetector."""

from __future__ import annotations

import pytest

from missy.security.secrets import SecretsDetector


@pytest.fixture()
def detector() -> SecretsDetector:
    return SecretsDetector()


# ---------------------------------------------------------------------------
# scan – individual pattern families
# ---------------------------------------------------------------------------


class TestScanAPIKey:
    def test_finds_api_key_with_equals(self, detector):
        text = 'api_key = "abcdefghijklmnopqrstu"'
        findings = detector.scan(text)
        types = [f["type"] for f in findings]
        assert "api_key" in types

    def test_finds_apikey_no_underscore(self, detector):
        text = 'apikey: "abcdefghijklmnopqrstu12345"'
        findings = detector.scan(text)
        types = [f["type"] for f in findings]
        assert "api_key" in types

    def test_api_key_too_short_not_flagged(self, detector):
        # The pattern requires 20+ chars after the separator
        text = 'api_key = "short"'
        findings = [f for f in detector.scan(text) if f["type"] == "api_key"]
        assert findings == []


class TestScanAWSKey:
    def test_finds_aws_access_key(self, detector):
        text = "My key is AKIAIOSFODNN7EXAMPLE in the config"
        findings = detector.scan(text)
        types = [f["type"] for f in findings]
        assert "aws_key" in types

    def test_non_akia_prefix_not_flagged(self, detector):
        text = "BKIAIOSFODNN7EXAMPLE is not an AWS key"
        findings = [f for f in detector.scan(text) if f["type"] == "aws_key"]
        assert findings == []


class TestScanPrivateKey:
    def test_finds_rsa_private_key(self, detector):
        text = "-----BEGIN RSA PRIVATE KEY-----\nMIIEpAIBAAKCAQEA..."
        findings = detector.scan(text)
        types = [f["type"] for f in findings]
        assert "private_key" in types

    def test_finds_ec_private_key(self, detector):
        text = "-----BEGIN EC PRIVATE KEY-----\nMHQCAQEE..."
        findings = detector.scan(text)
        types = [f["type"] for f in findings]
        assert "private_key" in types

    def test_finds_bare_private_key_header(self, detector):
        text = "-----BEGIN PRIVATE KEY-----"
        findings = detector.scan(text)
        types = [f["type"] for f in findings]
        assert "private_key" in types


class TestScanPassword:
    def test_finds_password_with_equals(self, detector):
        text = "password = supersecret123"
        findings = detector.scan(text)
        types = [f["type"] for f in findings]
        assert "password" in types

    def test_finds_passwd_colon(self, detector):
        text = "passwd: hunter2hunter"
        findings = detector.scan(text)
        types = [f["type"] for f in findings]
        assert "password" in types

    def test_password_too_short_not_flagged(self, detector):
        # pattern requires 8+ non-whitespace chars after separator
        text = "password = short"
        findings = [f for f in detector.scan(text) if f["type"] == "password"]
        assert findings == []


class TestScanToken:
    def test_finds_token_with_equals(self, detector):
        text = "token = abcdefghijklmnopqrstuvwxyz"
        findings = detector.scan(text)
        types = [f["type"] for f in findings]
        assert "token" in types

    def test_finds_secret_key(self, detector):
        text = 'secret = "my-very-long-secret-value-here"'
        findings = detector.scan(text)
        types = [f["type"] for f in findings]
        assert "token" in types


class TestScanGithubToken:
    def test_finds_github_pat(self, detector):
        # ghp_ followed by exactly 36 alphanumeric characters
        text = "ghp_" + "A" * 36
        findings = detector.scan(text)
        types = [f["type"] for f in findings]
        assert "github_token" in types


class TestScanClean:
    def test_clean_text_returns_empty_list(self, detector):
        text = "The quick brown fox jumps over the lazy dog."
        assert detector.scan(text) == []

    def test_empty_string_returns_empty_list(self, detector):
        assert detector.scan("") == []


# ---------------------------------------------------------------------------
# scan – result structure
# ---------------------------------------------------------------------------


class TestScanResultStructure:
    def test_finding_has_required_keys(self, detector):
        text = "AKIAIOSFODNN7EXAMPLE"
        findings = detector.scan(text)
        assert len(findings) >= 1
        for f in findings:
            assert "type" in f
            assert "match_start" in f
            assert "match_end" in f

    def test_findings_sorted_by_position(self, detector):
        # Two secrets at different positions
        text = "AKIAIOSFODNN7EXAMPLE and also -----BEGIN PRIVATE KEY-----"
        findings = detector.scan(text)
        if len(findings) >= 2:
            starts = [f["match_start"] for f in findings]
            assert starts == sorted(starts)

    def test_match_positions_are_correct(self, detector):
        aws_key = "AKIAIOSFODNN7EXAMPLE"
        text = f"key: {aws_key}"
        findings = [f for f in detector.scan(text) if f["type"] == "aws_key"]
        assert len(findings) == 1
        start = findings[0]["match_start"]
        end = findings[0]["match_end"]
        assert text[start:end] == aws_key


# ---------------------------------------------------------------------------
# redact
# ---------------------------------------------------------------------------


class TestRedact:
    def test_clean_text_returned_unchanged(self, detector):
        text = "Nothing secret here."
        assert detector.redact(text) == text

    def test_aws_key_is_redacted(self, detector):
        aws_key = "AKIAIOSFODNN7EXAMPLE"
        text = f"My key is {aws_key}"
        result = detector.redact(text)
        assert aws_key not in result
        assert "[REDACTED]" in result

    def test_redacted_text_does_not_contain_original_secret(self, detector):
        text = "-----BEGIN RSA PRIVATE KEY-----"
        result = detector.redact(text)
        assert "-----BEGIN" not in result

    def test_multiple_secrets_all_redacted(self, detector):
        aws_key = "AKIAIOSFODNN7EXAMPLE"
        private_key_header = "-----BEGIN PRIVATE KEY-----"
        text = f"{aws_key} and {private_key_header}"
        result = detector.redact(text)
        assert aws_key not in result
        assert "BEGIN PRIVATE" not in result
        assert result.count("[REDACTED]") >= 2

    def test_redact_preserves_non_secret_content(self, detector):
        text = "safe text AKIAIOSFODNN7EXAMPLE more safe text"
        result = detector.redact(text)
        assert "safe text" in result
        assert "more safe text" in result


# ---------------------------------------------------------------------------
# has_secrets
# ---------------------------------------------------------------------------


class TestHasSecrets:
    def test_returns_true_for_aws_key(self, detector):
        assert detector.has_secrets("AKIAIOSFODNN7EXAMPLE") is True

    def test_returns_true_for_private_key_header(self, detector):
        assert detector.has_secrets("-----BEGIN RSA PRIVATE KEY-----") is True

    def test_returns_false_for_clean_text(self, detector):
        assert detector.has_secrets("Hello, world!") is False

    def test_returns_false_for_empty_string(self, detector):
        assert detector.has_secrets("") is False

    def test_returns_true_for_github_token(self, detector):
        token = "ghp_" + "B" * 36
        assert detector.has_secrets(token) is True


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------


class TestModuleSingleton:
    def test_module_singleton_is_secrets_detector(self):
        from missy.security.secrets import secrets_detector

        assert isinstance(secrets_detector, SecretsDetector)
