"""Session 11: Secrets detection and redaction tests.

Tests SecretsDetector against various credential formats and
verifies the redaction mechanism works correctly.
"""

from __future__ import annotations

import pytest


class TestSecretsDetectionAPIKeys:
    """API key and token detection."""

    def test_detects_generic_api_key(self) -> None:
        from missy.security.secrets import SecretsDetector

        d = SecretsDetector()
        findings = d.scan('api_key = "sk_test_abc123def456ghi789jkl012"')
        assert len(findings) > 0
        assert any("api_key" in f["type"] or "token" in f["type"] for f in findings)

    def test_detects_aws_access_key(self) -> None:
        from missy.security.secrets import SecretsDetector

        d = SecretsDetector()
        findings = d.scan("AKIAIOSFODNN7EXAMPLE")
        assert any(f["type"] == "aws_key" for f in findings)

    def test_detects_github_pat(self) -> None:
        from missy.security.secrets import SecretsDetector

        d = SecretsDetector()
        findings = d.scan("ghp_ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghij")
        assert any(f["type"] == "github_token" for f in findings)

    def test_detects_github_oauth(self) -> None:
        from missy.security.secrets import SecretsDetector

        d = SecretsDetector()
        findings = d.scan("gho_ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghij")
        assert any(f["type"] == "github_oauth" for f in findings)

    def test_detects_anthropic_key(self) -> None:
        from missy.security.secrets import SecretsDetector

        d = SecretsDetector()
        findings = d.scan("sk-ant-api03-ABCDEFGHIJKLMNOPQRST")
        assert any(f["type"] == "anthropic_key" for f in findings)

    def test_detects_openai_key(self) -> None:
        from missy.security.secrets import SecretsDetector

        d = SecretsDetector()
        findings = d.scan("sk-proj-ABCDEFGHIJKLMNOPQRSTUVWXYZab")
        assert any(f["type"] == "openai_key" for f in findings)

    def test_detects_stripe_key(self) -> None:
        from missy.security.secrets import SecretsDetector

        d = SecretsDetector()
        findings = d.scan("sk_live_ABCDEFGHIJKLMNOPQRSTUVWXYZab")
        assert any(f["type"] == "stripe_key" for f in findings)

    def test_detects_slack_token(self) -> None:
        from missy.security.secrets import SecretsDetector

        d = SecretsDetector()
        findings = d.scan("xoxb-1234567890-abcdefghij")
        assert any(f["type"] == "slack_token" for f in findings)

    def test_detects_jwt(self) -> None:
        from missy.security.secrets import SecretsDetector

        d = SecretsDetector()
        jwt = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiIxMjM0NTY3ODkwIiwibmFtZSI6IkpvaG4gRG9lIiwiaWF0IjoxNTE2MjM5MDIyfQ.SflKxwRJSMeKKF2QT4fwpMeJf36POk6yJV_adQssw5c"
        findings = d.scan(jwt)
        assert any(f["type"] == "jwt" for f in findings)

    def test_detects_private_key_header(self) -> None:
        from missy.security.secrets import SecretsDetector

        d = SecretsDetector()
        findings = d.scan("-----BEGIN RSA PRIVATE KEY-----")
        assert any(f["type"] == "private_key" for f in findings)

    def test_detects_gcp_key(self) -> None:
        from missy.security.secrets import SecretsDetector

        d = SecretsDetector()
        findings = d.scan("AIzaSyA-ABCDEFGHIJKLMNOPQRSTUVWXYZ1234567")
        assert any(f["type"] == "gcp_key" for f in findings)

    def test_detects_huggingface_token(self) -> None:
        from missy.security.secrets import SecretsDetector

        d = SecretsDetector()
        findings = d.scan("hf_ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghij")
        assert any(f["type"] == "huggingface_token" for f in findings)


class TestSecretsDetectionDatabases:
    """Database connection string detection."""

    def test_detects_postgres_url(self) -> None:
        from missy.security.secrets import SecretsDetector

        d = SecretsDetector()
        findings = d.scan("postgres://admin:secretpass@db.example.com:5432/mydb")
        assert any(f["type"] == "db_connection_string" for f in findings)

    def test_detects_mongodb_url(self) -> None:
        from missy.security.secrets import SecretsDetector

        d = SecretsDetector()
        findings = d.scan("mongodb://user:password@cluster0.example.net/test")
        assert any(f["type"] == "db_connection_string" for f in findings)

    def test_detects_redis_url(self) -> None:
        from missy.security.secrets import SecretsDetector

        d = SecretsDetector()
        findings = d.scan("redis://default:mypassword@redis.example.com:6379")
        assert any(f["type"] == "db_connection_string" for f in findings)


class TestSecretsNoFalsePositives:
    """Ensure normal text doesn't trigger false positives."""

    def test_normal_text(self) -> None:
        from missy.security.secrets import SecretsDetector

        d = SecretsDetector()
        findings = d.scan("How do I configure nginx on Ubuntu 22.04?")
        assert len(findings) == 0

    def test_short_strings_not_matched(self) -> None:
        from missy.security.secrets import SecretsDetector

        d = SecretsDetector()
        findings = d.scan("API key is short")
        assert len(findings) == 0

    def test_code_comments_not_matched(self) -> None:
        from missy.security.secrets import SecretsDetector

        d = SecretsDetector()
        findings = d.scan("# This is a comment about the token handling logic")
        assert len(findings) == 0


class TestSecretsRedaction:
    """Test the redact() method."""

    def test_redact_replaces_secret(self) -> None:
        from missy.security.secrets import SecretsDetector

        d = SecretsDetector()
        text = "My key is ghp_ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghij"
        redacted = d.redact(text)
        assert "ghp_ABCDEF" not in redacted
        assert "***" in redacted or "REDACTED" in redacted.upper()

    def test_redact_preserves_context(self) -> None:
        from missy.security.secrets import SecretsDetector

        d = SecretsDetector()
        text = "Configured with key ghp_ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghij for access"
        redacted = d.redact(text)
        assert "Configured" in redacted
        assert "for access" in redacted

    def test_redact_no_secrets_returns_unchanged(self) -> None:
        from missy.security.secrets import SecretsDetector

        d = SecretsDetector()
        text = "This is clean text with no secrets."
        assert d.redact(text) == text

    def test_redact_multiple_secrets(self) -> None:
        from missy.security.secrets import SecretsDetector

        d = SecretsDetector()
        text = (
            "AWS: AKIAIOSFODNN7EXAMPLE and "
            "GitHub: ghp_ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghij"
        )
        redacted = d.redact(text)
        assert "AKIAIOSFODNN7EXAMPLE" not in redacted
        assert "ghp_ABCDEF" not in redacted

    def test_empty_text(self) -> None:
        from missy.security.secrets import SecretsDetector

        d = SecretsDetector()
        assert d.scan("") == []
        assert d.redact("") == ""

    def test_scan_returns_finding_structure(self) -> None:
        from missy.security.secrets import SecretsDetector

        d = SecretsDetector()
        findings = d.scan("AKIAIOSFODNN7EXAMPLE")
        assert len(findings) > 0
        finding = findings[0]
        assert "type" in finding
        assert "match_start" in finding
        assert "match_end" in finding
