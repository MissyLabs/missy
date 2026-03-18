"""Property-based tests for SecretsDetector using hypothesis.

Tests that the secrets detector never crashes on arbitrary input, reliably
detects known credential patterns, and correctly redacts detected secrets.
"""

from __future__ import annotations

import string

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from missy.security.secrets import SecretsDetector


@pytest.fixture
def detector():
    return SecretsDetector()


# ---------------------------------------------------------------------------
# Known credential samples that MUST be detected
# ---------------------------------------------------------------------------

KNOWN_SECRETS = {
    "aws_key": "AKIAIOSFODNN7EXAMPLE",
    "github_token": "ghp_ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghij",
    "github_oauth": "gho_ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghij",
    "private_key": "-----BEGIN RSA PRIVATE KEY-----",
    "stripe_key": "sk_live_ABCDEFGHIJKLMNOPQRSTUVwx",
    "slack_token": "xoxb-1234567890-abcdefghij",
    "jwt": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0.dozjgNryP4J3jVmNHl0w5N_XgL0n3I9PlFUP0THsR8U",
    "anthropic_key": "sk-ant-ABCDEFGHIJKLMNOPQRSTu",
    "openai_key": "sk-proj-ABCDEFGHIJKLMNOPQRSTu",
    "gcp_key": "AIzaSyBCDEFGHIJKLMNOPQRSTUVWXYZ01234567",
    "gitlab_token": "glpat-ABCDEFGHIJKLMNOPQRSTu",
    "npm_token": "npm_ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghij",
    "sendgrid_key": "SG.ABCDEFGHIJKLMNOPQRSTuv.ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrst",
    "newrelic_key": "NRAK-ABCDEFGHIJKLMNOPQRSTUVWXYZ0",
    "huggingface_token": "hf_ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefgh",
    "age_secret_key": "AGE-SECRET-KEY-" + "A" * 59,
    "render_key": "rnd_" + "A" * 32,
}


# ---------------------------------------------------------------------------
# Property tests
# ---------------------------------------------------------------------------


class TestDetectorNeverCrashes:
    """The detector must handle any text input without crashing."""

    @given(st.text(min_size=0, max_size=10000))
    @settings(max_examples=200, suppress_health_check=[HealthCheck.too_slow])
    def test_scan_never_crashes(self, text):
        d = SecretsDetector()
        findings = d.scan(text)
        assert isinstance(findings, list)
        for f in findings:
            assert "type" in f
            assert "match_start" in f
            assert "match_end" in f
            assert f["match_start"] >= 0
            assert f["match_end"] <= len(text)
            assert f["match_end"] > f["match_start"]

    @given(st.text(min_size=0, max_size=10000))
    @settings(max_examples=200, suppress_health_check=[HealthCheck.too_slow])
    def test_redact_never_crashes(self, text):
        d = SecretsDetector()
        result = d.redact(text)
        assert isinstance(result, str)

    @given(st.text(min_size=0, max_size=10000))
    @settings(max_examples=200, suppress_health_check=[HealthCheck.too_slow])
    def test_has_secrets_never_crashes(self, text):
        d = SecretsDetector()
        result = d.has_secrets(text)
        assert isinstance(result, bool)

    @given(st.binary(min_size=0, max_size=5000))
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_scan_decoded_binary(self, data):
        text = data.decode("utf-8", errors="replace")
        d = SecretsDetector()
        findings = d.scan(text)
        assert isinstance(findings, list)


class TestKnownSecretsDetected:
    """All known credential patterns must be detected."""

    @given(
        st.sampled_from(list(KNOWN_SECRETS.items())),
        st.text(min_size=0, max_size=100),
        st.text(min_size=0, max_size=100),
    )
    @settings(max_examples=200, suppress_health_check=[HealthCheck.too_slow])
    def test_known_secret_detected_with_context(self, secret_item, prefix, suffix):
        secret_type, secret_value = secret_item
        text = prefix + secret_value + suffix
        d = SecretsDetector()
        findings = d.scan(text)
        detected_types = {f["type"] for f in findings}
        assert secret_type in detected_types, (
            f"Secret type {secret_type!r} not detected in: {text[:200]!r}"
        )

    def test_all_known_secrets_in_single_scan(self):
        """All known secrets embedded in one text should all be found."""
        d = SecretsDetector()
        parts = []
        for secret_type, value in KNOWN_SECRETS.items():
            parts.append(f"config_{secret_type} = {value}")
        text = "\n".join(parts)
        findings = d.scan(text)
        detected = {f["type"] for f in findings}
        for secret_type in KNOWN_SECRETS:
            assert secret_type in detected, f"Missing detection of {secret_type}"


class TestRedaction:
    """Redaction must not leak any part of a detected secret."""

    @given(st.sampled_from(list(KNOWN_SECRETS.items())))
    @settings(max_examples=50)
    def test_redacted_text_hides_secret(self, secret_item):
        secret_type, secret_value = secret_item
        text = f"Here is a secret: {secret_value} end."
        d = SecretsDetector()
        redacted = d.redact(text)
        # The full secret value should not appear in the redacted text
        assert secret_value not in redacted
        assert "[REDACTED]" in redacted

    def test_overlapping_secrets_merged(self):
        """Overlapping or adjacent matches should be merged into one redaction."""
        d = SecretsDetector()
        # Two secrets that may overlap
        text = "api_key=sk-ant-ABCDEFGHIJKLMNOPQRSTu"
        redacted = d.redact(text)
        assert "[REDACTED]" in redacted
        # The key value should not be visible
        assert "sk-ant-" not in redacted

    @given(st.text(
        alphabet=string.ascii_lowercase + " ",
        min_size=10,
        max_size=200,
    ))
    @settings(max_examples=50)
    def test_clean_text_not_redacted(self, text):
        """Text with no secrets should be returned unchanged."""
        d = SecretsDetector()
        redacted = d.redact(text)
        assert redacted == text


class TestScanOrdering:
    """Scan results must be sorted by match position."""

    def test_multiple_secrets_sorted_by_position(self):
        d = SecretsDetector()
        text = "first: ghp_ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefgh second: AKIAIOSFODNN7EXAMPLE"
        findings = d.scan(text)
        positions = [f["match_start"] for f in findings]
        assert positions == sorted(positions)


class TestHasSecretsConsistency:
    """has_secrets must be consistent with scan results."""

    @given(st.sampled_from(list(KNOWN_SECRETS.values())))
    @settings(max_examples=50)
    def test_has_secrets_true_for_known(self, secret):
        d = SecretsDetector()
        assert d.has_secrets(secret) is True

    @given(st.text(
        alphabet=string.ascii_lowercase + " ",
        min_size=5,
        max_size=100,
    ))
    @settings(max_examples=50)
    def test_has_secrets_consistent_with_scan(self, text):
        d = SecretsDetector()
        has = d.has_secrets(text)
        findings = d.scan(text)
        assert has == (len(findings) > 0)


class TestDbConnectionStrings:
    """Database connection string patterns must be detected."""

    @given(st.sampled_from([
        "postgres://admin:password123@db.example.com:5432/mydb",
        "mysql://root:secret@localhost/testdb",
        "mongodb://user:pass@mongo.host:27017/db",
        "redis://default:myredispass@cache.internal:6379",
    ]))
    def test_db_connection_strings_detected(self, conn_string):
        d = SecretsDetector()
        findings = d.scan(f"DATABASE_URL={conn_string}")
        types = {f["type"] for f in findings}
        assert "db_connection_string" in types


class TestPasswordPatterns:
    """Password-like values must be detected."""

    @given(st.text(
        alphabet=string.ascii_letters + string.digits + "!@#$%",
        min_size=8,
        max_size=30,
    ))
    @settings(max_examples=30)
    def test_password_pattern(self, pw_value):
        d = SecretsDetector()
        text = f'password="{pw_value}"'
        findings = d.scan(text)
        types = {f["type"] for f in findings}
        assert "password" in types or "token" in types or len(findings) > 0
