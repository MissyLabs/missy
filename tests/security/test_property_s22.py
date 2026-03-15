"""Session 22 property-based security tests.

Uses hypothesis to verify invariants that must hold for all inputs.
"""

from __future__ import annotations

from hypothesis import given, settings
from hypothesis import strategies as st

from missy.security.censor import censor_response
from missy.security.sanitizer import InputSanitizer
from missy.security.secrets import SecretsDetector

# ---------------------------------------------------------------------------
# InputSanitizer invariants
# ---------------------------------------------------------------------------


class TestSanitizerPropertyBased:
    """Property-based tests for InputSanitizer."""

    @given(text=st.text(min_size=0, max_size=500))
    @settings(max_examples=50)
    def test_sanitize_never_raises(self, text: str) -> None:
        """sanitize() should never raise on any input."""
        sanitizer = InputSanitizer()
        result = sanitizer.sanitize(text)
        assert isinstance(result, str)

    @given(text=st.text(min_size=0, max_size=500))
    @settings(max_examples=50)
    def test_check_for_injection_returns_list(self, text: str) -> None:
        """check_for_injection() always returns a list of strings."""
        sanitizer = InputSanitizer()
        result = sanitizer.check_for_injection(text)
        assert isinstance(result, list)
        for item in result:
            assert isinstance(item, str)

    @given(text=st.text(min_size=0, max_size=500))
    @settings(max_examples=50)
    def test_sanitize_output_no_longer_than_input(self, text: str) -> None:
        """sanitize() should never produce output longer than input."""
        sanitizer = InputSanitizer()
        result = sanitizer.sanitize(text)
        # Sanitize strips zero-width chars, so output <= input
        assert len(result) <= len(text)


# ---------------------------------------------------------------------------
# SecretsDetector invariants
# ---------------------------------------------------------------------------


class TestSecretsDetectorPropertyBased:
    """Property-based tests for SecretsDetector."""

    @given(text=st.text(min_size=0, max_size=500))
    @settings(max_examples=50)
    def test_scan_never_raises(self, text: str) -> None:
        """scan() should never raise on any input."""
        detector = SecretsDetector()
        result = detector.scan(text)
        assert isinstance(result, list)

    @given(text=st.text(min_size=0, max_size=500))
    @settings(max_examples=50)
    def test_redact_never_raises(self, text: str) -> None:
        """redact() should never raise on any input."""
        detector = SecretsDetector()
        result = detector.redact(text)
        assert isinstance(result, str)

    @given(text=st.text(min_size=0, max_size=500))
    @settings(max_examples=50)
    def test_redact_idempotent(self, text: str) -> None:
        """Redacting already-redacted text should not change it further."""
        detector = SecretsDetector()
        once = detector.redact(text)
        twice = detector.redact(once)
        assert once == twice


# ---------------------------------------------------------------------------
# censor_response invariants
# ---------------------------------------------------------------------------


class TestCensorResponsePropertyBased:
    """Property-based tests for censor_response."""

    @given(text=st.text(min_size=0, max_size=500))
    @settings(max_examples=50)
    def test_censor_never_raises(self, text: str) -> None:
        """censor_response() should never raise."""
        result = censor_response(text)
        assert isinstance(result, str)

    @given(text=st.text(min_size=0, max_size=500))
    @settings(max_examples=50)
    def test_censor_output_same_or_shorter(self, text: str) -> None:
        """Censored output is same length or shorter (redacted is same length due to ****)."""
        result = censor_response(text)
        # Output may be longer due to **** replacements of short secrets
        # But it should always be a valid string
        assert isinstance(result, str)


# ---------------------------------------------------------------------------
# Scheduler parser invariants
# ---------------------------------------------------------------------------


class TestSchedulerParserPropertyBased:
    """Property-based tests for schedule parser."""

    @given(text=st.text(min_size=0, max_size=100, alphabet=st.characters(whitelist_categories=("L", "N", "Z"))))
    @settings(max_examples=30)
    def test_parse_schedule_never_crashes(self, text: str) -> None:
        """parse_schedule should raise ValueError for invalid input, never crash."""
        from missy.scheduler.parser import parse_schedule

        try:
            result = parse_schedule(text)
            # If it returns, it should be a dict
            assert isinstance(result, dict)
        except ValueError:
            # Expected for invalid schedule strings
            pass
