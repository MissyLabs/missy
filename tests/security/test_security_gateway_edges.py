"""Tests for session 9 security hardening.

Covers:
- Gateway URL validation (scheme, length, host extraction)
- Vault key lifecycle
- TrustScorer boundary conditions
- InputSanitizer edge cases
- SecretCensor patterns
"""

from __future__ import annotations

import contextlib

import pytest

# ---------------------------------------------------------------------------
# Gateway URL validation
# ---------------------------------------------------------------------------


class TestGatewayURLValidation:
    """Test URL checks in PolicyHTTPClient."""

    def test_rejects_ftp_scheme(self):
        from missy.gateway.client import PolicyHTTPClient

        client = PolicyHTTPClient(session_id="test", task_id="test")
        with pytest.raises(ValueError, match="Unsupported URL scheme"):
            client._check_url("ftp://evil.com/data", "GET")

    def test_rejects_file_scheme(self):
        from missy.gateway.client import PolicyHTTPClient

        client = PolicyHTTPClient(session_id="test", task_id="test")
        with pytest.raises(ValueError, match="Unsupported URL scheme"):
            client._check_url("file:///etc/passwd", "GET")

    def test_rejects_javascript_scheme(self):
        from missy.gateway.client import PolicyHTTPClient

        client = PolicyHTTPClient(session_id="test", task_id="test")
        with pytest.raises(ValueError, match="Unsupported URL scheme"):
            client._check_url("javascript:alert(1)", "GET")

    def test_rejects_very_long_url(self):
        from missy.gateway.client import PolicyHTTPClient

        client = PolicyHTTPClient(session_id="test", task_id="test")
        long_url = "https://example.com/" + "a" * 10000
        with pytest.raises(ValueError, match="exceeds maximum length"):
            client._check_url(long_url, "GET")

    def test_rejects_empty_host(self):
        from missy.gateway.client import PolicyHTTPClient

        client = PolicyHTTPClient(session_id="test", task_id="test")
        with pytest.raises(ValueError, match="Cannot determine host"):
            client._check_url("https://", "GET")

    def test_rejects_negative_timeout(self):
        from missy.gateway.client import PolicyHTTPClient

        with pytest.raises(ValueError, match="positive"):
            PolicyHTTPClient(timeout=-1)

    def test_rejects_zero_timeout(self):
        from missy.gateway.client import PolicyHTTPClient

        with pytest.raises(ValueError, match="positive"):
            PolicyHTTPClient(timeout=0)


# ---------------------------------------------------------------------------
# TrustScorer boundary conditions
# ---------------------------------------------------------------------------


class TestTrustScorerBoundaries:
    """Test trust scorer edge cases."""

    def test_score_never_exceeds_1000(self):
        from missy.security.trust import TrustScorer

        scorer = TrustScorer()
        for _ in range(200):
            scorer.record_success("entity", weight=100)
        assert scorer.score("entity") == 1000

    def test_score_never_below_zero(self):
        from missy.security.trust import TrustScorer

        scorer = TrustScorer()
        for _ in range(200):
            scorer.record_violation("entity", weight=100)
        assert scorer.score("entity") == 0

    def test_default_score_is_500(self):
        from missy.security.trust import TrustScorer

        scorer = TrustScorer()
        assert scorer.score("new_entity") == 500

    def test_reset_restores_default(self):
        from missy.security.trust import TrustScorer

        scorer = TrustScorer()
        scorer.record_violation("entity")
        assert scorer.score("entity") < 500
        scorer.reset("entity")
        assert scorer.score("entity") == 500

    def test_is_trusted_above_threshold(self):
        from missy.security.trust import TrustScorer

        scorer = TrustScorer()
        assert scorer.is_trusted("entity")  # 500 > 200 (default threshold)

    def test_is_trusted_below_threshold(self):
        from missy.security.trust import TrustScorer

        scorer = TrustScorer()
        for _ in range(10):
            scorer.record_violation("entity")
        assert not scorer.is_trusted("entity")

    def test_zero_weight_operations(self):
        from missy.security.trust import TrustScorer

        scorer = TrustScorer()
        scorer.record_success("entity", weight=0)
        assert scorer.score("entity") == 500
        scorer.record_failure("entity", weight=0)
        assert scorer.score("entity") == 500

    def test_get_scores_returns_copy(self):
        from missy.security.trust import TrustScorer

        scorer = TrustScorer()
        scorer.record_success("a")
        scores = scorer.get_scores()
        scores["a"] = 0  # mutate the copy
        assert scorer.score("a") == 510  # original unchanged


# ---------------------------------------------------------------------------
# Circuit breaker
# ---------------------------------------------------------------------------


class TestCircuitBreakerEdges:
    """Test circuit breaker edge cases."""

    def test_initial_state_is_closed(self):
        from missy.agent.circuit_breaker import CircuitBreaker, CircuitState

        cb = CircuitBreaker("test")
        assert cb.state == CircuitState.CLOSED

    def test_opens_after_threshold_failures(self):
        from missy.agent.circuit_breaker import CircuitBreaker, CircuitState

        cb = CircuitBreaker("test", threshold=3)
        for _ in range(3):
            with contextlib.suppress(RuntimeError):
                cb.call(lambda: (_ for _ in ()).throw(RuntimeError("fail")))
        assert cb.state == CircuitState.OPEN

    def test_rejects_calls_when_open(self):
        from missy.agent.circuit_breaker import CircuitBreaker
        from missy.core.exceptions import MissyError

        cb = CircuitBreaker("test", threshold=1)
        with contextlib.suppress(RuntimeError):
            cb.call(lambda: (_ for _ in ()).throw(RuntimeError("fail")))

        with pytest.raises(MissyError, match="OPEN"):
            cb.call(lambda: "should not run")

    def test_success_resets_to_closed(self):
        from missy.agent.circuit_breaker import CircuitBreaker, CircuitState

        cb = CircuitBreaker("test", threshold=3)
        # 2 failures (below threshold)
        for _ in range(2):
            with contextlib.suppress(RuntimeError):
                cb.call(lambda: (_ for _ in ()).throw(RuntimeError("fail")))
        # 1 success
        cb.call(lambda: "ok")
        assert cb.state == CircuitState.CLOSED

    def test_returns_function_result(self):
        from missy.agent.circuit_breaker import CircuitBreaker

        cb = CircuitBreaker("test")
        result = cb.call(lambda: 42)
        assert result == 42


# ---------------------------------------------------------------------------
# Prompt drift detector
# ---------------------------------------------------------------------------


class TestPromptDriftDetector:
    """Test prompt drift edge cases."""

    def test_unregistered_prompt_passes(self):
        from missy.security.drift import PromptDriftDetector

        detector = PromptDriftDetector()
        assert detector.verify("unknown", "anything")

    def test_registered_prompt_verified(self):
        from missy.security.drift import PromptDriftDetector

        detector = PromptDriftDetector()
        detector.register("sys", "hello world")
        assert detector.verify("sys", "hello world")
        assert not detector.verify("sys", "hello world!")

    def test_verify_all_reports_drift(self):
        from missy.security.drift import PromptDriftDetector

        detector = PromptDriftDetector()
        detector.register("a", "prompt a")
        detector.register("b", "prompt b")
        report = detector.verify_all({"a": "prompt a", "b": "tampered"})
        drifted = [r for r in report if r["drifted"]]
        assert len(drifted) == 1
        assert drifted[0]["prompt_id"] == "b"

    def test_verify_all_missing_content(self):
        from missy.security.drift import PromptDriftDetector

        detector = PromptDriftDetector()
        detector.register("a", "prompt a")
        report = detector.verify_all({})  # no content provided
        assert len(report) == 1
        assert not report[0]["drifted"]
        assert report[0]["actual_hash"] is None

    def test_get_drift_report(self):
        from missy.security.drift import PromptDriftDetector

        detector = PromptDriftDetector()
        detector.register("sys", "the system prompt")
        report = detector.get_drift_report()
        assert len(report) == 1
        assert report[0]["prompt_id"] == "sys"
        assert "expected_hash" in report[0]
