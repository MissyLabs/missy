"""Session 15: comprehensive tests for RateLimiter and RestPolicy.

Covers:
  missy/providers/rate_limiter.py  – RateLimiter, RateLimitExceeded
  missy/policy/rest_policy.py      – RestPolicy, RestRule

Strategy:
  - Time is mocked via unittest.mock.patch to avoid real sleeps and give
    deterministic control over the token-bucket refill math.
  - Thread-safety tests use real threads but with carefully sized buckets.
  - RestPolicy tests exercise the public check() and from_config() APIs
    against a wide matrix of host/method/path combinations.
"""

from __future__ import annotations

import contextlib
import threading
import time
from unittest.mock import patch

import pytest

from missy.policy.rest_policy import RestPolicy, RestRule
from missy.providers.rate_limiter import RateLimiter, RateLimitExceeded

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_limiter(
    rpm: int = 10,
    tpm: int = 0,
    max_wait: float = 0.0,
) -> RateLimiter:
    """Convenience factory with safe defaults for unit tests (no blocking)."""
    return RateLimiter(
        requests_per_minute=rpm,
        tokens_per_minute=tpm,
        max_wait_seconds=max_wait,
    )


# ===========================================================================
# RateLimitExceeded exception
# ===========================================================================


class TestRateLimitExceededException:
    """Public API of the exception class itself."""

    def test_message_contains_wait_seconds(self) -> None:
        exc = RateLimitExceeded(wait_seconds=4.5)
        assert "4.5" in str(exc)

    def test_wait_seconds_attribute_stored(self) -> None:
        exc = RateLimitExceeded(wait_seconds=12.34)
        assert exc.wait_seconds == pytest.approx(12.34)

    def test_zero_wait_seconds(self) -> None:
        exc = RateLimitExceeded(wait_seconds=0.0)
        assert exc.wait_seconds == 0.0

    def test_is_exception_subclass(self) -> None:
        assert issubclass(RateLimitExceeded, Exception)

    def test_can_be_raised_and_caught(self) -> None:
        with pytest.raises(RateLimitExceeded) as exc_info:
            raise RateLimitExceeded(wait_seconds=1.0)
        assert exc_info.value.wait_seconds == 1.0


# ===========================================================================
# RateLimiter – constructor
# ===========================================================================


class TestRateLimiterConstructor:
    """Constructor stores parameters and initialises buckets correctly."""

    def test_defaults_stored(self) -> None:
        rl = RateLimiter()
        assert rl._rpm == 60
        assert rl._tpm == 100_000
        assert rl._max_wait == 30.0

    def test_custom_rpm_stored(self) -> None:
        rl = RateLimiter(requests_per_minute=120)
        assert rl._rpm == 120

    def test_custom_tpm_stored(self) -> None:
        rl = RateLimiter(tokens_per_minute=50_000)
        assert rl._tpm == 50_000

    def test_custom_max_wait_stored(self) -> None:
        rl = RateLimiter(max_wait_seconds=5.0)
        assert rl._max_wait == 5.0

    def test_rpm_zero_bucket_starts_at_zero(self) -> None:
        """With rpm=0, the request bucket initialises to 0.0 (no limit tracked)."""
        rl = RateLimiter(requests_per_minute=0)
        assert rl._req_tokens == 0.0

    def test_tpm_zero_bucket_starts_at_zero(self) -> None:
        rl = RateLimiter(tokens_per_minute=0)
        assert rl._tok_tokens == 0.0

    def test_rpm_positive_bucket_starts_full(self) -> None:
        rl = RateLimiter(requests_per_minute=30)
        assert rl._req_tokens == pytest.approx(30.0)

    def test_tpm_positive_bucket_starts_full(self) -> None:
        rl = RateLimiter(tokens_per_minute=9000)
        assert rl._tok_tokens == pytest.approx(9000.0)

    def test_lock_is_thread_lock(self) -> None:
        rl = RateLimiter()
        assert isinstance(rl._lock, type(threading.Lock()))

    def test_very_high_rpm_accepted(self) -> None:
        rl = RateLimiter(requests_per_minute=1_000_000)
        assert rl._rpm == 1_000_000

    def test_rpm_one_single_token_in_bucket(self) -> None:
        rl = RateLimiter(requests_per_minute=1)
        assert rl._req_tokens == pytest.approx(1.0)


# ===========================================================================
# RateLimiter – acquire with mocked time
# ===========================================================================


class TestAcquireWithMockedTime:
    """Use patch('time.monotonic') and patch('time.sleep') for determinism."""

    def test_acquire_unlimited_returns_immediately(self) -> None:
        """Both rpm=0 and tpm=0: acquire is a no-op regardless of tokens arg."""
        rl = RateLimiter(requests_per_minute=0, tokens_per_minute=0)
        rl.acquire()
        rl.acquire(tokens=9_999_999)

    def test_acquire_deducts_request_token(self) -> None:
        rl = _make_limiter(rpm=10)
        before = rl._req_tokens
        rl.acquire()
        assert rl._req_tokens == pytest.approx(before - 1.0)

    def test_acquire_deducts_token_budget(self) -> None:
        rl = RateLimiter(requests_per_minute=1000, tokens_per_minute=1000, max_wait_seconds=0.0)
        rl.acquire(tokens=300)
        assert rl._tok_tokens == pytest.approx(700.0)

    def test_acquire_tokens_zero_does_not_touch_token_bucket(self) -> None:
        rl = RateLimiter(requests_per_minute=1000, tokens_per_minute=500, max_wait_seconds=0.0)
        before = rl._tok_tokens
        rl.acquire(tokens=0)
        assert rl._tok_tokens == pytest.approx(before)

    def test_acquire_raises_rate_limit_exceeded_on_empty_bucket(self) -> None:
        rl = RateLimiter(requests_per_minute=1, max_wait_seconds=0.0)
        rl.acquire()  # drain
        with pytest.raises(RateLimitExceeded):
            rl.acquire()

    def test_acquire_raises_when_token_budget_exhausted(self) -> None:
        rl = RateLimiter(requests_per_minute=100, tokens_per_minute=100, max_wait_seconds=0.0)
        rl.acquire(tokens=100)
        with pytest.raises(RateLimitExceeded):
            rl.acquire(tokens=1)

    def test_acquire_raises_when_ask_exceeds_full_token_bucket(self) -> None:
        """Asking for more tokens than the full bucket should raise immediately."""
        rl = RateLimiter(
            requests_per_minute=1000,
            tokens_per_minute=50,
            max_wait_seconds=0.0,
        )
        with pytest.raises(RateLimitExceeded):
            rl.acquire(tokens=51)

    def test_exact_token_budget_fills_bucket(self) -> None:
        """Consuming exactly the bucket capacity should succeed."""
        rl = RateLimiter(requests_per_minute=1000, tokens_per_minute=200, max_wait_seconds=0.0)
        rl.acquire(tokens=200)
        assert rl._tok_tokens == pytest.approx(0.0)

    def test_acquire_sleep_called_when_waiting(self) -> None:
        """When the bucket is empty, time.sleep should be invoked before raising."""
        rl = RateLimiter(requests_per_minute=1, max_wait_seconds=0.05)
        rl.acquire()  # drain bucket

        sleep_calls: list[float] = []
        real_sleep = time.sleep

        def capture_sleep(t: float) -> None:
            sleep_calls.append(t)
            real_sleep(t)

        with patch("missy.providers.rate_limiter.time.sleep", side_effect=capture_sleep), pytest.raises(RateLimitExceeded):
            rl.acquire()

        assert len(sleep_calls) >= 1, "Expected at least one sleep call while waiting"

    def test_acquire_refills_via_elapsed_time(self) -> None:
        """After draining the bucket, simulating elapsed time allows a new acquire."""
        rpm = 60
        rl = RateLimiter(requests_per_minute=rpm, tokens_per_minute=0, max_wait_seconds=0.0)
        # Drain completely
        for _ in range(rpm):
            rl.acquire()
        assert rl._req_tokens < 1.0

        # Simulate 1 second of elapsed time (= 1.0 token at 60 rpm)
        with rl._lock:
            rl._req_last_refill = time.monotonic() - 1.0

        # Should now succeed since 1 token refilled
        rl.acquire()


# ===========================================================================
# RateLimiter – capacity properties
# ===========================================================================


class TestCapacityProperties:
    """request_capacity and token_capacity reflect internal state after refill."""

    def test_request_capacity_unlimited_is_inf(self) -> None:
        rl = RateLimiter(requests_per_minute=0, tokens_per_minute=0)
        assert rl.request_capacity == float("inf")

    def test_token_capacity_unlimited_is_inf(self) -> None:
        rl = RateLimiter(requests_per_minute=0, tokens_per_minute=0)
        assert rl.token_capacity == float("inf")

    def test_request_capacity_after_full_drain(self) -> None:
        rl = _make_limiter(rpm=5)
        for _ in range(5):
            rl.acquire()
        assert rl.request_capacity < 1.0

    def test_token_capacity_after_partial_spend(self) -> None:
        rl = RateLimiter(requests_per_minute=1000, tokens_per_minute=400, max_wait_seconds=0.0)
        rl.acquire(tokens=100)
        assert rl.token_capacity == pytest.approx(300.0, abs=1.0)

    def test_request_capacity_never_exceeds_rpm(self) -> None:
        """Even after a long idle period, the bucket must not overflow."""
        rpm = 30
        rl = RateLimiter(requests_per_minute=rpm, tokens_per_minute=0, max_wait_seconds=0.0)
        # Back-date the refill timestamp by a very long time
        with rl._lock:
            rl._req_last_refill -= 10_000.0
        assert rl.request_capacity <= float(rpm) + 0.01

    def test_token_capacity_never_exceeds_tpm(self) -> None:
        tpm = 500
        rl = RateLimiter(requests_per_minute=1000, tokens_per_minute=tpm, max_wait_seconds=0.0)
        with rl._lock:
            rl._tok_last_refill -= 10_000.0
        assert rl.token_capacity <= float(tpm) + 0.01

    def test_request_capacity_increases_over_time(self) -> None:
        """After draining, capacity should grow when time passes."""
        rpm = 600  # 10 req/sec
        rl = _make_limiter(rpm=rpm)
        for _ in range(rpm):
            rl.acquire()
        cap_before = rl.request_capacity
        time.sleep(0.05)  # ~0.5 tokens at 10/sec
        cap_after = rl.request_capacity
        assert cap_after > cap_before

    def test_token_capacity_increases_over_time(self) -> None:
        tpm = 6000  # 100 tokens/sec
        rl = RateLimiter(requests_per_minute=10_000, tokens_per_minute=tpm, max_wait_seconds=0.0)
        rl.acquire(tokens=tpm)
        cap_before = rl.token_capacity
        time.sleep(0.05)  # ~5 tokens
        cap_after = rl.token_capacity
        assert cap_after > cap_before


# ===========================================================================
# RateLimiter – record_usage
# ===========================================================================


class TestRecordUsage:
    """record_usage adjusts the token bucket to reflect actual consumption."""

    def test_record_usage_deducts_sum_from_token_bucket(self) -> None:
        rl = RateLimiter(requests_per_minute=1000, tokens_per_minute=1000, max_wait_seconds=0.0)
        rl.acquire(tokens=200)  # _tok_tokens = 800
        rl.record_usage(prompt_tokens=50, completion_tokens=50)
        # 800 - 100 = 700
        assert rl._tok_tokens == pytest.approx(700.0)

    def test_record_usage_clamps_at_zero(self) -> None:
        """If actual usage pushes the bucket below zero, it must clamp to 0."""
        rl = RateLimiter(requests_per_minute=1000, tokens_per_minute=100, max_wait_seconds=0.0)
        rl.acquire(tokens=100)  # _tok_tokens = 0
        # Now record more usage than the bucket holds
        rl.record_usage(prompt_tokens=500, completion_tokens=500)
        assert rl._tok_tokens == 0.0

    def test_record_usage_no_op_when_tpm_is_zero(self) -> None:
        rl = RateLimiter(requests_per_minute=10, tokens_per_minute=0)
        before = rl._tok_tokens
        rl.record_usage(prompt_tokens=1000, completion_tokens=1000)
        assert rl._tok_tokens == before

    def test_record_usage_zero_values_is_no_op(self) -> None:
        rl = RateLimiter(requests_per_minute=10, tokens_per_minute=500)
        rl.acquire(tokens=100)
        before = rl._tok_tokens
        rl.record_usage(prompt_tokens=0, completion_tokens=0)
        assert rl._tok_tokens == pytest.approx(before)

    def test_record_usage_does_not_touch_request_bucket(self) -> None:
        rl = RateLimiter(requests_per_minute=100, tokens_per_minute=1000)
        rl.acquire(tokens=100)
        req_before = rl._req_tokens
        rl.record_usage(prompt_tokens=100, completion_tokens=100)
        assert rl._req_tokens == pytest.approx(req_before)

    def test_record_usage_large_values_clamps_not_negative(self) -> None:
        rl = RateLimiter(requests_per_minute=100, tokens_per_minute=1000)
        rl.record_usage(prompt_tokens=10_000_000, completion_tokens=10_000_000)
        assert rl._tok_tokens >= 0.0

    def test_record_usage_prompt_only(self) -> None:
        rl = RateLimiter(requests_per_minute=1000, tokens_per_minute=1000, max_wait_seconds=0.0)
        rl.acquire(tokens=0)
        rl.record_usage(prompt_tokens=300, completion_tokens=0)
        assert rl._tok_tokens == pytest.approx(700.0)

    def test_record_usage_completion_only(self) -> None:
        rl = RateLimiter(requests_per_minute=1000, tokens_per_minute=1000, max_wait_seconds=0.0)
        rl.acquire(tokens=0)
        rl.record_usage(prompt_tokens=0, completion_tokens=400)
        assert rl._tok_tokens == pytest.approx(600.0)


# ===========================================================================
# RateLimiter – on_rate_limit_response
# ===========================================================================


class TestOnRateLimitResponse:
    """429-response handler drains buckets and optionally sleeps."""

    def test_drains_request_bucket_to_zero(self) -> None:
        rl = _make_limiter(rpm=60)
        rl.on_rate_limit_response(retry_after=0.0)
        assert rl._req_tokens == 0.0

    def test_drains_token_bucket_to_zero_when_tpm_active(self) -> None:
        rl = RateLimiter(requests_per_minute=60, tokens_per_minute=10_000)
        rl.on_rate_limit_response(retry_after=0.0)
        assert rl._tok_tokens == 0.0

    def test_tpm_zero_token_bucket_unchanged_after_drain(self) -> None:
        """With tpm=0, on_rate_limit_response does not touch _tok_tokens."""
        rl = RateLimiter(requests_per_minute=60, tokens_per_minute=0)
        before = rl._tok_tokens
        rl.on_rate_limit_response(retry_after=0.0)
        assert rl._tok_tokens == before

    def test_retry_after_sleep_called_with_min_of_retry_and_max_wait(self) -> None:
        """Sleep duration is min(retry_after, max_wait_seconds)."""
        rl = RateLimiter(requests_per_minute=60, max_wait_seconds=1.0)
        with patch("missy.providers.rate_limiter.time.sleep") as mock_sleep:
            rl.on_rate_limit_response(retry_after=5.0)
        mock_sleep.assert_called_once_with(1.0)  # clamped to max_wait

    def test_retry_after_zero_no_sleep(self) -> None:
        """retry_after=0 must not call time.sleep."""
        rl = _make_limiter(rpm=10)
        with patch("missy.providers.rate_limiter.time.sleep") as mock_sleep:
            rl.on_rate_limit_response(retry_after=0.0)
        mock_sleep.assert_not_called()

    def test_retry_after_small_sleeps_exact_amount(self) -> None:
        """retry_after < max_wait_seconds: sleeps for exactly retry_after."""
        rl = RateLimiter(requests_per_minute=10, max_wait_seconds=30.0)
        with patch("missy.providers.rate_limiter.time.sleep") as mock_sleep:
            rl.on_rate_limit_response(retry_after=2.0)
        mock_sleep.assert_called_once_with(2.0)

    def test_buckets_remain_at_zero_after_drain_no_immediate_refill(self) -> None:
        """Immediately after on_rate_limit_response, both buckets read 0."""
        rl = RateLimiter(requests_per_minute=60, tokens_per_minute=1000)
        rl.on_rate_limit_response(retry_after=0.0)
        # Without sleeping, buckets haven't refilled yet
        assert rl._req_tokens == 0.0
        assert rl._tok_tokens == 0.0


# ===========================================================================
# RateLimiter – thread safety
# ===========================================================================


class TestRateLimiterThreadSafety:
    """Concurrent acquire calls must not corrupt internal state."""

    def test_concurrent_acquires_no_exceptions_other_than_rate_limit(self) -> None:
        """Only RateLimitExceeded should escape; no AttributeError or similar."""
        rl = RateLimiter(requests_per_minute=50, max_wait_seconds=0.0)
        unexpected: list[Exception] = []
        lock = threading.Lock()

        def worker() -> None:
            for _ in range(5):
                try:
                    rl.acquire()
                except RateLimitExceeded:
                    pass
                except Exception as exc:
                    with lock:
                        unexpected.append(exc)

        threads = [threading.Thread(target=worker, daemon=True) for _ in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=5)

        assert unexpected == [], f"Unexpected exceptions under concurrency: {unexpected}"

    def test_request_tokens_never_go_negative_under_concurrency(self) -> None:
        """The request bucket must never drop below 0.0 even under heavy contention."""
        rl = RateLimiter(requests_per_minute=20, max_wait_seconds=0.0)
        barrier = threading.Barrier(30)

        def worker() -> None:
            barrier.wait()
            with contextlib.suppress(RateLimitExceeded):
                rl.acquire()

        threads = [threading.Thread(target=worker, daemon=True) for _ in range(30)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=5)

        assert rl._req_tokens >= 0.0

    def test_token_bucket_never_goes_negative_under_concurrency(self) -> None:
        rl = RateLimiter(requests_per_minute=10_000, tokens_per_minute=200, max_wait_seconds=0.0)
        barrier = threading.Barrier(20)

        def worker() -> None:
            barrier.wait()
            with contextlib.suppress(RateLimitExceeded):
                rl.acquire(tokens=20)

        threads = [threading.Thread(target=worker, daemon=True) for _ in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=5)

        assert rl._tok_tokens >= 0.0

    def test_record_usage_concurrent_with_acquire(self) -> None:
        """record_usage and acquire can run simultaneously without crashing."""
        rl = RateLimiter(requests_per_minute=1000, tokens_per_minute=100_000, max_wait_seconds=1.0)
        errors: list[Exception] = []
        lock = threading.Lock()

        def acquirer() -> None:
            for _ in range(50):
                try:
                    rl.acquire(tokens=10)
                except RateLimitExceeded:
                    pass
                except Exception as exc:
                    with lock:
                        errors.append(exc)

        def recorder() -> None:
            for _ in range(50):
                try:
                    rl.record_usage(prompt_tokens=5, completion_tokens=5)
                except Exception as exc:
                    with lock:
                        errors.append(exc)

        threads = [threading.Thread(target=acquirer, daemon=True) for _ in range(5)]
        threads += [threading.Thread(target=recorder, daemon=True) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)

        assert errors == []


# ===========================================================================
# RestRule dataclass
# ===========================================================================


class TestRestRule:
    """RestRule is a frozen dataclass; test construction and immutability."""

    def test_fields_stored(self) -> None:
        rule = RestRule(host="api.example.com", method="GET", path="/foo/**", action="allow")
        assert rule.host == "api.example.com"
        assert rule.method == "GET"
        assert rule.path == "/foo/**"
        assert rule.action == "allow"

    def test_frozen_cannot_mutate(self) -> None:
        rule = RestRule(host="h", method="GET", path="/", action="allow")
        with pytest.raises((AttributeError, TypeError)):
            rule.host = "other"  # type: ignore[misc]

    def test_equality_by_value(self) -> None:
        r1 = RestRule(host="a", method="GET", path="/", action="allow")
        r2 = RestRule(host="a", method="GET", path="/", action="allow")
        assert r1 == r2

    def test_inequality_on_action(self) -> None:
        r1 = RestRule(host="a", method="GET", path="/", action="allow")
        r2 = RestRule(host="a", method="GET", path="/", action="deny")
        assert r1 != r2

    def test_hashable_usable_in_set(self) -> None:
        rule = RestRule(host="h", method="POST", path="/data", action="allow")
        s = {rule}
        assert rule in s


# ===========================================================================
# RestPolicy – construction
# ===========================================================================


class TestRestPolicyConstruction:
    """Construction via __init__ and from_config."""

    def test_empty_init_rules(self) -> None:
        policy = RestPolicy()
        assert policy._rules == []

    def test_init_with_none_rules(self) -> None:
        policy = RestPolicy(rules=None)
        assert policy._rules == []

    def test_init_with_rule_list(self) -> None:
        rules = [RestRule(host="h", method="GET", path="/", action="allow")]
        policy = RestPolicy(rules=rules)
        assert len(policy._rules) == 1

    def test_init_does_not_share_list_reference(self) -> None:
        """Mutating the original list must not affect the policy's rules."""
        rules = [RestRule(host="h", method="GET", path="/", action="allow")]
        policy = RestPolicy(rules=rules)
        rules.append(RestRule(host="h2", method="POST", path="/x", action="deny"))
        assert len(policy._rules) == 1

    def test_from_config_empty_list(self) -> None:
        policy = RestPolicy.from_config([])
        assert policy._rules == []

    def test_from_config_lowercases_host(self) -> None:
        policy = RestPolicy.from_config([
            {"host": "API.GitHub.COM", "method": "GET", "path": "/", "action": "allow"},
        ])
        assert policy._rules[0].host == "api.github.com"

    def test_from_config_uppercases_method(self) -> None:
        policy = RestPolicy.from_config([
            {"host": "h", "method": "delete", "path": "/", "action": "deny"},
        ])
        assert policy._rules[0].method == "DELETE"

    def test_from_config_lowercases_action(self) -> None:
        policy = RestPolicy.from_config([
            {"host": "h", "method": "GET", "path": "/", "action": "ALLOW"},
        ])
        assert policy._rules[0].action == "allow"

    def test_from_config_default_method_is_star(self) -> None:
        """Missing 'method' key defaults to '*'."""
        policy = RestPolicy.from_config([
            {"host": "h", "path": "/", "action": "allow"},
        ])
        assert policy._rules[0].method == "*"

    def test_from_config_default_path_is_glob_all(self) -> None:
        """Missing 'path' key defaults to '/**'."""
        policy = RestPolicy.from_config([
            {"host": "h", "method": "GET", "action": "allow"},
        ])
        assert policy._rules[0].path == "/**"

    def test_from_config_default_action_is_deny(self) -> None:
        """Missing 'action' key defaults to 'deny'."""
        policy = RestPolicy.from_config([
            {"host": "h", "method": "GET", "path": "/"},
        ])
        assert policy._rules[0].action == "deny"

    def test_from_config_default_host_is_empty_string(self) -> None:
        """Missing 'host' key defaults to '' (empty string)."""
        policy = RestPolicy.from_config([
            {"method": "GET", "path": "/", "action": "allow"},
        ])
        assert policy._rules[0].host == ""

    def test_from_config_multiple_rules_order_preserved(self) -> None:
        raw = [
            {"host": "a", "method": "GET", "path": "/1", "action": "allow"},
            {"host": "b", "method": "POST", "path": "/2", "action": "deny"},
            {"host": "c", "method": "DELETE", "path": "/3", "action": "allow"},
        ]
        policy = RestPolicy.from_config(raw)
        assert policy._rules[0].host == "a"
        assert policy._rules[1].host == "b"
        assert policy._rules[2].host == "c"


# ===========================================================================
# RestPolicy – check() return values
# ===========================================================================


class TestRestPolicyCheck:
    """check() returns 'allow', 'deny', or None."""

    def test_returns_none_when_no_rules(self) -> None:
        policy = RestPolicy()
        assert policy.check("api.github.com", "GET", "/repos") is None

    def test_returns_none_when_host_not_matched(self) -> None:
        policy = RestPolicy.from_config([
            {"host": "api.github.com", "method": "GET", "path": "/**", "action": "allow"},
        ])
        assert policy.check("api.example.com", "GET", "/repos") is None

    def test_returns_none_when_method_not_matched(self) -> None:
        policy = RestPolicy.from_config([
            {"host": "api.github.com", "method": "GET", "path": "/**", "action": "allow"},
        ])
        assert policy.check("api.github.com", "POST", "/repos") is None

    def test_returns_none_when_path_not_matched(self) -> None:
        policy = RestPolicy.from_config([
            {"host": "api.github.com", "method": "GET", "path": "/repos/**", "action": "allow"},
        ])
        assert policy.check("api.github.com", "GET", "/users/foo") is None

    def test_returns_allow_for_matching_rule(self) -> None:
        policy = RestPolicy.from_config([
            {"host": "api.github.com", "method": "GET", "path": "/repos/**", "action": "allow"},
        ])
        assert policy.check("api.github.com", "GET", "/repos/foo/bar") == "allow"

    def test_returns_deny_for_matching_rule(self) -> None:
        policy = RestPolicy.from_config([
            {"host": "api.github.com", "method": "DELETE", "path": "/**", "action": "deny"},
        ])
        assert policy.check("api.github.com", "DELETE", "/repos/foo") == "deny"

    def test_first_match_wins_allow_before_deny(self) -> None:
        policy = RestPolicy.from_config([
            {"host": "h", "method": "GET", "path": "/admin/**", "action": "allow"},
            {"host": "h", "method": "GET", "path": "/admin/**", "action": "deny"},
        ])
        assert policy.check("h", "GET", "/admin/secret") == "allow"

    def test_first_match_wins_deny_before_allow(self) -> None:
        policy = RestPolicy.from_config([
            {"host": "h", "method": "GET", "path": "/**", "action": "deny"},
            {"host": "h", "method": "GET", "path": "/public/**", "action": "allow"},
        ])
        assert policy.check("h", "GET", "/public/page") == "deny"

    def test_later_rule_reached_when_earlier_rule_host_differs(self) -> None:
        policy = RestPolicy.from_config([
            {"host": "other.com", "method": "GET", "path": "/**", "action": "deny"},
            {"host": "api.com", "method": "GET", "path": "/**", "action": "allow"},
        ])
        assert policy.check("api.com", "GET", "/data") == "allow"

    def test_later_rule_reached_when_earlier_rule_method_differs(self) -> None:
        policy = RestPolicy.from_config([
            {"host": "api.com", "method": "POST", "path": "/**", "action": "deny"},
            {"host": "api.com", "method": "GET", "path": "/**", "action": "allow"},
        ])
        assert policy.check("api.com", "GET", "/data") == "allow"


# ===========================================================================
# RestPolicy – host matching
# ===========================================================================


class TestRestPolicyHostMatching:
    """Host comparison is case-insensitive; only exact matches qualify."""

    def test_check_host_case_insensitive_upper_in_check(self) -> None:
        policy = RestPolicy.from_config([
            {"host": "api.github.com", "method": "GET", "path": "/**", "action": "allow"},
        ])
        assert policy.check("API.GITHUB.COM", "GET", "/foo") == "allow"

    def test_check_host_case_insensitive_mixed(self) -> None:
        policy = RestPolicy.from_config([
            {"host": "Api.GitHub.Com", "method": "GET", "path": "/**", "action": "allow"},
        ])
        assert policy.check("api.github.com", "GET", "/foo") == "allow"

    def test_subdomain_does_not_match_parent(self) -> None:
        """rules for api.github.com must not match sub.api.github.com."""
        policy = RestPolicy.from_config([
            {"host": "api.github.com", "method": "GET", "path": "/**", "action": "deny"},
        ])
        assert policy.check("sub.api.github.com", "GET", "/foo") is None

    def test_partial_host_suffix_does_not_match(self) -> None:
        policy = RestPolicy.from_config([
            {"host": "github.com", "method": "GET", "path": "/**", "action": "deny"},
        ])
        assert policy.check("api.github.com", "GET", "/foo") is None

    def test_empty_host_in_rule_only_matches_empty_host_in_check(self) -> None:
        """A rule with host='' only matches check calls where host is also ''."""
        policy = RestPolicy.from_config([
            {"host": "", "method": "GET", "path": "/**", "action": "allow"},
        ])
        assert policy.check("", "GET", "/foo") == "allow"
        assert policy.check("api.com", "GET", "/foo") is None


# ===========================================================================
# RestPolicy – method matching
# ===========================================================================


class TestRestPolicyMethodMatching:
    """Method matching is case-insensitive; '*' matches any method."""

    def test_method_wildcard_matches_get(self) -> None:
        policy = RestPolicy.from_config([
            {"host": "h", "method": "*", "path": "/**", "action": "allow"},
        ])
        assert policy.check("h", "GET", "/x") == "allow"

    def test_method_wildcard_matches_post(self) -> None:
        policy = RestPolicy.from_config([
            {"host": "h", "method": "*", "path": "/**", "action": "allow"},
        ])
        assert policy.check("h", "POST", "/x") == "allow"

    def test_method_wildcard_matches_delete(self) -> None:
        policy = RestPolicy.from_config([
            {"host": "h", "method": "*", "path": "/**", "action": "deny"},
        ])
        assert policy.check("h", "DELETE", "/x") == "deny"

    def test_method_wildcard_matches_patch(self) -> None:
        policy = RestPolicy.from_config([
            {"host": "h", "method": "*", "path": "/**", "action": "allow"},
        ])
        assert policy.check("h", "PATCH", "/x") == "allow"

    def test_method_wildcard_matches_put(self) -> None:
        policy = RestPolicy.from_config([
            {"host": "h", "method": "*", "path": "/**", "action": "allow"},
        ])
        assert policy.check("h", "PUT", "/x") == "allow"

    def test_method_wildcard_matches_head(self) -> None:
        policy = RestPolicy.from_config([
            {"host": "h", "method": "*", "path": "/**", "action": "allow"},
        ])
        assert policy.check("h", "HEAD", "/x") == "allow"

    def test_specific_method_does_not_match_other_methods(self) -> None:
        policy = RestPolicy.from_config([
            {"host": "h", "method": "GET", "path": "/**", "action": "allow"},
        ])
        for m in ("POST", "PUT", "DELETE", "PATCH", "OPTIONS"):
            assert policy.check("h", m, "/x") is None, f"Method {m!r} should not match"

    def test_method_case_insensitive_lower_in_rule_upper_in_check(self) -> None:
        policy = RestPolicy.from_config([
            {"host": "h", "method": "get", "path": "/**", "action": "allow"},
        ])
        assert policy.check("h", "GET", "/x") == "allow"

    def test_method_case_insensitive_upper_in_rule_lower_in_check(self) -> None:
        policy = RestPolicy.from_config([
            {"host": "h", "method": "DELETE", "path": "/**", "action": "deny"},
        ])
        assert policy.check("h", "delete", "/x") == "deny"


# ===========================================================================
# RestPolicy – path glob matching
# ===========================================================================


class TestRestPolicyPathGlob:
    """Path matching uses fnmatch; test various glob patterns."""

    def test_exact_path_match(self) -> None:
        policy = RestPolicy.from_config([
            {"host": "h", "method": "GET", "path": "/health", "action": "allow"},
        ])
        assert policy.check("h", "GET", "/health") == "allow"
        assert policy.check("h", "GET", "/health/check") is None

    def test_double_star_glob_matches_deep_paths(self) -> None:
        policy = RestPolicy.from_config([
            {"host": "h", "method": "GET", "path": "/repos/**", "action": "allow"},
        ])
        assert policy.check("h", "GET", "/repos/owner/repo/issues/42") == "allow"

    def test_double_star_glob_at_root_matches_any_path(self) -> None:
        policy = RestPolicy.from_config([
            {"host": "h", "method": "GET", "path": "/**", "action": "allow"},
        ])
        assert policy.check("h", "GET", "/a/b/c/d/e") == "allow"

    def test_single_star_matches_one_segment(self) -> None:
        """fnmatch '*' does not cross path separators in the file path sense,
        but in fnmatch it matches any characters including '/'.
        Verify the actual behaviour is consistent with fnmatch.fnmatch."""
        import fnmatch

        path = "/repos/owner"
        pattern = "/repos/*"
        expected = fnmatch.fnmatch(path, pattern)
        policy = RestPolicy.from_config([
            {"host": "h", "method": "GET", "path": pattern, "action": "allow"},
        ])
        result = policy.check("h", "GET", path)
        if expected:
            assert result == "allow"
        else:
            assert result is None

    def test_question_mark_glob_matches_single_char(self) -> None:
        """fnmatch '?' matches any single character."""
        policy = RestPolicy.from_config([
            {"host": "h", "method": "GET", "path": "/v?/status", "action": "allow"},
        ])
        assert policy.check("h", "GET", "/v1/status") == "allow"
        assert policy.check("h", "GET", "/v2/status") == "allow"

    def test_character_range_glob(self) -> None:
        """fnmatch '[abc]' matches one of the listed chars."""
        policy = RestPolicy.from_config([
            {"host": "h", "method": "GET", "path": "/api/v[123]/**", "action": "allow"},
        ])
        assert policy.check("h", "GET", "/api/v1/data") == "allow"
        assert policy.check("h", "GET", "/api/v2/data") == "allow"
        assert policy.check("h", "GET", "/api/v4/data") is None

    def test_leading_slash_required_for_match(self) -> None:
        policy = RestPolicy.from_config([
            {"host": "h", "method": "GET", "path": "/repos/**", "action": "allow"},
        ])
        assert policy.check("h", "GET", "repos/foo") is None

    def test_multiple_path_rules_per_host(self) -> None:
        """Different path rules on the same host each match independently."""
        policy = RestPolicy.from_config([
            {"host": "h", "method": "GET", "path": "/public/**", "action": "allow"},
            {"host": "h", "method": "GET", "path": "/private/**", "action": "deny"},
        ])
        assert policy.check("h", "GET", "/public/data") == "allow"
        assert policy.check("h", "GET", "/private/secret") == "deny"
        assert policy.check("h", "GET", "/other/stuff") is None


# ===========================================================================
# RestPolicy – realistic multi-rule scenarios
# ===========================================================================


class TestRestPolicyRealisticScenarios:
    """End-to-end rule sets that mirror actual config usage."""

    @pytest.fixture
    def github_policy(self) -> RestPolicy:
        return RestPolicy.from_config([
            {"host": "api.github.com", "method": "GET", "path": "/repos/**", "action": "allow"},
            {"host": "api.github.com", "method": "POST", "path": "/repos/**", "action": "allow"},
            {"host": "api.github.com", "method": "DELETE", "path": "/**", "action": "deny"},
            {"host": "api.github.com", "method": "GET", "path": "/user", "action": "allow"},
        ])

    def test_github_get_repos_allowed(self, github_policy: RestPolicy) -> None:
        assert github_policy.check("api.github.com", "GET", "/repos/owner/repo") == "allow"

    def test_github_post_repos_allowed(self, github_policy: RestPolicy) -> None:
        assert github_policy.check("api.github.com", "POST", "/repos/owner/repo/issues") == "allow"

    def test_github_delete_denied(self, github_policy: RestPolicy) -> None:
        assert github_policy.check("api.github.com", "DELETE", "/repos/owner/repo") == "deny"

    def test_github_get_user_allowed(self, github_policy: RestPolicy) -> None:
        assert github_policy.check("api.github.com", "GET", "/user") == "allow"

    def test_github_put_has_no_rule_returns_none(self, github_policy: RestPolicy) -> None:
        assert github_policy.check("api.github.com", "PUT", "/repos/owner/repo") is None

    def test_github_unrelated_host_no_match(self, github_policy: RestPolicy) -> None:
        assert github_policy.check("api.example.com", "DELETE", "/anything") is None

    def test_admin_deny_overrides_general_allow(self) -> None:
        """Deny rule for /admin/** placed first should block even when a later
        catch-all allow exists."""
        policy = RestPolicy.from_config([
            {"host": "api.com", "method": "GET", "path": "/admin/**", "action": "deny"},
            {"host": "api.com", "method": "GET", "path": "/**", "action": "allow"},
        ])
        assert policy.check("api.com", "GET", "/admin/users") == "deny"
        assert policy.check("api.com", "GET", "/public/data") == "allow"

    def test_read_only_policy_denies_all_writes(self) -> None:
        """A read-only profile: GET allowed, all other methods denied."""
        policy = RestPolicy.from_config([
            {"host": "api.com", "method": "GET", "path": "/**", "action": "allow"},
            {"host": "api.com", "method": "*", "path": "/**", "action": "deny"},
        ])
        assert policy.check("api.com", "GET", "/data") == "allow"
        for m in ("POST", "PUT", "PATCH", "DELETE"):
            assert policy.check("api.com", m, "/data") == "deny", f"Expected deny for {m}"

    def test_multiple_hosts_isolated(self) -> None:
        """Rules for different hosts are fully isolated from each other."""
        policy = RestPolicy.from_config([
            {"host": "a.com", "method": "GET", "path": "/**", "action": "allow"},
            {"host": "b.com", "method": "GET", "path": "/**", "action": "deny"},
        ])
        assert policy.check("a.com", "GET", "/x") == "allow"
        assert policy.check("b.com", "GET", "/x") == "deny"
        assert policy.check("c.com", "GET", "/x") is None

    def test_direct_rest_rule_objects_bypass_from_config(self) -> None:
        """Constructing via RestRule directly is equivalent to from_config."""
        rules = [
            RestRule(host="api.com", method="GET", path="/data/**", action="allow"),
            RestRule(host="api.com", method="POST", path="/data/**", action="deny"),
        ]
        policy = RestPolicy(rules=rules)
        assert policy.check("api.com", "GET", "/data/items") == "allow"
        assert policy.check("api.com", "POST", "/data/items") == "deny"
        assert policy.check("api.com", "DELETE", "/data/items") is None
