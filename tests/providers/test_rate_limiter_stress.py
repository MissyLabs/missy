"""Stress tests for the provider rate limiter.

These tests exercise concurrency, bucket exhaustion, refill accuracy, burst
behaviour, edge cases, and property-based invariants.  All ``max_wait_seconds``
values are kept short (0.05–0.5 s) so the suite runs in a few seconds.
"""

from __future__ import annotations

import contextlib
import math
import threading
import time
from typing import NamedTuple

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from missy.providers.rate_limiter import RateLimiter, RateLimitExceeded

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class AcquireResult(NamedTuple):
    succeeded: int
    failed: int
    errors: list[Exception]


def _concurrent_acquire(
    limiter: RateLimiter,
    *,
    n_threads: int,
    calls_per_thread: int,
    tokens: int = 0,
) -> AcquireResult:
    """Spawn ``n_threads`` threads each calling acquire ``calls_per_thread`` times.

    Returns counts of successes, RateLimitExceeded failures, and any unexpected
    exceptions.
    """
    succeeded = 0
    failed = 0
    unexpected: list[Exception] = []
    lock = threading.Lock()

    def worker() -> None:
        nonlocal succeeded, failed
        for _ in range(calls_per_thread):
            try:
                limiter.acquire(tokens=tokens)
                with lock:
                    succeeded += 1
            except RateLimitExceeded:
                with lock:
                    failed += 1
            except Exception as exc:  # noqa: BLE001
                with lock:
                    unexpected.append(exc)

    threads = [threading.Thread(target=worker, daemon=True) for _ in range(n_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=30)

    return AcquireResult(succeeded, failed, unexpected)


# ---------------------------------------------------------------------------
# 1. Concurrent acquire stress – RPM enforcement
# ---------------------------------------------------------------------------


class TestConcurrentAcquireStress:
    """Multiple threads racing on acquire(); verify RPM cap is never breached."""

    def test_concurrent_no_unexpected_errors(self) -> None:
        """No exception other than RateLimitExceeded should escape."""
        rl = RateLimiter(requests_per_minute=30, max_wait_seconds=0.1)
        result = _concurrent_acquire(rl, n_threads=10, calls_per_thread=5)
        assert result.errors == [], f"Unexpected exceptions: {result.errors}"

    def test_concurrent_total_throughput_bounded_by_rpm(self) -> None:
        """Successes inside a 1 s window must not exceed RPM + a small margin."""
        rpm = 20
        rl = RateLimiter(requests_per_minute=rpm, max_wait_seconds=0.2)

        start = time.monotonic()
        result = _concurrent_acquire(rl, n_threads=8, calls_per_thread=10)
        elapsed = time.monotonic() - start

        # Within the elapsed seconds the theoretical max is rpm/60 * elapsed + rpm
        # (bucket starts full).  Use a generous 20 % headroom for timing jitter.
        theoretical_max = rpm + (rpm / 60.0) * elapsed
        assert result.succeeded <= math.ceil(theoretical_max * 1.2), (
            f"succeeded={result.succeeded} exceeds theoretical_max={theoretical_max:.1f}"
        )

    def test_concurrent_all_succeed_when_limit_is_generous(self) -> None:
        """When RPM is far higher than demand, every acquire() should succeed."""
        rl = RateLimiter(requests_per_minute=10_000, max_wait_seconds=0.5)
        result = _concurrent_acquire(rl, n_threads=5, calls_per_thread=4)
        assert result.failed == 0
        assert result.succeeded == 20

    def test_concurrent_some_fail_when_rpm_is_tight(self) -> None:
        """When threads greatly outnumber the RPM budget, some must be rejected."""
        rl = RateLimiter(requests_per_minute=5, max_wait_seconds=0.05)
        result = _concurrent_acquire(rl, n_threads=20, calls_per_thread=5)
        # 20*5 = 100 attempts; only a handful should fit in ~5 + small refill tokens.
        assert result.failed > 0, "Expected at least some RateLimitExceeded"
        assert result.errors == []

    def test_concurrent_bucket_never_goes_negative(self) -> None:
        """Token bucket counters must never drop below zero under concurrency."""
        rl = RateLimiter(requests_per_minute=50, tokens_per_minute=5_000, max_wait_seconds=0.1)
        _concurrent_acquire(rl, n_threads=10, calls_per_thread=10, tokens=100)
        assert rl._req_tokens >= 0.0
        assert rl._tok_tokens >= 0.0


# ---------------------------------------------------------------------------
# 2. Token budget exhaustion
# ---------------------------------------------------------------------------


class TestTokenBudgetExhaustion:
    """Draining the token bucket with large per-request token counts."""

    def test_single_large_acquire_exhausts_token_bucket(self) -> None:
        tpm = 1_000
        # max_wait_seconds=0.0 so the refill loop is skipped entirely; the
        # bucket is fully consumed by the first acquire and the second must
        # raise without waiting for any refill to occur.
        rl = RateLimiter(requests_per_minute=1_000, tokens_per_minute=tpm, max_wait_seconds=0.0)
        rl.acquire(tokens=tpm)  # consume entire budget
        with pytest.raises(RateLimitExceeded):
            rl.acquire(tokens=1)

    def test_multiple_acquires_exhaust_token_bucket(self) -> None:
        rl = RateLimiter(requests_per_minute=1_000, tokens_per_minute=300, max_wait_seconds=0.1)
        rl.acquire(tokens=100)
        rl.acquire(tokens=100)
        rl.acquire(tokens=100)
        with pytest.raises(RateLimitExceeded):
            rl.acquire(tokens=100)

    def test_token_exhaustion_does_not_block_rpm_only_acquire(self) -> None:
        """An acquire() with tokens=0 should still succeed even if token bucket is empty."""
        rl = RateLimiter(requests_per_minute=100, tokens_per_minute=100, max_wait_seconds=0.1)
        rl.acquire(tokens=100)  # drain token bucket
        # tokens=0 should bypass token check entirely
        rl.acquire(tokens=0)

    def test_acquire_exactly_at_token_limit(self) -> None:
        """Requesting exactly the remaining tokens should succeed."""
        tpm = 500
        rl = RateLimiter(requests_per_minute=1_000, tokens_per_minute=tpm, max_wait_seconds=0.1)
        rl.acquire(tokens=tpm)  # should succeed – uses the whole bucket
        # next one should fail
        with pytest.raises(RateLimitExceeded):
            rl.acquire(tokens=1)

    def test_token_exceeded_error_carries_wait_seconds(self) -> None:
        tpm = 600  # 10 tokens/sec refill
        rl = RateLimiter(requests_per_minute=1_000, tokens_per_minute=tpm, max_wait_seconds=0.05)
        rl.acquire(tokens=tpm)
        try:
            rl.acquire(tokens=tpm)
        except RateLimitExceeded as exc:
            assert exc.wait_seconds > 0.0
        else:
            pytest.fail("Expected RateLimitExceeded")


# ---------------------------------------------------------------------------
# 3. Refill accuracy
# ---------------------------------------------------------------------------


class TestRefillAccuracy:
    """Partial refill after sleeping should restore a predictable fraction of capacity."""

    def test_request_bucket_partially_refills(self) -> None:
        rpm = 600  # 10 requests/sec → easy to test with short sleeps
        rl = RateLimiter(requests_per_minute=rpm, max_wait_seconds=0.05)

        # Drain completely
        for _ in range(rpm):
            try:
                rl.acquire()
            except RateLimitExceeded:
                break

        before = rl.request_capacity
        sleep_s = 0.2
        time.sleep(sleep_s)
        after = rl.request_capacity

        expected_refill = (rpm / 60.0) * sleep_s
        # Allow ±30 % tolerance for OS scheduling jitter
        assert after >= before + expected_refill * 0.7, (
            f"Expected at least {expected_refill * 0.7:.2f} refill; got {after - before:.2f}"
        )

    def test_token_bucket_partially_refills(self) -> None:
        tpm = 6_000  # 100 tokens/sec
        rl = RateLimiter(requests_per_minute=1_000, tokens_per_minute=tpm, max_wait_seconds=0.05)

        with contextlib.suppress(RateLimitExceeded):
            rl.acquire(tokens=tpm)

        before = rl.token_capacity
        sleep_s = 0.3
        time.sleep(sleep_s)
        after = rl.token_capacity

        expected_refill = (tpm / 60.0) * sleep_s
        assert after >= before + expected_refill * 0.7, (
            f"Expected ~{expected_refill:.0f} token refill; got {after - before:.1f}"
        )

    def test_full_refill_caps_at_maximum(self) -> None:
        """Sleeping much longer than needed should not overfill the bucket."""
        rpm = 10
        rl = RateLimiter(requests_per_minute=rpm, max_wait_seconds=0.1)
        rl.acquire()  # take one token
        time.sleep(0.3)  # wait long enough for multiple tokens to accrue
        cap = rl.request_capacity
        assert cap <= float(rpm) + 0.01, f"Bucket overflowed: {cap} > {rpm}"


# ---------------------------------------------------------------------------
# 4. Burst handling
# ---------------------------------------------------------------------------


class TestBurstHandling:
    """Rapid-fire requests; measure success/failure split."""

    def test_burst_limited_by_initial_bucket(self) -> None:
        """With max_wait=0 the bucket is effectively a hard cap."""
        rpm = 10
        rl = RateLimiter(requests_per_minute=rpm, max_wait_seconds=0.0)
        succeeded = 0
        failed = 0
        for _ in range(rpm * 3):
            try:
                rl.acquire()
                succeeded += 1
            except RateLimitExceeded:
                failed += 1

        # Exactly rpm tokens at start; with max_wait=0 there is no refill window.
        assert succeeded <= rpm + 2, f"Too many successes: {succeeded}"
        assert failed > 0

    def test_burst_over_token_budget(self) -> None:
        rl = RateLimiter(
            requests_per_minute=10_000,
            tokens_per_minute=500,
            max_wait_seconds=0.0,
        )
        succeeded = failed = 0
        for _ in range(20):
            try:
                rl.acquire(tokens=50)
                succeeded += 1
            except RateLimitExceeded:
                failed += 1

        assert succeeded <= 10 + 1  # 500 / 50 = 10 slots
        assert failed > 0

    def test_burst_succeeded_count_is_reproducible(self) -> None:
        """Two identical limiters burst-fired identically should yield same count."""

        def burst(rpm: int) -> int:
            rl = RateLimiter(requests_per_minute=rpm, max_wait_seconds=0.0)
            ok = 0
            for _ in range(rpm * 2):
                try:
                    rl.acquire()
                    ok += 1
                except RateLimitExceeded:
                    pass
            return ok

        a, b = burst(15), burst(15)
        assert a == b


# ---------------------------------------------------------------------------
# 5. Zero-limit (unlimited) behaviour
# ---------------------------------------------------------------------------


class TestZeroLimitsUnlimited:
    """RPM=0 and/or TPM=0 must never raise."""

    def test_both_zero_never_raises(self) -> None:
        rl = RateLimiter(requests_per_minute=0, tokens_per_minute=0, max_wait_seconds=0.0)
        for _ in range(1_000):
            rl.acquire(tokens=999_999)

    def test_rpm_zero_tpm_active_respects_token_limit(self) -> None:
        rl = RateLimiter(requests_per_minute=0, tokens_per_minute=100, max_wait_seconds=0.05)
        rl.acquire(tokens=100)
        with pytest.raises(RateLimitExceeded):
            rl.acquire(tokens=100)

    def test_tpm_zero_rpm_active_ignores_tokens(self) -> None:
        rl = RateLimiter(requests_per_minute=5, tokens_per_minute=0, max_wait_seconds=0.05)
        for _ in range(5):
            rl.acquire(tokens=1_000_000)  # token arg ignored
        with pytest.raises(RateLimitExceeded):
            rl.acquire()

    def test_both_zero_capacity_is_infinite(self) -> None:
        rl = RateLimiter(requests_per_minute=0, tokens_per_minute=0)
        assert rl.request_capacity == float("inf")
        assert rl.token_capacity == float("inf")

    def test_both_zero_no_internal_state_mutation(self) -> None:
        """Calling acquire on an unlimited limiter must not alter internal counters."""
        rl = RateLimiter(requests_per_minute=0, tokens_per_minute=0)
        before_req = rl._req_tokens
        before_tok = rl._tok_tokens
        for _ in range(100):
            rl.acquire(tokens=500)
        assert rl._req_tokens == before_req
        assert rl._tok_tokens == before_tok


# ---------------------------------------------------------------------------
# 6. Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Negative tokens, very large counts, max_wait=0."""

    def test_negative_tokens_treated_as_zero(self) -> None:
        """Negative token arg must not deduct from the bucket."""
        rl = RateLimiter(requests_per_minute=100, tokens_per_minute=100, max_wait_seconds=0.1)
        before = rl.token_capacity
        rl.acquire(tokens=-999)
        after = rl.token_capacity
        # Token bucket should be unchanged (negative deduction is clamped)
        assert after >= before - 1.0  # allow tiny refill jitter

    def test_zero_tokens_arg_skips_token_bucket_check(self) -> None:
        """tokens=0 must not cause a wait even when the token bucket is empty."""
        rl = RateLimiter(requests_per_minute=100, tokens_per_minute=10, max_wait_seconds=0.1)
        rl.acquire(tokens=10)  # drain token bucket
        rl.acquire(tokens=0)  # must not raise

    def test_very_large_token_count_raises_immediately(self) -> None:
        """Requesting more tokens than the entire TPM budget must raise quickly."""
        tpm = 1_000
        rl = RateLimiter(requests_per_minute=1_000, tokens_per_minute=tpm, max_wait_seconds=0.05)
        with pytest.raises(RateLimitExceeded):
            rl.acquire(tokens=tpm + 1)

    def test_max_wait_zero_raises_on_empty_bucket(self) -> None:
        rl = RateLimiter(requests_per_minute=1, max_wait_seconds=0.0)
        rl.acquire()
        with pytest.raises(RateLimitExceeded):
            rl.acquire()

    def test_rate_limit_exceeded_str(self) -> None:
        exc = RateLimitExceeded(wait_seconds=3.7)
        assert "3.7" in str(exc)
        assert exc.wait_seconds == pytest.approx(3.7)

    def test_capacity_after_complete_drain_is_zero(self) -> None:
        rpm = 5
        rl = RateLimiter(requests_per_minute=rpm, max_wait_seconds=0.0)
        for _ in range(rpm):
            try:
                rl.acquire()
            except RateLimitExceeded:
                break
        # Force refill timestamp alignment then check
        cap = rl.request_capacity
        assert cap <= 1.0, f"Expected near-zero capacity, got {cap}"

    def test_rpm_one_allows_exactly_one_call(self) -> None:
        rl = RateLimiter(requests_per_minute=1, max_wait_seconds=0.0)
        rl.acquire()  # first call must succeed
        with pytest.raises(RateLimitExceeded):
            rl.acquire()


# ---------------------------------------------------------------------------
# 7. 429 response handling via on_rate_limit_response
# ---------------------------------------------------------------------------


class TestRateLimitResponseHandling:
    """on_rate_limit_response drains both buckets; verify state and recovery."""

    def test_drains_request_bucket_to_zero(self) -> None:
        rl = RateLimiter(requests_per_minute=60)
        rl.on_rate_limit_response(retry_after=0.0)
        assert rl._req_tokens == 0.0

    def test_drains_token_bucket_to_zero(self) -> None:
        rl = RateLimiter(tokens_per_minute=100_000)
        rl.on_rate_limit_response(retry_after=0.0)
        assert rl._tok_tokens == 0.0

    def test_drains_both_buckets(self) -> None:
        rl = RateLimiter(requests_per_minute=60, tokens_per_minute=100_000)
        rl.on_rate_limit_response(retry_after=0.0)
        assert rl._req_tokens == 0.0
        assert rl._tok_tokens == 0.0

    def test_acquire_raises_immediately_after_drain(self) -> None:
        rl = RateLimiter(requests_per_minute=60, max_wait_seconds=0.05)
        rl.on_rate_limit_response(retry_after=0.0)
        with pytest.raises(RateLimitExceeded):
            rl.acquire()

    def test_recovery_after_drain_with_sleep(self) -> None:
        """After drain, sleeping long enough should restore enough capacity."""
        rpm = 600  # 10 req/sec – fast refill
        rl = RateLimiter(requests_per_minute=rpm, max_wait_seconds=0.5)
        rl.on_rate_limit_response(retry_after=0.0)
        time.sleep(0.15)  # should restore ~1.5 req-tokens
        rl.acquire()  # must not raise

    def test_retry_after_respected_as_sleep(self) -> None:
        """retry_after should cause a real blocking sleep inside the call."""
        rl = RateLimiter(requests_per_minute=60, max_wait_seconds=1.0)
        start = time.monotonic()
        rl.on_rate_limit_response(retry_after=0.1)
        elapsed = time.monotonic() - start
        assert elapsed >= 0.08, f"Expected ~0.1 s sleep, got {elapsed:.3f} s"

    def test_retry_after_capped_by_max_wait(self) -> None:
        """retry_after > max_wait must be clamped to max_wait."""
        rl = RateLimiter(requests_per_minute=60, max_wait_seconds=0.1)
        start = time.monotonic()
        rl.on_rate_limit_response(retry_after=60.0)
        elapsed = time.monotonic() - start
        assert elapsed < 0.5, f"Sleep was not capped; took {elapsed:.2f} s"

    def test_idempotent_double_drain(self) -> None:
        """Calling on_rate_limit_response twice must leave buckets at zero."""
        rl = RateLimiter(requests_per_minute=60, tokens_per_minute=1_000)
        rl.on_rate_limit_response(retry_after=0.0)
        rl.on_rate_limit_response(retry_after=0.0)
        assert rl._req_tokens == 0.0
        assert rl._tok_tokens == 0.0


# ---------------------------------------------------------------------------
# 8. Thread safety – interleaved acquire + record_usage
# ---------------------------------------------------------------------------


class TestThreadSafetyInterleaved:
    """Acquire and record_usage called concurrently must not corrupt state."""

    def test_acquire_and_record_interleaved_no_errors(self) -> None:
        rl = RateLimiter(
            requests_per_minute=1_000,
            tokens_per_minute=100_000,
            max_wait_seconds=0.2,
        )
        errors: list[Exception] = []
        lock = threading.Lock()

        def acquirer() -> None:
            for _ in range(20):
                try:
                    rl.acquire(tokens=100)
                except RateLimitExceeded:
                    pass
                except Exception as exc:  # noqa: BLE001
                    with lock:
                        errors.append(exc)

        def recorder() -> None:
            for _ in range(20):
                try:
                    rl.record_usage(prompt_tokens=80, completion_tokens=20)
                except Exception as exc:  # noqa: BLE001
                    with lock:
                        errors.append(exc)

        threads = [threading.Thread(target=acquirer, daemon=True) for _ in range(5)] + [
            threading.Thread(target=recorder, daemon=True) for _ in range(5)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=15)

        assert errors == [], f"Unexpected exceptions: {errors}"

    def test_capacity_always_non_negative_under_concurrency(self) -> None:
        """Token/request capacity must never report negative values."""
        rl = RateLimiter(
            requests_per_minute=200,
            tokens_per_minute=10_000,
            max_wait_seconds=0.1,
        )
        negatives: list[float] = []
        done = threading.Event()

        def sampler() -> None:
            while not done.is_set():
                rc = rl.request_capacity
                tc = rl.token_capacity
                if rc < 0 or tc < 0:
                    negatives.extend([rc, tc])

        def driver() -> None:
            for _ in range(50):
                with contextlib.suppress(RateLimitExceeded):
                    rl.acquire(tokens=50)
                rl.record_usage(prompt_tokens=30, completion_tokens=20)

        sampler_thread = threading.Thread(target=sampler, daemon=True)
        driver_threads = [threading.Thread(target=driver, daemon=True) for _ in range(5)]

        sampler_thread.start()
        for t in driver_threads:
            t.start()
        for t in driver_threads:
            t.join(timeout=15)
        done.set()
        sampler_thread.join(timeout=2)

        assert negatives == [], f"Capacity went negative: {negatives}"

    def test_concurrent_on_rate_limit_response_safe(self) -> None:
        """Concurrent 429 drains from multiple threads must not raise."""
        rl = RateLimiter(requests_per_minute=100, tokens_per_minute=10_000, max_wait_seconds=0.2)
        errors: list[Exception] = []
        lock = threading.Lock()

        def drain() -> None:
            try:
                rl.on_rate_limit_response(retry_after=0.0)
            except Exception as exc:  # noqa: BLE001
                with lock:
                    errors.append(exc)

        threads = [threading.Thread(target=drain, daemon=True) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=5)

        assert errors == []
        assert rl._req_tokens == 0.0
        assert rl._tok_tokens == 0.0

    def test_high_contention_final_state_consistent(self) -> None:
        """After a storm of concurrent operations, internal state must be self-consistent."""
        rpm = 500
        tpm = 50_000
        rl = RateLimiter(requests_per_minute=rpm, tokens_per_minute=tpm, max_wait_seconds=0.15)

        barrier = threading.Barrier(10)

        def worker() -> None:
            barrier.wait()  # all threads start simultaneously
            for i in range(10):
                with contextlib.suppress(RateLimitExceeded):
                    rl.acquire(tokens=100)
                if i % 3 == 0:
                    rl.record_usage(prompt_tokens=60, completion_tokens=40)

        threads = [threading.Thread(target=worker, daemon=True) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=15)

        assert 0.0 <= rl._req_tokens <= float(rpm) + 0.01
        assert 0.0 <= rl._tok_tokens <= float(tpm) + 0.01


# ---------------------------------------------------------------------------
# 9. Property-based tests with Hypothesis
# ---------------------------------------------------------------------------


class TestPropertyBased:
    """Hypothesis-driven invariant checks."""

    @given(
        rpm=st.integers(min_value=1, max_value=10_000),
        tpm=st.integers(min_value=1, max_value=1_000_000),
    )
    @settings(max_examples=60, deadline=None)
    def test_initial_capacity_equals_limit(self, rpm: int, tpm: int) -> None:
        """Freshly constructed limiters start at full capacity."""
        rl = RateLimiter(requests_per_minute=rpm, tokens_per_minute=tpm)
        assert rl.request_capacity == pytest.approx(float(rpm), abs=0.5)
        assert rl.token_capacity == pytest.approx(float(tpm), abs=0.5)

    @given(
        rpm=st.integers(min_value=1, max_value=500),
        n=st.integers(min_value=1, max_value=20),
    )
    @settings(max_examples=50, deadline=None)
    def test_capacity_never_exceeds_max_after_acquires(self, rpm: int, n: int) -> None:
        """Capacity must never exceed the configured RPM ceiling."""
        rl = RateLimiter(requests_per_minute=rpm, tokens_per_minute=0, max_wait_seconds=0.0)
        for _ in range(n):
            with contextlib.suppress(RateLimitExceeded):
                rl.acquire()
        assert rl.request_capacity <= float(rpm) + 0.01

    @given(tokens=st.integers(min_value=0, max_value=10_000))
    @settings(max_examples=60, deadline=None)
    def test_unlimited_limiter_never_raises(self, tokens: int) -> None:
        """With both limits at 0, acquire must always succeed regardless of token count."""
        rl = RateLimiter(requests_per_minute=0, tokens_per_minute=0, max_wait_seconds=0.0)
        rl.acquire(tokens=tokens)  # must not raise

    @given(
        rpm=st.integers(min_value=1, max_value=1_000),
        wait=st.floats(min_value=0.0, max_value=0.5),
    )
    @settings(max_examples=40, deadline=None)
    def test_capacity_non_negative_after_drain_and_sleep(self, rpm: int, wait: float) -> None:
        """After draining the bucket and sleeping, capacity must remain >= 0."""
        rl = RateLimiter(requests_per_minute=rpm, tokens_per_minute=0, max_wait_seconds=0.0)
        for _ in range(rpm):
            try:
                rl.acquire()
            except RateLimitExceeded:
                break
        time.sleep(wait)
        assert rl.request_capacity >= 0.0

    @given(
        rpm=st.integers(min_value=10, max_value=1_000),
        tpm=st.integers(min_value=100, max_value=100_000),
        token_ask=st.integers(min_value=1, max_value=200),
    )
    @settings(max_examples=50, deadline=None)
    def test_acquire_with_zero_tokens_never_raises_when_rpm_has_budget(
        self, rpm: int, tpm: int, token_ask: int
    ) -> None:
        """After draining the token bucket, acquire(tokens=0) must succeed if RPM is available."""
        rl = RateLimiter(requests_per_minute=rpm, tokens_per_minute=tpm, max_wait_seconds=0.1)
        # Drain token budget
        with contextlib.suppress(RateLimitExceeded):
            rl.acquire(tokens=tpm)
        # tokens=0 bypasses token bucket — must not raise while RPM has capacity
        rl.acquire(tokens=0)

    @given(
        retry_after=st.floats(min_value=0.0, max_value=0.05),
    )
    @settings(max_examples=30, deadline=None)
    def test_on_rate_limit_response_always_drains_to_zero(self, retry_after: float) -> None:
        """on_rate_limit_response must always zero both buckets."""
        rl = RateLimiter(
            requests_per_minute=100,
            tokens_per_minute=10_000,
            max_wait_seconds=0.1,
        )
        rl.on_rate_limit_response(retry_after=retry_after)
        assert rl._req_tokens == 0.0
        assert rl._tok_tokens == 0.0

    @given(
        prompt=st.integers(min_value=0, max_value=10_000),
        completion=st.integers(min_value=0, max_value=10_000),
    )
    @settings(max_examples=60, deadline=None)
    def test_record_usage_never_raises(self, prompt: int, completion: int) -> None:
        """record_usage must accept any non-negative int pair without raising."""
        rl = RateLimiter(tokens_per_minute=100_000)
        rl.record_usage(prompt_tokens=prompt, completion_tokens=completion)

    @given(
        max_wait=st.floats(min_value=0.0, max_value=0.1),
        n_acquires=st.integers(min_value=1, max_value=10),
    )
    @settings(max_examples=40, deadline=None)
    def test_exceeded_error_wait_seconds_is_positive(
        self, max_wait: float, n_acquires: int
    ) -> None:
        """When RateLimitExceeded is raised, wait_seconds must be positive."""
        rl = RateLimiter(requests_per_minute=1, max_wait_seconds=max_wait)
        rl.acquire()  # consume the single token
        try:
            for _ in range(n_acquires):
                rl.acquire()
        except RateLimitExceeded as exc:
            assert exc.wait_seconds > 0.0
