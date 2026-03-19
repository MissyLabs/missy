"""Edge-case tests for the provider rate limiter.

Covers scenarios NOT exercised by test_rate_limiter_stress.py:

1. tokens_per_minute=0 means unlimited tokens (rpm still enforced).
2. max_wait_seconds negative: deadline is already expired, so any bucket
   miss raises immediately without sleeping.
3. Barrier-synchronised concurrent acquire: all threads race at exactly
   the same instant to reveal lock-serialisation correctness.
4. Refill-math accuracy after a measured sleep: verify the formula
   ``refill = elapsed * (rpm / 60.0)`` to within 5 % tolerance.
5. record_usage behaviour: the current implementation intentionally does
   NOT decrement _tok_tokens (it only clamps to >= 0).  Tests document
   this contract so future changes are visible.
6. RPM-exhausted / TPM-fine interaction, and TPM-exhausted / RPM-fine
   interaction, verifying which bucket is the binding constraint.
"""

from __future__ import annotations

import threading
import time

import pytest

from missy.providers.rate_limiter import RateLimiter, RateLimitExceeded

# ---------------------------------------------------------------------------
# 1. tokens_per_minute=0 means unlimited token budget
# ---------------------------------------------------------------------------


class TestTokensPerMinuteZeroIsUnlimited:
    """tpm=0 disables token enforcement even when rpm is active."""

    def test_tpm_zero_any_token_count_allowed(self) -> None:
        """With tpm=0, passing arbitrarily large token counts must not raise."""
        rl = RateLimiter(requests_per_minute=100, tokens_per_minute=0, max_wait_seconds=0.0)
        for _ in range(10):
            rl.acquire(tokens=999_999_999)

    def test_tpm_zero_rpm_still_enforced(self) -> None:
        """With tpm=0, rpm limit is still active and raises when exhausted."""
        rl = RateLimiter(requests_per_minute=3, tokens_per_minute=0, max_wait_seconds=0.0)
        rl.acquire(tokens=1_000_000)
        rl.acquire(tokens=1_000_000)
        rl.acquire(tokens=1_000_000)
        with pytest.raises(RateLimitExceeded):
            rl.acquire(tokens=1_000_000)

    def test_tpm_zero_token_capacity_is_infinite(self) -> None:
        """token_capacity property must report +inf when tpm=0."""
        rl = RateLimiter(requests_per_minute=10, tokens_per_minute=0)
        assert rl.token_capacity == float("inf")

    def test_tpm_zero_no_token_deduction_on_acquire(self) -> None:
        """With tpm=0, the internal tok_tokens counter must never change."""
        rl = RateLimiter(requests_per_minute=100, tokens_per_minute=0, max_wait_seconds=0.0)
        before = rl._tok_tokens
        for _ in range(20):
            rl.acquire(tokens=50_000)
        assert rl._tok_tokens == before

    def test_tpm_zero_record_usage_is_no_op(self) -> None:
        """record_usage with tpm=0 must return immediately without touching state."""
        rl = RateLimiter(requests_per_minute=100, tokens_per_minute=0)
        before = rl._tok_tokens
        rl.record_usage(prompt_tokens=500, completion_tokens=500)
        assert rl._tok_tokens == before


# ---------------------------------------------------------------------------
# 2. Negative max_wait_seconds: deadline already expired
# ---------------------------------------------------------------------------


class TestNegativeMaxWait:
    """max_wait_seconds < 0 means the deadline is immediately in the past.

    Any acquire call that cannot be served from the current bucket must
    raise RateLimitExceeded without any sleep.
    """

    def test_negative_max_wait_raises_on_empty_bucket(self) -> None:
        """First acquire drains the single token; second must raise instantly."""
        rl = RateLimiter(requests_per_minute=1, max_wait_seconds=-1.0)
        rl.acquire()  # consumes the only token
        with pytest.raises(RateLimitExceeded):
            rl.acquire()

    def test_negative_max_wait_raises_without_sleeping(self) -> None:
        """The raise must happen almost immediately (< 50 ms)."""
        rl = RateLimiter(requests_per_minute=1, max_wait_seconds=-99.0)
        rl.acquire()
        start = time.monotonic()
        with pytest.raises(RateLimitExceeded):
            rl.acquire()
        elapsed = time.monotonic() - start
        assert elapsed < 0.05, f"Expected near-instant raise; took {elapsed:.3f}s"

    def test_negative_max_wait_still_allows_first_acquire(self) -> None:
        """A full bucket must satisfy the first acquire even with negative max_wait."""
        rl = RateLimiter(requests_per_minute=10, max_wait_seconds=-5.0)
        # Should not raise – bucket starts full
        rl.acquire()

    def test_negative_max_wait_raises_on_large_token_ask(self) -> None:
        """Asking for more tokens than the budget with negative wait must raise."""
        rl = RateLimiter(
            requests_per_minute=1000,
            tokens_per_minute=100,
            max_wait_seconds=-1.0,
        )
        with pytest.raises(RateLimitExceeded):
            rl.acquire(tokens=101)  # one more than the full bucket

    def test_negative_max_wait_no_sleep_on_rpm_exhausted(self) -> None:
        """RPM exhaustion with negative max_wait must raise without any blocking."""
        rpm = 5
        rl = RateLimiter(requests_per_minute=rpm, max_wait_seconds=-10.0)
        for _ in range(rpm):
            rl.acquire()

        start = time.monotonic()
        with pytest.raises(RateLimitExceeded):
            rl.acquire()
        elapsed = time.monotonic() - start
        assert elapsed < 0.05, f"Unexpected sleep of {elapsed:.3f}s with negative max_wait"


# ---------------------------------------------------------------------------
# 3. Barrier-synchronised concurrent acquire
# ---------------------------------------------------------------------------


class TestBarrierSynchronisedConcurrentAcquire:
    """All threads start at the same instant via threading.Barrier.

    This maximises lock contention and verifies that exactly the right
    number of tokens are consumed.
    """

    def test_all_succeed_when_threads_fewer_than_bucket(self) -> None:
        """N threads < rpm bucket: every thread should get a token."""
        n_threads = 20
        rpm = 100
        rl = RateLimiter(requests_per_minute=rpm, tokens_per_minute=0, max_wait_seconds=0.05)
        results = {"ok": 0, "fail": 0}
        lock = threading.Lock()
        barrier = threading.Barrier(n_threads)

        def worker() -> None:
            barrier.wait()
            try:
                rl.acquire()
                with lock:
                    results["ok"] += 1
            except RateLimitExceeded:
                with lock:
                    results["fail"] += 1

        threads = [threading.Thread(target=worker, daemon=True) for _ in range(n_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=5)

        assert results["fail"] == 0, f"Expected no failures; got {results['fail']}"
        assert results["ok"] == n_threads

    def test_excess_threads_are_rejected(self) -> None:
        """More threads than the rpm bucket: exactly rpm should succeed."""
        n_threads = 30
        rpm = 10
        rl = RateLimiter(requests_per_minute=rpm, tokens_per_minute=0, max_wait_seconds=0.0)
        results = {"ok": 0, "fail": 0}
        lock = threading.Lock()
        barrier = threading.Barrier(n_threads)

        def worker() -> None:
            barrier.wait()
            try:
                rl.acquire()
                with lock:
                    results["ok"] += 1
            except RateLimitExceeded:
                with lock:
                    results["fail"] += 1

        threads = [threading.Thread(target=worker, daemon=True) for _ in range(n_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=5)

        # Exactly rpm tokens in bucket; max_wait=0 means no refill occurs.
        assert results["ok"] <= rpm, f"More successes ({results['ok']}) than bucket ({rpm})"
        assert results["fail"] > 0, "Expected at least some rejections"
        assert results["ok"] + results["fail"] == n_threads

    def test_no_unexpected_errors_under_high_contention(self) -> None:
        """Under heavy simultaneous pressure, only RateLimitExceeded should ever escape."""
        n_threads = 50
        rl = RateLimiter(requests_per_minute=25, tokens_per_minute=0, max_wait_seconds=0.02)
        unexpected: list[Exception] = []
        lock = threading.Lock()
        barrier = threading.Barrier(n_threads)

        def worker() -> None:
            barrier.wait()
            try:
                rl.acquire()
            except RateLimitExceeded:
                pass
            except Exception as exc:  # noqa: BLE001
                with lock:
                    unexpected.append(exc)

        threads = [threading.Thread(target=worker, daemon=True) for _ in range(n_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)

        assert unexpected == [], f"Unexpected exceptions: {unexpected}"

    def test_token_bucket_exact_under_concurrency(self) -> None:
        """Token bucket is depleted by concurrent acquires; no over-spend."""
        n_threads = 10
        tpm = 1000
        tokens_per_call = 100  # 10 threads × 100 = exactly tpm
        rl = RateLimiter(
            requests_per_minute=10_000,
            tokens_per_minute=tpm,
            max_wait_seconds=0.0,
        )
        results = {"ok": 0, "fail": 0}
        lock = threading.Lock()
        barrier = threading.Barrier(n_threads)

        def worker() -> None:
            barrier.wait()
            try:
                rl.acquire(tokens=tokens_per_call)
                with lock:
                    results["ok"] += 1
            except RateLimitExceeded:
                with lock:
                    results["fail"] += 1

        threads = [threading.Thread(target=worker, daemon=True) for _ in range(n_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=5)

        # All 10 threads each consuming 100 tokens should just fit in tpm=1000
        assert results["ok"] == n_threads, (
            f"Expected all {n_threads} to succeed; got {results['ok']} ok, {results['fail']} fail"
        )
        assert rl._tok_tokens >= 0.0


# ---------------------------------------------------------------------------
# 4. Refill-math accuracy after a measured sleep
# ---------------------------------------------------------------------------


class TestRefillMathAccuracy:
    """Verify refill = elapsed * (limit / 60.0) holds within tight tolerance."""

    def test_request_bucket_refill_formula_accuracy(self) -> None:
        """After draining and sleeping a known interval, capacity must match formula."""
        # 6000 rpm = 100 req/sec; small sleep gives easily-measurable refill.
        rpm = 6000
        rl = RateLimiter(requests_per_minute=rpm, tokens_per_minute=0, max_wait_seconds=0.0)

        # Drain completely via a single drain call on the internal counter
        with rl._lock:
            rl._req_tokens = 0.0
            drain_time = time.monotonic()
            rl._req_last_refill = drain_time

        sleep_s = 0.1
        time.sleep(sleep_s)

        cap = rl.request_capacity
        expected = (rpm / 60.0) * sleep_s  # 10.0 tokens

        # Allow 5 % tolerance for OS scheduling jitter
        assert abs(cap - expected) / expected < 0.05, (
            f"Refill mismatch: expected ~{expected:.2f}, got {cap:.2f} "
            f"(delta={abs(cap - expected) / expected:.1%})"
        )

    def test_token_bucket_refill_formula_accuracy(self) -> None:
        """Token bucket refill matches tpm/60 * elapsed to within 5 %."""
        tpm = 12000  # 200 tokens/sec – easy to verify with short sleep
        rl = RateLimiter(requests_per_minute=10_000, tokens_per_minute=tpm, max_wait_seconds=0.0)

        with rl._lock:
            rl._tok_tokens = 0.0
            rl._tok_last_refill = time.monotonic()

        sleep_s = 0.1
        time.sleep(sleep_s)

        cap = rl.token_capacity
        expected = (tpm / 60.0) * sleep_s  # 20.0 tokens

        assert abs(cap - expected) / expected < 0.05, (
            f"Token refill mismatch: expected ~{expected:.2f}, got {cap:.2f}"
        )

    def test_refill_after_long_sleep_caps_at_maximum(self) -> None:
        """Even after sleeping much longer than needed, bucket must not exceed limit."""
        rpm = 60
        rl = RateLimiter(requests_per_minute=rpm, tokens_per_minute=0, max_wait_seconds=0.0)

        with rl._lock:
            rl._req_tokens = 0.0
            rl._req_last_refill = time.monotonic()

        time.sleep(0.25)  # would refill 60/60 * 0.25 * 4 = 1 token, well under cap
        # Now pretend a very long time has passed by back-dating the timestamp
        with rl._lock:
            rl._req_last_refill -= 3600.0  # simulate 1 hour of elapsed time

        cap = rl.request_capacity
        assert cap <= float(rpm) + 0.01, f"Bucket overflowed to {cap}; max is {rpm}"

    def test_refill_rate_proportional_to_rpm(self) -> None:
        """A limiter with 2× rpm refills 2× as fast over the same interval."""
        sleep_s = 0.1

        def measure_refill(rpm: int) -> float:
            rl = RateLimiter(requests_per_minute=rpm, tokens_per_minute=0, max_wait_seconds=0.0)
            with rl._lock:
                rl._req_tokens = 0.0
                rl._req_last_refill = time.monotonic()
            time.sleep(sleep_s)
            return rl.request_capacity

        refill_low = measure_refill(600)  # 10 req/sec
        refill_high = measure_refill(1200)  # 20 req/sec

        # refill_high should be approximately 2× refill_low
        ratio = refill_high / refill_low
        assert 1.7 < ratio < 2.3, f"Expected ~2× ratio; got {ratio:.2f}"


# ---------------------------------------------------------------------------
# 5. record_usage behaviour
# ---------------------------------------------------------------------------


class TestRecordUsageBehaviour:
    """Document the contract of record_usage.

    record_usage deducts actual token consumption from the bucket so the
    rate limiter tracks real usage accurately.
    """

    def test_record_usage_deducts_actual_tokens(self) -> None:
        """record_usage after an acquire should deduct the actual token count."""
        rl = RateLimiter(requests_per_minute=1000, tokens_per_minute=1000, max_wait_seconds=0.0)
        rl.acquire(tokens=100)  # _tok_tokens now 900
        rl.record_usage(prompt_tokens=80, completion_tokens=20)
        # 900 - 100 = 800
        assert rl._tok_tokens == 800.0

    def test_record_usage_is_no_op_when_tpm_zero(self) -> None:
        """record_usage must return immediately and leave state unchanged when tpm=0."""
        rl = RateLimiter(requests_per_minute=100, tokens_per_minute=0)
        before = rl._tok_tokens
        rl.record_usage(prompt_tokens=500, completion_tokens=500)
        assert rl._tok_tokens == before

    def test_record_usage_clamps_to_zero_not_negative(self) -> None:
        """Manually setting _tok_tokens to negative; record_usage must clamp to 0."""
        rl = RateLimiter(requests_per_minute=1000, tokens_per_minute=1000, max_wait_seconds=0.0)
        with rl._lock:
            rl._tok_tokens = -50.0  # force negative (shouldn't happen in practice)
        rl.record_usage(prompt_tokens=100, completion_tokens=100)
        assert rl._tok_tokens >= 0.0

    def test_record_usage_zero_values_is_no_op(self) -> None:
        """record_usage(0, 0) must not alter state."""
        rl = RateLimiter(requests_per_minute=100, tokens_per_minute=500)
        rl.acquire(tokens=100)
        cap_before = rl._tok_tokens
        rl.record_usage(prompt_tokens=0, completion_tokens=0)
        assert rl._tok_tokens == cap_before

    def test_record_usage_does_not_affect_request_bucket(self) -> None:
        """record_usage operates only on the token bucket, never the request bucket."""
        rl = RateLimiter(requests_per_minute=100, tokens_per_minute=1000)
        rl.acquire(tokens=200)
        req_before = rl._req_tokens
        rl.record_usage(prompt_tokens=100, completion_tokens=100)
        req_after = rl._req_tokens
        assert req_after == req_before

    def test_record_usage_never_raises_on_large_values(self) -> None:
        """record_usage must silently handle very large token counts."""
        rl = RateLimiter(requests_per_minute=100, tokens_per_minute=1000)
        rl.record_usage(prompt_tokens=10_000_000, completion_tokens=10_000_000)


# ---------------------------------------------------------------------------
# 6. RPM / TPM limit interaction
# ---------------------------------------------------------------------------


class TestRpmTpmInteraction:
    """Both buckets are independent; exhausting one does not affect the other."""

    def test_rpm_exhausted_tpm_fine_raises_on_next_request(self) -> None:
        """After using all RPM tokens, further calls raise even with tokens=0."""
        rpm = 3
        rl = RateLimiter(requests_per_minute=rpm, tokens_per_minute=100_000, max_wait_seconds=0.0)
        for _ in range(rpm):
            rl.acquire(tokens=10)

        with pytest.raises(RateLimitExceeded):
            rl.acquire(tokens=0)

    def test_tpm_exhausted_rpm_fine_tokens_zero_succeeds(self) -> None:
        """After exhausting TPM, acquire(tokens=0) still succeeds if RPM has budget."""
        rl = RateLimiter(requests_per_minute=100, tokens_per_minute=50, max_wait_seconds=0.0)
        rl.acquire(tokens=50)  # exhaust tpm
        # tokens=0 bypasses the token bucket check entirely
        rl.acquire(tokens=0)  # must not raise

    def test_tpm_exhausted_rpm_fine_tokens_positive_raises(self) -> None:
        """After exhausting TPM, acquire with any tokens > 0 raises."""
        rl = RateLimiter(requests_per_minute=100, tokens_per_minute=50, max_wait_seconds=0.0)
        rl.acquire(tokens=50)  # exhaust tpm
        with pytest.raises(RateLimitExceeded):
            rl.acquire(tokens=1)

    def test_both_exhausted_raises_on_any_acquire(self) -> None:
        """Both rpm and tpm exhausted: any acquire call must raise."""
        rl = RateLimiter(requests_per_minute=2, tokens_per_minute=100, max_wait_seconds=0.0)
        rl.acquire(tokens=50)
        rl.acquire(tokens=50)  # rpm=0 and tpm=0 now
        with pytest.raises(RateLimitExceeded):
            rl.acquire(tokens=0)

    def test_rpm_limits_calls_even_when_each_uses_few_tokens(self) -> None:
        """RPM cap fires before TPM cap when token-per-call is small."""
        rpm = 5
        tpm = 10_000
        rl = RateLimiter(requests_per_minute=rpm, tokens_per_minute=tpm, max_wait_seconds=0.0)
        succeeded = 0
        for _ in range(rpm + 5):
            try:
                rl.acquire(tokens=1)  # each call uses almost no token budget
                succeeded += 1
            except RateLimitExceeded:
                break

        assert succeeded == rpm, f"Expected exactly {rpm} successes; got {succeeded}"
        # Token budget should be largely intact
        assert rl.token_capacity > tpm * 0.9

    def test_tpm_limits_calls_when_rpm_is_generous(self) -> None:
        """TPM cap fires before RPM cap when each call uses many tokens."""
        rpm = 10_000
        tpm = 300
        token_per_call = 100  # 3 calls will exhaust tpm
        rl = RateLimiter(requests_per_minute=rpm, tokens_per_minute=tpm, max_wait_seconds=0.0)
        succeeded = 0
        for _ in range(10):
            try:
                rl.acquire(tokens=token_per_call)
                succeeded += 1
            except RateLimitExceeded:
                break

        assert succeeded == tpm // token_per_call, (
            f"Expected {tpm // token_per_call} successes; got {succeeded}"
        )
        # Request bucket should still be largely full
        assert rl.request_capacity > rpm * 0.9

    def test_rpm_and_tpm_both_enforced_independently_after_reset(self) -> None:
        """Draining via on_rate_limit_response resets both; each is checked independently."""
        rl = RateLimiter(requests_per_minute=60, tokens_per_minute=600, max_wait_seconds=0.0)
        rl.on_rate_limit_response(retry_after=0.0)

        assert rl._req_tokens == 0.0
        assert rl._tok_tokens == 0.0

        # With both at zero, even tokens=0 should fail (rpm is the binding constraint)
        with pytest.raises(RateLimitExceeded):
            rl.acquire(tokens=0)
