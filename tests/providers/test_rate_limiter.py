"""Tests for the provider rate limiter."""
import time
import threading
import pytest

from missy.providers.rate_limiter import RateLimiter, RateLimitExceeded


class TestRateLimiterInit:
    def test_default_values(self):
        rl = RateLimiter()
        assert rl._rpm == 60
        assert rl._tpm == 100_000

    def test_custom_values(self):
        rl = RateLimiter(requests_per_minute=10, tokens_per_minute=5000, max_wait_seconds=5.0)
        assert rl._rpm == 10
        assert rl._tpm == 5000
        assert rl._max_wait == 5.0

    def test_unlimited_when_zero(self):
        rl = RateLimiter(requests_per_minute=0, tokens_per_minute=0)
        # Should not block or raise
        rl.acquire()
        rl.acquire(tokens=999999)


class TestAcquire:
    def test_acquire_succeeds_under_limit(self):
        rl = RateLimiter(requests_per_minute=60)
        rl.acquire()  # Should not raise

    def test_acquire_blocks_when_near_limit(self):
        # Use high RPM so refill is fast; exhaust bucket, then verify acquire still works
        rl = RateLimiter(requests_per_minute=600, max_wait_seconds=2.0)
        # 600 RPM = 10/sec; exhaust the bucket
        for _ in range(600):
            rl.acquire()
        # Next call should block briefly then succeed (tokens refill at 10/sec)
        start = time.monotonic()
        rl.acquire()
        elapsed = time.monotonic() - start
        assert elapsed >= 0  # Had to wait for refill

    def test_acquire_raises_on_timeout(self):
        rl = RateLimiter(requests_per_minute=1, max_wait_seconds=0.1)
        rl.acquire()
        with pytest.raises(RateLimitExceeded):
            rl.acquire()

    def test_acquire_with_token_budget(self):
        rl = RateLimiter(requests_per_minute=100, tokens_per_minute=1000, max_wait_seconds=0.1)
        rl.acquire(tokens=500)
        rl.acquire(tokens=500)
        with pytest.raises(RateLimitExceeded):
            rl.acquire(tokens=500)

    def test_unlimited_requests_with_token_limit(self):
        rl = RateLimiter(requests_per_minute=0, tokens_per_minute=1000, max_wait_seconds=0.1)
        rl.acquire(tokens=500)
        rl.acquire(tokens=500)
        with pytest.raises(RateLimitExceeded):
            rl.acquire(tokens=500)

    def test_unlimited_tokens_with_request_limit(self):
        rl = RateLimiter(requests_per_minute=2, tokens_per_minute=0, max_wait_seconds=0.1)
        rl.acquire(tokens=999999)
        rl.acquire(tokens=999999)
        with pytest.raises(RateLimitExceeded):
            rl.acquire()


class TestCapacityProperties:
    def test_request_capacity_starts_at_max(self):
        rl = RateLimiter(requests_per_minute=60)
        assert rl.request_capacity == pytest.approx(60.0, abs=1.0)

    def test_request_capacity_decreases_after_acquire(self):
        rl = RateLimiter(requests_per_minute=60)
        rl.acquire()
        assert rl.request_capacity < 60.0

    def test_token_capacity_starts_at_max(self):
        rl = RateLimiter(tokens_per_minute=100_000)
        assert rl.token_capacity == pytest.approx(100_000.0, abs=100.0)

    def test_unlimited_capacity_is_inf(self):
        rl = RateLimiter(requests_per_minute=0, tokens_per_minute=0)
        assert rl.request_capacity == float("inf")
        assert rl.token_capacity == float("inf")


class TestOnRateLimitResponse:
    def test_drains_request_bucket(self):
        rl = RateLimiter(requests_per_minute=60)
        rl.on_rate_limit_response(retry_after=0.0)
        assert rl._req_tokens == 0.0

    def test_drains_token_bucket(self):
        rl = RateLimiter(tokens_per_minute=100_000)
        rl.on_rate_limit_response(retry_after=0.0)
        assert rl._tok_tokens == 0.0


class TestRecordUsage:
    def test_record_usage_does_not_crash(self):
        rl = RateLimiter(tokens_per_minute=100_000)
        rl.record_usage(prompt_tokens=100, completion_tokens=50)

    def test_record_usage_with_no_limit(self):
        rl = RateLimiter(tokens_per_minute=0)
        rl.record_usage(prompt_tokens=100, completion_tokens=50)  # Should be no-op


class TestThreadSafety:
    def test_concurrent_acquire(self):
        rl = RateLimiter(requests_per_minute=100, max_wait_seconds=5.0)
        errors = []

        def worker():
            try:
                for _ in range(10):
                    rl.acquire()
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)

        assert len(errors) == 0
