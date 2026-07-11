"""Unit tests for missy.channels.discord.rate_limit.DiscordUserRateLimiter.

DISC-CMD-008 (task #10 validation): no per-user rate limiter previously
existed for incoming Discord commands. These tests exercise the
limiter in isolation; tests/channels/test_discord_channel_coverage.py
covers the real dispatch-path wiring (_handle_message/_handle_interaction).
"""

from __future__ import annotations

import time

from missy.channels.discord.rate_limit import DiscordUserRateLimiter, RateLimitResult


class TestDiscordUserRateLimiter:
    def test_first_request_from_a_fresh_bucket_is_allowed(self):
        """Regression: a bucket's initial capacity must equal exactly
        rpm on the very first check(), not one token short. An earlier
        version called time.monotonic() twice at slightly different
        points (once by the caller, once inside the bucket constructor)
        and subtracted them in the wrong order, producing a negative
        elapsed time and denying every user's first-ever request."""
        limiter = DiscordUserRateLimiter(requests_per_minute=1)
        result = limiter.check("user-1")
        assert result == RateLimitResult(allowed=True, retry_after_seconds=0.0)

    def test_exceeding_capacity_denies_and_reports_wait_time(self):
        limiter = DiscordUserRateLimiter(requests_per_minute=1)
        assert limiter.check("user-1").allowed is True
        second = limiter.check("user-1")
        assert second.allowed is False
        # ~1 request/minute means the wait should be close to 60s.
        assert 55.0 < second.retry_after_seconds <= 60.0

    def test_denied_request_does_not_consume_a_token(self):
        """A refused request must not itself burn budget -- otherwise a
        user spamming past their limit would never recover even after
        the window naturally refills, since every refused attempt would
        keep draining tokens that were never actually granted."""
        limiter = DiscordUserRateLimiter(requests_per_minute=1)
        assert limiter.check("user-1").allowed is True
        for _ in range(5):
            assert limiter.check("user-1").allowed is False
        # Manually advance the bucket's clock to simulate the window
        # passing, then confirm exactly one more request is allowed --
        # if the repeated denials above had wrongly drained tokens
        # further into negative territory, this would still be denied.
        bucket = limiter._buckets["user-1"]
        bucket.last_refill -= 61.0
        assert limiter.check("user-1").allowed is True

    def test_bucket_refills_over_time(self):
        limiter = DiscordUserRateLimiter(requests_per_minute=1)  # capacity == 1 token
        assert limiter.check("user-1").allowed is True
        assert limiter.check("user-1").allowed is False
        # Halfway through the 60s window: still not enough for a full token.
        bucket = limiter._buckets["user-1"]
        bucket.last_refill -= 30.0
        assert limiter.check("user-1").allowed is False
        # The rest of the window elapses: bucket is back to full capacity.
        bucket.last_refill -= 30.0
        assert limiter.check("user-1").allowed is True

    def test_zero_requests_per_minute_disables_limiting(self):
        limiter = DiscordUserRateLimiter(requests_per_minute=0)
        for _ in range(100):
            assert limiter.check("user-1").allowed is True
        # No bucket should even be created when disabled.
        assert limiter.tracked_user_count == 0

    def test_negative_requests_per_minute_treated_as_disabled(self):
        limiter = DiscordUserRateLimiter(requests_per_minute=-5)
        assert limiter.check("user-1").allowed is True

    def test_empty_user_id_always_allowed(self):
        limiter = DiscordUserRateLimiter(requests_per_minute=1)
        assert limiter.check("").allowed is True
        assert limiter.check("").allowed is True  # never consumes a real bucket

    def test_different_users_have_independent_buckets(self):
        limiter = DiscordUserRateLimiter(requests_per_minute=1)
        assert limiter.check("user-a").allowed is True
        assert limiter.check("user-a").allowed is False
        # user-b's budget is untouched by user-a's usage.
        assert limiter.check("user-b").allowed is True
        assert limiter.tracked_user_count == 2

    def test_idle_buckets_are_evicted(self):
        limiter = DiscordUserRateLimiter(requests_per_minute=1)
        limiter.check("user-old")
        assert limiter.tracked_user_count == 1
        # Push the bucket's last_touched far into the past to simulate
        # a long-idle user, then trigger eviction via a new check() for
        # a different user (eviction runs on every check()).
        bucket = limiter._buckets["user-old"]
        bucket.last_touched = time.monotonic() - 7200.0
        limiter.check("user-new")
        assert "user-old" not in limiter._buckets
        assert limiter.tracked_user_count == 1  # only user-new remains

    def test_recently_touched_buckets_are_not_evicted(self):
        limiter = DiscordUserRateLimiter(requests_per_minute=1)
        limiter.check("user-1")
        limiter.check("user-2")
        assert limiter.tracked_user_count == 2
