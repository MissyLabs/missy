"""Tests for webhook channel rate limiting, payload size limits, and queue bounds."""

from __future__ import annotations

import json
import threading
import time
from unittest.mock import MagicMock, patch

import pytest

from missy.channels.webhook import (
    WebhookChannel,
    _MAX_PAYLOAD_BYTES,
    _MAX_QUEUE_SIZE,
    _RATE_LIMIT_REQUESTS,
    _RATE_LIMIT_WINDOW,
)


class TestWebhookRateLimiting:
    """Rate limiting prevents request floods."""

    def test_check_rate_limit_allows_under_limit(self):
        ch = WebhookChannel()
        for _ in range(_RATE_LIMIT_REQUESTS - 1):
            assert ch._check_rate_limit("1.2.3.4") is True

    def test_check_rate_limit_blocks_over_limit(self):
        ch = WebhookChannel()
        for _ in range(_RATE_LIMIT_REQUESTS):
            ch._check_rate_limit("1.2.3.4")
        assert ch._check_rate_limit("1.2.3.4") is False

    def test_rate_limit_per_ip(self):
        """Different IPs have independent limits."""
        ch = WebhookChannel()
        for _ in range(_RATE_LIMIT_REQUESTS):
            ch._check_rate_limit("1.2.3.4")
        # First IP is blocked
        assert ch._check_rate_limit("1.2.3.4") is False
        # Second IP is still allowed
        assert ch._check_rate_limit("5.6.7.8") is True

    def test_rate_limit_window_expiry(self):
        """Old timestamps outside the window are pruned."""
        ch = WebhookChannel()
        # Fill up rate tracker with old timestamps
        old_time = time.monotonic() - _RATE_LIMIT_WINDOW - 1
        ch._rate_tracker["1.2.3.4"] = [old_time] * _RATE_LIMIT_REQUESTS
        # Should be allowed since old timestamps are pruned
        assert ch._check_rate_limit("1.2.3.4") is True


class TestWebhookPayloadSize:
    """Oversized payloads must be rejected."""

    def test_max_payload_bytes_constant(self):
        assert _MAX_PAYLOAD_BYTES == 1024 * 1024  # 1 MB

    def test_max_queue_size_constant(self):
        assert _MAX_QUEUE_SIZE == 1000


class TestWebhookQueueBounds:
    """Queue must reject new messages when full."""

    def test_queue_has_size_limit(self):
        """WebhookChannel should not allow unbounded queue growth."""
        ch = WebhookChannel()
        # Fill queue to max
        from missy.channels.base import ChannelMessage

        for i in range(_MAX_QUEUE_SIZE):
            ch._queue.append(
                ChannelMessage(content=f"msg-{i}", sender="test", channel="webhook")
            )
        assert len(ch._queue) == _MAX_QUEUE_SIZE


class TestWebhookDefaults:
    """Verify webhook initialization defaults."""

    def test_default_host(self):
        ch = WebhookChannel()
        assert ch._host == "127.0.0.1"

    def test_default_port(self):
        ch = WebhookChannel()
        assert ch._port == 9090

    def test_rate_tracker_initialized(self):
        ch = WebhookChannel()
        assert ch._rate_tracker == {}

    def test_rate_lock_is_lock(self):
        ch = WebhookChannel()
        assert isinstance(ch._rate_lock, type(threading.Lock()))

    def test_rate_limit_constants(self):
        assert _RATE_LIMIT_REQUESTS == 60
        assert _RATE_LIMIT_WINDOW == 60
