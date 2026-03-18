"""Integration tests for WebhookChannel security features.

Starts a real HTTP server on a random high port and sends actual HTTP
requests via http.client to verify end-to-end behaviour of:
  - Rate limiting (429 after exceeding per-IP limit)
  - Oversized payload rejection (413)
  - Queue overflow rejection (503)
  - Normal POST accepted (202)
"""

from __future__ import annotations

import http.client
import json
import socket
import time
import unittest.mock
from contextlib import contextmanager

import missy.channels.webhook as webhook_module
from missy.channels.base import ChannelMessage
from missy.channels.webhook import WebhookChannel

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _free_port() -> int:
    """Return an available TCP port on loopback."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@contextmanager
def _running_channel(**kwargs):
    """Start a WebhookChannel, yield it, then stop it."""
    ch = WebhookChannel(**kwargs)
    ch.start()
    # Give the server thread a moment to bind and start accepting.
    time.sleep(0.05)
    try:
        yield ch
    finally:
        ch.stop()


def _post(port: int, body: bytes, headers: dict | None = None) -> http.client.HTTPResponse:
    """Send a POST to / on 127.0.0.1:port with the given raw body bytes."""
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
    h = {"Content-Type": "application/json", "Content-Length": str(len(body))}
    if headers:
        h.update(headers)
    conn.request("POST", "/", body=body, headers=h)
    return conn.getresponse()


def _json_post(port: int, data: dict) -> http.client.HTTPResponse:
    """POST a JSON-serialisable dict and return the response."""
    return _post(port, json.dumps(data).encode())


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestNormalPost:
    """A well-formed POST must be accepted and queued."""

    def test_valid_post_returns_202(self):
        port = _free_port()
        with _running_channel(port=port):
            resp = _json_post(port, {"prompt": "hello world"})
            assert resp.status == 202

    def test_valid_post_response_body_is_json_queued(self):
        port = _free_port()
        with _running_channel(port=port):
            resp = _json_post(port, {"prompt": "hello world"})
            body = json.loads(resp.read())
            assert body == {"status": "queued"}

    def test_valid_post_message_queued(self):
        port = _free_port()
        with _running_channel(port=port) as ch:
            _json_post(port, {"prompt": "integration test"})
            msg = ch.receive()
        assert msg is not None
        assert msg.content == "integration test"
        assert msg.channel == "webhook"

    def test_valid_post_custom_sender_preserved(self):
        port = _free_port()
        with _running_channel(port=port) as ch:
            _json_post(port, {"prompt": "task", "sender": "ci-bot"})
            msg = ch.receive()
        assert msg is not None
        assert msg.sender == "ci-bot"

    def test_missing_prompt_returns_400(self):
        port = _free_port()
        with _running_channel(port=port):
            resp = _json_post(port, {"not_a_prompt": "value"})
            assert resp.status == 400

    def test_empty_prompt_returns_400(self):
        port = _free_port()
        with _running_channel(port=port):
            resp = _json_post(port, {"prompt": "   "})
            assert resp.status == 400

    def test_malformed_json_returns_400(self):
        port = _free_port()
        with _running_channel(port=port):
            resp = _post(port, b"{not valid json}")
            assert resp.status == 400


class TestOversizedPayload:
    """Content-Length exceeding the limit must be rejected with 413."""

    def test_oversized_content_length_returns_413(self):
        port = _free_port()
        with _running_channel(port=port):
            # Send a real small body but lie about Content-Length to exceed
            # the limit. The server checks the header before reading.
            large_length = webhook_module._MAX_PAYLOAD_BYTES + 1
            conn = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
            # Send a tiny body; server rejects on header check before reading.
            tiny_body = b'{"prompt":"x"}'
            conn.request(
                "POST",
                "/",
                body=tiny_body,
                headers={
                    "Content-Type": "application/json",
                    "Content-Length": str(large_length),
                },
            )
            resp = conn.getresponse()
            assert resp.status == 413

    def test_exactly_at_limit_is_accepted(self):
        """Content-Length equal to the limit must not be rejected as oversized."""
        port = _free_port()
        with _running_channel(port=port):
            # Build a payload whose JSON-encoded length equals _MAX_PAYLOAD_BYTES.
            # Keep prompt short (under _MAX_PROMPT_LENGTH) and pad with a
            # separate JSON field so the overall body hits the byte limit.
            limit = webhook_module._MAX_PAYLOAD_BYTES
            prefix = b'{"prompt":"hello","pad":"'
            suffix = b'"}'
            padding_len = limit - len(prefix) - len(suffix)
            body = prefix + b"x" * padding_len + suffix
            assert len(body) == limit
            resp = _post(port, body)
            # The body is valid JSON and the prompt is non-empty, so 202.
            assert resp.status == 202

    def test_one_byte_over_limit_returns_413(self):
        port = _free_port()
        with _running_channel(port=port):
            over_limit = webhook_module._MAX_PAYLOAD_BYTES + 1
            conn = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
            tiny_body = b'{"prompt":"x"}'
            conn.request(
                "POST",
                "/",
                body=tiny_body,
                headers={
                    "Content-Type": "application/json",
                    "Content-Length": str(over_limit),
                },
            )
            resp = conn.getresponse()
            assert resp.status == 413


class TestRateLimiting:
    """Sending more requests than the per-IP limit must trigger 429."""

    def test_rate_limit_triggers_429(self):
        """Send _RATE_LIMIT_REQUESTS + 1 rapid requests; the last must be 429."""
        port = _free_port()
        with _running_channel(port=port):
            limit = webhook_module._RATE_LIMIT_REQUESTS
            statuses = []
            payload = json.dumps({"prompt": "flood"}).encode()
            for _ in range(limit + 1):
                conn = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
                conn.request(
                    "POST",
                    "/",
                    body=payload,
                    headers={
                        "Content-Type": "application/json",
                        "Content-Length": str(len(payload)),
                    },
                )
                resp = conn.getresponse()
                resp.read()  # drain so the connection is reusable
                statuses.append(resp.status)

            assert 429 in statuses, (
                f"Expected at least one 429 after {limit + 1} requests, got: {set(statuses)}"
            )

    def test_rate_limit_429_has_retry_after_header(self):
        """The 429 response must include a Retry-After header."""
        port = _free_port()
        with _running_channel(port=port):
            limit = webhook_module._RATE_LIMIT_REQUESTS
            payload = json.dumps({"prompt": "flood"}).encode()
            last_resp = None
            for _ in range(limit + 1):
                conn = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
                conn.request(
                    "POST",
                    "/",
                    body=payload,
                    headers={
                        "Content-Type": "application/json",
                        "Content-Length": str(len(payload)),
                    },
                )
                last_resp = conn.getresponse()
                last_resp.read()

            # The final response should be 429 with Retry-After.
            assert last_resp is not None
            assert last_resp.status == 429
            assert last_resp.getheader("Retry-After") is not None

    def test_different_ips_are_rate_limited_independently(self):
        """Rate limiting must be per-IP; a second IP is not affected by the first."""
        port = _free_port()
        with _running_channel(port=port) as ch:
            limit = webhook_module._RATE_LIMIT_REQUESTS

            # Exhaust the limit for 10.0.0.1 by directly manipulating the tracker.
            # (We cannot spoof the source IP over loopback, so we use the internal
            # API to pre-fill the tracker for a different IP and then verify that
            # 127.0.0.1 is still accepted.)
            import time as _time

            now = _time.monotonic()
            ch._rate_tracker["10.0.0.99"] = [now] * limit  # exhausted

            # 127.0.0.1 (our loopback address) should still be allowed.
            assert ch._check_rate_limit("127.0.0.1") is True
            # 10.0.0.99 should be blocked.
            assert ch._check_rate_limit("10.0.0.99") is False

    def test_rate_limit_with_patched_small_limit(self):
        """Using a patched-down limit makes the flood test run in milliseconds."""
        port = _free_port()
        with (
            unittest.mock.patch.object(webhook_module, "_RATE_LIMIT_REQUESTS", 3),
            _running_channel(port=port),
        ):
            payload = json.dumps({"prompt": "fast flood"}).encode()
            statuses = []
            for _ in range(5):
                conn = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
                conn.request(
                    "POST",
                    "/",
                    body=payload,
                    headers={
                        "Content-Type": "application/json",
                        "Content-Length": str(len(payload)),
                    },
                )
                resp = conn.getresponse()
                resp.read()
                statuses.append(resp.status)

            assert 429 in statuses


class TestQueueOverflow:
    """When the queue is full, new POSTs must be rejected with 503."""

    def test_full_queue_returns_503(self):
        port = _free_port()
        with _running_channel(port=port) as ch:
            # Pre-fill the queue to capacity without going through HTTP.
            for i in range(webhook_module._MAX_QUEUE_SIZE):
                ch._queue.append(
                    ChannelMessage(content=f"pre-{i}", sender="test", channel="webhook")
                )

            resp = _json_post(port, {"prompt": "one more"})
            assert resp.status == 503

    def test_full_queue_does_not_add_message(self):
        port = _free_port()
        with _running_channel(port=port) as ch:
            for i in range(webhook_module._MAX_QUEUE_SIZE):
                ch._queue.append(
                    ChannelMessage(content=f"pre-{i}", sender="test", channel="webhook")
                )

            _json_post(port, {"prompt": "overflow"})
            assert len(ch._queue) == webhook_module._MAX_QUEUE_SIZE

    def test_queue_accepts_after_draining(self):
        """After consuming messages the queue should accept new ones."""
        port = _free_port()
        with _running_channel(port=port) as ch:
            # Fill to capacity.
            for i in range(webhook_module._MAX_QUEUE_SIZE):
                ch._queue.append(
                    ChannelMessage(content=f"pre-{i}", sender="test", channel="webhook")
                )

            # Verify overflow.
            assert _json_post(port, {"prompt": "overflow"}).status == 503

            # Drain one slot.
            ch.receive()

            # Now there is room for one more.
            assert _json_post(port, {"prompt": "after drain"}).status == 202

    def test_503_with_patched_small_queue(self):
        """Patch queue size to 2 so the test needs almost no pre-fill."""
        port = _free_port()
        with (
            unittest.mock.patch.object(webhook_module, "_MAX_QUEUE_SIZE", 2),
            _running_channel(port=port),
        ):
            # First two POSTs should succeed.
            r1 = _json_post(port, {"prompt": "msg1"})
            r2 = _json_post(port, {"prompt": "msg2"})
            # Third should overflow.
            r3 = _json_post(port, {"prompt": "msg3"})

            assert r1.status == 202
            assert r2.status == 202
            assert r3.status == 503


class TestHmacSignature:
    """Requests must pass HMAC validation when a secret is configured."""

    def _sign(self, secret: str, body: bytes) -> str:
        import hashlib
        import hmac as _hmac

        return "sha256=" + _hmac.new(secret.encode(), body, hashlib.sha256).hexdigest()

    def test_valid_signature_accepted(self):
        port = _free_port()
        secret = "s3cr3t"
        with _running_channel(port=port, secret=secret):
            body = json.dumps({"prompt": "signed"}).encode()
            sig = self._sign(secret, body)
            resp = _post(port, body, headers={"X-Missy-Signature": sig})
            assert resp.status == 202

    def test_missing_signature_returns_401(self):
        port = _free_port()
        with _running_channel(port=port, secret="s3cr3t"):
            resp = _json_post(port, {"prompt": "unsigned"})
            assert resp.status == 401

    def test_wrong_signature_returns_401(self):
        port = _free_port()
        with _running_channel(port=port, secret="s3cr3t"):
            body = json.dumps({"prompt": "tampered"}).encode()
            resp = _post(port, body, headers={"X-Missy-Signature": "sha256=deadbeef"})
            assert resp.status == 401

    def test_no_secret_configured_ignores_signature_header(self):
        """When no secret is set, any (or no) signature header is accepted."""
        port = _free_port()
        with _running_channel(port=port):  # no secret
            body = json.dumps({"prompt": "no secret needed"}).encode()
            resp = _post(port, body, headers={"X-Missy-Signature": "sha256=whatever"})
            assert resp.status == 202


class TestServerLifecycle:
    """Server must start cleanly and shut down without hanging."""

    def test_stop_before_start_does_not_raise(self):
        ch = WebhookChannel(port=_free_port())
        ch.stop()  # no-op, must not raise

    def test_start_and_stop(self):
        port = _free_port()
        ch = WebhookChannel(port=port)
        ch.start()
        time.sleep(0.05)
        # Should be able to POST while running.
        resp = _json_post(port, {"prompt": "lifecycle"})
        assert resp.status == 202
        ch.stop()

    def test_receive_returns_none_on_empty_queue(self):
        ch = WebhookChannel(port=_free_port())
        assert ch.receive() is None

    def test_send_does_not_raise(self):
        ch = WebhookChannel(port=_free_port())
        ch.send("any output")  # logs only, must not raise

    def test_no_secret_on_non_loopback_logs_warning(self, caplog):
        """Starting without a secret on non-loopback should log a warning."""
        import logging

        port = _free_port()
        ch = WebhookChannel(host="0.0.0.0", port=port, secret="")
        with caplog.at_level(logging.WARNING):
            ch.start()
            time.sleep(0.05)
            ch.stop()
        assert any("no HMAC secret" in msg for msg in caplog.messages)

    def test_with_secret_no_warning(self, caplog):
        """Starting with a secret should not log the unauthenticated warning."""
        import logging

        port = _free_port()
        ch = WebhookChannel(host="0.0.0.0", port=port, secret="s3cr3t")
        with caplog.at_level(logging.WARNING):
            ch.start()
            time.sleep(0.05)
            ch.stop()
        assert not any("no HMAC secret" in msg for msg in caplog.messages)
