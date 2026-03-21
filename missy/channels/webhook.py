"""Webhook channel: receive agent tasks via HTTP POST."""

from __future__ import annotations

import hashlib
import hmac
import json
import logging
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer

from missy.channels.base import BaseChannel, ChannelMessage

logger = logging.getLogger(__name__)

# Maximum request body size (1 MB).
_MAX_PAYLOAD_BYTES = 1024 * 1024
# Maximum prompt length (characters) to prevent excessive token usage.
_MAX_PROMPT_LENGTH = 32_000
# Maximum queued messages before rejecting new ones.
_MAX_QUEUE_SIZE = 1000
# Rate limit: max requests per IP per window.
_RATE_LIMIT_REQUESTS = 60
_RATE_LIMIT_WINDOW = 60  # seconds
# Maximum tracked IPs before evicting the oldest entries.
_MAX_TRACKED_IPS = 10_000


class WebhookChannel(BaseChannel):
    """HTTP webhook channel that queues inbound POST requests as agent tasks.

    Listens on a local HTTP port. Each POST to / with a JSON body
    ``{"prompt": "..."}`` creates a ChannelMessage.

    Args:
        host: Bind address (default 127.0.0.1).
        port: Bind port (default 9090).
        secret: Optional HMAC-SHA256 shared secret for request validation.
        trust_proxy: When ``True``, use ``X-Forwarded-For`` header for
            rate-limiting IP resolution.  Only enable behind a trusted
            reverse proxy.
    """

    name = "webhook"

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 9090,
        secret: str = "",
        trust_proxy: bool = False,
    ):
        self._host = host
        self._port = port
        self._secret = secret.encode() if secret else b""
        self._trust_proxy = trust_proxy
        self._queue: list[ChannelMessage] = []
        self._lock = threading.Lock()
        self._server: HTTPServer | None = None
        self._thread: threading.Thread | None = None
        # Per-IP rate tracking: {ip: [timestamps]}
        self._rate_tracker: dict[str, list[float]] = {}
        self._rate_lock = threading.Lock()

    def _check_rate_limit(self, client_ip: str) -> bool:
        """Return True if the request is within rate limits."""
        now = time.monotonic()
        cutoff = now - _RATE_LIMIT_WINDOW
        with self._rate_lock:
            timestamps = self._rate_tracker.get(client_ip, [])
            timestamps = [t for t in timestamps if t > cutoff]
            if len(timestamps) >= _RATE_LIMIT_REQUESTS:
                self._rate_tracker[client_ip] = timestamps
                return False
            timestamps.append(now)
            self._rate_tracker[client_ip] = timestamps

            # Evict stale IPs to prevent unbounded memory growth.
            if len(self._rate_tracker) > _MAX_TRACKED_IPS:
                self._evict_stale_ips(cutoff)
            return True

    def _evict_stale_ips(self, cutoff: float) -> None:
        """Remove IPs whose timestamps have all expired.

        Must be called while holding ``_rate_lock``.
        """
        stale = [
            ip
            for ip, ts_list in self._rate_tracker.items()
            if not ts_list or all(t <= cutoff for t in ts_list)
        ]
        for ip in stale:
            del self._rate_tracker[ip]

    def start(self) -> None:
        if self._trust_proxy and self._host != "127.0.0.1":
            logger.warning(
                "WebhookChannel: trust_proxy=True on non-loopback bind (%s) — "
                "X-Forwarded-For can be spoofed if not behind a trusted proxy.",
                self._host,
            )
        if not self._secret and self._host != "127.0.0.1":
            logger.warning(
                "WebhookChannel: no HMAC secret configured on non-loopback bind (%s) — "
                "webhook accepts unauthenticated requests from any client that can reach this port. "
                "Set 'secret' to enable request signing.",
                self._host,
            )
        channel_ref = self

        class Handler(BaseHTTPRequestHandler):
            server_version = "missy"
            sys_version = ""

            def version_string(self) -> str:
                return "missy"

            def _send_security_headers(self) -> None:
                self.send_header("X-Content-Type-Options", "nosniff")
                self.send_header("X-Frame-Options", "DENY")
                self.send_header("Cache-Control", "no-store")

            def do_GET(self) -> None:
                self.send_response(405)
                self._send_security_headers()
                self.end_headers()

            do_PUT = do_GET
            do_DELETE = do_GET
            do_PATCH = do_GET

            def log_message(self, format: str, *args: object) -> None:
                logger.debug("Webhook: " + format, *args)

            def _get_client_ip(self) -> str:
                """Return the client IP, respecting X-Forwarded-For when behind a trusted proxy."""
                forwarded = self.headers.get("X-Forwarded-For")
                if forwarded and channel_ref._trust_proxy:
                    # Take the leftmost (client-closest) IP.
                    return forwarded.split(",")[0].strip()
                return self.client_address[0]

            def do_POST(self) -> None:
                # Rate limiting
                client_ip = self._get_client_ip()
                if not channel_ref._check_rate_limit(client_ip):
                    self.send_response(429)
                    self.send_header("Retry-After", str(_RATE_LIMIT_WINDOW))
                    self._send_security_headers()
                    self.end_headers()
                    return

                # Require JSON Content-Type to prevent CSRF via form submissions.
                content_type = (self.headers.get("Content-Type") or "").split(";")[0].strip()
                if content_type != "application/json":
                    self.send_response(415)
                    self._send_security_headers()
                    self.end_headers()
                    return

                # Parse Content-Length safely (reject non-integer / negative).
                try:
                    length = int(self.headers.get("Content-Length", 0))
                    if length < 0:
                        raise ValueError("negative")
                except (ValueError, TypeError):
                    self.send_response(400)
                    self._send_security_headers()
                    self.end_headers()
                    return

                # Reject oversized payloads
                if length > _MAX_PAYLOAD_BYTES:
                    self.send_response(413)
                    self._send_security_headers()
                    self.end_headers()
                    return

                body = self.rfile.read(length)

                # Validate HMAC if secret configured
                if channel_ref._secret:
                    sig = self.headers.get("X-Missy-Signature", "")
                    expected = (
                        "sha256=" + hmac.new(channel_ref._secret, body, hashlib.sha256).hexdigest()
                    )
                    if not hmac.compare_digest(sig, expected):
                        self.send_response(401)
                        self._send_security_headers()
                        self.end_headers()
                        return

                try:
                    data = json.loads(body)
                except json.JSONDecodeError:
                    self.send_response(400)
                    self._send_security_headers()
                    self.end_headers()
                    return

                prompt = (data.get("prompt") or "").strip()
                if not prompt:
                    self.send_response(400)
                    self._send_security_headers()
                    self.end_headers()
                    return

                if len(prompt) > _MAX_PROMPT_LENGTH:
                    self.send_response(413)
                    self._send_security_headers()
                    self.end_headers()
                    return

                # Validate sender: cap length, strip control characters
                raw_sender = str(data.get("sender", "webhook"))[:64]
                safe_sender = (
                    "".join(c for c in raw_sender if c.isalnum() or c in "-_. @") or "webhook"
                )

                msg = ChannelMessage(
                    content=prompt,
                    sender=safe_sender,
                    channel="webhook",
                    metadata={
                        "webhook_headers": {
                            k: v
                            for k, v in self.headers.items()
                            if k.lower()
                            in (
                                "content-type",
                                "user-agent",
                                "x-request-id",
                                "x-missy-signature",
                            )
                        },
                    },
                )
                with channel_ref._lock:
                    if len(channel_ref._queue) >= _MAX_QUEUE_SIZE:
                        self.send_response(503)
                        self._send_security_headers()
                        self.end_headers()
                        return
                    channel_ref._queue.append(msg)

                self.send_response(202)
                self.send_header("Content-Type", "application/json")
                self._send_security_headers()
                self.end_headers()
                self.wfile.write(b'{"status": "queued"}')

        self._server = HTTPServer((self._host, self._port), Handler)
        self._thread = threading.Thread(
            target=self._server.serve_forever, daemon=True, name="missy-webhook"
        )
        self._thread.start()
        logger.info("Webhook channel listening on %s:%d", self._host, self._port)

    def stop(self) -> None:
        if self._server:
            self._server.shutdown()

    def receive(self) -> ChannelMessage | None:
        with self._lock:
            return self._queue.pop(0) if self._queue else None

    def send(self, message: str) -> None:
        logger.debug("Webhook response: %d chars", len(message))
