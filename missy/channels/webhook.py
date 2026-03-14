"""Webhook channel: receive agent tasks via HTTP POST."""
from __future__ import annotations

import hashlib
import hmac
import json
import logging
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Optional

from missy.channels.base import BaseChannel, ChannelMessage

logger = logging.getLogger(__name__)


class WebhookChannel(BaseChannel):
    """HTTP webhook channel that queues inbound POST requests as agent tasks.

    Listens on a local HTTP port. Each POST to / with a JSON body
    ``{"prompt": "..."}`` creates a ChannelMessage.

    Args:
        host: Bind address (default 127.0.0.1).
        port: Bind port (default 9090).
        secret: Optional HMAC-SHA256 shared secret for request validation.
    """

    name = "webhook"

    def __init__(self, host: str = "127.0.0.1", port: int = 9090, secret: str = ""):
        self._host = host
        self._port = port
        self._secret = secret.encode() if secret else b""
        self._queue: list[ChannelMessage] = []
        self._lock = threading.Lock()
        self._server: Optional[HTTPServer] = None
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        channel_ref = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, format, *args):
                logger.debug("Webhook: " + format, *args)

            def do_POST(self):
                length = int(self.headers.get("Content-Length", 0))
                body = self.rfile.read(length)

                # Validate HMAC if secret configured
                if channel_ref._secret:
                    sig = self.headers.get("X-Missy-Signature", "")
                    expected = "sha256=" + hmac.new(channel_ref._secret, body, hashlib.sha256).hexdigest()
                    if not hmac.compare_digest(sig, expected):
                        self.send_response(401)
                        self.end_headers()
                        return

                try:
                    data = json.loads(body)
                except json.JSONDecodeError:
                    self.send_response(400)
                    self.end_headers()
                    return

                prompt = data.get("prompt", "").strip()
                if not prompt:
                    self.send_response(400)
                    self.end_headers()
                    return

                msg = ChannelMessage(
                    content=prompt,
                    sender=data.get("sender", "webhook"),
                    channel="webhook",
                    metadata={"webhook_headers": dict(self.headers)},
                )
                with channel_ref._lock:
                    channel_ref._queue.append(msg)

                self.send_response(202)
                self.send_header("Content-Type", "application/json")
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

    def receive(self) -> Optional[ChannelMessage]:
        with self._lock:
            return self._queue.pop(0) if self._queue else None

    def send(self, message: str) -> None:
        logger.info("Webhook response: %s", message[:200])
