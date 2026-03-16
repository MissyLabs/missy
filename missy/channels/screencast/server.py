"""Combined HTTP + WebSocket server for the screencast channel.

Serves the static capture page over HTTP and handles authenticated binary
frame streaming over WebSocket, all on a single port.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import ssl
import time
from pathlib import Path
from typing import Any

import websockets.exceptions

try:
    from websockets.asyncio.server import ServerConnection as WebSocketServerProtocol
    from websockets.asyncio.server import serve as _ws_serve
except ImportError:  # websockets < 13
    from websockets import serve as _ws_serve  # type: ignore[assignment]
    from websockets.server import WebSocketServerProtocol  # type: ignore[assignment]

from missy.channels.screencast.auth import ScreencastTokenRegistry
from missy.channels.screencast.session_manager import (
    FrameMetadata,
    SessionManager,
    SessionState,
)
from missy.core.events import AuditEvent, event_bus

logger = logging.getLogger(__name__)

# Maximum frame payload (5 MB).
_MAX_FRAME_BYTES = 5 * 1024 * 1024

# Maximum concurrent WebSocket connections.
_MAX_CONCURRENT_CONNECTIONS = 20

# Seconds to wait for the first auth frame.
_AUTH_TIMEOUT_SECONDS = 10.0

# Minimum interval between frames from a single session (seconds).
_MIN_FRAME_INTERVAL = 2.0

# Dimension bounds for frames.
_MIN_DIMENSION = 64
_MAX_DIMENSION = 7680

# JPEG magic bytes.
_JPEG_MAGIC = b"\xff\xd8\xff"

# PNG magic bytes.
_PNG_MAGIC = b"\x89PNG\r\n\x1a\n"

# Audit task ID.
_TASK_ID = "screencast-server"

# Path to the capture.html file.
_CAPTURE_HTML_PATH = Path(__file__).parent / "web" / "capture.html"

# Security headers for the capture page.
_SECURITY_HEADERS = {
    "X-Content-Type-Options": "nosniff",
    "X-Frame-Options": "DENY",
    "Cache-Control": "no-store",
    "Content-Security-Policy": (
        "default-src 'none'; "
        "script-src 'unsafe-inline'; "
        "style-src 'unsafe-inline'; "
        "connect-src 'self' ws: wss:; "
        "img-src 'self' blob:; "
        "media-src 'self' blob:"
    ),
}


def _emit(
    session_id: str,
    event_type: str,
    result: str,
    detail: dict[str, Any] | None = None,
) -> None:
    """Publish a screencast-server audit event."""
    try:
        event_bus.publish(
            AuditEvent.now(
                session_id=session_id,
                task_id=_TASK_ID,
                event_type=event_type,
                category="plugin",
                result=result,  # type: ignore[arg-type]
                detail=detail or {},
            )
        )
    except Exception:
        logger.debug("screencast server: audit emit failed for %r", event_type, exc_info=True)


class ScreencastServer:
    """Combined HTTP + WebSocket server for screencast frame streaming.

    HTTP routes:
        ``GET /`` — serve ``capture.html``
        ``GET /status`` — JSON server status

    WebSocket protocol:
        See module docstring in ``__init__.py`` for the full protocol spec.

    Args:
        token_registry: Authentication registry for session tokens.
        session_manager: Multi-session state and frame queue manager.
        host: Interface to bind on.
        port: TCP port to listen on.
    """

    def __init__(
        self,
        token_registry: ScreencastTokenRegistry,
        session_manager: SessionManager,
        host: str = "127.0.0.1",
        port: int = 8780,
        tls_cert_dir: str = "",
    ) -> None:
        self._registry = token_registry
        self._sessions = session_manager
        self._host = host
        self._port = port
        self._tls_cert_dir = tls_cert_dir or str(
            Path("~/.missy/secrets").expanduser()
        )
        self._running = False
        self._ws_server: Any | None = None
        self._active_connections: int = 0
        self._capture_html: str | None = None

    async def start(self) -> None:
        """Start accepting connections."""
        if self._running:
            return

        if self._host == "0.0.0.0":
            logger.warning(
                "ScreencastServer: binding to 0.0.0.0 exposes the server on all interfaces."
            )
            _emit(
                "system",
                "screencast.bind.warning",
                "allow",
                {"host": self._host, "port": self._port},
            )

        # Pre-load the capture HTML.
        self._capture_html = self._load_capture_html()

        # Create the frame queue on this event loop.
        queue: asyncio.Queue[tuple[FrameMetadata, bytes]] = asyncio.Queue(maxsize=50)
        self._sessions.set_queue(queue)

        # Build TLS context — getDisplayMedia() requires a secure context.
        # Localhost is already a secure context in browsers, so skip TLS there.
        if self._host in ("127.0.0.1", "::1", "localhost"):
            ssl_ctx = None
        else:
            ssl_ctx = self._get_or_create_ssl_context()
        scheme = "https" if ssl_ctx else "http"

        self._ws_server = await _ws_serve(
            self._handle_connection,
            self._host,
            self._port,
            max_size=_MAX_FRAME_BYTES + 4096,  # frame + header overhead
            process_request=self._process_http_request,
            ssl=ssl_ctx,
        )
        self._running = True
        self._tls_enabled = ssl_ctx is not None
        logger.info(
            "ScreencastServer: listening on %s://%s:%d",
            scheme,
            self._host,
            self._port,
        )

    async def stop(self) -> None:
        """Close all connections and shut down."""
        if not self._running:
            return

        self._running = False

        if self._ws_server is not None:
            self._ws_server.close()
            await self._ws_server.wait_closed()
            self._ws_server = None

        logger.info("ScreencastServer: stopped.")

    def _load_capture_html(self) -> str:
        """Load the capture.html file from disk."""
        try:
            return _CAPTURE_HTML_PATH.read_text(encoding="utf-8")
        except Exception:
            logger.error("ScreencastServer: failed to load capture.html", exc_info=True)
            return "<html><body><h1>Capture page not found</h1></body></html>"

    def _get_or_create_ssl_context(self) -> ssl.SSLContext | None:
        """Return an SSL context with a self-signed cert for HTTPS.

        ``getDisplayMedia()`` requires a secure context — browsers refuse it
        on plain ``http://`` for non-localhost origins.  A self-signed cert
        triggers a one-time browser warning but satisfies the requirement.

        Cert/key are stored at ``{tls_cert_dir}/screencast.{crt,key}`` and
        reused across restarts.
        """
        cert_dir = Path(self._tls_cert_dir)
        cert_path = cert_dir / "screencast.crt"
        key_path = cert_dir / "screencast.key"

        # Generate if missing.
        if not cert_path.exists() or not key_path.exists():
            try:
                self._generate_self_signed_cert(cert_path, key_path)
            except Exception:
                logger.warning(
                    "ScreencastServer: could not generate TLS cert — falling back to HTTP. "
                    "getDisplayMedia() will NOT work from non-localhost origins.",
                    exc_info=True,
                )
                return None

        try:
            ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
            ctx.load_cert_chain(str(cert_path), str(key_path))
            logger.info("ScreencastServer: TLS enabled with %s", cert_path)
            return ctx
        except Exception:
            logger.warning(
                "ScreencastServer: TLS cert load failed — falling back to HTTP.",
                exc_info=True,
            )
            return None

    @staticmethod
    def _generate_self_signed_cert(cert_path: Path, key_path: Path) -> None:
        """Generate a self-signed cert+key pair using the cryptography library."""
        import datetime
        import ipaddress

        from cryptography import x509
        from cryptography.hazmat.primitives import hashes, serialization
        from cryptography.hazmat.primitives.asymmetric import ec
        from cryptography.x509.oid import NameOID

        # Generate EC key (fast, small).
        key = ec.generate_private_key(ec.SECP256R1())

        subject = issuer = x509.Name([
            x509.NameAttribute(NameOID.COMMON_NAME, "Missy Screencast"),
            x509.NameAttribute(NameOID.ORGANIZATION_NAME, "Missy"),
        ])

        # SANs: localhost + common LAN ranges.
        san_entries: list[x509.GeneralName] = [
            x509.DNSName("localhost"),
        ]
        # Add the machine's LAN IP.
        try:
            import socket

            with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
                s.connect(("10.255.255.255", 1))
                lan_ip = s.getsockname()[0]
            san_entries.append(x509.IPAddress(ipaddress.ip_address(lan_ip)))
        except Exception:
            pass
        san_entries.append(x509.IPAddress(ipaddress.ip_address("127.0.0.1")))

        cert = (
            x509.CertificateBuilder()
            .subject_name(subject)
            .issuer_name(issuer)
            .public_key(key.public_key())
            .serial_number(x509.random_serial_number())
            .not_valid_before(datetime.datetime.now(datetime.UTC))
            .not_valid_after(
                datetime.datetime.now(datetime.UTC)
                + datetime.timedelta(days=3650)
            )
            .add_extension(
                x509.SubjectAlternativeName(san_entries),
                critical=False,
            )
            .sign(key, hashes.SHA256())
        )

        cert_path.parent.mkdir(parents=True, exist_ok=True)

        key_path.write_bytes(
            key.private_bytes(
                serialization.Encoding.PEM,
                serialization.PrivateFormat.PKCS8,
                serialization.NoEncryption(),
            )
        )
        key_path.chmod(0o600)

        cert_path.write_bytes(cert.public_bytes(serialization.Encoding.PEM))
        cert_path.chmod(0o644)

        logger.info(
            "ScreencastServer: generated self-signed TLS cert at %s",
            cert_path,
        )

    def _process_http_request(
        self,
        connection: Any,
        request: Any,
    ) -> Any:
        """Handle plain HTTP requests before WebSocket upgrade.

        In websockets >= 13 (asyncio API), ``process_request`` receives
        ``(connection, request)`` and should return a ``Response`` to short-
        circuit the handshake, or ``None`` to proceed with WebSocket upgrade.
        """
        from websockets.datastructures import Headers as WsHeaders
        from websockets.http11 import Response as WsResponse

        # Extract the path from the request object.
        path = getattr(request, "path", "/")
        clean_path = str(path).split("?")[0].rstrip("/") or "/"

        # Check for Upgrade header — if present, let WebSocket handle it.
        req_headers = getattr(request, "headers", {})
        upgrade = ""
        if hasattr(req_headers, "get"):
            upgrade = req_headers.get("Upgrade", "") or ""
        if upgrade.lower() == "websocket":
            return None

        headers = WsHeaders(list(_SECURITY_HEADERS.items()))

        if clean_path == "/":
            html = self._capture_html or ""
            body = html.encode("utf-8")
            headers["Content-Type"] = "text/html; charset=utf-8"
            headers["Content-Length"] = str(len(body))
            return WsResponse(200, "OK", headers, body)

        if clean_path == "/status":
            status = self.get_status()
            body = json.dumps(status).encode("utf-8")
            headers["Content-Type"] = "application/json"
            headers["Content-Length"] = str(len(body))
            return WsResponse(200, "OK", headers, body)

        headers["Content-Type"] = "text/plain"
        return WsResponse(404, "Not Found", headers, b"Not Found")

    async def _handle_connection(self, websocket: WebSocketServerProtocol) -> None:
        """Handle a single WebSocket connection lifecycle."""
        remote_addr = websocket.remote_address if hasattr(websocket, "remote_address") else ("?", 0)
        logger.debug("ScreencastServer: new connection from %s", remote_addr)

        # Connection flood protection.
        if self._active_connections >= _MAX_CONCURRENT_CONNECTIONS:
            logger.warning(
                "ScreencastServer: connection limit reached (%d) — rejecting %s",
                _MAX_CONCURRENT_CONNECTIONS,
                remote_addr,
            )
            _emit(
                "system",
                "screencast.connection.rejected_capacity",
                "deny",
                {"remote": str(remote_addr), "limit": _MAX_CONCURRENT_CONNECTIONS},
            )
            await websocket.close(1013, "Server at capacity")
            return

        self._active_connections += 1
        session_id: str | None = None

        try:
            # First frame must be auth.
            try:
                raw_first = await asyncio.wait_for(
                    websocket.recv(),
                    timeout=_AUTH_TIMEOUT_SECONDS,
                )
            except TimeoutError:
                _emit("system", "screencast.connection.auth_fail", "deny", {"reason": "timeout"})
                await websocket.close(1008, "Authentication timeout")
                return

            if isinstance(raw_first, bytes):
                await websocket.close(1008, "First frame must be JSON auth")
                return

            try:
                first_msg: dict[str, Any] = json.loads(raw_first)
            except json.JSONDecodeError:
                await websocket.close(1008, "Malformed JSON")
                return

            msg_type = first_msg.get("type", "")
            if msg_type != "auth":
                await self._send_json(
                    websocket,
                    {"type": "error", "message": "First message must be auth"},
                )
                await websocket.close(1008, "Protocol violation")
                return

            # Authenticate.
            session_id = str(first_msg.get("session_id", ""))
            token = str(first_msg.get("token", ""))

            if not session_id or not token:
                await self._send_json(
                    websocket,
                    {"type": "auth_fail", "reason": "Missing session_id or token"},
                )
                _emit("system", "screencast.connection.auth_fail", "deny", {"reason": "missing_fields"})
                await websocket.close(1008, "Missing credentials")
                return

            if not self._registry.verify_token(session_id, token):
                await self._send_json(
                    websocket,
                    {"type": "auth_fail", "reason": "Invalid session or token"},
                )
                _emit(
                    session_id,
                    "screencast.connection.auth_fail",
                    "deny",
                    {"reason": "invalid_token"},
                )
                await websocket.close(1008, "Authentication failed")
                return

            # Check session capacity.
            if self._sessions.at_capacity:
                await self._send_json(
                    websocket,
                    {"type": "auth_fail", "reason": "Too many active sessions"},
                )
                _emit(session_id, "screencast.connection.rejected_capacity", "deny")
                await websocket.close(1013, "Session limit reached")
                return

            # Auth OK — register connection.
            remote_str = f"{remote_addr[0]}:{remote_addr[1]}" if isinstance(remote_addr, tuple) else str(remote_addr)
            state = self._sessions.register_connection(session_id, remote_str)
            await self._send_json(websocket, {"type": "auth_ok", "session_id": session_id})

            _emit(
                session_id,
                "screencast.connection.auth_ok",
                "allow",
                {"remote": remote_str},
            )
            logger.info("ScreencastServer: authenticated session %s from %s", session_id, remote_str)

            try:
                await self._message_loop(websocket, session_id, state)
            finally:
                self._sessions.unregister_connection(session_id)
                _emit(session_id, "screencast.connection.closed", "allow")
                logger.info("ScreencastServer: session %s disconnected", session_id)

        except websockets.exceptions.ConnectionClosed:
            pass
        except Exception:
            logger.error(
                "ScreencastServer: unexpected error for %s",
                remote_addr,
                exc_info=True,
            )
        finally:
            self._active_connections = max(0, self._active_connections - 1)

    async def _message_loop(
        self,
        websocket: WebSocketServerProtocol,
        session_id: str,
        state: SessionState,
    ) -> None:
        """Dispatch incoming messages for an authenticated session."""
        last_frame_time: float = 0.0
        pending_meta: FrameMetadata | None = None

        async for raw in websocket:
            if not self._running:
                break

            if isinstance(raw, bytes):
                # Binary frame — this is the frame data following a "frame" JSON message.
                if pending_meta is None:
                    logger.debug(
                        "ScreencastServer: unexpected binary frame from %s — ignoring.",
                        session_id,
                    )
                    continue

                meta = pending_meta
                pending_meta = None

                # Validate frame size.
                if len(raw) > _MAX_FRAME_BYTES:
                    _emit(
                        session_id,
                        "screencast.frame.rejected",
                        "deny",
                        {"reason": "too_large", "size": len(raw)},
                    )
                    await self._send_json(
                        websocket,
                        {"type": "error", "message": f"Frame too large ({len(raw)} bytes)"},
                    )
                    continue

                # Validate magic bytes.
                if not self._validate_image_magic(raw):
                    _emit(
                        session_id,
                        "screencast.frame.rejected",
                        "deny",
                        {"reason": "invalid_magic"},
                    )
                    await self._send_json(
                        websocket,
                        {"type": "error", "message": "Invalid image data (bad magic bytes)"},
                    )
                    continue

                # Rate limit: minimum interval between frames.
                now = time.monotonic()
                if now - last_frame_time < _MIN_FRAME_INTERVAL:
                    continue  # Silently drop.
                last_frame_time = now

                meta.size_bytes = len(raw)
                state.frame_count += 1

                # Update the registry stats.
                self._registry.update_frame_stats(
                    session_id,
                    frame_count=state.frame_count,
                )

                _emit(
                    session_id,
                    "screencast.frame.received",
                    "allow",
                    {
                        "frame_number": meta.frame_number,
                        "size_bytes": meta.size_bytes,
                        "format": meta.format,
                    },
                )

                # Enqueue for analysis.
                if not self._sessions.enqueue_frame(meta, raw):
                    await self._send_json(websocket, {"type": "backpressure"})
                    logger.debug(
                        "ScreencastServer: backpressure for session %s",
                        session_id,
                    )

                continue

            # JSON text frame.
            try:
                msg: dict[str, Any] = json.loads(raw)
            except json.JSONDecodeError:
                await self._send_json(websocket, {"type": "error", "message": "Malformed JSON"})
                continue

            msg_type = msg.get("type", "")

            if msg_type == "frame":
                # Prepare metadata; binary data follows in the next frame.
                fmt = str(msg.get("format", "jpeg")).lower()
                if fmt not in ("jpeg", "png"):
                    fmt = "jpeg"

                width = int(msg.get("width", 0))
                height = int(msg.get("height", 0))
                seq = int(msg.get("seq", state.frame_count + 1))

                # Validate dimensions.
                if width and height and not (_MIN_DIMENSION <= width <= _MAX_DIMENSION and _MIN_DIMENSION <= height <= _MAX_DIMENSION):
                    await self._send_json(
                        websocket,
                        {"type": "error", "message": f"Invalid dimensions: {width}x{height}"},
                    )
                    continue

                pending_meta = FrameMetadata(
                    session_id=session_id,
                    frame_number=seq,
                    format=fmt,
                    width=width,
                    height=height,
                )

            elif msg_type == "heartbeat":
                # No-op acknowledgement — keeps the connection alive.
                pass

            elif msg_type == "config":
                # Client requesting a capture interval change.
                interval_ms = int(msg.get("interval_ms", state.capture_interval_ms))
                interval_ms = max(2000, min(interval_ms, 300000))  # 2s–5min
                state.capture_interval_ms = interval_ms
                logger.debug(
                    "ScreencastServer: session %s capture interval set to %dms",
                    session_id,
                    interval_ms,
                )

            else:
                await self._send_json(
                    websocket,
                    {"type": "error", "message": f"Unknown message type: {msg_type}"},
                )

    @staticmethod
    def _validate_image_magic(data: bytes) -> bool:
        """Check that the data starts with JPEG or PNG magic bytes."""
        if len(data) < 8:
            return False
        return data[:3] == _JPEG_MAGIC or data[:8] == _PNG_MAGIC

    @staticmethod
    async def _send_json(websocket: WebSocketServerProtocol, data: dict[str, Any]) -> None:
        """Send a JSON message to the client."""
        with contextlib.suppress(Exception):
            await websocket.send(json.dumps(data))

    async def send_analysis(self, session_id: str, seq: int, text: str) -> None:
        """Send an analysis result back to a connected client.

        Called by the analyzer when results are ready.
        """
        state = self._sessions.get_connection(session_id)
        if state is None:
            return
        # We don't have a direct reference to the websocket here.
        # Analysis results are stored in the session manager and the
        # client can poll or we can push via a separate mechanism.
        # For now, results are delivered via the Discord callback.

    def get_status(self) -> dict[str, Any]:
        """Return server operational status."""
        return {
            "running": self._running,
            "host": self._host,
            "port": self._port,
            "active_connections": self._active_connections,
            "sessions": self._sessions.get_status(),
        }
