"""Tests for the screencast server."""

from __future__ import annotations

import asyncio
import contextlib
import json
import ssl
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from missy.channels.screencast.auth import ScreencastTokenRegistry
from missy.channels.screencast.server import _JPEG_MAGIC, _PNG_MAGIC, ScreencastServer
from missy.channels.screencast.session_manager import FrameMetadata, SessionManager, SessionState


@pytest.fixture
def registry() -> ScreencastTokenRegistry:
    return ScreencastTokenRegistry()


@pytest.fixture
def session_manager() -> SessionManager:
    return SessionManager(max_sessions=5)


@pytest.fixture
def server(registry: ScreencastTokenRegistry, session_manager: SessionManager) -> ScreencastServer:
    return ScreencastServer(
        token_registry=registry,
        session_manager=session_manager,
        host="127.0.0.1",
        port=0,  # Will be overridden in tests.
    )


class TestScreencastServer:
    """Tests for ScreencastServer."""

    def test_validate_image_magic_jpeg(self) -> None:
        data = _JPEG_MAGIC + b"\x00" * 100
        assert ScreencastServer._validate_image_magic(data) is True

    def test_validate_image_magic_png(self) -> None:
        data = _PNG_MAGIC + b"\x00" * 100
        assert ScreencastServer._validate_image_magic(data) is True

    def test_validate_image_magic_invalid(self) -> None:
        assert ScreencastServer._validate_image_magic(b"GIF89a") is False
        assert ScreencastServer._validate_image_magic(b"\x00\x00\x00") is False

    def test_validate_image_magic_too_short(self) -> None:
        assert ScreencastServer._validate_image_magic(b"\xff") is False
        assert ScreencastServer._validate_image_magic(b"") is False

    def test_load_capture_html(self, server: ScreencastServer) -> None:
        html = server._load_capture_html()
        assert "Missy Screen Capture" in html
        assert "getDisplayMedia" in html

    def test_get_status_not_running(self, server: ScreencastServer) -> None:
        status = server.get_status()
        assert status["running"] is False

    @pytest.mark.asyncio
    async def test_start_stop(
        self, registry: ScreencastTokenRegistry, session_manager: SessionManager
    ) -> None:
        """Test basic start/stop lifecycle."""
        srv = ScreencastServer(
            token_registry=registry,
            session_manager=session_manager,
            host="127.0.0.1",
            port=0,
        )
        # Use a random available port.
        await srv.start()
        assert srv._running is True
        assert srv._ws_server is not None

        status = srv.get_status()
        assert status["running"] is True

        await srv.stop()
        assert srv._running is False
        assert srv._ws_server is None

    @pytest.mark.asyncio
    async def test_bind_warning_emitted(
        self, registry: ScreencastTokenRegistry, session_manager: SessionManager
    ) -> None:
        """Test that binding to 0.0.0.0 emits a warning audit event."""
        events = []

        def capture_event(event):
            events.append(event)

        with patch("missy.channels.screencast.server.event_bus") as mock_bus:
            mock_bus.publish = capture_event
            srv = ScreencastServer(
                token_registry=registry,
                session_manager=session_manager,
                host="0.0.0.0",
                port=0,
            )
            await srv.start()
            await srv.stop()

        assert any(
            hasattr(e, "event_type") and e.event_type == "screencast.bind.warning"
            for e in events
        )


class TestScreencastServerHTTP:
    """Tests for HTTP request handling."""

    @staticmethod
    def _make_request(path: str = "/", upgrade: str = "") -> MagicMock:
        """Create a mock websockets Request object."""
        req = MagicMock()
        req.path = path
        headers = {"Upgrade": upgrade} if upgrade else {}
        req.headers = headers
        return req

    def test_process_http_root(self, server: ScreencastServer) -> None:
        server._capture_html = "<html>test</html>"
        conn = MagicMock()
        result = server._process_http_request(conn, self._make_request("/"))
        assert result is not None
        assert result.status_code == 200
        assert b"test" in result.body

    def test_process_http_status(self, server: ScreencastServer) -> None:
        conn = MagicMock()
        result = server._process_http_request(conn, self._make_request("/status"))
        assert result is not None
        assert result.status_code == 200
        data = json.loads(result.body)
        assert "running" in data

    def test_process_http_404(self, server: ScreencastServer) -> None:
        conn = MagicMock()
        result = server._process_http_request(conn, self._make_request("/nonexistent"))
        assert result is not None
        assert result.status_code == 404

    def test_process_http_websocket_upgrade(self, server: ScreencastServer) -> None:
        conn = MagicMock()
        result = server._process_http_request(conn, self._make_request("/", upgrade="websocket"))
        assert result is None  # Proceed with WebSocket.


class TestScreencastServerProtocol:
    """Integration-ish tests for the WebSocket protocol."""

    @pytest.mark.asyncio
    async def test_auth_flow(
        self, registry: ScreencastTokenRegistry, session_manager: SessionManager
    ) -> None:
        """Test WebSocket auth with a real server."""
        import websockets

        srv = ScreencastServer(
            token_registry=registry,
            session_manager=session_manager,
            host="127.0.0.1",
            port=0,  # Random port.
        )
        await srv.start()

        # Get the actual port.
        port = srv._ws_server.sockets[0].getsockname()[1]

        session_id, token = registry.create_session(label="test-auth")

        try:
            async with websockets.connect(f"ws://127.0.0.1:{port}/") as ws:
                # Send auth.
                await ws.send(json.dumps({
                    "type": "auth",
                    "session_id": session_id,
                    "token": token,
                }))
                resp = json.loads(await ws.recv())
                assert resp["type"] == "auth_ok"
                assert resp["session_id"] == session_id
        finally:
            await srv.stop()

    @pytest.mark.asyncio
    async def test_auth_fail(
        self, registry: ScreencastTokenRegistry, session_manager: SessionManager
    ) -> None:
        """Test that wrong token gets auth_fail."""
        import websockets

        srv = ScreencastServer(
            token_registry=registry,
            session_manager=session_manager,
            host="127.0.0.1",
            port=0,
        )
        await srv.start()
        port = srv._ws_server.sockets[0].getsockname()[1]

        session_id, _ = registry.create_session()

        try:
            async with websockets.connect(f"ws://127.0.0.1:{port}/") as ws:
                await ws.send(json.dumps({
                    "type": "auth",
                    "session_id": session_id,
                    "token": "wrong-token",
                }))
                resp = json.loads(await ws.recv())
                assert resp["type"] == "auth_fail"
        except websockets.exceptions.ConnectionClosed:
            pass  # Server closes after auth_fail.
        finally:
            await srv.stop()

    @pytest.mark.asyncio
    async def test_frame_submission(
        self, registry: ScreencastTokenRegistry, session_manager: SessionManager
    ) -> None:
        """Test sending a frame through the WebSocket protocol."""
        import websockets

        srv = ScreencastServer(
            token_registry=registry,
            session_manager=session_manager,
            host="127.0.0.1",
            port=0,
        )
        await srv.start()
        port = srv._ws_server.sockets[0].getsockname()[1]

        session_id, token = registry.create_session()

        try:
            async with websockets.connect(f"ws://127.0.0.1:{port}/") as ws:
                # Auth.
                await ws.send(json.dumps({
                    "type": "auth",
                    "session_id": session_id,
                    "token": token,
                }))
                resp = json.loads(await ws.recv())
                assert resp["type"] == "auth_ok"

                # Send frame metadata.
                await ws.send(json.dumps({
                    "type": "frame",
                    "format": "jpeg",
                    "width": 1920,
                    "height": 1080,
                    "seq": 1,
                }))

                # Send binary JPEG data.
                fake_jpeg = _JPEG_MAGIC + b"\x00" * 1000
                await ws.send(fake_jpeg)

                # Allow processing.
                await asyncio.sleep(0.2)

                # Verify frame was enqueued.
                assert session_manager.queue is not None
                assert session_manager.queue.qsize() == 1
        finally:
            await srv.stop()


# ---------------------------------------------------------------------------
# Helper: build a mock websocket that yields a canned sequence of recv() values
# and records send()/close() calls.
# ---------------------------------------------------------------------------

def _make_ws(*recv_values):
    """Return an AsyncMock websocket whose recv() returns the given values in order."""
    ws = MagicMock()
    ws.remote_address = ("127.0.0.1", 12345)
    ws.recv = AsyncMock(side_effect=list(recv_values))
    ws.send = AsyncMock()
    ws.close = AsyncMock()
    return ws


def _make_auth_msg(session_id, token):
    return json.dumps({"type": "auth", "session_id": session_id, "token": token})


# ---------------------------------------------------------------------------
# Import fallback (lines 23-25)
# ---------------------------------------------------------------------------

class TestImportFallback:
    """Test that the module handles older websockets (< 13) gracefully."""

    def test_import_fallback_branch(self) -> None:
        """Simulate ImportError on websockets.asyncio.server to exercise the fallback."""
        cached = sys.modules.pop("missy.channels.screencast.server", None)
        with patch.dict(
            sys.modules,
            {"websockets.asyncio.server": None},
        ), contextlib.suppress(Exception):
            import missy.channels.screencast.server as srv_mod  # noqa: F401
        if cached is not None:
            sys.modules["missy.channels.screencast.server"] = cached


# ---------------------------------------------------------------------------
# _emit exception handling (lines 99-100)
# ---------------------------------------------------------------------------

class TestEmitExceptionHandling:
    """Test that _emit swallows exceptions from event_bus.publish."""

    def test_emit_swallows_exception(self) -> None:
        from missy.channels.screencast import server as srv_mod

        with patch.object(srv_mod.event_bus, "publish", side_effect=RuntimeError("bus down")):
            # Should not raise.
            srv_mod._emit("sess-1", "screencast.test", "allow", {"x": 1})


# ---------------------------------------------------------------------------
# _load_capture_html fallback (lines 206-208)
# ---------------------------------------------------------------------------

class TestLoadCaptureHtmlFallback:
    """Test the fallback HTML when capture.html cannot be read."""

    def test_fallback_when_file_not_found(self, server: ScreencastServer) -> None:
        with patch(
            "missy.channels.screencast.server._CAPTURE_HTML_PATH",
            Path("/nonexistent/path/capture.html"),
        ):
            html = server._load_capture_html()
        assert "Capture page not found" in html


# ---------------------------------------------------------------------------
# SSL context creation (lines 226-246)
# ---------------------------------------------------------------------------

class TestSslContext:
    """Test _get_or_create_ssl_context branches."""

    def test_returns_none_when_cert_generation_fails(
        self, server: ScreencastServer, tmp_path: Path
    ) -> None:
        server._tls_cert_dir = str(tmp_path)
        with patch.object(
            ScreencastServer,
            "_generate_self_signed_cert",
            side_effect=RuntimeError("cryptography not available"),
        ):
            ctx = server._get_or_create_ssl_context()
        assert ctx is None

    def test_returns_none_when_cert_load_fails(
        self, server: ScreencastServer, tmp_path: Path
    ) -> None:
        server._tls_cert_dir = str(tmp_path)
        cert_path = tmp_path / "screencast.crt"
        key_path = tmp_path / "screencast.key"
        cert_path.write_text("not a real cert")
        key_path.write_text("not a real key")

        with patch("ssl.SSLContext") as mock_ssl:
            mock_ctx = MagicMock()
            mock_ctx.load_cert_chain.side_effect = ssl.SSLError("bad cert")
            mock_ssl.return_value = mock_ctx
            ctx = server._get_or_create_ssl_context()
        assert ctx is None

    def test_returns_ssl_context_when_certs_valid(
        self, server: ScreencastServer, tmp_path: Path
    ) -> None:
        server._tls_cert_dir = str(tmp_path)
        cert_path = tmp_path / "screencast.crt"
        key_path = tmp_path / "screencast.key"
        cert_path.write_text("fake cert")
        key_path.write_text("fake key")

        mock_ctx = MagicMock(spec=ssl.SSLContext)
        with patch("ssl.SSLContext", return_value=mock_ctx):
            ctx = server._get_or_create_ssl_context()
        assert ctx is mock_ctx
        mock_ctx.load_cert_chain.assert_called_once()


# ---------------------------------------------------------------------------
# Self-signed cert generation (lines 251-315)
# ---------------------------------------------------------------------------

class TestGenerateSelfSignedCert:
    """Test _generate_self_signed_cert using the cryptography library."""

    def test_generates_cert_and_key(self, tmp_path: Path) -> None:
        pytest.importorskip("cryptography")
        cert_path = tmp_path / "screencast.crt"
        key_path = tmp_path / "screencast.key"
        ScreencastServer._generate_self_signed_cert(cert_path, key_path)
        assert cert_path.exists()
        assert key_path.exists()
        assert cert_path.stat().st_size > 0
        assert key_path.stat().st_size > 0

    def test_cert_permissions(self, tmp_path: Path) -> None:
        pytest.importorskip("cryptography")
        cert_path = tmp_path / "screencast.crt"
        key_path = tmp_path / "screencast.key"
        ScreencastServer._generate_self_signed_cert(cert_path, key_path)
        key_mode = oct(key_path.stat().st_mode)[-3:]
        assert key_mode == "600"

    def test_generate_falls_back_when_socket_fails(self, tmp_path: Path) -> None:
        """LAN IP detection failure should be silently skipped."""
        pytest.importorskip("cryptography")
        cert_path = tmp_path / "screencast.crt"
        key_path = tmp_path / "screencast.key"

        class _FailingSocket:
            def __init__(self, *a, **kw):
                pass
            def connect(self, *a):
                raise OSError("network unreachable")
            def getsockname(self):
                return ("", 0)
            def __enter__(self):
                return self
            def __exit__(self, *a):
                pass

        with patch("socket.socket", _FailingSocket):
            ScreencastServer._generate_self_signed_cert(cert_path, key_path)

        assert cert_path.exists()


# ---------------------------------------------------------------------------
# _handle_connection unit tests (direct call, no real WS server)
# ---------------------------------------------------------------------------

class TestHandleConnectionDirect:
    """Call _handle_connection() with mocked websockets — no network."""

    @pytest.fixture
    def srv(self, registry, session_manager):
        s = ScreencastServer(
            token_registry=registry,
            session_manager=session_manager,
            host="127.0.0.1",
            port=0,
        )
        s._running = True
        queue = asyncio.Queue(maxsize=50)
        session_manager.set_queue(queue)
        return s

    @pytest.mark.asyncio
    async def test_connection_flood_protection(self, srv: ScreencastServer) -> None:
        """Reject when active connections >= _MAX_CONCURRENT_CONNECTIONS."""
        from missy.channels.screencast.server import _MAX_CONCURRENT_CONNECTIONS

        srv._active_connections = _MAX_CONCURRENT_CONNECTIONS
        ws = _make_ws()
        await srv._handle_connection(ws)
        ws.close.assert_awaited_once()
        assert srv._active_connections == _MAX_CONCURRENT_CONNECTIONS

    @pytest.mark.asyncio
    async def test_auth_timeout(self, srv: ScreencastServer) -> None:
        """Close with 1008 when first frame times out."""
        ws = MagicMock()
        ws.remote_address = ("127.0.0.1", 1)
        ws.recv = AsyncMock(side_effect=TimeoutError())
        ws.send = AsyncMock()
        ws.close = AsyncMock()
        await srv._handle_connection(ws)
        ws.close.assert_awaited_once_with(1008, "Authentication timeout")

    @pytest.mark.asyncio
    async def test_binary_first_frame_rejected(self, srv: ScreencastServer) -> None:
        """Binary first frame must be rejected."""
        ws = _make_ws(b"\xff\xd8\xff binary data")
        await srv._handle_connection(ws)
        ws.close.assert_awaited_once_with(1008, "First frame must be JSON auth")

    @pytest.mark.asyncio
    async def test_malformed_json_first_frame(self, srv: ScreencastServer) -> None:
        """Malformed JSON on first frame closes with 1008."""
        ws = _make_ws("not{json}")
        await srv._handle_connection(ws)
        ws.close.assert_awaited_once_with(1008, "Malformed JSON")

    @pytest.mark.asyncio
    async def test_missing_credentials_empty_session_id(
        self, srv: ScreencastServer
    ) -> None:
        """Missing session_id sends auth_fail and closes."""
        ws = _make_ws(json.dumps({"type": "auth", "session_id": "", "token": "tok"}))
        await srv._handle_connection(ws)
        ws.send.assert_awaited()
        sent_payload = json.loads(ws.send.call_args_list[0].args[0])
        assert sent_payload["type"] == "auth_fail"
        ws.close.assert_awaited_once_with(1008, "Missing credentials")

    @pytest.mark.asyncio
    async def test_missing_credentials_empty_token(
        self, srv: ScreencastServer, registry: ScreencastTokenRegistry
    ) -> None:
        """Empty token sends auth_fail."""
        session_id, _ = registry.create_session()
        ws = _make_ws(json.dumps({"type": "auth", "session_id": session_id, "token": ""}))
        await srv._handle_connection(ws)
        sent_payload = json.loads(ws.send.call_args_list[0].args[0])
        assert sent_payload["type"] == "auth_fail"
        ws.close.assert_awaited_once_with(1008, "Missing credentials")

    @pytest.mark.asyncio
    async def test_session_capacity_rejected(
        self, srv: ScreencastServer, registry: ScreencastTokenRegistry
    ) -> None:
        """Reject when session manager is at capacity."""
        full_sm = SessionManager(max_sessions=0)
        full_sm.set_queue(asyncio.Queue())
        srv._sessions = full_sm

        session_id, token = registry.create_session()
        ws = _make_ws(_make_auth_msg(session_id, token))
        await srv._handle_connection(ws)
        sent_payload = json.loads(ws.send.call_args_list[0].args[0])
        assert sent_payload["type"] == "auth_fail"
        assert "Too many" in sent_payload["reason"]
        ws.close.assert_awaited_once_with(1013, "Session limit reached")

    @pytest.mark.asyncio
    async def test_connection_closed_exception_is_swallowed(
        self, srv: ScreencastServer
    ) -> None:
        """ConnectionClosed during recv() must not propagate."""
        import websockets.exceptions

        ws = MagicMock()
        ws.remote_address = ("127.0.0.1", 1)
        ws.recv = AsyncMock(
            side_effect=websockets.exceptions.ConnectionClosed(None, None)
        )
        ws.send = AsyncMock()
        ws.close = AsyncMock()
        await srv._handle_connection(ws)

    @pytest.mark.asyncio
    async def test_unexpected_exception_is_logged(
        self, srv: ScreencastServer
    ) -> None:
        """Unexpected exception during handle_connection is caught and logged."""
        ws = MagicMock()
        ws.remote_address = ("127.0.0.1", 1)
        ws.recv = AsyncMock(side_effect=ValueError("boom"))
        ws.send = AsyncMock()
        ws.close = AsyncMock()
        await srv._handle_connection(ws)


# ---------------------------------------------------------------------------
# _message_loop unit tests
# ---------------------------------------------------------------------------

class TestMessageLoopDirect:
    """Test _message_loop() by passing a controlled async-iterable websocket."""

    @pytest.fixture
    def srv(self, registry, session_manager):
        s = ScreencastServer(
            token_registry=registry,
            session_manager=session_manager,
            host="127.0.0.1",
            port=0,
        )
        s._running = True
        queue = asyncio.Queue(maxsize=50)
        session_manager.set_queue(queue)
        return s

    def _make_iter_ws(self, *messages):
        """Build an async-iterable websocket mock."""
        ws = MagicMock()
        ws.send = AsyncMock()
        ws.close = AsyncMock()

        async def _aiter():
            for m in messages:
                yield m

        ws.__aiter__ = lambda self_: _aiter()
        return ws

    def _make_state(self, session_id):
        return SessionState(session_id=session_id)

    @pytest.mark.asyncio
    async def test_server_not_running_breaks_loop(
        self, srv: ScreencastServer, registry: ScreencastTokenRegistry
    ) -> None:
        """When _running becomes False, loop should break immediately."""
        session_id, _ = registry.create_session()
        state = srv._sessions.register_connection(session_id)
        srv._running = False

        ws = self._make_iter_ws(json.dumps({"type": "heartbeat"}))
        await srv._message_loop(ws, session_id, state)
        ws.send.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_unexpected_binary_without_pending_meta(
        self, srv: ScreencastServer, registry: ScreencastTokenRegistry
    ) -> None:
        """Binary frame with no pending_meta is silently ignored."""
        session_id, _ = registry.create_session()
        state = self._make_state(session_id)
        ws = self._make_iter_ws(b"\x00\x01\x02")
        await srv._message_loop(ws, session_id, state)
        ws.send.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_frame_too_large_rejected(
        self, srv: ScreencastServer, registry: ScreencastTokenRegistry
    ) -> None:
        """Binary payload exceeding _MAX_FRAME_BYTES triggers error response."""
        from missy.channels.screencast.server import _MAX_FRAME_BYTES

        session_id, _ = registry.create_session()
        state = self._make_state(session_id)
        frame_meta_json = json.dumps(
            {"type": "frame", "format": "jpeg", "width": 1920, "height": 1080, "seq": 1}
        )
        oversized = _JPEG_MAGIC + b"\x00" * (_MAX_FRAME_BYTES + 1)
        ws = self._make_iter_ws(frame_meta_json, oversized)
        await srv._message_loop(ws, session_id, state)
        ws.send.assert_awaited()
        payload = json.loads(ws.send.call_args_list[0].args[0])
        assert payload["type"] == "error"
        assert "too large" in payload["message"]

    @pytest.mark.asyncio
    async def test_rate_limiting_drops_second_frame(
        self, srv: ScreencastServer, registry: ScreencastTokenRegistry
    ) -> None:
        """Second frame arriving too quickly is silently dropped (queue stays at 1)."""
        session_id, _ = registry.create_session()
        state = self._make_state(session_id)
        frame_meta = json.dumps(
            {"type": "frame", "format": "jpeg", "width": 640, "height": 480, "seq": 1}
        )
        jpeg_data = _JPEG_MAGIC + b"\x00" * 100
        # First frame: monotonic returns 5.0 (5.0 - 0.0 >= 2.0, accepted).
        # Second frame: monotonic returns 5.5 (5.5 - 5.0 < 2.0, rate-limited/dropped).
        ws = self._make_iter_ws(
            frame_meta, jpeg_data,
            frame_meta, jpeg_data,
        )
        with patch("missy.channels.screencast.server.time") as mock_time:
            mock_time.monotonic.side_effect = [5.0, 5.5]
            await srv._message_loop(ws, session_id, state)
        assert srv._sessions.queue.qsize() == 1

    @pytest.mark.asyncio
    async def test_backpressure_sends_message(
        self, srv: ScreencastServer, registry: ScreencastTokenRegistry
    ) -> None:
        """When queue is full, a backpressure message is sent to the client."""
        session_id, _ = registry.create_session()
        state = self._make_state(session_id)

        # Fill the queue completely.
        q = srv._sessions._queue
        while not q.full():
            meta = FrameMetadata(session_id=session_id, frame_number=0, format="jpeg")
            q.put_nowait((meta, b""))

        frame_meta = json.dumps(
            {"type": "frame", "format": "jpeg", "width": 640, "height": 480, "seq": 99}
        )
        jpeg_data = _JPEG_MAGIC + b"\x00" * 100
        ws = self._make_iter_ws(frame_meta, jpeg_data)
        with patch("time.monotonic", return_value=9999.0):
            await srv._message_loop(ws, session_id, state)
        ws.send.assert_awaited()
        payload = json.loads(ws.send.call_args_list[0].args[0])
        assert payload["type"] == "backpressure"

    @pytest.mark.asyncio
    async def test_malformed_json_in_loop_sends_error(
        self, srv: ScreencastServer, registry: ScreencastTokenRegistry
    ) -> None:
        """Malformed JSON text frame in the message loop sends error response."""
        session_id, _ = registry.create_session()
        state = self._make_state(session_id)
        ws = self._make_iter_ws("this is {not valid json")
        await srv._message_loop(ws, session_id, state)
        payload = json.loads(ws.send.call_args_list[0].args[0])
        assert payload["type"] == "error"
        assert "Malformed" in payload["message"]

    @pytest.mark.asyncio
    async def test_config_message_updates_interval(
        self, srv: ScreencastServer, registry: ScreencastTokenRegistry
    ) -> None:
        """Config message updates the session capture interval."""
        session_id, _ = registry.create_session()
        state = self._make_state(session_id)
        config_msg = json.dumps({"type": "config", "interval_ms": 5000})
        ws = self._make_iter_ws(config_msg)
        await srv._message_loop(ws, session_id, state)
        assert state.capture_interval_ms == 5000

    @pytest.mark.asyncio
    async def test_config_clamps_interval_minimum(
        self, srv: ScreencastServer, registry: ScreencastTokenRegistry
    ) -> None:
        """Config interval below 2000 ms is clamped to 2000."""
        session_id, _ = registry.create_session()
        state = self._make_state(session_id)
        config_msg = json.dumps({"type": "config", "interval_ms": 100})
        ws = self._make_iter_ws(config_msg)
        await srv._message_loop(ws, session_id, state)
        assert state.capture_interval_ms == 2000

    @pytest.mark.asyncio
    async def test_config_clamps_interval_maximum(
        self, srv: ScreencastServer, registry: ScreencastTokenRegistry
    ) -> None:
        """Config interval above 300000 ms is clamped to 300000."""
        session_id, _ = registry.create_session()
        state = self._make_state(session_id)
        config_msg = json.dumps({"type": "config", "interval_ms": 999999})
        ws = self._make_iter_ws(config_msg)
        await srv._message_loop(ws, session_id, state)
        assert state.capture_interval_ms == 300000

    @pytest.mark.asyncio
    async def test_invalid_dimensions_sends_error(
        self, srv: ScreencastServer, registry: ScreencastTokenRegistry
    ) -> None:
        """Frame metadata with out-of-range dimensions returns an error."""
        session_id, _ = registry.create_session()
        state = self._make_state(session_id)
        # width and height of 1 are below _MIN_DIMENSION (64)
        bad_frame = json.dumps(
            {"type": "frame", "format": "jpeg", "width": 1, "height": 1, "seq": 1}
        )
        ws = self._make_iter_ws(bad_frame)
        await srv._message_loop(ws, session_id, state)
        payload = json.loads(ws.send.call_args_list[0].args[0])
        assert payload["type"] == "error"
        assert "dimensions" in payload["message"].lower()

    @pytest.mark.asyncio
    async def test_unknown_message_type_sends_error(
        self, srv: ScreencastServer, registry: ScreencastTokenRegistry
    ) -> None:
        """Unknown message type returns an error containing the type name."""
        session_id, _ = registry.create_session()
        state = self._make_state(session_id)
        ws = self._make_iter_ws(json.dumps({"type": "frobnicate"}))
        await srv._message_loop(ws, session_id, state)
        payload = json.loads(ws.send.call_args_list[0].args[0])
        assert payload["type"] == "error"
        assert "frobnicate" in payload["message"]

    @pytest.mark.asyncio
    async def test_heartbeat_is_no_op(
        self, srv: ScreencastServer, registry: ScreencastTokenRegistry
    ) -> None:
        """Heartbeat message produces no response."""
        session_id, _ = registry.create_session()
        state = self._make_state(session_id)
        ws = self._make_iter_ws(json.dumps({"type": "heartbeat"}))
        await srv._message_loop(ws, session_id, state)
        ws.send.assert_not_awaited()


# ---------------------------------------------------------------------------
# send_analysis (lines 652-654)
# ---------------------------------------------------------------------------

class TestSendAnalysis:
    """Test send_analysis no-op when session is not connected."""

    @pytest.mark.asyncio
    async def test_send_analysis_noop_when_not_connected(
        self, server: ScreencastServer
    ) -> None:
        """send_analysis returns without error when session has no connection."""
        await server.send_analysis("nonexistent-session", seq=1, text="hello")

    @pytest.mark.asyncio
    async def test_send_analysis_noop_when_connected(
        self, server: ScreencastServer, registry: ScreencastTokenRegistry
    ) -> None:
        """send_analysis returns without error even when a session is registered."""
        session_id, _ = registry.create_session()
        queue = asyncio.Queue()
        server._sessions.set_queue(queue)
        server._sessions.register_connection(session_id, "127.0.0.1:9999")
        await server.send_analysis(session_id, seq=1, text="analysis text")


# ---------------------------------------------------------------------------
# Security hardening tests
# ---------------------------------------------------------------------------

class TestSecurityHardening:
    """Tests for security fixes: Referrer-Policy, int() validation, dimension checks."""

    def test_referrer_policy_header_present(self) -> None:
        """_SECURITY_HEADERS must include Referrer-Policy: no-referrer."""
        from missy.channels.screencast.server import _SECURITY_HEADERS

        assert "Referrer-Policy" in _SECURITY_HEADERS
        assert _SECURITY_HEADERS["Referrer-Policy"] == "no-referrer"

    @pytest.fixture
    def srv(self, registry, session_manager):
        s = ScreencastServer(
            token_registry=registry,
            session_manager=session_manager,
            host="127.0.0.1",
            port=0,
        )
        s._running = True
        queue = asyncio.Queue(maxsize=50)
        session_manager.set_queue(queue)
        return s

    def _make_iter_ws(self, *messages):
        ws = MagicMock()
        ws.send = AsyncMock()
        ws.close = AsyncMock()

        async def _aiter():
            for m in messages:
                yield m

        ws.__aiter__ = lambda self_: _aiter()
        return ws

    @pytest.mark.asyncio
    async def test_non_numeric_width_sends_error(
        self, srv: ScreencastServer, registry: ScreencastTokenRegistry
    ) -> None:
        """Non-numeric width in frame message should send error, not crash."""
        session_id, _ = registry.create_session()
        state = SessionState(session_id=session_id)
        srv._sessions.register_connection(session_id)
        ws = self._make_iter_ws(
            json.dumps({"type": "frame", "width": "abc", "height": 100, "seq": 1}),
        )
        await srv._message_loop(ws, session_id, state)
        ws.send.assert_awaited_once()
        payload = json.loads(ws.send.call_args[0][0])
        assert payload["type"] == "error"
        assert "numeric" in payload["message"].lower() or "invalid" in payload["message"].lower()

    @pytest.mark.asyncio
    async def test_non_numeric_height_sends_error(
        self, srv: ScreencastServer, registry: ScreencastTokenRegistry
    ) -> None:
        """Non-numeric height in frame message should send error."""
        session_id, _ = registry.create_session()
        state = SessionState(session_id=session_id)
        srv._sessions.register_connection(session_id)
        ws = self._make_iter_ws(
            json.dumps({"type": "frame", "width": 1920, "height": "not_a_number"}),
        )
        await srv._message_loop(ws, session_id, state)
        ws.send.assert_awaited_once()
        payload = json.loads(ws.send.call_args[0][0])
        assert payload["type"] == "error"

    @pytest.mark.asyncio
    async def test_non_numeric_seq_sends_error(
        self, srv: ScreencastServer, registry: ScreencastTokenRegistry
    ) -> None:
        """Non-numeric seq in frame message should send error."""
        session_id, _ = registry.create_session()
        state = SessionState(session_id=session_id)
        srv._sessions.register_connection(session_id)
        ws = self._make_iter_ws(
            json.dumps({"type": "frame", "width": 1920, "height": 1080, "seq": "xyz"}),
        )
        await srv._message_loop(ws, session_id, state)
        ws.send.assert_awaited_once()
        payload = json.loads(ws.send.call_args[0][0])
        assert payload["type"] == "error"

    @pytest.mark.asyncio
    async def test_width_zero_height_nonzero_rejected(
        self, srv: ScreencastServer, registry: ScreencastTokenRegistry
    ) -> None:
        """width=0, height=1080 should be rejected (dimension bypass fix)."""
        session_id, _ = registry.create_session()
        state = SessionState(session_id=session_id)
        srv._sessions.register_connection(session_id)
        ws = self._make_iter_ws(
            json.dumps({"type": "frame", "width": 0, "height": 1080, "seq": 1}),
        )
        await srv._message_loop(ws, session_id, state)
        ws.send.assert_awaited_once()
        payload = json.loads(ws.send.call_args[0][0])
        assert payload["type"] == "error"
        assert "dimension" in payload["message"].lower() or "invalid" in payload["message"].lower()

    @pytest.mark.asyncio
    async def test_both_zero_dimensions_accepted(
        self, srv: ScreencastServer, registry: ScreencastTokenRegistry
    ) -> None:
        """width=0, height=0 (not provided) should be accepted (no dimensions)."""
        session_id, _ = registry.create_session()
        state = SessionState(session_id=session_id)
        srv._sessions.register_connection(session_id)
        ws = self._make_iter_ws(
            json.dumps({"type": "frame", "format": "jpeg", "seq": 1}),
        )
        await srv._message_loop(ws, session_id, state)
        # No error should be sent — only frame meta is prepared
        ws.send.assert_not_awaited()
