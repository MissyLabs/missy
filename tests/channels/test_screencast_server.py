"""Tests for the screencast server."""

from __future__ import annotations

import asyncio
import json
from unittest.mock import MagicMock, patch

import pytest

from missy.channels.screencast.auth import ScreencastTokenRegistry
from missy.channels.screencast.server import ScreencastServer, _JPEG_MAGIC, _PNG_MAGIC
from missy.channels.screencast.session_manager import SessionManager


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
