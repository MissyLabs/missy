"""Gap coverage tests for missy/channels/voice/server.py.

Targets remaining uncovered lines:
  293-300   : _handle_connection — auth success → message_loop → cleanup (node tracking)
  333-348   : _handle_connection — mark_offline raises; ConnectionClosed after auth
  353-356   : _handle_connection — ConnectionClosed before auth
  368-369   : _handle_connection — unexpected exception, send_json also raises
  472       : _message_loop — unknown message type logged and skipped
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import websockets.exceptions

from missy.channels.voice.registry import EdgeNode
from missy.channels.voice.server import VoiceServer


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_edge_node(
    node_id: str = "node-1",
    room: str = "office",
    paired: bool = True,
    policy_mode: str = "full",
    audio_logging: bool = False,
    audio_log_dir: str = "",
) -> EdgeNode:
    return EdgeNode(
        node_id=node_id,
        friendly_name="Test Node",
        room=room,
        ip_address="192.168.1.100",
        paired=paired,
        policy_mode=policy_mode,
        audio_logging=audio_logging,
        audio_log_dir=audio_log_dir,
    )


def _make_stt_engine(transcript_text: str = "hello world", confidence: float = 0.95):
    from missy.channels.voice.stt.base import TranscriptionResult

    engine = MagicMock()
    engine.name = "mock-stt"
    engine.load = MagicMock()
    engine.unload = MagicMock()
    engine.transcribe = AsyncMock(
        return_value=TranscriptionResult(
            text=transcript_text,
            confidence=confidence,
            processing_ms=50,
        )
    )
    return engine


def _make_tts_engine(audio_data: bytes = b"\x00" * 4096):
    from missy.channels.voice.tts.base import AudioBuffer

    engine = MagicMock()
    engine.name = "mock-tts"
    engine.load = MagicMock()
    engine.unload = MagicMock()
    engine.synthesize = AsyncMock(
        return_value=AudioBuffer(
            data=audio_data,
            sample_rate=22050,
            channels=1,
            format="wav",
        )
    )
    return engine


def _make_websocket(remote_address=("127.0.0.1", 12345)):
    ws = AsyncMock()
    ws.remote_address = remote_address
    ws.send = AsyncMock()
    ws.close = AsyncMock()
    return ws


def _make_server(
    *,
    node: EdgeNode | None = None,
    token_valid: bool = True,
    host: str = "127.0.0.1",
    debug_transcripts: bool = False,
    agent_response: str = "OK",
) -> VoiceServer:
    registry = MagicMock()
    registry.verify_token.return_value = token_valid
    registry.get_node.return_value = node or _make_edge_node()
    registry.mark_offline = MagicMock()
    registry.mark_online = MagicMock()

    pairing_manager = MagicMock()
    pairing_manager.initiate_pairing.return_value = "new-node-id"

    presence_store = MagicMock()
    presence_store.update = MagicMock()

    stt = _make_stt_engine()
    tts = _make_tts_engine()
    agent_callback = AsyncMock(return_value=agent_response)

    return VoiceServer(
        registry=registry,
        pairing_manager=pairing_manager,
        presence_store=presence_store,
        stt_engine=stt,
        tts_engine=tts,
        agent_callback=agent_callback,
        host=host,
        debug_transcripts=debug_transcripts,
    )


# ---------------------------------------------------------------------------
# _handle_connection — auth success → message_loop → cleanup  (lines 293-300)
# ---------------------------------------------------------------------------


class TestHandleConnectionAuthSuccess:
    @pytest.mark.asyncio
    async def test_auth_success_adds_to_connected_nodes(self):
        """Lines 333-338: After auth, node is added to _connected_nodes."""
        node = _make_edge_node(node_id="auth-node")
        server = _make_server(node=node)
        ws = _make_websocket()

        # First recv returns valid auth frame; after that message_loop gets empty iter.
        auth_frame = json.dumps({"type": "auth", "node_id": "auth-node", "token": "valid"})

        async def fake_aiter():
            return
            yield  # make it an async generator that yields nothing

        ws.recv = AsyncMock(return_value=auth_frame)
        ws.__aiter__ = lambda _self: fake_aiter()

        with patch("missy.channels.voice.server._emit"):
            await server._handle_connection(ws)

        # node should have been removed after the connection ends (discard called).
        assert "auth-node" not in server._connected_nodes

    @pytest.mark.asyncio
    async def test_auth_success_calls_mark_offline_on_cleanup(self):
        """Lines 345-350: mark_offline is called after message_loop exits."""
        node = _make_edge_node(node_id="cleanup-node")
        server = _make_server(node=node)
        ws = _make_websocket()

        auth_frame = json.dumps({"type": "auth", "node_id": "cleanup-node", "token": "valid"})

        async def fake_aiter():
            return
            yield

        ws.recv = AsyncMock(return_value=auth_frame)
        ws.__aiter__ = lambda _self: fake_aiter()

        with patch("missy.channels.voice.server._emit"):
            await server._handle_connection(ws)

        server._registry.mark_offline.assert_called_with("cleanup-node")

    @pytest.mark.asyncio
    async def test_mark_offline_failure_is_swallowed(self):
        """Lines 347-350: mark_offline raises → exception is logged, not propagated."""
        node = _make_edge_node(node_id="crash-node")
        server = _make_server(node=node)
        server._registry.mark_offline.side_effect = RuntimeError("db locked")
        ws = _make_websocket()

        auth_frame = json.dumps({"type": "auth", "node_id": "crash-node", "token": "valid"})

        async def fake_aiter():
            return
            yield

        ws.recv = AsyncMock(return_value=auth_frame)
        ws.__aiter__ = lambda _self: fake_aiter()

        with patch("missy.channels.voice.server._emit"):
            # Should not raise even though mark_offline fails.
            await server._handle_connection(ws)


# ---------------------------------------------------------------------------
# _handle_connection — ConnectionClosed after auth  (lines 353-356)
# ---------------------------------------------------------------------------


class TestHandleConnectionClosedAfterAuth:
    @pytest.mark.asyncio
    async def test_connection_closed_after_auth_is_handled_gracefully(self):
        """Lines 353-354: ConnectionClosed raised inside message_loop → logged, not propagated."""
        node = _make_edge_node(node_id="close-after-auth")
        server = _make_server(node=node)
        ws = _make_websocket()

        auth_frame = json.dumps({"type": "auth", "node_id": "close-after-auth", "token": "valid"})
        ws.recv = AsyncMock(return_value=auth_frame)

        async def fake_aiter():
            raise websockets.exceptions.ConnectionClosed(None, None)
            yield  # pragma: no cover

        ws.__aiter__ = lambda _self: fake_aiter()

        with patch("missy.channels.voice.server._emit"):
            # Should complete without raising.
            await server._handle_connection(ws)


# ---------------------------------------------------------------------------
# _handle_connection — auth failure → early return  (line 300)
# ---------------------------------------------------------------------------


class TestHandleConnectionAuthFailureEarlyReturn:
    @pytest.mark.asyncio
    async def test_auth_failure_returns_early_without_message_loop(self):
        """Line 300: _handle_auth returns None → function returns before message_loop."""
        server = _make_server(token_valid=False)
        ws = _make_websocket()
        auth_frame = json.dumps({"type": "auth", "node_id": "bad-node", "token": "wrong"})
        ws.recv = AsyncMock(return_value=auth_frame)

        # Provide a fake __aiter__ that should NEVER be called (we return before it).
        message_loop_called = []

        async def _fake_message_loop(websocket, node):
            message_loop_called.append(True)

        server._message_loop = _fake_message_loop

        with patch("missy.channels.voice.server._emit"):
            await server._handle_connection(ws)

        # _message_loop should not have been called.
        assert len(message_loop_called) == 0


# ---------------------------------------------------------------------------
# _handle_connection — ConnectionClosed before auth  (lines 355-356)
# ---------------------------------------------------------------------------


class TestHandleConnectionClosedBeforeAuth:
    @pytest.mark.asyncio
    async def test_connection_closed_before_auth_when_node_is_none(self):
        """Lines 355-356: ConnectionClosed raised and node is None at that point."""
        server = _make_server(token_valid=False)
        ws = _make_websocket()

        # First recv raises ConnectionClosed directly — simulates close before first frame.
        ws.recv = AsyncMock(side_effect=websockets.exceptions.ConnectionClosed(None, None))

        # Should complete without raising.
        await server._handle_connection(ws)

    @pytest.mark.asyncio
    async def test_connection_closed_before_auth_logs_remote_addr(self):
        """Line 356: ConnectionClosed before auth with node=None → else branch logged."""
        server = _make_server()
        ws = _make_websocket(remote_address=("10.0.0.1", 9999))

        # Malformed JSON first → parse fails → close. Then we need ConnectionClosed
        # after json.loads fails, not in recv. Let's just use recv side_effect directly.
        ws.recv = AsyncMock(side_effect=websockets.exceptions.ConnectionClosed(None, None))

        await server._handle_connection(ws)
        # node is None here → else branch of "if node is not None" is logged.


# ---------------------------------------------------------------------------
# _handle_connection — unexpected exception, send_json also raises  (lines 368-369)
# ---------------------------------------------------------------------------


class TestHandleConnectionUnexpectedExceptionSendFails:
    @pytest.mark.asyncio
    async def test_unexpected_exception_send_json_also_fails(self):
        """Lines 368-369: Exception path where send_json also raises → suppressed."""
        server = _make_server()
        ws = _make_websocket()

        # recv raises a non-websockets exception.
        ws.recv = AsyncMock(side_effect=ValueError("unexpected non-ws error"))
        # send also raises to hit the inner except pass branch.
        ws.send = AsyncMock(side_effect=RuntimeError("send failed too"))

        with patch("missy.channels.voice.server._emit"):
            # Should not propagate either exception.
            await server._handle_connection(ws)


# ---------------------------------------------------------------------------
# _message_loop — unknown message type  (line 472)
# ---------------------------------------------------------------------------


class TestMessageLoopUnknownMessageType:
    @pytest.mark.asyncio
    async def test_unknown_message_type_is_skipped(self):
        """Line 472: unknown message type is logged and loop continues."""
        node = _make_edge_node()
        server = _make_server(node=node)
        ws = _make_websocket()

        async def fake_aiter():
            yield json.dumps({"type": "ping"})   # unknown type
            yield json.dumps({"type": "pong"})   # another unknown type

        ws.__aiter__ = lambda _self: fake_aiter()

        with patch("missy.channels.voice.server._emit"):
            # Should complete without error.
            await server._message_loop(ws, node)

        # No transcription or agent call should have occurred.
        server._stt.transcribe.assert_not_awaited()
        server._agent_callback.assert_not_awaited()
