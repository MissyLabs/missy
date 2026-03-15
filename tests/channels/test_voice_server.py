"""Tests for missy/channels/voice/server.py — VoiceServer.

All external I/O (websockets, STT, TTS, audit events, registry) is mocked.
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from missy.channels.voice.registry import EdgeNode
from missy.channels.voice.server import VoiceServer, _emit

# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------


def _make_edge_node(
    node_id: str = "node-1",
    room: str = "office",
    paired: bool = True,
    policy_mode: str = "full",
    audio_logging: bool = False,
    audio_log_dir: str | None = None,
) -> EdgeNode:
    return EdgeNode(
        node_id=node_id,
        friendly_name="Test Node",
        room=room,
        ip_address="192.168.1.100",
        paired=paired,
        policy_mode=policy_mode,
        audio_logging=audio_logging,
        audio_log_dir=audio_log_dir or "",
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
    agent_response: str = "I am Missy.",
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
# _emit helper
# ---------------------------------------------------------------------------


class TestEmit:
    def test_emit_publishes_event_without_raising(self) -> None:
        with patch("missy.channels.voice.server.event_bus") as mock_bus:
            _emit("session-1", "voice.test.event", "allow", {"key": "val"})
        mock_bus.publish.assert_called_once()

    def test_emit_handles_exception_without_propagating(self) -> None:
        with patch("missy.channels.voice.server.event_bus") as mock_bus:
            mock_bus.publish.side_effect = RuntimeError("bus failure")
            # Should not raise.
            _emit("session-1", "voice.test.event", "allow")

    def test_emit_uses_empty_dict_when_detail_is_none(self) -> None:
        with patch("missy.channels.voice.server.event_bus") as mock_bus:
            _emit("s", "e", "allow", None)
        published = mock_bus.publish.call_args[0][0]
        assert published.detail == {}


# ---------------------------------------------------------------------------
# VoiceServer — lifecycle
# ---------------------------------------------------------------------------


class TestVoiceServerLifecycle:
    def test_initial_state(self) -> None:
        server = _make_server()
        assert server._running is False
        assert server._ws_server is None
        assert server.get_connected_nodes() == []

    @pytest.mark.asyncio
    async def test_start_loads_engines(self) -> None:
        server = _make_server()

        mock_ws_server = MagicMock()
        mock_ws_server.close = MagicMock()
        mock_ws_server.wait_closed = AsyncMock()

        # websockets.serve returns an awaitable that resolves to the server object.
        async def fake_serve(*args, **kwargs):
            return mock_ws_server

        with patch("missy.channels.voice.server._ws_serve", side_effect=fake_serve):
            await server.start()

        server._stt.load.assert_called_once()
        server._tts.load.assert_called_once()
        assert server._running is True

    @pytest.mark.asyncio
    async def test_start_is_idempotent(self) -> None:
        server = _make_server()
        server._running = True

        with patch("missy.channels.voice.server._ws_serve") as mock_serve:
            await server.start()

        mock_serve.assert_not_called()

    @pytest.mark.asyncio
    async def test_start_emits_warning_for_0000_host(self) -> None:
        server = _make_server(host="0.0.0.0")

        mock_ws_server = MagicMock()
        mock_ws_server.close = MagicMock()
        mock_ws_server.wait_closed = AsyncMock()

        async def fake_serve(*args, **kwargs):
            return mock_ws_server

        with (
            patch("missy.channels.voice.server._ws_serve", side_effect=fake_serve),
            patch("missy.channels.voice.server._emit") as mock_emit,
        ):
            await server.start()

        # Should have emitted a warning about binding to all interfaces.
        # _emit called with keyword args: session_id, event_type, result, detail.
        call_kwargs = [c.kwargs for c in mock_emit.call_args_list]
        event_types = [k.get("event_type", "") for k in call_kwargs]
        assert "voice.bind.warning" in event_types

    @pytest.mark.asyncio
    async def test_stop_when_not_running_is_noop(self) -> None:
        server = _make_server()
        assert server._running is False
        await server.stop()  # should not raise

    @pytest.mark.asyncio
    async def test_stop_unloads_engines(self) -> None:
        server = _make_server()
        server._running = True

        mock_ws_server = MagicMock()
        mock_ws_server.close = MagicMock()
        mock_ws_server.wait_closed = AsyncMock()
        server._ws_server = mock_ws_server

        await server.stop()

        server._stt.unload.assert_called_once()
        server._tts.unload.assert_called_once()
        assert server._running is False
        assert server._ws_server is None

    @pytest.mark.asyncio
    async def test_stop_clears_connected_nodes(self) -> None:
        server = _make_server()
        server._running = True
        server._connected_nodes = {"node-1", "node-2"}

        mock_ws_server = MagicMock()
        mock_ws_server.close = MagicMock()
        mock_ws_server.wait_closed = AsyncMock()
        server._ws_server = mock_ws_server

        await server.stop()

        assert len(server._connected_nodes) == 0


# ---------------------------------------------------------------------------
# VoiceServer — get_status
# ---------------------------------------------------------------------------


class TestGetStatus:
    def test_status_when_not_running(self) -> None:
        server = _make_server()
        status = server.get_status()
        assert status["running"] is False
        assert status["connected_nodes"] == 0
        assert status["host"] == "127.0.0.1"
        assert status["port"] == 8765
        assert status["stt_engine"] == "mock-stt"
        assert status["tts_engine"] == "mock-tts"

    def test_status_connected_node_count(self) -> None:
        server = _make_server()
        server._running = True
        server._connected_nodes = {"a", "b"}
        status = server.get_status()
        assert status["running"] is True
        assert status["connected_nodes"] == 2


# ---------------------------------------------------------------------------
# VoiceServer — _handle_connection
# ---------------------------------------------------------------------------


class TestHandleConnection:
    @pytest.mark.asyncio
    async def test_rejects_binary_first_frame(self) -> None:
        server = _make_server()
        ws = _make_websocket()
        ws.recv = AsyncMock(return_value=b"\xff\xfe")  # binary frame

        with patch("missy.channels.voice.server._emit"):
            await server._handle_connection(ws)

        ws.close.assert_awaited()

    @pytest.mark.asyncio
    async def test_rejects_malformed_json(self) -> None:
        server = _make_server()
        ws = _make_websocket()
        ws.recv = AsyncMock(return_value="not valid json {{{{")

        with patch("missy.channels.voice.server._emit"):
            await server._handle_connection(ws)

        ws.close.assert_awaited()

    @pytest.mark.asyncio
    async def test_rejects_unexpected_first_message_type(self) -> None:
        server = _make_server()
        ws = _make_websocket()
        ws.recv = AsyncMock(return_value=json.dumps({"type": "heartbeat"}))

        with patch("missy.channels.voice.server._emit"):
            await server._handle_connection(ws)

        ws.close.assert_awaited()

    @pytest.mark.asyncio
    async def test_handles_connection_closed_before_first_frame(self) -> None:
        import websockets.exceptions as wsexc

        server = _make_server()
        ws = _make_websocket()
        ws.recv = AsyncMock(side_effect=wsexc.ConnectionClosed(None, None))

        # Should return without error.
        await server._handle_connection(ws)

    @pytest.mark.asyncio
    async def test_handles_pair_request(self) -> None:
        server = _make_server()
        ws = _make_websocket()
        ws.recv = AsyncMock(
            return_value=json.dumps(
                {
                    "type": "pair_request",
                    "friendly_name": "New Node",
                    "room": "Kitchen",
                    "hardware_profile": {"platform": "linux"},
                }
            )
        )

        with patch("missy.channels.voice.server._emit"):
            await server._handle_connection(ws)

        # pair_pending should have been sent.
        sent_frames = [
            json.loads(c[0][0]) for c in ws.send.call_args_list if not isinstance(c[0][0], bytes)
        ]
        types = [f.get("type") for f in sent_frames]
        assert "pair_pending" in types

    @pytest.mark.asyncio
    async def test_handles_unexpected_exception_gracefully(self) -> None:
        server = _make_server()
        ws = _make_websocket()
        ws.recv = AsyncMock(side_effect=RuntimeError("unexpected crash"))

        with patch("missy.channels.voice.server._emit"):
            # Should not propagate.
            await server._handle_connection(ws)


# ---------------------------------------------------------------------------
# VoiceServer — _handle_auth
# ---------------------------------------------------------------------------


class TestHandleAuth:
    @pytest.mark.asyncio
    async def test_auth_fail_invalid_token(self) -> None:
        server = _make_server(token_valid=False)
        ws = _make_websocket()

        with patch("missy.channels.voice.server._emit"):
            result = await server._handle_auth(ws, node_id="n1", token="bad")

        assert result is None
        sent = [json.loads(c[0][0]) for c in ws.send.call_args_list]
        assert any(f["type"] == "auth_fail" for f in sent)

    @pytest.mark.asyncio
    async def test_auth_fail_node_not_found_after_verify(self) -> None:
        server = _make_server()
        server._registry.get_node.return_value = None  # token valid but node vanished
        ws = _make_websocket()

        with patch("missy.channels.voice.server._emit"):
            result = await server._handle_auth(ws, node_id="n1", token="valid")

        assert result is None
        sent = [json.loads(c[0][0]) for c in ws.send.call_args_list]
        assert any(f["type"] == "auth_fail" for f in sent)

    @pytest.mark.asyncio
    async def test_auth_fail_node_not_paired(self) -> None:
        node = _make_edge_node(paired=False)
        server = _make_server(node=node)
        ws = _make_websocket()

        with patch("missy.channels.voice.server._emit"):
            result = await server._handle_auth(ws, node_id="node-1", token="valid")

        assert result is None
        sent = [json.loads(c[0][0]) for c in ws.send.call_args_list]
        assert any(f["type"] == "auth_fail" for f in sent)

    @pytest.mark.asyncio
    async def test_auth_fail_node_muted(self) -> None:
        node = _make_edge_node(paired=True, policy_mode="muted")
        server = _make_server(node=node)
        ws = _make_websocket()

        with patch("missy.channels.voice.server._emit"):
            result = await server._handle_auth(ws, node_id="node-1", token="valid")

        assert result is None
        sent = [json.loads(c[0][0]) for c in ws.send.call_args_list]
        assert any(f["type"] == "muted" for f in sent)

    @pytest.mark.asyncio
    async def test_auth_success(self) -> None:
        node = _make_edge_node(paired=True, policy_mode="full")
        server = _make_server(node=node)
        ws = _make_websocket()

        with patch("missy.channels.voice.server._emit"):
            result = await server._handle_auth(ws, node_id="node-1", token="valid")

        assert result is node
        sent = [json.loads(c[0][0]) for c in ws.send.call_args_list]
        assert any(f["type"] == "auth_ok" for f in sent)


# ---------------------------------------------------------------------------
# VoiceServer — _handle_pair_request
# ---------------------------------------------------------------------------


class TestHandlePairRequest:
    @pytest.mark.asyncio
    async def test_pair_request_sends_pair_pending(self) -> None:
        server = _make_server()
        ws = _make_websocket()

        data = {
            "type": "pair_request",
            "friendly_name": "Living Room",
            "room": "lounge",
            "hardware_profile": {},
        }

        with patch("missy.channels.voice.server._emit"):
            await server._handle_pair_request(ws, data)

        sent = [json.loads(c[0][0]) for c in ws.send.call_args_list]
        assert any(f["type"] == "pair_pending" for f in sent)
        assert any(f.get("node_id") == "new-node-id" for f in sent)

    @pytest.mark.asyncio
    async def test_pair_request_closes_connection_after_sending(self) -> None:
        server = _make_server()
        ws = _make_websocket()

        with patch("missy.channels.voice.server._emit"):
            await server._handle_pair_request(ws, {"friendly_name": "X", "room": "Y"})

        ws.close.assert_awaited()

    @pytest.mark.asyncio
    async def test_pair_request_extracts_ip_from_remote_address(self) -> None:
        server = _make_server()
        ws = _make_websocket(remote_address=("10.0.0.42", 54321))

        with patch("missy.channels.voice.server._emit"):
            await server._handle_pair_request(ws, {"friendly_name": "X", "room": "Y"})

        kwargs = server._pairing_manager.initiate_pairing.call_args[1]
        assert kwargs["ip_address"] == "10.0.0.42"

    @pytest.mark.asyncio
    async def test_pair_request_handles_none_remote_address(self) -> None:
        server = _make_server()
        ws = _make_websocket()
        ws.remote_address = None

        with patch("missy.channels.voice.server._emit"):
            await server._handle_pair_request(ws, {"friendly_name": "X", "room": "Y"})

        kwargs = server._pairing_manager.initiate_pairing.call_args[1]
        assert kwargs["ip_address"] == "unknown"


# ---------------------------------------------------------------------------
# VoiceServer — _message_loop
# ---------------------------------------------------------------------------


class TestMessageLoop:
    @pytest.mark.asyncio
    async def test_binary_outside_audio_session_is_ignored(self) -> None:
        node = _make_edge_node()
        server = _make_server(node=node)
        ws = _make_websocket()

        async def fake_aiter():
            yield b"\xff\xfe"

        ws.__aiter__ = lambda _self: fake_aiter()

        await server._message_loop(ws, node)

        # Should not have sent any error about binary outside session.

    @pytest.mark.asyncio
    async def test_audio_accumulation_and_end(self) -> None:
        node = _make_edge_node()
        server = _make_server(node=node)
        ws = _make_websocket()

        async def fake_aiter():
            yield json.dumps({"type": "audio_start", "sample_rate": 16000, "channels": 1})
            yield b"\x00" * 100
            yield b"\x01" * 100
            yield json.dumps({"type": "audio_end"})

        ws.__aiter__ = lambda _self: fake_aiter()

        with patch("missy.channels.voice.server._emit"):
            await server._message_loop(ws, node)

        # Audio should have been transcribed.
        server._stt.transcribe.assert_awaited_once()
        call_args = server._stt.transcribe.call_args
        assert call_args[0][0] == b"\x00" * 100 + b"\x01" * 100

    @pytest.mark.asyncio
    async def test_audio_end_outside_session_is_ignored(self) -> None:
        node = _make_edge_node()
        server = _make_server(node=node)
        ws = _make_websocket()

        async def fake_aiter():
            yield json.dumps({"type": "audio_end"})  # outside any audio session

        ws.__aiter__ = lambda _self: fake_aiter()

        with patch("missy.channels.voice.server._emit"):
            await server._message_loop(ws, node)

        # No transcription should have occurred.
        server._stt.transcribe.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_heartbeat_updates_presence(self) -> None:
        node = _make_edge_node()
        server = _make_server(node=node)
        ws = _make_websocket()

        async def fake_aiter():
            yield json.dumps(
                {
                    "type": "heartbeat",
                    "occupancy": True,
                    "noise_level": 0.3,
                    "wake_word_fp": False,
                }
            )

        ws.__aiter__ = lambda _self: fake_aiter()

        with patch("missy.channels.voice.server._emit"):
            await server._message_loop(ws, node)

        server._presence_store.update.assert_called_once_with(
            node.node_id,
            occupancy=True,
            noise_level=0.3,
            wake_word_fp=False,
        )

    @pytest.mark.asyncio
    async def test_malformed_json_sends_error_frame(self) -> None:
        node = _make_edge_node()
        server = _make_server(node=node)
        ws = _make_websocket()

        async def fake_aiter():
            yield "{{not json}}"

        ws.__aiter__ = lambda _self: fake_aiter()

        with patch("missy.channels.voice.server._emit"):
            await server._message_loop(ws, node)

        sent = [json.loads(c[0][0]) for c in ws.send.call_args_list]
        assert any(f["type"] == "error" for f in sent)

    @pytest.mark.asyncio
    async def test_spurious_auth_frame_is_ignored(self) -> None:
        node = _make_edge_node()
        server = _make_server(node=node)
        ws = _make_websocket()

        async def fake_aiter():
            yield json.dumps({"type": "auth", "node_id": "n1", "token": "t"})

        ws.__aiter__ = lambda _self: fake_aiter()

        with patch("missy.channels.voice.server._emit"):
            await server._message_loop(ws, node)

        # No auth processing should have happened (just logged and ignored).
        server._registry.verify_token.assert_not_called()

    @pytest.mark.asyncio
    async def test_audio_buffer_overflow_closes_connection(self) -> None:
        node = _make_edge_node()
        server = _make_server(node=node)
        ws = _make_websocket()

        big_chunk = b"\xff" * (11 * 1024 * 1024)  # 11 MB > 10 MB limit

        async def fake_aiter():
            yield json.dumps({"type": "audio_start", "sample_rate": 16000, "channels": 1})
            yield big_chunk

        ws.__aiter__ = lambda _self: fake_aiter()

        with patch("missy.channels.voice.server._emit"):
            await server._message_loop(ws, node)

        ws.close.assert_awaited()


# ---------------------------------------------------------------------------
# VoiceServer — _handle_audio
# ---------------------------------------------------------------------------


class TestHandleAudio:
    @pytest.mark.asyncio
    async def test_stt_failure_sends_error_frame(self) -> None:
        node = _make_edge_node()
        server = _make_server(node=node)
        server._stt.transcribe = AsyncMock(side_effect=RuntimeError("STT crashed"))
        ws = _make_websocket()

        with patch("missy.channels.voice.server._emit"):
            await server._handle_audio(ws, node=node, audio_buffer=b"\x00" * 100, sample_rate=16000)

        sent = [json.loads(c[0][0]) for c in ws.send.call_args_list if isinstance(c[0][0], str)]
        assert any(f["type"] == "error" and "Speech" in f.get("message", "") for f in sent)

    @pytest.mark.asyncio
    async def test_empty_transcript_skips_agent_call(self) -> None:
        from missy.channels.voice.stt.base import TranscriptionResult

        node = _make_edge_node()
        server = _make_server(node=node)
        server._stt.transcribe = AsyncMock(
            return_value=TranscriptionResult(
                text="   ",  # empty after strip
                confidence=0.0,
                processing_ms=10,
            )
        )
        ws = _make_websocket()

        with patch("missy.channels.voice.server._emit"):
            await server._handle_audio(ws, node=node, audio_buffer=b"\x00" * 100, sample_rate=16000)

        server._agent_callback.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_debug_transcripts_sends_transcript_frame(self) -> None:
        node = _make_edge_node()
        server = _make_server(node=node, debug_transcripts=True)
        ws = _make_websocket()

        with patch("missy.channels.voice.server._emit"):
            await server._handle_audio(ws, node=node, audio_buffer=b"\x00" * 100, sample_rate=16000)

        # Filter out binary frames (audio chunks), only parse JSON strings.
        sent = [json.loads(c[0][0]) for c in ws.send.call_args_list if isinstance(c[0][0], str)]
        assert any(f["type"] == "transcript" for f in sent)

    @pytest.mark.asyncio
    async def test_agent_callback_failure_sends_error_frame(self) -> None:
        node = _make_edge_node()
        server = _make_server(node=node)
        server._agent_callback = AsyncMock(side_effect=RuntimeError("Agent failure"))
        ws = _make_websocket()

        with patch("missy.channels.voice.server._emit"):
            await server._handle_audio(ws, node=node, audio_buffer=b"\x00" * 100, sample_rate=16000)

        sent = [json.loads(c[0][0]) for c in ws.send.call_args_list if isinstance(c[0][0], str)]
        assert any(f["type"] == "error" and "Agent" in f.get("message", "") for f in sent)

    @pytest.mark.asyncio
    async def test_tts_failure_sends_text_only(self) -> None:
        node = _make_edge_node()
        server = _make_server(node=node)
        server._tts.synthesize = AsyncMock(side_effect=RuntimeError("TTS crashed"))
        ws = _make_websocket()

        with patch("missy.channels.voice.server._emit"):
            await server._handle_audio(ws, node=node, audio_buffer=b"\x00" * 100, sample_rate=16000)

        sent = [json.loads(c[0][0]) for c in ws.send.call_args_list if isinstance(c[0][0], str)]
        # response_text should have been sent even when TTS fails.
        assert any(f["type"] == "response_text" for f in sent)
        # But no audio_start should be present.
        assert not any(f["type"] == "audio_start" for f in sent)

    @pytest.mark.asyncio
    async def test_successful_audio_pipeline_sends_all_frames(self) -> None:
        node = _make_edge_node()
        server = _make_server(node=node, agent_response="Hello!")
        ws = _make_websocket()

        with patch("missy.channels.voice.server._emit"):
            await server._handle_audio(ws, node=node, audio_buffer=b"\x00" * 100, sample_rate=16000)

        sent = [
            json.loads(c[0][0]) for c in ws.send.call_args_list if not isinstance(c[0][0], bytes)
        ]
        types = [f["type"] for f in sent]
        assert "response_text" in types
        assert "audio_start" in types
        assert "audio_end" in types

    @pytest.mark.asyncio
    async def test_audio_streamed_in_chunks(self) -> None:
        node = _make_edge_node()
        audio_data = b"\xab" * 8192  # 2 chunks of 4096
        server = _make_server(node=node, agent_response="OK")
        server._tts.synthesize = AsyncMock(
            return_value=__import__(
                "missy.channels.voice.tts.base", fromlist=["AudioBuffer"]
            ).AudioBuffer(data=audio_data, sample_rate=22050, channels=1, format="wav")
        )
        ws = _make_websocket()

        with patch("missy.channels.voice.server._emit"):
            await server._handle_audio(ws, node=node, audio_buffer=b"\x00" * 100, sample_rate=16000)

        # Check that binary sends happened.
        binary_sends = [c for c in ws.send.call_args_list if isinstance(c[0][0], bytes)]
        assert len(binary_sends) == 2  # 8192 bytes / 4096 chunk_size

    @pytest.mark.asyncio
    async def test_audio_logging_writes_to_disk(self, tmp_path) -> None:
        log_dir = tmp_path / "audio_logs"
        node = _make_edge_node(audio_logging=True, audio_log_dir=str(log_dir))
        server = _make_server(node=node)
        ws = _make_websocket()

        with patch("missy.channels.voice.server._emit"):
            await server._handle_audio(ws, node=node, audio_buffer=b"\x00" * 200, sample_rate=16000)

        pcm_files = list(log_dir.glob("*.pcm"))
        assert len(pcm_files) == 1
        assert pcm_files[0].read_bytes() == b"\x00" * 200


# ---------------------------------------------------------------------------
# VoiceServer — _handle_heartbeat
# ---------------------------------------------------------------------------


class TestHandleHeartbeat:
    @pytest.mark.asyncio
    async def test_heartbeat_calls_mark_online(self) -> None:
        node = _make_edge_node()
        server = _make_server(node=node)
        ws = _make_websocket(remote_address=("192.168.1.5", 9999))

        data = {"type": "heartbeat", "occupancy": False, "noise_level": 0.1}
        await server._handle_heartbeat(ws, node=node, data=data)

        server._registry.mark_online.assert_called_once_with(node.node_id, ip_address="192.168.1.5")

    @pytest.mark.asyncio
    async def test_heartbeat_presence_exception_is_logged(self) -> None:
        node = _make_edge_node()
        server = _make_server(node=node)
        server._presence_store.update.side_effect = RuntimeError("store error")
        ws = _make_websocket()

        # Should not propagate.
        await server._handle_heartbeat(ws, node=node, data={})

    @pytest.mark.asyncio
    async def test_heartbeat_mark_online_exception_is_logged(self) -> None:
        node = _make_edge_node()
        server = _make_server(node=node)
        server._registry.mark_online.side_effect = RuntimeError("registry error")
        ws = _make_websocket()

        # Should not propagate.
        await server._handle_heartbeat(ws, node=node, data={})

    @pytest.mark.asyncio
    async def test_heartbeat_wake_word_fp_defaults_false(self) -> None:
        node = _make_edge_node()
        server = _make_server(node=node)
        ws = _make_websocket()

        await server._handle_heartbeat(ws, node=node, data={"type": "heartbeat"})

        server._presence_store.update.assert_called_once_with(
            node.node_id,
            occupancy=None,
            noise_level=None,
            wake_word_fp=False,
        )


# ---------------------------------------------------------------------------
# VoiceServer — _log_audio_to_disk
# ---------------------------------------------------------------------------


class TestLogAudioToDisk:
    @pytest.mark.asyncio
    async def test_creates_pcm_file(self, tmp_path) -> None:
        log_dir = tmp_path / "audio"
        node = _make_edge_node(audio_logging=True, audio_log_dir=str(log_dir))
        server = _make_server(node=node)

        with patch("missy.channels.voice.server._emit"):
            await server._log_audio_to_disk(node, b"\xff" * 512, 16000, 1)

        pcm_files = list(log_dir.glob("*.pcm"))
        assert len(pcm_files) == 1
        assert pcm_files[0].read_bytes() == b"\xff" * 512

    @pytest.mark.asyncio
    async def test_handles_write_failure_gracefully(self) -> None:
        node = _make_edge_node(audio_logging=True, audio_log_dir="/nonexistent/path")
        server = _make_server(node=node)

        with (
            patch("missy.channels.voice.server._emit"),
            patch("pathlib.Path.mkdir", side_effect=PermissionError("denied")),
        ):
            # Should not propagate.
            await server._log_audio_to_disk(node, b"\x00" * 100, 16000, 1)
