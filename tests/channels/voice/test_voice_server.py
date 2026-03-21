"""Tests for the VoiceServer WebSocket protocol handler.

Tests the voice server's authentication, pair request handling, message loop,
and audio processing logic using mocked dependencies.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from unittest.mock import AsyncMock, MagicMock

import pytest

from missy.channels.voice.stt.base import TranscriptionResult
from missy.channels.voice.tts.base import AudioBuffer

# ---------------------------------------------------------------------------
# Mock edge node
# ---------------------------------------------------------------------------


@dataclass
class MockEdgeNode:
    node_id: str = "node-1"
    friendly_name: str = "Test Node"
    room: str = "office"
    paired: bool = True
    policy_mode: str = "full"
    ip_address: str = "192.168.1.100"
    hardware_profile: dict = None
    audio_logging: bool = False
    audio_log_dir: str = ""

    def __post_init__(self):
        if self.hardware_profile is None:
            self.hardware_profile = {}


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_registry():
    reg = MagicMock()
    reg.verify_token = MagicMock(return_value=True)
    reg.get_node = MagicMock(return_value=MockEdgeNode())
    reg.mark_online = MagicMock()
    reg.mark_offline = MagicMock()
    return reg


@pytest.fixture
def mock_pairing():
    pm = MagicMock()
    pm.initiate_pairing = MagicMock(return_value="new-node-id")
    return pm


@pytest.fixture
def mock_presence():
    return MagicMock()


@pytest.fixture
def mock_stt():
    stt = MagicMock()
    stt.name = "test-stt"
    stt.load = MagicMock()
    stt.unload = MagicMock()
    stt.transcribe = AsyncMock(
        return_value=TranscriptionResult(
            text="hello world", confidence=0.95, processing_ms=100, language="en"
        )
    )
    return stt


@pytest.fixture
def mock_tts():
    tts = MagicMock()
    tts.name = "test-tts"
    tts.load = MagicMock()
    tts.unload = MagicMock()
    tts.synthesize = AsyncMock(
        return_value=AudioBuffer(data=b"\x00" * 44100, sample_rate=22050, channels=1, format="wav")
    )
    return tts


@pytest.fixture
def mock_agent_callback():
    return AsyncMock(return_value="Agent response")


@pytest.fixture
def server(mock_registry, mock_pairing, mock_presence, mock_stt, mock_tts, mock_agent_callback):
    from missy.channels.voice.server import VoiceServer

    return VoiceServer(
        registry=mock_registry,
        pairing_manager=mock_pairing,
        presence_store=mock_presence,
        stt_engine=mock_stt,
        tts_engine=mock_tts,
        agent_callback=mock_agent_callback,
        host="127.0.0.1",
        port=0,  # won't actually bind in tests
    )


# ---------------------------------------------------------------------------
# Auth handler tests
# ---------------------------------------------------------------------------


class TestHandleAuth:
    @pytest.mark.asyncio
    async def test_auth_success(self, server, mock_registry):
        ws = AsyncMock()
        ws.remote_address = ("127.0.0.1", 12345)
        node = await server._handle_auth(ws, node_id="node-1", token="valid-token")
        assert node is not None
        assert node.node_id == "node-1"
        # Should have sent auth_ok
        ws.send.assert_awaited()
        sent = json.loads(ws.send.call_args[0][0])
        assert sent["type"] == "auth_ok"

    @pytest.mark.asyncio
    async def test_auth_fail_invalid_token(self, server, mock_registry):
        mock_registry.verify_token = MagicMock(return_value=False)
        ws = AsyncMock()
        ws.remote_address = ("127.0.0.1", 12345)
        node = await server._handle_auth(ws, node_id="node-1", token="bad-token")
        assert node is None
        ws.close.assert_awaited()

    @pytest.mark.asyncio
    async def test_auth_fail_node_not_found(self, server, mock_registry):
        mock_registry.get_node = MagicMock(return_value=None)
        ws = AsyncMock()
        ws.remote_address = ("127.0.0.1", 12345)
        node = await server._handle_auth(ws, node_id="ghost", token="token")
        assert node is None
        ws.close.assert_awaited()

    @pytest.mark.asyncio
    async def test_auth_fail_not_paired(self, server, mock_registry):
        mock_registry.get_node = MagicMock(return_value=MockEdgeNode(paired=False))
        ws = AsyncMock()
        ws.remote_address = ("127.0.0.1", 12345)
        node = await server._handle_auth(ws, node_id="node-1", token="token")
        assert node is None

    @pytest.mark.asyncio
    async def test_auth_fail_muted_node(self, server, mock_registry):
        mock_registry.get_node = MagicMock(return_value=MockEdgeNode(policy_mode="muted"))
        ws = AsyncMock()
        ws.remote_address = ("127.0.0.1", 12345)
        node = await server._handle_auth(ws, node_id="node-1", token="token")
        assert node is None
        # Should have sent "muted" frame
        ws.send.assert_awaited()
        sent = json.loads(ws.send.call_args[0][0])
        assert sent["type"] == "muted"


# ---------------------------------------------------------------------------
# Pair request handler tests
# ---------------------------------------------------------------------------


class TestHandlePairRequest:
    @pytest.mark.asyncio
    async def test_pair_request(self, server, mock_pairing):
        ws = AsyncMock()
        ws.remote_address = ("192.168.1.50", 9999)
        data = {
            "type": "pair_request",
            "friendly_name": "Kitchen Pi",
            "room": "kitchen",
            "hardware_profile": {"type": "rpi4"},
        }
        await server._handle_pair_request(ws, data)
        mock_pairing.initiate_pairing.assert_called_once()
        # Should send pair_pending and close
        ws.send.assert_awaited()
        sent = json.loads(ws.send.call_args[0][0])
        assert sent["type"] == "pair_pending"
        assert sent["node_id"] == "new-node-id"
        ws.close.assert_awaited()


# ---------------------------------------------------------------------------
# Heartbeat handler tests
# ---------------------------------------------------------------------------


class TestHandleHeartbeat:
    @pytest.mark.asyncio
    async def test_heartbeat_updates_presence(self, server, mock_presence, mock_registry):
        ws = AsyncMock()
        ws.remote_address = ("192.168.1.100", 12345)
        node = MockEdgeNode()
        data = {
            "type": "heartbeat",
            "occupancy": True,
            "noise_level": 0.3,
            "wake_word_fp": False,
        }
        await server._handle_heartbeat(ws, node=node, data=data)
        mock_presence.update.assert_called_once_with(
            "node-1",
            occupancy=True,
            noise_level=0.3,
            wake_word_fp=False,
        )
        mock_registry.mark_online.assert_called_once()

    @pytest.mark.asyncio
    async def test_heartbeat_graceful_on_presence_error(self, server, mock_presence, mock_registry):
        mock_presence.update = MagicMock(side_effect=RuntimeError("DB down"))
        ws = AsyncMock()
        ws.remote_address = ("192.168.1.100", 12345)
        node = MockEdgeNode()
        data = {"type": "heartbeat"}
        # Should not raise
        await server._handle_heartbeat(ws, node=node, data=data)


# ---------------------------------------------------------------------------
# Audio processing tests
# ---------------------------------------------------------------------------


class TestHandleAudio:
    @pytest.mark.asyncio
    async def test_audio_full_pipeline(self, server, mock_stt, mock_tts, mock_agent_callback):
        ws = AsyncMock()
        ws.remote_address = ("192.168.1.100", 12345)
        node = MockEdgeNode()
        audio = b"\x00\x01" * 16000  # 1 second of audio

        await server._handle_audio(ws, node=node, audio_buffer=audio, sample_rate=16000)

        mock_stt.transcribe.assert_awaited_once()
        mock_agent_callback.assert_awaited_once()
        # Should send response_text and audio frames
        calls = ws.send.await_args_list
        sent_types = []
        for c in calls:
            arg = c[0][0]
            if isinstance(arg, str):
                sent_types.append(json.loads(arg).get("type"))
        assert "response_text" in sent_types
        assert "audio_start" in sent_types
        assert "audio_end" in sent_types

    @pytest.mark.asyncio
    async def test_audio_stt_failure(self, server, mock_stt):
        mock_stt.transcribe = AsyncMock(side_effect=RuntimeError("STT crash"))
        ws = AsyncMock()
        ws.remote_address = ("192.168.1.100", 12345)
        node = MockEdgeNode()

        await server._handle_audio(ws, node=node, audio_buffer=b"\x00" * 100, sample_rate=16000)

        # Should send error frame
        ws.send.assert_awaited()
        sent = json.loads(ws.send.call_args[0][0])
        assert sent["type"] == "error"
        assert "recognition" in sent["message"].lower()

    @pytest.mark.asyncio
    async def test_audio_empty_transcript(self, server, mock_stt, mock_agent_callback):
        mock_stt.transcribe = AsyncMock(
            return_value=TranscriptionResult(text="   ", confidence=0.1, processing_ms=50)
        )
        ws = AsyncMock()
        ws.remote_address = ("192.168.1.100", 12345)
        node = MockEdgeNode()

        await server._handle_audio(ws, node=node, audio_buffer=b"\x00" * 100, sample_rate=16000)

        # Agent should not be called for empty transcripts
        mock_agent_callback.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_audio_agent_failure(self, server, mock_stt, mock_agent_callback):
        mock_agent_callback.side_effect = RuntimeError("Agent crashed")
        ws = AsyncMock()
        ws.remote_address = ("192.168.1.100", 12345)
        node = MockEdgeNode()

        await server._handle_audio(ws, node=node, audio_buffer=b"\x00" * 100, sample_rate=16000)

        ws.send.assert_awaited()

    @pytest.mark.asyncio
    async def test_audio_tts_failure_sends_text_only(self, server, mock_tts, mock_agent_callback):
        mock_tts.synthesize = AsyncMock(side_effect=RuntimeError("TTS crash"))
        ws = AsyncMock()
        ws.remote_address = ("192.168.1.100", 12345)
        node = MockEdgeNode()

        await server._handle_audio(ws, node=node, audio_buffer=b"\x00" * 100, sample_rate=16000)

        # response_text should still be sent
        calls = ws.send.await_args_list
        sent_types = []
        for c in calls:
            arg = c[0][0]
            if isinstance(arg, str):
                sent_types.append(json.loads(arg).get("type"))
        assert "response_text" in sent_types
        # No audio frames
        assert "audio_start" not in sent_types


# ---------------------------------------------------------------------------
# Server status / lifecycle tests
# ---------------------------------------------------------------------------


class TestServerStatus:
    def test_get_status(self, server):
        status = server.get_status()
        assert status["running"] is False
        assert status["connected_nodes"] == 0
        assert status["stt_engine"] == "test-stt"
        assert status["tts_engine"] == "test-tts"

    def test_get_connected_nodes_empty(self, server):
        assert server.get_connected_nodes() == []

    def test_get_connected_nodes_tracks(self, server):
        server._connected_nodes.add("node-1")
        server._connected_nodes.add("node-2")
        nodes = server.get_connected_nodes()
        assert set(nodes) == {"node-1", "node-2"}

    def test_debug_transcripts_default(self, server):
        assert server._debug_transcripts is False
