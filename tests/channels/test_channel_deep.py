"""Deep integration tests for voice and webhook channels.

Covers:
  - DeviceRegistry persistence (JSON round-trip via tmp_path)
  - DeviceRegistry PBKDF2 token hashing and verification
  - DeviceRegistry policy-mode fields and enforcement surface
  - DeviceRegistry duplicate/idempotent node handling
  - VoiceServer WebSocket protocol state machine (auth → streaming → idle)
  - VoiceServer binary PCM frame accumulation and overflow guard
  - VoiceServer concurrent connections tracking
  - WebhookChannel HTTP POST receive pipeline
  - WebhookChannel HMAC signature authentication
  - WebhookChannel malformed-request error handling
  - CLIChannel send/receive basic operations
  - Channel type identity and BaseChannel contract
"""

from __future__ import annotations

import hashlib
import hmac as _hmac
import http.client
import json
import socket
import sys
import time
from io import StringIO
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from missy.channels.base import BaseChannel, ChannelMessage
from missy.channels.cli_channel import CLIChannel
from missy.channels.voice.pairing import PairingManager
from missy.channels.voice.presence import PresenceStore
from missy.channels.voice.registry import DeviceRegistry, EdgeNode
from missy.channels.voice.server import VoiceServer
from missy.channels.voice.stt.base import STTEngine, TranscriptionResult
from missy.channels.voice.tts.base import AudioBuffer, TTSEngine
from missy.channels.webhook import WebhookChannel


# ---------------------------------------------------------------------------
# WebSocket mock helper
# ---------------------------------------------------------------------------


class _FakeWebSocket:
    """A minimal fake WebSocket that supports ``async for`` iteration.

    ``_message_loop`` calls ``async for raw in websocket`` — this class
    implements the async iterator protocol correctly so MagicMock's
    dunder-assignment quirks are avoided entirely.
    """

    def __init__(self, frames: list[str | bytes]) -> None:
        self._frames = list(frames)
        self._idx = 0
        self.remote_address = ("127.0.0.1", 9000)
        self._sent: list[str | bytes] = []
        self._closed: bool = False
        self._close_args: tuple = ()

    # Async iterator protocol
    def __aiter__(self):
        return self

    async def __anext__(self) -> str | bytes:
        if self._idx >= len(self._frames):
            raise StopAsyncIteration
        frame = self._frames[self._idx]
        self._idx += 1
        return frame

    async def send(self, data: str | bytes) -> None:
        self._sent.append(data)

    async def close(self, code: int = 1000, reason: str = "") -> None:
        self._closed = True
        self._close_args = (code, reason)

    def sent_json(self) -> list[dict]:
        return [json.loads(m) for m in self._sent if isinstance(m, str)]

    def sent_binary(self) -> list[bytes]:
        return [m for m in self._sent if isinstance(m, bytes)]

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _free_port() -> int:
    """Return an unused TCP port on loopback."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _make_node(
    node_id: str = "node-test-1",
    friendly_name: str = "Test Node",
    room: str = "lab",
    ip_address: str = "10.0.0.1",
    paired: bool = True,
    policy_mode: str = "full",
) -> EdgeNode:
    return EdgeNode(
        node_id=node_id,
        friendly_name=friendly_name,
        room=room,
        ip_address=ip_address,
        paired=paired,
        policy_mode=policy_mode,
    )


def _registry_at(tmp_path: Path) -> DeviceRegistry:
    return DeviceRegistry(registry_path=str(tmp_path / "devices.json"))


def _stub_stt(text: str = "hello world", confidence: float = 0.9) -> MagicMock:
    engine = MagicMock(spec=STTEngine)
    engine.name = "stub-stt"
    engine.load = MagicMock()
    engine.unload = MagicMock()
    engine.transcribe = AsyncMock(
        return_value=TranscriptionResult(text=text, confidence=confidence, processing_ms=10)
    )
    return engine


def _stub_tts(wav_data: bytes = b"\x00" * 512) -> MagicMock:
    engine = MagicMock(spec=TTSEngine)
    engine.name = "stub-tts"
    engine.load = MagicMock()
    engine.unload = MagicMock()
    engine.synthesize = AsyncMock(
        return_value=AudioBuffer(data=wav_data, sample_rate=22050, channels=1, format="wav")
    )
    return engine


def _make_server(
    registry: DeviceRegistry,
    stt: MagicMock | None = None,
    tts: MagicMock | None = None,
    agent_cb: AsyncMock | None = None,
) -> VoiceServer:
    pairing_mgr = PairingManager(registry)
    presence_store = PresenceStore(registry)
    if stt is None:
        stt = _stub_stt()
    if tts is None:
        tts = _stub_tts()
    if agent_cb is None:
        agent_cb = AsyncMock(return_value="pong")
    return VoiceServer(
        registry=registry,
        pairing_manager=pairing_mgr,
        presence_store=presence_store,
        stt_engine=stt,
        tts_engine=tts,
        agent_callback=agent_cb,
        host="127.0.0.1",
        port=0,
    )


# ---------------------------------------------------------------------------
# 1. DeviceRegistry — JSON persistence
# ---------------------------------------------------------------------------


class TestDeviceRegistryPersistence:
    """Registry saves to and loads from disk correctly."""

    def test_save_and_reload_empty_registry(self, tmp_path: Path) -> None:
        reg = _registry_at(tmp_path)
        reg.save()
        reg2 = _registry_at(tmp_path)
        reg2.load()
        assert reg2.list_nodes() == []

    def test_save_and_reload_single_node(self, tmp_path: Path) -> None:
        reg = _registry_at(tmp_path)
        node = _make_node()
        reg.add_node(node)

        reg2 = _registry_at(tmp_path)
        reg2.load()
        loaded = reg2.get_node(node.node_id)
        assert loaded is not None
        assert loaded.node_id == node.node_id
        assert loaded.friendly_name == node.friendly_name
        assert loaded.room == node.room
        assert loaded.ip_address == node.ip_address

    def test_save_and_reload_multiple_nodes(self, tmp_path: Path) -> None:
        reg = _registry_at(tmp_path)
        nodes = [_make_node(node_id=f"n{i}", room=f"room-{i}") for i in range(5)]
        for n in nodes:
            reg.add_node(n)

        reg2 = _registry_at(tmp_path)
        reg2.load()
        loaded_ids = {n.node_id for n in reg2.list_nodes()}
        assert loaded_ids == {n.node_id for n in nodes}

    def test_missing_file_loads_empty(self, tmp_path: Path) -> None:
        reg = DeviceRegistry(registry_path=str(tmp_path / "nonexistent.json"))
        reg.load()
        assert reg.list_nodes() == []

    def test_corrupt_file_loads_empty(self, tmp_path: Path) -> None:
        p = tmp_path / "devices.json"
        p.write_text("not valid json", encoding="utf-8")
        reg = DeviceRegistry(registry_path=str(p))
        reg.load()
        assert reg.list_nodes() == []

    def test_atomic_write_creates_valid_json(self, tmp_path: Path) -> None:
        reg = _registry_at(tmp_path)
        reg.add_node(_make_node())
        path = tmp_path / "devices.json"
        assert path.exists()
        data = json.loads(path.read_text(encoding="utf-8"))
        assert isinstance(data, list)
        assert len(data) == 1
        assert data[0]["node_id"] == "node-test-1"

    def test_reload_preserves_paired_state(self, tmp_path: Path) -> None:
        reg = _registry_at(tmp_path)
        node = _make_node(paired=False)
        reg.add_node(node)
        reg.approve_node(node.node_id)

        reg2 = _registry_at(tmp_path)
        reg2.load()
        assert reg2.get_node(node.node_id).paired is True

    def test_reload_preserves_policy_mode(self, tmp_path: Path) -> None:
        reg = _registry_at(tmp_path)
        node = _make_node(policy_mode="safe-chat")
        reg.add_node(node)

        reg2 = _registry_at(tmp_path)
        reg2.load()
        assert reg2.get_node(node.node_id).policy_mode == "safe-chat"


# ---------------------------------------------------------------------------
# 2. DeviceRegistry — PBKDF2 token hashing
# ---------------------------------------------------------------------------


class TestDeviceRegistryTokenHashing:
    """Token generation uses PBKDF2-HMAC-SHA256; verification is constant-time."""

    def test_generate_token_returns_plaintext(self, tmp_path: Path) -> None:
        reg = _registry_at(tmp_path)
        reg.add_node(_make_node())
        token = reg.generate_token("node-test-1")
        assert isinstance(token, str)
        assert len(token) > 10

    def test_verify_correct_token(self, tmp_path: Path) -> None:
        reg = _registry_at(tmp_path)
        reg.add_node(_make_node())
        token = reg.generate_token("node-test-1")
        assert reg.verify_token("node-test-1", token) is True

    def test_verify_wrong_token_rejected(self, tmp_path: Path) -> None:
        reg = _registry_at(tmp_path)
        reg.add_node(_make_node())
        reg.generate_token("node-test-1")
        assert reg.verify_token("node-test-1", "completely-wrong-token") is False

    def test_verify_unknown_node_returns_false(self, tmp_path: Path) -> None:
        reg = _registry_at(tmp_path)
        assert reg.verify_token("ghost-node", "any-token") is False

    def test_token_hash_stored_not_plaintext(self, tmp_path: Path) -> None:
        reg = _registry_at(tmp_path)
        reg.add_node(_make_node())
        token = reg.generate_token("node-test-1")
        node = reg.get_node("node-test-1")
        assert node is not None
        assert node.token_hash != token
        assert len(node.token_hash) == 64  # hex-encoded SHA-256 = 32 bytes = 64 hex chars

    def test_regenerate_token_invalidates_old(self, tmp_path: Path) -> None:
        reg = _registry_at(tmp_path)
        reg.add_node(_make_node())
        old_token = reg.generate_token("node-test-1")
        _new_token = reg.generate_token("node-test-1")
        assert reg.verify_token("node-test-1", old_token) is False

    def test_token_salted_per_node(self, tmp_path: Path) -> None:
        """Same plaintext on two different nodes must produce different hashes."""
        reg = _registry_at(tmp_path)
        n1 = _make_node(node_id="n1")
        n2 = _make_node(node_id="n2")
        reg.add_node(n1)
        reg.add_node(n2)
        # Manually set the same token for both to test salt differentiation.
        fixed_token = "same-plaintext-token"
        reg.update_node("n1", token_hash=reg._hash_token("n1", fixed_token))
        reg.update_node("n2", token_hash=reg._hash_token("n2", fixed_token))

        h1 = reg.get_node("n1").token_hash
        h2 = reg.get_node("n2").token_hash
        assert h1 != h2

    def test_verify_node_with_empty_hash_returns_false(self, tmp_path: Path) -> None:
        reg = _registry_at(tmp_path)
        node = _make_node()
        node.token_hash = ""
        reg.add_node(node)
        assert reg.verify_token("node-test-1", "any-token") is False

    def test_pbkdf2_hash_uses_correct_algorithm(self, tmp_path: Path) -> None:
        """Cross-validate _hash_token output against a manual PBKDF2 computation."""
        reg = _registry_at(tmp_path)
        reg.add_node(_make_node(node_id="verify-me"))
        token = "test-token-value"
        expected = hashlib.pbkdf2_hmac(
            "sha256",
            token.encode(),
            "verify-me".encode(),
            iterations=100_000,
        ).hex()
        actual = reg._hash_token("verify-me", token)
        assert actual == expected


# ---------------------------------------------------------------------------
# 3. DeviceRegistry — policy modes
# ---------------------------------------------------------------------------


class TestDeviceRegistryPolicyModes:
    """Policy mode values are stored and retrievable."""

    @pytest.mark.parametrize("mode", ["full", "safe-chat", "muted"])
    def test_policy_mode_stored_correctly(self, tmp_path: Path, mode: str) -> None:
        reg = _registry_at(tmp_path)
        node = _make_node(policy_mode=mode)
        reg.add_node(node)
        assert reg.get_node(node.node_id).policy_mode == mode

    def test_policy_mode_update(self, tmp_path: Path) -> None:
        reg = _registry_at(tmp_path)
        reg.add_node(_make_node(policy_mode="full"))
        reg.update_node("node-test-1", policy_mode="muted")
        assert reg.get_node("node-test-1").policy_mode == "muted"

    def test_policy_mode_round_trips_through_disk(self, tmp_path: Path) -> None:
        reg = _registry_at(tmp_path)
        reg.add_node(_make_node(policy_mode="safe-chat"))
        reg2 = _registry_at(tmp_path)
        reg2.load()
        assert reg2.get_node("node-test-1").policy_mode == "safe-chat"

    def test_muted_node_is_reported_by_status_field(self, tmp_path: Path) -> None:
        reg = _registry_at(tmp_path)
        node = _make_node(policy_mode="muted")
        node.status = "muted"
        reg.add_node(node)
        retrieved = reg.get_node(node.node_id)
        assert retrieved.status == "muted"
        assert retrieved.policy_mode == "muted"


# ---------------------------------------------------------------------------
# 4. DeviceRegistry — duplicate handling
# ---------------------------------------------------------------------------


class TestDeviceRegistryDuplicates:
    """add_node replaces silently; PairingManager.initiate_pairing is idempotent."""

    def test_add_node_replaces_existing(self, tmp_path: Path) -> None:
        reg = _registry_at(tmp_path)
        n1 = _make_node(friendly_name="Original")
        reg.add_node(n1)
        n2 = _make_node(friendly_name="Replacement")
        reg.add_node(n2)
        assert len(reg.list_nodes()) == 1
        assert reg.get_node("node-test-1").friendly_name == "Replacement"

    def test_initiate_pairing_idempotent_on_existing_node(self, tmp_path: Path) -> None:
        reg = _registry_at(tmp_path)
        node = _make_node(node_id="pair-me", paired=False)
        reg.add_node(node)
        mgr = PairingManager(reg)
        returned_id = mgr.initiate_pairing(
            node_id="pair-me",
            friendly_name="New Name",
            room="new-room",
            ip_address="1.2.3.4",
            hardware_profile={},
        )
        assert returned_id == "pair-me"
        # Original node unchanged because it already existed.
        assert reg.get_node("pair-me").friendly_name == _make_node(node_id="pair-me").friendly_name

    def test_initiate_pairing_assigns_uuid_when_node_id_empty(self, tmp_path: Path) -> None:
        reg = _registry_at(tmp_path)
        mgr = PairingManager(reg)
        node_id = mgr.initiate_pairing(
            node_id="",
            friendly_name="Auto",
            room="hall",
            ip_address="192.168.0.5",
            hardware_profile={},
        )
        assert node_id != ""
        assert reg.get_node(node_id) is not None

    def test_two_distinct_nodes_coexist(self, tmp_path: Path) -> None:
        reg = _registry_at(tmp_path)
        reg.add_node(_make_node(node_id="a"))
        reg.add_node(_make_node(node_id="b"))
        assert len(reg.list_nodes()) == 2

    def test_remove_nonexistent_is_noop(self, tmp_path: Path) -> None:
        reg = _registry_at(tmp_path)
        reg.remove_node("ghost")  # must not raise
        assert reg.list_nodes() == []


# ---------------------------------------------------------------------------
# 5 & 6. VoiceServer — protocol state machine (auth / streaming / idle)
#         and binary PCM frame accumulation
# ---------------------------------------------------------------------------


class TestVoiceServerProtocol:
    """Tests drive the server handler methods directly with mocked WebSocket objects."""

    def _make_ws(self, frames: list[str | bytes]) -> MagicMock:
        """Return a mock WebSocket whose recv() yields *frames* in order."""
        ws = MagicMock()
        ws.remote_address = ("127.0.0.1", 9999)
        frames_iter = iter(frames)
        ws.recv = AsyncMock(side_effect=frames_iter)
        ws.send = AsyncMock()
        ws.close = AsyncMock()
        return ws

    def _async_iter_ws(self, ws: MagicMock, frames: list[str | bytes]) -> MagicMock:
        """Patch __aiter__ so that ``async for raw in ws`` yields *frames*."""
        ws.__aiter__ = MagicMock(return_value=ws)
        ws.__anext__ = AsyncMock(side_effect=list(frames) + [StopAsyncIteration()])
        return ws

    @pytest.fixture
    def registry(self, tmp_path: Path) -> DeviceRegistry:
        reg = _registry_at(tmp_path)
        node = _make_node(node_id="n1", paired=True, policy_mode="full")
        reg.add_node(node)
        reg.generate_token("n1")
        return reg

    @pytest.mark.asyncio
    async def test_auth_success_sends_auth_ok(self, registry: DeviceRegistry) -> None:
        token = registry.generate_token("n1")
        server = _make_server(registry)
        ws = MagicMock()
        ws.remote_address = ("127.0.0.1", 9000)
        ws.send = AsyncMock()
        ws.close = AsyncMock()

        result = await server._handle_auth(ws, node_id="n1", token=token)
        assert result is not None
        assert result.node_id == "n1"
        sent_payload = json.loads(ws.send.call_args_list[0][0][0])
        assert sent_payload["type"] == "auth_ok"
        assert sent_payload["node_id"] == "n1"

    @pytest.mark.asyncio
    async def test_auth_failure_wrong_token(self, registry: DeviceRegistry) -> None:
        server = _make_server(registry)
        ws = MagicMock()
        ws.remote_address = ("127.0.0.1", 9000)
        ws.send = AsyncMock()
        ws.close = AsyncMock()

        result = await server._handle_auth(ws, node_id="n1", token="wrong-token")
        assert result is None
        sent_payload = json.loads(ws.send.call_args_list[0][0][0])
        assert sent_payload["type"] == "auth_fail"

    @pytest.mark.asyncio
    async def test_auth_failure_unknown_node(self, registry: DeviceRegistry) -> None:
        server = _make_server(registry)
        ws = MagicMock()
        ws.remote_address = ("127.0.0.1", 9000)
        ws.send = AsyncMock()
        ws.close = AsyncMock()

        result = await server._handle_auth(ws, node_id="ghost", token="irrelevant")
        assert result is None
        sent_payload = json.loads(ws.send.call_args_list[0][0][0])
        assert sent_payload["type"] == "auth_fail"

    @pytest.mark.asyncio
    async def test_auth_failure_unpaired_node(self, tmp_path: Path) -> None:
        reg = _registry_at(tmp_path)
        node = _make_node(node_id="unpaired", paired=False)
        reg.add_node(node)
        token = reg.generate_token("unpaired")
        server = _make_server(reg)
        ws = MagicMock()
        ws.remote_address = ("127.0.0.1", 9000)
        ws.send = AsyncMock()
        ws.close = AsyncMock()

        result = await server._handle_auth(ws, node_id="unpaired", token=token)
        assert result is None
        payload = json.loads(ws.send.call_args_list[0][0][0])
        assert payload["type"] == "auth_fail"
        assert "not yet approved" in payload["reason"]

    @pytest.mark.asyncio
    async def test_auth_muted_node_sends_muted_frame(self, tmp_path: Path) -> None:
        reg = _registry_at(tmp_path)
        node = _make_node(node_id="muted-node", paired=True, policy_mode="muted")
        reg.add_node(node)
        token = reg.generate_token("muted-node")
        server = _make_server(reg)
        ws = MagicMock()
        ws.remote_address = ("127.0.0.1", 9000)
        ws.send = AsyncMock()
        ws.close = AsyncMock()

        result = await server._handle_auth(ws, node_id="muted-node", token=token)
        assert result is None
        payload = json.loads(ws.send.call_args_list[0][0][0])
        assert payload["type"] == "muted"

    @pytest.mark.asyncio
    async def test_message_loop_audio_session_full_cycle(self, tmp_path: Path) -> None:
        """audio_start → binary frames → audio_end triggers STT + agent + TTS."""
        reg = _registry_at(tmp_path)
        node = _make_node(node_id="n1")
        reg.add_node(node)

        stt = _stub_stt(text="set a timer")
        agent_cb = AsyncMock(return_value="Timer set.")
        tts = _stub_tts(wav_data=b"\xAB\xCD" * 256)
        server = _make_server(reg, stt=stt, tts=tts, agent_cb=agent_cb)

        pcm_frames = b"\x01\x02" * 64

        ws = _FakeWebSocket([
            json.dumps({"type": "audio_start", "sample_rate": 16000, "channels": 1}),
            pcm_frames,
            json.dumps({"type": "audio_end"}),
        ])

        await server._message_loop(ws, node)

        stt.transcribe.assert_awaited_once()
        agent_cb.assert_awaited_once()
        tts.synthesize.assert_awaited_once()

        types_sent = [m["type"] for m in ws.sent_json()]
        assert "response_text" in types_sent
        assert "audio_start" in types_sent
        assert "audio_end" in types_sent

    @pytest.mark.asyncio
    async def test_binary_frame_outside_audio_session_ignored(self, tmp_path: Path) -> None:
        reg = _registry_at(tmp_path)
        node = _make_node(node_id="n1")
        reg.add_node(node)
        stt = _stub_stt()
        server = _make_server(reg, stt=stt)

        ws = _FakeWebSocket([b"\x00\xFF\x00"])  # binary before audio_start

        await server._message_loop(ws, node)
        stt.transcribe.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_audio_buffer_overflow_closes_connection(self, tmp_path: Path) -> None:
        """Exceeding 10 MB in the audio buffer sends an error and closes."""
        reg = _registry_at(tmp_path)
        node = _make_node(node_id="n1")
        reg.add_node(node)
        server = _make_server(reg)

        # 11 MB binary payload in a single chunk.
        oversized = b"\x00" * (11 * 1024 * 1024)

        ws = _FakeWebSocket([
            json.dumps({"type": "audio_start", "sample_rate": 16000, "channels": 1}),
            oversized,
        ])

        await server._message_loop(ws, node)

        assert ws._closed
        assert any(m.get("type") == "error" for m in ws.sent_json())

    @pytest.mark.asyncio
    async def test_malformed_json_in_message_loop_sends_error(self, tmp_path: Path) -> None:
        reg = _registry_at(tmp_path)
        node = _make_node(node_id="n1")
        reg.add_node(node)
        server = _make_server(reg)

        ws = _FakeWebSocket(["not-valid-json {{{"])

        await server._message_loop(ws, node)

        assert any(m["type"] == "error" for m in ws.sent_json())

    @pytest.mark.asyncio
    async def test_audio_end_without_start_is_ignored(self, tmp_path: Path) -> None:
        reg = _registry_at(tmp_path)
        node = _make_node(node_id="n1")
        reg.add_node(node)
        stt = _stub_stt()
        server = _make_server(reg, stt=stt)

        ws = _FakeWebSocket([json.dumps({"type": "audio_end"})])

        await server._message_loop(ws, node)
        stt.transcribe.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_empty_transcript_skips_agent_call(self, tmp_path: Path) -> None:
        reg = _registry_at(tmp_path)
        node = _make_node(node_id="n1")
        reg.add_node(node)

        stt = _stub_stt(text="   ")  # whitespace-only
        agent_cb = AsyncMock(return_value="reply")
        server = _make_server(reg, stt=stt, agent_cb=agent_cb)

        ws = _FakeWebSocket([
            json.dumps({"type": "audio_start", "sample_rate": 16000, "channels": 1}),
            b"\x00" * 100,
            json.dumps({"type": "audio_end"}),
        ])

        await server._message_loop(ws, node)
        agent_cb.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_debug_transcripts_sends_transcript_frame(self, tmp_path: Path) -> None:
        reg = _registry_at(tmp_path)
        node = _make_node(node_id="n1")
        reg.add_node(node)

        stt = _stub_stt(text="debug me", confidence=0.88)
        tts = _stub_tts()
        agent_cb = AsyncMock(return_value="ok")
        pairing_mgr = PairingManager(reg)
        presence = PresenceStore(reg)
        server = VoiceServer(
            registry=reg,
            pairing_manager=pairing_mgr,
            presence_store=presence,
            stt_engine=stt,
            tts_engine=tts,
            agent_callback=agent_cb,
            debug_transcripts=True,
        )

        ws = _FakeWebSocket([
            json.dumps({"type": "audio_start", "sample_rate": 16000, "channels": 1}),
            b"\x01\x02" * 50,
            json.dumps({"type": "audio_end"}),
        ])

        await server._message_loop(ws, node)

        types_sent = [m["type"] for m in ws.sent_json()]
        assert "transcript" in types_sent
        transcript_frame = next(m for m in ws.sent_json() if m["type"] == "transcript")
        assert transcript_frame["text"] == "debug me"
        assert abs(transcript_frame["confidence"] - 0.88) < 0.001


# ---------------------------------------------------------------------------
# 6. VoiceServer binary frame handling (detailed accumulation)
# ---------------------------------------------------------------------------


class TestVoiceServerBinaryFrames:
    """Verify multi-chunk PCM accumulation reaches the STT engine intact."""

    @pytest.mark.asyncio
    async def test_multiple_binary_chunks_concatenated(self, tmp_path: Path) -> None:
        reg = _registry_at(tmp_path)
        node = _make_node(node_id="n1")
        reg.add_node(node)

        received_audio: list[bytes] = []

        async def capture_stt(audio: bytes, sample_rate: int, channels: int) -> TranscriptionResult:
            received_audio.append(audio)
            return TranscriptionResult(text="chunks", confidence=0.9, processing_ms=5)

        stt = MagicMock(spec=STTEngine)
        stt.name = "capture-stt"
        stt.load = MagicMock()
        stt.unload = MagicMock()
        stt.transcribe = capture_stt
        agent_cb = AsyncMock(return_value="reply")
        tts = _stub_tts()
        server = _make_server(reg, stt=stt, tts=tts, agent_cb=agent_cb)

        chunk1 = b"\xAA" * 100
        chunk2 = b"\xBB" * 200
        chunk3 = b"\xCC" * 50

        ws = _FakeWebSocket([
            json.dumps({"type": "audio_start", "sample_rate": 16000, "channels": 1}),
            chunk1,
            chunk2,
            chunk3,
            json.dumps({"type": "audio_end"}),
        ])

        await server._message_loop(ws, node)
        assert len(received_audio) == 1
        assert received_audio[0] == chunk1 + chunk2 + chunk3

    @pytest.mark.asyncio
    async def test_sample_rate_clipped_to_bounds(self, tmp_path: Path) -> None:
        """Sample-rate values outside [8000, 48000] are clipped, not rejected."""
        reg = _registry_at(tmp_path)
        node = _make_node(node_id="n1")
        reg.add_node(node)

        captured_rates: list[int] = []

        async def capture_stt(audio: bytes, sample_rate: int, channels: int) -> TranscriptionResult:
            captured_rates.append(sample_rate)
            return TranscriptionResult(text="ok", confidence=1.0, processing_ms=1)

        stt = MagicMock(spec=STTEngine)
        stt.name = "rate-stt"
        stt.load = MagicMock()
        stt.unload = MagicMock()
        stt.transcribe = capture_stt
        server = _make_server(reg, stt=stt)

        # Attempt to send an out-of-bounds sample rate (100 Hz is below minimum).
        ws = _FakeWebSocket([
            json.dumps({"type": "audio_start", "sample_rate": 100, "channels": 1}),
            b"\x00" * 32,
            json.dumps({"type": "audio_end"}),
        ])

        await server._message_loop(ws, node)
        assert captured_rates == [8000]  # clipped to minimum

    @pytest.mark.asyncio
    async def test_audio_buffer_reset_between_sessions(self, tmp_path: Path) -> None:
        """A second audio_start resets the buffer; no data leaks between sessions."""
        reg = _registry_at(tmp_path)
        node = _make_node(node_id="n1")
        reg.add_node(node)

        received_audio: list[bytes] = []

        async def capture_stt(audio: bytes, sample_rate: int, channels: int) -> TranscriptionResult:
            received_audio.append(audio)
            return TranscriptionResult(text="session", confidence=0.9, processing_ms=1)

        stt = MagicMock(spec=STTEngine)
        stt.name = "buf-stt"
        stt.load = MagicMock()
        stt.unload = MagicMock()
        stt.transcribe = capture_stt
        agent_cb = AsyncMock(return_value="ok")
        server = _make_server(reg, stt=stt, agent_cb=agent_cb)

        session1_data = b"\x11" * 64
        session2_data = b"\x22" * 32

        ws = _FakeWebSocket([
            json.dumps({"type": "audio_start", "sample_rate": 16000, "channels": 1}),
            session1_data,
            json.dumps({"type": "audio_end"}),
            json.dumps({"type": "audio_start", "sample_rate": 16000, "channels": 1}),
            session2_data,
            json.dumps({"type": "audio_end"}),
        ])

        await server._message_loop(ws, node)
        assert len(received_audio) == 2
        assert received_audio[0] == session1_data
        assert received_audio[1] == session2_data  # no contamination from session 1


# ---------------------------------------------------------------------------
# 7. VoiceServer — concurrent connections tracking
# ---------------------------------------------------------------------------


class TestVoiceServerConcurrentConnections:
    """get_connected_nodes and get_status reflect the live connection set."""

    def test_initial_state_no_connected_nodes(self, tmp_path: Path) -> None:
        reg = _registry_at(tmp_path)
        server = _make_server(reg)
        assert server.get_connected_nodes() == []

    def test_get_status_reflects_running_flag(self, tmp_path: Path) -> None:
        reg = _registry_at(tmp_path)
        server = _make_server(reg)
        status = server.get_status()
        assert status["running"] is False
        assert status["connected_nodes"] == 0

    def test_connected_nodes_count_in_status(self, tmp_path: Path) -> None:
        reg = _registry_at(tmp_path)
        server = _make_server(reg)
        # Manually simulate two connected nodes.
        server._connected_nodes.add("n1")
        server._connected_nodes.add("n2")
        status = server.get_status()
        assert status["connected_nodes"] == 2

    def test_get_connected_nodes_snapshot_is_a_copy(self, tmp_path: Path) -> None:
        reg = _registry_at(tmp_path)
        server = _make_server(reg)
        server._connected_nodes.add("n1")
        snapshot = server.get_connected_nodes()
        snapshot.append("n99")
        assert "n99" not in server._connected_nodes

    def test_connection_limit_guard(self, tmp_path: Path) -> None:
        """When at capacity, _active_connections >= 50, new connections are refused."""
        from missy.channels.voice import server as server_module

        reg = _registry_at(tmp_path)
        voice_server = _make_server(reg)
        voice_server._active_connections = server_module._MAX_CONCURRENT_CONNECTIONS

        ws = MagicMock()
        ws.remote_address = ("127.0.0.1", 9001)
        ws.close = AsyncMock()
        ws.recv = AsyncMock()

        import asyncio

        asyncio.get_event_loop().run_until_complete(voice_server._handle_connection(ws))
        ws.close.assert_awaited_with(1013, "Server at capacity")

    def test_status_includes_engine_names(self, tmp_path: Path) -> None:
        reg = _registry_at(tmp_path)
        stt = _stub_stt()
        tts = _stub_tts()
        server = _make_server(reg, stt=stt, tts=tts)
        status = server.get_status()
        assert status["stt_engine"] == "stub-stt"
        assert status["tts_engine"] == "stub-tts"


# ---------------------------------------------------------------------------
# 8. WebhookChannel — receive pipeline
# ---------------------------------------------------------------------------


def _start_webhook(**kwargs) -> tuple[WebhookChannel, int]:
    port = _free_port()
    ch = WebhookChannel(host="127.0.0.1", port=port, **kwargs)
    ch.start()
    time.sleep(0.05)
    return ch, port


def _post(
    port: int,
    body: bytes,
    content_type: str = "application/json",
    extra_headers: dict[str, str] | None = None,
) -> http.client.HTTPResponse:
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
    headers = {
        "Content-Type": content_type,
        "Content-Length": str(len(body)),
    }
    if extra_headers:
        headers.update(extra_headers)
    conn.request("POST", "/", body=body, headers=headers)
    return conn.getresponse()


class TestWebhookChannelReceive:
    def test_valid_post_returns_202(self) -> None:
        ch, port = _start_webhook()
        try:
            body = json.dumps({"prompt": "hello world"}).encode()
            resp = _post(port, body)
            assert resp.status == 202
        finally:
            ch.stop()

    def test_message_queued_after_post(self) -> None:
        ch, port = _start_webhook()
        try:
            body = json.dumps({"prompt": "do something"}).encode()
            _post(port, body)
            msg = ch.receive()
            assert msg is not None
            assert msg.content == "do something"
            assert msg.channel == "webhook"
        finally:
            ch.stop()

    def test_sender_field_propagated(self) -> None:
        ch, port = _start_webhook()
        try:
            body = json.dumps({"prompt": "ping", "sender": "ci-bot"}).encode()
            _post(port, body)
            msg = ch.receive()
            assert msg is not None
            assert msg.sender == "ci-bot"
        finally:
            ch.stop()

    def test_empty_queue_returns_none(self) -> None:
        ch, port = _start_webhook()
        try:
            assert ch.receive() is None
        finally:
            ch.stop()

    def test_sender_sanitised_strips_control_chars(self) -> None:
        ch, port = _start_webhook()
        try:
            body = json.dumps({"prompt": "test", "sender": "valid\x00evil"}).encode()
            _post(port, body)
            msg = ch.receive()
            assert msg is not None
            assert "\x00" not in msg.sender
        finally:
            ch.stop()

    def test_get_request_returns_405(self) -> None:
        ch, port = _start_webhook()
        try:
            conn = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
            conn.request("GET", "/")
            resp = conn.getresponse()
            assert resp.status == 405
        finally:
            ch.stop()

    def test_multiple_messages_queued_in_order(self) -> None:
        ch, port = _start_webhook()
        try:
            for i in range(3):
                body = json.dumps({"prompt": f"msg-{i}"}).encode()
                _post(port, body)
            messages = []
            for _ in range(3):
                m = ch.receive()
                if m is not None:
                    messages.append(m.content)
            assert messages == ["msg-0", "msg-1", "msg-2"]
        finally:
            ch.stop()


# ---------------------------------------------------------------------------
# 9. WebhookChannel — HMAC authentication
# ---------------------------------------------------------------------------


class TestWebhookChannelAuth:
    def _sign(self, secret: str, body: bytes) -> str:
        return "sha256=" + _hmac.new(secret.encode(), body, hashlib.sha256).hexdigest()

    def test_correct_signature_accepted(self) -> None:
        ch, port = _start_webhook(secret="my-secret")
        try:
            body = json.dumps({"prompt": "signed"}).encode()
            sig = self._sign("my-secret", body)
            resp = _post(port, body, extra_headers={"X-Missy-Signature": sig})
            assert resp.status == 202
        finally:
            ch.stop()

    def test_wrong_signature_rejected_with_401(self) -> None:
        ch, port = _start_webhook(secret="my-secret")
        try:
            body = json.dumps({"prompt": "tampered"}).encode()
            resp = _post(port, body, extra_headers={"X-Missy-Signature": "sha256=badhash"})
            assert resp.status == 401
        finally:
            ch.stop()

    def test_missing_signature_rejected_with_401(self) -> None:
        ch, port = _start_webhook(secret="my-secret")
        try:
            body = json.dumps({"prompt": "no-sig"}).encode()
            resp = _post(port, body)
            assert resp.status == 401
        finally:
            ch.stop()

    def test_no_secret_configured_accepts_any_post(self) -> None:
        ch, port = _start_webhook()
        try:
            body = json.dumps({"prompt": "open"}).encode()
            resp = _post(port, body)
            assert resp.status == 202
        finally:
            ch.stop()

    def test_wrong_signature_does_not_enqueue_message(self) -> None:
        ch, port = _start_webhook(secret="strict")
        try:
            body = json.dumps({"prompt": "stealth"}).encode()
            _post(port, body, extra_headers={"X-Missy-Signature": "sha256=fake"})
            assert ch.receive() is None
        finally:
            ch.stop()


# ---------------------------------------------------------------------------
# 10. WebhookChannel — error handling
# ---------------------------------------------------------------------------


class TestWebhookChannelErrors:
    def test_malformed_json_returns_400(self) -> None:
        ch, port = _start_webhook()
        try:
            resp = _post(port, b"not-json-at-all")
            assert resp.status == 400
        finally:
            ch.stop()

    def test_missing_prompt_field_returns_400(self) -> None:
        ch, port = _start_webhook()
        try:
            body = json.dumps({"other": "field"}).encode()
            resp = _post(port, body)
            assert resp.status == 400
        finally:
            ch.stop()

    def test_empty_prompt_returns_400(self) -> None:
        ch, port = _start_webhook()
        try:
            body = json.dumps({"prompt": "   "}).encode()
            resp = _post(port, body)
            assert resp.status == 400
        finally:
            ch.stop()

    def test_wrong_content_type_returns_415(self) -> None:
        ch, port = _start_webhook()
        try:
            body = b"prompt=hello"
            resp = _post(port, body, content_type="application/x-www-form-urlencoded")
            assert resp.status == 415
        finally:
            ch.stop()

    def test_oversized_payload_returns_413(self) -> None:
        ch, port = _start_webhook()
        try:
            big_prompt = "x" * (33_000)
            body = json.dumps({"prompt": big_prompt}).encode()
            resp = _post(port, body)
            assert resp.status == 413
        finally:
            ch.stop()

    def test_negative_content_length_returns_400(self) -> None:
        ch, port = _start_webhook()
        try:
            body = json.dumps({"prompt": "hello"}).encode()
            resp = _post(port, body, extra_headers={"Content-Length": "-1"})
            assert resp.status == 400
        finally:
            ch.stop()

    def test_rate_limit_returns_429(self) -> None:
        import missy.channels.webhook as wh_mod

        original = wh_mod._RATE_LIMIT_REQUESTS
        wh_mod._RATE_LIMIT_REQUESTS = 2
        try:
            ch, port = _start_webhook()
            try:
                body = json.dumps({"prompt": "ok"}).encode()
                for _ in range(2):
                    _post(port, body)
                resp = _post(port, body)
                assert resp.status == 429
            finally:
                ch.stop()
        finally:
            wh_mod._RATE_LIMIT_REQUESTS = original

    def test_channel_name_is_webhook(self) -> None:
        ch = WebhookChannel()
        assert ch.name == "webhook"


# ---------------------------------------------------------------------------
# 11. CLIChannel — basic operations
# ---------------------------------------------------------------------------


class TestCLIChannelBasicOperations:
    def test_receive_returns_channel_message(self) -> None:
        ch = CLIChannel()
        with patch("builtins.input", return_value="hello cli"):
            msg = ch.receive()
        assert msg is not None
        assert isinstance(msg, ChannelMessage)
        assert msg.content == "hello cli"
        assert msg.sender == "user"
        assert msg.channel == "cli"

    def test_receive_eof_returns_none(self) -> None:
        ch = CLIChannel()
        with patch("builtins.input", side_effect=EOFError):
            result = ch.receive()
        assert result is None

    def test_receive_keyboard_interrupt_returns_none(self) -> None:
        ch = CLIChannel()
        with patch("builtins.input", side_effect=KeyboardInterrupt):
            result = ch.receive()
        assert result is None

    def test_send_writes_to_stdout(self) -> None:
        ch = CLIChannel()
        buf = StringIO()
        with patch("sys.stdout", buf):
            ch.send("hello output")
        assert "hello output" in buf.getvalue()

    def test_receive_preserves_whitespace_in_content(self) -> None:
        ch = CLIChannel()
        with patch("builtins.input", return_value="  spaced out  "):
            msg = ch.receive()
        assert msg is not None
        assert msg.content == "  spaced out  "

    def test_receive_empty_string_returns_message(self) -> None:
        ch = CLIChannel()
        with patch("builtins.input", return_value=""):
            msg = ch.receive()
        assert msg is not None
        assert msg.content == ""

    def test_metadata_is_empty_dict(self) -> None:
        ch = CLIChannel()
        with patch("builtins.input", return_value="test"):
            msg = ch.receive()
        assert msg is not None
        assert msg.metadata == {}

    def test_custom_prompt_passed_to_input(self) -> None:
        ch = CLIChannel(prompt=">> ")
        with patch("builtins.input", return_value="x") as mock_input:
            ch.receive()
        mock_input.assert_called_once_with(">> ")

    def test_channel_name_attribute(self) -> None:
        assert CLIChannel.name == "cli"


# ---------------------------------------------------------------------------
# 12. Channel factory / registry — correct type instantiation
# ---------------------------------------------------------------------------


class TestChannelTypeContracts:
    """Verify that channel classes satisfy the BaseChannel contract and are
    distinguishable by their ``name`` attribute."""

    def test_cli_channel_is_basechannel(self) -> None:
        ch = CLIChannel()
        assert isinstance(ch, BaseChannel)

    def test_webhook_channel_is_basechannel(self) -> None:
        ch = WebhookChannel()
        assert isinstance(ch, BaseChannel)

    def test_cli_channel_name(self) -> None:
        assert CLIChannel().name == "cli"

    def test_webhook_channel_name(self) -> None:
        assert WebhookChannel().name == "webhook"

    def test_channel_message_defaults(self) -> None:
        msg = ChannelMessage(content="x")
        assert msg.sender == "user"
        assert msg.channel == "cli"
        assert msg.metadata == {}

    def test_channel_message_content_preserved(self) -> None:
        msg = ChannelMessage(content="important data", sender="bot", channel="webhook")
        assert msg.content == "important data"
        assert msg.sender == "bot"
        assert msg.channel == "webhook"

    def test_basechannel_enforces_abstract_methods(self) -> None:
        """Concrete subclasses that omit receive/send must fail to instantiate."""

        class Incomplete(BaseChannel):
            name = "incomplete"

        with pytest.raises(TypeError):
            Incomplete()  # type: ignore[abstract]

    def test_webhook_channel_send_censors_output(self) -> None:
        """WebhookChannel.send() logs (does not raise) and censors the message."""
        ch = WebhookChannel()
        with patch("missy.security.censor.censor_response", return_value="[REDACTED]") as mock_c:
            ch.send("some secret output")
        mock_c.assert_called_once()

    def test_channel_message_metadata_is_independent_across_instances(self) -> None:
        m1 = ChannelMessage(content="a")
        m2 = ChannelMessage(content="b")
        m1.metadata["k"] = "v"
        assert "k" not in m2.metadata

    def test_voice_channel_package_exports(self) -> None:
        """The voice channel package exports the expected public symbols."""
        import missy.channels.voice as vc

        assert hasattr(vc, "VoiceServer")
        assert hasattr(vc, "DeviceRegistry")
        assert hasattr(vc, "EdgeNode")
        assert hasattr(vc, "PairingManager")
        assert hasattr(vc, "PresenceStore")
        assert hasattr(vc, "VoiceChannel")
