"""End-to-end WebSocket protocol tests for VoiceServer.

These tests exercise the server's connection-handling methods directly, using
mock WebSocket objects with async send/recv/close methods.  No real TCP socket
is bound; all I/O is intercepted via unittest.mock.

Test coverage:
- Connection flood protection (>50 concurrent connections rejected with 1013)
- Auth timeout (no first frame within 10 s closes with 1008)
- Binary first frame rejected
- Malformed JSON first frame rejected
- Unexpected message type as first frame rejected
- Audio buffer overflow (>10 MB) causes disconnect
- Audio session state machine: binary frames outside audio session are ignored
- audio_end outside session is ignored
- Sample-rate clamping (< 8000 -> 8000, > 48000 -> 48000)
- Channel clamping (< 1 -> 1, > 2 -> 2)
- Malformed JSON during authenticated message loop sends error but continues
- Spurious auth frame during authenticated loop is silently ignored
- Unknown message type during authenticated loop is silently skipped
- get_status and get_connected_nodes introspection
- start() idempotent when already running
- stop() idempotent when not running
"""

from __future__ import annotations

import contextlib
import json
from dataclasses import dataclass
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from missy.channels.voice.stt.base import TranscriptionResult
from missy.channels.voice.tts.base import AudioBuffer

# ---------------------------------------------------------------------------
# Shared mock edge-node dataclass
# ---------------------------------------------------------------------------


@dataclass
class _Node:
    node_id: str = "proto-node"
    friendly_name: str = "Protocol Test Node"
    room: str = "lab"
    paired: bool = True
    policy_mode: str = "full"
    ip_address: str = "10.0.0.1"
    hardware_profile: dict = None
    audio_logging: bool = False
    audio_log_dir: str = ""

    def __post_init__(self) -> None:
        if self.hardware_profile is None:
            self.hardware_profile = {}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _MockWebSocket:
    """Mock WebSocket that satisfies both the ``recv()`` call interface (used by
    ``_handle_connection`` via ``asyncio.wait_for``) and the ``async for raw in
    websocket`` iteration interface (used by ``_message_loop``).

    Items in *messages* are returned in order.  When the list is exhausted a
    :class:`~websockets.exceptions.ConnectionClosed` is raised to cleanly exit
    any loop.  Any item that is already a :class:`BaseException` instance is
    raised rather than returned.
    """

    def __init__(
        self,
        messages: list[Any],
        remote: tuple[str, int] = ("127.0.0.1", 55000),
    ) -> None:
        import websockets.exceptions

        self._messages = list(messages)
        self._index = 0
        self._closed_exc = websockets.exceptions.ConnectionClosed(None, None)
        self.remote_address = remote
        self.send = AsyncMock()
        self.close = AsyncMock()

    def _next_item(self) -> Any:
        """Pop and return the next message, raising ConnectionClosed when exhausted."""
        if self._index >= len(self._messages):
            raise self._closed_exc
        item = self._messages[self._index]
        self._index += 1
        if isinstance(item, BaseException):
            raise item
        return item

    async def recv(self) -> Any:
        """Called by ``_handle_connection`` through ``asyncio.wait_for``."""
        return self._next_item()

    def __aiter__(self) -> _MockWebSocket:
        """Support ``async for raw in websocket`` used by ``_message_loop``."""
        return self

    async def __anext__(self) -> Any:
        """Raise ``StopAsyncIteration`` when the connection is 'closed'."""
        import websockets.exceptions

        try:
            return self._next_item()
        except websockets.exceptions.ConnectionClosed:
            raise StopAsyncIteration from None


def _make_ws(
    messages: list[Any] | None = None,
    *,
    remote: tuple[str, int] = ("127.0.0.1", 55000),
) -> _MockWebSocket:
    """Convenience constructor for :class:`_MockWebSocket`."""
    return _MockWebSocket(messages=list(messages or []), remote=remote)


def _sent_json_frames(ws: AsyncMock) -> list[dict[str, Any]]:
    """Collect all JSON frames that were sent over *ws* (ignores binary sends)."""
    frames = []
    for call in ws.send.await_args_list:
        arg = call.args[0] if call.args else call.kwargs.get("data", "")
        if isinstance(arg, str):
            with contextlib.suppress(json.JSONDecodeError):
                frames.append(json.loads(arg))
    return frames


def _sent_types(ws: AsyncMock) -> list[str]:
    """Return the ``type`` field of every JSON frame sent over *ws*."""
    return [f.get("type", "") for f in _sent_json_frames(ws)]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def mock_registry() -> MagicMock:
    reg = MagicMock()
    reg.verify_token = MagicMock(return_value=True)
    reg.get_node = MagicMock(return_value=_Node())
    reg.mark_online = MagicMock()
    reg.mark_offline = MagicMock()
    return reg


@pytest.fixture()
def mock_pairing() -> MagicMock:
    pm = MagicMock()
    pm.initiate_pairing = MagicMock(return_value="new-node-99")
    return pm


@pytest.fixture()
def mock_presence() -> MagicMock:
    return MagicMock()


@pytest.fixture()
def mock_stt() -> MagicMock:
    stt = MagicMock()
    stt.name = "mock-stt"
    stt.load = MagicMock()
    stt.unload = MagicMock()
    stt.transcribe = AsyncMock(
        return_value=TranscriptionResult(
            text="test transcript", confidence=0.9, processing_ms=80, language="en"
        )
    )
    return stt


@pytest.fixture()
def mock_tts() -> MagicMock:
    tts = MagicMock()
    tts.name = "mock-tts"
    tts.load = MagicMock()
    tts.unload = MagicMock()
    tts.synthesize = AsyncMock(
        return_value=AudioBuffer(data=b"\x00" * 8192, sample_rate=22050, channels=1, format="wav")
    )
    return tts


@pytest.fixture()
def mock_agent() -> AsyncMock:
    return AsyncMock(return_value="Agent says hello")


@pytest.fixture()
def server(
    mock_registry: MagicMock,
    mock_pairing: MagicMock,
    mock_presence: MagicMock,
    mock_stt: MagicMock,
    mock_tts: MagicMock,
    mock_agent: AsyncMock,
) -> Any:
    from missy.channels.voice.server import VoiceServer

    return VoiceServer(
        registry=mock_registry,
        pairing_manager=mock_pairing,
        presence_store=mock_presence,
        stt_engine=mock_stt,
        tts_engine=mock_tts,
        agent_callback=mock_agent,
        host="127.0.0.1",
        port=0,
    )


# ---------------------------------------------------------------------------
# 1. Connection flood protection
# ---------------------------------------------------------------------------


class TestConnectionFloodProtection:
    async def test_connection_rejected_at_limit(self, server: Any) -> None:
        """_handle_connection must close with 1013 when at the connection cap."""
        from missy.channels.voice.server import _MAX_CONCURRENT_CONNECTIONS

        # Pin the counter at the maximum.
        server._active_connections = _MAX_CONCURRENT_CONNECTIONS
        ws = _make_ws()  # send nothing — should never be reached

        await server._handle_connection(ws)

        ws.close.assert_awaited_once_with(1013, "Server at capacity")
        # The counter must not have been incremented beyond the cap.
        assert server._active_connections == _MAX_CONCURRENT_CONNECTIONS

    async def test_connection_accepted_one_below_limit(self, server: Any) -> None:
        """One slot below the cap: the connection must proceed past flood check."""
        from missy.channels.voice.server import _MAX_CONCURRENT_CONNECTIONS

        server._active_connections = _MAX_CONCURRENT_CONNECTIONS - 1

        auth_frame = json.dumps({"type": "auth", "node_id": "proto-node", "token": "good-token"})
        ws = _make_ws([auth_frame])

        await server._handle_connection(ws)

        # close was NOT called with 1013.
        for call in ws.close.await_args_list:
            code = call.args[0] if call.args else None
            assert code != 1013, "Connection should not have been rejected with 1013"

    async def test_active_connections_decremented_on_exit(self, server: Any) -> None:
        """_active_connections must be decremented even when the connection ends cleanly."""
        auth_frame = json.dumps({"type": "auth", "node_id": "proto-node", "token": "good-token"})
        ws = _make_ws([auth_frame])  # recv will raise ConnectionClosed after auth

        before = server._active_connections
        await server._handle_connection(ws)
        assert server._active_connections == before


# ---------------------------------------------------------------------------
# 2. Auth timeout
# ---------------------------------------------------------------------------


class TestAuthTimeout:
    async def test_connection_closed_on_auth_timeout(self, server: Any) -> None:
        """No first frame within the timeout window must close with 1008."""

        # recv raises TimeoutError to simulate asyncio.wait_for expiry.
        ws = _make_ws([TimeoutError()])

        await server._handle_connection(ws)

        ws.close.assert_awaited_once_with(1008, "Authentication timeout")

    async def test_active_connections_decremented_after_timeout(self, server: Any) -> None:
        """The connection counter must be restored even on timeout."""
        ws = _make_ws([TimeoutError()])
        before = server._active_connections
        await server._handle_connection(ws)
        assert server._active_connections == before


# ---------------------------------------------------------------------------
# 3. Binary first frame
# ---------------------------------------------------------------------------


class TestBinaryFirstFrame:
    async def test_binary_first_frame_rejected(self, server: Any) -> None:
        """A binary frame before auth must cause close with 1008."""
        ws = _make_ws([b"\x00\x01\x02\x03"])

        await server._handle_connection(ws)

        ws.close.assert_awaited_once_with(1008, "First frame must be JSON auth or pair_request")

    async def test_binary_first_frame_no_auth_ok(self, server: Any) -> None:
        """auth_ok must never be sent when the first frame is binary."""
        ws = _make_ws([b"\xff" * 512])

        await server._handle_connection(ws)

        assert "auth_ok" not in _sent_types(ws)


# ---------------------------------------------------------------------------
# 4. Malformed JSON first frame
# ---------------------------------------------------------------------------


class TestMalformedJsonFirstFrame:
    async def test_malformed_json_rejected(self, server: Any) -> None:
        """An invalid JSON string as first frame must close with 1008."""
        ws = _make_ws(["{not valid json"])

        await server._handle_connection(ws)

        ws.close.assert_awaited_once_with(1008, "Malformed JSON")

    async def test_malformed_json_no_auth_ok(self, server: Any) -> None:
        ws = _make_ws(["<xml>definitely not json</xml>"])

        await server._handle_connection(ws)

        assert "auth_ok" not in _sent_types(ws)


# ---------------------------------------------------------------------------
# 5. Unexpected message type as first frame
# ---------------------------------------------------------------------------


class TestUnexpectedFirstFrameType:
    async def test_heartbeat_as_first_frame_rejected(self, server: Any) -> None:
        """A heartbeat as the first frame must close with 1008 (Protocol violation)."""
        ws = _make_ws([json.dumps({"type": "heartbeat", "node_id": "proto-node"})])

        await server._handle_connection(ws)

        ws.close.assert_awaited_once_with(1008, "Protocol violation")

    async def test_error_frame_sends_error_json(self, server: Any) -> None:
        """An error JSON frame must be sent before the protocol-violation close."""
        ws = _make_ws([json.dumps({"type": "heartbeat"})])

        await server._handle_connection(ws)

        assert "error" in _sent_types(ws)

    async def test_audio_start_as_first_frame_rejected(self, server: Any) -> None:
        """audio_start before auth must be treated as protocol violation."""
        ws = _make_ws([json.dumps({"type": "audio_start", "sample_rate": 16000, "channels": 1})])

        await server._handle_connection(ws)

        ws.close.assert_awaited_once_with(1008, "Protocol violation")

    async def test_empty_type_as_first_frame_rejected(self, server: Any) -> None:
        """A JSON frame with no type field is also a protocol violation."""
        ws = _make_ws([json.dumps({"payload": "mystery"})])

        await server._handle_connection(ws)

        ws.close.assert_awaited_once_with(1008, "Protocol violation")


# ---------------------------------------------------------------------------
# 6. Audio buffer overflow
# ---------------------------------------------------------------------------


class TestAudioBufferOverflow:
    async def test_overflow_sends_error_and_closes(self, server: Any) -> None:
        """Accumulating more than 10 MB of audio must close with 1009."""
        node = _Node()
        _make_ws()

        # Simulate: audio_start, then one giant binary frame exceeding 10 MB.
        # We exercise _message_loop directly.
        from missy.channels.voice.server import _MAX_AUDIO_BYTES

        oversized_payload = b"\x00" * (_MAX_AUDIO_BYTES + 1)

        # Build an async generator to feed the server's `async for raw in websocket` loop.

        frames: list[Any] = [
            json.dumps({"type": "audio_start", "sample_rate": 16000, "channels": 1}),
            oversized_payload,
        ]
        ws2 = _make_ws(frames)

        await server._message_loop(ws2, node)

        ws2.close.assert_awaited_once_with(1009, "Audio buffer too large")
        assert "error" in _sent_types(ws2)

    async def test_overflow_error_message_mentions_limit(self, server: Any) -> None:
        """The error frame must reference the 10 MB limit."""
        from missy.channels.voice.server import _MAX_AUDIO_BYTES

        node = _Node()
        frames: list[Any] = [
            json.dumps({"type": "audio_start", "sample_rate": 16000, "channels": 1}),
            b"\x00" * (_MAX_AUDIO_BYTES + 1),
        ]
        ws = _make_ws(frames)

        await server._message_loop(ws, node)

        error_frames = [f for f in _sent_json_frames(ws) if f.get("type") == "error"]
        assert error_frames, "Expected at least one error frame"
        assert "10 MB" in error_frames[0].get("message", "")


# ---------------------------------------------------------------------------
# 7. Binary frames outside audio session are ignored
# ---------------------------------------------------------------------------


class TestBinaryOutsideAudioSession:
    async def test_binary_before_audio_start_ignored(self, server: Any) -> None:
        """Binary frames arriving before audio_start must be discarded silently."""
        node = _Node()

        # Send binary data with no preceding audio_start, then a heartbeat to
        # confirm the connection is still alive and processing.
        frames: list[Any] = [
            b"\xde\xad\xbe\xef",
            json.dumps(
                {
                    "type": "heartbeat",
                    "node_id": "proto-node",
                    "occupancy": None,
                    "noise_level": None,
                    "wake_word_fp": False,
                }
            ),
        ]
        ws = _make_ws(frames)

        # Patch presence_store.update so we can check heartbeat was processed.
        server._presence_store.update = MagicMock()

        await server._message_loop(ws, node)

        # Connection must not have been closed due to the rogue binary frame.
        ws.close.assert_not_awaited()
        # Heartbeat must have been dispatched (presence update called).
        server._presence_store.update.assert_called_once()

    async def test_binary_after_audio_end_ignored(self, server: Any) -> None:
        """Binary frames after the session ends (but before a new audio_start) are ignored."""
        from missy.channels.voice.stt.base import TranscriptionResult

        node = _Node()
        server._stt.transcribe = AsyncMock(
            return_value=TranscriptionResult(
                text="hello", confidence=0.9, processing_ms=50, language="en"
            )
        )
        server._agent_callback = AsyncMock(return_value="ok")

        frames: list[Any] = [
            json.dumps({"type": "audio_start", "sample_rate": 16000, "channels": 1}),
            b"\x00" * 100,
            json.dumps({"type": "audio_end"}),
            # Rogue binary frame *after* the session ended.
            b"\xff" * 100,
        ]
        ws = _make_ws(frames)

        await server._message_loop(ws, node)

        ws.close.assert_not_awaited()


# ---------------------------------------------------------------------------
# 8. audio_end outside session
# ---------------------------------------------------------------------------


class TestAudioEndOutsideSession:
    async def test_audio_end_without_audio_start_ignored(self, server: Any) -> None:
        """audio_end with no preceding audio_start must be silently skipped."""
        node = _Node()

        frames: list[Any] = [
            json.dumps({"type": "audio_end"}),
        ]
        ws = _make_ws(frames)

        await server._message_loop(ws, node)

        # No close, no error.
        ws.close.assert_not_awaited()
        assert "error" not in _sent_types(ws)
        # Agent must not have been called.
        server._agent_callback.assert_not_awaited()

    async def test_audio_end_outside_session_does_not_call_stt(self, server: Any) -> None:
        """STT must not be invoked when audio_end is received outside a session."""
        node = _Node()
        ws = _make_ws([json.dumps({"type": "audio_end"})])

        await server._message_loop(ws, node)

        server._stt.transcribe.assert_not_awaited()


# ---------------------------------------------------------------------------
# 9. Sample-rate clamping
# ---------------------------------------------------------------------------


class TestSampleRateClamping:
    async def _run_audio_start(self, server: Any, sample_rate: Any) -> tuple[int, int]:
        """Drive _message_loop with a single audio_start and return captured
        (audio_sample_rate, audio_channels) via a spy on _handle_audio."""
        node = _Node()
        captured: dict[str, Any] = {}

        async def _spy_handle_audio(ws, node, audio_buffer, sample_rate, channels=1):
            captured["sample_rate"] = sample_rate
            captured["channels"] = channels

        server._handle_audio = _spy_handle_audio

        frames: list[Any] = [
            json.dumps({"type": "audio_start", "sample_rate": sample_rate, "channels": 1}),
            b"\x00" * 16,
            json.dumps({"type": "audio_end"}),
        ]
        ws = _make_ws(frames)
        await server._message_loop(ws, node)
        return captured.get("sample_rate", -1), captured.get("channels", -1)

    async def test_sample_rate_below_minimum_clamped_to_8000(self, server: Any) -> None:
        rate, _ = await self._run_audio_start(server, sample_rate=100)
        assert rate == 8000

    async def test_sample_rate_at_minimum_accepted(self, server: Any) -> None:
        rate, _ = await self._run_audio_start(server, sample_rate=8000)
        assert rate == 8000

    async def test_sample_rate_in_range_accepted(self, server: Any) -> None:
        rate, _ = await self._run_audio_start(server, sample_rate=16000)
        assert rate == 16000

    async def test_sample_rate_above_maximum_clamped_to_48000(self, server: Any) -> None:
        rate, _ = await self._run_audio_start(server, sample_rate=96000)
        assert rate == 48000

    async def test_sample_rate_at_maximum_accepted(self, server: Any) -> None:
        rate, _ = await self._run_audio_start(server, sample_rate=48000)
        assert rate == 48000

    async def test_non_numeric_sample_rate_defaults_to_16000(self, server: Any) -> None:
        """A non-numeric sample_rate must fall back to the default 16000 Hz."""
        rate, _ = await self._run_audio_start(server, sample_rate="fast")
        assert rate == 16000


# ---------------------------------------------------------------------------
# 10. Channel clamping
# ---------------------------------------------------------------------------


class TestChannelClamping:
    async def _run_with_channels(self, server: Any, channels: Any) -> int:
        """Drive _message_loop with the given channels value; return the clamped result."""
        node = _Node()
        captured: dict[str, Any] = {}

        async def _spy(ws, node, audio_buffer, sample_rate, channels=1):
            captured["channels"] = channels

        server._handle_audio = _spy

        frames: list[Any] = [
            json.dumps({"type": "audio_start", "sample_rate": 16000, "channels": channels}),
            b"\x00" * 16,
            json.dumps({"type": "audio_end"}),
        ]
        ws = _make_ws(frames)
        await server._message_loop(ws, node)
        return captured.get("channels", -1)

    async def test_channels_below_minimum_clamped_to_1(self, server: Any) -> None:
        result = await self._run_with_channels(server, channels=0)
        assert result == 1

    async def test_channels_negative_clamped_to_1(self, server: Any) -> None:
        result = await self._run_with_channels(server, channels=-5)
        assert result == 1

    async def test_channels_1_accepted(self, server: Any) -> None:
        result = await self._run_with_channels(server, channels=1)
        assert result == 1

    async def test_channels_2_accepted(self, server: Any) -> None:
        result = await self._run_with_channels(server, channels=2)
        assert result == 2

    async def test_channels_above_maximum_clamped_to_2(self, server: Any) -> None:
        result = await self._run_with_channels(server, channels=8)
        assert result == 2

    async def test_non_numeric_channels_defaults_to_1(self, server: Any) -> None:
        result = await self._run_with_channels(server, channels="stereo")
        assert result == 1


# ---------------------------------------------------------------------------
# 11. Malformed JSON during message loop
# ---------------------------------------------------------------------------


class TestMalformedJsonInLoop:
    async def test_malformed_json_sends_error_frame(self, server: Any) -> None:
        """JSON parse failure in the authenticated loop must send an error frame."""
        node = _Node()
        frames: list[Any] = [
            "this is not json }{",
        ]
        ws = _make_ws(frames)

        await server._message_loop(ws, node)

        assert "error" in _sent_types(ws)

    async def test_malformed_json_does_not_close_connection(self, server: Any) -> None:
        """A JSON parse failure must not terminate the connection."""
        node = _Node()
        # Malformed frame followed by a valid heartbeat — confirms the loop
        # continued processing after the parse error.
        frames: list[Any] = [
            "{{broken",
            json.dumps({"type": "heartbeat", "occupancy": None}),
        ]
        ws = _make_ws(frames)
        server._presence_store.update = MagicMock()

        await server._message_loop(ws, node)

        ws.close.assert_not_awaited()
        # Heartbeat was still processed.
        server._presence_store.update.assert_called_once()

    async def test_multiple_malformed_frames_all_produce_errors(self, server: Any) -> None:
        """Each malformed frame in a row must each produce a separate error frame."""
        node = _Node()
        frames: list[Any] = [
            "bad1",
            "bad2",
            "bad3",
        ]
        ws = _make_ws(frames)

        await server._message_loop(ws, node)

        error_count = _sent_types(ws).count("error")
        assert error_count == 3


# ---------------------------------------------------------------------------
# 12. Spurious auth frame
# ---------------------------------------------------------------------------


class TestSpuriousAuthFrame:
    async def test_auth_after_authenticated_is_ignored(self, server: Any) -> None:
        """A second auth frame on an already-authenticated connection must be a no-op."""
        node = _Node()

        frames: list[Any] = [
            json.dumps({"type": "auth", "node_id": "proto-node", "token": "whatever"}),
        ]
        ws = _make_ws(frames)

        await server._message_loop(ws, node)

        # The registry's verify_token is not called from within the loop.
        server._registry.verify_token.assert_not_called()
        # No auth_ok or auth_fail should be sent.
        sent = _sent_types(ws)
        assert "auth_ok" not in sent
        assert "auth_fail" not in sent

    async def test_auth_frame_does_not_close_connection(self, server: Any) -> None:
        """A spurious auth frame must not cause the connection to close."""
        node = _Node()
        frames: list[Any] = [
            json.dumps({"type": "auth", "node_id": "proto-node", "token": "re-auth"}),
        ]
        ws = _make_ws(frames)

        await server._message_loop(ws, node)

        ws.close.assert_not_awaited()


# ---------------------------------------------------------------------------
# 13. Unknown message type
# ---------------------------------------------------------------------------


class TestUnknownMessageType:
    async def test_unknown_type_skipped_silently(self, server: Any) -> None:
        """An unknown message type must produce no error and no close."""
        node = _Node()
        frames: list[Any] = [
            json.dumps({"type": "totally_unknown_command", "data": 42}),
        ]
        ws = _make_ws(frames)

        await server._message_loop(ws, node)

        ws.close.assert_not_awaited()
        assert "error" not in _sent_types(ws)

    async def test_multiple_unknown_types_all_skipped(self, server: Any) -> None:
        """Multiple unknown types in sequence must all be silently skipped."""
        node = _Node()
        frames: list[Any] = [
            json.dumps({"type": "boo"}),
            json.dumps({"type": "ghost"}),
            json.dumps({"type": "phantom"}),
        ]
        ws = _make_ws(frames)

        await server._message_loop(ws, node)

        assert _sent_types(ws) == []
        ws.close.assert_not_awaited()

    async def test_unknown_type_followed_by_valid_heartbeat(self, server: Any) -> None:
        """The loop must continue to process valid frames after skipping unknown ones."""
        node = _Node()
        server._presence_store.update = MagicMock()

        frames: list[Any] = [
            json.dumps({"type": "mystery"}),
            json.dumps({"type": "heartbeat", "occupancy": True, "noise_level": 0.1}),
        ]
        ws = _make_ws(frames)

        await server._message_loop(ws, node)

        server._presence_store.update.assert_called_once()


# ---------------------------------------------------------------------------
# 14. get_status and get_connected_nodes
# ---------------------------------------------------------------------------


class TestIntrospection:
    def test_get_status_initial_state(self, server: Any) -> None:
        status = server.get_status()
        assert status["running"] is False
        assert status["connected_nodes"] == 0
        assert status["host"] == "127.0.0.1"
        assert status["port"] == 0
        assert status["stt_engine"] == "mock-stt"
        assert status["tts_engine"] == "mock-tts"

    def test_get_connected_nodes_empty_initially(self, server: Any) -> None:
        assert server.get_connected_nodes() == []

    def test_get_connected_nodes_reflects_set(self, server: Any) -> None:
        server._connected_nodes.update({"alpha", "beta", "gamma"})
        nodes = server.get_connected_nodes()
        assert set(nodes) == {"alpha", "beta", "gamma"}

    def test_get_status_connected_nodes_count(self, server: Any) -> None:
        server._connected_nodes.update({"x", "y"})
        assert server.get_status()["connected_nodes"] == 2

    def test_get_connected_nodes_returns_snapshot(self, server: Any) -> None:
        """Mutating the returned list must not affect the server's internal set."""
        server._connected_nodes.add("snap-node")
        result = server.get_connected_nodes()
        result.clear()
        assert "snap-node" in server._connected_nodes

    def test_get_status_running_true_when_running(self, server: Any) -> None:
        server._running = True
        assert server.get_status()["running"] is True


# ---------------------------------------------------------------------------
# 15. start() idempotent
# ---------------------------------------------------------------------------


class TestStartIdempotent:
    async def test_start_when_already_running_is_noop(self, server: Any) -> None:
        """Calling start() on a running server must return without re-loading engines."""
        server._running = True  # Simulate already-started state.

        await server.start()

        # STT and TTS load must NOT have been called.
        server._stt.load.assert_not_called()
        server._tts.load.assert_not_called()

    async def test_start_idempotent_does_not_overwrite_ws_server(self, server: Any) -> None:
        """The existing websocket server handle must not be replaced on a redundant start()."""
        sentinel = object()
        server._running = True
        server._ws_server = sentinel

        await server.start()

        assert server._ws_server is sentinel


# ---------------------------------------------------------------------------
# 16. stop() idempotent
# ---------------------------------------------------------------------------


class TestStopIdempotent:
    async def test_stop_when_not_running_is_noop(self, server: Any) -> None:
        """Calling stop() when not running must not raise and must not call unload."""
        assert server._running is False

        await server.stop()  # must not raise

        server._stt.unload.assert_not_called()
        server._tts.unload.assert_not_called()

    async def test_stop_twice_is_safe(self, server: Any) -> None:
        """Calling stop() twice in a row must not raise."""
        # First stop on a server that was never started.
        await server.stop()
        # Second stop — should still be a no-op.
        await server.stop()

        server._stt.unload.assert_not_called()

    async def test_stop_clears_connected_nodes(self, server: Any) -> None:
        """stop() must clear the connected-nodes set when it runs."""
        server._running = True
        server._ws_server = MagicMock()
        server._ws_server.close = MagicMock()
        server._ws_server.wait_closed = AsyncMock()
        server._connected_nodes.update({"n1", "n2"})

        await server.stop()

        assert server._connected_nodes == set()

    async def test_stop_sets_running_to_false(self, server: Any) -> None:
        server._running = True
        server._ws_server = MagicMock()
        server._ws_server.close = MagicMock()
        server._ws_server.wait_closed = AsyncMock()

        await server.stop()

        assert server._running is False
