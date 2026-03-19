"""Comprehensive unit tests for missy/channels/voice/server.py.


Focus: module constants, constructor parameter storage, initial state, lifecycle
idempotency, connection flood protection, _emit helper robustness, STT/TTS
engine lifecycle ordering, status introspection, and edge-case inputs.

All external I/O (websockets, event_bus, STT, TTS) is mocked throughout.
No real network sockets are opened.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from missy.channels.voice.registry import EdgeNode
from missy.channels.voice.server import (
    _AUTH_TIMEOUT_SECONDS,
    _DEFAULT_SAMPLE_RATE,
    _MAX_AUDIO_BYTES,
    _MAX_CHANNELS,
    _MAX_CONCURRENT_CONNECTIONS,
    _MAX_SAMPLE_RATE,
    _MAX_WS_FRAME_BYTES,
    _MIN_CHANNELS,
    _MIN_SAMPLE_RATE,
    _TASK_ID,
    VoiceServer,
    _emit,
)

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _make_stt(name: str = "test-stt") -> MagicMock:
    engine = MagicMock()
    engine.name = name
    engine.load = MagicMock()
    engine.unload = MagicMock()
    engine.transcribe = AsyncMock()
    return engine


def _make_tts(name: str = "test-tts") -> MagicMock:
    engine = MagicMock()
    engine.name = name
    engine.load = MagicMock()
    engine.unload = MagicMock()
    engine.synthesize = AsyncMock()
    return engine


def _make_registry(
    *,
    verify_token: bool = True,
    node: EdgeNode | None = None,
) -> MagicMock:
    reg = MagicMock()
    reg.verify_token = MagicMock(return_value=verify_token)
    reg.get_node = MagicMock(return_value=node)
    reg.mark_online = MagicMock()
    reg.mark_offline = MagicMock()
    return reg


def _make_pairing() -> MagicMock:
    pm = MagicMock()
    pm.initiate_pairing = MagicMock(return_value="generated-node-id")
    return pm


def _make_presence() -> MagicMock:
    ps = MagicMock()
    ps.update = MagicMock()
    return ps


def _make_server(**kwargs: Any) -> VoiceServer:
    """Build a VoiceServer with fully-mocked dependencies.

    Any keyword arg overrides the corresponding constructor parameter.
    """
    defaults: dict[str, Any] = {
        "registry": _make_registry(),
        "pairing_manager": _make_pairing(),
        "presence_store": _make_presence(),
        "stt_engine": _make_stt(),
        "tts_engine": _make_tts(),
        "agent_callback": AsyncMock(return_value="response"),
    }
    defaults.update(kwargs)
    return VoiceServer(**defaults)


def _make_ws(remote_address: tuple = ("127.0.0.1", 9999)) -> AsyncMock:
    ws = AsyncMock()
    ws.remote_address = remote_address
    ws.send = AsyncMock()
    ws.close = AsyncMock()
    ws.recv = AsyncMock()
    return ws


def _make_edge_node(**kwargs: Any) -> EdgeNode:
    defaults = {
        "node_id": "node-unit-1",
        "friendly_name": "Unit Test Node",
        "room": "lab",
        "ip_address": "10.0.0.1",
        "paired": True,
        "policy_mode": "full",
        "audio_logging": False,
        "audio_log_dir": "",
    }
    defaults.update(kwargs)
    return EdgeNode(**defaults)


def _make_ws_server() -> MagicMock:
    """Minimal mock for a running websockets server object."""
    ws_srv = MagicMock()
    ws_srv.close = MagicMock()
    ws_srv.wait_closed = AsyncMock()
    return ws_srv


async def _fake_serve(*args: Any, **kwargs: Any) -> MagicMock:
    return _make_ws_server()


# ---------------------------------------------------------------------------
# 1. Module constants
# ---------------------------------------------------------------------------


class TestModuleConstants:
    def test_max_audio_bytes_is_10_mb(self) -> None:
        assert _MAX_AUDIO_BYTES == 10 * 1024 * 1024

    def test_max_ws_frame_bytes_is_1_mb(self) -> None:
        assert _MAX_WS_FRAME_BYTES == 1 * 1024 * 1024

    def test_max_concurrent_connections_is_50(self) -> None:
        assert _MAX_CONCURRENT_CONNECTIONS == 50

    def test_auth_timeout_is_10_seconds(self) -> None:
        assert _AUTH_TIMEOUT_SECONDS == 10.0

    def test_min_sample_rate_is_8000(self) -> None:
        assert _MIN_SAMPLE_RATE == 8000

    def test_max_sample_rate_is_48000(self) -> None:
        assert _MAX_SAMPLE_RATE == 48000

    def test_default_sample_rate_is_16000(self) -> None:
        assert _DEFAULT_SAMPLE_RATE == 16000

    def test_min_channels_is_1(self) -> None:
        assert _MIN_CHANNELS == 1

    def test_max_channels_is_2(self) -> None:
        assert _MAX_CHANNELS == 2

    def test_task_id_is_voice_server(self) -> None:
        assert _TASK_ID == "voice-server"

    def test_max_audio_bytes_exact_arithmetic(self) -> None:
        assert _MAX_AUDIO_BYTES == 10_485_760

    def test_max_ws_frame_bytes_exact_arithmetic(self) -> None:
        assert _MAX_WS_FRAME_BYTES == 1_048_576

    def test_sample_rate_bounds_ordering(self) -> None:
        assert _MIN_SAMPLE_RATE < _DEFAULT_SAMPLE_RATE < _MAX_SAMPLE_RATE

    def test_channel_bounds_ordering(self) -> None:
        assert _MIN_CHANNELS < _MAX_CHANNELS


# ---------------------------------------------------------------------------
# 2. Constructor parameter storage
# ---------------------------------------------------------------------------


class TestConstructorParameterStorage:
    def test_registry_is_stored(self) -> None:
        reg = _make_registry()
        server = _make_server(registry=reg)
        assert server._registry is reg

    def test_pairing_manager_is_stored(self) -> None:
        pm = _make_pairing()
        server = _make_server(pairing_manager=pm)
        assert server._pairing_manager is pm

    def test_presence_store_is_stored(self) -> None:
        ps = _make_presence()
        server = _make_server(presence_store=ps)
        assert server._presence_store is ps

    def test_stt_engine_is_stored(self) -> None:
        stt = _make_stt("whisper-large")
        server = _make_server(stt_engine=stt)
        assert server._stt is stt

    def test_tts_engine_is_stored(self) -> None:
        tts = _make_tts("piper-neural")
        server = _make_server(tts_engine=tts)
        assert server._tts is tts

    def test_agent_callback_is_stored(self) -> None:
        cb = AsyncMock(return_value="ok")
        server = _make_server(agent_callback=cb)
        assert server._agent_callback is cb

    def test_custom_host_is_stored(self) -> None:
        server = _make_server(host="192.168.1.50")
        assert server._host == "192.168.1.50"

    def test_custom_port_is_stored(self) -> None:
        server = _make_server(port=9000)
        assert server._port == 9000

    def test_custom_audio_chunk_size_is_stored(self) -> None:
        server = _make_server(audio_chunk_size=8192)
        assert server._audio_chunk_size == 8192

    def test_debug_transcripts_true_is_stored(self) -> None:
        server = _make_server(debug_transcripts=True)
        assert server._debug_transcripts is True


# ---------------------------------------------------------------------------
# 3. Constructor defaults
# ---------------------------------------------------------------------------


class TestConstructorDefaults:
    def test_default_host_is_loopback(self) -> None:
        server = _make_server()
        assert server._host == "127.0.0.1"

    def test_default_port_is_8765(self) -> None:
        server = _make_server()
        assert server._port == 8765

    def test_default_audio_chunk_size_is_4096(self) -> None:
        server = _make_server()
        assert server._audio_chunk_size == 4096

    def test_default_debug_transcripts_is_false(self) -> None:
        server = _make_server()
        assert server._debug_transcripts is False


# ---------------------------------------------------------------------------
# 4. Initial state
# ---------------------------------------------------------------------------


class TestInitialState:
    def test_running_is_false_at_init(self) -> None:
        server = _make_server()
        assert server._running is False

    def test_ws_server_is_none_at_init(self) -> None:
        server = _make_server()
        assert server._ws_server is None

    def test_connected_nodes_is_empty_set_at_init(self) -> None:
        server = _make_server()
        assert isinstance(server._connected_nodes, set)
        assert len(server._connected_nodes) == 0

    def test_active_connections_is_zero_at_init(self) -> None:
        server = _make_server()
        assert server._active_connections == 0

    def test_get_connected_nodes_returns_empty_list_at_init(self) -> None:
        server = _make_server()
        result = server.get_connected_nodes()
        assert result == []

    def test_get_connected_nodes_returns_list_not_set(self) -> None:
        server = _make_server()
        assert isinstance(server.get_connected_nodes(), list)


# ---------------------------------------------------------------------------
# 5. get_status introspection
# ---------------------------------------------------------------------------


class TestGetStatus:
    def test_status_contains_all_expected_keys(self) -> None:
        server = _make_server()
        status = server.get_status()
        for key in ("running", "connected_nodes", "host", "port", "stt_engine", "tts_engine"):
            assert key in status, f"missing key: {key}"

    def test_status_running_false_when_not_started(self) -> None:
        server = _make_server()
        assert server.get_status()["running"] is False

    def test_status_running_true_when_flag_set(self) -> None:
        server = _make_server()
        server._running = True
        assert server.get_status()["running"] is True

    def test_status_reflects_custom_host(self) -> None:
        server = _make_server(host="0.0.0.0")
        assert server.get_status()["host"] == "0.0.0.0"

    def test_status_reflects_custom_port(self) -> None:
        server = _make_server(port=9999)
        assert server.get_status()["port"] == 9999

    def test_status_reflects_stt_engine_name(self) -> None:
        server = _make_server(stt_engine=_make_stt("my-stt"))
        assert server.get_status()["stt_engine"] == "my-stt"

    def test_status_reflects_tts_engine_name(self) -> None:
        server = _make_server(tts_engine=_make_tts("my-tts"))
        assert server.get_status()["tts_engine"] == "my-tts"

    def test_status_connected_nodes_count_matches_set(self) -> None:
        server = _make_server()
        server._connected_nodes = {"alpha", "beta", "gamma"}
        assert server.get_status()["connected_nodes"] == 3

    def test_status_connected_nodes_count_zero_initially(self) -> None:
        server = _make_server()
        assert server.get_status()["connected_nodes"] == 0


# ---------------------------------------------------------------------------
# 6. get_connected_nodes
# ---------------------------------------------------------------------------


class TestGetConnectedNodes:
    def test_returns_snapshot_list(self) -> None:
        server = _make_server()
        server._connected_nodes = {"n1", "n2"}
        result = server.get_connected_nodes()
        # Mutating the returned list must not affect internal state.
        result.clear()
        assert len(server._connected_nodes) == 2

    def test_snapshot_contains_all_node_ids(self) -> None:
        server = _make_server()
        server._connected_nodes = {"x", "y", "z"}
        assert set(server.get_connected_nodes()) == {"x", "y", "z"}


# ---------------------------------------------------------------------------
# 7. stop() idempotency
# ---------------------------------------------------------------------------


class TestStopIdempotency:
    @pytest.mark.asyncio
    async def test_stop_on_non_running_server_is_noop(self) -> None:
        server = _make_server()
        assert server._running is False
        # Must not raise and must not call unload on engines.
        await server.stop()
        server._stt.unload.assert_not_called()
        server._tts.unload.assert_not_called()

    @pytest.mark.asyncio
    async def test_stop_called_twice_is_safe(self) -> None:
        server = _make_server()
        server._running = True
        server._ws_server = _make_ws_server()

        await server.stop()
        # Second call: _running is now False → should be a no-op.
        await server.stop()
        # unload should have been called exactly once each.
        server._stt.unload.assert_called_once()
        server._tts.unload.assert_called_once()

    @pytest.mark.asyncio
    async def test_stop_with_none_ws_server_does_not_raise(self) -> None:
        server = _make_server()
        server._running = True
        server._ws_server = None  # no bound server object
        await server.stop()
        server._stt.unload.assert_called_once()


# ---------------------------------------------------------------------------
# 8. start() idempotency
# ---------------------------------------------------------------------------


class TestStartIdempotency:
    @pytest.mark.asyncio
    async def test_start_on_already_running_is_noop(self) -> None:
        server = _make_server()
        server._running = True

        with patch("missy.channels.voice.server._ws_serve") as mock_serve:
            await server.start()

        mock_serve.assert_not_called()
        server._stt.load.assert_not_called()
        server._tts.load.assert_not_called()


# ---------------------------------------------------------------------------
# 9. start() STT/TTS load ordering and host warning
# ---------------------------------------------------------------------------


class TestStartLifecycle:
    @pytest.mark.asyncio
    async def test_start_calls_stt_load_then_tts_load(self) -> None:
        stt = _make_stt()
        tts = _make_tts()
        call_order: list[str] = []

        stt.load.side_effect = lambda: call_order.append("stt.load")
        tts.load.side_effect = lambda: call_order.append("tts.load")

        server = _make_server(stt_engine=stt, tts_engine=tts)

        with patch("missy.channels.voice.server._ws_serve", side_effect=_fake_serve):
            await server.start()

        assert call_order == ["stt.load", "tts.load"]

    @pytest.mark.asyncio
    async def test_start_sets_running_true(self) -> None:
        server = _make_server()
        with patch("missy.channels.voice.server._ws_serve", side_effect=_fake_serve):
            await server.start()
        assert server._running is True

    @pytest.mark.asyncio
    async def test_start_stores_ws_server_object(self) -> None:
        server = _make_server()
        with patch("missy.channels.voice.server._ws_serve", side_effect=_fake_serve):
            await server.start()
        assert server._ws_server is not None

    @pytest.mark.asyncio
    async def test_start_with_0000_host_emits_bind_warning(self) -> None:
        server = _make_server(host="0.0.0.0")

        with (
            patch("missy.channels.voice.server._ws_serve", side_effect=_fake_serve),
            patch("missy.channels.voice.server._emit") as mock_emit,
        ):
            await server.start()

        emitted_types = [c.kwargs.get("event_type") for c in mock_emit.call_args_list]
        assert "voice.bind.warning" in emitted_types

    @pytest.mark.asyncio
    async def test_start_with_0000_host_bind_warning_includes_host_port(self) -> None:
        server = _make_server(host="0.0.0.0", port=1234)

        with (
            patch("missy.channels.voice.server._ws_serve", side_effect=_fake_serve),
            patch("missy.channels.voice.server._emit") as mock_emit,
        ):
            await server.start()

        warning_calls = [
            c for c in mock_emit.call_args_list
            if c.kwargs.get("event_type") == "voice.bind.warning"
        ]
        assert len(warning_calls) == 1
        detail = warning_calls[0].kwargs.get("detail", {})
        assert detail.get("host") == "0.0.0.0"
        assert detail.get("port") == 1234

    @pytest.mark.asyncio
    async def test_start_with_loopback_host_does_not_emit_bind_warning(self) -> None:
        server = _make_server(host="127.0.0.1")

        with (
            patch("missy.channels.voice.server._ws_serve", side_effect=_fake_serve),
            patch("missy.channels.voice.server._emit") as mock_emit,
        ):
            await server.start()

        emitted_types = [c.kwargs.get("event_type") for c in mock_emit.call_args_list]
        assert "voice.bind.warning" not in emitted_types

    @pytest.mark.asyncio
    async def test_start_passes_max_size_to_ws_serve(self) -> None:
        server = _make_server()
        captured_kwargs: dict[str, Any] = {}

        async def capturing_serve(*args: Any, **kwargs: Any) -> MagicMock:
            captured_kwargs.update(kwargs)
            return _make_ws_server()

        with patch("missy.channels.voice.server._ws_serve", side_effect=capturing_serve):
            await server.start()

        assert captured_kwargs.get("max_size") == _MAX_WS_FRAME_BYTES


# ---------------------------------------------------------------------------
# 10. stop() engine unload ordering and cleanup
# ---------------------------------------------------------------------------


class TestStopLifecycle:
    @pytest.mark.asyncio
    async def test_stop_calls_stt_unload_then_tts_unload(self) -> None:
        stt = _make_stt()
        tts = _make_tts()
        call_order: list[str] = []

        stt.unload.side_effect = lambda: call_order.append("stt.unload")
        tts.unload.side_effect = lambda: call_order.append("tts.unload")

        server = _make_server(stt_engine=stt, tts_engine=tts)
        server._running = True
        server._ws_server = _make_ws_server()

        await server.stop()

        assert call_order == ["stt.unload", "tts.unload"]

    @pytest.mark.asyncio
    async def test_stop_clears_connected_nodes(self) -> None:
        server = _make_server()
        server._running = True
        server._ws_server = _make_ws_server()
        server._connected_nodes = {"alpha", "beta"}

        await server.stop()

        assert len(server._connected_nodes) == 0

    @pytest.mark.asyncio
    async def test_stop_sets_running_false(self) -> None:
        server = _make_server()
        server._running = True
        server._ws_server = _make_ws_server()

        await server.stop()

        assert server._running is False

    @pytest.mark.asyncio
    async def test_stop_sets_ws_server_to_none(self) -> None:
        server = _make_server()
        server._running = True
        server._ws_server = _make_ws_server()

        await server.stop()

        assert server._ws_server is None

    @pytest.mark.asyncio
    async def test_stop_calls_ws_server_close_and_wait_closed(self) -> None:
        server = _make_server()
        ws_srv = _make_ws_server()
        server._running = True
        server._ws_server = ws_srv

        await server.stop()

        ws_srv.close.assert_called_once()
        ws_srv.wait_closed.assert_awaited_once()


# ---------------------------------------------------------------------------
# 11. Connection flood protection
# ---------------------------------------------------------------------------


class TestConnectionFloodProtection:
    @pytest.mark.asyncio
    async def test_rejects_connection_at_limit(self) -> None:
        server = _make_server()
        server._active_connections = _MAX_CONCURRENT_CONNECTIONS
        ws = _make_ws()

        await server._handle_connection(ws)

        ws.close.assert_awaited_once_with(1013, "Server at capacity")

    @pytest.mark.asyncio
    async def test_rejects_connection_above_limit(self) -> None:
        server = _make_server()
        server._active_connections = _MAX_CONCURRENT_CONNECTIONS + 10
        ws = _make_ws()

        await server._handle_connection(ws)

        ws.close.assert_awaited_once_with(1013, "Server at capacity")

    @pytest.mark.asyncio
    async def test_accepts_connection_one_below_limit(self) -> None:
        """One connection below the limit must be processed (not immediately closed with 1013)."""
        server = _make_server()
        server._active_connections = _MAX_CONCURRENT_CONNECTIONS - 1
        ws = _make_ws()

        import websockets.exceptions as wsexc

        ws.recv = AsyncMock(side_effect=wsexc.ConnectionClosed(None, None))

        # Should not see the 1013 close — it may close for other reasons.
        await server._handle_connection(ws)

        # Verify the close was NOT the capacity-reject close.
        for call_item in ws.close.call_args_list:
            args = call_item[0]
            if args:
                assert args[0] != 1013, "Should not reject with 1013 below the limit"

    @pytest.mark.asyncio
    async def test_active_connections_decremented_after_reject_at_limit(self) -> None:
        server = _make_server()
        server._active_connections = _MAX_CONCURRENT_CONNECTIONS
        ws = _make_ws()

        before = server._active_connections
        await server._handle_connection(ws)
        # Counter must not increase when rejected before incrementing.
        assert server._active_connections == before

    @pytest.mark.asyncio
    async def test_active_connections_incremented_then_decremented_normally(self) -> None:
        """Counter goes up by 1 on entry and back down by 1 on exit."""
        server = _make_server()
        assert server._active_connections == 0

        import websockets.exceptions as wsexc

        ws = _make_ws()
        ws.recv = AsyncMock(side_effect=wsexc.ConnectionClosed(None, None))

        await server._handle_connection(ws)

        # Net effect: back to 0.
        assert server._active_connections == 0

    @pytest.mark.asyncio
    async def test_active_connections_never_goes_negative(self) -> None:
        """The max(0, ...) guard ensures _active_connections never dips below zero."""
        server = _make_server()
        server._active_connections = 0

        import websockets.exceptions as wsexc

        ws = _make_ws()
        ws.recv = AsyncMock(side_effect=wsexc.ConnectionClosed(None, None))

        await server._handle_connection(ws)

        assert server._active_connections >= 0


# ---------------------------------------------------------------------------
# 12. _emit helper
# ---------------------------------------------------------------------------


class TestEmitHelper:
    def test_emit_calls_event_bus_publish(self) -> None:
        with patch("missy.channels.voice.server.event_bus") as mock_bus:
            _emit("sid", "voice.test", "allow", {"k": "v"})
        mock_bus.publish.assert_called_once()

    def test_emit_published_event_has_correct_session_id(self) -> None:
        with patch("missy.channels.voice.server.event_bus") as mock_bus:
            _emit("my-session", "voice.test", "allow")
        event = mock_bus.publish.call_args[0][0]
        assert event.session_id == "my-session"

    def test_emit_published_event_has_correct_event_type(self) -> None:
        with patch("missy.channels.voice.server.event_bus") as mock_bus:
            _emit("s", "voice.custom.event", "allow")
        event = mock_bus.publish.call_args[0][0]
        assert event.event_type == "voice.custom.event"

    def test_emit_published_event_has_correct_result(self) -> None:
        with patch("missy.channels.voice.server.event_bus") as mock_bus:
            _emit("s", "e", "deny")
        event = mock_bus.publish.call_args[0][0]
        assert event.result == "deny"

    def test_emit_published_event_has_correct_detail(self) -> None:
        with patch("missy.channels.voice.server.event_bus") as mock_bus:
            _emit("s", "e", "allow", {"node": "x"})
        event = mock_bus.publish.call_args[0][0]
        assert event.detail == {"node": "x"}

    def test_emit_uses_empty_dict_when_detail_is_none(self) -> None:
        with patch("missy.channels.voice.server.event_bus") as mock_bus:
            _emit("s", "e", "allow", None)
        event = mock_bus.publish.call_args[0][0]
        assert event.detail == {}

    def test_emit_uses_task_id_voice_server(self) -> None:
        with patch("missy.channels.voice.server.event_bus") as mock_bus:
            _emit("s", "e", "allow")
        event = mock_bus.publish.call_args[0][0]
        assert event.task_id == "voice-server"

    def test_emit_does_not_raise_when_publish_raises_runtime_error(self) -> None:
        with patch("missy.channels.voice.server.event_bus") as mock_bus:
            mock_bus.publish.side_effect = RuntimeError("bus failure")
            # Must not propagate.
            _emit("s", "e", "error")

    def test_emit_does_not_raise_when_publish_raises_attribute_error(self) -> None:
        with patch("missy.channels.voice.server.event_bus") as mock_bus:
            mock_bus.publish.side_effect = AttributeError("bus gone")
            _emit("s", "e", "allow")

    def test_emit_does_not_raise_when_publish_raises_os_error(self) -> None:
        with patch("missy.channels.voice.server.event_bus") as mock_bus:
            mock_bus.publish.side_effect = OSError("disk full")
            _emit("s", "e", "allow")

    def test_emit_with_error_result_publishes(self) -> None:
        with patch("missy.channels.voice.server.event_bus") as mock_bus:
            _emit("s", "e", "error", {"code": 500})
        mock_bus.publish.assert_called_once()

    def test_emit_with_deny_result_publishes(self) -> None:
        with patch("missy.channels.voice.server.event_bus") as mock_bus:
            _emit("s", "e", "deny", {"reason": "blocked"})
        mock_bus.publish.assert_called_once()


# ---------------------------------------------------------------------------
# 13. Sample rate clamping in _message_loop audio_start
# ---------------------------------------------------------------------------


def _make_full_server_for_clamping() -> VoiceServer:
    """Build a server with proper STT/TTS mocks for clamping tests.

    The stt.transcribe returns an empty transcript so the agent path is
    short-circuited, avoiding the need to mock TTS synthesize output.
    """
    from missy.channels.voice.stt.base import TranscriptionResult

    stt = _make_stt()
    # Empty transcript prevents agent callback → TTS path entirely.
    stt.transcribe = AsyncMock(return_value=TranscriptionResult(
        text="   ", confidence=0.0, processing_ms=5
    ))
    return _make_server(stt_engine=stt)


class TestSampleRateClamping:
    @pytest.mark.asyncio
    async def test_sample_rate_below_min_is_clamped_to_min(self) -> None:
        """audio_start with sample_rate=100 should clamp to _MIN_SAMPLE_RATE."""
        import json

        from missy.channels.voice.stt.base import TranscriptionResult

        node = _make_edge_node()
        server = _make_full_server_for_clamping()
        ws = _make_ws()

        captured_rate: list[int] = []

        async def capture_transcribe(buf: bytes, *, sample_rate: int, channels: int):
            captured_rate.append(sample_rate)
            return TranscriptionResult(text="   ", confidence=0.0, processing_ms=5)

        server._stt.transcribe = capture_transcribe

        async def fake_aiter():
            yield json.dumps({"type": "audio_start", "sample_rate": 100, "channels": 1})
            yield b"\x00" * 64
            yield json.dumps({"type": "audio_end"})

        ws.__aiter__ = lambda _self: fake_aiter()

        with patch("missy.channels.voice.server._emit"):
            await server._message_loop(ws, node)

        assert captured_rate == [_MIN_SAMPLE_RATE]

    @pytest.mark.asyncio
    async def test_sample_rate_above_max_is_clamped_to_max(self) -> None:
        import json

        from missy.channels.voice.stt.base import TranscriptionResult

        node = _make_edge_node()
        server = _make_full_server_for_clamping()
        ws = _make_ws()

        captured_rate: list[int] = []

        async def capture_transcribe(buf: bytes, *, sample_rate: int, channels: int):
            captured_rate.append(sample_rate)
            return TranscriptionResult(text="   ", confidence=0.0, processing_ms=5)

        server._stt.transcribe = capture_transcribe

        async def fake_aiter():
            yield json.dumps({"type": "audio_start", "sample_rate": 999_999, "channels": 1})
            yield b"\x00" * 64
            yield json.dumps({"type": "audio_end"})

        ws.__aiter__ = lambda _self: fake_aiter()

        with patch("missy.channels.voice.server._emit"):
            await server._message_loop(ws, node)

        assert captured_rate == [_MAX_SAMPLE_RATE]

    @pytest.mark.asyncio
    async def test_invalid_sample_rate_falls_back_to_default(self) -> None:
        import json

        from missy.channels.voice.stt.base import TranscriptionResult

        node = _make_edge_node()
        server = _make_full_server_for_clamping()
        ws = _make_ws()

        captured_rate: list[int] = []

        async def capture_transcribe(buf: bytes, *, sample_rate: int, channels: int):
            captured_rate.append(sample_rate)
            return TranscriptionResult(text="   ", confidence=0.0, processing_ms=5)

        server._stt.transcribe = capture_transcribe

        async def fake_aiter():
            # non-numeric sample_rate
            yield json.dumps({"type": "audio_start", "sample_rate": "not-a-number", "channels": 1})
            yield b"\x00" * 64
            yield json.dumps({"type": "audio_end"})

        ws.__aiter__ = lambda _self: fake_aiter()

        with patch("missy.channels.voice.server._emit"):
            await server._message_loop(ws, node)

        assert captured_rate == [_DEFAULT_SAMPLE_RATE]


# ---------------------------------------------------------------------------
# 14. Channel count clamping in _message_loop audio_start
# ---------------------------------------------------------------------------


class TestChannelClamping:
    @pytest.mark.asyncio
    async def test_channels_below_min_is_clamped_to_min(self) -> None:
        import json

        from missy.channels.voice.stt.base import TranscriptionResult

        node = _make_edge_node()
        server = _make_full_server_for_clamping()
        ws = _make_ws()

        captured_channels: list[int] = []

        async def capture_transcribe(buf: bytes, *, sample_rate: int, channels: int):
            captured_channels.append(channels)
            return TranscriptionResult(text="   ", confidence=0.0, processing_ms=5)

        server._stt.transcribe = capture_transcribe

        async def fake_aiter():
            yield json.dumps({"type": "audio_start", "sample_rate": 16000, "channels": 0})
            yield b"\x00" * 64
            yield json.dumps({"type": "audio_end"})

        ws.__aiter__ = lambda _self: fake_aiter()

        with patch("missy.channels.voice.server._emit"):
            await server._message_loop(ws, node)

        assert captured_channels == [_MIN_CHANNELS]

    @pytest.mark.asyncio
    async def test_channels_above_max_is_clamped_to_max(self) -> None:
        import json

        from missy.channels.voice.stt.base import TranscriptionResult

        node = _make_edge_node()
        server = _make_full_server_for_clamping()
        ws = _make_ws()

        captured_channels: list[int] = []

        async def capture_transcribe(buf: bytes, *, sample_rate: int, channels: int):
            captured_channels.append(channels)
            return TranscriptionResult(text="   ", confidence=0.0, processing_ms=5)

        server._stt.transcribe = capture_transcribe

        async def fake_aiter():
            yield json.dumps({"type": "audio_start", "sample_rate": 16000, "channels": 100})
            yield b"\x00" * 64
            yield json.dumps({"type": "audio_end"})

        ws.__aiter__ = lambda _self: fake_aiter()

        with patch("missy.channels.voice.server._emit"):
            await server._message_loop(ws, node)

        assert captured_channels == [_MAX_CHANNELS]

    @pytest.mark.asyncio
    async def test_invalid_channels_type_falls_back_to_min(self) -> None:
        import json

        from missy.channels.voice.stt.base import TranscriptionResult

        node = _make_edge_node()
        server = _make_full_server_for_clamping()
        ws = _make_ws()

        captured_channels: list[int] = []

        async def capture_transcribe(buf: bytes, *, sample_rate: int, channels: int):
            captured_channels.append(channels)
            return TranscriptionResult(text="   ", confidence=0.0, processing_ms=5)

        server._stt.transcribe = capture_transcribe

        async def fake_aiter():
            yield json.dumps({"type": "audio_start", "sample_rate": 16000, "channels": "stereo"})
            yield b"\x00" * 64
            yield json.dumps({"type": "audio_end"})

        ws.__aiter__ = lambda _self: fake_aiter()

        with patch("missy.channels.voice.server._emit"):
            await server._message_loop(ws, node)

        assert captured_channels == [_MIN_CHANNELS]


# ---------------------------------------------------------------------------
# 15. Auth timeout constant drives wait_for timeout
# ---------------------------------------------------------------------------


class TestAuthTimeoutConstant:
    @pytest.mark.asyncio
    async def test_auth_timeout_closes_connection_with_1008(self) -> None:
        server = _make_server()
        ws = _make_ws()

        async def slow_recv() -> str:
            raise TimeoutError()

        ws.recv = AsyncMock(side_effect=TimeoutError)

        # Patch asyncio.wait_for so that TimeoutError propagates as the server expects.

        async def instant_timeout(coro, timeout):
            # Discard the coro to avoid ResourceWarning.
            coro.close()
            raise TimeoutError()

        with patch("asyncio.wait_for", side_effect=instant_timeout):
            await server._handle_connection(ws)

        ws.close.assert_awaited_with(1008, "Authentication timeout")


# ---------------------------------------------------------------------------
# 16. Port=0 edge case (OS assigns port)
# ---------------------------------------------------------------------------


class TestPortZeroEdgeCase:
    def test_port_zero_is_stored_as_zero(self) -> None:
        server = _make_server(port=0)
        assert server._port == 0

    def test_port_zero_appears_in_status(self) -> None:
        server = _make_server(port=0)
        assert server.get_status()["port"] == 0

    @pytest.mark.asyncio
    async def test_start_with_port_zero_passes_zero_to_serve(self) -> None:
        captured: list[Any] = []

        async def capturing_serve(*args: Any, **kwargs: Any) -> MagicMock:
            captured.extend(args)
            return _make_ws_server()

        server = _make_server(port=0)

        with patch("missy.channels.voice.server._ws_serve", side_effect=capturing_serve):
            await server.start()

        # positional args to _ws_serve: handler, host, port
        assert captured[2] == 0


# ---------------------------------------------------------------------------
# 17. Empty registry (no nodes registered)
# ---------------------------------------------------------------------------


class TestEmptyRegistry:
    @pytest.mark.asyncio
    async def test_auth_with_empty_registry_returns_none(self) -> None:

        reg = _make_registry(verify_token=False)
        server = _make_server(registry=reg)
        ws = _make_ws()

        with patch("missy.channels.voice.server._emit"):
            result = await server._handle_auth(ws, node_id="ghost", token="bad")

        assert result is None

    def test_get_connected_nodes_with_empty_registry(self) -> None:
        reg = _make_registry()
        server = _make_server(registry=reg)
        assert server.get_connected_nodes() == []


# ---------------------------------------------------------------------------
# 18. Audio chunk size parameter storage and streaming
# ---------------------------------------------------------------------------


class TestAudioChunkSize:
    def test_chunk_size_1024_stored(self) -> None:
        server = _make_server(audio_chunk_size=1024)
        assert server._audio_chunk_size == 1024

    def test_chunk_size_16384_stored(self) -> None:
        server = _make_server(audio_chunk_size=16384)
        assert server._audio_chunk_size == 16384

    @pytest.mark.asyncio
    async def test_custom_chunk_size_used_when_streaming_audio(self) -> None:
        from missy.channels.voice.stt.base import TranscriptionResult
        from missy.channels.voice.tts.base import AudioBuffer

        chunk_size = 1024
        audio_size = 4096  # exactly 4 chunks

        node = _make_edge_node()
        tts = _make_tts()
        tts.synthesize = AsyncMock(return_value=AudioBuffer(
            data=b"\xab" * audio_size, sample_rate=22050, channels=1, format="wav"
        ))
        stt = _make_stt()
        stt.transcribe = AsyncMock(return_value=TranscriptionResult(
            text="test", confidence=0.9, processing_ms=10
        ))

        server = _make_server(stt_engine=stt, tts_engine=tts, audio_chunk_size=chunk_size)
        ws = _make_ws()

        with patch("missy.channels.voice.server._emit"):
            await server._handle_audio(
                ws, node=node, audio_buffer=b"\x00" * 100, sample_rate=16000
            )

        binary_sends = [c for c in ws.send.call_args_list if isinstance(c[0][0], bytes)]
        # 4096 bytes / 1024 chunk = 4 binary sends
        assert len(binary_sends) == 4
        for call_item in binary_sends:
            assert len(call_item[0][0]) == chunk_size


# ---------------------------------------------------------------------------
# 19. Multiple start/stop cycles
# ---------------------------------------------------------------------------


class TestMultipleStartStopCycles:
    @pytest.mark.asyncio
    async def test_start_stop_start_stop_cycle(self) -> None:
        stt = _make_stt()
        tts = _make_tts()
        server = _make_server(stt_engine=stt, tts_engine=tts)

        with patch("missy.channels.voice.server._ws_serve", side_effect=_fake_serve):
            await server.start()

        assert server._running is True
        await server.stop()
        assert server._running is False

        # Reset mocks to count a fresh start.
        stt.load.reset_mock()
        tts.load.reset_mock()

        with patch("missy.channels.voice.server._ws_serve", side_effect=_fake_serve):
            await server.start()

        assert server._running is True
        stt.load.assert_called_once()
        tts.load.assert_called_once()

        await server.stop()
        assert server._running is False


# ---------------------------------------------------------------------------
# 20. debug_transcripts=True sends transcript frame
# ---------------------------------------------------------------------------


class TestDebugTranscripts:
    @pytest.mark.asyncio
    async def test_debug_transcripts_true_sends_transcript_json(self) -> None:
        import json

        from missy.channels.voice.stt.base import TranscriptionResult
        from missy.channels.voice.tts.base import AudioBuffer

        node = _make_edge_node()
        stt = _make_stt()
        stt.transcribe = AsyncMock(return_value=TranscriptionResult(
            text="hello there", confidence=0.87, processing_ms=50
        ))
        tts = _make_tts()
        tts.synthesize = AsyncMock(return_value=AudioBuffer(
            data=b"\x00" * 256, sample_rate=22050, channels=1, format="wav"
        ))

        server = _make_server(stt_engine=stt, tts_engine=tts, debug_transcripts=True)
        ws = _make_ws()

        with patch("missy.channels.voice.server._emit"):
            await server._handle_audio(ws, node=node, audio_buffer=b"\x00" * 64, sample_rate=16000)

        sent_json = [
            json.loads(c[0][0]) for c in ws.send.call_args_list if isinstance(c[0][0], str)
        ]
        transcript_frames = [f for f in sent_json if f.get("type") == "transcript"]
        assert len(transcript_frames) == 1
        assert transcript_frames[0]["text"] == "hello there"
        assert transcript_frames[0]["confidence"] == pytest.approx(0.87)

    @pytest.mark.asyncio
    async def test_debug_transcripts_false_does_not_send_transcript_json(self) -> None:
        import json

        from missy.channels.voice.stt.base import TranscriptionResult
        from missy.channels.voice.tts.base import AudioBuffer

        node = _make_edge_node()
        stt = _make_stt()
        stt.transcribe = AsyncMock(return_value=TranscriptionResult(
            text="hello", confidence=0.9, processing_ms=30
        ))
        tts = _make_tts()
        tts.synthesize = AsyncMock(return_value=AudioBuffer(
            data=b"\x00" * 256, sample_rate=22050, channels=1, format="wav"
        ))

        server = _make_server(stt_engine=stt, tts_engine=tts, debug_transcripts=False)
        ws = _make_ws()

        with patch("missy.channels.voice.server._emit"):
            await server._handle_audio(ws, node=node, audio_buffer=b"\x00" * 64, sample_rate=16000)

        sent_json = [
            json.loads(c[0][0]) for c in ws.send.call_args_list if isinstance(c[0][0], str)
        ]
        transcript_frames = [f for f in sent_json if f.get("type") == "transcript"]
        assert len(transcript_frames) == 0
