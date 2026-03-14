"""Gap coverage tests for missy/channels/voice/channel.py.

Targets remaining uncovered lines:
  241-245   : _run_loop() — while server._running loop executes then exits
  249-250   : _run_loop() — loop.run_until_complete raises Exception → logged
  272       : start() — logger.info after successful start
  323       : get_server() — returns _server when set
"""

from __future__ import annotations

import threading
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from missy.channels.voice.channel import VoiceChannel, _build_agent_callback

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_channel(**kwargs) -> VoiceChannel:
    """Return a VoiceChannel with all heavy dependencies mocked."""
    with (
        patch("missy.channels.voice.channel.FasterWhisperSTT"),
        patch("missy.channels.voice.channel.PiperTTS"),
        patch("missy.channels.voice.channel.DeviceRegistry"),
    ):
        return VoiceChannel(**kwargs)


# ---------------------------------------------------------------------------
# _build_agent_callback — both code paths
# ---------------------------------------------------------------------------


class TestBuildAgentCallback:
    @pytest.mark.asyncio
    async def test_async_runtime_uses_run_async(self):
        """Uses run_async when agent_runtime has it as a coroutine function."""
        runtime = MagicMock()
        runtime.run_async = AsyncMock(return_value="async response")

        cb = _build_agent_callback(runtime)
        result = await cb("hello", "session-1", {})
        assert result == "async response"
        runtime.run_async.assert_awaited_once_with("hello", "session-1", {})

    @pytest.mark.asyncio
    async def test_sync_runtime_wraps_run_in_executor(self):
        """Falls back to executor when run_async is not a coroutine function."""
        runtime = MagicMock()
        runtime.run = MagicMock(return_value="sync response")
        # Ensure run_async is not present (or is a sync mock).
        del runtime.run_async

        cb = _build_agent_callback(runtime)
        result = await cb("hi", "session-2", {})
        assert result == "sync response"


# ---------------------------------------------------------------------------
# get_server() — returns _server when set  (line 323)
# ---------------------------------------------------------------------------


class TestGetServer:
    def test_get_server_returns_none_before_start(self):
        """get_server() returns None when channel has not been started."""
        ch = _make_channel()
        assert ch.get_server() is None

    def test_get_server_returns_server_when_set(self):
        """Line 323: get_server() returns the _server attribute."""
        ch = _make_channel()
        mock_server = MagicMock()
        ch._server = mock_server
        assert ch.get_server() is mock_server


# ---------------------------------------------------------------------------
# start() — logger.info after successful start  (line 272)
# and the _run_loop while-loop path  (lines 241-245)
# ---------------------------------------------------------------------------


class TestVoiceChannelStartSuccessPath:
    def test_start_success_logs_info(self, caplog):
        """Line 272: logger.info('VoiceChannel: started on ...') is emitted."""
        import logging

        mock_server = MagicMock()
        # Make server.start() succeed and _running flip to True then False quickly.

        async def fake_start():
            mock_server._running = True

        mock_server.start = AsyncMock(side_effect=fake_start)
        # _running will be True once, then False so the while loop exits.
        mock_server._running = False

        mock_reg = MagicMock()
        mock_pairing = MagicMock()
        mock_presence = MagicMock()
        mock_stt = MagicMock()
        mock_tts = MagicMock()

        ch = _make_channel(host="127.0.0.1", port=19876)

        with (
            patch("missy.channels.voice.channel.DeviceRegistry", return_value=mock_reg),
            patch("missy.channels.voice.channel.PairingManager", return_value=mock_pairing),
            patch("missy.channels.voice.channel.PresenceStore", return_value=mock_presence),
            patch("missy.channels.voice.channel.FasterWhisperSTT", return_value=mock_stt),
            patch("missy.channels.voice.channel.PiperTTS", return_value=mock_tts),
            patch("missy.channels.voice.channel.VoiceServer", return_value=mock_server),
            caplog.at_level(logging.INFO, logger="missy.channels.voice.channel"),
        ):
            ch.start(MagicMock())

        assert any("started on" in r.message for r in caplog.records)
        ch.stop()

    def test_start_while_loop_runs_until_server_stops(self):
        """Lines 241-245: The while server._running loop keeps the event loop alive."""
        tick_count = [0]

        async def fake_start():
            # Mark running True so the while loop executes at least once.
            mock_server._running = True

        mock_server = MagicMock()
        mock_server.start = AsyncMock(side_effect=fake_start)
        mock_server.stop = AsyncMock()

        # After the server starts, set _running to False after a brief delay
        # to let the while loop body (asyncio.sleep(0.25)) execute once.

        class _RunningDescriptor:
            """Simulate _running flipping after first check in the while loop."""

            def __get__(self, obj, objtype=None):
                tick_count[0] += 1
                return tick_count[0] <= 2  # True for first 2 checks, then False

        # Patch the VoiceServer _running attribute via property-like mechanism.
        # Use a simpler approach: mock_server._running is a regular attribute that
        # we'll flip from a separate thread.
        mock_server._running = True

        stop_event = threading.Event()

        def _flip_running():
            stop_event.wait(timeout=0.4)
            mock_server._running = False

        mock_reg = MagicMock()
        mock_pairing = MagicMock()
        mock_presence = MagicMock()
        mock_stt = MagicMock()
        mock_tts = MagicMock()

        ch = _make_channel(host="127.0.0.1", port=19877)

        flipper = threading.Thread(target=_flip_running, daemon=True)
        flipper.start()
        stop_event.set()

        with (
            patch("missy.channels.voice.channel.DeviceRegistry", return_value=mock_reg),
            patch("missy.channels.voice.channel.PairingManager", return_value=mock_pairing),
            patch("missy.channels.voice.channel.PresenceStore", return_value=mock_presence),
            patch("missy.channels.voice.channel.FasterWhisperSTT", return_value=mock_stt),
            patch("missy.channels.voice.channel.PiperTTS", return_value=mock_tts),
            patch("missy.channels.voice.channel.VoiceServer", return_value=mock_server),
        ):
            ch.start(MagicMock())

        flipper.join(timeout=2)
        # If start() completed, the while loop did run.
        assert ch._server is not None
        ch.stop()


# ---------------------------------------------------------------------------
# _run_loop() — loop.run_until_complete raises Exception  (lines 249-250)
# ---------------------------------------------------------------------------


class TestRunLoopRunUntilCompleteError:
    def test_start_handles_run_until_complete_exception(self, caplog):
        """Lines 249-250: loop.run_until_complete raises → error logged, thread finishes."""
        import logging

        mock_server = MagicMock()

        async def fake_start_that_raises():
            raise RuntimeError("event loop error during start")

        mock_server.start = AsyncMock(side_effect=fake_start_that_raises)
        mock_server._running = False

        mock_reg = MagicMock()
        mock_pairing = MagicMock()
        mock_presence = MagicMock()
        mock_stt = MagicMock()
        mock_tts = MagicMock()

        ch = _make_channel(host="127.0.0.1", port=19878)

        with (
            patch("missy.channels.voice.channel.DeviceRegistry", return_value=mock_reg),
            patch("missy.channels.voice.channel.PairingManager", return_value=mock_pairing),
            patch("missy.channels.voice.channel.PresenceStore", return_value=mock_presence),
            patch("missy.channels.voice.channel.FasterWhisperSTT", return_value=mock_stt),
            patch("missy.channels.voice.channel.PiperTTS", return_value=mock_tts),
            patch("missy.channels.voice.channel.VoiceServer", return_value=mock_server),
            caplog.at_level(logging.ERROR, logger="missy.channels.voice.channel"),
            pytest.raises(RuntimeError, match="VoiceChannel failed to start"),
        ):
            # server.start() raises → error_holder is filled → RuntimeError is raised.
            ch.start(MagicMock())
