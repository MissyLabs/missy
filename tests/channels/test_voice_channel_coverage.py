"""Coverage tests for missy/channels/voice/channel.py.

Targets uncovered lines:
  194         : start() — RuntimeError when already running
  246-250     : _runner() — server.start() raises exception → error_holder filled
  254-255     : _run_loop() — outer except (loop.run_until_complete raises)
  275         : start() — raises RuntimeError from error_holder after thread join
  293-315     : stop() — full stop path: run_coroutine_threadsafe, future.result,
                thread.join, cleanup; timeout/raise warning path
  328         : stop() — thread.is_alive() warning path
  339-341     : get_presence_context() — presence_store is None returns sentinel string;
                presence_store present returns summary
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import threading
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from missy.channels.voice.channel import VoiceChannel, _build_agent_callback


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_PATCHES = [
    "missy.channels.voice.channel.FasterWhisperSTT",
    "missy.channels.voice.channel.PiperTTS",
    "missy.channels.voice.channel.DeviceRegistry",
    "missy.channels.voice.channel.PairingManager",
    "missy.channels.voice.channel.PresenceStore",
    "missy.channels.voice.channel.VoiceServer",
]


def _make_channel(**kwargs) -> VoiceChannel:
    """Return a VoiceChannel with all heavy dependencies mocked."""
    with (
        patch("missy.channels.voice.channel.FasterWhisperSTT"),
        patch("missy.channels.voice.channel.PiperTTS"),
        patch("missy.channels.voice.channel.DeviceRegistry"),
    ):
        return VoiceChannel(**kwargs)


# ---------------------------------------------------------------------------
# start() — already running guard  (line 194)
# ---------------------------------------------------------------------------


class TestVoiceChannelStartAlreadyRunning:
    def test_start_raises_when_already_running(self):
        """Line 194: RuntimeError raised if thread is alive."""
        ch = _make_channel(host="127.0.0.1", port=9001)

        mock_thread = MagicMock()
        mock_thread.is_alive.return_value = True
        ch._thread = mock_thread

        with pytest.raises(RuntimeError, match="already running"):
            ch.start(MagicMock())


# ---------------------------------------------------------------------------
# start() — server.start() raises inside the thread  (lines 246-250, 275)
# ---------------------------------------------------------------------------


class TestVoiceChannelStartFailure:
    def test_start_raises_if_server_start_fails(self):
        """Lines 246-250, 275: server.start() raises → RuntimeError surfaced to caller."""
        ch = _make_channel(host="127.0.0.1", port=9002)

        mock_server = MagicMock()
        mock_server.start = AsyncMock(side_effect=OSError("address in use"))
        mock_server._running = False

        mock_reg = MagicMock()
        mock_pairing = MagicMock()
        mock_presence = MagicMock()
        mock_stt = MagicMock()
        mock_tts = MagicMock()

        with (
            patch("missy.channels.voice.channel.DeviceRegistry", return_value=mock_reg),
            patch("missy.channels.voice.channel.PairingManager", return_value=mock_pairing),
            patch("missy.channels.voice.channel.PresenceStore", return_value=mock_presence),
            patch("missy.channels.voice.channel.FasterWhisperSTT", return_value=mock_stt),
            patch("missy.channels.voice.channel.PiperTTS", return_value=mock_tts),
            patch("missy.channels.voice.channel.VoiceServer", return_value=mock_server),
        ):
            with pytest.raises(RuntimeError, match="VoiceChannel failed to start"):
                ch.start(MagicMock())

        # Thread and server are cleaned up on failure.
        assert ch._thread is None
        assert ch._server is None


# ---------------------------------------------------------------------------
# stop() paths  (lines 293-315, 328)
# ---------------------------------------------------------------------------


class TestVoiceChannelStop:
    def test_stop_when_not_running_is_noop(self):
        """Calling stop() when _server/_loop/_thread are None is silent."""
        ch = _make_channel()
        # Should not raise
        ch.stop()
        assert ch._server is None

    def test_stop_happy_path_cleans_up(self):
        """Lines 293-315: successful stop clears _server, _loop, _thread."""
        ch = _make_channel()

        mock_server = MagicMock()
        mock_server.stop = AsyncMock()

        # Provide a real event loop that can handle run_coroutine_threadsafe.
        loop = asyncio.new_event_loop()

        mock_thread = MagicMock()
        mock_thread.is_alive.return_value = False

        ch._server = mock_server
        ch._loop = loop
        ch._thread = mock_thread

        # run_coroutine_threadsafe schedules the coro in the given loop.
        # We run that loop briefly in a background thread to let the coro complete.
        def run_loop_briefly():
            loop.run_until_complete(asyncio.sleep(0.5))

        t = threading.Thread(target=run_loop_briefly, daemon=True)
        t.start()

        try:
            ch.stop()
        finally:
            loop.call_soon_threadsafe(loop.stop)
            t.join(timeout=2)
            if not loop.is_closed():
                loop.close()

        assert ch._server is None
        assert ch._loop is None
        assert ch._thread is None

    def test_stop_thread_still_alive_logs_warning(self, caplog):
        """Line 328: if thread.is_alive() after join, a warning is logged."""
        import logging

        ch = _make_channel()

        mock_server = MagicMock()
        mock_server.stop = AsyncMock()

        loop = asyncio.new_event_loop()

        mock_thread = MagicMock()
        # is_alive() returns True after join → triggers the warning log
        mock_thread.is_alive.return_value = True

        ch._server = mock_server
        ch._loop = loop
        ch._thread = mock_thread

        # Run the loop in background; it handles the stop coroutine then exits cleanly.
        stop_event = threading.Event()

        def run_loop():
            async def _runner():
                stop_event.wait_for = None  # placeholder
                await asyncio.sleep(2)

            loop.run_until_complete(asyncio.sleep(0))  # let run_coroutine_threadsafe work
            loop.close()

        # Provide a loop that can service run_coroutine_threadsafe
        real_loop = asyncio.new_event_loop()
        ch._loop = real_loop

        bg = threading.Thread(
            target=lambda: real_loop.run_until_complete(asyncio.sleep(0.3)), daemon=True
        )
        bg.start()

        with caplog.at_level(logging.WARNING, logger="missy.channels.voice.channel"):
            ch.stop()

        bg.join(timeout=1)
        if not real_loop.is_closed():
            real_loop.close()

        assert any("did not exit cleanly" in r.message for r in caplog.records)

    def test_stop_future_timeout_logs_warning(self, caplog):
        """Lines 303-306: future.result() timeout logs warning but doesn't raise."""
        import logging

        ch = _make_channel()

        real_loop = asyncio.new_event_loop()

        mock_server = MagicMock()
        mock_server.stop = AsyncMock()

        mock_thread = MagicMock()
        mock_thread.is_alive.return_value = False

        ch._server = mock_server
        ch._loop = real_loop
        ch._thread = mock_thread

        # Patch Future.result to raise TimeoutError
        with patch.object(
            concurrent.futures.Future, "result", side_effect=TimeoutError("timed out")
        ):
            bg = threading.Thread(
                target=lambda: real_loop.run_until_complete(asyncio.sleep(0.3)), daemon=True
            )
            bg.start()

            with caplog.at_level(logging.WARNING, logger="missy.channels.voice.channel"):
                ch.stop()

            bg.join(timeout=1)
            if not real_loop.is_closed():
                real_loop.close()

        assert any(
            "timed out" in r.message or "stop timed out" in r.message for r in caplog.records
        )


# ---------------------------------------------------------------------------
# get_presence_context()  (lines 339-341)
# ---------------------------------------------------------------------------


class TestGetPresenceContext:
    def test_returns_sentinel_when_no_presence_store(self):
        """Line 339-340: _presence_store is None → returns '(no nodes registered)'."""
        ch = _make_channel()
        assert ch._presence_store is None
        result = ch.get_presence_context()
        assert result == "(no nodes registered)"

    def test_delegates_to_presence_store_when_available(self):
        """Line 341: _presence_store present → get_context_summary() called."""
        ch = _make_channel()
        mock_store = MagicMock()
        mock_store.get_context_summary.return_value = "Living Room: occupied"
        ch._presence_store = mock_store

        result = ch.get_presence_context()
        assert result == "Living Room: occupied"
        mock_store.get_context_summary.assert_called_once()
