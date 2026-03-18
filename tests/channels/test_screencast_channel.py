"""Tests for missy/channels/screencast/channel.py."""

from __future__ import annotations

import asyncio
import concurrent.futures
import threading
import time
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from missy.channels.screencast.channel import ScreencastChannel, _get_lan_ip
from missy.channels.screencast.session_manager import AnalysisResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_channel(**kwargs: Any) -> ScreencastChannel:
    """Return a ScreencastChannel that has NOT been started."""
    defaults = {
        "host": "127.0.0.1",
        "port": 8780,
        "max_sessions": 5,
    }
    defaults.update(kwargs)
    return ScreencastChannel(**defaults)


# ---------------------------------------------------------------------------
# Basic attribute / interface tests
# ---------------------------------------------------------------------------

class TestChannelName:
    def test_channel_name(self) -> None:
        ch = _make_channel()
        assert ch.name == "screencast"


class TestNotImplementedMethods:
    def test_receive_raises(self) -> None:
        ch = _make_channel()
        with pytest.raises(NotImplementedError):
            ch.receive()

    def test_send_raises(self) -> None:
        ch = _make_channel()
        with pytest.raises(NotImplementedError):
            ch.send("hello")


# ---------------------------------------------------------------------------
# "Not running" fallback paths (no server started)
# ---------------------------------------------------------------------------

class TestNotRunningFallbacks:
    def test_create_session_when_not_running(self) -> None:
        ch = _make_channel()
        with pytest.raises(RuntimeError, match="not running"):
            ch.create_session(created_by="u1", discord_channel_id="c1", label="test")

    def test_revoke_session_when_not_running(self) -> None:
        ch = _make_channel()
        result = ch.revoke_session("fake-session-id")
        assert result is False

    def test_get_active_sessions_when_not_running(self) -> None:
        ch = _make_channel()
        result = ch.get_active_sessions()
        assert result == []

    def test_get_latest_analysis_when_not_running(self) -> None:
        ch = _make_channel()
        result = ch.get_latest_analysis("fake-session-id")
        assert result is None

    def test_get_results_when_not_running(self) -> None:
        ch = _make_channel()
        result = ch.get_results("fake-session-id", limit=5)
        assert result == []

    def test_get_status_when_not_running(self) -> None:
        ch = _make_channel()
        status = ch.get_status()
        assert status == {"running": False}

    def test_stop_when_not_running(self) -> None:
        ch = _make_channel()
        # Must be a no-op and must not raise.
        ch.stop()


# ---------------------------------------------------------------------------
# Discord REST attachment
# ---------------------------------------------------------------------------

class TestSetDiscordRest:
    def test_set_discord_rest(self) -> None:
        ch = _make_channel()
        mock_rest = MagicMock()
        ch.set_discord_rest(mock_rest)
        assert ch._discord_rest is mock_rest

    def test_set_discord_rest_none(self) -> None:
        ch = _make_channel()
        ch.set_discord_rest(None)
        assert ch._discord_rest is None


# ---------------------------------------------------------------------------
# _get_lan_ip
# ---------------------------------------------------------------------------

class TestGetLanIp:
    def test_get_lan_ip_returns_string(self) -> None:
        ip = _get_lan_ip()
        assert isinstance(ip, str)
        assert len(ip) > 0

    def test_get_lan_ip_fallback_on_error(self) -> None:
        """When the socket call fails, the function must return 127.0.0.1."""
        import socket
        with patch("missy.channels.screencast.channel.socket.socket") as mock_sock_cls:
            mock_sock = MagicMock()
            mock_sock.__enter__ = MagicMock(return_value=mock_sock)
            mock_sock.__exit__ = MagicMock(return_value=False)
            mock_sock.connect.side_effect = OSError("network unreachable")
            mock_sock_cls.return_value = mock_sock
            ip = _get_lan_ip()
        assert ip == "127.0.0.1"


# ---------------------------------------------------------------------------
# create_session with mocked token registry
# ---------------------------------------------------------------------------

class TestCreateSessionMocked:
    def test_create_session_with_mocked_registry(self) -> None:
        ch = _make_channel(host="127.0.0.1", port=8780)

        mock_registry = MagicMock()
        mock_registry.create_session.return_value = ("sess-abc", "tok-xyz")
        ch._token_registry = mock_registry

        session_id, token, share_url = ch.create_session(
            created_by="user1",
            discord_channel_id="chan1",
            label="my label",
        )

        assert session_id == "sess-abc"
        assert token == "tok-xyz"
        assert "sess-abc" in share_url
        assert "tok-xyz" in share_url

    def test_create_session_url_contains_session_and_token(self) -> None:
        ch = _make_channel(host="127.0.0.1", port=9000)

        mock_registry = MagicMock()
        mock_registry.create_session.return_value = ("sid-001", "tok-001")
        ch._token_registry = mock_registry

        _, _, share_url = ch.create_session()

        assert "session_id=sid-001" in share_url
        assert "token=tok-001" in share_url

    def test_create_session_with_capture_url_base(self) -> None:
        """A custom capture_url_base must be used verbatim in the share URL."""
        custom_base = "https://my.proxy.example.com"
        ch = _make_channel(capture_url_base=custom_base)

        mock_registry = MagicMock()
        mock_registry.create_session.return_value = ("sid-002", "tok-002")
        ch._token_registry = mock_registry

        _, _, share_url = ch.create_session(label="proxied")

        assert share_url.startswith(custom_base)
        assert "sid-002" in share_url
        assert "tok-002" in share_url

    def test_create_session_0000_host_uses_lan_ip(self) -> None:
        """When host is 0.0.0.0 the share URL should use the LAN IP, not 0.0.0.0."""
        ch = _make_channel(host="0.0.0.0", port=8780)

        mock_registry = MagicMock()
        mock_registry.create_session.return_value = ("sid-003", "tok-003")
        ch._token_registry = mock_registry

        with patch(
            "missy.channels.screencast.channel._get_lan_ip",
            return_value="192.168.1.50",
        ):
            _, _, share_url = ch.create_session()

        assert "0.0.0.0" not in share_url
        assert "192.168.1.50" in share_url


# ---------------------------------------------------------------------------
# get_active_sessions with mocked registry
# ---------------------------------------------------------------------------

class TestGetActiveSessionsMocked:
    def test_get_active_sessions_with_mocked_registry(self) -> None:
        ch = _make_channel()

        fake_session = MagicMock()
        fake_session.session_id = "s-001"
        fake_session.label = "work"
        fake_session.created_by = "user99"
        fake_session.created_at = time.time()
        fake_session.frame_count = 10
        fake_session.analysis_count = 3
        fake_session.last_frame_at = time.time()

        mock_registry = MagicMock()
        mock_registry.list_active.return_value = [fake_session]
        ch._token_registry = mock_registry

        sessions = ch.get_active_sessions()

        assert len(sessions) == 1
        assert sessions[0]["session_id"] == "s-001"
        assert sessions[0]["label"] == "work"
        assert sessions[0]["frame_count"] == 10

    def test_get_active_sessions_empty_registry(self) -> None:
        ch = _make_channel()

        mock_registry = MagicMock()
        mock_registry.list_active.return_value = []
        ch._token_registry = mock_registry

        sessions = ch.get_active_sessions()
        assert sessions == []


# ---------------------------------------------------------------------------
# Already-running guard on start()
# ---------------------------------------------------------------------------

class TestAlreadyRunning:
    def test_already_running_raises(self) -> None:
        """If _thread is alive, start() must raise RuntimeError."""
        ch = _make_channel()

        mock_thread = MagicMock(spec=threading.Thread)
        mock_thread.is_alive.return_value = True
        ch._thread = mock_thread

        with pytest.raises(RuntimeError, match="already running"):
            ch.start()


# ---------------------------------------------------------------------------
# start() lifecycle tests
# ---------------------------------------------------------------------------

def _make_mock_server(running_flag_holder: list[bool] | None = None) -> MagicMock:
    """Return a MagicMock for ScreencastServer whose start() sets _running=True."""
    mock_server = MagicMock()

    async def _fake_start() -> None:
        mock_server._running = True

    async def _fake_stop() -> None:
        mock_server._running = False

    mock_server.start = AsyncMock(side_effect=_fake_start)
    mock_server.stop = AsyncMock(side_effect=_fake_stop)
    mock_server._running = False
    mock_server.get_status = MagicMock(return_value={"running": True, "sessions": 0})
    return mock_server


def _make_mock_analyzer() -> MagicMock:
    """Return a MagicMock for FrameAnalyzer."""
    mock_analyzer = MagicMock()
    mock_analyzer.start = AsyncMock()
    mock_analyzer.stop = AsyncMock()
    return mock_analyzer


class TestStartLifecycle:
    """Tests for the start() full-lifecycle path."""

    def _patch_constructors(
        self,
        mock_server: MagicMock,
        mock_analyzer: MagicMock,
    ):
        """Return a context-manager stack that patches ScreencastServer and FrameAnalyzer."""
        import contextlib

        @contextlib.contextmanager
        def _combined():
            with patch(
                "missy.channels.screencast.channel.ScreencastServer",
                return_value=mock_server,
            ) as ps, patch(
                "missy.channels.screencast.channel.FrameAnalyzer",
                return_value=mock_analyzer,
            ) as pa:
                yield ps, pa

        return _combined()

    def test_start_sets_running_state(self) -> None:
        """After start() the channel must have a live thread and loop."""
        mock_server = _make_mock_server()
        mock_analyzer = _make_mock_analyzer()

        ch = _make_channel()
        with self._patch_constructors(mock_server, mock_analyzer):
            ch.start()

        try:
            assert ch._thread is not None
            assert ch._thread.is_alive()
        finally:
            # Drive the server into stopped state so the daemon loop exits.
            mock_server._running = False
            ch._thread.join(timeout=3)

    def test_start_calls_server_and_analyzer_start(self) -> None:
        """start() must await server.start() and analyzer.start()."""
        mock_server = _make_mock_server()
        mock_analyzer = _make_mock_analyzer()

        ch = _make_channel()
        with self._patch_constructors(mock_server, mock_analyzer):
            ch.start()

        # Give the daemon thread a moment to run the coroutines.
        deadline = time.monotonic() + 3.0
        while time.monotonic() < deadline:
            if mock_server.start.called and mock_analyzer.start.called:
                break
            time.sleep(0.05)

        assert mock_server.start.called
        assert mock_analyzer.start.called

        mock_server._running = False

    def test_start_populates_subsystem_references(self) -> None:
        """start() must set _token_registry, _session_manager, _analyzer, _server."""
        mock_server = _make_mock_server()
        mock_analyzer = _make_mock_analyzer()

        ch = _make_channel()
        with self._patch_constructors(mock_server, mock_analyzer):
            ch.start()

        try:
            assert ch._token_registry is not None
            assert ch._session_manager is not None
            assert ch._analyzer is mock_analyzer
            assert ch._server is mock_server
        finally:
            mock_server._running = False

    def test_start_raises_when_server_start_fails(self) -> None:
        """If server.start() raises, start() must propagate a RuntimeError."""
        mock_server = _make_mock_server()
        mock_server.start = AsyncMock(side_effect=OSError("address already in use"))
        mock_analyzer = _make_mock_analyzer()

        ch = _make_channel()
        with self._patch_constructors(mock_server, mock_analyzer):
            with pytest.raises(RuntimeError, match="failed to start"):
                ch.start()

        # After failure, _thread and _server must be cleared.
        assert ch._thread is None
        assert ch._server is None

    def test_start_raises_when_analyzer_start_fails(self) -> None:
        """If analyzer.start() raises, start() must propagate a RuntimeError."""
        mock_server = _make_mock_server()
        mock_analyzer = _make_mock_analyzer()
        mock_analyzer.start = AsyncMock(side_effect=RuntimeError("analyzer init failed"))

        ch = _make_channel()
        with self._patch_constructors(mock_server, mock_analyzer):
            with pytest.raises(RuntimeError, match="failed to start"):
                ch.start()

    def test_start_thread_is_daemon(self) -> None:
        """The background thread must be a daemon so it doesn't block process exit."""
        mock_server = _make_mock_server()
        mock_analyzer = _make_mock_analyzer()

        ch = _make_channel()
        with self._patch_constructors(mock_server, mock_analyzer):
            ch.start()

        try:
            assert ch._thread is not None
            assert ch._thread.daemon is True
        finally:
            mock_server._running = False

    def test_start_thread_name(self) -> None:
        """The background thread must be named 'missy-screencast-server'."""
        mock_server = _make_mock_server()
        mock_analyzer = _make_mock_analyzer()

        ch = _make_channel()
        with self._patch_constructors(mock_server, mock_analyzer):
            ch.start()

        try:
            assert ch._thread is not None
            assert ch._thread.name == "missy-screencast-server"
        finally:
            mock_server._running = False

    def test_get_status_after_start(self) -> None:
        """get_status() must delegate to server.get_status() after start()."""
        mock_server = _make_mock_server()
        mock_analyzer = _make_mock_analyzer()

        ch = _make_channel()
        with self._patch_constructors(mock_server, mock_analyzer):
            ch.start()

        try:
            status = ch.get_status()
            assert status["running"] is True
        finally:
            mock_server._running = False


# ---------------------------------------------------------------------------
# stop() tests
# ---------------------------------------------------------------------------

class TestStopLifecycle:
    """Tests for the stop() method."""

    def _start_channel_with_mocks(self, mock_server, mock_analyzer):
        """Start a channel with mocked dependencies and return it."""
        ch = _make_channel()
        with patch(
            "missy.channels.screencast.channel.ScreencastServer",
            return_value=mock_server,
        ), patch(
            "missy.channels.screencast.channel.FrameAnalyzer",
            return_value=mock_analyzer,
        ):
            ch.start()
        return ch

    def test_stop_clears_server_and_thread(self) -> None:
        """After stop(), _server, _thread, and _loop must all be None."""
        mock_server = _make_mock_server()
        mock_analyzer = _make_mock_analyzer()

        ch = self._start_channel_with_mocks(mock_server, mock_analyzer)
        ch.stop()

        assert ch._server is None
        assert ch._thread is None
        assert ch._loop is None

    def test_stop_calls_analyzer_and_server_stop(self) -> None:
        """stop() must call analyzer.stop() then server.stop() on the background loop."""
        mock_server = _make_mock_server()
        mock_analyzer = _make_mock_analyzer()

        ch = self._start_channel_with_mocks(mock_server, mock_analyzer)
        ch.stop()

        mock_analyzer.stop.assert_called_once()
        mock_server.stop.assert_called_once()

    def test_stop_is_idempotent_when_not_running(self) -> None:
        """Calling stop() on a channel that was never started must be a no-op."""
        ch = _make_channel()
        # Must not raise.
        ch.stop()
        ch.stop()

        assert ch._server is None
        assert ch._thread is None

    def test_stop_handles_future_timeout(self) -> None:
        """If the stop coroutine future times out, stop() must log and continue."""
        mock_server = MagicMock()
        mock_analyzer = MagicMock()

        # Manually wire up the channel as if it were running.
        ch = _make_channel()
        ch._server = mock_server
        ch._analyzer = mock_analyzer

        # Create a real event loop in a background thread but give it a future
        # that never completes to force the timeout path.
        loop = asyncio.new_event_loop()
        loop_started = threading.Event()

        def _run_loop():
            asyncio.set_event_loop(loop)
            loop_started.set()
            loop.run_forever()

        bg_thread = threading.Thread(target=_run_loop, daemon=True)
        bg_thread.start()
        loop_started.wait(timeout=3)

        ch._loop = loop
        ch._thread = bg_thread

        # Patch Future.result to raise TimeoutError simulating a slow shutdown.
        real_future_result = concurrent.futures.Future.result

        def _slow_result(self, timeout=None):
            raise concurrent.futures.TimeoutError("simulated timeout")

        with patch.object(concurrent.futures.Future, "result", _slow_result):
            # stop() should not raise even though the future times out.
            ch.stop()

        # Clean up the background loop.
        loop.call_soon_threadsafe(loop.stop)
        bg_thread.join(timeout=3)

        assert ch._server is None
        assert ch._loop is None

    def test_stop_handles_non_joining_thread(self) -> None:
        """If the thread does not join in time, stop() must log and continue without raising."""
        mock_server = _make_mock_server()
        mock_analyzer = _make_mock_analyzer()

        ch = self._start_channel_with_mocks(mock_server, mock_analyzer)

        # Replace thread with a mock that claims to still be alive after join.
        fake_thread = MagicMock(spec=threading.Thread)
        fake_thread.is_alive.return_value = True  # Simulate thread not dying.
        ch._thread = fake_thread

        # stop() must not raise even though the thread won't join cleanly.
        ch.stop()

        fake_thread.join.assert_called_once()
        assert ch._thread is None


# ---------------------------------------------------------------------------
# get_latest_analysis() and get_results() delegation tests
# ---------------------------------------------------------------------------

class TestSessionManagerDelegation:
    """Tests for get_latest_analysis() and get_results() when session manager is set."""

    def test_get_latest_analysis_delegates_to_session_manager(self) -> None:
        ch = _make_channel()

        fake_result = MagicMock(spec=AnalysisResult)
        mock_sm = MagicMock()
        mock_sm.get_latest_result.return_value = fake_result
        ch._session_manager = mock_sm

        result = ch.get_latest_analysis("sess-xyz")

        mock_sm.get_latest_result.assert_called_once_with("sess-xyz")
        assert result is fake_result

    def test_get_latest_analysis_returns_none_when_not_found(self) -> None:
        ch = _make_channel()

        mock_sm = MagicMock()
        mock_sm.get_latest_result.return_value = None
        ch._session_manager = mock_sm

        result = ch.get_latest_analysis("no-such-session")

        assert result is None

    def test_get_results_delegates_to_session_manager(self) -> None:
        ch = _make_channel()

        fake_results = [MagicMock(spec=AnalysisResult), MagicMock(spec=AnalysisResult)]
        mock_sm = MagicMock()
        mock_sm.get_results.return_value = fake_results
        ch._session_manager = mock_sm

        results = ch.get_results("sess-abc", limit=5)

        mock_sm.get_results.assert_called_once_with("sess-abc", 5)
        assert results is fake_results

    def test_get_results_default_limit(self) -> None:
        ch = _make_channel()

        mock_sm = MagicMock()
        mock_sm.get_results.return_value = []
        ch._session_manager = mock_sm

        ch.get_results("sess-abc")

        mock_sm.get_results.assert_called_once_with("sess-abc", 10)

    def test_get_results_returns_empty_when_no_session_manager(self) -> None:
        ch = _make_channel()
        assert ch.get_results("sess-xyz") == []

    def test_get_latest_analysis_returns_none_when_no_session_manager(self) -> None:
        ch = _make_channel()
        assert ch.get_latest_analysis("sess-xyz") is None


# ---------------------------------------------------------------------------
# _post_to_discord() async method tests
# ---------------------------------------------------------------------------

class TestPostToDiscord:
    """Tests for the _post_to_discord() async method."""

    @pytest.mark.asyncio
    async def test_post_to_discord_no_rest_client_returns_early(self) -> None:
        """When _discord_rest is None, _post_to_discord() must return without error."""
        ch = _make_channel()
        ch._discord_rest = None

        # Must not raise.
        await ch._post_to_discord("sess-1", "chan-1", "analysis text")

    @pytest.mark.asyncio
    async def test_post_to_discord_empty_channel_id_returns_early(self) -> None:
        """When discord_channel_id is empty, _post_to_discord() must return without error."""
        ch = _make_channel()
        ch._discord_rest = MagicMock()

        await ch._post_to_discord("sess-1", "", "analysis text")

        ch._discord_rest.send_message.assert_not_called()

    @pytest.mark.asyncio
    async def test_post_to_discord_sends_message_with_label(self) -> None:
        """When a session has a label, the Discord message header must include it."""
        ch = _make_channel()

        mock_rest = MagicMock()
        mock_rest.send_message = MagicMock()
        ch._discord_rest = mock_rest

        fake_session = MagicMock()
        fake_session.label = "my-label"
        mock_registry = MagicMock()
        mock_registry.get_session.return_value = fake_session
        ch._token_registry = mock_registry

        await ch._post_to_discord("sess-1", "chan-abc", "The screen shows X.")

        mock_rest.send_message.assert_called_once()
        _channel_arg, text_arg = mock_rest.send_message.call_args[0]
        assert _channel_arg == "chan-abc"
        assert "my-label" in text_arg
        assert "The screen shows X." in text_arg

    @pytest.mark.asyncio
    async def test_post_to_discord_sends_message_without_label(self) -> None:
        """When a session has no label, the header must not include parentheses for label."""
        ch = _make_channel()

        mock_rest = MagicMock()
        mock_rest.send_message = MagicMock()
        ch._discord_rest = mock_rest

        fake_session = MagicMock()
        fake_session.label = ""
        mock_registry = MagicMock()
        mock_registry.get_session.return_value = fake_session
        ch._token_registry = mock_registry

        await ch._post_to_discord("sess-1", "chan-abc", "Screen content here.")

        mock_rest.send_message.assert_called_once()
        _, text_arg = mock_rest.send_message.call_args[0]
        assert "Screen analysis:" in text_arg
        assert "(" not in text_arg.split("\n")[0]  # No label in the header line.

    @pytest.mark.asyncio
    async def test_post_to_discord_sends_message_no_registry(self) -> None:
        """When no token registry is set, label is empty and message is still sent."""
        ch = _make_channel()

        mock_rest = MagicMock()
        mock_rest.send_message = MagicMock()
        ch._discord_rest = mock_rest

        await ch._post_to_discord("sess-1", "chan-abc", "Screen content.")

        mock_rest.send_message.assert_called_once()
        _, text_arg = mock_rest.send_message.call_args[0]
        assert "Screen analysis:" in text_arg

    @pytest.mark.asyncio
    async def test_post_to_discord_truncates_long_analysis(self) -> None:
        """Analysis text that exceeds the budget must be truncated with ellipsis."""
        ch = _make_channel()

        mock_rest = MagicMock()
        mock_rest.send_message = MagicMock()
        ch._discord_rest = mock_rest

        # 3000-character body — well over the 2000-char Discord limit.
        long_text = "A" * 3000

        await ch._post_to_discord("sess-1", "chan-abc", long_text)

        mock_rest.send_message.assert_called_once()
        _, text_arg = mock_rest.send_message.call_args[0]
        assert len(text_arg) <= 2000
        assert text_arg.endswith("...")

    @pytest.mark.asyncio
    async def test_post_to_discord_short_analysis_not_truncated(self) -> None:
        """Analysis text within the budget must be sent verbatim."""
        ch = _make_channel()

        mock_rest = MagicMock()
        mock_rest.send_message = MagicMock()
        ch._discord_rest = mock_rest

        short_text = "Short analysis."
        await ch._post_to_discord("sess-1", "chan-abc", short_text)

        _, text_arg = mock_rest.send_message.call_args[0]
        assert short_text in text_arg
        assert not text_arg.endswith("...")

    @pytest.mark.asyncio
    async def test_post_to_discord_handles_send_message_exception(self) -> None:
        """If rest.send_message raises, _post_to_discord() must catch it and not re-raise."""
        ch = _make_channel()

        mock_rest = MagicMock()
        mock_rest.send_message = MagicMock(side_effect=ConnectionError("Discord unavailable"))
        ch._discord_rest = mock_rest

        # Must not propagate the exception.
        await ch._post_to_discord("sess-1", "chan-abc", "Analysis text.")

    @pytest.mark.asyncio
    async def test_post_to_discord_session_not_found_in_registry(self) -> None:
        """When get_session() returns None, no label is used but message is still sent."""
        ch = _make_channel()

        mock_rest = MagicMock()
        mock_rest.send_message = MagicMock()
        ch._discord_rest = mock_rest

        mock_registry = MagicMock()
        mock_registry.get_session.return_value = None
        ch._token_registry = mock_registry

        await ch._post_to_discord("sess-unknown", "chan-abc", "Analysis here.")

        mock_rest.send_message.assert_called_once()
        _, text_arg = mock_rest.send_message.call_args[0]
        assert "Screen analysis:" in text_arg
