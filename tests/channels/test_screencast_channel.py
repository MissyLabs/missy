"""Tests for missy/channels/screencast/channel.py."""

from __future__ import annotations

import threading
import time
from typing import Any
from unittest.mock import MagicMock, patch

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
