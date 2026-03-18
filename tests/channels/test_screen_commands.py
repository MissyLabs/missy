"""Tests for missy/channels/discord/screen_commands.py."""

from __future__ import annotations

import time
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from missy.channels.discord.screen_commands import (
    ScreenCommandResult,
    _format_duration,
    maybe_handle_screen_command,
)
from missy.channels.screencast.session_manager import AnalysisResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fake_session(
    session_id: str = "s-001",
    label: str = "test",
    created_by: str = "user1",
    frame_count: int = 5,
    analysis_count: int = 2,
    last_frame_at: float = 0.0,
) -> dict:
    return {
        "session_id": session_id,
        "label": label,
        "created_by": created_by,
        "created_at": time.time() - 60,
        "frame_count": frame_count,
        "analysis_count": analysis_count,
        "last_frame_at": last_frame_at,
    }


async def _invoke(
    content: str,
    screencast: object | None = None,
    channel_id: str = "chan-1",
    author_id: str = "user-1",
) -> ScreenCommandResult:
    return await maybe_handle_screen_command(
        content=content,
        channel_id=channel_id,
        author_id=author_id,
        screencast=screencast,
    )


# ---------------------------------------------------------------------------
# Routing / dispatch
# ---------------------------------------------------------------------------

class TestRouting:
    async def test_non_screen_command(self) -> None:
        result = await _invoke("hello world")
        assert result.handled is False
        assert result.reply is None

    async def test_bare_screen_command(self) -> None:
        result = await _invoke("!screen")
        assert result.handled is True
        assert result.reply is not None
        assert "Usage" in result.reply

    async def test_unknown_subcommand(self) -> None:
        result = await _invoke("!screen foo")
        assert result.handled is False

    async def test_screencast_none_any_subcommand(self) -> None:
        result = await _invoke("!screen share", screencast=None)
        assert result.handled is True
        assert result.reply is not None
        assert "not enabled" in result.reply.lower()


# ---------------------------------------------------------------------------
# !screen share
# ---------------------------------------------------------------------------

class TestShare:
    async def test_share_success(self) -> None:
        sc = MagicMock()
        sc.create_session.return_value = (
            "sess-abc",
            "tok-xyz",
            "http://127.0.0.1:8780/?session_id=sess-abc&token=tok-xyz",
        )

        result = await _invoke("!screen share my session", screencast=sc)

        assert result.handled is True
        assert result.reply is not None
        assert "sess-abc" in result.reply
        assert "http://127.0.0.1:8780" in result.reply

    async def test_share_failure(self) -> None:
        sc = MagicMock()
        sc.create_session.side_effect = RuntimeError("server unavailable")

        result = await _invoke("!screen share", screencast=sc)

        assert result.handled is True
        assert result.reply is not None
        assert "Failed" in result.reply or "failed" in result.reply


# ---------------------------------------------------------------------------
# !screen list
# ---------------------------------------------------------------------------

class TestList:
    async def test_list_no_sessions(self) -> None:
        sc = MagicMock()
        sc.get_active_sessions.return_value = []

        result = await _invoke("!screen list", screencast=sc)

        assert result.handled is True
        assert result.reply is not None
        assert "No active" in result.reply

    async def test_list_with_sessions(self) -> None:
        sc = MagicMock()
        sc.get_active_sessions.return_value = [
            _fake_session("s-001", label="work", frame_count=10, analysis_count=3),
            _fake_session("s-002", label="demo", frame_count=0, analysis_count=0),
        ]

        result = await _invoke("!screen list", screencast=sc)

        assert result.handled is True
        assert result.reply is not None
        assert "s-001" in result.reply
        assert "s-002" in result.reply
        assert "work" in result.reply


# ---------------------------------------------------------------------------
# !screen stop
# ---------------------------------------------------------------------------

class TestStop:
    async def test_stop_no_sessions(self) -> None:
        sc = MagicMock()
        sc.get_active_sessions.return_value = []

        result = await _invoke("!screen stop", screencast=sc)

        assert result.handled is True
        assert result.reply is not None
        assert "No active sessions to stop" in result.reply

    async def test_stop_with_id_success(self) -> None:
        sc = MagicMock()
        sc.revoke_session.return_value = True

        result = await _invoke("!screen stop s-001", screencast=sc)

        assert result.handled is True
        assert result.reply is not None
        assert "stopped" in result.reply.lower()
        sc.revoke_session.assert_called_once_with("s-001")

    async def test_stop_not_found(self) -> None:
        sc = MagicMock()
        sc.revoke_session.return_value = False

        result = await _invoke("!screen stop missing-id", screencast=sc)

        assert result.handled is True
        assert result.reply is not None
        assert "not found" in result.reply.lower()

    async def test_stop_no_id_uses_last_session(self) -> None:
        """When no session_id given, the last active session is stopped."""
        sc = MagicMock()
        sessions = [
            _fake_session("s-first"),
            _fake_session("s-last"),
        ]
        sc.get_active_sessions.return_value = sessions
        sc.revoke_session.return_value = True

        result = await _invoke("!screen stop", screencast=sc)

        assert result.handled is True
        sc.revoke_session.assert_called_once_with("s-last")


# ---------------------------------------------------------------------------
# !screen analyze
# ---------------------------------------------------------------------------

class TestAnalyze:
    async def test_analyze_no_sessions(self) -> None:
        sc = MagicMock()
        sc.get_active_sessions.return_value = []

        result = await _invoke("!screen analyze", screencast=sc)

        assert result.handled is True
        assert result.reply is not None
        assert "No active sessions" in result.reply

    async def test_analyze_with_result(self) -> None:
        sc = MagicMock()
        analysis = AnalysisResult(
            session_id="s-001",
            frame_number=42,
            analysis_text="The screen shows a terminal.",
            model="llava",
            processing_ms=350,
        )
        sc.get_latest_analysis.return_value = analysis

        result = await _invoke("!screen analyze s-001", screencast=sc)

        assert result.handled is True
        assert result.reply is not None
        assert "s-001" in result.reply
        assert "terminal" in result.reply
        assert "42" in result.reply
        assert "llava" in result.reply
        assert "350ms" in result.reply

    async def test_analyze_no_result(self) -> None:
        sc = MagicMock()
        sc.get_latest_analysis.return_value = None

        result = await _invoke("!screen analyze s-001", screencast=sc)

        assert result.handled is True
        assert result.reply is not None
        assert "No analysis results" in result.reply


# ---------------------------------------------------------------------------
# !screen status
# ---------------------------------------------------------------------------

class TestStatus:
    async def test_status_not_running(self) -> None:
        sc = MagicMock()
        sc.get_status.return_value = {"running": False}

        result = await _invoke("!screen status", screencast=sc)

        assert result.handled is True
        assert result.reply is not None
        assert "not running" in result.reply.lower()

    async def test_status_running(self) -> None:
        sc = MagicMock()
        sc.get_status.return_value = {
            "running": True,
            "host": "127.0.0.1",
            "port": 8780,
            "active_connections": 2,
            "sessions": {
                "connected_sessions": 2,
                "max_sessions": 20,
                "queue_size": 1,
            },
        }

        result = await _invoke("!screen status", screencast=sc)

        assert result.handled is True
        assert result.reply is not None
        assert "Running: yes" in result.reply
        assert "127.0.0.1:8780" in result.reply
        assert "2/20" in result.reply


# ---------------------------------------------------------------------------
# _format_duration
# ---------------------------------------------------------------------------

class TestFormatDuration:
    def test_format_duration_seconds(self) -> None:
        assert _format_duration(45) == "45s"

    def test_format_duration_zero(self) -> None:
        assert _format_duration(0) == "0s"

    def test_format_duration_exactly_one_minute(self) -> None:
        assert _format_duration(60) == "1m 0s"

    def test_format_duration_minutes(self) -> None:
        assert _format_duration(125) == "2m 5s"

    def test_format_duration_hours(self) -> None:
        assert _format_duration(3725) == "1h 2m"

    def test_format_duration_exactly_one_hour(self) -> None:
        assert _format_duration(3600) == "1h 0m"

    def test_format_duration_large(self) -> None:
        # 2h 30m
        assert _format_duration(9000) == "2h 30m"
