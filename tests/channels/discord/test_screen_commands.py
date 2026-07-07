"""Tests for Discord screencast command handlers.

Tests both the bang-prefixed ``!screen ...`` syntax and the
natural-language fallback in
:func:`~missy.channels.discord.screen_commands.maybe_handle_screen_command`
and :func:`~missy.channels.discord.screen_commands.infer_screen_intent`.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from missy.channels.discord.screen_commands import (
    ScreenIntent,
    infer_screen_intent,
    maybe_handle_screen_command,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_screencast():
    """Mock ScreencastChannel with all methods used by the handler."""
    screencast = MagicMock()
    screencast.create_session.return_value = (
        "sess-1",
        "tok-1",
        "https://example.test/share/sess-1",
    )
    screencast.get_active_sessions.return_value = []
    screencast.revoke_session.return_value = True
    screencast.get_latest_analysis.return_value = None
    screencast.get_status.return_value = {"running": True, "host": "0.0.0.0", "port": 9000}
    return screencast


@pytest.fixture
def base_kwargs(mock_screencast):
    return {
        "content": "",
        "channel_id": "12345",
        "author_id": "11111",
        "screencast": mock_screencast,
    }


# ---------------------------------------------------------------------------
# infer_screen_intent — positive natural-language phrasings
# ---------------------------------------------------------------------------


class TestInferShare:
    def test_share_my_screen(self):
        assert infer_screen_intent("share my screen") == ScreenIntent(action="share")

    def test_share_the_screen(self):
        assert infer_screen_intent("share the screen") == ScreenIntent(action="share")

    def test_start_a_screen_share(self):
        assert infer_screen_intent("start a screen share") == ScreenIntent(action="share")

    def test_start_a_screen_share_with_label(self):
        result = infer_screen_intent("start a screen share labelled Demo Time")
        assert result == ScreenIntent(action="share", label="Demo Time")

    def test_share_my_screen_with_label(self):
        result = infer_screen_intent("share my screen called Standup")
        assert result == ScreenIntent(action="share", label="Standup")

    def test_let_me_show_you_my_screen(self):
        assert infer_screen_intent("let me show you my screen") == ScreenIntent(action="share")

    def test_show_you_my_screen(self):
        assert infer_screen_intent("show you my screen") == ScreenIntent(action="share")

    def test_politeness_stripped(self):
        assert infer_screen_intent("can you share my screen") == ScreenIntent(action="share")


class TestInferList:
    def test_list_screen_sessions(self):
        assert infer_screen_intent("list screen sessions") == ScreenIntent(action="list")

    def test_what_screen_shares_are_active(self):
        assert infer_screen_intent("what screen shares are active") == ScreenIntent(action="list")

    def test_show_active_screen_sessions(self):
        assert infer_screen_intent("show active screen sessions") == ScreenIntent(action="list")

    def test_show_screen_shares(self):
        assert infer_screen_intent("show screen shares") == ScreenIntent(action="list")


class TestInferStop:
    def test_stop_the_screen_share(self):
        assert infer_screen_intent("stop the screen share") == ScreenIntent(action="stop")

    def test_stop_screen_share_with_id(self):
        result = infer_screen_intent("stop screen share abc123")
        assert result == ScreenIntent(action="stop", session_id="abc123")

    def test_end_my_screen_session(self):
        assert infer_screen_intent("end my screen session") == ScreenIntent(action="stop")

    def test_end_the_screen_share_with_id(self):
        result = infer_screen_intent("end the screen share abc123")
        assert result == ScreenIntent(action="stop", session_id="abc123")


class TestInferAnalyze:
    def test_analyze_the_screen(self):
        assert infer_screen_intent("analyze the screen") == ScreenIntent(action="analyze")

    def test_analyze_the_screen_share_with_id(self):
        result = infer_screen_intent("analyze the screen share abc123")
        assert result == ScreenIntent(action="analyze", session_id="abc123")

    def test_whats_on_the_screen(self):
        assert infer_screen_intent("what's on the screen") == ScreenIntent(action="analyze")

    def test_whats_on_the_screen_no_apostrophe(self):
        assert infer_screen_intent("whats on the screen") == ScreenIntent(action="analyze")

    def test_look_at_my_screen_share(self):
        assert infer_screen_intent("look at my screen share") == ScreenIntent(action="analyze")


class TestTrailingFiller:
    """Trailing conversational filler must not defeat the fullmatch patterns."""

    @pytest.mark.parametrize(
        "text",
        [
            "what's on the screen right now",
            "what's on the screen now",
            "analyze the screen currently",
            "share my screen please",
            "screencast status now",
        ],
    )
    def test_filler_still_matches(self, text):
        assert infer_screen_intent(text) is not None

    def test_real_session_id_survives_filler_stripping(self):
        # A genuine id (not a filler word) must be preserved.
        assert infer_screen_intent("stop the screen share abc123") == ScreenIntent(
            action="stop", session_id="abc123"
        )

    def test_trailing_now_is_stripped_not_treated_as_id(self):
        assert infer_screen_intent("stop the screen share now") == ScreenIntent(
            action="stop", session_id=None
        )


class TestInferStatus:
    def test_screen_server_status(self):
        assert infer_screen_intent("screen server status") == ScreenIntent(action="status")

    def test_screencast_status(self):
        assert infer_screen_intent("screencast status") == ScreenIntent(action="status")

    def test_screen_status(self):
        assert infer_screen_intent("screen status") == ScreenIntent(action="status")


# ---------------------------------------------------------------------------
# infer_screen_intent — negative cases (must stay conservative)
# ---------------------------------------------------------------------------


class TestInferNegative:
    @pytest.mark.parametrize(
        "text",
        [
            "hello world",
            "how are you today",
            "thanks a lot",
            "what time is it",
            "stop the music",
            "share my location",
            "list my tasks",
            "",
            "   ",
        ],
    )
    def test_ordinary_conversation_not_matched(self, text):
        assert infer_screen_intent(text) is None

    def test_none_content_not_matched(self):
        assert infer_screen_intent(None) is None

    def test_bang_command_not_matched_by_nl_parser(self):
        # Bang commands are handled separately; the NL parser should not
        # also claim them.
        assert infer_screen_intent("!screen share") is None

    @pytest.mark.parametrize(
        "text",
        [
            "save the screenshot",
            "analyze this picture",
            "analyze the screenshot",
            "analyze this image",
            "take a screenshot",
            "!screenshot save",
        ],
    )
    def test_screenshot_and_image_phrasings_not_matched(self, text):
        # These belong to the image command handler, not the screencast one.
        assert infer_screen_intent(text) is None


# ---------------------------------------------------------------------------
# maybe_handle_screen_command — bang commands still work
# ---------------------------------------------------------------------------


class TestBangCommandsBackwardCompatible:
    @pytest.mark.asyncio
    async def test_bang_share(self, base_kwargs, mock_screencast):
        base_kwargs["content"] = "!screen share My Label"
        result = await maybe_handle_screen_command(**base_kwargs)
        assert result.handled is True
        mock_screencast.create_session.assert_called_once_with(
            created_by="11111",
            discord_channel_id="12345",
            label="My Label",
        )
        assert "sess-1" in result.reply

    @pytest.mark.asyncio
    async def test_bang_list(self, base_kwargs, mock_screencast):
        base_kwargs["content"] = "!screen list"
        result = await maybe_handle_screen_command(**base_kwargs)
        assert result.handled is True
        assert "No active" in result.reply

    @pytest.mark.asyncio
    async def test_bang_stop(self, base_kwargs, mock_screencast):
        base_kwargs["content"] = "!screen stop abc123"
        result = await maybe_handle_screen_command(**base_kwargs)
        assert result.handled is True
        mock_screencast.revoke_session.assert_called_once_with("abc123")

    @pytest.mark.asyncio
    async def test_bang_analyze(self, base_kwargs, mock_screencast):
        mock_screencast.get_active_sessions.return_value = [
            {"session_id": "sess-1", "created_at": 0, "last_frame_at": None}
        ]
        base_kwargs["content"] = "!screen analyze"
        result = await maybe_handle_screen_command(**base_kwargs)
        assert result.handled is True
        mock_screencast.get_latest_analysis.assert_called_once_with("sess-1")

    @pytest.mark.asyncio
    async def test_bang_status(self, base_kwargs):
        base_kwargs["content"] = "!screen status"
        result = await maybe_handle_screen_command(**base_kwargs)
        assert result.handled is True
        assert "Screencast server status" in result.reply

    @pytest.mark.asyncio
    async def test_bang_missing_subcommand_shows_usage(self, base_kwargs):
        base_kwargs["content"] = "!screen"
        result = await maybe_handle_screen_command(**base_kwargs)
        assert result.handled is True
        assert "Usage" in result.reply

    @pytest.mark.asyncio
    async def test_bang_unknown_subcommand_not_handled(self, base_kwargs):
        base_kwargs["content"] = "!screen bogus"
        result = await maybe_handle_screen_command(**base_kwargs)
        assert result.handled is False

    @pytest.mark.asyncio
    async def test_bang_screenshot_not_handled_by_screen_handler(self, base_kwargs):
        # "!screenshot" starts with the "!screen" prefix but is a distinct
        # command owned by the image handler.
        base_kwargs["content"] = "!screenshot save"
        result = await maybe_handle_screen_command(**base_kwargs)
        assert result.handled is False


# ---------------------------------------------------------------------------
# maybe_handle_screen_command — natural-language fallback
# ---------------------------------------------------------------------------


class TestNaturalLanguageHandling:
    @pytest.mark.asyncio
    async def test_share_my_screen(self, base_kwargs, mock_screencast):
        base_kwargs["content"] = "share my screen"
        result = await maybe_handle_screen_command(**base_kwargs)
        assert result.handled is True
        mock_screencast.create_session.assert_called_once_with(
            created_by="11111",
            discord_channel_id="12345",
            label="screen share",
        )

    @pytest.mark.asyncio
    async def test_start_screen_share_with_label(self, base_kwargs, mock_screencast):
        base_kwargs["content"] = "start a screen share labelled Standup"
        result = await maybe_handle_screen_command(**base_kwargs)
        assert result.handled is True
        mock_screencast.create_session.assert_called_once_with(
            created_by="11111",
            discord_channel_id="12345",
            label="Standup",
        )

    @pytest.mark.asyncio
    async def test_list_screen_sessions(self, base_kwargs, mock_screencast):
        base_kwargs["content"] = "list screen sessions"
        result = await maybe_handle_screen_command(**base_kwargs)
        assert result.handled is True
        mock_screencast.get_active_sessions.assert_called_once()

    @pytest.mark.asyncio
    async def test_stop_the_screen_share(self, base_kwargs, mock_screencast):
        mock_screencast.get_active_sessions.return_value = [
            {"session_id": "sess-9", "created_at": 0, "last_frame_at": None}
        ]
        base_kwargs["content"] = "stop the screen share"
        result = await maybe_handle_screen_command(**base_kwargs)
        assert result.handled is True
        mock_screencast.revoke_session.assert_called_once_with("sess-9")

    @pytest.mark.asyncio
    async def test_stop_screen_share_with_id(self, base_kwargs, mock_screencast):
        base_kwargs["content"] = "stop screen share abc123"
        result = await maybe_handle_screen_command(**base_kwargs)
        assert result.handled is True
        mock_screencast.revoke_session.assert_called_once_with("abc123")

    @pytest.mark.asyncio
    async def test_whats_on_the_screen(self, base_kwargs, mock_screencast):
        mock_screencast.get_active_sessions.return_value = [
            {"session_id": "sess-2", "created_at": 0, "last_frame_at": None}
        ]
        base_kwargs["content"] = "what's on the screen"
        result = await maybe_handle_screen_command(**base_kwargs)
        assert result.handled is True
        mock_screencast.get_latest_analysis.assert_called_once_with("sess-2")

    @pytest.mark.asyncio
    async def test_analyze_screen_share_with_id(self, base_kwargs, mock_screencast):
        base_kwargs["content"] = "analyze the screen share abc123"
        result = await maybe_handle_screen_command(**base_kwargs)
        assert result.handled is True
        mock_screencast.get_latest_analysis.assert_called_once_with("abc123")

    @pytest.mark.asyncio
    async def test_screencast_status(self, base_kwargs):
        base_kwargs["content"] = "screencast status"
        result = await maybe_handle_screen_command(**base_kwargs)
        assert result.handled is True
        assert "Screencast server status" in result.reply

    @pytest.mark.asyncio
    async def test_ordinary_conversation_not_handled(self, base_kwargs):
        base_kwargs["content"] = "hello, how's it going?"
        result = await maybe_handle_screen_command(**base_kwargs)
        assert result.handled is False
        assert result.reply is None

    @pytest.mark.asyncio
    async def test_screenshot_phrasing_not_handled(self, base_kwargs, mock_screencast):
        base_kwargs["content"] = "analyze this picture"
        result = await maybe_handle_screen_command(**base_kwargs)
        assert result.handled is False
        mock_screencast.get_latest_analysis.assert_not_called()

    @pytest.mark.asyncio
    async def test_save_screenshot_phrasing_not_handled(self, base_kwargs, mock_screencast):
        base_kwargs["content"] = "save the screenshot"
        result = await maybe_handle_screen_command(**base_kwargs)
        assert result.handled is False


# ---------------------------------------------------------------------------
# Guard conditions
# ---------------------------------------------------------------------------


class TestGuardConditions:
    @pytest.mark.asyncio
    async def test_screencast_not_enabled_bang(self, base_kwargs):
        base_kwargs["content"] = "!screen status"
        base_kwargs["screencast"] = None
        result = await maybe_handle_screen_command(**base_kwargs)
        assert result.handled is True
        assert "not enabled" in result.reply.lower()

    @pytest.mark.asyncio
    async def test_screencast_not_enabled_nl(self, base_kwargs):
        base_kwargs["content"] = "screencast status"
        base_kwargs["screencast"] = None
        result = await maybe_handle_screen_command(**base_kwargs)
        assert result.handled is True
        assert "not enabled" in result.reply.lower()
