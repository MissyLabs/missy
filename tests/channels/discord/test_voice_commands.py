"""Tests for Discord voice command handlers.

Tests the command parsing and routing in
:func:`~missy.channels.discord.voice_commands.maybe_handle_voice_command`.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from missy.channels.discord.voice_commands import (
    maybe_handle_voice_command,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_voice():
    """Mock DiscordVoiceManager with all methods."""
    voice = MagicMock()
    voice.is_ready = True
    voice.can_listen = True
    voice.can_speak = True
    voice.join = AsyncMock(return_value="General")
    voice.leave = AsyncMock(return_value="General")
    voice.say = AsyncMock()
    return voice


@pytest.fixture
def base_kwargs(mock_voice):
    return {
        "content": "",
        "channel_id": "12345",
        "guild_id": "99999",
        "author_id": "11111",
        "voice": mock_voice,
    }


# ---------------------------------------------------------------------------
# Non-command messages
# ---------------------------------------------------------------------------


class TestNonCommands:
    @pytest.mark.asyncio
    async def test_non_command_not_handled(self, base_kwargs):
        base_kwargs["content"] = "hello world"
        result = await maybe_handle_voice_command(**base_kwargs)
        assert result.handled is False
        assert result.reply is None

    @pytest.mark.asyncio
    async def test_empty_string_not_handled(self, base_kwargs):
        base_kwargs["content"] = ""
        result = await maybe_handle_voice_command(**base_kwargs)
        assert result.handled is False

    @pytest.mark.asyncio
    async def test_none_content_not_handled(self, base_kwargs):
        base_kwargs["content"] = None
        result = await maybe_handle_voice_command(**base_kwargs)
        assert result.handled is False

    @pytest.mark.asyncio
    async def test_unknown_bang_command_not_handled(self, base_kwargs):
        base_kwargs["content"] = "!unknown"
        result = await maybe_handle_voice_command(**base_kwargs)
        assert result.handled is False

    @pytest.mark.asyncio
    async def test_whitespace_only_not_handled(self, base_kwargs):
        base_kwargs["content"] = "   "
        result = await maybe_handle_voice_command(**base_kwargs)
        assert result.handled is False


# ---------------------------------------------------------------------------
# Guard conditions
# ---------------------------------------------------------------------------


class TestGuardConditions:
    @pytest.mark.asyncio
    async def test_no_guild_id(self, base_kwargs):
        base_kwargs["content"] = "!join"
        base_kwargs["guild_id"] = None
        result = await maybe_handle_voice_command(**base_kwargs)
        assert result.handled is True
        assert "servers" in result.reply.lower()

    @pytest.mark.asyncio
    async def test_no_voice_manager(self, base_kwargs):
        base_kwargs["content"] = "!join"
        base_kwargs["voice"] = None
        result = await maybe_handle_voice_command(**base_kwargs)
        assert result.handled is True
        assert "not enabled" in result.reply.lower()

    @pytest.mark.asyncio
    async def test_voice_not_ready(self, base_kwargs, mock_voice):
        base_kwargs["content"] = "!join"
        mock_voice.is_ready = False
        result = await maybe_handle_voice_command(**base_kwargs)
        assert result.handled is True
        assert "starting up" in result.reply.lower()


# ---------------------------------------------------------------------------
# !join command
# ---------------------------------------------------------------------------


class TestJoinCommand:
    @pytest.mark.asyncio
    async def test_join_no_args_uses_user_id(self, base_kwargs, mock_voice):
        base_kwargs["content"] = "!join"
        result = await maybe_handle_voice_command(**base_kwargs)
        assert result.handled is True
        mock_voice.join.assert_awaited_once_with(99999, user_id=11111)
        assert "General" in result.reply

    @pytest.mark.asyncio
    async def test_join_by_channel_name(self, base_kwargs, mock_voice):
        base_kwargs["content"] = "!join Music"
        result = await maybe_handle_voice_command(**base_kwargs)
        assert result.handled is True
        mock_voice.join.assert_awaited_once_with(99999, channel_name="Music")
        assert "General" in result.reply

    @pytest.mark.asyncio
    async def test_join_by_channel_id(self, base_kwargs, mock_voice):
        base_kwargs["content"] = "!join 777888999"
        result = await maybe_handle_voice_command(**base_kwargs)
        assert result.handled is True
        mock_voice.join.assert_awaited_once_with(99999, channel_id=777888999)

    @pytest.mark.asyncio
    async def test_join_shows_capabilities(self, base_kwargs, mock_voice):
        base_kwargs["content"] = "!join"
        mock_voice.can_listen = True
        mock_voice.can_speak = True
        result = await maybe_handle_voice_command(**base_kwargs)
        assert "listening" in result.reply
        assert "speaking" in result.reply

    @pytest.mark.asyncio
    async def test_join_listen_only(self, base_kwargs, mock_voice):
        base_kwargs["content"] = "!join"
        mock_voice.can_listen = True
        mock_voice.can_speak = False
        result = await maybe_handle_voice_command(**base_kwargs)
        assert "listening" in result.reply
        assert "speaking" not in result.reply

    @pytest.mark.asyncio
    async def test_join_error_returns_message(self, base_kwargs, mock_voice):
        from missy.channels.discord.voice import DiscordVoiceError

        base_kwargs["content"] = "!join"
        mock_voice.join = AsyncMock(side_effect=DiscordVoiceError("Not in a voice channel"))
        result = await maybe_handle_voice_command(**base_kwargs)
        assert result.handled is True
        assert "Not in a voice channel" in result.reply


# ---------------------------------------------------------------------------
# !leave command
# ---------------------------------------------------------------------------


class TestLeaveCommand:
    @pytest.mark.asyncio
    async def test_leave_success(self, base_kwargs, mock_voice):
        base_kwargs["content"] = "!leave"
        result = await maybe_handle_voice_command(**base_kwargs)
        assert result.handled is True
        mock_voice.leave.assert_awaited_once_with(99999)
        assert "General" in result.reply

    @pytest.mark.asyncio
    async def test_leave_not_in_channel(self, base_kwargs, mock_voice):
        base_kwargs["content"] = "!leave"
        mock_voice.leave = AsyncMock(return_value=None)
        result = await maybe_handle_voice_command(**base_kwargs)
        assert result.handled is True
        assert "not in" in result.reply.lower()

    @pytest.mark.asyncio
    async def test_leave_error(self, base_kwargs, mock_voice):
        from missy.channels.discord.voice import DiscordVoiceError

        base_kwargs["content"] = "!leave"
        mock_voice.leave = AsyncMock(side_effect=DiscordVoiceError("Connection lost"))
        result = await maybe_handle_voice_command(**base_kwargs)
        assert result.handled is True
        assert "Connection lost" in result.reply


# ---------------------------------------------------------------------------
# !say command
# ---------------------------------------------------------------------------


class TestSayCommand:
    @pytest.mark.asyncio
    async def test_say_success(self, base_kwargs, mock_voice):
        base_kwargs["content"] = "!say Hello world"
        result = await maybe_handle_voice_command(**base_kwargs)
        assert result.handled is True
        mock_voice.say.assert_awaited_once_with(99999, "Hello world")
        assert result.reply is None  # No reply on success

    @pytest.mark.asyncio
    async def test_say_no_text(self, base_kwargs, mock_voice):
        base_kwargs["content"] = "!say"
        result = await maybe_handle_voice_command(**base_kwargs)
        assert result.handled is True
        assert "Usage" in result.reply

    @pytest.mark.asyncio
    async def test_say_only_whitespace(self, base_kwargs, mock_voice):
        base_kwargs["content"] = "!say   "
        result = await maybe_handle_voice_command(**base_kwargs)
        assert result.handled is True
        assert "Usage" in result.reply

    @pytest.mark.asyncio
    async def test_say_error(self, base_kwargs, mock_voice):
        from missy.channels.discord.voice import DiscordVoiceError

        base_kwargs["content"] = "!say test"
        mock_voice.say = AsyncMock(side_effect=DiscordVoiceError("TTS not configured"))
        result = await maybe_handle_voice_command(**base_kwargs)
        assert result.handled is True
        assert "TTS not configured" in result.reply


# ---------------------------------------------------------------------------
# Case sensitivity
# ---------------------------------------------------------------------------


class TestCaseSensitivity:
    @pytest.mark.asyncio
    async def test_join_uppercase(self, base_kwargs, mock_voice):
        base_kwargs["content"] = "!JOIN"
        result = await maybe_handle_voice_command(**base_kwargs)
        assert result.handled is True
        mock_voice.join.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_leave_mixed_case(self, base_kwargs, mock_voice):
        base_kwargs["content"] = "!Leave"
        result = await maybe_handle_voice_command(**base_kwargs)
        assert result.handled is True
        mock_voice.leave.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_say_mixed_case(self, base_kwargs, mock_voice):
        base_kwargs["content"] = "!SAY hello"
        result = await maybe_handle_voice_command(**base_kwargs)
        assert result.handled is True
        mock_voice.say.assert_awaited_once()
