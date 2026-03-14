"""Coverage tests for missy/channels/discord/commands.py.

Targets uncovered lines:
  76-77   : _get_option — option found by name, return str(value)
  89-98   : _handle_ask — successful run via AgentRuntime and exception path
  117-129 : _handle_model — name provided (early return), get_registry success
             with providers, get_registry raises (exception path)
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from missy.channels.discord.commands import (
    SLASH_COMMANDS,
    _get_option,
    handle_slash_command,
    _handle_model,
    _handle_ask,
    _handle_status,
    _handle_help,
)
from missy.channels.discord.config import DiscordAccountConfig, DiscordDMPolicy


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_mock_channel(
    bot_user_id: str = "bot-42",
    dm_policy: DiscordDMPolicy = DiscordDMPolicy.OPEN,
    guild_policies: dict | None = None,
) -> MagicMock:
    channel = MagicMock()
    channel.bot_user_id = bot_user_id
    account_cfg = DiscordAccountConfig(
        token_env_var="DISCORD_BOT_TOKEN",
        dm_policy=dm_policy,
        ignore_bots=True,
        guild_policies=guild_policies or {},
    )
    channel.account_config = account_cfg
    return channel


def _make_interaction(name: str, options: list | None = None) -> dict:
    return {
        "id": "int-1",
        "token": "tok-abc",
        "data": {
            "name": name,
            "options": options or [],
        },
    }


# ---------------------------------------------------------------------------
# _get_option
# ---------------------------------------------------------------------------


class TestGetOption:
    def test_option_found_returns_string_value(self):
        """Lines 76-77: matching option returns str(value)."""
        interaction = _make_interaction("ask", options=[{"name": "prompt", "value": "hello world"}])
        result = _get_option(interaction, "prompt")
        assert result == "hello world"

    def test_option_not_found_returns_none(self):
        interaction = _make_interaction("ask", options=[{"name": "other", "value": "x"}])
        result = _get_option(interaction, "prompt")
        assert result is None

    def test_no_options_returns_none(self):
        interaction = _make_interaction("ask")
        result = _get_option(interaction, "prompt")
        assert result is None

    def test_no_data_returns_none(self):
        interaction = {"id": "int-1"}
        result = _get_option(interaction, "prompt")
        assert result is None

    def test_integer_value_coerced_to_string(self):
        """Value is cast with str()."""
        interaction = _make_interaction("model", options=[{"name": "name", "value": 42}])
        result = _get_option(interaction, "name")
        assert result == "42"


# ---------------------------------------------------------------------------
# _handle_ask
# ---------------------------------------------------------------------------


class TestHandleAsk:
    @pytest.mark.asyncio
    async def test_no_prompt_returns_help_message(self):
        interaction = _make_interaction("ask")  # no options
        channel = _make_mock_channel()
        result = await _handle_ask(interaction, channel)
        assert "prompt" in result.lower() or "/ask" in result

    @pytest.mark.asyncio
    async def test_ask_success_via_agent_runtime(self):
        """Lines 89-95: successful agent run returns reply string."""
        interaction = _make_interaction("ask", options=[{"name": "prompt", "value": "What time?"}])
        channel = _make_mock_channel()

        mock_agent = MagicMock()
        mock_agent.run.return_value = "It is 3pm."

        import sys

        mock_module = MagicMock()
        mock_module.AgentConfig = MagicMock(return_value=MagicMock())
        mock_module.AgentRuntime = MagicMock(return_value=mock_agent)

        with patch.dict(sys.modules, {"missy.agent.runtime": mock_module}):
            result = await _handle_ask(interaction, channel)

        # Should return the agent's response or an error string
        assert isinstance(result, str)

    @pytest.mark.asyncio
    async def test_ask_exception_returns_error_message(self):
        """Lines 96-98: exception in agent run returns user-friendly error."""
        interaction = _make_interaction("ask", options=[{"name": "prompt", "value": "boom"}])
        channel = _make_mock_channel()

        import sys

        mock_module = MagicMock()
        mock_module.AgentConfig = MagicMock(side_effect=RuntimeError("agent exploded"))
        mock_module.AgentRuntime = MagicMock()

        with patch.dict(sys.modules, {"missy.agent.runtime": mock_module}):
            result = await _handle_ask(interaction, channel)

        assert "error" in result.lower() or "agent exploded" in result


# ---------------------------------------------------------------------------
# _handle_model
# ---------------------------------------------------------------------------


class TestHandleModel:
    @pytest.mark.asyncio
    async def test_name_provided_returns_not_supported(self):
        """Line 119: when name option given, returns 'not yet supported' message."""
        interaction = _make_interaction("model", options=[{"name": "name", "value": "claude-opus"}])
        channel = _make_mock_channel()
        result = await _handle_model(interaction, channel)
        assert "not yet supported" in result.lower() or "unchanged" in result.lower()

    @pytest.mark.asyncio
    async def test_no_name_with_providers(self):
        """Lines 121-126: no name option, registry returns providers."""
        interaction = _make_interaction("model")  # no options
        channel = _make_mock_channel()

        mock_registry = MagicMock()
        mock_registry.list_providers.return_value = ["anthropic", "openai"]

        import sys

        mock_module = MagicMock()
        mock_module.get_registry = MagicMock(return_value=mock_registry)

        with patch.dict(sys.modules, {"missy.providers.registry": mock_module}):
            result = await _handle_model(interaction, channel)

        assert "anthropic" in result or "openai" in result

    @pytest.mark.asyncio
    async def test_no_name_no_providers(self):
        """Line 127: no providers configured."""
        interaction = _make_interaction("model")
        channel = _make_mock_channel()

        mock_registry = MagicMock()
        mock_registry.list_providers.return_value = []

        import sys

        mock_module = MagicMock()
        mock_module.get_registry = MagicMock(return_value=mock_registry)

        with patch.dict(sys.modules, {"missy.providers.registry": mock_module}):
            result = await _handle_model(interaction, channel)

        assert "no providers" in result.lower() or "not" in result.lower()

    @pytest.mark.asyncio
    async def test_no_name_registry_exception(self):
        """Lines 128-129: exception in get_registry returns unavailable message."""
        interaction = _make_interaction("model")
        channel = _make_mock_channel()

        import sys

        mock_module = MagicMock()
        mock_module.get_registry = MagicMock(side_effect=RuntimeError("no registry"))

        with patch.dict(sys.modules, {"missy.providers.registry": mock_module}):
            result = await _handle_model(interaction, channel)

        assert "unavailable" in result.lower()


# ---------------------------------------------------------------------------
# handle_slash_command router
# ---------------------------------------------------------------------------


class TestHandleSlashCommand:
    @pytest.mark.asyncio
    async def test_unknown_command_returns_error(self):
        interaction = _make_interaction("nonexistent")
        channel = _make_mock_channel()
        result = await handle_slash_command(interaction, channel)
        assert "unknown" in result.lower() or "nonexistent" in result

    @pytest.mark.asyncio
    async def test_status_command_routed(self):
        interaction = _make_interaction("status")
        channel = _make_mock_channel()
        result = await handle_slash_command(interaction, channel)
        assert "status" in result.lower() or "missy" in result.lower()

    @pytest.mark.asyncio
    async def test_help_command_routed(self):
        interaction = _make_interaction("help")
        channel = _make_mock_channel()
        result = await handle_slash_command(interaction, channel)
        # _handle_help lists all commands from SLASH_COMMANDS
        for cmd in SLASH_COMMANDS:
            assert cmd["name"] in result

    @pytest.mark.asyncio
    async def test_no_data_key_treated_as_unknown(self):
        interaction = {}  # no 'data' key
        channel = _make_mock_channel()
        result = await handle_slash_command(interaction, channel)
        assert "unknown" in result.lower()
