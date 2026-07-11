"""Coverage tests for missy/channels/discord/commands.py.

Targets uncovered lines:
  76-77   : _get_option — option found by name, return str(value)
  89-98   : _handle_ask — successful run via AgentRuntime and exception path
  117-129 : _handle_model — name provided (early return), get_registry success
             with providers, get_registry raises (exception path)
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from missy.channels.discord.commands import (
    SLASH_COMMANDS,
    _get_option,
    _handle_ask,
    _handle_model,
    _interaction_author_id,
    handle_slash_command,
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
        """Successful agent run returns reply string."""
        interaction = _make_interaction("ask", options=[{"name": "prompt", "value": "What time?"}])
        channel = _make_mock_channel()

        mock_agent = MagicMock()
        mock_agent.run.return_value = "It is 3pm."
        channel._agent_runtime = mock_agent

        result = await _handle_ask(interaction, channel)

        assert result == "It is 3pm."

    @pytest.mark.asyncio
    async def test_ask_no_runtime_returns_unavailable(self):
        """When agent runtime is None, returns unavailable message."""
        interaction = _make_interaction("ask", options=[{"name": "prompt", "value": "hi"}])
        channel = _make_mock_channel()
        channel._agent_runtime = None

        result = await _handle_ask(interaction, channel)

        assert "not available" in result.lower()

    @pytest.mark.asyncio
    async def test_ask_exception_returns_error_message(self):
        """Exception in agent run returns user-friendly error."""
        interaction = _make_interaction("ask", options=[{"name": "prompt", "value": "boom"}])
        channel = _make_mock_channel()

        mock_agent = MagicMock()
        mock_agent.run.side_effect = RuntimeError("agent exploded")
        channel._agent_runtime = mock_agent

        result = await _handle_ask(interaction, channel)

        assert "error" in result.lower() or "agent exploded" in result

    @pytest.mark.asyncio
    async def test_ask_preserves_whitespace_multiline_and_quotes_verbatim(self):
        """DISC-CMD-001/002 (task #10 validation): a prompt with extra
        leading/trailing whitespace, embedded blank lines, a quoted
        phrase, and a tab character must reach agent.run() byte-for-byte
        -- no trimming, no silent truncation, no mangling. Discord's
        slash-command UI collects the whole `prompt` option as one
        opaque string; there is no free-text tokenizer in this path to
        misparse whitespace/quotes, but nothing should alter the value
        either."""
        raw_prompt = (
            "   leading/trailing whitespace   \n\n"
            'multiline\ncontent   with "quoted phrase" and \tembedded tab'
        )
        interaction = _make_interaction("ask", options=[{"name": "prompt", "value": raw_prompt}])
        channel = _make_mock_channel()

        mock_agent = MagicMock()
        mock_agent.run.return_value = "ok"
        channel._agent_runtime = mock_agent

        await _handle_ask(interaction, channel)

        assert mock_agent.run.call_args[0][0] == raw_prompt

    @pytest.mark.asyncio
    async def test_ask_preserves_long_multi_requirement_prompt_without_truncation(self):
        """DISC-CMD-002: a long, multi-requirement brief must not be
        silently dropped or truncated before it reaches the agent."""
        long_prompt = "Requirement 1: do X.\n" * 200 + "Final constraint: never do Y."
        interaction = _make_interaction("ask", options=[{"name": "prompt", "value": long_prompt}])
        channel = _make_mock_channel()

        mock_agent = MagicMock()
        mock_agent.run.return_value = "ok"
        channel._agent_runtime = mock_agent

        await _handle_ask(interaction, channel)

        forwarded = mock_agent.run.call_args[0][0]
        assert forwarded == long_prompt
        assert "Final constraint: never do Y." in forwarded


# ---------------------------------------------------------------------------
# _interaction_author_id / per-user session scoping (SR-1.13 critical fix)
#
# _handle_ask() used to hardcode session_id="discord" for every user in
# every guild, so every /ask interaction across the whole bot shared one
# conversation history -- one user's prompts and the agent's replies to
# them became context for every other user's /ask calls.
# ---------------------------------------------------------------------------


class TestInteractionAuthorId:
    def test_extracts_from_guild_member(self):
        interaction = {"member": {"user": {"id": "guild-user-1"}}}
        assert _interaction_author_id(interaction) == "guild-user-1"

    def test_extracts_from_dm_user(self):
        interaction = {"user": {"id": "dm-user-1"}}
        assert _interaction_author_id(interaction) == "dm-user-1"

    def test_member_user_preferred_over_top_level_user(self):
        interaction = {"member": {"user": {"id": "guild-user"}}, "user": {"id": "dm-user"}}
        assert _interaction_author_id(interaction) == "guild-user"

    def test_missing_author_returns_empty_string(self):
        assert _interaction_author_id({}) == ""


class TestHandleAskSessionScopedPerUser:
    @pytest.mark.asyncio
    async def test_session_id_is_the_invoking_users_id_not_hardcoded(self):
        interaction = _make_interaction("ask", options=[{"name": "prompt", "value": "hi"}])
        interaction["member"] = {"user": {"id": "user-alice"}}
        channel = _make_mock_channel()

        mock_agent = MagicMock()
        mock_agent.run.return_value = "reply"
        channel._agent_runtime = mock_agent

        await _handle_ask(interaction, channel)

        mock_agent.run.assert_called_once_with("hi", "user-alice")

    @pytest.mark.asyncio
    async def test_two_different_users_get_two_different_session_ids(self):
        channel = _make_mock_channel()
        mock_agent = MagicMock()
        mock_agent.run.return_value = "reply"
        channel._agent_runtime = mock_agent

        alice_interaction = _make_interaction("ask", options=[{"name": "prompt", "value": "a"}])
        alice_interaction["member"] = {"user": {"id": "user-alice"}}
        await _handle_ask(alice_interaction, channel)

        bob_interaction = _make_interaction("ask", options=[{"name": "prompt", "value": "b"}])
        bob_interaction["member"] = {"user": {"id": "user-bob"}}
        await _handle_ask(bob_interaction, channel)

        session_ids = [call.args[1] for call in mock_agent.run.call_args_list]
        assert session_ids == ["user-alice", "user-bob"]
        assert session_ids[0] != session_ids[1]

    @pytest.mark.asyncio
    async def test_dm_interaction_session_id_from_top_level_user(self):
        interaction = _make_interaction("ask", options=[{"name": "prompt", "value": "hi"}])
        interaction["user"] = {"id": "dm-carol"}
        channel = _make_mock_channel()

        mock_agent = MagicMock()
        mock_agent.run.return_value = "reply"
        channel._agent_runtime = mock_agent

        await _handle_ask(interaction, channel)

        mock_agent.run.assert_called_once_with("hi", "dm-carol")

    @pytest.mark.asyncio
    async def test_missing_author_falls_back_to_discord_literal(self):
        # No member/user field at all (shouldn't happen in practice, but
        # must not crash) -- falls back to the old shared literal rather
        # than an empty string.
        interaction = _make_interaction("ask", options=[{"name": "prompt", "value": "hi"}])
        channel = _make_mock_channel()

        mock_agent = MagicMock()
        mock_agent.run.return_value = "reply"
        channel._agent_runtime = mock_agent

        await _handle_ask(interaction, channel)

        mock_agent.run.assert_called_once_with("hi", "discord")


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
