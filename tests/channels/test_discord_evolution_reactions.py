"""Tests for Discord evolution reaction-based approval workflow.

Covers:
- add_evolution_reactions() adds ✅/❌ and tracks the proposal
- _handle_reaction() routes approve/reject based on emoji
- Ignores reactions from the bot itself
- Ignores reactions on non-tracked messages
- Ignores non-approve/reject emoji
- send_to() returns the last message ID
- Gateway intents include reaction bits
"""

from __future__ import annotations

import asyncio
from unittest.mock import MagicMock, patch

import pytest

from missy.channels.discord.channel import DiscordChannel
from missy.channels.discord.config import DiscordAccountConfig, DiscordDMPolicy

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def mock_rest() -> MagicMock:
    rest = MagicMock()
    rest.add_reaction.return_value = None
    rest.trigger_typing.return_value = None
    msg_response = MagicMock()
    msg_response.get.return_value = "sent-msg-123"
    rest.send_message.return_value = {"id": "sent-msg-123"}
    return rest


@pytest.fixture()
def channel(mock_rest: MagicMock) -> DiscordChannel:
    acct = DiscordAccountConfig(
        token_env_var="DISCORD_BOT_TOKEN",
        account_id="bot-001",
        dm_policy=DiscordDMPolicy.OPEN,
    )
    ch = DiscordChannel(account_config=acct)
    ch._rest = mock_rest
    ch._bot_user_id = "bot-001"
    return ch


# ---------------------------------------------------------------------------
# add_evolution_reactions
# ---------------------------------------------------------------------------


class TestAddEvolutionReactions:
    def test_adds_reactions_and_tracks(self, channel: DiscordChannel, mock_rest: MagicMock):
        channel.add_evolution_reactions("ch-1", "msg-1", "evo-abc")

        assert mock_rest.add_reaction.call_count == 2
        calls = [c.args for c in mock_rest.add_reaction.call_args_list]
        assert ("ch-1", "msg-1", "\u2705") in calls
        assert ("ch-1", "msg-1", "\u274c") in calls
        assert channel._pending_evolutions["msg-1"] == "evo-abc"

    def test_cleans_up_on_failure(self, channel: DiscordChannel, mock_rest: MagicMock):
        mock_rest.add_reaction.side_effect = Exception("no perms")
        channel.add_evolution_reactions("ch-1", "msg-1", "evo-xyz")
        assert "msg-1" not in channel._pending_evolutions


# ---------------------------------------------------------------------------
# _handle_reaction
# ---------------------------------------------------------------------------


class TestHandleReaction:
    def test_ignores_own_reaction(self, channel: DiscordChannel, mock_rest: MagicMock):
        channel._pending_evolutions["msg-1"] = "evo-1"
        data = {
            "message_id": "msg-1",
            "user_id": "bot-001",
            "channel_id": "ch-1",
            "emoji": {"name": "\u2705"},
        }
        asyncio.get_event_loop().run_until_complete(channel._handle_reaction(data))
        mock_rest.send_message.assert_not_called()

    def test_ignores_untracked_message(self, channel: DiscordChannel, mock_rest: MagicMock):
        data = {
            "message_id": "msg-unknown",
            "user_id": "user-42",
            "channel_id": "ch-1",
            "emoji": {"name": "\u2705"},
        }
        asyncio.get_event_loop().run_until_complete(channel._handle_reaction(data))
        mock_rest.send_message.assert_not_called()

    def test_ignores_irrelevant_emoji(self, channel: DiscordChannel, mock_rest: MagicMock):
        channel._pending_evolutions["msg-1"] = "evo-1"
        data = {
            "message_id": "msg-1",
            "user_id": "user-42",
            "channel_id": "ch-1",
            "emoji": {"name": "\U0001f44d"},  # 👍 — not ✅ or ❌
        }
        asyncio.get_event_loop().run_until_complete(channel._handle_reaction(data))
        mock_rest.send_message.assert_not_called()

    def test_approve_reaction(self, channel: DiscordChannel, mock_rest: MagicMock):
        channel._pending_evolutions["msg-1"] = "evo-1"
        mock_mgr = MagicMock()
        mock_mgr.approve.return_value = True

        data = {
            "message_id": "msg-1",
            "user_id": "user-42",
            "channel_id": "ch-1",
            "emoji": {"name": "\u2705"},
        }
        with patch(
            "missy.agent.code_evolution.CodeEvolutionManager",
            return_value=mock_mgr,
        ):
            asyncio.get_event_loop().run_until_complete(channel._handle_reaction(data))

        mock_mgr.approve.assert_called_once_with("evo-1")
        mock_rest.send_message.assert_called_once()
        sent_content = mock_rest.send_message.call_args.kwargs.get(
            "content", mock_rest.send_message.call_args[1].get("content", "")
        )
        # Handle both positional and keyword args
        if not sent_content:
            sent_content = mock_rest.send_message.call_args[0][1] if len(mock_rest.send_message.call_args[0]) > 1 else ""
        assert "approved" in sent_content.lower()
        assert "msg-1" not in channel._pending_evolutions

    def test_reject_reaction(self, channel: DiscordChannel, mock_rest: MagicMock):
        channel._pending_evolutions["msg-1"] = "evo-1"
        mock_mgr = MagicMock()
        mock_mgr.reject.return_value = True

        data = {
            "message_id": "msg-1",
            "user_id": "user-42",
            "channel_id": "ch-1",
            "emoji": {"name": "\u274c"},
        }
        with patch(
            "missy.agent.code_evolution.CodeEvolutionManager",
            return_value=mock_mgr,
        ):
            asyncio.get_event_loop().run_until_complete(channel._handle_reaction(data))

        mock_mgr.reject.assert_called_once_with("evo-1")
        assert "msg-1" not in channel._pending_evolutions

    def test_approve_failure_sends_error(self, channel: DiscordChannel, mock_rest: MagicMock):
        channel._pending_evolutions["msg-1"] = "evo-1"
        mock_mgr = MagicMock()
        mock_mgr.approve.return_value = False

        data = {
            "message_id": "msg-1",
            "user_id": "user-42",
            "channel_id": "ch-1",
            "emoji": {"name": "\u2705"},
        }
        with patch(
            "missy.agent.code_evolution.CodeEvolutionManager",
            return_value=mock_mgr,
        ):
            asyncio.get_event_loop().run_until_complete(channel._handle_reaction(data))

        # Should still send a message (error message)
        mock_rest.send_message.assert_called_once()


# ---------------------------------------------------------------------------
# send_to returns message ID
# ---------------------------------------------------------------------------


class TestSendToReturnsId:
    def test_returns_last_message_id(self, channel: DiscordChannel, mock_rest: MagicMock):
        result = asyncio.get_event_loop().run_until_complete(
            channel.send_to("ch-1", "hello")
        )
        assert result == "sent-msg-123"

    def test_raises_on_error(self, channel: DiscordChannel, mock_rest: MagicMock):
        from missy.channels.discord.channel import DiscordSendError

        mock_rest.send_message.side_effect = Exception("fail")
        with pytest.raises(DiscordSendError):
            asyncio.get_event_loop().run_until_complete(
                channel.send_to("ch-1", "hello")
            )


# ---------------------------------------------------------------------------
# Gateway intents
# ---------------------------------------------------------------------------


class TestGatewayIntents:
    def test_intents_include_reaction_bits(self):
        from missy.channels.discord.gateway import _INTENTS

        guild_message_reactions = 1024
        direct_message_reactions = 8192
        assert _INTENTS & guild_message_reactions == guild_message_reactions
        assert _INTENTS & direct_message_reactions == direct_message_reactions


# ---------------------------------------------------------------------------
# Gateway forwards reaction events
# ---------------------------------------------------------------------------


class TestGatewayForwardsReactions:
    def test_message_reaction_add_forwarded(self):
        from missy.channels.discord.gateway import DiscordGatewayClient

        received = []

        async def on_msg(payload):
            received.append(payload)

        gw = DiscordGatewayClient(bot_token="Bot test", on_message=on_msg)
        # Simulate a DISPATCH with MESSAGE_REACTION_ADD
        asyncio.get_event_loop().run_until_complete(
            gw._handle_dispatch("MESSAGE_REACTION_ADD", {"message_id": "m1"})
        )
        assert len(received) == 1
        assert received[0]["t"] == "MESSAGE_REACTION_ADD"
