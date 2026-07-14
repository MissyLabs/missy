"""Tests for the Discord evolution reaction workflow.

Covers:
- add_evolution_reactions() adds ✅/❌ and tracks the proposal
- _handle_reaction() routes approve/reject based on emoji
- Ignores reactions from the bot itself
- Ignores reactions on non-tracked messages
- Ignores non-approve/reject emoji
- SR-1.2/1.3 + owner allowlist: a ✅/❌ reaction from a Discord user is
  only treated as real approval/rejection when that user's ID is in
  DiscordAccountConfig.owner_ids -- CodeEvolutionManager.approve()/
  .reject() must never be called for a non-owner reactor, since a bare
  Discord reaction is not by itself an authenticated human operator.
  With no owners configured (the default), both actions stay fully
  refused, matching the original SR-1.2/1.3 behavior; `missy evolve
  approve/reject <id>` run from a terminal on the host always works
  regardless of owner configuration.
- A denied (non-owner) or failed (already-resolved) reaction must not
  clear the pending-tracking entry, so the real owner can still act on
  the same message afterward.
- A successful owner approval posts a follow-up confirmation message
  with its own apply/cancel reactions (_pending_applies, tracked
  separately from _pending_evolutions so the two stages can never be
  confused with each other).
- _handle_evolution_apply_reaction(): owner-gated apply (real
  CodeEvolutionManager.apply() call, including the failure/auto-revert
  path) and cancel (reject()) on that follow-up message.
- send_to() returns the last message ID
- Gateway intents include reaction bits
"""

from __future__ import annotations

import asyncio
import time
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


def _run(coro):
    return asyncio.run(coro)


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
        _run(channel._handle_reaction(data))
        mock_rest.send_message.assert_not_called()

    def test_ignores_untracked_message(self, channel: DiscordChannel, mock_rest: MagicMock):
        data = {
            "message_id": "msg-unknown",
            "user_id": "user-42",
            "channel_id": "ch-1",
            "emoji": {"name": "\u2705"},
        }
        _run(channel._handle_reaction(data))
        mock_rest.send_message.assert_not_called()

    def test_ignores_irrelevant_emoji(self, channel: DiscordChannel, mock_rest: MagicMock):
        channel._pending_evolutions["msg-1"] = "evo-1"
        data = {
            "message_id": "msg-1",
            "user_id": "user-42",
            "channel_id": "ch-1",
            "emoji": {"name": "\U0001f44d"},  # 👍 — not ✅ or ❌
        }
        _run(channel._handle_reaction(data))
        mock_rest.send_message.assert_not_called()

    def test_approve_reaction_is_refused_for_non_owner(
        self, channel: DiscordChannel, mock_rest: MagicMock
    ):
        # SR-1.2/1.3 + owner allowlist: a non-owner Discord user reacting
        # with an emoji must never be able to approve a code-evolution
        # proposal. The `channel` fixture configures no owner_ids, so
        # every reactor is a non-owner here. mgr.approve() must never be
        # called from this path.
        channel._pending_evolutions["msg-1"] = "evo-1"
        mock_mgr = MagicMock()

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
            _run(channel._handle_reaction(data))

        mock_mgr.approve.assert_not_called()
        mock_rest.send_message.assert_called_once()
        sent_content = mock_rest.send_message.call_args.kwargs.get(
            "content", mock_rest.send_message.call_args[1].get("content", "")
        )
        # Handle both positional and keyword args
        if not sent_content:
            sent_content = (
                mock_rest.send_message.call_args[0][1]
                if len(mock_rest.send_message.call_args[0]) > 1
                else ""
            )
        assert "not a configured owner" in sent_content
        assert "missy evolve approve" in sent_content
        # A denial must not consume the pending tracking entry -- the
        # real owner must still be able to act on this message afterward.
        assert channel._pending_evolutions.get("msg-1") == "evo-1"

    def test_approve_reaction_succeeds_for_configured_owner(
        self, channel: DiscordChannel, mock_rest: MagicMock
    ):
        channel.account_config.owner_ids = ["owner-1"]
        channel._pending_evolutions["msg-1"] = "evo-1"
        mock_mgr = MagicMock()
        mock_mgr.approve.return_value = True

        data = {
            "message_id": "msg-1",
            "user_id": "owner-1",
            "channel_id": "ch-1",
            "emoji": {"name": "\u2705"},
        }
        with patch(
            "missy.agent.code_evolution.CodeEvolutionManager",
            return_value=mock_mgr,
        ):
            _run(channel._handle_reaction(data))

        mock_mgr.approve.assert_called_once_with("evo-1")
        sent_content = mock_rest.send_message.call_args[0][1]
        assert "approved" in sent_content.lower()
        assert "missy evolve apply" in sent_content
        # Resolved -- tracking is cleared.
        assert "msg-1" not in channel._pending_evolutions
        # A successful approve posts a follow-up confirmation message and
        # starts tracking apply/cancel reactions on it (sent-msg-123 is
        # the mock_rest fixture's stubbed new-message id).
        assert channel._pending_applies.get("sent-msg-123") == "evo-1"
        assert mock_rest.add_reaction.call_count == 2

    def test_reject_reaction_denied_for_non_owner(
        self, channel: DiscordChannel, mock_rest: MagicMock
    ):
        """Reject was previously open to any reactor; it is now
        owner-gated the same as approve, for a consistent security model
        across both real approve/reject decisions."""
        channel._pending_evolutions["msg-1"] = "evo-1"
        mock_mgr = MagicMock()

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
            _run(channel._handle_reaction(data))

        mock_mgr.reject.assert_not_called()
        assert channel._pending_evolutions.get("msg-1") == "evo-1"

    def test_reject_reaction_succeeds_for_configured_owner(
        self, channel: DiscordChannel, mock_rest: MagicMock
    ):
        channel.account_config.owner_ids = ["owner-1"]
        channel._pending_evolutions["msg-1"] = "evo-1"
        mock_mgr = MagicMock()
        mock_mgr.reject.return_value = True

        data = {
            "message_id": "msg-1",
            "user_id": "owner-1",
            "channel_id": "ch-1",
            "emoji": {"name": "\u274c"},
        }
        with patch(
            "missy.agent.code_evolution.CodeEvolutionManager",
            return_value=mock_mgr,
        ):
            _run(channel._handle_reaction(data))

        mock_mgr.reject.assert_called_once_with("evo-1")
        assert "msg-1" not in channel._pending_evolutions

    def test_approve_reaction_emits_deny_audit_event(
        self, channel: DiscordChannel, mock_rest: MagicMock
    ):
        channel._pending_evolutions["msg-1"] = "evo-1"
        mock_mgr = MagicMock()

        data = {
            "message_id": "msg-1",
            "user_id": "user-42",
            "channel_id": "ch-1",
            "emoji": {"name": "\u2705"},
        }
        with (
            patch("missy.agent.code_evolution.CodeEvolutionManager", return_value=mock_mgr),
            patch.object(channel, "_emit_audit") as mock_audit,
        ):
            _run(channel._handle_reaction(data))

        mock_audit.assert_called_once()
        call_args = mock_audit.call_args.args
        assert call_args[0] == "discord.evolution.approve_denied"
        assert call_args[1] == "deny"

    def test_owner_ids_defaults_to_empty_fail_closed(self, channel: DiscordChannel) -> None:
        assert channel.account_config.owner_ids == []


# ---------------------------------------------------------------------------
# _handle_evolution_apply_reaction -- apply/cancel on the follow-up
# confirmation message posted after a successful approve.
# ---------------------------------------------------------------------------


class TestHandleEvolutionApplyReaction:
    def test_apply_succeeds_for_configured_owner(
        self, channel: DiscordChannel, mock_rest: MagicMock
    ):
        channel.account_config.owner_ids = ["owner-1"]
        channel._pending_applies["apply-msg-1"] = "evo-1"
        mock_mgr = MagicMock()
        mock_mgr.apply.return_value = {
            "success": True,
            "message": "Evolution applied and committed: abc12345",
            "commit_sha": "abc12345def",
            "test_output": "",
        }

        data = {
            "message_id": "apply-msg-1",
            "user_id": "owner-1",
            "channel_id": "ch-1",
            "emoji": {"name": "✅"},
        }
        with patch(
            "missy.agent.code_evolution.CodeEvolutionManager",
            return_value=mock_mgr,
        ):
            _run(channel._handle_reaction(data))

        mock_mgr.apply.assert_called_once_with("evo-1")
        sent_content = mock_rest.send_message.call_args[0][1]
        assert "applied" in sent_content.lower()
        assert "abc12345" in sent_content
        assert "apply-msg-1" not in channel._pending_applies
        # FX-apply-blocking: an immediate progress message is sent
        # *before* the (potentially slow) apply() call, proving the
        # dispatch is the non-blocking send-then-await-executor shape
        # rather than one single blocking call with one final message.
        assert mock_rest.send_message.call_count == 2
        progress_content = mock_rest.send_message.call_args_list[0][0][1]
        assert "applying" in progress_content.lower()

    def test_apply_dispatched_via_executor_not_blocking_event_loop(
        self, channel: DiscordChannel, mock_rest: MagicMock
    ):
        """FX-apply-blocking: live-observed root cause of "Missy stopped
        responding" after hitting apply via Discord -- mgr.apply() was
        called directly on the coroutine handling the reaction, blocking
        the single-threaded asyncio event loop the whole gateway
        connection (heartbeats, every other channel's messages) runs on
        for however long the real test-suite subprocess took. Proven
        here by running a second, concurrent coroutine on the same event
        loop alongside the reaction handler -- it must keep making
        progress (real interleaving) instead of stalling until apply()
        returns."""
        channel.account_config.owner_ids = ["owner-1"]
        channel._pending_applies["apply-msg-1"] = "evo-1"
        mock_mgr = MagicMock()

        def _slow_apply(_proposal_id):
            time.sleep(0.2)
            return {"success": True, "message": "ok", "commit_sha": "abc123", "test_output": ""}

        mock_mgr.apply.side_effect = _slow_apply

        data = {
            "message_id": "apply-msg-1",
            "user_id": "owner-1",
            "channel_id": "ch-1",
            "emoji": {"name": "✅"},
        }

        async def _scenario():
            other_progress = []

            async def _other_coroutine():
                for i in range(10):
                    other_progress.append(i)
                    await asyncio.sleep(0.02)

            with patch(
                "missy.agent.code_evolution.CodeEvolutionManager",
                return_value=mock_mgr,
            ):
                await asyncio.gather(
                    channel._handle_reaction(data),
                    _other_coroutine(),
                )
            return other_progress

        progress = asyncio.run(_scenario())
        # If apply() had blocked the event loop, _other_coroutine()
        # couldn't have interleaved any of its awaits during the ~0.2s
        # apply() call, and progress would be far shorter (or empty).
        assert len(progress) == 10

    def test_apply_failure_reports_message_and_stays_resolved(
        self, channel: DiscordChannel, mock_rest: MagicMock
    ):
        """A failed apply (e.g. tests failed, auto-reverted by
        CodeEvolutionManager.apply() itself) is still a resolved apply
        *attempt* -- the confirmation message's buttons shouldn't stay
        active for a retry, since the proposal is now in a terminal
        FAILED state."""
        channel.account_config.owner_ids = ["owner-1"]
        channel._pending_applies["apply-msg-1"] = "evo-1"
        mock_mgr = MagicMock()
        mock_mgr.apply.return_value = {
            "success": False,
            "message": "Tests failed. Changes reverted.",
            "commit_sha": "",
            "test_output": "AssertionError: ...",
        }

        data = {
            "message_id": "apply-msg-1",
            "user_id": "owner-1",
            "channel_id": "ch-1",
            "emoji": {"name": "✅"},
        }
        with patch(
            "missy.agent.code_evolution.CodeEvolutionManager",
            return_value=mock_mgr,
        ):
            _run(channel._handle_reaction(data))

        sent_content = mock_rest.send_message.call_args[0][1]
        assert "failed" in sent_content.lower()
        assert "Tests failed. Changes reverted." in sent_content
        assert "apply-msg-1" not in channel._pending_applies

    def test_apply_denied_for_non_owner(self, channel: DiscordChannel, mock_rest: MagicMock):
        channel._pending_applies["apply-msg-1"] = "evo-1"
        mock_mgr = MagicMock()

        data = {
            "message_id": "apply-msg-1",
            "user_id": "random-user",
            "channel_id": "ch-1",
            "emoji": {"name": "✅"},
        }
        with patch(
            "missy.agent.code_evolution.CodeEvolutionManager",
            return_value=mock_mgr,
        ):
            _run(channel._handle_reaction(data))

        mock_mgr.apply.assert_not_called()
        sent_content = mock_rest.send_message.call_args[0][1]
        assert "not a configured owner" in sent_content
        # Denial must not consume the pending tracking entry.
        assert channel._pending_applies.get("apply-msg-1") == "evo-1"

    def test_cancel_succeeds_for_configured_owner(
        self, channel: DiscordChannel, mock_rest: MagicMock
    ):
        channel.account_config.owner_ids = ["owner-1"]
        channel._pending_applies["apply-msg-1"] = "evo-1"
        mock_mgr = MagicMock()
        mock_mgr.reject.return_value = True

        data = {
            "message_id": "apply-msg-1",
            "user_id": "owner-1",
            "channel_id": "ch-1",
            "emoji": {"name": "❌"},
        }
        with patch(
            "missy.agent.code_evolution.CodeEvolutionManager",
            return_value=mock_mgr,
        ):
            _run(channel._handle_reaction(data))

        mock_mgr.reject.assert_called_once_with("evo-1")
        mock_mgr.apply.assert_not_called()
        sent_content = mock_rest.send_message.call_args[0][1]
        assert "cancelled" in sent_content.lower()
        assert "apply-msg-1" not in channel._pending_applies

    def test_cancel_denied_for_non_owner(self, channel: DiscordChannel, mock_rest: MagicMock):
        channel._pending_applies["apply-msg-1"] = "evo-1"
        mock_mgr = MagicMock()

        data = {
            "message_id": "apply-msg-1",
            "user_id": "random-user",
            "channel_id": "ch-1",
            "emoji": {"name": "❌"},
        }
        with patch(
            "missy.agent.code_evolution.CodeEvolutionManager",
            return_value=mock_mgr,
        ):
            _run(channel._handle_reaction(data))

        mock_mgr.reject.assert_not_called()
        assert channel._pending_applies.get("apply-msg-1") == "evo-1"

    def test_apply_and_approval_tracking_dicts_are_independent(
        self, channel: DiscordChannel, mock_rest: MagicMock
    ):
        """A reaction on a proposal message must route through approval
        handling even while a *different* message is pending apply, and
        vice versa -- the two dicts must never cross-contaminate."""
        channel.account_config.owner_ids = ["owner-1"]
        channel._pending_evolutions["proposal-msg"] = "evo-1"
        channel._pending_applies["apply-msg"] = "evo-2"
        mock_mgr = MagicMock()
        mock_mgr.approve.return_value = True

        data = {
            "message_id": "proposal-msg",
            "user_id": "owner-1",
            "channel_id": "ch-1",
            "emoji": {"name": "✅"},
        }
        with patch(
            "missy.agent.code_evolution.CodeEvolutionManager",
            return_value=mock_mgr,
        ):
            _run(channel._handle_reaction(data))

        mock_mgr.approve.assert_called_once_with("evo-1")
        mock_mgr.apply.assert_not_called()
        # The unrelated pending apply entry is untouched.
        assert channel._pending_applies.get("apply-msg") == "evo-2"

    def test_reaction_on_untracked_message_is_ignored(
        self, channel: DiscordChannel, mock_rest: MagicMock
    ):
        data = {
            "message_id": "totally-unknown",
            "user_id": "owner-1",
            "channel_id": "ch-1",
            "emoji": {"name": "✅"},
        }
        _run(channel._handle_reaction(data))
        mock_rest.send_message.assert_not_called()


# ---------------------------------------------------------------------------
# send_to returns message ID
# ---------------------------------------------------------------------------


class TestSendToReturnsId:
    def test_returns_last_message_id(self, channel: DiscordChannel, mock_rest: MagicMock):
        result = _run(channel.send_to("ch-1", "hello"))
        assert result == "sent-msg-123"

    def test_raises_on_error(self, channel: DiscordChannel, mock_rest: MagicMock):
        from missy.channels.discord.channel import DiscordSendError

        mock_rest.send_message.side_effect = Exception("fail")
        with pytest.raises(DiscordSendError):
            _run(channel.send_to("ch-1", "hello"))


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
        _run(gw._handle_dispatch("MESSAGE_REACTION_ADD", {"message_id": "m1"}))
        assert len(received) == 1
        assert received[0]["t"] == "MESSAGE_REACTION_ADD"
