"""Coverage tests for missy/channels/discord/channel.py.

Targets the most impactful uncovered paths (by line count):
  119         : __init__ — no token found → logs error
  150         : set_agent_runtime()
  162-175     : start() — application_id set → register_slash_commands called; exception logged
  179-186     : stop() — gateway_task cancel/await
  201         : receive() — raises NotImplementedError
  213         : areceive() — returns queue item
  225-232     : send() — no channel context → drop; no event loop → drop
  261-262     : send_to() — thread_id provided as target; message chunk sent
  417-433     : _on_gateway_event() — GUILD_CREATE, INTERACTION_CREATE, MESSAGE_REACTION_ADD
  467         : _handle_message — own-bot message is dropped
  482-483     : _handle_message — voice command handled → early return
  511-512     : _handle_message — secrets detected, message deleted path
  532-547     : _handle_message — bot author filtered
  618-659     : _maybe_handle_voice_command — various voice command paths
  663-693     : _handle_interaction() — interaction response
  725         : _check_dm_policy — PAIRING path falls through to _check_pairing
  757         : _check_dm_policy — unknown policy returns False
  781-784     : _check_pairing — !pair deny command
  826-851     : _check_guild_policy — channel allowlist deny, user allowlist deny
  869-870     : _check_guild_policy — require_mention with no own_id (fallback)
  1071        : _handle_reaction — reject path
  1080-1082   : _handle_reaction — exception in CodeEvolutionManager
  1108-1109   : _emit_audit — exception swallowed
"""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from missy.channels.discord.channel import DiscordChannel
from missy.channels.discord.config import (
    DiscordAccountConfig,
    DiscordDMPolicy,
    DiscordGuildPolicy,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_account(
    dm_policy: DiscordDMPolicy = DiscordDMPolicy.OPEN,
    account_id: str = "bot-001",
    guild_policies: dict | None = None,
    dm_allowlist: list[str] | None = None,
    ignore_bots: bool = True,
    allow_bots_if_mention_only: bool = False,
    auto_thread_threshold: int = 0,
) -> DiscordAccountConfig:
    return DiscordAccountConfig(
        token="direct-token",
        token_env_var="DISCORD_BOT_TOKEN",
        account_id=account_id,
        dm_policy=dm_policy,
        guild_policies=guild_policies or {},
        dm_allowlist=dm_allowlist or [],
        ignore_bots=ignore_bots,
        allow_bots_if_mention_only=allow_bots_if_mention_only,
        auto_thread_threshold=auto_thread_threshold,
    )


def _make_channel(account: DiscordAccountConfig | None = None) -> DiscordChannel:
    if account is None:
        account = _make_account()
    with (
        patch("missy.channels.discord.channel.DiscordGatewayClient"),
        patch("missy.channels.discord.channel.DiscordRestClient"),
    ):
        return DiscordChannel(account_config=account)


def _make_message(
    author_id: str = "user-1",
    content: str = "hello",
    guild_id: str | None = None,
    channel_id: str = "chan-1",
    is_bot: bool = False,
    message_id: str = "msg-1",
    attachments: list | None = None,
) -> dict[str, Any]:
    return {
        "id": message_id,
        "channel_id": channel_id,
        "guild_id": guild_id,
        "content": content,
        "author": {"id": author_id, "username": "TestUser", "bot": is_bot},
        "attachments": attachments or [],
    }


# ---------------------------------------------------------------------------
# __init__ — no token found (line 119)
# ---------------------------------------------------------------------------


class TestInitNoToken:
    def test_no_token_logs_error(self, caplog):
        import logging

        account = DiscordAccountConfig(
            token=None,
            token_env_var="DISCORD_MISSING_ENV_VAR_12345",
        )
        with (
            patch("missy.channels.discord.channel.DiscordGatewayClient"),
            patch("missy.channels.discord.channel.DiscordRestClient"),
            caplog.at_level(logging.ERROR, logger="missy.channels.discord.channel"),
        ):
            DiscordChannel(account_config=account)

        assert any("no bot token" in r.message.lower() for r in caplog.records)


# ---------------------------------------------------------------------------
# set_agent_runtime (line 150)
# ---------------------------------------------------------------------------


class TestSetAgentRuntime:
    def test_set_agent_runtime_stores_reference(self):
        ch = _make_channel()
        runtime = MagicMock()
        ch.set_agent_runtime(runtime)
        assert ch._agent_runtime is runtime


# ---------------------------------------------------------------------------
# start() — slash command registration (lines 162-175)
# ---------------------------------------------------------------------------


class TestStart:
    @pytest.mark.asyncio
    async def test_start_registers_slash_commands_when_app_id_set(self):
        account = _make_account()
        account.application_id = "app-123"
        ch = _make_channel(account)

        mock_task = AsyncMock()

        with (
            patch.object(ch._gateway, "run", return_value=AsyncMock()),
            patch("asyncio.create_task", return_value=mock_task),
        ):
            ch._rest.register_slash_commands = MagicMock()
            await ch.start()

        ch._rest.register_slash_commands.assert_called_once()

    @pytest.mark.asyncio
    async def test_start_logs_warning_when_slash_registration_fails(self, caplog):
        import logging

        account = _make_account()
        account.application_id = "app-123"
        ch = _make_channel(account)

        ch._rest.register_slash_commands = MagicMock(side_effect=RuntimeError("403 Forbidden"))

        with (
            patch("asyncio.create_task", return_value=AsyncMock()),
            caplog.at_level(logging.WARNING, logger="missy.channels.discord.channel"),
        ):
            await ch.start()

        assert any("slash command registration" in r.message.lower() for r in caplog.records)

    @pytest.mark.asyncio
    async def test_start_skips_registration_when_no_app_id(self):
        account = _make_account()
        account.application_id = ""
        ch = _make_channel(account)
        ch._rest.register_slash_commands = MagicMock()

        with patch("asyncio.create_task", return_value=AsyncMock()):
            await ch.start()

        ch._rest.register_slash_commands.assert_not_called()


# ---------------------------------------------------------------------------
# stop() — task cancel (lines 179-186)
# ---------------------------------------------------------------------------


class TestStop:
    @pytest.mark.asyncio
    async def test_stop_cancels_gateway_task(self):
        ch = _make_channel()
        ch._gateway.disconnect = AsyncMock()

        mock_task = MagicMock()
        mock_task.cancel = MagicMock()
        mock_task.__await__ = lambda self: iter([])

        async def _cancelled():
            raise asyncio.CancelledError()

        ch._gateway_task = asyncio.ensure_future(_cancelled())
        await ch.stop()

        assert ch._gateway_task is None

    @pytest.mark.asyncio
    async def test_stop_with_no_task_is_noop(self):
        ch = _make_channel()
        ch._gateway.disconnect = AsyncMock()
        ch._gateway_task = None
        await ch.stop()  # should not raise


# ---------------------------------------------------------------------------
# receive() raises NotImplementedError (line 201)
# ---------------------------------------------------------------------------


class TestReceive:
    def test_receive_raises_not_implemented(self):
        ch = _make_channel()
        with pytest.raises(NotImplementedError):
            ch.receive()


# ---------------------------------------------------------------------------
# areceive() (line 213)
# ---------------------------------------------------------------------------


class TestAReceive:
    @pytest.mark.asyncio
    async def test_areceive_returns_queued_message(self):
        from missy.channels.base import ChannelMessage

        ch = _make_channel()
        msg = ChannelMessage(content="hi", sender="user-1", channel="discord")
        await ch._queue.put(msg)
        result = await ch.areceive()
        assert result is msg


# ---------------------------------------------------------------------------
# send() (lines 225-232)
# ---------------------------------------------------------------------------


class TestSend:
    def test_send_drops_when_no_channel_context(self, caplog):
        import logging

        ch = _make_channel()
        ch._current_channel_id = None
        with caplog.at_level(logging.WARNING, logger="missy.channels.discord.channel"):
            ch.send("test message")

        assert any("no current channel context" in r.message for r in caplog.records)

    def test_send_drops_when_no_event_loop(self, caplog):
        import logging

        ch = _make_channel()
        ch._current_channel_id = "chan-1"

        with (
            patch("asyncio.get_event_loop", side_effect=RuntimeError("no loop")),
            caplog.at_level(logging.WARNING, logger="missy.channels.discord.channel"),
        ):
            ch.send("test message")

        assert any("no running event loop" in r.message for r in caplog.records)


# ---------------------------------------------------------------------------
# _on_gateway_event routing (lines 417-433)
# ---------------------------------------------------------------------------


class TestOnGatewayEvent:
    @pytest.mark.asyncio
    async def test_guild_create_logs_debug(self, caplog):
        import logging

        ch = _make_channel()
        payload = {"t": "GUILD_CREATE", "d": {"id": "guild-777"}}
        with caplog.at_level(logging.DEBUG, logger="missy.channels.discord.channel"):
            await ch._on_gateway_event(payload)

    @pytest.mark.asyncio
    async def test_interaction_create_routes_to_handler(self):
        ch = _make_channel()
        ch._handle_interaction = AsyncMock()
        payload = {"t": "INTERACTION_CREATE", "d": {"id": "int-1", "token": "tok"}}
        await ch._on_gateway_event(payload)
        ch._handle_interaction.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_message_reaction_add_routes_to_handler(self):
        ch = _make_channel()
        ch._handle_reaction = AsyncMock()
        payload = {"t": "MESSAGE_REACTION_ADD", "d": {"message_id": "m-1"}}
        await ch._on_gateway_event(payload)
        ch._handle_reaction.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_bot_user_id_set_from_gateway_on_event(self):
        ch = _make_channel()
        ch._bot_user_id = None
        ch._gateway.bot_user_id = "bot-gateway-id"
        ch._handle_interaction = AsyncMock()
        payload = {"t": "INTERACTION_CREATE", "d": {}}
        await ch._on_gateway_event(payload)
        assert ch._bot_user_id == "bot-gateway-id"


# ---------------------------------------------------------------------------
# _handle_message — own-bot message dropped (line 467)
# ---------------------------------------------------------------------------


class TestHandleMessageOwnBot:
    @pytest.mark.asyncio
    async def test_own_bot_message_is_dropped(self):
        ch = _make_channel()
        ch._bot_user_id = "bot-001"
        data = _make_message(author_id="bot-001")
        await ch._handle_message(data)
        assert ch._queue.empty()


# ---------------------------------------------------------------------------
# _handle_message — bot author filtered (lines 532-547)
# ---------------------------------------------------------------------------


class TestHandleMessageBotFilter:
    @pytest.mark.asyncio
    async def test_bot_author_filtered_when_ignore_bots(self):
        ch = _make_channel(_make_account(ignore_bots=True))
        data = _make_message(author_id="other-bot", is_bot=True)
        await ch._handle_message(data)
        assert ch._queue.empty()


# ---------------------------------------------------------------------------
# _handle_message — attachment policy deny (lines 532-547)
# ---------------------------------------------------------------------------


class TestHandleMessageAttachment:
    @pytest.mark.asyncio
    async def test_message_with_attachment_is_denied(self):
        ch = _make_channel()
        data = _make_message(attachments=[{"id": "att-1", "filename": "file.exe"}])
        await ch._handle_message(data)
        assert ch._queue.empty()


# ---------------------------------------------------------------------------
# _check_dm_policy — PAIRING (line 725)
# ---------------------------------------------------------------------------


class TestCheckDMPolicyPairing:
    def test_pairing_policy_delegates_to_check_pairing(self):
        account = _make_account(dm_policy=DiscordDMPolicy.PAIRING, dm_allowlist=["user-paired"])
        ch = _make_channel(account)
        # user is already in the allowlist
        assert ch._check_dm_policy("user-paired", "hello") is True

    def test_pairing_policy_rejects_unpaired_user(self):
        account = _make_account(dm_policy=DiscordDMPolicy.PAIRING)
        ch = _make_channel(account)
        assert ch._check_dm_policy("stranger", "hello") is False


# ---------------------------------------------------------------------------
# _check_pairing — !pair deny (lines 781-784)
# ---------------------------------------------------------------------------


class TestCheckPairingDeny:
    def test_pair_deny_removes_from_pending(self):
        account = _make_account(dm_policy=DiscordDMPolicy.PAIRING)
        ch = _make_channel(account)
        ch._pending_pairs.add("user-pending")

        result = ch._check_pairing("admin", "!pair deny user-pending")

        assert "user-pending" not in ch._pending_pairs
        assert result is False


# ---------------------------------------------------------------------------
# _check_guild_policy — channel allowlist deny (lines 826-851)
# ---------------------------------------------------------------------------


class TestCheckGuildPolicyAllowlist:
    def test_channel_not_in_allowlist_denied(self):
        policy = DiscordGuildPolicy(
            enabled=True,
            require_mention=False,
            allowed_channels=["allowed-chan"],
            allowed_users=[],
        )
        account = _make_account(guild_policies={"guild-1": policy})
        ch = _make_channel(account)
        data = {"channel": {"name": "other-chan"}}
        result = ch._check_guild_policy("guild-1", "not-allowed-chan", "user-1", "hello", data)
        assert result is False

    def test_user_not_in_allowlist_denied(self):
        policy = DiscordGuildPolicy(
            enabled=True,
            require_mention=False,
            allowed_channels=[],
            allowed_users=["user-allowed"],
        )
        account = _make_account(guild_policies={"guild-1": policy})
        ch = _make_channel(account)
        result = ch._check_guild_policy("guild-1", "chan-1", "user-not-allowed", "hello", {})
        assert result is False

    def test_guild_disabled_is_denied(self):
        policy = DiscordGuildPolicy(enabled=False)
        account = _make_account(guild_policies={"guild-1": policy})
        ch = _make_channel(account)
        result = ch._check_guild_policy("guild-1", "chan-1", "user-1", "hello", {})
        assert result is False


# ---------------------------------------------------------------------------
# _check_guild_policy — require_mention, no own_id fallback (lines 869-870)
# ---------------------------------------------------------------------------


class TestCheckGuildPolicyRequireMentionFallback:
    def test_require_mention_no_own_id_uses_mentions_list(self):
        """When no bot user ID is known, fall back to checking the mentions list."""
        policy = DiscordGuildPolicy(enabled=True, require_mention=True)
        account = _make_account(account_id=None, guild_policies={"guild-1": policy})
        ch = _make_channel(account)
        ch._bot_user_id = None  # force fallback path

        # Message does not mention the bot → denied
        data: dict[str, Any] = {"mentions": []}
        result = ch._check_guild_policy("guild-1", "chan-1", "user-1", "hello", data)
        assert result is False


# ---------------------------------------------------------------------------
# _handle_interaction (lines 663-693)
# ---------------------------------------------------------------------------


class TestHandleInteraction:
    @pytest.mark.asyncio
    async def test_handle_interaction_calls_slash_handler(self):
        import sys

        ch = _make_channel()

        mock_http = MagicMock()
        mock_http.post = MagicMock()

        mock_commands_module = MagicMock()
        mock_commands_module.handle_slash_command = AsyncMock(return_value="ok reply")

        mock_gateway_module = MagicMock()
        mock_gateway_module.create_client = MagicMock(return_value=mock_http)

        with (
            patch.dict(sys.modules, {"missy.channels.discord.commands": mock_commands_module}),
            patch.dict(sys.modules, {"missy.gateway.client": mock_gateway_module}),
        ):
            data = {
                "id": "int-1",
                "token": "tok-abc",
                "channel_id": "chan-1",
                "data": {"name": "help", "options": []},
            }
            await ch._handle_interaction(data)

        assert ch._current_channel_id == "chan-1"

    @pytest.mark.asyncio
    async def test_handle_interaction_logs_on_http_error(self, caplog):
        import logging
        import sys

        ch = _make_channel()

        mock_commands_module = MagicMock()
        mock_commands_module.handle_slash_command = AsyncMock(return_value="resp")

        mock_gateway_module = MagicMock()
        mock_gateway_module.create_client = MagicMock(side_effect=RuntimeError("network error"))

        with (
            patch.dict(sys.modules, {"missy.channels.discord.commands": mock_commands_module}),
            patch.dict(sys.modules, {"missy.gateway.client": mock_gateway_module}),
            caplog.at_level(logging.ERROR, logger="missy.channels.discord.channel"),
        ):
            data = {"id": "int-1", "token": "tok", "channel_id": "chan-1", "data": {"name": "help"}}
            await ch._handle_interaction(data)

        assert any("interaction response failed" in r.message for r in caplog.records)


# ---------------------------------------------------------------------------
# _handle_reaction — reject path (line 1071) and exception path (lines 1080-1082)
# ---------------------------------------------------------------------------


class TestHandleReaction:
    @pytest.mark.asyncio
    async def test_reaction_reject_sends_rejection_message(self):
        import sys

        ch = _make_channel()
        ch._bot_user_id = "bot-001"
        ch._pending_evolutions["msg-99"] = "proposal-abc"

        mock_mgr = MagicMock()
        mock_mgr.reject.return_value = True
        ch._rest.send_message = MagicMock()

        mock_evo_module = MagicMock()
        mock_evo_module.CodeEvolutionManager = MagicMock(return_value=mock_mgr)

        with patch.dict(sys.modules, {"missy.agent.code_evolution": mock_evo_module}):
            data = {
                "message_id": "msg-99",
                "user_id": "user-2",
                "channel_id": "chan-1",
                "emoji": {"name": "\u274c"},
            }
            await ch._handle_reaction(data)

        mock_mgr.reject.assert_called_once_with("proposal-abc")
        ch._rest.send_message.assert_called()

    @pytest.mark.asyncio
    async def test_reaction_reject_failed_approve_sends_could_not_message(self):
        import sys

        ch = _make_channel()
        ch._bot_user_id = "bot-001"
        ch._pending_evolutions["msg-99"] = "proposal-abc"

        mock_mgr = MagicMock()
        mock_mgr.reject.return_value = False  # already resolved
        ch._rest.send_message = MagicMock()

        mock_evo_module = MagicMock()
        mock_evo_module.CodeEvolutionManager = MagicMock(return_value=mock_mgr)

        with patch.dict(sys.modules, {"missy.agent.code_evolution": mock_evo_module}):
            data = {
                "message_id": "msg-99",
                "user_id": "user-2",
                "channel_id": "chan-1",
                "emoji": {"name": "\u274c"},
            }
            await ch._handle_reaction(data)

        ch._rest.send_message.assert_called()
        msg_arg = ch._rest.send_message.call_args[0][1]
        assert "Could not reject" in msg_arg

    @pytest.mark.asyncio
    async def test_reaction_evolution_exception_sends_error_message(self):
        """Lines 1080-1082: exception in CodeEvolutionManager → error message sent."""
        import sys

        ch = _make_channel()
        ch._bot_user_id = "bot-001"
        ch._pending_evolutions["msg-99"] = "proposal-abc"
        ch._rest.send_message = MagicMock()

        mock_evo_module = MagicMock()
        mock_evo_module.CodeEvolutionManager = MagicMock(side_effect=RuntimeError("db locked"))

        with patch.dict(sys.modules, {"missy.agent.code_evolution": mock_evo_module}):
            data = {
                "message_id": "msg-99",
                "user_id": "user-2",
                "channel_id": "chan-1",
                "emoji": {"name": "\u2705"},
            }
            await ch._handle_reaction(data)

        ch._rest.send_message.assert_called()
        msg_arg = ch._rest.send_message.call_args[0][1]
        assert "Error" in msg_arg or "db locked" in msg_arg

    @pytest.mark.asyncio
    async def test_reaction_from_own_bot_ignored(self):
        ch = _make_channel()
        ch._bot_user_id = "bot-001"
        ch._pending_evolutions["msg-99"] = "proposal-abc"
        ch._rest.send_message = MagicMock()

        data = {
            "message_id": "msg-99",
            "user_id": "bot-001",
            "channel_id": "chan-1",
            "emoji": {"name": "\u2705"},
        }
        await ch._handle_reaction(data)
        ch._rest.send_message.assert_not_called()

    @pytest.mark.asyncio
    async def test_reaction_on_untracked_message_ignored(self):
        ch = _make_channel()
        ch._bot_user_id = "bot-001"
        ch._rest.send_message = MagicMock()

        data = {
            "message_id": "untracked-msg",
            "user_id": "user-2",
            "channel_id": "chan-1",
            "emoji": {"name": "\u2705"},
        }
        await ch._handle_reaction(data)
        ch._rest.send_message.assert_not_called()

    @pytest.mark.asyncio
    async def test_unrelated_emoji_ignored(self):
        ch = _make_channel()
        ch._bot_user_id = "bot-001"
        ch._pending_evolutions["msg-99"] = "proposal-abc"
        ch._rest.send_message = MagicMock()

        data = {
            "message_id": "msg-99",
            "user_id": "user-2",
            "channel_id": "chan-1",
            "emoji": {"name": "\U0001f44d"},  # thumbs up — not tracked
        }
        await ch._handle_reaction(data)
        ch._rest.send_message.assert_not_called()


# ---------------------------------------------------------------------------
# _emit_audit — exception swallowed (lines 1108-1109)
# ---------------------------------------------------------------------------


class TestEmitAuditExceptionSwallowed:
    def test_emit_audit_exception_is_swallowed(self):
        """Lines 1108-1109: exception inside _emit_audit is caught; no raise."""
        ch = _make_channel()

        with patch("missy.channels.discord.channel.event_bus") as mock_bus:
            mock_bus.publish.side_effect = RuntimeError("bus error")
            # Should not raise
            ch._emit_audit("test.event", "allow", {})

    def test_emit_audit_publishes_to_event_bus(self):
        ch = _make_channel()

        from missy.core.events import event_bus as real_bus

        with patch("missy.channels.discord.channel.event_bus") as mock_bus:
            mock_bus.publish = MagicMock()
            ch._emit_audit("discord.test", "allow", {"key": "val"})
            mock_bus.publish.assert_called_once()
            event_arg = mock_bus.publish.call_args[0][0]
            assert event_arg.event_type == "discord.test"
            assert event_arg.result == "allow"
