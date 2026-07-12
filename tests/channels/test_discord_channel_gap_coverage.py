"""Gap coverage tests for missy/channels/discord/channel.py.

Targets remaining uncovered lines after existing coverage files:
  227          : send() — happy path: loop.create_task called
  431          : _on_gateway_event — bot_user_id synced from gateway when not None
  474          : _handle_message — voice command handled, early return
  489-490      : _handle_message — delete_message raises, warning logged
  624-668      : _maybe_handle_voice_command — various sub-paths
  732          : _is_bot_allowed — allow_bots_if_mention_only with <@!id> mention
  764          : _check_guild_policy — mention requirement satisfied (returns True)
  878-879      : _check_guild_policy — require_mention mentioned=True path (GUILD_CREATE emit)
"""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from missy.channels.discord import voice_binding
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
) -> dict[str, Any]:
    return {
        "id": message_id,
        "channel_id": channel_id,
        "guild_id": guild_id,
        "content": content,
        "author": {"id": author_id, "username": "TestUser", "bot": is_bot},
        "attachments": [],
    }


# ---------------------------------------------------------------------------
# send() — happy path: loop.create_task is called  (line 227)
# ---------------------------------------------------------------------------


class TestSendHappyPath:
    def test_send_schedules_task_when_loop_available(self):
        """Line 227: loop.create_task(send_with_retry(...)) is called."""
        ch = _make_channel()
        ch._current_channel_id = "chan-42"

        mock_loop = MagicMock()
        mock_loop.create_task = MagicMock()

        with patch("asyncio.get_running_loop", return_value=mock_loop):
            ch.send("hello discord")

        mock_loop.create_task.assert_called_once()


# ---------------------------------------------------------------------------
# _on_gateway_event — bot_user_id set from gateway when previously None  (line 431)
# ---------------------------------------------------------------------------


class TestOnGatewayEventBotUserIdSync:
    @pytest.mark.asyncio
    async def test_bot_user_id_synced_from_gateway_on_message_create(self):
        """Line 431: _bot_user_id is None → synced from gateway.bot_user_id."""
        ch = _make_channel()
        ch._bot_user_id = None
        ch._gateway.bot_user_id = "gw-bot-id"
        ch._handle_message = AsyncMock()

        payload = {"t": "MESSAGE_CREATE", "d": {"id": "m-1", "content": "hi"}}
        await ch._on_gateway_event(payload)

        assert ch._bot_user_id == "gw-bot-id"

    @pytest.mark.asyncio
    async def test_bot_user_id_not_overwritten_when_already_set(self):
        """Line 431: When _bot_user_id is already set, gateway value is not applied."""
        ch = _make_channel()
        ch._bot_user_id = "existing-id"
        ch._gateway.bot_user_id = "different-gw-id"
        ch._handle_message = AsyncMock()

        payload = {"t": "MESSAGE_CREATE", "d": {}}
        await ch._on_gateway_event(payload)

        assert ch._bot_user_id == "existing-id"

    @pytest.mark.asyncio
    async def test_guild_create_event_is_handled(self, caplog):
        """Lines 436-437: GUILD_CREATE logs debug and doesn't raise."""
        import logging

        ch = _make_channel()
        payload = {"t": "GUILD_CREATE", "d": {"id": "guild-xyz"}}

        with caplog.at_level(logging.DEBUG, logger="missy.channels.discord.channel"):
            await ch._on_gateway_event(payload)

        # No exception raised; optional debug log may or may not appear
        # depending on log level — just verify no error is raised.


# ---------------------------------------------------------------------------
# _handle_message — voice command handled, early return  (line 474)
# ---------------------------------------------------------------------------


class TestHandleMessageVoiceCommandEarlyReturn:
    @pytest.mark.asyncio
    async def test_voice_command_handled_means_message_not_queued(self):
        """When _maybe_handle_voice_command returns True, message is dropped.

        SR-1.13: voice-command dispatch only runs after guild policy
        authorization, so this test must authorize "guild-1" for the
        dispatch to be reached at all.
        """
        account = _make_account(guild_policies={"guild-1": DiscordGuildPolicy(enabled=True)})
        ch = _make_channel(account)
        ch._bot_user_id = "other-bot"  # not own message
        ch._maybe_handle_voice_command = AsyncMock(return_value=True)

        data = _make_message(
            author_id="user-1",
            content="join the general voice channel",
            guild_id="guild-1",
        )
        await ch._handle_message(data)

        assert ch._queue.empty()
        ch._maybe_handle_voice_command.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_voice_command_not_dispatched_when_guild_unauthorized(self):
        """SR-1.13: an unauthorized guild must never reach voice-command
        dispatch -- the pre-fix ordering let any guild message trigger
        real side effects (joining voice, etc.) before authorization."""
        ch = _make_channel()  # default account has no guild_policies at all
        ch._bot_user_id = "other-bot"
        ch._maybe_handle_voice_command = AsyncMock(return_value=True)

        data = _make_message(
            author_id="user-1",
            content="join the general voice channel",
            guild_id="unauthorized-guild",
        )
        await ch._handle_message(data)

        ch._maybe_handle_voice_command.assert_not_awaited()


# ---------------------------------------------------------------------------
# SR-1.13: uniform Discord ingress authorization.
#
# _handle_message() previously dispatched voice/image/screencast commands
# BEFORE _check_dm_policy()/_check_guild_policy() ran (explicit comment:
# "handled before policy gates"), so any message in any guild channel
# (ignoring allowed_channels/allowed_roles/allowed_users/require_mention)
# or any DM (ignoring dm_policy DISABLED/ALLOWLIST/PAIRING) could join a
# voice channel, capture/analyze a screenshot, or start a screen share
# with real side effects before authorization was ever checked.
# ---------------------------------------------------------------------------


class TestUniformIngressAuthorizationSR113:
    @pytest.mark.asyncio
    async def test_image_command_not_dispatched_when_guild_unauthorized(self):
        ch = _make_channel()  # no guild_policies configured -> default deny
        ch._bot_user_id = "other-bot"
        ch._maybe_handle_image_command = AsyncMock(return_value=True)

        data = _make_message(
            author_id="user-1",
            content="!analyze",
            guild_id="unauthorized-guild",
        )
        await ch._handle_message(data)

        ch._maybe_handle_image_command.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_image_command_dispatched_when_guild_authorized(self):
        account = _make_account(guild_policies={"guild-1": DiscordGuildPolicy(enabled=True)})
        ch = _make_channel(account)
        ch._bot_user_id = "other-bot"
        ch._maybe_handle_image_command = AsyncMock(return_value=True)

        data = _make_message(author_id="user-1", content="!analyze", guild_id="guild-1")
        await ch._handle_message(data)

        ch._maybe_handle_image_command.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_screencast_command_not_dispatched_when_guild_unauthorized(self):
        ch = _make_channel()
        ch._bot_user_id = "other-bot"
        ch._maybe_handle_screen_command = AsyncMock(return_value=True)

        data = _make_message(
            author_id="user-1",
            content="!screen share",
            guild_id="unauthorized-guild",
        )
        await ch._handle_message(data)

        ch._maybe_handle_screen_command.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_screencast_command_dispatched_when_guild_authorized(self):
        account = _make_account(guild_policies={"guild-1": DiscordGuildPolicy(enabled=True)})
        ch = _make_channel(account)
        ch._bot_user_id = "other-bot"
        ch._maybe_handle_screen_command = AsyncMock(return_value=True)

        data = _make_message(author_id="user-1", content="!screen share", guild_id="guild-1")
        await ch._handle_message(data)

        ch._maybe_handle_screen_command.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_image_command_not_dispatched_for_channel_not_in_allowlist(self):
        # Guild is enabled overall but this specific channel isn't allowed --
        # a real allowed_channels policy must still gate special commands.
        account = _make_account(
            guild_policies={
                "guild-1": DiscordGuildPolicy(enabled=True, allowed_channels=["other-channel"])
            }
        )
        ch = _make_channel(account)
        ch._bot_user_id = "other-bot"
        ch._maybe_handle_image_command = AsyncMock(return_value=True)

        data = _make_message(
            author_id="user-1",
            content="!analyze",
            guild_id="guild-1",
            channel_id="chan-1",
        )
        await ch._handle_message(data)

        ch._maybe_handle_image_command.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_image_command_not_dispatched_for_user_not_in_allowlist(self):
        account = _make_account(
            guild_policies={
                "guild-1": DiscordGuildPolicy(enabled=True, allowed_users=["someone-else"])
            }
        )
        ch = _make_channel(account)
        ch._bot_user_id = "other-bot"
        ch._maybe_handle_image_command = AsyncMock(return_value=True)

        data = _make_message(author_id="user-1", content="!analyze", guild_id="guild-1")
        await ch._handle_message(data)

        ch._maybe_handle_image_command.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_image_command_not_dispatched_when_dm_policy_disabled(self):
        account = _make_account(dm_policy=DiscordDMPolicy.DISABLED)
        ch = _make_channel(account)
        ch._bot_user_id = "other-bot"
        ch._maybe_handle_image_command = AsyncMock(return_value=True)

        data = _make_message(author_id="user-1", content="!analyze", guild_id=None)
        await ch._handle_message(data)

        ch._maybe_handle_image_command.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_screencast_command_not_dispatched_when_dm_unpaired(self):
        # PAIRING policy: an unpaired DM sender must not reach screencast
        # dispatch (screen sharing has real side effects on the host).
        account = _make_account(dm_policy=DiscordDMPolicy.PAIRING)
        ch = _make_channel(account)
        ch._bot_user_id = "other-bot"
        ch._maybe_handle_screen_command = AsyncMock(return_value=True)

        data = _make_message(author_id="unpaired-user", content="!screen share", guild_id=None)
        await ch._handle_message(data)

        ch._maybe_handle_screen_command.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_authorization_runs_before_voice_image_and_screencast_all_three(self):
        # A single unauthorized message must not reach ANY of the three
        # special-command dispatchers.
        ch = _make_channel()
        ch._bot_user_id = "other-bot"
        ch._maybe_handle_voice_command = AsyncMock(return_value=False)
        ch._maybe_handle_image_command = AsyncMock(return_value=False)
        ch._maybe_handle_screen_command = AsyncMock(return_value=False)

        data = _make_message(
            author_id="user-1",
            content="join voice and !analyze and !screen share",
            guild_id="unauthorized-guild",
        )
        await ch._handle_message(data)

        ch._maybe_handle_voice_command.assert_not_awaited()
        ch._maybe_handle_image_command.assert_not_awaited()
        ch._maybe_handle_screen_command.assert_not_awaited()


# ---------------------------------------------------------------------------
# _handle_message — delete_message raises exception  (lines 489-490)
# ---------------------------------------------------------------------------


class TestHandleMessageDeleteRaisesWarning:
    @pytest.mark.asyncio
    @pytest.mark.filterwarnings("ignore::RuntimeWarning")
    async def test_delete_message_exception_logged_as_warning(self, caplog):
        """Lines 489-490: delete_message raises → warning is logged, processing continues."""
        import logging

        ch = _make_channel()
        ch._bot_user_id = "other-bot"
        ch._rest.delete_message = MagicMock(side_effect=RuntimeError("403 Forbidden"))

        # Provide content that will trigger the secrets detector.
        # We patch the detector to always return True for simplicity.
        with patch("missy.security.secrets.SecretsDetector.has_secrets", return_value=True):
            data = _make_message(
                author_id="user-1",
                content="AKIA1234567890ABCDEF",  # fake AWS key pattern
                guild_id=None,
                message_id="msg-del",
            )
            with caplog.at_level(logging.WARNING, logger="missy.channels.discord.channel"):
                await ch._handle_message(data)

        assert any("delete" in r.message.lower() for r in caplog.records)


# ---------------------------------------------------------------------------
# _maybe_handle_voice_command — various sub-paths  (lines 624-668)
# ---------------------------------------------------------------------------


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
class TestMaybeHandleVoiceCommand:
    @pytest.mark.asyncio
    async def test_regular_message_returns_false(self):
        """Non-voice content returns False immediately."""
        ch = _make_channel()
        result = await ch._maybe_handle_voice_command(
            guild_id="g1", channel_id="c1", author_id="u1", content="hello world"
        )
        assert result is False

    @pytest.mark.asyncio
    async def test_unknown_voice_command_returns_false(self):
        """Unrecognized voice-like content returns False."""
        ch = _make_channel()
        result = await ch._maybe_handle_voice_command(
            guild_id="g1", channel_id="c1", author_id="u1", content="start the disco lights"
        )
        assert result is False

    @pytest.mark.asyncio
    async def test_mention_prefix_stripped_before_check(self):
        """Lines 624-626: leading <@mention> is stripped before parsing."""
        ch = _make_channel()
        result = await ch._maybe_handle_voice_command(
            guild_id="g1", channel_id="c1", author_id="u1", content="<@123> hello world"
        )
        assert result is False

    @pytest.mark.asyncio
    async def test_voice_command_with_existing_voice_manager(self):
        """Lines 658-668: voice manager already created, dispatches to maybe_handle_voice_command."""
        ch = _make_channel()
        ch._voice = MagicMock()  # already initialized

        mock_result = MagicMock()
        mock_result.handled = True
        mock_result.reply = "Joined channel!"

        ch._rest.send_message = MagicMock()

        with patch(
            "missy.channels.discord.voice_commands.maybe_handle_voice_command",
            AsyncMock(return_value=mock_result),
        ):
            result = await ch._maybe_handle_voice_command(
                guild_id="g1",
                channel_id="c1",
                author_id="u1",
                content="join the general voice channel",
            )

        assert result is True
        ch._rest.send_message.assert_called_once()

    @pytest.mark.asyncio
    async def test_voice_command_no_reply_does_not_send_message(self):
        """Lines 665-667: handled=True but reply is None → send_message not called."""
        ch = _make_channel()
        ch._voice = MagicMock()

        mock_result = MagicMock()
        mock_result.handled = True
        mock_result.reply = None

        ch._rest.send_message = MagicMock()

        with patch(
            "missy.channels.discord.voice_commands.maybe_handle_voice_command",
            AsyncMock(return_value=mock_result),
        ):
            result = await ch._maybe_handle_voice_command(
                guild_id="g1",
                channel_id="c1",
                author_id="u1",
                content="join the general voice channel",
            )

        assert result is True
        ch._rest.send_message.assert_not_called()

    @pytest.mark.asyncio
    async def test_voice_manager_init_failure_returns_true_and_sends_error(self):
        """Lines 647-651: DiscordVoiceManager init/start raises → error sent, returns True."""
        ch = _make_channel()
        ch._voice = None
        voice_binding.set_voice_binding(
            MagicMock(),
            asyncio.get_running_loop(),
            account_id=ch._voice_account_id(),
            guild_id="g1",
        )

        ch._rest.send_message = MagicMock()

        with patch(
            "missy.channels.discord.voice.DiscordVoiceManager",
            side_effect=RuntimeError("voice unavailable"),
        ):
            result = await ch._maybe_handle_voice_command(
                guild_id="g1",
                channel_id="c1",
                author_id="u1",
                content="join the general voice channel",
            )

        assert result is True
        ch._rest.send_message.assert_called_once()
        assert "Voice unavailable" in ch._rest.send_message.call_args[0][1]
        assert voice_binding.get_voice_binding(guild_id="g1") is None

    @pytest.mark.asyncio
    async def test_voice_command_with_agent_runtime_creates_callback(self):
        """Lines 636-645: agent_runtime is set → async callback is created."""
        ch = _make_channel()
        ch._voice = None
        ch._agent_runtime = MagicMock()
        ch._agent_runtime.run = MagicMock(return_value="response")

        mock_vm = MagicMock()
        mock_vm.start = AsyncMock()

        mock_result = MagicMock()
        mock_result.handled = True
        mock_result.reply = None

        ch._rest.send_message = MagicMock()

        with (
            patch("missy.channels.discord.voice.DiscordVoiceManager", return_value=mock_vm),
            patch(
                "missy.channels.discord.voice_commands.maybe_handle_voice_command",
                AsyncMock(return_value=mock_result),
            ),
        ):
            result = await ch._maybe_handle_voice_command(
                guild_id="g1",
                channel_id="c1",
                author_id="u1",
                content="join the general voice channel",
            )

        assert result is True
        binding = voice_binding.get_voice_binding(account_id=ch._voice_account_id(), guild_id="g1")
        assert binding is not None
        assert binding.manager is mock_vm
        assert binding.account_id == "bot-001"
        assert binding.guild_id == "g1"
        voice_binding.clear_voice_binding()

    @pytest.mark.asyncio
    async def test_voice_agent_callback_blocks_secrets_before_forwarding(self):
        """Regression: typed-text messages run through SecretsDetector
        before ever reaching the agent ("1b. Credential / secrets
        detection" in _handle_message), deleting the message and
        blocking it entirely. AgentRuntime.run() itself only applies
        InputSanitizer (prompt-injection detection), never
        SecretsDetector -- so the real _voice_agent_cb built here fed a
        voice transcript straight into _rt.run() with no check of its
        own, bypassing secrets screening completely for voice input
        (unlike identical text typed in chat).
        """
        ch = _make_channel()
        ch._voice = None
        ch._agent_runtime = MagicMock()
        ch._agent_runtime.run = MagicMock(return_value="response")

        mock_vm = MagicMock()
        mock_vm.start = AsyncMock()

        mock_result = MagicMock()
        mock_result.handled = True
        mock_result.reply = None

        ch._rest.send_message = MagicMock()

        captured_kwargs: dict = {}

        def _capture_voice_manager(**kwargs):
            captured_kwargs.update(kwargs)
            return mock_vm

        with (
            patch(
                "missy.channels.discord.voice.DiscordVoiceManager",
                side_effect=_capture_voice_manager,
            ),
            patch(
                "missy.channels.discord.voice_commands.maybe_handle_voice_command",
                AsyncMock(return_value=mock_result),
            ),
        ):
            await ch._maybe_handle_voice_command(
                guild_id="g1",
                channel_id="c1",
                author_id="u1",
                content="join the general voice channel",
            )

        voice_binding.clear_voice_binding()

        agent_callback = captured_kwargs["agent_callback"]
        result = await agent_callback("My AWS key is AKIAIOSFODNN7EXAMPLE123456", "sess-1")

        ch._agent_runtime.run.assert_not_called()
        assert "credential" in result.lower() or "secret" in result.lower()

    @pytest.mark.asyncio
    async def test_voice_agent_callback_forwards_normal_prompt(self):
        ch = _make_channel()
        ch._voice = None
        ch._agent_runtime = MagicMock()
        ch._agent_runtime.run = MagicMock(return_value="It is 3pm.")

        mock_vm = MagicMock()
        mock_vm.start = AsyncMock()

        mock_result = MagicMock()
        mock_result.handled = True
        mock_result.reply = None

        ch._rest.send_message = MagicMock()

        captured_kwargs: dict = {}

        def _capture_voice_manager(**kwargs):
            captured_kwargs.update(kwargs)
            return mock_vm

        with (
            patch(
                "missy.channels.discord.voice.DiscordVoiceManager",
                side_effect=_capture_voice_manager,
            ),
            patch(
                "missy.channels.discord.voice_commands.maybe_handle_voice_command",
                AsyncMock(return_value=mock_result),
            ),
        ):
            await ch._maybe_handle_voice_command(
                guild_id="g1",
                channel_id="c1",
                author_id="u1",
                content="join the general voice channel",
            )

        voice_binding.clear_voice_binding()

        agent_callback = captured_kwargs["agent_callback"]
        result = await agent_callback("What time is it?", "sess-1")

        ch._agent_runtime.run.assert_called_once()
        assert result == "It is 3pm."

    @pytest.mark.asyncio
    async def test_existing_voice_manager_registers_new_guild_scope(self):
        ch = _make_channel()
        ch._voice = MagicMock()

        mock_result = MagicMock()
        mock_result.handled = True
        mock_result.reply = None

        with patch(
            "missy.channels.discord.voice_commands.maybe_handle_voice_command",
            AsyncMock(return_value=mock_result),
        ):
            result = await ch._maybe_handle_voice_command(
                guild_id="g2",
                channel_id="c1",
                author_id="u1",
                content="join the general voice channel",
            )

        assert result is True
        binding = voice_binding.get_voice_binding(account_id=ch._voice_account_id(), guild_id="g2")
        assert binding is not None
        assert binding.manager is ch._voice
        voice_binding.clear_voice_binding()

    @pytest.mark.asyncio
    async def test_stop_stops_voice_manager_and_clears_binding(self):
        ch = _make_channel()
        mock_voice = MagicMock()
        mock_voice.stop = AsyncMock()
        ch._voice = mock_voice
        ch._gateway.disconnect = AsyncMock()
        voice_binding.set_voice_binding(
            mock_voice,
            asyncio.get_running_loop(),
            account_id=ch._voice_account_id(),
            guild_id="g1",
        )

        await ch.stop()

        mock_voice.stop.assert_awaited_once_with()
        assert ch._voice is None
        assert voice_binding.get_voice_binding() is None


# ---------------------------------------------------------------------------
# _allow_bot_author — allow_bots_if_mention_only with <@!id> mention  (line 732)
# ---------------------------------------------------------------------------


class TestAllowBotAuthorAlternateMentionFormat:
    def test_alt_mention_format_allows_bot(self):
        """Line 732: <@!own_id> mention format also allows the bot through."""
        account = _make_account(
            account_id="bot-777",
            ignore_bots=True,
            allow_bots_if_mention_only=True,
        )
        ch = _make_channel(account)
        ch._bot_user_id = "bot-777"

        author = {"id": "other-bot", "bot": True}
        result = ch._allow_bot_author(author=author, content="<@!bot-777> hello", guild_id=None)
        assert result is True

    def test_standard_mention_format_allows_bot(self):
        """Line 730: <@own_id> mention format allows the bot through."""
        account = _make_account(
            account_id="bot-777",
            ignore_bots=True,
            allow_bots_if_mention_only=True,
        )
        ch = _make_channel(account)
        ch._bot_user_id = "bot-777"

        author = {"id": "other-bot", "bot": True}
        result = ch._allow_bot_author(author=author, content="<@bot-777> hello", guild_id=None)
        assert result is True

    def test_no_mention_returns_false(self):
        """allow_bots_if_mention_only but no mention → False."""
        account = _make_account(
            account_id="bot-777",
            ignore_bots=True,
            allow_bots_if_mention_only=True,
        )
        ch = _make_channel(account)
        ch._bot_user_id = "bot-777"

        author = {"id": "other-bot", "bot": True}
        result = ch._allow_bot_author(author=author, content="hello without mention", guild_id=None)
        assert result is False


# ---------------------------------------------------------------------------
# _check_guild_policy — mention requirement satisfied (returns True)  (line 764)
# and GUILD_CREATE / require_mention True path  (lines 878-879)
# ---------------------------------------------------------------------------


class TestCheckGuildPolicyMentionSatisfied:
    def test_require_mention_satisfied_returns_true(self):
        """Line 764: content contains mention → mentioned=True → returns True."""
        policy = DiscordGuildPolicy(
            enabled=True,
            require_mention=True,
            allowed_channels=[],
            allowed_users=[],
        )
        account = _make_account(account_id="bot-888", guild_policies={"guild-1": policy})
        ch = _make_channel(account)
        ch._bot_user_id = "bot-888"

        content = "<@bot-888> please help"
        result = ch._check_guild_policy("guild-1", "chan-1", "user-1", content, {})
        assert result is True

    def test_require_mention_alt_format_satisfied_returns_true(self):
        """Lines 878-879 area: <@!id> mention format also passes require_mention."""
        policy = DiscordGuildPolicy(
            enabled=True,
            require_mention=True,
            allowed_channels=[],
            allowed_users=[],
        )
        account = _make_account(account_id="bot-888", guild_policies={"guild-1": policy})
        ch = _make_channel(account)
        ch._bot_user_id = "bot-888"

        content = "<@!bot-888> help me"
        result = ch._check_guild_policy("guild-1", "chan-1", "user-1", content, {})
        assert result is True

    def test_require_mention_no_own_id_with_populated_mentions_list(self):
        """Lines 878-879: own_id is None, mentions list is checked but fails."""
        policy = DiscordGuildPolicy(
            enabled=True,
            require_mention=True,
            allowed_channels=[],
            allowed_users=[],
        )
        account = _make_account(account_id=None, guild_policies={"guild-1": policy})
        ch = _make_channel(account)
        ch._bot_user_id = None

        # Mentions list contains some other user, not the bot.
        data: dict[str, Any] = {"mentions": [{"id": "other-user"}]}
        result = ch._check_guild_policy("guild-1", "chan-1", "user-1", "hello", data)
        assert result is False

    def test_require_mention_no_own_id_empty_mentions_list(self):
        """Lines 878-879: own_id is None, empty mentions list → mentioned=False."""
        policy = DiscordGuildPolicy(
            enabled=True,
            require_mention=True,
            allowed_channels=[],
            allowed_users=[],
        )
        account = _make_account(account_id=None, guild_policies={"guild-1": policy})
        ch = _make_channel(account)
        ch._bot_user_id = None

        # Empty mentions list → mentioned=False → denied.
        data: dict[str, Any] = {"mentions": []}
        result = ch._check_guild_policy("guild-1", "chan-1", "user-1", "hello there", data)
        assert result is False
