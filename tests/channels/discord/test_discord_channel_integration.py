"""Integration tests for DiscordChannel — construction, properties, and access control.

Covers methods not exercised by other test files:

- DiscordChannel construction with full and minimal configs
- set_screencast / set_agent_runtime setter methods
- bot_user_id property resolution (from _bot_user_id and gateway fallback)
- _is_own_message helper
- _allow_bot_author helper (ignore_bots / allow_bots_if_mention_only)
- _check_dm_policy (DISABLED / OPEN / ALLOWLIST / PAIRING)
- _check_pairing (initiate, accept, deny, regular message)
- _check_guild_policy (no policy, disabled, channel allowlist, user allowlist,
  mention requirement)
- accept_pair / deny_pair / get_pending_pairs management
- get_thread_session / set_thread_session helpers
- send() with no channel context (dropped) and with context (fire-and-forget)
- areceive() dequeues correctly
- receive() raises NotImplementedError

All Discord Gateway and REST I/O is mocked so no network is needed.
"""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from missy.channels.discord.channel import DiscordChannel, DiscordSendError
from missy.channels.discord.config import (
    DiscordAccountConfig,
    DiscordDMPolicy,
    DiscordGuildPolicy,
)

# ---------------------------------------------------------------------------
# Helpers / shared fixtures
# ---------------------------------------------------------------------------


def _make_account(
    *,
    token: str = "test-token",
    dm_policy: DiscordDMPolicy = DiscordDMPolicy.DISABLED,
    dm_allowlist: list[str] | None = None,
    ignore_bots: bool = True,
    allow_bots_if_mention_only: bool = False,
    account_id: str | None = None,
    guild_policies: dict[str, DiscordGuildPolicy] | None = None,
    auto_thread_threshold: int = 0,
    application_id: str = "",
) -> DiscordAccountConfig:
    return DiscordAccountConfig(
        token=token,
        dm_policy=dm_policy,
        dm_allowlist=dm_allowlist or [],
        ignore_bots=ignore_bots,
        allow_bots_if_mention_only=allow_bots_if_mention_only,
        account_id=account_id,
        guild_policies=guild_policies or {},
        auto_thread_threshold=auto_thread_threshold,
        application_id=application_id,
    )


@pytest.fixture
def account_cfg() -> DiscordAccountConfig:
    return _make_account()


@pytest.fixture
def mock_gateway() -> MagicMock:
    gw = MagicMock()
    gw.bot_user_id = None
    gw.run = AsyncMock()
    gw.disconnect = AsyncMock()
    return gw


@pytest.fixture
def mock_rest() -> MagicMock:
    rest = MagicMock()
    rest.send_message = MagicMock(return_value={"id": "msg-001"})
    rest.trigger_typing = MagicMock()
    rest.register_slash_commands = MagicMock()
    return rest


@pytest.fixture
def channel(account_cfg, mock_gateway, mock_rest) -> DiscordChannel:
    """A DiscordChannel with mocked gateway and REST clients."""
    with (
        patch(
            "missy.channels.discord.channel.DiscordGatewayClient",
            return_value=mock_gateway,
        ),
        patch(
            "missy.channels.discord.channel.DiscordRestClient",
            return_value=mock_rest,
        ),
    ):
        ch = DiscordChannel(account_config=account_cfg)
    # Attach mocks as attributes for easy access in tests.
    ch._gateway = mock_gateway
    ch._rest = mock_rest
    return ch


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


class TestDiscordChannelConstruction:
    def test_channel_name_attribute(self, channel):
        assert channel.name == "discord"

    def test_account_config_stored(self, channel, account_cfg):
        assert channel.account_config is account_cfg

    def test_queue_is_empty_on_construction(self, channel):
        assert channel._queue.empty()

    def test_bot_user_id_none_before_ready(self, channel):
        channel._bot_user_id = None
        channel._gateway.bot_user_id = None
        assert channel.bot_user_id is None

    def test_pending_pairs_empty_on_construction(self, channel):
        assert channel._pending_pairs == set()

    def test_thread_sessions_empty_on_construction(self, channel):
        assert channel._thread_sessions == {}

    def test_screencast_none_on_construction(self, channel):
        assert channel._screencast is None

    def test_agent_runtime_none_on_construction(self, channel):
        assert channel._agent_runtime is None

    def test_voice_none_on_construction(self, channel):
        assert channel._voice is None

    def test_custom_queue_max(self, account_cfg, mock_gateway, mock_rest):
        with (
            patch(
                "missy.channels.discord.channel.DiscordGatewayClient",
                return_value=mock_gateway,
            ),
            patch(
                "missy.channels.discord.channel.DiscordRestClient",
                return_value=mock_rest,
            ),
        ):
            ch = DiscordChannel(account_config=account_cfg, queue_max=8)
        assert ch._queue.maxsize == 8

    def test_session_id_and_task_id_stored(self, account_cfg, mock_gateway, mock_rest):
        with (
            patch(
                "missy.channels.discord.channel.DiscordGatewayClient",
                return_value=mock_gateway,
            ),
            patch(
                "missy.channels.discord.channel.DiscordRestClient",
                return_value=mock_rest,
            ),
        ):
            ch = DiscordChannel(
                account_config=account_cfg,
                session_id="s-custom",
                task_id="t-custom",
            )
        assert ch._session_id == "s-custom"
        assert ch._task_id == "t-custom"

    def test_auto_thread_threshold_from_config(self, mock_gateway, mock_rest):
        cfg = _make_account(auto_thread_threshold=5)
        with (
            patch(
                "missy.channels.discord.channel.DiscordGatewayClient",
                return_value=mock_gateway,
            ),
            patch(
                "missy.channels.discord.channel.DiscordRestClient",
                return_value=mock_rest,
            ),
        ):
            ch = DiscordChannel(account_config=cfg)
        assert ch._auto_thread_threshold == 5


# ---------------------------------------------------------------------------
# set_screencast / set_agent_runtime
# ---------------------------------------------------------------------------


class TestSetters:
    def test_set_screencast_stores_reference(self, channel):
        sentinel = object()
        channel.set_screencast(sentinel)
        assert channel._screencast is sentinel

    def test_set_screencast_overrides_previous(self, channel):
        first = object()
        second = object()
        channel.set_screencast(first)
        channel.set_screencast(second)
        assert channel._screencast is second

    def test_set_screencast_none(self, channel):
        channel.set_screencast(object())
        channel.set_screencast(None)
        assert channel._screencast is None

    def test_set_agent_runtime_stores_reference(self, channel):
        runtime = MagicMock()
        channel.set_agent_runtime(runtime)
        assert channel._agent_runtime is runtime

    def test_set_agent_runtime_overrides_previous(self, channel):
        first = MagicMock()
        second = MagicMock()
        channel.set_agent_runtime(first)
        channel.set_agent_runtime(second)
        assert channel._agent_runtime is second

    def test_set_agent_runtime_none(self, channel):
        channel.set_agent_runtime(MagicMock())
        channel.set_agent_runtime(None)
        assert channel._agent_runtime is None


# ---------------------------------------------------------------------------
# bot_user_id property
# ---------------------------------------------------------------------------


class TestBotUserId:
    def test_returns_local_value_when_set(self, channel):
        channel._bot_user_id = "bot-123"
        channel._gateway.bot_user_id = "gw-456"
        assert channel.bot_user_id == "bot-123"

    def test_falls_back_to_gateway_when_local_is_none(self, channel):
        channel._bot_user_id = None
        channel._gateway.bot_user_id = "gw-456"
        assert channel.bot_user_id == "gw-456"

    def test_returns_none_when_both_unset(self, channel):
        channel._bot_user_id = None
        channel._gateway.bot_user_id = None
        assert channel.bot_user_id is None

    def test_empty_string_local_does_not_shadow_gateway(self, channel):
        # Empty string is falsy — gateway value should still take effect.
        channel._bot_user_id = ""
        channel._gateway.bot_user_id = "gw-789"
        # Property: `return self._bot_user_id or self._gateway.bot_user_id`
        # "" is falsy, so we expect gateway value.
        assert channel.bot_user_id == "gw-789"


# ---------------------------------------------------------------------------
# _is_own_message
# ---------------------------------------------------------------------------


class TestIsOwnMessage:
    def test_matches_bot_user_id(self, channel):
        channel._bot_user_id = "bot-111"
        assert channel._is_own_message("bot-111") is True

    def test_does_not_match_different_author(self, channel):
        channel._bot_user_id = "bot-111"
        assert channel._is_own_message("user-999") is False

    def test_uses_account_id_when_bot_user_id_absent(self, channel):
        channel._bot_user_id = None
        channel._gateway.bot_user_id = None
        channel.account_config.account_id = "acct-222"
        assert channel._is_own_message("acct-222") is True

    def test_returns_false_when_no_id_configured(self, channel):
        channel._bot_user_id = None
        channel._gateway.bot_user_id = None
        channel.account_config.account_id = None
        assert channel._is_own_message("anyone") is False


# ---------------------------------------------------------------------------
# _allow_bot_author
# ---------------------------------------------------------------------------


class TestAllowBotAuthor:
    def test_non_bot_always_allowed(self, channel):
        author = {"id": "u-1", "bot": False}
        assert channel._allow_bot_author(author, "hello", "guild-1") is True

    def test_non_bot_missing_bot_key(self, channel):
        author = {"id": "u-1"}
        assert channel._allow_bot_author(author, "hello", "guild-1") is True

    def test_bot_ignored_when_ignore_bots_true(self, channel):
        channel.account_config.ignore_bots = True
        channel.account_config.allow_bots_if_mention_only = False
        author = {"id": "other-bot", "bot": True}
        assert channel._allow_bot_author(author, "hello", "guild-1") is False

    def test_bot_allowed_when_ignore_bots_false(self, channel):
        channel.account_config.ignore_bots = False
        author = {"id": "other-bot", "bot": True}
        assert channel._allow_bot_author(author, "hello", "guild-1") is True

    def test_bot_allowed_via_mention_exemption(self, channel):
        channel._bot_user_id = "bot-123"
        channel.account_config.ignore_bots = True
        channel.account_config.allow_bots_if_mention_only = True
        author = {"id": "other-bot", "bot": True}
        # Message includes a mention of this bot.
        content = "Hey <@bot-123> can you help?"
        assert channel._allow_bot_author(author, content, "guild-1") is True

    def test_bot_with_nickname_mention_exemption(self, channel):
        channel._bot_user_id = "bot-123"
        channel.account_config.ignore_bots = True
        channel.account_config.allow_bots_if_mention_only = True
        author = {"id": "other-bot", "bot": True}
        # Nickname mention format <@!id>
        content = "Hey <@!bot-123> can you help?"
        assert channel._allow_bot_author(author, content, "guild-1") is True

    def test_bot_mention_exemption_but_no_mention(self, channel):
        channel._bot_user_id = "bot-123"
        channel.account_config.ignore_bots = True
        channel.account_config.allow_bots_if_mention_only = True
        author = {"id": "other-bot", "bot": True}
        content = "Hello without a mention"
        assert channel._allow_bot_author(author, content, "guild-1") is False


# ---------------------------------------------------------------------------
# _check_dm_policy
# ---------------------------------------------------------------------------


class TestCheckDMPolicy:
    def test_disabled_policy_denies(self, channel):
        channel.account_config.dm_policy = DiscordDMPolicy.DISABLED
        assert channel._check_dm_policy("user-1", "hello") is False

    def test_open_policy_allows_all(self, channel):
        channel.account_config.dm_policy = DiscordDMPolicy.OPEN
        assert channel._check_dm_policy("user-any", "hello") is True

    def test_allowlist_allows_listed_user(self, channel):
        channel.account_config.dm_policy = DiscordDMPolicy.ALLOWLIST
        channel.account_config.dm_allowlist = ["user-allowed"]
        assert channel._check_dm_policy("user-allowed", "hello") is True

    def test_allowlist_denies_unlisted_user(self, channel):
        channel.account_config.dm_policy = DiscordDMPolicy.ALLOWLIST
        channel.account_config.dm_allowlist = ["user-allowed"]
        assert channel._check_dm_policy("user-other", "hello") is False

    def test_allowlist_empty_denies_all(self, channel):
        channel.account_config.dm_policy = DiscordDMPolicy.ALLOWLIST
        channel.account_config.dm_allowlist = []
        assert channel._check_dm_policy("user-any", "hello") is False

    def test_pairing_denies_unregistered_user(self, channel):
        channel.account_config.dm_policy = DiscordDMPolicy.PAIRING
        channel.account_config.dm_allowlist = []
        assert channel._check_dm_policy("stranger", "hello") is False

    def test_pairing_allows_already_paired_user(self, channel):
        channel.account_config.dm_policy = DiscordDMPolicy.PAIRING
        channel.account_config.dm_allowlist = ["paired-user"]
        assert channel._check_dm_policy("paired-user", "hello") is True


# ---------------------------------------------------------------------------
# _check_pairing
# ---------------------------------------------------------------------------


class TestCheckPairing:
    def test_pair_command_adds_to_pending(self, channel):
        channel._check_pairing("u-new", "!pair")
        assert "u-new" in channel._pending_pairs

    def test_pair_command_returns_false(self, channel):
        result = channel._check_pairing("u-new", "!pair")
        assert result is False

    def test_slash_pair_command(self, channel):
        channel._check_pairing("u-new", "/pair")
        assert "u-new" in channel._pending_pairs

    def test_accept_command_moves_to_allowlist(self, channel):
        channel._pending_pairs.add("u-pending")
        channel._check_pairing("admin", "!pair accept u-pending")
        assert "u-pending" not in channel._pending_pairs
        assert "u-pending" in channel.account_config.dm_allowlist

    def test_accept_command_returns_false(self, channel):
        channel._pending_pairs.add("u-pending")
        result = channel._check_pairing("admin", "!pair accept u-pending")
        assert result is False

    def test_deny_command_removes_from_pending(self, channel):
        channel._pending_pairs.add("u-pending")
        channel._check_pairing("admin", "!pair deny u-pending")
        assert "u-pending" not in channel._pending_pairs

    def test_deny_command_returns_false(self, channel):
        channel._pending_pairs.add("u-pending")
        result = channel._check_pairing("admin", "!pair deny u-pending")
        assert result is False

    def test_regular_message_denied_when_not_in_allowlist(self, channel):
        channel.account_config.dm_allowlist = []
        result = channel._check_pairing("u-stranger", "Hello bot!")
        assert result is False

    def test_regular_message_allowed_when_in_allowlist(self, channel):
        channel.account_config.dm_allowlist = ["u-paired"]
        result = channel._check_pairing("u-paired", "Hello bot!")
        assert result is True


# ---------------------------------------------------------------------------
# _check_guild_policy
# ---------------------------------------------------------------------------


class TestCheckGuildPolicy:
    def test_no_policy_for_guild_denies(self, channel):
        channel.account_config.guild_policies = {}
        result = channel._check_guild_policy("guild-x", "ch-1", "user-1", "hi", {})
        assert result is False

    def test_disabled_guild_denies(self, channel):
        channel.account_config.guild_policies = {"guild-1": DiscordGuildPolicy(enabled=False)}
        result = channel._check_guild_policy("guild-1", "ch-1", "user-1", "hi", {})
        assert result is False

    def test_enabled_guild_no_restrictions_allows(self, channel):
        channel.account_config.guild_policies = {"guild-1": DiscordGuildPolicy(enabled=True)}
        result = channel._check_guild_policy("guild-1", "ch-1", "user-1", "hi", {})
        assert result is True

    def test_channel_allowlist_allows_listed_channel_id(self, channel):
        channel.account_config.guild_policies = {
            "guild-1": DiscordGuildPolicy(enabled=True, allowed_channels=["ch-allowed"])
        }
        result = channel._check_guild_policy("guild-1", "ch-allowed", "user-1", "hi", {})
        assert result is True

    def test_channel_allowlist_denies_unlisted_channel(self, channel):
        channel.account_config.guild_policies = {
            "guild-1": DiscordGuildPolicy(enabled=True, allowed_channels=["ch-allowed"])
        }
        result = channel._check_guild_policy("guild-1", "ch-other", "user-1", "hi", {})
        assert result is False

    def test_channel_allowlist_matches_by_name_in_data(self, channel):
        channel.account_config.guild_policies = {
            "guild-1": DiscordGuildPolicy(enabled=True, allowed_channels=["general"])
        }
        data = {"channel": {"name": "general"}}
        result = channel._check_guild_policy("guild-1", "ch-id-99", "user-1", "hi", data)
        assert result is True

    def test_user_allowlist_allows_listed_user(self, channel):
        channel.account_config.guild_policies = {
            "guild-1": DiscordGuildPolicy(enabled=True, allowed_users=["u-permitted"])
        }
        result = channel._check_guild_policy("guild-1", "ch-1", "u-permitted", "hi", {})
        assert result is True

    def test_user_allowlist_denies_unlisted_user(self, channel):
        channel.account_config.guild_policies = {
            "guild-1": DiscordGuildPolicy(enabled=True, allowed_users=["u-permitted"])
        }
        result = channel._check_guild_policy("guild-1", "ch-1", "u-stranger", "hi", {})
        assert result is False

    def test_require_mention_allows_when_mentioned(self, channel):
        channel._bot_user_id = "bot-111"
        channel.account_config.guild_policies = {
            "guild-1": DiscordGuildPolicy(enabled=True, require_mention=True)
        }
        content = "Hey <@bot-111> help me"
        result = channel._check_guild_policy("guild-1", "ch-1", "user-1", content, {})
        assert result is True

    def test_require_mention_denies_when_not_mentioned(self, channel):
        channel._bot_user_id = "bot-111"
        channel.account_config.guild_policies = {
            "guild-1": DiscordGuildPolicy(enabled=True, require_mention=True)
        }
        result = channel._check_guild_policy(
            "guild-1", "ch-1", "user-1", "hello without mention", {}
        )
        assert result is False

    def test_require_mention_nickname_format(self, channel):
        channel._bot_user_id = "bot-111"
        channel.account_config.guild_policies = {
            "guild-1": DiscordGuildPolicy(enabled=True, require_mention=True)
        }
        content = "Hey <@!bot-111> are you there?"
        result = channel._check_guild_policy("guild-1", "ch-1", "user-1", content, {})
        assert result is True

    def test_require_mention_fallback_to_mentions_list_when_no_bot_id(self, channel):
        channel._bot_user_id = None
        channel._gateway.bot_user_id = None
        channel.account_config.account_id = None
        channel.account_config.guild_policies = {
            "guild-1": DiscordGuildPolicy(enabled=True, require_mention=True)
        }
        # Without own_id the code checks the mentions list; no mentions → deny.
        result = channel._check_guild_policy("guild-1", "ch-1", "user-1", "hello", {})
        assert result is False


# ---------------------------------------------------------------------------
# Pairing management (accept_pair / deny_pair / get_pending_pairs)
# ---------------------------------------------------------------------------


class TestPairingManagement:
    def test_get_pending_pairs_returns_copy(self, channel):
        channel._pending_pairs = {"u-1", "u-2"}
        pairs = channel.get_pending_pairs()
        assert pairs == {"u-1", "u-2"}
        # Mutating the returned set must not affect internal state.
        pairs.add("u-extra")
        assert "u-extra" not in channel._pending_pairs

    def test_accept_pair_removes_from_pending(self, channel):
        channel._pending_pairs.add("u-1")
        channel.accept_pair("u-1")
        assert "u-1" not in channel._pending_pairs

    def test_accept_pair_adds_to_allowlist(self, channel):
        channel.account_config.dm_allowlist = []
        channel._pending_pairs.add("u-1")
        channel.accept_pair("u-1")
        assert "u-1" in channel.account_config.dm_allowlist

    def test_accept_pair_does_not_duplicate_in_allowlist(self, channel):
        channel.account_config.dm_allowlist = ["u-1"]
        channel._pending_pairs.add("u-1")
        channel.accept_pair("u-1")
        assert channel.account_config.dm_allowlist.count("u-1") == 1

    def test_deny_pair_removes_from_pending(self, channel):
        channel._pending_pairs.add("u-1")
        channel.deny_pair("u-1")
        assert "u-1" not in channel._pending_pairs

    def test_deny_pair_does_not_add_to_allowlist(self, channel):
        channel.account_config.dm_allowlist = []
        channel._pending_pairs.add("u-1")
        channel.deny_pair("u-1")
        assert "u-1" not in channel.account_config.dm_allowlist

    def test_deny_pair_noop_for_unknown_user(self, channel):
        # Should not raise.
        channel.deny_pair("unknown-user")

    def test_accept_pair_noop_for_unknown_user(self, channel):
        # Not in pending; should still add to allowlist without error.
        channel.account_config.dm_allowlist = []
        channel.accept_pair("unknown-user")
        assert "unknown-user" in channel.account_config.dm_allowlist


# ---------------------------------------------------------------------------
# Thread session management
# ---------------------------------------------------------------------------


class TestThreadSessionManagement:
    def test_get_thread_session_returns_none_for_unknown(self, channel):
        assert channel.get_thread_session("nonexistent-thread") is None

    def test_set_and_get_thread_session(self, channel):
        channel.set_thread_session("thread-1", "session-abc")
        assert channel.get_thread_session("thread-1") == "session-abc"

    def test_set_thread_session_overwrites(self, channel):
        channel.set_thread_session("thread-1", "session-old")
        channel.set_thread_session("thread-1", "session-new")
        assert channel.get_thread_session("thread-1") == "session-new"

    def test_multiple_thread_sessions_independent(self, channel):
        channel.set_thread_session("t-1", "s-1")
        channel.set_thread_session("t-2", "s-2")
        assert channel.get_thread_session("t-1") == "s-1"
        assert channel.get_thread_session("t-2") == "s-2"


# ---------------------------------------------------------------------------
# BaseChannel interface
# ---------------------------------------------------------------------------


class TestBaseChannelInterface:
    def test_receive_raises_not_implemented(self, channel):
        with pytest.raises(NotImplementedError):
            channel.receive()

    @pytest.mark.asyncio
    async def test_areceive_returns_enqueued_message(self, channel):
        from missy.channels.base import ChannelMessage

        msg = ChannelMessage(content="hello", sender="u-1", channel="discord")
        await channel._queue.put(msg)
        result = await channel.areceive()
        assert result is msg

    def test_send_drops_when_no_channel_context(self, channel):
        channel._current_channel_id = None
        # Should not raise; message silently dropped.
        channel.send("hello")
        channel._rest.send_message.assert_not_called()

    @pytest.mark.asyncio
    async def test_send_creates_task_with_running_loop(self, channel):
        channel._current_channel_id = "ch-99"

        # Patch send_with_retry to a no-op coroutine so the task doesn't
        # attempt actual network calls.
        async def _noop(*_args: Any, **_kwargs: Any) -> str:
            return "msg-ok"

        channel.send_with_retry = _noop  # type: ignore[method-assign]

        # call send() inside a running event loop to verify it creates a task.
        asyncio.get_event_loop()
        channel.send("hello there")
        # Give the newly scheduled task a chance to start.
        await asyncio.sleep(0)


# ---------------------------------------------------------------------------
# send_to — message chunking and error handling
# ---------------------------------------------------------------------------


class TestSendTo:
    @pytest.mark.asyncio
    async def test_send_short_message(self, channel):
        channel._rest.send_message = MagicMock(return_value={"id": "m-1"})
        msg_id = await channel.send_to("ch-1", "Hello")
        assert msg_id == "m-1"
        channel._rest.send_message.assert_called_once()

    @pytest.mark.asyncio
    async def test_send_long_message_splits_into_chunks(self, channel):
        sent_chunks: list[str] = []

        def _capture(channel_id: str, content: str, **_kwargs: Any) -> dict:
            sent_chunks.append(content)
            return {"id": f"m-{len(sent_chunks)}"}

        channel._rest.send_message = _capture
        # 1990-char limit per chunk; send 3 990-char blocks = 2970 chars → 2 chunks.
        big_message = "A" * 2970
        await channel.send_to("ch-1", big_message)
        assert len(sent_chunks) == 2
        assert all(len(c) <= 1990 for c in sent_chunks)

    @pytest.mark.asyncio
    async def test_send_to_raises_discord_send_error_on_failure(self, channel):
        channel._rest.send_message = MagicMock(side_effect=RuntimeError("network error"))
        with pytest.raises(DiscordSendError):
            await channel.send_to("ch-1", "Hello")

    @pytest.mark.asyncio
    async def test_send_to_uses_thread_id_when_provided(self, channel):
        channel._rest.send_message = MagicMock(return_value={"id": "m-1"})
        await channel.send_to("ch-parent", "Hello", thread_id="thread-99")
        call_kwargs = channel._rest.send_message.call_args
        # thread_id should override channel_id as the target.
        assert call_kwargs[1]["channel_id"] == "thread-99" or (
            call_kwargs[0] and call_kwargs[0][0] == "thread-99"
        )


# ---------------------------------------------------------------------------
# Minimal config (no optional fields)
# ---------------------------------------------------------------------------


class TestMinimalConfig:
    def test_minimal_account_constructs(self, mock_gateway, mock_rest):
        """DiscordChannel must accept a bare-minimum DiscordAccountConfig."""
        cfg = DiscordAccountConfig()  # all defaults
        with (
            patch(
                "missy.channels.discord.channel.DiscordGatewayClient",
                return_value=mock_gateway,
            ),
            patch(
                "missy.channels.discord.channel.DiscordRestClient",
                return_value=mock_rest,
            ),
        ):
            ch = DiscordChannel(account_config=cfg)
        assert ch.name == "discord"
        assert ch.bot_user_id is None

    def test_minimal_config_dm_policy_defaults_to_disabled(self):
        cfg = DiscordAccountConfig()
        assert cfg.dm_policy == DiscordDMPolicy.DISABLED

    def test_minimal_config_guild_policies_empty(self):
        cfg = DiscordAccountConfig()
        assert cfg.guild_policies == {}

    def test_minimal_config_no_application_id(self, mock_gateway, mock_rest):
        cfg = DiscordAccountConfig(application_id="")
        with (
            patch(
                "missy.channels.discord.channel.DiscordGatewayClient",
                return_value=mock_gateway,
            ),
            patch(
                "missy.channels.discord.channel.DiscordRestClient",
                return_value=mock_rest,
            ),
        ):
            ch = DiscordChannel(account_config=cfg)
        # No application_id → skip slash command registration in start()
        assert ch.account_config.application_id == ""


# ---------------------------------------------------------------------------
# Lifecycle: start / stop
# ---------------------------------------------------------------------------


class TestLifecycle:
    @pytest.mark.asyncio
    async def test_start_creates_gateway_task(self, channel):
        channel._gateway.run = AsyncMock()
        await channel.start()
        assert channel._gateway_task is not None
        # Clean up.
        channel._gateway_task.cancel()
        import contextlib

        with contextlib.suppress(asyncio.CancelledError):
            await channel._gateway_task

    @pytest.mark.asyncio
    async def test_start_registers_slash_commands_when_app_id_set(self, channel):
        channel.account_config.application_id = "app-123"
        channel._gateway.run = AsyncMock()

        with patch("missy.channels.discord.channel.DiscordGatewayClient"):
            pass  # gateway already mocked

        await channel.start()

        # Slash command registration should have been attempted.
        channel._rest.register_slash_commands.assert_called_once()
        channel._gateway_task.cancel()
        import contextlib

        with contextlib.suppress(asyncio.CancelledError):
            await channel._gateway_task

    @pytest.mark.asyncio
    async def test_stop_disconnects_gateway(self, channel):
        channel._gateway.disconnect = AsyncMock()
        # Simulate a running task to cancel.
        channel._gateway_task = asyncio.create_task(asyncio.sleep(100))
        await channel.stop()
        channel._gateway.disconnect.assert_awaited_once()
        assert channel._gateway_task is None


# ---------------------------------------------------------------------------
# _on_gateway_event routing
# ---------------------------------------------------------------------------


class TestOnGatewayEvent:
    @pytest.mark.asyncio
    async def test_bot_user_id_populated_from_gateway_on_event(self, channel):
        channel._bot_user_id = None
        channel._gateway.bot_user_id = "gw-bot-999"

        # Dispatch a GUILD_CREATE event — handled but should populate bot_user_id.
        payload = {"t": "GUILD_CREATE", "d": {"id": "guild-1"}}
        await channel._on_gateway_event(payload)

        assert channel._bot_user_id == "gw-bot-999"

    @pytest.mark.asyncio
    async def test_unknown_event_type_does_not_raise(self, channel):
        payload = {"t": "UNKNOWN_EVENT", "d": {}}
        # Should complete without raising.
        await channel._on_gateway_event(payload)

    @pytest.mark.asyncio
    async def test_guild_create_event_does_not_enqueue_message(self, channel):
        payload = {"t": "GUILD_CREATE", "d": {"id": "guild-1"}}
        await channel._on_gateway_event(payload)
        assert channel._queue.empty()


# ---------------------------------------------------------------------------
# _handle_message — access control integration
# ---------------------------------------------------------------------------


class TestHandleMessageAccessControl:
    def _make_message_payload(
        self,
        *,
        author_id: str = "user-1",
        channel_id: str = "ch-1",
        guild_id: str | None = "guild-1",
        content: str = "hello",
        is_bot: bool = False,
        message_id: str = "msg-1",
    ) -> dict:
        return {
            "id": message_id,
            "channel_id": channel_id,
            "guild_id": guild_id,
            "content": content,
            "author": {"id": author_id, "bot": is_bot},
        }

    @pytest.mark.asyncio
    async def test_own_message_not_enqueued(self, channel):
        channel._bot_user_id = "bot-id"
        data = self._make_message_payload(author_id="bot-id")
        await channel._handle_message(data)
        assert channel._queue.empty()

    @pytest.mark.asyncio
    async def test_dm_disabled_policy_drops_message(self, channel):
        channel.account_config.dm_policy = DiscordDMPolicy.DISABLED
        data = self._make_message_payload(guild_id=None, author_id="user-x")
        await channel._handle_message(data)
        assert channel._queue.empty()

    @pytest.mark.asyncio
    async def test_dm_open_policy_enqueues_message(self, channel):
        channel.account_config.dm_policy = DiscordDMPolicy.OPEN
        data = self._make_message_payload(guild_id=None, author_id="user-x")
        await channel._handle_message(data)
        assert not channel._queue.empty()
        msg = channel._queue.get_nowait()
        assert msg.content == "hello"

    @pytest.mark.asyncio
    async def test_guild_message_no_policy_drops(self, channel):
        channel.account_config.guild_policies = {}
        data = self._make_message_payload(guild_id="guild-unknown")
        await channel._handle_message(data)
        assert channel._queue.empty()

    @pytest.mark.asyncio
    async def test_guild_message_with_open_policy_enqueues(self, channel):
        channel.account_config.guild_policies = {"guild-1": DiscordGuildPolicy(enabled=True)}
        data = self._make_message_payload(guild_id="guild-1", content="hi")
        await channel._handle_message(data)
        assert not channel._queue.empty()
        msg = channel._queue.get_nowait()
        assert msg.sender == "user-1"

    @pytest.mark.asyncio
    async def test_guild_message_sets_current_channel_id(self, channel):
        channel.account_config.guild_policies = {"guild-1": DiscordGuildPolicy(enabled=True)}
        data = self._make_message_payload(guild_id="guild-1", channel_id="ch-active")
        await channel._handle_message(data)
        assert channel._current_channel_id == "ch-active"

    @pytest.mark.asyncio
    async def test_bot_message_filtered_when_ignore_bots(self, channel):
        channel.account_config.ignore_bots = True
        channel.account_config.allow_bots_if_mention_only = False
        channel.account_config.guild_policies = {"guild-1": DiscordGuildPolicy(enabled=True)}
        data = self._make_message_payload(guild_id="guild-1", is_bot=True, author_id="other-bot")
        await channel._handle_message(data)
        assert channel._queue.empty()
