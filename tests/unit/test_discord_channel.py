"""Unit tests for Discord channel components.

Tests cover:
- DiscordRestClient request construction (mocked httpx)
- Access control: bot filtering, DM policy, guild allowlist
- Pairing workflow state transitions
- Slash command routing
- Audit events emitted on deny/allow

No real Discord connectivity is required — everything is mocked.
"""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from missy.channels.discord.channel import DiscordChannel
from missy.channels.discord.config import (
    DiscordAccountConfig,
    DiscordDMPolicy,
    DiscordGuildPolicy,
)
from missy.channels.discord.rest import BASE, DiscordRestClient
from missy.core.events import AuditEvent, EventBus

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def event_bus_fresh() -> EventBus:
    """A clean EventBus isolated from the global singleton."""
    return EventBus()


@pytest.fixture()
def mock_http_client() -> MagicMock:
    """A PolicyHTTPClient mock that records calls."""
    client = MagicMock()
    # Make response objects behave like httpx.Response
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {}
    mock_response.raise_for_status.return_value = None
    client.get.return_value = mock_response
    client.post.return_value = mock_response
    return client


@pytest.fixture()
def rest_client(mock_http_client: MagicMock) -> DiscordRestClient:
    """DiscordRestClient wired to the mock HTTP client."""
    return DiscordRestClient(bot_token="Test.Token.Here", http_client=mock_http_client)


@pytest.fixture()
def open_dm_account() -> DiscordAccountConfig:
    return DiscordAccountConfig(
        token_env_var="DISCORD_BOT_TOKEN",
        account_id="bot-001",
        dm_policy=DiscordDMPolicy.OPEN,
        ignore_bots=True,
    )


@pytest.fixture()
def disabled_dm_account() -> DiscordAccountConfig:
    return DiscordAccountConfig(
        token_env_var="DISCORD_BOT_TOKEN",
        account_id="bot-001",
        dm_policy=DiscordDMPolicy.DISABLED,
        ignore_bots=True,
    )


@pytest.fixture()
def allowlist_dm_account() -> DiscordAccountConfig:
    return DiscordAccountConfig(
        token_env_var="DISCORD_BOT_TOKEN",
        account_id="bot-001",
        dm_policy=DiscordDMPolicy.ALLOWLIST,
        dm_allowlist=["user-allowed"],
    )


@pytest.fixture()
def pairing_dm_account() -> DiscordAccountConfig:
    return DiscordAccountConfig(
        token_env_var="DISCORD_BOT_TOKEN",
        account_id="bot-001",
        dm_policy=DiscordDMPolicy.PAIRING,
    )


@pytest.fixture()
def guild_account() -> DiscordAccountConfig:
    return DiscordAccountConfig(
        token_env_var="DISCORD_BOT_TOKEN",
        account_id="bot-001",
        dm_policy=DiscordDMPolicy.DISABLED,
        guild_policies={
            "guild-111": DiscordGuildPolicy(
                enabled=True,
                require_mention=False,
                allowed_channels=[],
                allowed_users=[],
            )
        },
    )


def _make_channel(account: DiscordAccountConfig, bus: EventBus | None = None) -> DiscordChannel:
    """Build a DiscordChannel with patched gateway and rest clients."""
    with (
        patch("missy.channels.discord.channel.DiscordGatewayClient"),
        patch("missy.channels.discord.channel.DiscordRestClient"),
    ):
        channel = DiscordChannel(account_config=account)
    if bus is not None:
        # Patch the channel's audit emitter to use our isolated bus.

        def patched_emit(event_type: str, result: str, detail: dict[str, Any]) -> None:
            event = AuditEvent.now(
                session_id="test",
                task_id="test",
                event_type=event_type,
                category="network",
                result=result,  # type: ignore[arg-type]
                detail=detail,
            )
            bus.publish(event)

        channel._emit_audit = patched_emit  # type: ignore[method-assign]
    return channel


def _make_message(
    author_id: str = "user-123",
    content: str = "hello",
    guild_id: str | None = None,
    channel_id: str = "chan-1",
    is_bot: bool = False,
) -> dict[str, Any]:
    return {
        "id": "msg-1",
        "channel_id": channel_id,
        "guild_id": guild_id,
        "content": content,
        "author": {
            "id": author_id,
            "username": "TestUser",
            "bot": is_bot,
        },
    }


# ---------------------------------------------------------------------------
# DiscordRestClient request construction
# ---------------------------------------------------------------------------


class TestDiscordRestClient:
    def test_token_prefix_added(self, mock_http_client: MagicMock) -> None:
        client = DiscordRestClient(bot_token="rawtoken", http_client=mock_http_client)
        assert client._token == "Bot rawtoken"

    def test_token_prefix_not_doubled(self, mock_http_client: MagicMock) -> None:
        client = DiscordRestClient(bot_token="Bot mytoken", http_client=mock_http_client)
        assert client._token == "Bot mytoken"

    def test_get_current_user_calls_correct_url(
        self, rest_client: DiscordRestClient, mock_http_client: MagicMock
    ) -> None:
        mock_http_client.get.return_value.json.return_value = {"id": "123", "username": "TestBot"}
        result = rest_client.get_current_user()
        mock_http_client.get.assert_called_once()
        call_url = mock_http_client.get.call_args[0][0]
        assert call_url == f"{BASE}/users/@me"
        assert result["username"] == "TestBot"

    def test_get_gateway_bot_calls_correct_url(
        self, rest_client: DiscordRestClient, mock_http_client: MagicMock
    ) -> None:
        mock_http_client.get.return_value.json.return_value = {"url": "wss://gateway.discord.gg"}
        rest_client.get_gateway_bot()
        call_url = mock_http_client.get.call_args[0][0]
        assert call_url == f"{BASE}/gateway/bot"

    def test_send_message_no_reply(
        self, rest_client: DiscordRestClient, mock_http_client: MagicMock
    ) -> None:
        mock_http_client.post.return_value.json.return_value = {"id": "200000000000000001"}
        rest_client.send_message(channel_id="100000000000000001", content="Hello!")
        mock_http_client.post.assert_called_once()
        call_url = mock_http_client.post.call_args[0][0]
        assert "100000000000000001/messages" in call_url
        call_kwargs = mock_http_client.post.call_args[1]
        assert call_kwargs["json"]["content"] == "Hello!"
        assert "message_reference" not in call_kwargs["json"]

    def test_send_message_with_reply(
        self, rest_client: DiscordRestClient, mock_http_client: MagicMock
    ) -> None:
        mock_http_client.post.return_value.json.return_value = {"id": "200000000000000002"}
        rest_client.send_message(
            channel_id="100000000000000001",
            content="Reply!",
            reply_to_message_id="200000000000000010",
        )
        call_kwargs = mock_http_client.post.call_args[1]
        assert call_kwargs["json"]["message_reference"]["message_id"] == "200000000000000010"

    def test_trigger_typing_calls_correct_url(
        self, rest_client: DiscordRestClient, mock_http_client: MagicMock
    ) -> None:
        mock_http_client.post.return_value.status_code = 204
        rest_client.trigger_typing("100000000000000001")
        call_url = mock_http_client.post.call_args[0][0]
        assert "100000000000000001/typing" in call_url

    def test_register_slash_commands_global(
        self, rest_client: DiscordRestClient, mock_http_client: MagicMock
    ) -> None:
        mock_http_client.put.return_value.json.return_value = [{"name": "ask"}]
        commands = [{"name": "ask", "description": "Ask Missy"}]
        rest_client.register_slash_commands(
            application_id="app-111",
            commands=commands,
        )
        call_url = mock_http_client.put.call_args[0][0]
        assert "app-111/commands" in call_url
        assert "guilds" not in call_url

    def test_register_slash_commands_guild(
        self, rest_client: DiscordRestClient, mock_http_client: MagicMock
    ) -> None:
        mock_http_client.put.return_value.json.return_value = [{"name": "ask"}]
        rest_client.register_slash_commands(
            application_id="app-111",
            commands=[],
            guild_id="guild-222",
        )
        call_url = mock_http_client.put.call_args[0][0]
        assert "guilds/guild-222/commands" in call_url

    def test_authorization_header_present(
        self, rest_client: DiscordRestClient, mock_http_client: MagicMock
    ) -> None:
        mock_http_client.get.return_value.json.return_value = {}
        rest_client.get_current_user()
        headers = mock_http_client.get.call_args[1]["headers"]
        assert "Authorization" in headers
        assert headers["Authorization"].startswith("Bot ")


# ---------------------------------------------------------------------------
# Bot filtering
# ---------------------------------------------------------------------------


class TestBotFiltering:
    def test_own_message_filtered(self) -> None:
        account = DiscordAccountConfig(account_id="bot-001", dm_policy=DiscordDMPolicy.OPEN)
        channel = _make_channel(account)
        channel._bot_user_id = "bot-001"
        assert channel._is_own_message("bot-001") is True
        assert channel._is_own_message("other-user") is False

    def test_bot_author_ignored_when_ignore_bots(self) -> None:
        account = DiscordAccountConfig(account_id="bot-001", ignore_bots=True)
        channel = _make_channel(account)
        bot_author = {"id": "other-bot", "bot": True}
        assert channel._allow_bot_author(bot_author, "hello", None) is False

    def test_bot_author_allowed_when_ignore_bots_false(self) -> None:
        account = DiscordAccountConfig(account_id="bot-001", ignore_bots=False)
        channel = _make_channel(account)
        bot_author = {"id": "other-bot", "bot": True}
        assert channel._allow_bot_author(bot_author, "hello", None) is True

    def test_bot_author_allowed_with_mention_exemption(self) -> None:
        account = DiscordAccountConfig(
            account_id="bot-001",
            ignore_bots=True,
            allow_bots_if_mention_only=True,
        )
        channel = _make_channel(account)
        channel._bot_user_id = "bot-001"
        bot_author = {"id": "other-bot", "bot": True}
        # Message explicitly mentions bot-001.
        assert channel._allow_bot_author(bot_author, "hello <@bot-001>", None) is True
        # Message does not mention the bot.
        assert channel._allow_bot_author(bot_author, "just talking", None) is False

    def test_human_author_always_passes_bot_filter(self) -> None:
        account = DiscordAccountConfig(account_id="bot-001", ignore_bots=True)
        channel = _make_channel(account)
        human_author = {"id": "human-1", "bot": False}
        assert channel._allow_bot_author(human_author, "hello", None) is True


# ---------------------------------------------------------------------------
# DM policy
# ---------------------------------------------------------------------------


class TestDMPolicy:
    def test_disabled_policy_denies_all(self, event_bus_fresh: EventBus) -> None:
        account = DiscordAccountConfig(account_id="b", dm_policy=DiscordDMPolicy.DISABLED)
        channel = _make_channel(account, bus=event_bus_fresh)
        assert channel._check_dm_policy("anyone", "hello") is False
        denied = event_bus_fresh.get_events(result="deny")
        assert len(denied) == 1
        assert denied[0].event_type == "discord.channel.message_denied"

    def test_open_policy_allows_all(self) -> None:
        account = DiscordAccountConfig(account_id="b", dm_policy=DiscordDMPolicy.OPEN)
        channel = _make_channel(account)
        assert channel._check_dm_policy("anyone", "hello") is True

    def test_allowlist_allows_listed_user(self) -> None:
        account = DiscordAccountConfig(
            account_id="b",
            dm_policy=DiscordDMPolicy.ALLOWLIST,
            dm_allowlist=["user-allowed"],
        )
        channel = _make_channel(account)
        assert channel._check_dm_policy("user-allowed", "hi") is True

    def test_allowlist_denies_unlisted_user(self, event_bus_fresh: EventBus) -> None:
        account = DiscordAccountConfig(
            account_id="b",
            dm_policy=DiscordDMPolicy.ALLOWLIST,
            dm_allowlist=["user-allowed"],
        )
        channel = _make_channel(account, bus=event_bus_fresh)
        assert channel._check_dm_policy("unknown-user", "hi") is False
        denied = event_bus_fresh.get_events(result="deny")
        assert any(e.event_type == "discord.channel.allowlist_denied" for e in denied)

    def test_pairing_denies_unpaired_user(self, event_bus_fresh: EventBus) -> None:
        account = DiscordAccountConfig(account_id="b", dm_policy=DiscordDMPolicy.PAIRING)
        channel = _make_channel(account, bus=event_bus_fresh)
        assert channel._check_dm_policy("new-user", "hello") is False

    def test_pairing_allows_user_already_in_allowlist(self) -> None:
        account = DiscordAccountConfig(
            account_id="b",
            dm_policy=DiscordDMPolicy.PAIRING,
            dm_allowlist=["paired-user"],
        )
        channel = _make_channel(account)
        assert channel._check_dm_policy("paired-user", "hello") is True


# ---------------------------------------------------------------------------
# Pairing workflow
# ---------------------------------------------------------------------------


class TestPairingWorkflow:
    def test_pair_command_adds_to_pending(self, event_bus_fresh: EventBus) -> None:
        account = DiscordAccountConfig(account_id="b", dm_policy=DiscordDMPolicy.PAIRING)
        channel = _make_channel(account, bus=event_bus_fresh)

        result = channel._check_pairing("user-new", "!pair")
        assert result is False  # Pairing command itself is not forwarded.
        assert "user-new" in channel.get_pending_pairs()

        # A pairing_wait audit event should be emitted.
        events = event_bus_fresh.get_events(event_type="discord.channel.pairing_wait")
        assert len(events) == 1

    def test_accept_pair_moves_to_allowlist(self) -> None:
        account = DiscordAccountConfig(account_id="b", dm_policy=DiscordDMPolicy.PAIRING)
        channel = _make_channel(account)
        channel._pending_pairs.add("user-new")

        channel.accept_pair("user-new")

        assert "user-new" not in channel.get_pending_pairs()
        assert "user-new" in channel.account_config.dm_allowlist

    def test_deny_pair_removes_from_pending(self) -> None:
        account = DiscordAccountConfig(account_id="b", dm_policy=DiscordDMPolicy.PAIRING)
        channel = _make_channel(account)
        channel._pending_pairs.add("user-new")

        channel.deny_pair("user-new")

        assert "user-new" not in channel.get_pending_pairs()
        assert "user-new" not in channel.account_config.dm_allowlist

    def test_accepted_user_can_send_messages(self) -> None:
        account = DiscordAccountConfig(account_id="b", dm_policy=DiscordDMPolicy.PAIRING)
        channel = _make_channel(account)
        channel.accept_pair("user-abc")

        assert channel._check_dm_policy("user-abc", "hello!") is True

    def test_denied_user_cannot_send_messages(self, event_bus_fresh: EventBus) -> None:
        account = DiscordAccountConfig(account_id="b", dm_policy=DiscordDMPolicy.PAIRING)
        channel = _make_channel(account, bus=event_bus_fresh)
        channel._pending_pairs.add("user-bad")
        channel.deny_pair("user-bad")

        result = channel._check_dm_policy("user-bad", "hello!")
        assert result is False

    def test_accept_pair_via_command_message(self) -> None:
        account = DiscordAccountConfig(account_id="b", dm_policy=DiscordDMPolicy.PAIRING)
        channel = _make_channel(account)
        channel._pending_pairs.add("target-user")

        result = channel._check_pairing("admin", "!pair accept target-user")
        assert result is False  # Command itself not forwarded.
        assert "target-user" not in channel.get_pending_pairs()
        assert "target-user" in channel.account_config.dm_allowlist


# ---------------------------------------------------------------------------
# Guild access control
# ---------------------------------------------------------------------------


class TestGuildPolicy:
    def test_no_guild_policy_denies(self, event_bus_fresh: EventBus) -> None:
        account = DiscordAccountConfig(account_id="b", guild_policies={})
        channel = _make_channel(account, bus=event_bus_fresh)
        result = channel._check_guild_policy("unknown-guild", "chan-1", "user-1", "hi", {})
        assert result is False
        denied = event_bus_fresh.get_events(result="deny")
        assert any(e.detail.get("reason") == "no_guild_policy" for e in denied)

    def test_disabled_guild_denies(self, event_bus_fresh: EventBus) -> None:
        account = DiscordAccountConfig(
            account_id="b",
            guild_policies={
                "guild-1": DiscordGuildPolicy(enabled=False),
            },
        )
        channel = _make_channel(account, bus=event_bus_fresh)
        result = channel._check_guild_policy("guild-1", "chan-1", "user-1", "hi", {})
        assert result is False
        denied = event_bus_fresh.get_events(result="deny")
        assert any(e.detail.get("reason") == "guild_disabled" for e in denied)

    def test_enabled_guild_allows(self) -> None:
        account = DiscordAccountConfig(
            account_id="b",
            guild_policies={"guild-1": DiscordGuildPolicy(enabled=True)},
        )
        channel = _make_channel(account)
        result = channel._check_guild_policy("guild-1", "chan-1", "user-1", "hi", {})
        assert result is True

    def test_user_allowlist_denies_unknown_user(self, event_bus_fresh: EventBus) -> None:
        account = DiscordAccountConfig(
            account_id="b",
            guild_policies={
                "guild-1": DiscordGuildPolicy(
                    enabled=True,
                    allowed_users=["user-allowed"],
                )
            },
        )
        channel = _make_channel(account, bus=event_bus_fresh)
        result = channel._check_guild_policy("guild-1", "chan-1", "unknown-user", "hi", {})
        assert result is False
        denied = event_bus_fresh.get_events(result="deny")
        assert any(e.detail.get("reason") == "user_not_in_allowlist" for e in denied)

    def test_user_allowlist_allows_known_user(self) -> None:
        account = DiscordAccountConfig(
            account_id="b",
            guild_policies={
                "guild-1": DiscordGuildPolicy(
                    enabled=True,
                    allowed_users=["user-allowed"],
                )
            },
        )
        channel = _make_channel(account)
        result = channel._check_guild_policy("guild-1", "chan-1", "user-allowed", "hi", {})
        assert result is True

    def test_require_mention_filters_unmention(self, event_bus_fresh: EventBus) -> None:
        account = DiscordAccountConfig(
            account_id="bot-001",
            guild_policies={"guild-1": DiscordGuildPolicy(enabled=True, require_mention=True)},
        )
        channel = _make_channel(account, bus=event_bus_fresh)
        channel._bot_user_id = "bot-001"
        result = channel._check_guild_policy(
            "guild-1", "chan-1", "user-1", "just a message no mention", {}
        )
        assert result is False
        denied = event_bus_fresh.get_events(result="deny")
        assert any(e.detail.get("reason") == "mention_required" for e in denied)

    def test_require_mention_allows_when_mentioned(self) -> None:
        account = DiscordAccountConfig(
            account_id="bot-001",
            guild_policies={"guild-1": DiscordGuildPolicy(enabled=True, require_mention=True)},
        )
        channel = _make_channel(account)
        channel._bot_user_id = "bot-001"
        result = channel._check_guild_policy(
            "guild-1", "chan-1", "user-1", "hey <@bot-001> what's up?", {}
        )
        assert result is True


# ---------------------------------------------------------------------------
# Message handling (asyncio)
# ---------------------------------------------------------------------------


class TestMessageHandling:
    @pytest.mark.asyncio
    async def test_valid_dm_enqueues_message(self) -> None:
        account = DiscordAccountConfig(
            account_id="bot-001",
            dm_policy=DiscordDMPolicy.OPEN,
        )
        channel = _make_channel(account)
        channel._bot_user_id = "bot-001"

        msg_data = _make_message(author_id="user-1", content="Hello Missy", guild_id=None)
        await channel._handle_message(msg_data)

        assert not channel._queue.empty()
        queued = channel._queue.get_nowait()
        assert queued.content == "Hello Missy"
        assert queued.sender == "user-1"
        assert queued.channel == "discord"

    @pytest.mark.asyncio
    async def test_own_message_not_enqueued(self) -> None:
        account = DiscordAccountConfig(
            account_id="bot-001",
            dm_policy=DiscordDMPolicy.OPEN,
        )
        channel = _make_channel(account)
        channel._bot_user_id = "bot-001"

        msg_data = _make_message(author_id="bot-001", content="My own message", guild_id=None)
        await channel._handle_message(msg_data)

        assert channel._queue.empty()

    @pytest.mark.asyncio
    async def test_bot_author_not_enqueued_when_ignore_bots(self) -> None:
        account = DiscordAccountConfig(
            account_id="bot-001",
            dm_policy=DiscordDMPolicy.OPEN,
            ignore_bots=True,
        )
        channel = _make_channel(account)
        channel._bot_user_id = "bot-001"

        msg_data = _make_message(author_id="other-bot", is_bot=True, guild_id=None)
        await channel._handle_message(msg_data)

        assert channel._queue.empty()

    @pytest.mark.asyncio
    async def test_disabled_dm_not_enqueued(self, event_bus_fresh: EventBus) -> None:
        account = DiscordAccountConfig(
            account_id="bot-001",
            dm_policy=DiscordDMPolicy.DISABLED,
        )
        channel = _make_channel(account, bus=event_bus_fresh)
        channel._bot_user_id = "bot-001"

        msg_data = _make_message(author_id="user-1", guild_id=None)
        await channel._handle_message(msg_data)

        assert channel._queue.empty()
        denied = event_bus_fresh.get_events(result="deny")
        assert len(denied) >= 1


# ---------------------------------------------------------------------------
# Slash command routing
# ---------------------------------------------------------------------------


class TestSlashCommandRouting:
    @pytest.mark.asyncio
    async def test_help_command_returns_text(self) -> None:
        from missy.channels.discord.commands import handle_slash_command

        account = DiscordAccountConfig(account_id="bot-001")
        channel = _make_channel(account)

        interaction = {
            "id": "interaction-1",
            "token": "token-abc",
            "data": {"name": "help", "options": []},
        }
        response = await handle_slash_command(interaction, channel)
        assert "/ask" in response
        assert "/help" in response
        assert "/status" in response

    @pytest.mark.asyncio
    async def test_status_command_returns_text(self) -> None:
        from missy.channels.discord.commands import handle_slash_command

        account = DiscordAccountConfig(
            account_id="bot-001",
            dm_policy=DiscordDMPolicy.OPEN,
        )
        channel = _make_channel(account)
        channel._bot_user_id = "bot-001"

        interaction = {
            "id": "interaction-2",
            "token": "token-abc",
            "data": {"name": "status", "options": []},
        }
        response = await handle_slash_command(interaction, channel)
        assert "bot-001" in response or "Status" in response

    @pytest.mark.asyncio
    async def test_unknown_command_returns_error_message(self) -> None:
        from missy.channels.discord.commands import handle_slash_command

        account = DiscordAccountConfig(account_id="bot-001")
        channel = _make_channel(account)

        interaction = {"data": {"name": "nonexistent", "options": []}}
        response = await handle_slash_command(interaction, channel)
        assert "Unknown command" in response or "nonexistent" in response

    @pytest.mark.asyncio
    async def test_ask_without_prompt_returns_guidance(self) -> None:
        from missy.channels.discord.commands import handle_slash_command

        account = DiscordAccountConfig(account_id="bot-001")
        channel = _make_channel(account)

        interaction = {"data": {"name": "ask", "options": []}}
        response = await handle_slash_command(interaction, channel)
        assert "prompt" in response.lower() or "provide" in response.lower()

    def test_slash_commands_list_has_required_commands(self) -> None:
        from missy.channels.discord.commands import SLASH_COMMANDS

        names = {cmd["name"] for cmd in SLASH_COMMANDS}
        assert "ask" in names
        assert "status" in names
        assert "model" in names
        assert "help" in names

    def test_all_commands_have_description(self) -> None:
        from missy.channels.discord.commands import SLASH_COMMANDS

        for cmd in SLASH_COMMANDS:
            assert "description" in cmd and cmd["description"]


# ---------------------------------------------------------------------------
# Audit events
# ---------------------------------------------------------------------------


class TestAuditEvents:
    def test_message_allowed_emits_audit_event(self, event_bus_fresh: EventBus) -> None:
        account = DiscordAccountConfig(
            account_id="bot-001",
            dm_policy=DiscordDMPolicy.OPEN,
        )
        channel = _make_channel(account, bus=event_bus_fresh)
        channel._emit_audit("discord.channel.message_received", "allow", {"author_id": "u1"})

        events = event_bus_fresh.get_events(event_type="discord.channel.message_received")
        assert len(events) == 1
        assert events[0].result == "allow"

    def test_message_denied_emits_audit_event(self, event_bus_fresh: EventBus) -> None:
        account = DiscordAccountConfig(
            account_id="bot-001",
            dm_policy=DiscordDMPolicy.DISABLED,
        )
        channel = _make_channel(account, bus=event_bus_fresh)
        channel._check_dm_policy("user-x", "hello")

        denied = event_bus_fresh.get_events(result="deny")
        assert len(denied) >= 1

    def test_bot_filtered_emits_audit_event(self, event_bus_fresh: EventBus) -> None:
        account = DiscordAccountConfig(account_id="bot-001", ignore_bots=True)
        channel = _make_channel(account, bus=event_bus_fresh)
        channel._emit_audit(
            "discord.channel.bot_filtered", "deny", {"author_id": "bot-x", "is_bot": True}
        )

        events = event_bus_fresh.get_events(event_type="discord.channel.bot_filtered")
        assert len(events) == 1
        assert events[0].detail["is_bot"] is True

    def test_pairing_wait_emits_event(self, event_bus_fresh: EventBus) -> None:
        account = DiscordAccountConfig(account_id="b", dm_policy=DiscordDMPolicy.PAIRING)
        channel = _make_channel(account, bus=event_bus_fresh)
        channel._check_pairing("new-user", "!pair")

        events = event_bus_fresh.get_events(event_type="discord.channel.pairing_wait")
        assert len(events) == 1

    def test_allowlist_denied_emits_event(self, event_bus_fresh: EventBus) -> None:
        account = DiscordAccountConfig(
            account_id="b",
            dm_policy=DiscordDMPolicy.ALLOWLIST,
            dm_allowlist=[],
        )
        channel = _make_channel(account, bus=event_bus_fresh)
        channel._check_dm_policy("blocked-user", "hi")

        events = event_bus_fresh.get_events(event_type="discord.channel.allowlist_denied")
        assert len(events) == 1

    def test_require_mention_filtered_emits_event(self, event_bus_fresh: EventBus) -> None:
        account = DiscordAccountConfig(
            account_id="bot-001",
            guild_policies={"guild-1": DiscordGuildPolicy(enabled=True, require_mention=True)},
        )
        channel = _make_channel(account, bus=event_bus_fresh)
        channel._bot_user_id = "bot-001"
        channel._check_guild_policy("guild-1", "c", "user", "no mention", {})

        events = event_bus_fresh.get_events(event_type="discord.channel.require_mention_filtered")
        assert len(events) == 1


# ---------------------------------------------------------------------------
# Thread management
# ---------------------------------------------------------------------------


class TestDiscordThreadManagement:
    """Tests for thread creation and thread-scoped sessions."""

    def test_set_and_get_thread_session(self, open_dm_account: DiscordAccountConfig) -> None:
        channel = _make_channel(open_dm_account)
        assert channel.get_thread_session("thread-1") is None
        channel.set_thread_session("thread-1", "session-abc")
        assert channel.get_thread_session("thread-1") == "session-abc"

    def test_thread_session_in_metadata(self, open_dm_account: DiscordAccountConfig) -> None:
        """Thread session ID is included in enqueued message metadata."""
        channel = _make_channel(open_dm_account)
        channel._bot_user_id = "bot-001"
        channel.set_thread_session("thread-42", "sess-xyz")

        msg = _make_message(content="hi from thread", channel_id="thread-42")
        msg["channel_type"] = 11  # PUBLIC_THREAD

        loop = asyncio.new_event_loop()
        try:
            loop.run_until_complete(channel._handle_message(msg))
            queued = loop.run_until_complete(channel._queue.get())
            assert queued.metadata["discord_thread_id"] == "thread-42"
            assert queued.metadata["discord_thread_session_id"] == "sess-xyz"
        finally:
            loop.close()

    def test_non_thread_message_has_empty_thread_session(
        self, open_dm_account: DiscordAccountConfig
    ) -> None:
        channel = _make_channel(open_dm_account)
        channel._bot_user_id = "bot-001"

        msg = _make_message(content="normal message")

        loop = asyncio.new_event_loop()
        try:
            loop.run_until_complete(channel._handle_message(msg))
            queued = loop.run_until_complete(channel._queue.get())
            assert queued.metadata["discord_thread_id"] == ""
            assert queued.metadata["discord_thread_session_id"] == ""
        finally:
            loop.close()

    def test_create_thread_success(self, open_dm_account: DiscordAccountConfig) -> None:
        channel = _make_channel(open_dm_account)
        channel._rest = MagicMock()
        channel._rest.create_thread.return_value = {"id": "new-thread-123"}

        loop = asyncio.new_event_loop()
        try:
            thread_id = loop.run_until_complete(
                channel.create_thread("chan-1", "Test Thread", session_id="sess-1")
            )
            assert thread_id == "new-thread-123"
            assert channel.get_thread_session("new-thread-123") == "sess-1"
        finally:
            loop.close()

    def test_create_thread_failure(self, open_dm_account: DiscordAccountConfig) -> None:
        channel = _make_channel(open_dm_account)
        channel._rest = MagicMock()
        channel._rest.create_thread.side_effect = Exception("API error")

        loop = asyncio.new_event_loop()
        try:
            thread_id = loop.run_until_complete(channel.create_thread("chan-1", "Test Thread"))
            assert thread_id is None
        finally:
            loop.close()

    def test_create_thread_with_message_id(self, open_dm_account: DiscordAccountConfig) -> None:
        channel = _make_channel(open_dm_account)
        channel._rest = MagicMock()
        channel._rest.create_thread.return_value = {"id": "thread-from-msg"}

        loop = asyncio.new_event_loop()
        try:
            thread_id = loop.run_until_complete(
                channel.create_thread("chan-1", "Thread", message_id="msg-42")
            )
            assert thread_id == "thread-from-msg"
            channel._rest.create_thread.assert_called_once_with(
                channel_id="chan-1", name="Thread", message_id="msg-42"
            )
        finally:
            loop.close()

    def test_auto_thread_threshold_config(self) -> None:
        account = DiscordAccountConfig(
            token_env_var="DISCORD_BOT_TOKEN",
            account_id="bot-001",
            dm_policy=DiscordDMPolicy.OPEN,
            auto_thread_threshold=5,
        )
        channel = _make_channel(account)
        assert channel._auto_thread_threshold == 5

    def test_channel_message_count_tracking(self, event_bus_fresh: EventBus) -> None:
        """Messages in guild channels increment the message count."""
        account = DiscordAccountConfig(
            token_env_var="DISCORD_BOT_TOKEN",
            account_id="bot-001",
            dm_policy=DiscordDMPolicy.DISABLED,
            guild_policies={
                "guild-1": DiscordGuildPolicy(enabled=True),
            },
            auto_thread_threshold=10,
        )
        channel = _make_channel(account, bus=event_bus_fresh)
        channel._bot_user_id = "bot-001"

        msg = _make_message(content="hello", guild_id="guild-1", channel_id="chan-1")

        loop = asyncio.new_event_loop()
        try:
            loop.run_until_complete(channel._handle_message(msg))
            assert channel._channel_message_counts.get("chan-1") == 1
        finally:
            loop.close()


# ---------------------------------------------------------------------------
# REST client thread operations
# ---------------------------------------------------------------------------


class TestDiscordRestThreads:
    """Tests for REST client thread creation and channel fetching."""

    def test_create_thread_without_message(
        self, rest_client: DiscordRestClient, mock_http_client: MagicMock
    ) -> None:
        mock_http_client.post.return_value.json.return_value = {"id": "500000000000000001"}
        result = rest_client.create_thread("100000000000000001", "My Thread")
        assert result["id"] == "500000000000000001"
        call_args = mock_http_client.post.call_args
        assert "/channels/100000000000000001/threads" in call_args[0][0]
        body = call_args[1]["json"]
        assert body["name"] == "My Thread"
        assert body["type"] == 11  # PUBLIC_THREAD

    def test_create_thread_with_message(
        self, rest_client: DiscordRestClient, mock_http_client: MagicMock
    ) -> None:
        mock_http_client.post.return_value.json.return_value = {"id": "500000000000000002"}
        result = rest_client.create_thread("100000000000000001", "Thread", message_id="200000000000000099")
        assert result["id"] == "500000000000000002"
        call_args = mock_http_client.post.call_args
        assert "/messages/200000000000000099/threads" in call_args[0][0]

    def test_create_thread_name_truncation(
        self, rest_client: DiscordRestClient, mock_http_client: MagicMock
    ) -> None:
        mock_http_client.post.return_value.json.return_value = {"id": "500000000000000003"}
        long_name = "x" * 200
        rest_client.create_thread("100000000000000001", long_name)
        call_args = mock_http_client.post.call_args
        assert len(call_args[1]["json"]["name"]) == 100

    def test_get_channel(self, rest_client: DiscordRestClient, mock_http_client: MagicMock) -> None:
        mock_http_client.get.return_value.json.return_value = {"id": "100000000000000001", "type": 0}
        result = rest_client.get_channel("100000000000000001")
        assert result["id"] == "100000000000000001"
        call_args = mock_http_client.get.call_args
        assert "/channels/100000000000000001" in call_args[0][0]
