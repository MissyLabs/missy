"""Unit tests for missy.channels.discord.config."""

from __future__ import annotations

import pytest

from missy.channels.discord.config import (
    DiscordAccountConfig,
    DiscordConfig,
    DiscordDMPolicy,
    DiscordGuildPolicy,
    _parse_account,
    _parse_guild_policy,
    parse_discord_config,
)

# ---------------------------------------------------------------------------
# DiscordDMPolicy enum
# ---------------------------------------------------------------------------


class TestDiscordDMPolicy:
    def test_values_exist(self) -> None:
        assert DiscordDMPolicy.PAIRING.value == "pairing"
        assert DiscordDMPolicy.ALLOWLIST.value == "allowlist"
        assert DiscordDMPolicy.OPEN.value == "open"
        assert DiscordDMPolicy.DISABLED.value == "disabled"

    def test_is_string_enum(self) -> None:
        assert DiscordDMPolicy.OPEN == "open"
        assert DiscordDMPolicy.DISABLED == "disabled"

    def test_from_string(self) -> None:
        assert DiscordDMPolicy("pairing") is DiscordDMPolicy.PAIRING
        assert DiscordDMPolicy("open") is DiscordDMPolicy.OPEN

    def test_invalid_value_raises(self) -> None:
        with pytest.raises(ValueError):
            DiscordDMPolicy("unknown")


# ---------------------------------------------------------------------------
# DiscordGuildPolicy defaults
# ---------------------------------------------------------------------------


class TestDiscordGuildPolicy:
    def test_defaults(self) -> None:
        gp = DiscordGuildPolicy()
        assert gp.enabled is True
        assert gp.require_mention is False
        assert gp.allowed_channels == []
        assert gp.allowed_roles == []
        assert gp.allowed_users == []
        assert gp.mode == "full"

    def test_custom_values(self) -> None:
        gp = DiscordGuildPolicy(
            enabled=False,
            require_mention=True,
            allowed_channels=["general"],
            allowed_roles=["admin"],
            allowed_users=["123"],
            mode="no_tools",
        )
        assert gp.enabled is False
        assert gp.require_mention is True
        assert gp.allowed_channels == ["general"]
        assert gp.mode == "no_tools"

    def test_parse_from_dict(self) -> None:
        data = {
            "enabled": True,
            "require_mention": True,
            "allowed_channels": ["bot-commands"],
            "allowed_roles": [],
            "allowed_users": ["999"],
            "mode": "safe_chat_only",
        }
        gp = _parse_guild_policy(data)
        assert gp.require_mention is True
        assert gp.allowed_channels == ["bot-commands"]
        assert gp.allowed_users == ["999"]
        assert gp.mode == "safe_chat_only"

    def test_parse_partial_dict(self) -> None:
        gp = _parse_guild_policy({})
        assert gp.enabled is True
        assert gp.mode == "full"


# ---------------------------------------------------------------------------
# DiscordAccountConfig defaults
# ---------------------------------------------------------------------------


class TestDiscordAccountConfig:
    def test_defaults(self) -> None:
        ac = DiscordAccountConfig()
        assert ac.token_env_var == "DISCORD_BOT_TOKEN"
        assert ac.account_id is None
        assert ac.application_id == ""
        assert ac.guild_policies == {}
        assert ac.dm_policy is DiscordDMPolicy.DISABLED
        assert ac.dm_allowlist == []
        assert ac.ack_reaction == ""
        assert ac.ignore_bots is True
        assert ac.allow_bots_if_mention_only is False

    def test_resolve_token_present(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("TEST_DISCORD_TOKEN", "my-secret-token")
        ac = DiscordAccountConfig(token_env_var="TEST_DISCORD_TOKEN")
        assert ac.resolve_token() == "my-secret-token"

    def test_resolve_token_missing(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("MISSING_DISCORD_TOKEN", raising=False)
        ac = DiscordAccountConfig(token_env_var="MISSING_DISCORD_TOKEN")
        assert ac.resolve_token() is None

    def test_parse_account_full(self) -> None:
        data = {
            "token_env_var": "MY_BOT_TOKEN",
            "account_id": "111",
            "application_id": "222",
            "dm_policy": "allowlist",
            "dm_allowlist": ["333", "444"],
            "ack_reaction": "eyes",
            "ignore_bots": False,
            "allow_bots_if_mention_only": True,
            "guild_policies": {
                "555": {
                    "enabled": True,
                    "require_mention": True,
                    "allowed_channels": ["general"],
                    "mode": "no_tools",
                }
            },
        }
        ac = _parse_account(data)
        assert ac.token_env_var == "MY_BOT_TOKEN"
        assert ac.account_id == "111"
        assert ac.application_id == "222"
        assert ac.dm_policy is DiscordDMPolicy.ALLOWLIST
        assert ac.dm_allowlist == ["333", "444"]
        assert ac.ack_reaction == "eyes"
        assert ac.ignore_bots is False
        assert ac.allow_bots_if_mention_only is True
        assert "555" in ac.guild_policies
        assert ac.guild_policies["555"].require_mention is True

    def test_parse_account_invalid_dm_policy_defaults_disabled(self) -> None:
        data = {"dm_policy": "nonsense_value"}
        ac = _parse_account(data)
        assert ac.dm_policy is DiscordDMPolicy.DISABLED

    def test_parse_account_minimal(self) -> None:
        ac = _parse_account({})
        assert ac.token_env_var == "DISCORD_BOT_TOKEN"
        assert ac.dm_policy is DiscordDMPolicy.DISABLED
        assert ac.guild_policies == {}


# ---------------------------------------------------------------------------
# DiscordConfig / parse_discord_config
# ---------------------------------------------------------------------------


class TestDiscordConfig:
    def test_defaults(self) -> None:
        dc = DiscordConfig()
        assert dc.accounts == []
        assert dc.enabled is False

    def test_parse_empty_dict(self) -> None:
        dc = parse_discord_config({})
        assert dc.enabled is False
        assert dc.accounts == []

    def test_parse_non_dict_returns_empty(self) -> None:
        dc = parse_discord_config(None)  # type: ignore[arg-type]
        assert dc.enabled is False
        assert dc.accounts == []

    def test_parse_enabled_with_accounts(self) -> None:
        data = {
            "enabled": True,
            "accounts": [
                {
                    "token_env_var": "BOT_TOKEN_A",
                    "application_id": "100",
                    "dm_policy": "open",
                },
                {
                    "token_env_var": "BOT_TOKEN_B",
                    "dm_policy": "pairing",
                },
            ],
        }
        dc = parse_discord_config(data)
        assert dc.enabled is True
        assert len(dc.accounts) == 2
        assert dc.accounts[0].token_env_var == "BOT_TOKEN_A"
        assert dc.accounts[0].dm_policy is DiscordDMPolicy.OPEN
        assert dc.accounts[1].dm_policy is DiscordDMPolicy.PAIRING

    def test_parse_skips_non_dict_accounts(self) -> None:
        data = {
            "enabled": True,
            "accounts": [
                "not-a-dict",
                {"token_env_var": "GOOD_TOKEN"},
            ],
        }
        dc = parse_discord_config(data)
        # "not-a-dict" is skipped; only the valid dict is parsed.
        assert len(dc.accounts) == 1
        assert dc.accounts[0].token_env_var == "GOOD_TOKEN"

    def test_parse_guild_policies_nested(self) -> None:
        data = {
            "enabled": True,
            "accounts": [
                {
                    "token_env_var": "BOT_TOKEN",
                    "guild_policies": {
                        "guild123": {
                            "enabled": True,
                            "require_mention": True,
                            "allowed_channels": ["bot"],
                            "mode": "full",
                        }
                    },
                }
            ],
        }
        dc = parse_discord_config(data)
        account = dc.accounts[0]
        assert "guild123" in account.guild_policies
        gp = account.guild_policies["guild123"]
        assert gp.require_mention is True
        assert gp.allowed_channels == ["bot"]


# ---------------------------------------------------------------------------
# Token resolution from environment variables
# ---------------------------------------------------------------------------


class TestTokenResolution:
    def test_token_from_custom_env_var(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("MY_CUSTOM_BOT_TOKEN", "secret-abc-123")
        ac = DiscordAccountConfig(token_env_var="MY_CUSTOM_BOT_TOKEN")
        assert ac.resolve_token() == "secret-abc-123"

    def test_token_not_set(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("DISCORD_BOT_TOKEN", raising=False)
        ac = DiscordAccountConfig()
        assert ac.resolve_token() is None

    def test_token_env_var_is_not_stored_as_value(self) -> None:
        """Verify the dataclass stores the variable name, not the resolved secret."""
        ac = DiscordAccountConfig(token_env_var="SOME_SECRET_VAR")
        assert ac.token_env_var == "SOME_SECRET_VAR"
        # The token field should be None (not pre-resolved from env).
        assert ac.token is None

    def test_multiple_accounts_different_env_vars(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("BOT_A", "token-a")
        monkeypatch.setenv("BOT_B", "token-b")
        a = DiscordAccountConfig(token_env_var="BOT_A")
        b = DiscordAccountConfig(token_env_var="BOT_B")
        assert a.resolve_token() == "token-a"
        assert b.resolve_token() == "token-b"
