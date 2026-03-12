"""Discord channel configuration dataclasses and enumerations.

Configuration is loaded from a ``discord:`` section in the Missy YAML file.
Bot tokens are **never** stored directly in the config; the ``token_env_var``
field names the environment variable that holds the token at runtime.

Example YAML::

    discord:
      enabled: true
      accounts:
        - token_env_var: DISCORD_BOT_TOKEN
          application_id: "1234567890"
          dm_policy: pairing
          guild_policies:
            "987654321":
              enabled: true
              require_mention: true
              allowed_channels: ["general", "bot-commands"]
              mode: full
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional


class DiscordDMPolicy(str, Enum):
    """Controls how the bot handles direct messages.

    Attributes:
        PAIRING: Users must complete a pairing handshake before the bot
            responds to their DMs.
        ALLOWLIST: Only users explicitly listed in ``dm_allowlist`` may
            send DMs to the bot.
        OPEN: Any user may send DMs without restriction.
        DISABLED: The bot ignores all direct messages.
    """

    PAIRING = "pairing"
    ALLOWLIST = "allowlist"
    OPEN = "open"
    DISABLED = "disabled"


@dataclass
class DiscordGuildPolicy:
    """Access control policy for a single Discord guild (server).

    Attributes:
        enabled: When ``False`` the bot ignores all messages from this guild.
        require_mention: When ``True`` the bot only responds if it is
            explicitly @-mentioned in the message.
        allowed_channels: Whitelist of channel names (not IDs) that the bot
            will respond in.  Empty list means all channels are permitted.
        allowed_roles: Whitelist of role names that users must hold to
            interact with the bot.  Empty means all roles are permitted.
        allowed_users: Whitelist of user IDs permitted to interact.  Empty
            means all users are permitted (subject to role rules).
        mode: Feature mode for this guild.  One of ``"safe_chat_only"``,
            ``"no_tools"``, or ``"full"``.
    """

    enabled: bool = True
    require_mention: bool = False
    allowed_channels: list[str] = field(default_factory=list)
    allowed_roles: list[str] = field(default_factory=list)
    allowed_users: list[str] = field(default_factory=list)
    mode: str = "full"


@dataclass
class DiscordAccountConfig:
    """Configuration for a single Discord bot account.

    The bot token is retrieved from an environment variable at runtime so
    that it is never hardcoded or committed to source control.

    Attributes:
        token_env_var: Name of the environment variable that contains the
            bot token (e.g. ``"DISCORD_BOT_TOKEN"``).
        account_id: Optional explicit bot user ID.  When omitted the ID is
            fetched from Discord on startup via ``GET /users/@me``.
        application_id: Discord application ID (required for slash command
            registration).
        guild_policies: Mapping of guild ID strings to
            :class:`DiscordGuildPolicy` instances.
        dm_policy: Controls how the bot handles direct messages.
        dm_allowlist: Explicit user IDs permitted for DM when
            ``dm_policy == DiscordDMPolicy.ALLOWLIST``.
        ack_reaction: Emoji used to acknowledge receipt of a message
            (e.g. ``"eyes"``).  Set to an empty string to disable.
        ignore_bots: When ``True`` the bot ignores messages from other bots.
        allow_bots_if_mention_only: When ``True`` and ``ignore_bots`` is
            ``True``, bot messages that explicitly @-mention this bot are
            not ignored.
    """

    token_env_var: str = "DISCORD_BOT_TOKEN"
    token: Optional[str] = None          # direct token (takes precedence over token_env_var)
    account_id: Optional[str] = None
    application_id: str = ""
    guild_policies: dict[str, DiscordGuildPolicy] = field(default_factory=dict)
    dm_policy: DiscordDMPolicy = DiscordDMPolicy.DISABLED
    dm_allowlist: list[str] = field(default_factory=list)
    ack_reaction: str = ""
    ignore_bots: bool = True
    allow_bots_if_mention_only: bool = False
    auto_thread_threshold: int = 0  # 0 = disabled; N = create thread after N messages

    def resolve_token(self) -> Optional[str]:
        """Return the bot token — checks direct token, env var, and vault in order.

        Returns:
            The token string, or ``None`` when none are configured.
        """
        if self.token:
            # Resolve vault:// references
            if self.token.startswith("vault://"):
                try:
                    from missy.security.vault import Vault
                    key = self.token[len("vault://"):]
                    return Vault().get(key)
                except Exception:
                    pass
            return self.token
        return os.environ.get(self.token_env_var)


@dataclass
class DiscordConfig:
    """Top-level Discord integration configuration.

    Attributes:
        accounts: List of Discord bot account configurations.  Most
            deployments will have exactly one entry.
        enabled: Master switch.  When ``False`` the Discord channel is not
            started even if ``accounts`` is populated.
    """

    accounts: list[DiscordAccountConfig] = field(default_factory=list)
    enabled: bool = False


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------


def _parse_guild_policy(data: dict[str, Any]) -> DiscordGuildPolicy:
    """Construct a :class:`DiscordGuildPolicy` from a raw YAML dict."""
    return DiscordGuildPolicy(
        enabled=bool(data.get("enabled", True)),
        require_mention=bool(data.get("require_mention", False)),
        allowed_channels=list(data.get("allowed_channels", [])),
        allowed_roles=list(data.get("allowed_roles", [])),
        allowed_users=list(data.get("allowed_users", [])),
        mode=str(data.get("mode", "full")),
    )


def _parse_account(data: dict[str, Any]) -> DiscordAccountConfig:
    """Construct a :class:`DiscordAccountConfig` from a raw YAML dict."""
    raw_dm_policy = data.get("dm_policy", "disabled")
    try:
        dm_policy = DiscordDMPolicy(raw_dm_policy)
    except ValueError:
        dm_policy = DiscordDMPolicy.DISABLED

    raw_guild_policies = data.get("guild_policies") or {}
    guild_policies: dict[str, DiscordGuildPolicy] = {}
    for guild_id, gp_data in raw_guild_policies.items():
        if isinstance(gp_data, dict):
            guild_policies[str(guild_id)] = _parse_guild_policy(gp_data)

    return DiscordAccountConfig(
        token_env_var=str(data.get("token_env_var", "DISCORD_BOT_TOKEN")),
        token=data.get("token") or None,
        account_id=data.get("account_id") or None,
        application_id=str(data.get("application_id", "")),
        guild_policies=guild_policies,
        dm_policy=dm_policy,
        dm_allowlist=list(data.get("dm_allowlist", [])),
        ack_reaction=str(data.get("ack_reaction", "")),
        ignore_bots=bool(data.get("ignore_bots", True)),
        allow_bots_if_mention_only=bool(data.get("allow_bots_if_mention_only", False)),
        auto_thread_threshold=int(data.get("auto_thread_threshold", 0)),
    )


def parse_discord_config(data: dict[str, Any]) -> DiscordConfig:
    """Parse a ``discord:`` YAML section into a :class:`DiscordConfig`.

    Args:
        data: The raw dict from the YAML ``discord:`` key.

    Returns:
        A populated :class:`DiscordConfig`.
    """
    if not isinstance(data, dict):
        return DiscordConfig()

    raw_accounts = data.get("accounts") or []
    accounts = [_parse_account(a) for a in raw_accounts if isinstance(a, dict)]

    return DiscordConfig(
        accounts=accounts,
        enabled=bool(data.get("enabled", False)),
    )
