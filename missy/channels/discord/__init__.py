"""Discord channel integration for the Missy framework.

Provides a full Discord bot channel implementation that connects via the
Discord Gateway WebSocket API and exposes Discord messages as
:class:`~missy.channels.base.ChannelMessage` instances.

All outbound HTTP requests to Discord's REST API are routed through
:class:`~missy.gateway.client.PolicyHTTPClient` so that network policy
is enforced on every call.

Minimal public exports — callers should import concrete classes directly::

    from missy.channels.discord.channel import DiscordChannel
    from missy.channels.discord.config import DiscordConfig, DiscordAccountConfig
"""

from __future__ import annotations

from missy.channels.discord.channel import DiscordChannel, DiscordSendError
from missy.channels.discord.config import (
    DiscordAccountConfig,
    DiscordConfig,
    DiscordDMPolicy,
    DiscordGuildPolicy,
)

__all__ = [
    "DiscordChannel",
    "DiscordSendError",
    "DiscordConfig",
    "DiscordAccountConfig",
    "DiscordDMPolicy",
    "DiscordGuildPolicy",
]
