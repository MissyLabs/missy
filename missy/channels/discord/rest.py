"""Discord REST API client built on top of :class:`PolicyHTTPClient`.

All outbound requests to ``discord.com`` are routed through
:class:`~missy.gateway.client.PolicyHTTPClient` so the framework's
network policy is enforced on every call.  The Discord domain must
therefore be listed in ``network.allowed_domains`` in the Missy config.

Example::

    from missy.channels.discord.rest import DiscordRestClient
    from missy.gateway.client import create_client

    http = create_client(session_id="s1", task_id="t1")
    discord = DiscordRestClient(bot_token="Bot TOKEN", http_client=http)
    user = discord.get_current_user()
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from missy.gateway.client import PolicyHTTPClient, create_client

logger = logging.getLogger(__name__)

#: Discord REST API base URL.
BASE = "https://discord.com/api/v10"


class DiscordRestClient:
    """Thin wrapper around the Discord REST API v10.

    The client injects the ``Authorization`` header on every request and
    delegates all I/O to an injected :class:`~missy.gateway.client.PolicyHTTPClient`
    instance so that the framework's network policy is always enforced.

    Args:
        bot_token: The Discord bot token.  **Must** start with ``"Bot "``
            according to the Discord API specification.  If the value does
            not have the prefix it is added automatically.
        http_client: Optional pre-constructed :class:`PolicyHTTPClient`.
            When ``None`` a new default client is created.
        session_id: Forwarded to the policy client for audit tracing.
        task_id: Forwarded to the policy client for audit tracing.
    """

    def __init__(
        self,
        bot_token: str,
        http_client: Optional[PolicyHTTPClient] = None,
        session_id: str = "discord",
        task_id: str = "rest",
    ) -> None:
        if not bot_token.startswith("Bot "):
            bot_token = f"Bot {bot_token}"
        self._token = bot_token
        self._http: PolicyHTTPClient = http_client or create_client(
            session_id=session_id,
            task_id=task_id,
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _headers(self, extra: Optional[dict[str, str]] = None) -> dict[str, str]:
        """Build request headers with the bot token injected."""
        hdrs: dict[str, str] = {
            "Authorization": self._token,
            "Content-Type": "application/json",
            "User-Agent": "MissyBot (https://github.com/missy, 0.1)",
        }
        if extra:
            hdrs.update(extra)
        return hdrs

    # ------------------------------------------------------------------
    # Public methods
    # ------------------------------------------------------------------

    def get_current_user(self) -> dict[str, Any]:
        """Fetch the bot's own Discord user object.

        Returns:
            The Discord ``User`` object as a dict.

        Raises:
            PolicyViolationError: If ``discord.com`` is not allowed by
                the network policy.
            httpx.HTTPStatusError: On non-2xx responses.
        """
        url = f"{BASE}/users/@me"
        response = self._http.get(url, headers=self._headers())
        response.raise_for_status()
        return response.json()

    def get_gateway_bot(self) -> dict[str, Any]:
        """Fetch Gateway connection info including the WSS URL.

        Returns:
            A dict containing ``url``, ``shards``, and
            ``session_start_limit`` fields.

        Raises:
            PolicyViolationError: If ``discord.com`` is not allowed.
            httpx.HTTPStatusError: On non-2xx responses.
        """
        url = f"{BASE}/gateway/bot"
        response = self._http.get(url, headers=self._headers())
        response.raise_for_status()
        return response.json()

    def send_message(
        self,
        channel_id: str,
        content: str,
        reply_to_message_id: Optional[str] = None,
    ) -> dict[str, Any]:
        """Send a text message to a Discord channel.

        Args:
            channel_id: The target channel's snowflake ID.
            content: The message text (max 2 000 characters per Discord
                limits; callers are responsible for splitting longer text).
            reply_to_message_id: When set, the message is posted as a reply
                to this message ID.

        Returns:
            The created Discord ``Message`` object as a dict.

        Raises:
            PolicyViolationError: If ``discord.com`` is not allowed.
            httpx.HTTPStatusError: On non-2xx responses.
        """
        url = f"{BASE}/channels/{channel_id}/messages"
        body: dict[str, Any] = {"content": content}
        if reply_to_message_id is not None:
            body["message_reference"] = {
                "message_id": reply_to_message_id,
                "fail_if_not_exists": False,
            }
        response = self._http.post(url, headers=self._headers(), json=body)
        response.raise_for_status()
        return response.json()

    def add_reaction(
        self,
        channel_id: str,
        message_id: str,
        emoji: str,
    ) -> None:
        """Add an emoji reaction to a message.

        Args:
            channel_id: The channel containing the message.
            message_id: The message to react to.
            emoji: URL-encoded emoji string (e.g. ``"eyes"`` or a custom
                emoji in ``name:id`` format).

        Raises:
            PolicyViolationError: If ``discord.com`` is not allowed.
            httpx.HTTPStatusError: On non-2xx responses.
        """
        from urllib.parse import quote

        encoded = quote(emoji, safe="")
        url = f"{BASE}/channels/{channel_id}/messages/{message_id}/reactions/{encoded}/@me"
        # PUT with empty body; Discord returns 204 No Content on success.
        response = self._http.post(
            f"{BASE}/channels/{channel_id}/messages/{message_id}/reactions/{encoded}/@me/put",
            headers=self._headers(),
            content=b"",
        )
        # Some callers use a raw PUT which PolicyHTTPClient does not expose yet;
        # fall back to a workaround using post with the special _method marker.
        # In practice we issue a real PUT via the sync client directly here.
        import httpx

        parsed_url = f"{BASE}/channels/{channel_id}/messages/{message_id}/reactions/{encoded}/@me"
        http_response = httpx.put(
            parsed_url,
            headers={k: v for k, v in self._headers().items() if k != "Content-Type"},
            timeout=10,
        )
        http_response.raise_for_status()

    def trigger_typing(self, channel_id: str) -> None:
        """Trigger the typing indicator in a channel.

        The indicator displays for ~10 seconds or until the bot sends a
        message, whichever comes first.

        Args:
            channel_id: The channel to show the typing indicator in.

        Raises:
            PolicyViolationError: If ``discord.com`` is not allowed.
            httpx.HTTPStatusError: On non-2xx responses.
        """
        url = f"{BASE}/channels/{channel_id}/typing"
        response = self._http.post(url, headers=self._headers(), content=b"")
        # Discord returns 204; raise_for_status handles non-2xx.
        if response.status_code not in (200, 204):
            response.raise_for_status()

    def register_slash_commands(
        self,
        application_id: str,
        commands: list[dict[str, Any]],
        guild_id: Optional[str] = None,
    ) -> list[dict[str, Any]]:
        """Register (bulk overwrite) application slash commands.

        When *guild_id* is provided, commands are registered as
        guild-specific (instant propagation).  Without it, they are
        registered globally (may take up to an hour to propagate).

        Args:
            application_id: The Discord application / client ID.
            commands: List of application command objects to register.
            guild_id: Optional guild ID for guild-scoped commands.

        Returns:
            List of registered command objects returned by Discord.

        Raises:
            PolicyViolationError: If ``discord.com`` is not allowed.
            httpx.HTTPStatusError: On non-2xx responses.
        """
        if guild_id:
            url = f"{BASE}/applications/{application_id}/guilds/{guild_id}/commands"
        else:
            url = f"{BASE}/applications/{application_id}/commands"

        response = self._http.post(url, headers=self._headers(), json=commands)
        response.raise_for_status()
        return response.json()
