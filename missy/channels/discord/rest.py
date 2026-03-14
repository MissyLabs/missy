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
import random
import re
import time
from typing import Any, Optional

from missy.gateway.client import PolicyHTTPClient, create_client

logger = logging.getLogger(__name__)

#: Discord REST API base URL.
BASE = "https://discord.com/api/v10"


_MENTION_ID_RE = re.compile(r"<@!?(?:\\d+)>|<@&(?:\\d+)>|<#(?:\\d+)>")


def _mask_mentions(s: str) -> str:
    """Redact snowflake IDs inside common mention tokens for safer logging."""
    return _MENTION_ID_RE.sub(lambda m: re.sub(r"\\d+", "redacted", m.group(0)), s or "")


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
        mention_user_ids: Optional[list[str]] = None,
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
            httpx.HTTPStatusError: On non-2xx responses (after retries).
            RuntimeError: If Discord returns a payload without a message id.
        """
        url = f"{BASE}/channels/{channel_id}/messages"
        body: dict[str, Any] = {
            "content": content,
            # Prevent Discord from parsing any mentions in outbound content
            # unless specific user IDs are explicitly allowlisted.
            "allowed_mentions": {"parse": [], "users": mention_user_ids or []},
        }
        if reply_to_message_id is not None:
            body["message_reference"] = {
                "message_id": reply_to_message_id,
                "fail_if_not_exists": False,
            }

        retry_statuses = {429, 502, 503, 504}
        backoffs = (1.0, 2.0, 4.0)
        attempt_count = len(backoffs) + 1

        def _log_final_failure(*, response: Any | None, exc: Exception | None, attempt_index: int) -> None:
            try:
                status_code = getattr(response, "status_code", None)
                response_text = ""
                if response is not None:
                    try:
                        response_text = getattr(response, "text", "") or ""
                    except Exception:
                        response_text = ""

                response_body = _mask_mentions(response_text)[:500]

                payload_preview = _mask_mentions(content)[:200]
                payload_len = len(content) if content is not None else 0

                logger.error(
                    "Discord send_message final failure (channel_id=%s attempt_count=%d status_code=%s payload_len=%d payload_preview=%r response_body=%r exc=%s)",
                    channel_id,
                    attempt_index,
                    status_code,
                    payload_len,
                    payload_preview,
                    response_body,
                    repr(exc) if exc else None,
                )
            except Exception:
                logger.exception("Discord send_message final failure logging failed")

        for attempt in range(attempt_count):
            response = None
            try:
                response = self._http.post(url, headers=self._headers(), json=body)

                if response.status_code in retry_statuses:
                    delay: Optional[float] = None
                    if response.status_code == 429:
                        ra = response.headers.get("Retry-After") if hasattr(response, "headers") else None
                        if ra:
                            try:
                                delay = float(ra)
                            except Exception:
                                delay = None
                    if delay is None:
                        if attempt >= len(backoffs):
                            response.raise_for_status()
                        delay = backoffs[attempt]

                    delay = float(delay) + random.uniform(0.0, 0.25)
                    logger.warning(
                        "Discord send_message transient HTTP %d; retrying in %.2fs (attempt %d/%d)",
                        response.status_code,
                        delay,
                        attempt + 1,
                        attempt_count,
                    )
                    time.sleep(delay)
                    continue

                response.raise_for_status()
                payload = response.json()
                msg_id = payload.get("id") if isinstance(payload, dict) else None
                if not msg_id:
                    raise RuntimeError(f"Discord send_message missing id in response: {payload!r}")
                return payload

            except Exception as exc:
                if attempt >= len(backoffs):
                    _log_final_failure(response=response, exc=exc, attempt_index=attempt_count)
                    raise
                delay = backoffs[attempt] + random.uniform(0.0, 0.25)
                logger.warning(
                    "Discord send_message exception; retrying in %.2fs (attempt %d/%d): %s",
                    delay,
                    attempt + 1,
                    attempt_count,
                    exc,
                )
                time.sleep(delay)

        raise RuntimeError("Discord send_message failed without exception")

    def upload_file(
        self,
        channel_id: str,
        file_path: str,
        caption: str = "",
    ) -> dict[str, Any]:
        """Upload a file to a Discord channel as an attachment.

        Args:
            channel_id: The target channel snowflake ID.
            file_path: Absolute path to the file to upload.
            caption: Optional message text to include with the file.

        Returns:
            The created Discord Message object as a dict.
        """
        import mimetypes
        from pathlib import Path

        import httpx

        path = Path(file_path).expanduser()
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        mime = mimetypes.guess_type(str(path))[0] or "application/octet-stream"
        url = f"{BASE}/channels/{channel_id}/messages"
        headers = {k: v for k, v in self._headers().items() if k != "Content-Type"}

        with path.open("rb") as fh:
            files = {"file": (path.name, fh, mime)}
            data = {"content": caption} if caption else {}
            response = httpx.post(url, headers=headers, files=files, data=data, timeout=60)
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
            emoji: Unicode emoji (e.g. ``"\u2705"``) or a custom emoji in
                ``name:id`` format.

        Raises:
            PolicyViolationError: If ``discord.com`` is not allowed.
            httpx.HTTPStatusError: On non-2xx responses.
        """
        from urllib.parse import quote

        import httpx

        encoded = quote(emoji, safe="")
        url = f"{BASE}/channels/{channel_id}/messages/{message_id}/reactions/{encoded}/@me"
        # Discord expects a PUT with empty body; returns 204 No Content.
        response = httpx.put(
            url,
            headers={k: v for k, v in self._headers().items() if k != "Content-Type"},
            timeout=10,
        )
        response.raise_for_status()

    def trigger_typing(self, channel_id: str) -> None:
        """Send a typing indicator to *channel_id*.

        Discord displays a "Bot is typing..." indicator for ~10 seconds.
        Call this before sending a long response to give users feedback.

        Args:
            channel_id: The channel snowflake ID.
        """
        try:
            self._http.post(
                f"{BASE}/channels/{channel_id}/typing",
                headers=self._headers(),
            )
        except Exception as exc:
            logger.debug("typing indicator failed for %s: %s", channel_id, exc)

    def delete_message(self, channel_id: str, message_id: str) -> bool:
        """Delete a message from a Discord channel.

        Used to remove messages that contain credentials or secrets detected
        by SecretsDetector to prevent them from sitting in chat history.

        Args:
            channel_id: The channel containing the message.
            message_id: The message snowflake ID to delete.

        Returns:
            True on success (HTTP 204), False if the message was not found
                (HTTP 404) or the bot lacks permissions (HTTP 403).

        Raises:
            PolicyViolationError: If discord.com is not in the network policy.
        """
        url = f"{BASE}/channels/{channel_id}/messages/{message_id}"
        try:
            import httpx

            response = httpx.delete(
                url,
                headers={k: v for k, v in self._headers().items() if k != "Content-Type"},
                timeout=10,
            )
            if response.status_code == 204:
                return True
            if response.status_code in (403, 404):
                logger.warning(
                    "Could not delete Discord message %s/%s: HTTP %d",
                    channel_id,
                    message_id,
                    response.status_code,
                )
                return False
            response.raise_for_status()
            return True
        except Exception as exc:
            logger.warning("delete_message failed for %s/%s: %s", channel_id, message_id, exc)
            return False

    def create_thread(
        self,
        channel_id: str,
        name: str,
        message_id: Optional[str] = None,
        auto_archive_duration: int = 1440,
    ) -> dict[str, Any]:
        """Create a new thread in a Discord channel.

        When *message_id* is provided, creates a thread attached to that
        message.  Otherwise creates a standalone thread (no starter message).

        Args:
            channel_id: Parent channel snowflake ID.
            name: Thread name (max 100 characters).
            message_id: Optional message to start the thread from.
            auto_archive_duration: Minutes of inactivity before auto-archive
                (60, 1440, 4320, or 10080).

        Returns:
            The created channel (thread) object as a dict.
        """
        if message_id:
            url = f"{BASE}/channels/{channel_id}/messages/{message_id}/threads"
            body: dict[str, Any] = {
                "name": name[:100],
                "auto_archive_duration": auto_archive_duration,
            }
        else:
            url = f"{BASE}/channels/{channel_id}/threads"
            body = {
                "name": name[:100],
                "auto_archive_duration": auto_archive_duration,
                "type": 11,  # PUBLIC_THREAD
            }
        response = self._http.post(url, headers=self._headers(), json=body)
        response.raise_for_status()
        return response.json()

    def get_channel(self, channel_id: str) -> dict[str, Any]:
        """Fetch a channel object by ID.

        Args:
            channel_id: The channel snowflake ID.

        Returns:
            The Discord Channel object as a dict.
        """
        url = f"{BASE}/channels/{channel_id}"
        response = self._http.get(url, headers=self._headers())
        response.raise_for_status()
        return response.json()

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

        # PUT = bulk overwrite; POST expects a single command object, not a list.
        response = self._http.put(url, headers=self._headers(), json=commands)
        response.raise_for_status()
        return response.json()
