"""Discord channel implementation.

:class:`DiscordChannel` wires together the Gateway client, REST client,
access-control logic, and pairing workflow into a :class:`BaseChannel`
that the Missy agent framework can use like any other channel.

Access-control pipeline (evaluated in order):

1. Filter own-bot messages by comparing ``author.id`` to :attr:`bot_user_id`.
2. If the message author is a bot and ``ignore_bots`` is enabled, check
   ``allow_bots_if_mention_only`` before deciding whether to drop.
3. For DMs (``guild_id`` absent): apply :class:`~.config.DiscordDMPolicy`.
4. For guild messages: look up :class:`~.config.DiscordGuildPolicy` and
   apply channel allowlist, user allowlist, role allowlist, and mention
   requirement.

Audit events are emitted for every allow/deny decision.

Example::

    import asyncio
    from missy.channels.discord.channel import DiscordChannel
    from missy.channels.discord.config import DiscordAccountConfig, DiscordDMPolicy

    account_cfg = DiscordAccountConfig(
        token_env_var="DISCORD_BOT_TOKEN",
        dm_policy=DiscordDMPolicy.OPEN,
    )
    channel = DiscordChannel(account_config=account_cfg)
    asyncio.run(channel.start())
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import time
from typing import Any

from missy.channels.base import BaseChannel, ChannelMessage
from missy.channels.discord.config import DiscordAccountConfig, DiscordDMPolicy
from missy.channels.discord.gateway import DiscordGatewayClient
from missy.channels.discord.rest import DiscordRestClient
from missy.core.events import AuditEvent, event_bus

logger = logging.getLogger(__name__)

#: How long a guild's resolved role-ID-to-name map stays cached before
#: being re-fetched from Discord's REST API (allowed_roles enforcement).
_GUILD_ROLES_CACHE_TTL_SECONDS = 300.0


class DiscordSendError(Exception):
    """Raised when a Discord message could not be delivered."""

    def __init__(
        self,
        message: str,
        *,
        channel_id: str = "",
        original_error: Exception | None = None,
    ) -> None:
        super().__init__(message)
        self.channel_id = channel_id
        self.original_error = original_error


class DiscordChannel(BaseChannel):
    """Missy channel backed by the Discord Gateway WebSocket API.

    Args:
        account_config: The :class:`DiscordAccountConfig` that governs this
            channel's behaviour.
        session_id: Forwarded to audit events and the HTTP client.
        task_id: Forwarded to audit events and the HTTP client.
        queue_max: Maximum number of pending inbound messages.
    """

    name = "discord"

    def __init__(
        self,
        account_config: DiscordAccountConfig,
        session_id: str = "discord",
        task_id: str = "channel",
        queue_max: int = 256,
    ) -> None:
        self.account_config = account_config
        self._session_id = session_id
        self._task_id = task_id
        self._queue: asyncio.Queue[ChannelMessage] = asyncio.Queue(maxsize=queue_max)
        self._current_channel_id: str | None = None

        # Pairing state: set of Discord user IDs that have initiated pairing
        # but not yet been confirmed.
        self._pending_pairs: set[str] = set()

        # Resolved bot user ID (populated after READY).
        self._bot_user_id: str | None = None

        # Thread-scoped session mapping: thread_id -> session_id
        # Enables conversation continuity within Discord threads.
        self._thread_sessions: dict[str, str] = {}

        # Thread-to-parent-channel mapping: thread_id -> parent_channel_id.
        # A Gateway MESSAGE_CREATE for a message posted inside a thread
        # carries channel_id = the thread's own snowflake, never the
        # parent's -- but guild_policy.allowed_channels is naturally
        # configured with parent-channel IDs/names (operators can't know a
        # dynamically-created thread's ID ahead of time). Without this,
        # any message inside a thread this bot created (via
        # auto_thread_threshold) is silently denied by the channel
        # allowlist forever, even though its parent channel is allowed.
        # Populated only for threads this bot itself creates (see
        # create_thread() below); a thread created by a Discord user
        # directly is not covered without also handling the Gateway
        # THREAD_CREATE event, which is a larger, separate effort.
        self._thread_parents: dict[str, str] = {}

        # Message count per channel for auto-thread creation.
        self._channel_message_counts: dict[str, int] = {}

        # Auto-thread threshold: create a thread after N messages in a channel.
        self._auto_thread_threshold: int = getattr(account_config, "auto_thread_threshold", 0)

        # Pending evolution reactions: message_id -> proposal_id
        self._pending_evolutions: dict[str, str] = {}

        # allowed_roles enforcement: cache of guild_id -> (fetched_at,
        # {role_id: role_name}), refreshed via the REST API on a TTL so
        # every message doesn't need its own round trip to Discord.
        self._guild_roles_cache: dict[str, tuple[float, dict[str, str]]] = {}

        # Optional voice manager (lazy import so text-only deployments don't need voice deps)
        self._voice = None

        # Optional screencast channel reference — set via set_screencast() from main.py
        self._screencast: Any = None

        # Agent runtime reference — set via set_agent_runtime() from main.py
        # so voice can call the agent for conversational responses.
        self._agent_runtime: Any = None

        # DISC-CMD-008: per-user command rate limiting. Checked before any
        # command-producing dispatch (slash interaction or natural-language
        # message) so a single user can't spam paid LLM calls unbounded --
        # previously only the overall session CostTracker budget backstopped
        # this, with no per-user throttle at all.
        from missy.channels.discord.rate_limit import DiscordUserRateLimiter

        self._rate_limiter = DiscordUserRateLimiter(
            requests_per_minute=getattr(account_config, "rate_limit_per_minute", 10)
        )

        token = account_config.resolve_token() or ""
        if not token:
            logger.error(
                "Discord: no bot token found — set env var %r or add 'token:' to config",
                account_config.token_env_var,
            )
        else:
            logger.info("Discord: bot token resolved (length=%d)", len(token))
        self._rest = DiscordRestClient(
            bot_token=token,
            session_id=session_id,
            task_id=task_id,
        )
        self._gateway = DiscordGatewayClient(
            bot_token=token,
            on_message=self._on_gateway_event,
            session_id=session_id,
            task_id=task_id,
        )
        self._slash_registration_status: dict[str, Any] = {
            "attempted": False,
            "ok": None,
            "scope": None,
            "command_count": 0,
            "error": None,
        }

        self._gateway_task: asyncio.Task[None] | None = None

    # ------------------------------------------------------------------
    # Public properties
    # ------------------------------------------------------------------

    @property
    def bot_user_id(self) -> str | None:
        """The Discord user ID of the connected bot, available after READY."""
        return self._bot_user_id or self._gateway.bot_user_id

    def get_diagnostics(self) -> dict[str, Any]:
        """Return redacted channel lifecycle diagnostics."""
        return {
            "gateway": self._gateway.get_diagnostics(),
            "slash_registration": dict(self._slash_registration_status),
        }

    def set_agent_runtime(self, agent_runtime: Any) -> None:
        """Provide the agent runtime for voice conversation support."""
        self._agent_runtime = agent_runtime

    def set_screencast(self, screencast: Any) -> None:
        """Provide the screencast channel for !screen command support."""
        self._screencast = screencast

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """Start the Discord Gateway connection and register slash commands.

        This method is non-blocking: the Gateway runs as a background
        asyncio task.
        """
        self._gateway_task = asyncio.create_task(self._gateway.run())

        # Register global slash commands if an application ID is configured.
        if self.account_config.application_id:
            try:
                from missy.channels.discord.commands import SLASH_COMMANDS

                self._rest.register_slash_commands(
                    application_id=self.account_config.application_id,
                    commands=SLASH_COMMANDS,
                )
                self._slash_registration_status = {
                    "attempted": True,
                    "ok": True,
                    "scope": "global",
                    "command_count": len(SLASH_COMMANDS),
                    "error": None,
                }
                self._emit_audit(
                    "discord.slash_commands.registered",
                    "allow",
                    {
                        "scope": "global",
                        "command_count": len(SLASH_COMMANDS),
                    },
                )
                logger.info("Discord: slash commands registered globally")
            except Exception as exc:
                self._slash_registration_status = {
                    "attempted": True,
                    "ok": False,
                    "scope": "global",
                    "command_count": 0,
                    "error": str(exc),
                }
                self._emit_audit(
                    "discord.slash_commands.registration_failed",
                    "error",
                    {"scope": "global", "error": str(exc)},
                )
                logger.warning("Discord: slash command registration failed: %s", exc)

    async def stop(self) -> None:
        """Disconnect from the Gateway and cancel the background task."""
        if self._voice is not None:
            with contextlib.suppress(Exception):
                await self._voice.stop()
            with contextlib.suppress(Exception):
                from missy.channels.discord.voice_binding import clear_voice_binding

                clear_voice_binding(manager=self._voice)
            self._voice = None
        with contextlib.suppress(Exception):
            from missy.channels.discord.voice_binding import clear_voice_binding

            clear_voice_binding(account_id=self._voice_account_id())
        await self._gateway.disconnect()
        if self._gateway_task is not None:
            self._gateway_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._gateway_task
            self._gateway_task = None

    # ------------------------------------------------------------------
    # BaseChannel interface
    # ------------------------------------------------------------------

    def receive(self) -> ChannelMessage | None:
        """Raise :exc:`NotImplementedError` — Discord is async-only.

        Use :meth:`areceive` or the internal queue directly when building
        an async consumer loop.

        Raises:
            NotImplementedError: Always.
        """
        raise NotImplementedError(
            "DiscordChannel is async-only. "
            "Use 'await channel.areceive()' or consume the asyncio.Queue directly."
        )

    async def areceive(self) -> ChannelMessage | None:
        """Await the next inbound :class:`ChannelMessage` from Discord.

        Returns:
            The next message from the internal queue, or ``None`` when
            the channel is stopped.
        """
        return await self._queue.get()

    def send(self, message: str) -> None:
        """Send *message* to the most recently active channel (sync stub).

        Because Discord I/O is async this method creates a fire-and-forget
        coroutine via the running event loop.  If no channel context has
        been established yet, the call is silently dropped.

        Args:
            message: The text to send.
        """
        if self._current_channel_id is None:
            logger.warning("DiscordChannel.send(): no current channel context — message dropped")
            return
        try:
            loop = asyncio.get_running_loop()
            loop.create_task(self.send_with_retry(self._current_channel_id, message))
        except RuntimeError:
            logger.warning("DiscordChannel.send(): no running event loop — message dropped")

    async def send_to(
        self,
        channel_id: str,
        message: str,
        reply_to: str | None = None,
        thread_id: str | None = None,
        mention_user_ids: list[str] | None = None,
    ) -> str:
        """Send *message* to a specific Discord channel asynchronously.

        Args:
            channel_id: Discord channel snowflake ID.
            message: Text to send (max 2 000 characters).
            reply_to: Optional message ID to reply to.
            thread_id: Optional thread snowflake ID; when set, the message
                is sent to the thread rather than the parent channel.
            mention_user_ids: Optional list of user IDs to allow mentions for.

        Returns:
            The snowflake ID of the last message sent.

        Raises:
            DiscordSendError: If the message could not be delivered.
        """
        # Send typing indicator as an in-progress UX signal.
        with contextlib.suppress(Exception):
            self._rest.trigger_typing(channel_id)

        target_channel = thread_id if thread_id else channel_id

        # Discord hard limit is 2000 characters per message. Split if needed.
        _DISCORD_MAX = 1990
        chunks = [
            message[i : i + _DISCORD_MAX] for i in range(0, max(len(message), 1), _DISCORD_MAX)
        ]

        last_message_id: str | None = None
        try:
            for idx, chunk in enumerate(chunks):
                result = self._rest.send_message(
                    channel_id=target_channel,
                    content=chunk,
                    reply_to_message_id=reply_to if idx == 0 else None,
                    mention_user_ids=mention_user_ids,
                )
                last_message_id = str(result.get("id", "")) if result else None
            self._emit_audit(
                "discord.channel.reply_sent",
                "allow",
                {"channel_id": channel_id, "reply_to": reply_to},
            )
        except Exception as exc:
            logger.error("Discord send_to failed: %s", exc)
            self._emit_audit(
                "discord.channel.reply_sent",
                "error",
                {"channel_id": channel_id, "error": str(exc)},
            )
            raise DiscordSendError(
                f"Failed to send message to channel {channel_id}: {exc}",
                channel_id=channel_id,
                original_error=exc,
            ) from exc
        if not last_message_id:
            raise DiscordSendError(
                f"Discord returned no message ID for channel {channel_id}",
                channel_id=channel_id,
            )
        return last_message_id

    async def send_with_retry(
        self,
        channel_id: str,
        message: str,
        reply_to: str | None = None,
        thread_id: str | None = None,
        mention_user_ids: list[str] | None = None,
        max_attempts: int = 6,
        max_total_seconds: float = 300.0,
    ) -> str:
        """Send a message with exponential backoff retry on failure.

        Retries with delays of 2s, 4s, 8s, 16s, 32s, … (capped at 60s per
        wait) until the message is delivered or *max_total_seconds* (default
        5 minutes) has elapsed.

        Args:
            channel_id: Discord channel snowflake ID.
            message: Text to send.
            reply_to: Optional message ID to reply to.
            thread_id: Optional thread ID.
            mention_user_ids: Optional user IDs to allow mentions for.
            max_attempts: Maximum number of send attempts (default 6).
            max_total_seconds: Give up after this many seconds total
                (default 300 = 5 minutes).

        Returns:
            The snowflake ID of the last message sent.

        Raises:
            DiscordSendError: If all retry attempts are exhausted.
        """
        import time as _time

        start = _time.monotonic()
        base_delay = 2.0
        last_error: Exception | None = None

        for attempt in range(1, max_attempts + 1):
            elapsed = _time.monotonic() - start
            if attempt > 1 and elapsed >= max_total_seconds:
                break

            try:
                msg_id = await self.send_to(
                    channel_id=channel_id,
                    message=message,
                    reply_to=reply_to,
                    thread_id=thread_id,
                    mention_user_ids=mention_user_ids,
                )
                if attempt > 1:
                    logger.info(
                        "Discord send_with_retry succeeded on attempt %d/%d "
                        "after %.1fs (channel=%s)",
                        attempt,
                        max_attempts,
                        _time.monotonic() - start,
                        channel_id,
                    )
                return msg_id
            except DiscordSendError as exc:
                last_error = exc
                delay = min(base_delay * (2 ** (attempt - 1)), 60.0)
                remaining = max_total_seconds - (_time.monotonic() - start)
                if attempt >= max_attempts or remaining <= 0:
                    break
                delay = min(delay, remaining)
                logger.warning(
                    "Discord send_with_retry attempt %d/%d failed "
                    "(channel=%s); retrying in %.1fs: %s",
                    attempt,
                    max_attempts,
                    channel_id,
                    delay,
                    exc,
                )
                self._emit_audit(
                    "discord.channel.send_retry",
                    "retry",
                    {
                        "channel_id": channel_id,
                        "attempt": attempt,
                        "max_attempts": max_attempts,
                        "delay_seconds": delay,
                        "error": str(exc),
                    },
                )
                await asyncio.sleep(delay)

        total_elapsed = _time.monotonic() - start
        logger.error(
            "Discord send_with_retry exhausted all %d attempts over %.1fs (channel=%s)",
            max_attempts,
            total_elapsed,
            channel_id,
        )
        self._emit_audit(
            "discord.channel.send_failed",
            "error",
            {
                "channel_id": channel_id,
                "attempts": max_attempts,
                "elapsed_seconds": round(total_elapsed, 1),
                "error": str(last_error),
            },
        )
        raise DiscordSendError(
            f"Failed to send message to channel {channel_id} after "
            f"{max_attempts} attempts over {total_elapsed:.1f}s: {last_error}",
            channel_id=channel_id,
            original_error=last_error,
        )

    # ------------------------------------------------------------------
    # Gateway event handler
    # ------------------------------------------------------------------

    async def _on_gateway_event(self, payload: dict[str, Any]) -> None:
        """Receive a raw Gateway payload and route it."""
        event_name: str = payload.get("t", "")
        data: dict[str, Any] = payload.get("d") or {}

        # Update bot user ID from READY (gateway sets it).
        if self._bot_user_id is None:
            gw_id = self._gateway.bot_user_id
            if gw_id:
                self._bot_user_id = gw_id

        if event_name == "MESSAGE_CREATE":
            await self._handle_message(data)
        elif event_name == "INTERACTION_CREATE":
            await self._handle_interaction(data)
        elif event_name == "MESSAGE_REACTION_ADD":
            await self._handle_reaction(data)
        elif event_name == "GUILD_CREATE":
            logger.debug("Discord: GUILD_CREATE for guild %s", data.get("id"))

    # ------------------------------------------------------------------
    # Message handling
    # ------------------------------------------------------------------

    async def _handle_message(self, data: dict[str, Any]) -> None:
        """Apply access control and enqueue an allowed message."""
        author: dict[str, Any] = data.get("author") or {}
        author_id: str = str(author.get("id", ""))
        channel_id: str = str(data.get("channel_id", ""))
        guild_id: str | None = data.get("guild_id") or None
        content: str = str(data.get("content", ""))
        thread_id: str | None = data.get("thread_id") or None

        logger.info(
            "Discord: MESSAGE_CREATE guild=%s channel=%s author=%s content=%r",
            guild_id,
            channel_id,
            author_id,
            content[:80],
        )

        # 1. Filter own-bot messages.
        if self._is_own_message(author_id):
            logger.debug("Discord: ignoring own message (bot_id=%s)", self._bot_user_id)
            return

        # 1b. Credential / secrets detection — delete message and warn if secrets found.
        #
        # Runs before authorization so credentials are always scrubbed
        # regardless of channel/DM policy, but produces no side effect
        # beyond deleting the offending message and warning -- it does not
        # dispatch to voice/image/screencast handling.
        if content:
            try:
                from missy.security.secrets import SecretsDetector

                _detector = SecretsDetector()
                if _detector.has_secrets(content):
                    message_id = str(data.get("id", ""))
                    # Attempt to delete the message from Discord.
                    deleted = False
                    if message_id and self._rest is not None:
                        try:
                            deleted = self._rest.delete_message(channel_id, message_id)
                        except Exception as _del_exc:
                            logger.warning("Failed to delete credential message: %s", _del_exc)

                    self._emit_audit(
                        "discord.channel.credential_detected",
                        "deny",
                        {
                            "author_id": author_id,
                            "channel_id": channel_id,
                            "message_id": message_id,
                            "message_deleted": deleted,
                        },
                    )
                    logger.warning(
                        "Discord: credentials detected in message from %s in channel %s — message %s",
                        author_id,
                        channel_id,
                        "deleted" if deleted else "could not be deleted",
                    )
                    # Send a warning back to the channel.
                    if self._rest is not None:
                        with contextlib.suppress(Exception):
                            self._rest.send_message(
                                channel_id,
                                f"\u26a0\ufe0f <@{author_id}> Your message appeared to contain"
                                f" credentials or secrets and has been"
                                f" {'removed from this channel' if deleted else 'flagged'}."
                                f" Please rotate any exposed keys immediately.",
                            )
                    # Do NOT enqueue this message — drop it after warning.
                    return
            except Exception as _sec_exc:
                logger.debug("Secrets detection error: %s", _sec_exc)

        # 2. Filter other bots.
        if not self._allow_bot_author(author, content, guild_id):
            self._emit_audit(
                "discord.channel.bot_filtered",
                "deny",
                {"author_id": author_id, "is_bot": True},
            )
            return

        # SR-1.13: DM/guild access control must run before ANY command
        # dispatch that can produce a side effect (voice, image,
        # screencast, or the eventual agent enqueue). This used to run
        # last (as step "4"), with voice/image/screencast handled first
        # under a comment literally reading "handled before policy gates"
        # -- meaning any message in any guild channel (ignoring
        # allowed_channels/allowed_roles/allowed_users/require_mention)
        # or any DM (ignoring dm_policy DISABLED/ALLOWLIST/PAIRING) could
        # join a voice channel, capture/analyze a screenshot, or start a
        # screen share before authorization was ever checked. No pre-gate
        # command may produce side effects.
        if guild_id is None:
            allowed = self._check_dm_policy(author_id, content)
        else:
            allowed = self._check_guild_policy(guild_id, channel_id, author_id, content, data)

        if not allowed:
            return

        # 2a2. Per-user rate limit (DISC-CMD-008) — after authorization
        # (so it never leaks whether an unauthorized user exists) but
        # before any command dispatch that could produce a side effect
        # or an LLM call.
        rate_result = self._rate_limiter.check(author_id)
        if not rate_result.allowed:
            self._emit_audit(
                "discord.channel.rate_limited",
                "deny",
                {
                    "author_id": author_id,
                    "channel_id": channel_id,
                    "retry_after_seconds": round(rate_result.retry_after_seconds, 1),
                },
            )
            with contextlib.suppress(Exception):
                self._rest.send_message(
                    channel_id,
                    f"⏳ <@{author_id}> You're sending commands too quickly. "
                    f"Please wait {rate_result.retry_after_seconds:.0f}s and try again.",
                )
            return

        # 2b. Voice commands (MESSAGE_CREATE) — only after authorization.
        if guild_id and content:
            handled = await self._maybe_handle_voice_command(
                guild_id=str(guild_id),
                channel_id=channel_id,
                author_id=author_id,
                content=content,
            )
            if handled:
                return

        # 2c. Image commands (!analyze, !screenshot) — only after authorization.
        if content:
            handled = await self._maybe_handle_image_command(
                channel_id=channel_id,
                content=content,
            )
            if handled:
                return

        # 2d. Screencast commands (!screen ...) — only after authorization.
        if content:
            handled = await self._maybe_handle_screen_command(
                channel_id=channel_id,
                author_id=author_id,
                content=content,
            )
            if handled:
                return

        # 3. Attachment policy gate.
        #
        # Three-way classification: image (for vision analysis), text-like
        # (for direct reading -- .md/.txt/.json/.yaml/.csv/.log, spliced
        # into the prompt as content once downloaded), and everything
        # else, which is still denied outright. A message with even one
        # denied attachment is dropped entirely (matches the prior
        # image-only gate's all-or-nothing behavior) rather than silently
        # processing a partial attachment set.
        attachments: list[dict] = data.get("attachments") or []
        image_attachments: list[tuple[dict, Any]] = []
        text_attachments: list[tuple[dict, Any]] = []
        if attachments:
            from missy.channels.discord.image_analyze import (
                AttachmentValidation,
                is_image_attachment,
                validate_image_attachment,
            )
            from missy.channels.discord.text_attachment import (
                MAX_TEXT_ATTACHMENT_BYTES,
                is_text_attachment,
                validate_text_attachment,
            )

            denied_attachments: list[tuple[dict, Any]] = []
            for attachment in attachments:
                if is_image_attachment(attachment):
                    validation = validate_image_attachment(attachment)
                    (image_attachments if validation.allowed else denied_attachments).append(
                        (attachment, validation)
                    )
                elif is_text_attachment(attachment):
                    validation = validate_text_attachment(attachment)
                    (text_attachments if validation.allowed else denied_attachments).append(
                        (attachment, validation)
                    )
                else:
                    filename = attachment.get("filename") or "attachment"
                    denied_attachments.append(
                        (
                            attachment,
                            AttachmentValidation(
                                allowed=False,
                                reasons=["unsupported_attachment_type"],
                                details={"filename": filename},
                            ),
                        )
                    )

            if denied_attachments:
                denied_details = [
                    {**validation.details, "reasons": validation.reasons}
                    for _attachment, validation in denied_attachments
                ]
                self._emit_audit(
                    "discord.channel.attachment_denied",
                    "deny",
                    {
                        "author_id": author_id,
                        "channel_id": channel_id,
                        "attachment_count": len(denied_attachments),
                        "attachments": denied_details,
                        "reason": "attachment_metadata_not_permitted",
                    },
                )
                logger.info(
                    "Discord: message with %d denied attachment(s) from %s denied by policy",
                    len(denied_attachments),
                    author_id,
                )
                with contextlib.suppress(Exception):
                    names = ", ".join(
                        a.get("filename", "attachment") for a, _v in denied_attachments
                    )
                    self._rest.send_message(
                        channel_id,
                        f"⚠️ <@{author_id}> I can't accept {names} — only image and "
                        f"text file (.md/.txt/.json/.yaml/.csv/.log, under "
                        f"{MAX_TEXT_ATTACHMENT_BYTES // 1024}KB) attachments are "
                        f"supported right now. Paste the content as text instead if "
                        f"you'd like me to look at it.",
                    )
                return

            if image_attachments:
                allowed_details = [
                    {**validation.details, "reasons": []}
                    for _attachment, validation in image_attachments
                ]
                self._emit_audit(
                    "discord.channel.image_attachment_allowed",
                    "allow",
                    {
                        "author_id": author_id,
                        "channel_id": channel_id,
                        "image_count": len(image_attachments),
                        "attachments": allowed_details,
                    },
                )
                logger.info(
                    "Discord: allowing %d image attachment(s) from %s for analysis",
                    len(image_attachments),
                    author_id,
                )

            if text_attachments:
                allowed_text_details = [
                    {**validation.details, "reasons": []}
                    for _attachment, validation in text_attachments
                ]
                self._emit_audit(
                    "discord.channel.text_attachment_allowed",
                    "allow",
                    {
                        "author_id": author_id,
                        "channel_id": channel_id,
                        "text_count": len(text_attachments),
                        "attachments": allowed_text_details,
                    },
                )
                logger.info(
                    "Discord: allowing %d text attachment(s) from %s for reading",
                    len(text_attachments),
                    author_id,
                )

        # 5. Resolve thread-scoped session if applicable.
        effective_thread_id = thread_id
        # Discord thread channels have type 11 (PUBLIC_THREAD) or 12 (PRIVATE_THREAD).
        # If the message came from a thread, the channel_id IS the thread ID.
        channel_type = data.get("channel_type") or data.get("type")
        if channel_type in (11, 12):
            effective_thread_id = channel_id

        thread_session_id = ""
        if effective_thread_id:
            thread_session_id = self._thread_sessions.get(effective_thread_id, "")

        # Track message counts for auto-thread threshold. The counter was
        # previously written but never read anywhere -- an operator setting
        # auto_thread_threshold: N got a counter that silently incremented
        # forever and a feature that never actually fired.
        if guild_id and not effective_thread_id and self._auto_thread_threshold > 0:
            count = self._channel_message_counts.get(channel_id, 0) + 1
            if count >= self._auto_thread_threshold:
                self._channel_message_counts[channel_id] = 0
                thread_name = content[:80] if content else f"Thread ({count} messages)"
                new_thread_id = await self.create_thread(
                    channel_id=channel_id,
                    name=thread_name,
                    message_id=str(data.get("id", "")) or None,
                )
                if new_thread_id:
                    effective_thread_id = new_thread_id
            else:
                self._channel_message_counts[channel_id] = count

        # 6. Enqueue.
        self._current_channel_id = channel_id

        # Include allowed attachment info in metadata for downstream
        # processing (vision analysis for images, direct reading for
        # text). Reuses image_attachments/text_attachments computed by
        # the policy gate above rather than re-validating -- by this
        # point any denied attachment has already caused an early
        # `return`, so these are exactly the final, allowed sets.
        image_attachment_data: list[dict] = [
            {
                "url": attachment.get("url", ""),
                "proxy_url": attachment.get("proxy_url", ""),
                "filename": validation.details["filename"],
                "content_type": validation.details["content_type"],
                "size": validation.details["size"] or 0,
                "width": validation.details["width"] or 0,
                "height": validation.details["height"] or 0,
            }
            for attachment, validation in image_attachments
        ]
        text_attachment_data: list[dict] = [
            {
                "url": attachment.get("url", ""),
                "proxy_url": attachment.get("proxy_url", ""),
                "filename": validation.details["filename"],
                "content_type": validation.details["content_type"],
                "size": validation.details["size"] or 0,
            }
            for attachment, validation in text_attachments
        ]

        msg = ChannelMessage(
            content=content,
            sender=author_id,
            channel=self.name,
            metadata={
                "discord_message_id": str(data.get("id", "")),
                "discord_channel_id": channel_id,
                "discord_guild_id": guild_id or "",
                "discord_thread_id": effective_thread_id or "",
                "discord_thread_session_id": thread_session_id,
                "discord_author": author,
                "discord_author_is_bot": bool(author.get("bot", False)),
                "discord_image_attachments": image_attachment_data,
                "discord_text_attachments": text_attachment_data,
            },
        )
        self._emit_audit(
            "discord.channel.message_received",
            "allow",
            {
                "author_id": author_id,
                "channel_id": channel_id,
                "guild_id": guild_id or "dm",
                "thread_id": effective_thread_id or "",
                "thread_session_id": thread_session_id,
            },
        )
        await self._queue.put(msg)

    async def _maybe_handle_voice_command(
        self,
        guild_id: str,
        channel_id: str,
        author_id: str,
        content: str,
    ) -> bool:
        text = content.strip()
        from missy.channels.discord.voice_commands import (
            maybe_handle_voice_command,
            parse_voice_intent,
        )

        if parse_voice_intent(text) is None:
            return False

        # Lazy-start the voice manager on first voice command.
        if self._voice is None:
            try:
                from missy.channels.discord.voice import DiscordVoiceManager

                # Build async agent callback for voice conversations.
                agent_cb = None
                if self._agent_runtime is not None:
                    _rt = self._agent_runtime

                    async def _voice_agent_cb(prompt: str, session_id: str) -> str:
                        # Regular typed-text messages run through
                        # SecretsDetector before ever reaching the agent
                        # ("1b. Credential / secrets detection" above),
                        # deleting the message and blocking it entirely.
                        # AgentRuntime.run() itself only applies
                        # InputSanitizer (prompt-injection detection), never
                        # SecretsDetector -- so a voice transcript feeding
                        # straight into _rt.run() with no check of its own
                        # bypassed secrets screening completely (e.g.
                        # dictating an API key to "read it back to me"
                        # reached the LLM provider, session history, and
                        # TTS reply unscrubbed). There's no Discord message
                        # to delete for a live voice utterance, so the
                        # equivalent action is refusing to forward the
                        # transcript and returning a spoken warning instead.
                        try:
                            from missy.security.secrets import SecretsDetector

                            if SecretsDetector().has_secrets(prompt):
                                self._emit_audit(
                                    "discord.channel.credential_detected",
                                    "deny",
                                    {"session_id": session_id, "source": "voice_transcript"},
                                )
                                return (
                                    "Your message appeared to contain credentials or"
                                    " secrets, so I didn't process it. Please rotate"
                                    " any exposed keys immediately."
                                )
                        except Exception:
                            logger.debug("Secrets detection error in voice callback", exc_info=True)

                        loop = asyncio.get_running_loop()
                        return await loop.run_in_executor(
                            None,
                            _rt.run,
                            prompt,
                            session_id,
                        )

                    agent_cb = _voice_agent_cb

                self._voice = DiscordVoiceManager(agent_callback=agent_cb)
                token = self.account_config.resolve_token() or ""
                await self._voice.start(token)
                from missy.channels.discord.voice_binding import set_voice_binding

                account_id = self._voice_account_id()
                set_voice_binding(
                    self._voice,
                    asyncio.get_running_loop(),
                    account_id=account_id,
                    guild_id=guild_id,
                )
                self._emit_audit(
                    "discord.voice.binding_registered",
                    "allow",
                    {
                        "account_id": account_id,
                        "guild_id": guild_id,
                        "channel_id": channel_id,
                    },
                )
            except Exception as exc:
                self._voice = None
                with contextlib.suppress(Exception):
                    from missy.channels.discord.voice_binding import clear_voice_binding

                    clear_voice_binding(account_id=self._voice_account_id(), guild_id=guild_id)
                self._emit_audit(
                    "discord.voice.start_failed",
                    "error",
                    {
                        "account_id": self._voice_account_id(),
                        "guild_id": guild_id,
                        "channel_id": channel_id,
                        "error": str(exc),
                    },
                )
                self._rest.send_message(channel_id, f"Voice unavailable: {exc}")
                return True
        else:
            from missy.channels.discord.voice_binding import set_voice_binding

            set_voice_binding(
                self._voice,
                asyncio.get_running_loop(),
                account_id=self._voice_account_id(),
                guild_id=guild_id,
            )

        result = await maybe_handle_voice_command(
            content=text,
            channel_id=channel_id,
            guild_id=guild_id,
            author_id=author_id,
            voice=self._voice,
        )
        if result.handled and result.reply:
            self._rest.send_message(channel_id, result.reply)
        return result.handled

    async def _maybe_handle_image_command(
        self,
        channel_id: str,
        content: str,
    ) -> bool:
        """Handle image requests, either as !analyze / !screenshot bang
        commands or as natural-language phrasings ("what's in this
        screenshot?", "save that image to /tmp")."""
        import re

        text = content.strip()
        # Strip leading bot mentions so "@Missy !analyze" works.
        text = re.sub(r"^(<@!?\d+>\s*)+", "", text).strip()

        from missy.channels.discord.image_commands import (
            infer_image_intent,
            maybe_handle_image_command,
        )

        # Accept either an explicit bang command (!analyze / !screenshot) or a
        # natural-language image request. Anything else falls through to the
        # normal agent path so ordinary conversation is unaffected.
        is_bang = text.startswith("!") and text.split()[0].lower() in ("!analyze", "!screenshot")
        if not is_bang and infer_image_intent(text) is None:
            return False

        # Show typing indicator while processing.
        if self._rest is not None:
            self._rest.trigger_typing(channel_id)

        result = await maybe_handle_image_command(
            content=text,
            channel_id=channel_id,
            rest_client=self._rest,
        )
        if result.handled and result.reply:
            # Analysis responses can be long — use message splitting.
            reply = result.reply
            if len(reply) <= 2000:
                self._rest.send_message(channel_id, reply)
            else:
                # Split into 1990-char chunks.
                for i in range(0, len(reply), 1990):
                    self._rest.send_message(channel_id, reply[i : i + 1990])
        return result.handled

    async def _maybe_handle_screen_command(
        self,
        channel_id: str,
        author_id: str,
        content: str,
    ) -> bool:
        """Handle screencast requests, either as "!screen ..." bang commands
        or as natural-language phrasings ("share my screen", "what's on the
        screen", "stop the screen share")."""
        import re

        text = content.strip()
        # Strip leading bot mentions so "@Missy !screen share" works.
        text = re.sub(r"^(<@!?\d+>\s*)+", "", text).strip()

        from missy.channels.discord.screen_commands import (
            infer_screen_intent,
            maybe_handle_screen_command,
        )

        # Accept either an explicit "!screen ..." command or a natural-language
        # screen request; otherwise fall through to the normal agent path.
        if not text.startswith("!screen") and infer_screen_intent(text) is None:
            return False

        result = await maybe_handle_screen_command(
            content=text,
            channel_id=channel_id,
            author_id=author_id,
            screencast=self._screencast,
        )
        if result.handled and result.reply:
            reply = result.reply
            if len(reply) <= 2000:
                self._rest.send_message(channel_id, reply)
            else:
                for i in range(0, len(reply), 1990):
                    self._rest.send_message(channel_id, reply[i : i + 1990])
        return result.handled

    async def _handle_interaction(self, data: dict[str, Any]) -> None:
        """Handle a slash command interaction.

        Sends a deferred response (type 5) immediately, then runs the
        command handler and edits the original response with the result.
        This avoids the 3-second interaction timeout for slow providers.
        """
        from missy.channels.discord.commands import handle_slash_command

        interaction_id: str = str(data.get("id", ""))
        interaction_token: str = str(data.get("token", ""))
        channel_id: str = str(data.get("channel_id", ""))
        guild_id: str | None = data.get("guild_id") or None
        self._current_channel_id = channel_id

        # SR-1.13: slash-command interactions arrive over a completely
        # separate Gateway event (INTERACTION_CREATE) from regular
        # messages and previously had NO authorization check at all --
        # /ask dispatched straight to the full agent for any Discord
        # user in any guild/channel/DM, ignoring allowed_channels/
        # allowed_users/dm_policy entirely. Guild interactions carry the
        # invoking user under member.user; DM interactions carry it
        # under user directly.
        member = data.get("member") or {}
        interaction_user = member.get("user") or data.get("user") or {}
        author_id: str = str(interaction_user.get("id", ""))

        if guild_id is None:
            allowed = self._check_dm_policy(author_id, "")
        else:
            allowed = self._check_guild_policy(
                guild_id, channel_id, author_id, "", data, skip_mention_check=True
            )

        if not allowed:
            try:
                self._rest.send_interaction_response(
                    interaction_id,
                    interaction_token,
                    response_type=4,  # CHANNEL_MESSAGE_WITH_SOURCE
                    data={"content": "You're not authorized to use Missy commands here."},
                )
            except Exception as exc:
                logger.error("Discord: interaction denial response failed: %s", exc)
            return

        # DISC-CMD-008: per-user rate limit, checked after authorization
        # (never leaks whether an unauthorized user exists) but before
        # dispatching to the potentially LLM-calling command handler.
        rate_result = self._rate_limiter.check(author_id)
        if not rate_result.allowed:
            self._emit_audit(
                "discord.channel.rate_limited",
                "deny",
                {
                    "author_id": author_id,
                    "channel_id": channel_id,
                    "retry_after_seconds": round(rate_result.retry_after_seconds, 1),
                },
            )
            try:
                self._rest.send_interaction_response(
                    interaction_id,
                    interaction_token,
                    response_type=4,  # CHANNEL_MESSAGE_WITH_SOURCE
                    data={
                        "content": (
                            f"⏳ You're sending commands too quickly. "
                            f"Please wait {rate_result.retry_after_seconds:.0f}s and try again."
                        )
                    },
                )
            except Exception as exc:
                logger.error("Discord: rate-limit response failed: %s", exc)
            return

        # Send deferred response immediately (type 5 = DEFERRED_CHANNEL_MESSAGE_WITH_SOURCE)
        try:
            self._rest.send_interaction_response(interaction_id, interaction_token, response_type=5)
        except Exception as exc:
            logger.error("Discord: deferred interaction response failed: %s", exc)
            return

        # Run the command handler (may take a while for /ask with slow providers)
        try:
            response_text = await handle_slash_command(data, self)
        except Exception as exc:
            logger.exception("Discord: slash command handler failed: %s", exc)
            response_text = f"Sorry, I encountered an error: {exc}"

        # Edit the deferred response with the actual result
        try:
            app_id = self.account_config.application_id
            if not app_id:
                logger.error(
                    "Discord: no application_id configured, cannot edit interaction response"
                )
                return
            self._rest.edit_interaction_response(app_id, interaction_token, response_text)
        except Exception as exc:
            logger.error("Discord: editing interaction response failed: %s", exc)

    # ------------------------------------------------------------------
    # Access control
    # ------------------------------------------------------------------

    def _is_own_message(self, author_id: str) -> bool:
        """Return True if *author_id* matches this bot's own user ID."""
        own_id = self.bot_user_id or self.account_config.account_id
        return bool(own_id and author_id == own_id)

    def _allow_bot_author(
        self,
        author: dict[str, Any],
        content: str,
        guild_id: str | None,
    ) -> bool:
        """Return True if the message from a bot author should be processed."""
        if not author.get("bot"):
            return True  # Not a bot — always continue.

        if not self.account_config.ignore_bots:
            return True  # Configured to accept bot messages.

        # ignore_bots is True: check the exemption flag.
        if self.account_config.allow_bots_if_mention_only:
            own_id = self.bot_user_id or self.account_config.account_id
            if own_id and f"<@{own_id}>" in content:
                return True
            if own_id and f"<@!{own_id}>" in content:
                return True

        return False

    def _check_dm_policy(self, author_id: str, content: str) -> bool:
        """Evaluate DM policy for a direct message."""
        policy = self.account_config.dm_policy

        if policy == DiscordDMPolicy.DISABLED:
            self._emit_audit(
                "discord.channel.message_denied",
                "deny",
                {"reason": "dm_disabled", "author_id": author_id},
            )
            return False

        if policy == DiscordDMPolicy.OPEN:
            return True

        if policy == DiscordDMPolicy.ALLOWLIST:
            if author_id in self.account_config.dm_allowlist:
                return True
            self._emit_audit(
                "discord.channel.allowlist_denied",
                "deny",
                {"reason": "dm_allowlist", "author_id": author_id},
            )
            return False

        if policy == DiscordDMPolicy.PAIRING:
            return self._check_pairing(author_id, content)

        return False

    def _check_pairing(self, author_id: str, content: str) -> bool:
        """Handle pairing workflow; return True only when paired."""
        # Command to initiate pairing.
        if content.strip().lower() in ("!pair", "/pair"):
            self._pending_pairs.add(author_id)
            self._emit_audit(
                "discord.channel.pairing_wait",
                "allow",
                {"author_id": author_id},
            )
            logger.info("Discord: pairing requested by %s", author_id)
            return False  # Don't forward the pairing command itself.

        # SR-1.12: pairing approval/denial must NEVER be reachable from an
        # in-band DM message. There is no way to authenticate the sender of
        # a Discord DM as an authorized operator from within this handler --
        # any unpaired stranger could otherwise DM `!pair` followed by
        # `!pair accept <their own id>` and grant themselves access with no
        # authentication at all. accept_pair()/deny_pair() remain the only
        # way to resolve a pending request, and must only ever be invoked
        # from an authenticated operator surface (the Web console/API,
        # which shares this process under `missy gateway start`) -- never
        # from message content. Treat these as unrecognised commands and
        # audit the attempt.
        lowered = content.strip().lower()
        if lowered.startswith(("!pair accept ", "!pair deny ")):
            self._emit_audit(
                "discord.channel.pairing_decision_denied",
                "deny",
                {
                    "reason": "pairing_decisions_not_available_via_dm",
                    "author_id": author_id,
                },
            )
            logger.warning(
                "Discord: rejected in-band pairing decision command from %s "
                "-- pairing must be approved via an authenticated operator "
                "surface, not DM content.",
                author_id,
            )
            return False

        # Regular message: allowed only if already in the allowlist.
        if author_id in self.account_config.dm_allowlist:
            return True

        self._emit_audit(
            "discord.channel.message_denied",
            "deny",
            {"reason": "pairing_required", "author_id": author_id},
        )
        return False

    def _check_guild_policy(
        self,
        guild_id: str,
        channel_id: str,
        author_id: str,
        content: str,
        data: dict[str, Any],
        *,
        skip_mention_check: bool = False,
    ) -> bool:
        """Evaluate the guild-level access policy.

        Args:
            skip_mention_check: When ``True``, the ``require_mention``
                rule is not evaluated. Slash-command interactions are
                inherently addressed to this bot by Discord's own
                command routing -- there is no message text to contain
                an ``@mention``, and requiring one would incorrectly
                block every slash command in guilds configured with
                ``require_mention: true``.
        """
        guild_policy = self.account_config.guild_policies.get(guild_id)
        if guild_policy is None:
            # No policy defined for this guild — default deny.
            self._emit_audit(
                "discord.channel.message_denied",
                "deny",
                {"reason": "no_guild_policy", "guild_id": guild_id, "author_id": author_id},
            )
            return False

        if not guild_policy.enabled:
            self._emit_audit(
                "discord.channel.message_denied",
                "deny",
                {"reason": "guild_disabled", "guild_id": guild_id, "author_id": author_id},
            )
            return False

        # Channel allowlist check — match by ID or name.
        if guild_policy.allowed_channels:
            channel_name = str(
                (data.get("channel") or {}).get("name", "") or data.get("channel_name", "")
            )
            # A message posted inside a thread carries channel_id = the
            # thread's own snowflake, never its parent's -- but the
            # allowlist is naturally configured with parent-channel
            # IDs/names. Check the thread's known parent (if this bot
            # created the thread) in addition to the raw channel_id, so a
            # thread under an allowed channel isn't silently denied.
            parent_channel_id = self._thread_parents.get(channel_id)
            # A message always has channel_id; name may be absent from Gateway events.
            in_allowlist = (
                channel_id in guild_policy.allowed_channels
                or (channel_name and channel_name in guild_policy.allowed_channels)
                or (parent_channel_id and parent_channel_id in guild_policy.allowed_channels)
            )
            logger.debug(
                "Discord: channel check guild=%s channel_id=%s channel_name=%r allowlist=%s match=%s",
                guild_id,
                channel_id,
                channel_name,
                guild_policy.allowed_channels,
                in_allowlist,
            )
            if not in_allowlist:
                self._emit_audit(
                    "discord.channel.allowlist_denied",
                    "deny",
                    {
                        "reason": "channel_not_allowed",
                        "channel_id": channel_id,
                        "channel_name": channel_name,
                        "guild_id": guild_id,
                        "author_id": author_id,
                    },
                )
                return False

        # User allowlist check.
        if guild_policy.allowed_users and author_id not in guild_policy.allowed_users:
            self._emit_audit(
                "discord.channel.allowlist_denied",
                "deny",
                {"reason": "user_not_in_allowlist", "guild_id": guild_id, "author_id": author_id},
            )
            return False

        # Role allowlist check. The Gateway's message `member` object
        # carries the author's role IDs (snowflakes); allowed_roles is
        # documented and configured as role *names*, so the IDs are
        # resolved via a cached guild role lookup before comparing.
        if guild_policy.allowed_roles:
            member = data.get("member") or {}
            member_role_ids = member.get("roles") or []
            member_role_names = self._resolve_role_names(guild_id, member_role_ids)
            if not member_role_names & set(guild_policy.allowed_roles):
                self._emit_audit(
                    "discord.channel.allowlist_denied",
                    "deny",
                    {
                        "reason": "role_not_in_allowlist",
                        "guild_id": guild_id,
                        "author_id": author_id,
                    },
                )
                return False

        # Mention requirement check.
        if guild_policy.require_mention and not skip_mention_check:
            own_id = self.bot_user_id or self.account_config.account_id
            if own_id:
                mentioned = f"<@{own_id}>" in content or f"<@!{own_id}>" in content
            else:
                # Fallback: check mentions list in message data.
                mentions = data.get("mentions") or []
                mentioned = any(str(m.get("id", "")) == (own_id or "") for m in mentions)

            if not mentioned:
                self._emit_audit(
                    "discord.channel.require_mention_filtered",
                    "deny",
                    {"reason": "mention_required", "guild_id": guild_id, "author_id": author_id},
                )
                return False

        return True

    def _resolve_role_names(self, guild_id: str, role_ids: list[str]) -> set[str]:
        """Resolve role ID snowflakes to role names for ``guild_id``.

        Discord's Gateway ``message.member.roles`` field only carries
        role ID snowflakes, but ``DiscordGuildPolicy.allowed_roles`` is
        documented and configured as human-readable role *names* — this
        bridges the two via a cached (TTL
        ``_GUILD_ROLES_CACHE_TTL_SECONDS``) call to
        ``GET /guilds/{id}/roles``, so a normal message doesn't need its
        own REST round trip.

        Args:
            guild_id: The guild the message was sent in.
            role_ids: Role ID snowflakes from the message's ``member``
                object.

        Returns:
            The set of role names corresponding to ``role_ids``.  On a
            REST failure, returns an empty set (fail closed — an
            unresolvable role can never satisfy an allowlist).
        """
        if not role_ids:
            return set()

        now = time.monotonic()
        cached = self._guild_roles_cache.get(guild_id)
        if cached is None or (now - cached[0]) >= _GUILD_ROLES_CACHE_TTL_SECONDS:
            try:
                roles = self._rest.get_guild_roles(guild_id)
                role_map = {str(r["id"]): str(r["name"]) for r in roles}
            except Exception:
                logger.warning(
                    "Discord: failed to fetch guild roles for %s -- "
                    "allowed_roles check fails closed for this message.",
                    guild_id,
                    exc_info=True,
                )
                return set()
            self._guild_roles_cache[guild_id] = (now, role_map)
            cached = self._guild_roles_cache[guild_id]

        role_map = cached[1]
        return {role_map[rid] for rid in role_ids if rid in role_map}

    # ------------------------------------------------------------------
    # Pairing management
    # ------------------------------------------------------------------

    def get_pending_pairs(self) -> set[str]:
        """Return the set of user IDs awaiting pairing confirmation."""
        return set(self._pending_pairs)

    def accept_pair(self, user_id: str) -> None:
        """Accept a pending pairing request from *user_id*."""
        self._pending_pairs.discard(user_id)
        if user_id not in self.account_config.dm_allowlist:
            self.account_config.dm_allowlist.append(user_id)

    def deny_pair(self, user_id: str) -> None:
        """Deny and remove a pending pairing request from *user_id*."""
        self._pending_pairs.discard(user_id)

    # ------------------------------------------------------------------
    # Thread management
    # ------------------------------------------------------------------

    def get_thread_session(self, thread_id: str) -> str | None:
        """Return the session ID associated with a thread, if any."""
        return self._thread_sessions.get(thread_id)

    def set_thread_session(self, thread_id: str, session_id: str) -> None:
        """Associate a session ID with a Discord thread."""
        self._thread_sessions[thread_id] = session_id

    async def create_thread(
        self,
        channel_id: str,
        name: str,
        message_id: str | None = None,
        session_id: str | None = None,
    ) -> str | None:
        """Create a new Discord thread and optionally bind it to a session.

        Args:
            channel_id: Parent channel snowflake ID.
            name: Thread name.
            message_id: Optional message ID to start thread from.
            session_id: Optional session ID to bind to this thread.

        Returns:
            The thread ID on success, or None on failure.
        """
        try:
            result = self._rest.create_thread(
                channel_id=channel_id,
                name=name,
                message_id=message_id,
            )
            thread_id = str(result.get("id", ""))
            if thread_id:
                self._thread_parents[thread_id] = channel_id
            if thread_id and session_id:
                self._thread_sessions[thread_id] = session_id
            self._emit_audit(
                "discord.thread.created",
                "allow",
                {
                    "channel_id": channel_id,
                    "thread_id": thread_id,
                    "thread_name": name,
                    "session_id": session_id or "",
                },
            )
            logger.info("Discord: created thread %s (%s) in %s", thread_id, name, channel_id)
            return thread_id
        except Exception as exc:
            logger.error("Discord: thread creation failed: %s", exc)
            self._emit_audit(
                "discord.thread.create_failed",
                "error",
                {"channel_id": channel_id, "error": str(exc)},
            )
            return None

    # ------------------------------------------------------------------
    # Evolution reaction workflow
    # ------------------------------------------------------------------

    def add_evolution_reactions(
        self,
        channel_id: str,
        message_id: str,
        proposal_id: str,
    ) -> None:
        """Add approve/reject reaction buttons to an evolution proposal message.

        Args:
            channel_id: The channel containing the message.
            message_id: The bot's response message snowflake ID.
            proposal_id: The evolution proposal ID to track.
        """
        self._pending_evolutions[message_id] = proposal_id
        try:
            self._rest.add_reaction(channel_id, message_id, "\u2705")  # ✅
            self._rest.add_reaction(channel_id, message_id, "\u274c")  # ❌
            self._emit_audit(
                "discord.evolution.reactions_added",
                "allow",
                {
                    "channel_id": channel_id,
                    "message_id": message_id,
                    "proposal_id": proposal_id,
                },
            )
            logger.info(
                "Discord: added evolution reactions to message %s (proposal %s)",
                message_id,
                proposal_id,
            )
        except Exception as exc:
            logger.error("Discord: failed to add evolution reactions: %s", exc)
            self._pending_evolutions.pop(message_id, None)

    async def _handle_reaction(self, data: dict[str, Any]) -> None:
        """Handle a MESSAGE_REACTION_ADD event for evolution approval."""
        message_id = str(data.get("message_id", ""))
        user_id = str(data.get("user_id", ""))
        channel_id = str(data.get("channel_id", ""))
        emoji = data.get("emoji", {})
        emoji_name = emoji.get("name", "")

        # Ignore reactions from the bot itself.
        if self._is_own_message(user_id):
            return

        # Only process reactions on tracked evolution messages.
        proposal_id = self._pending_evolutions.get(message_id)
        if not proposal_id:
            return

        if emoji_name not in ("\u2705", "\u274c"):
            return

        action = "approve" if emoji_name == "\u2705" else "reject"
        logger.info(
            "Discord: evolution %s by user %s for proposal %s",
            action,
            user_id,
            proposal_id,
        )

        try:
            from missy.agent.code_evolution import CodeEvolutionManager

            mgr = CodeEvolutionManager()

            if action == "approve":
                # SR-1.2/1.3: a Discord user reacting with an emoji is not an
                # authenticated human operator -- Discord identity is soft,
                # and any user able to see and react to this message could
                # otherwise approve a change to Missy's own source code with
                # no authentication at all. Approval is only available via
                # `missy evolve approve <id>` run from a terminal session on
                # the host. Do not call mgr.approve() here.
                self._rest.send_message(
                    channel_id,
                    f"\u26a0\ufe0f Evolution **{proposal_id}** cannot be approved from "
                    "Discord. An operator must run "
                    f"`missy evolve approve {proposal_id}` from a terminal on "
                    f"the host, then `missy evolve apply {proposal_id}` to apply it.",
                )
                self._emit_audit(
                    "discord.evolution.approve_denied",
                    "deny",
                    {
                        "proposal_id": proposal_id,
                        "user_id": user_id,
                        "channel_id": channel_id,
                        "reason": "discord_reaction_cannot_approve_code_evolution",
                    },
                )
            else:
                if mgr.reject(proposal_id):
                    self._rest.send_message(
                        channel_id,
                        f"\u274c Evolution **{proposal_id}** rejected by <@{user_id}>.",
                    )
                    self._emit_audit(
                        "discord.evolution.rejected",
                        "allow",
                        {
                            "proposal_id": proposal_id,
                            "user_id": user_id,
                            "channel_id": channel_id,
                        },
                    )
                else:
                    self._rest.send_message(
                        channel_id,
                        f"Could not reject evolution **{proposal_id}** — "
                        f"it may already be resolved.",
                    )

            # Remove from pending once acted upon.
            self._pending_evolutions.pop(message_id, None)

        except Exception as exc:
            logger.error("Discord: evolution reaction handling failed: %s", exc)
            self._rest.send_message(
                channel_id,
                f"Error processing evolution reaction: {exc}",
            )

    # ------------------------------------------------------------------
    # Audit helpers
    # ------------------------------------------------------------------

    def _voice_account_id(self) -> str:
        """Return the stable account key used for Discord voice binding scopes."""
        return str(self.account_config.account_id or self.bot_user_id or self._session_id).strip()

    def _emit_audit(
        self,
        event_type: str,
        result: str,
        detail: dict[str, Any],
    ) -> None:
        """Publish an audit event onto the process-level event bus."""
        try:
            event = AuditEvent.now(
                session_id=self._session_id,
                task_id=self._task_id,
                event_type=event_type,
                category="network",
                result=result,  # type: ignore[arg-type]
                detail=detail,
            )
            event_bus.publish(event)
        except Exception as exc:
            logger.debug("Audit emit failed: %s", exc)
