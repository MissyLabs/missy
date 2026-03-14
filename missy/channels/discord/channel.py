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
   apply channel allowlist, user allowlist, and mention requirement.

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
import logging
from typing import Any, Optional

from missy.channels.base import BaseChannel, ChannelMessage
from missy.channels.discord.config import DiscordAccountConfig, DiscordDMPolicy
from missy.channels.discord.gateway import DiscordGatewayClient
from missy.channels.discord.rest import DiscordRestClient
from missy.core.events import AuditEvent, event_bus

logger = logging.getLogger(__name__)


class DiscordSendError(Exception):
    """Raised when a Discord message could not be delivered."""

    def __init__(
        self,
        message: str,
        *,
        channel_id: str = "",
        original_error: Optional[Exception] = None,
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
        self._current_channel_id: Optional[str] = None

        # Pairing state: set of Discord user IDs that have initiated pairing
        # but not yet been confirmed.
        self._pending_pairs: set[str] = set()

        # Resolved bot user ID (populated after READY).
        self._bot_user_id: Optional[str] = None

        # Thread-scoped session mapping: thread_id -> session_id
        # Enables conversation continuity within Discord threads.
        self._thread_sessions: dict[str, str] = {}

        # Message count per channel for auto-thread creation.
        self._channel_message_counts: dict[str, int] = {}

        # Auto-thread threshold: create a thread after N messages in a channel.
        self._auto_thread_threshold: int = getattr(
            account_config, "auto_thread_threshold", 0
        )

        # Pending evolution reactions: message_id -> proposal_id
        self._pending_evolutions: dict[str, str] = {}

        # Optional voice manager (lazy import so text-only deployments don't need voice deps)
        self._voice = None

        # Agent runtime reference — set via set_agent_runtime() from main.py
        # so voice can call the agent for conversational responses.
        self._agent_runtime: Any = None

        token = account_config.resolve_token() or ""
        if not token:
            logger.error(
                "Discord: no bot token found — set env var %r or add 'token:' to config",
                account_config.token_env_var,
            )
        else:
            logger.info("Discord: bot token resolved (%.8s…)", token.lstrip("Bot ").strip()[:8])
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

        self._gateway_task: Optional[asyncio.Task[None]] = None

    # ------------------------------------------------------------------
    # Public properties
    # ------------------------------------------------------------------

    @property
    def bot_user_id(self) -> Optional[str]:
        """The Discord user ID of the connected bot, available after READY."""
        return self._bot_user_id or self._gateway.bot_user_id

    def set_agent_runtime(self, agent_runtime: Any) -> None:
        """Provide the agent runtime for voice conversation support."""
        self._agent_runtime = agent_runtime

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
                logger.info("Discord: slash commands registered globally")
            except Exception as exc:
                logger.warning("Discord: slash command registration failed: %s", exc)

    async def stop(self) -> None:
        """Disconnect from the Gateway and cancel the background task."""
        await self._gateway.disconnect()
        if self._gateway_task is not None:
            self._gateway_task.cancel()
            try:
                await self._gateway_task
            except asyncio.CancelledError:
                pass
            self._gateway_task = None

    # ------------------------------------------------------------------
    # BaseChannel interface
    # ------------------------------------------------------------------

    def receive(self) -> Optional[ChannelMessage]:
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

    async def areceive(self) -> Optional[ChannelMessage]:
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
            loop = asyncio.get_event_loop()
            loop.create_task(self.send_with_retry(self._current_channel_id, message))
        except RuntimeError:
            logger.warning("DiscordChannel.send(): no running event loop — message dropped")

    async def send_to(
        self,
        channel_id: str,
        message: str,
        reply_to: Optional[str] = None,
        thread_id: Optional[str] = None,
        mention_user_ids: Optional[list[str]] = None,
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
        try:
            self._rest.trigger_typing(channel_id)
        except Exception:
            pass

        target_channel = thread_id if thread_id else channel_id

        # Discord hard limit is 2000 characters per message. Split if needed.
        _DISCORD_MAX = 1990
        chunks = [message[i:i + _DISCORD_MAX] for i in range(0, max(len(message), 1), _DISCORD_MAX)]

        last_message_id: Optional[str] = None
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
        reply_to: Optional[str] = None,
        thread_id: Optional[str] = None,
        mention_user_ids: Optional[list[str]] = None,
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
        last_error: Optional[Exception] = None

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
                        attempt, max_attempts, _time.monotonic() - start,
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
                    attempt, max_attempts, channel_id, delay, exc,
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
            "Discord send_with_retry exhausted all %d attempts over %.1fs "
            "(channel=%s)",
            max_attempts, total_elapsed, channel_id,
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
        guild_id: Optional[str] = data.get("guild_id") or None
        content: str = str(data.get("content", ""))
        thread_id: Optional[str] = data.get("thread_id") or None

        logger.info(
            "Discord: MESSAGE_CREATE guild=%s channel=%s author=%s content=%r",
            guild_id, channel_id, author_id, content[:80],
        )

        # 1. Filter own-bot messages.
        if self._is_own_message(author_id):
            logger.debug("Discord: ignoring own message (bot_id=%s)", self._bot_user_id)
            return

        # 0. Voice MVP commands (MESSAGE_CREATE) — handled before policy gates.
        if guild_id and content:
            handled = await self._maybe_handle_voice_command(
                guild_id=str(guild_id),
                channel_id=channel_id,
                author_id=author_id,
                content=content,
            )
            if handled:
                return

        # 1b. Credential / secrets detection — delete message and warn if secrets found.
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
                        try:
                            self._rest.send_message(
                                channel_id,
                                f"\u26a0\ufe0f <@{author_id}> Your message appeared to contain"
                                f" credentials or secrets and has been"
                                f" {'removed from this channel' if deleted else 'flagged'}."
                                f" Please rotate any exposed keys immediately.",
                            )
                        except Exception:
                            pass
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

        # 3. Attachment policy gate.
        attachments: list[dict] = data.get("attachments") or []
        if attachments:
            # Attachments are policy-gated: deny unless explicitly configured.
            # For now: log and drop messages with attachments (safe default).
            self._emit_audit(
                "discord.channel.attachment_denied",
                "deny",
                {
                    "author_id": author_id,
                    "channel_id": channel_id,
                    "attachment_count": len(attachments),
                    "reason": "attachments_not_permitted",
                },
            )
            logger.info(
                "Discord: message with %d attachment(s) from %s denied by policy",
                len(attachments),
                author_id,
            )
            return

        # 4. Route to DM or guild access control.
        if guild_id is None:
            allowed = self._check_dm_policy(author_id, content)
        else:
            allowed = self._check_guild_policy(guild_id, channel_id, author_id, content, data)

        if not allowed:
            return

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

        # Track message counts for auto-thread threshold.
        if guild_id and not effective_thread_id and self._auto_thread_threshold > 0:
            count = self._channel_message_counts.get(channel_id, 0) + 1
            self._channel_message_counts[channel_id] = count

        # 6. Enqueue.
        self._current_channel_id = channel_id
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
        # Strip leading bot mentions so "@Missy !join General" works.
        import re
        text = re.sub(r"^(<@!?\d+>\s*)+", "", text).strip()
        if not text.startswith("!"):
            return False

        cmd = text.split()[0].lower()
        if cmd not in ("!join", "!leave", "!say"):
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
                        loop = asyncio.get_event_loop()
                        return await loop.run_in_executor(
                            None, _rt.run, prompt, session_id,
                        )

                    agent_cb = _voice_agent_cb

                self._voice = DiscordVoiceManager(agent_callback=agent_cb)
                token = self.account_config.resolve_token() or ""
                await self._voice.start(token)
            except Exception as exc:
                self._voice = None
                self._rest.send_message(channel_id, f"Voice unavailable: {exc}")
                return True

        from missy.channels.discord.voice_commands import maybe_handle_voice_command

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

    async def _handle_interaction(self, data: dict[str, Any]) -> None:
        """Handle a slash command interaction."""
        from missy.channels.discord.commands import handle_slash_command

        interaction_id: str = str(data.get("id", ""))
        interaction_token: str = str(data.get("token", ""))
        channel_id: str = str(data.get("channel_id", ""))
        self._current_channel_id = channel_id

        response_text = await handle_slash_command(data, self)

        # Respond to the interaction via the REST API.
        try:
            url = (
                f"https://discord.com/api/v10/interactions/"
                f"{interaction_id}/{interaction_token}/callback"
            )
            from missy.gateway.client import create_client

            http = create_client(session_id=self._session_id, task_id="interaction")
            http.post(
                url,
                headers={
                    "Authorization": self._rest._token,
                    "Content-Type": "application/json",
                },
                json={
                    "type": 4,  # CHANNEL_MESSAGE_WITH_SOURCE
                    "data": {"content": response_text[:2000]},
                },
            )
        except Exception as exc:
            logger.error("Discord: interaction response failed: %s", exc)

    # ------------------------------------------------------------------
    # Access control
    # ------------------------------------------------------------------

    def _is_own_message(self, author_id: str) -> bool:
        """Return True if *author_id* matches this bot's own user ID."""
        own_id = self.bot_user_id or self.account_config.account_id
        if own_id and author_id == own_id:
            return True
        return False

    def _allow_bot_author(
        self,
        author: dict[str, Any],
        content: str,
        guild_id: Optional[str],
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

        # Commands to accept/deny pending pair requests (admin only — simplified).
        if content.strip().lower().startswith("!pair accept "):
            target_id = content.strip().split()[-1]
            self._pending_pairs.discard(target_id)
            self.account_config.dm_allowlist.append(target_id)
            logger.info("Discord: pairing accepted for %s", target_id)
            return False

        if content.strip().lower().startswith("!pair deny "):
            target_id = content.strip().split()[-1]
            self._pending_pairs.discard(target_id)
            logger.info("Discord: pairing denied for %s", target_id)
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
    ) -> bool:
        """Evaluate the guild-level access policy."""
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
                (data.get("channel") or {}).get("name", "")
                or data.get("channel_name", "")
            )
            # A message always has channel_id; name may be absent from Gateway events.
            in_allowlist = (
                channel_id in guild_policy.allowed_channels
                or (channel_name and channel_name in guild_policy.allowed_channels)
            )
            logger.debug(
                "Discord: channel check guild=%s channel_id=%s channel_name=%r allowlist=%s match=%s",
                guild_id, channel_id, channel_name, guild_policy.allowed_channels, in_allowlist,
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

        # Mention requirement check.
        if guild_policy.require_mention:
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

    def get_thread_session(self, thread_id: str) -> Optional[str]:
        """Return the session ID associated with a thread, if any."""
        return self._thread_sessions.get(thread_id)

    def set_thread_session(self, thread_id: str, session_id: str) -> None:
        """Associate a session ID with a Discord thread."""
        self._thread_sessions[thread_id] = session_id

    async def create_thread(
        self,
        channel_id: str,
        name: str,
        message_id: Optional[str] = None,
        session_id: Optional[str] = None,
    ) -> Optional[str]:
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
                if mgr.approve(proposal_id):
                    self._rest.send_message(
                        channel_id,
                        f"\u2705 Evolution **{proposal_id}** approved by <@{user_id}>.\n"
                        f"Use `code_evolve(action='apply', proposal_id='{proposal_id}')` "
                        f"to apply the change.",
                    )
                    self._emit_audit(
                        "discord.evolution.approved",
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
                        f"Could not approve evolution **{proposal_id}** — "
                        f"it may already be approved or in an invalid state.",
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
