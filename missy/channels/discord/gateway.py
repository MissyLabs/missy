"""Discord Gateway WebSocket client.

Implements the Discord Gateway protocol (API v10) directly using the
``websockets`` library — **without** discord.py or any other bot framework.

The client handles the full connection lifecycle:

1. Connect to the Gateway WSS URL.
2. Receive the HELLO opcode and start the heartbeat loop.
3. Send IDENTIFY.
4. Receive READY and store the session ID / resume URL.
5. Forward MESSAGE_CREATE, GUILD_CREATE, and INTERACTION_CREATE events to
   the registered ``on_message`` callback.
6. Resume from the last sequence number when reconnecting.

Audit events are emitted for: ``discord.gateway.connect``,
``discord.gateway.disconnect``, ``discord.gateway.heartbeat_sent``,
``discord.gateway.session_resumed``.

Example::

    import asyncio
    from missy.channels.discord.gateway import DiscordGatewayClient

    async def handle(event: dict) -> None:
        print(event)

    gw = DiscordGatewayClient(bot_token="Bot TOKEN", on_message=handle)
    asyncio.run(gw.run())
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
from collections.abc import Callable, Coroutine
from typing import Any

from missy.core.events import AuditEvent, event_bus

logger = logging.getLogger(__name__)

# Discord Gateway opcodes
_OP_DISPATCH = 0
_OP_HEARTBEAT = 1
_OP_IDENTIFY = 2
_OP_RESUME = 6
_OP_RECONNECT = 7
_OP_INVALID_SESSION = 9
_OP_HELLO = 10
_OP_HEARTBEAT_ACK = 11

# Gateway API version
_GATEWAY_URL = "wss://gateway.discord.gg/?v=10&encoding=json"

# Intents: GUILDS | GUILD_MESSAGES | GUILD_MESSAGE_REACTIONS
#         | DIRECT_MESSAGES | DIRECT_MESSAGE_REACTIONS | MESSAGE_CONTENT
_INTENTS = 1 | 512 | 1024 | 4096 | 8192 | 32768

AsyncMessageCallback = Callable[[dict[str, Any]], Coroutine[Any, Any, None]]


class DiscordGatewayClient:
    """Async Discord Gateway client.

    Args:
        bot_token: The Discord bot token (with or without ``"Bot "`` prefix).
        on_message: Async callback invoked for every dispatched event.
        gateway_url: Override the default WSS Gateway URL.
        session_id: Identifier forwarded to audit events.
        task_id: Identifier forwarded to audit events.
    """

    def __init__(
        self,
        bot_token: str,
        on_message: AsyncMessageCallback,
        gateway_url: str = _GATEWAY_URL,
        session_id: str = "discord",
        task_id: str = "gateway",
    ) -> None:
        if not bot_token.startswith("Bot "):
            bot_token = f"Bot {bot_token}"
        self._token = bot_token
        self._on_message = on_message
        self._gateway_url = gateway_url
        self._session_id_audit = session_id
        self._task_id_audit = task_id

        # Runtime state
        self._ws: Any = None  # websockets.WebSocketClientProtocol
        self._heartbeat_task: asyncio.Task[None] | None = None
        self._sequence: int | None = None
        self._discord_session_id: str | None = None
        self._resume_gateway_url: str | None = None
        self._bot_user_id: str | None = None
        self._running: bool = False

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    @property
    def bot_user_id(self) -> str | None:
        """The Discord user ID of the connected bot, available after READY."""
        return self._bot_user_id

    async def connect(self) -> None:
        """Open the Gateway WebSocket connection and complete the handshake.

        After this method returns the heartbeat loop is running and the
        client has sent IDENTIFY (or RESUME on a reconnect).

        Raises:
            RuntimeError: If ``websockets`` is not installed.
            Exception: Propagates WebSocket / network errors.
        """
        try:
            import websockets  # type: ignore[import-untyped]
        except ImportError as exc:
            raise RuntimeError(
                "The 'websockets' package is required for Discord Gateway support. "
                "Install it with: pip install websockets>=12.0"
            ) from exc

        url = self._resume_gateway_url or self._gateway_url
        logger.debug("Discord Gateway: connecting to %s", url)
        self._ws = await websockets.connect(url)
        self._emit_audit("discord.gateway.connect", "allow", {"url": url})

    async def disconnect(self) -> None:
        """Close the Gateway connection gracefully."""
        self._running = False
        if self._heartbeat_task is not None:
            self._heartbeat_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._heartbeat_task
            self._heartbeat_task = None

        if self._ws is not None:
            try:
                await self._ws.close()
            except Exception as exc:
                logger.debug("Gateway close error (ignored): %s", exc)
            self._ws = None

        self._emit_audit("discord.gateway.disconnect", "allow", {})

    async def run(self) -> None:
        """Connect and run the event receive loop until disconnected.

        Automatically reconnects on transient errors.  Call
        :meth:`disconnect` to stop cleanly.
        """
        self._running = True
        while self._running:
            try:
                await self.connect()
                await self._receive_loop()
            except Exception as exc:
                if not self._running:
                    break
                logger.warning("Gateway disconnected: %s — reconnecting in 5s", exc)
                self._emit_audit(
                    "discord.gateway.disconnect",
                    "error",
                    {"error": str(exc)},
                )
                await asyncio.sleep(5)

    # ------------------------------------------------------------------
    # Internal loop
    # ------------------------------------------------------------------

    async def _receive_loop(self) -> None:
        """Read messages from the WebSocket until the connection closes."""
        async for raw in self._ws:
            if not self._running:
                break
            try:
                payload = json.loads(raw)
            except json.JSONDecodeError as exc:
                logger.warning("Gateway: invalid JSON payload: %s", exc)
                continue

            await self._handle_payload(payload)

    async def _handle_payload(self, payload: dict[str, Any]) -> None:
        """Route a Gateway payload to the appropriate handler."""
        op: int = payload.get("op", -1)
        data: Any = payload.get("d")
        seq: int | None = payload.get("s")
        event_name: str | None = payload.get("t")

        if seq is not None:
            self._sequence = seq

        if op == _OP_HELLO:
            heartbeat_interval = data["heartbeat_interval"] / 1000.0
            await self._start_heartbeat(heartbeat_interval)
            await self._identify_or_resume()

        elif op == _OP_DISPATCH:
            await self._handle_dispatch(event_name, data)

        elif op == _OP_HEARTBEAT:
            # Discord requests an immediate heartbeat.
            await self._send_heartbeat()

        elif op == _OP_HEARTBEAT_ACK:
            logger.debug("Gateway: heartbeat acknowledged")

        elif op == _OP_RECONNECT:
            logger.info("Gateway: server requested reconnect")
            await self._ws.close()

        elif op == _OP_INVALID_SESSION:
            resumable: bool = bool(data)
            logger.warning("Gateway: invalid session (resumable=%s)", resumable)
            if not resumable:
                self._discord_session_id = None
                self._resume_gateway_url = None
                self._sequence = None
            await asyncio.sleep(2)
            await self._ws.close()

        else:
            logger.debug("Gateway: unhandled opcode %d", op)

    async def _handle_dispatch(self, event_name: str | None, data: Any) -> None:
        """Handle a DISPATCH (opcode 0) event."""
        if event_name == "READY":
            self._discord_session_id = data.get("session_id")
            self._resume_gateway_url = data.get("resume_gateway_url")
            bot_user = data.get("user", {})
            self._bot_user_id = str(bot_user.get("id", ""))
            logger.info(
                "Gateway: READY as %s#%s (id=%s)",
                bot_user.get("username"),
                bot_user.get("discriminator"),
                self._bot_user_id,
            )
            self._emit_audit(
                "discord.gateway.connect",
                "allow",
                {"event": "READY", "bot_user_id": self._bot_user_id},
            )
            return

        if event_name == "RESUMED":
            self._emit_audit("discord.gateway.session_resumed", "allow", {})
            logger.info("Gateway: session resumed")
            return

        # Forward dispatched events to the callback.
        if event_name in (
            "MESSAGE_CREATE",
            "GUILD_CREATE",
            "INTERACTION_CREATE",
            "MESSAGE_REACTION_ADD",
        ):
            event_payload = {"t": event_name, "d": data}
            try:
                await self._on_message(event_payload)
            except Exception as exc:
                logger.exception("on_message callback raised: %s", exc)

    # ------------------------------------------------------------------
    # Heartbeat
    # ------------------------------------------------------------------

    async def _start_heartbeat(self, interval: float) -> None:
        """Cancel any existing heartbeat task and start a new one."""
        if self._heartbeat_task is not None:
            self._heartbeat_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._heartbeat_task

        self._heartbeat_task = asyncio.create_task(self._heartbeat_loop(interval))

    async def _heartbeat_loop(self, interval: float) -> None:
        """Send heartbeats on the given interval forever."""
        # Jitter: wait a random fraction of the interval before the first beat.
        import random

        await asyncio.sleep(interval * random.random())
        while True:
            await self._send_heartbeat()
            await asyncio.sleep(interval)

    async def _send_heartbeat(self) -> None:
        """Send a single heartbeat payload to the Gateway."""
        if self._ws is None:
            return
        payload = json.dumps({"op": _OP_HEARTBEAT, "d": self._sequence})
        try:
            await self._ws.send(payload)
            self._emit_audit("discord.gateway.heartbeat_sent", "allow", {"seq": self._sequence})
            logger.debug("Gateway: heartbeat sent (seq=%s)", self._sequence)
        except Exception as exc:
            logger.warning("Gateway: heartbeat failed: %s", exc)

    # ------------------------------------------------------------------
    # Identify / Resume
    # ------------------------------------------------------------------

    async def _identify_or_resume(self) -> None:
        """Send IDENTIFY or RESUME depending on session state."""
        if self._discord_session_id and self._sequence is not None:
            await self._send_resume()
        else:
            await self._send_identify()

    async def _send_identify(self) -> None:
        """Send the IDENTIFY payload to authenticate the bot."""
        payload = {
            "op": _OP_IDENTIFY,
            "d": {
                "token": self._token,
                "intents": _INTENTS,
                "properties": {
                    "os": "linux",
                    "browser": "missy",
                    "device": "missy",
                },
            },
        }
        await self._ws.send(json.dumps(payload))
        logger.debug("Gateway: IDENTIFY sent")

    async def _send_resume(self) -> None:
        """Send the RESUME payload to restore an existing session."""
        payload = {
            "op": _OP_RESUME,
            "d": {
                "token": self._token,
                "session_id": self._discord_session_id,
                "seq": self._sequence,
            },
        }
        await self._ws.send(json.dumps(payload))
        logger.info("Gateway: RESUME sent (seq=%s)", self._sequence)

    # ------------------------------------------------------------------
    # Audit helpers
    # ------------------------------------------------------------------

    def _emit_audit(self, event_type: str, result: str, detail: dict[str, Any]) -> None:
        """Publish an audit event onto the process-level event bus."""
        try:
            event = AuditEvent.now(
                session_id=self._session_id_audit,
                task_id=self._task_id_audit,
                event_type=event_type,
                category="network",
                result=result,  # type: ignore[arg-type]
                detail=detail,
            )
            event_bus.publish(event)
        except Exception as exc:
            logger.debug("Audit emit failed: %s", exc)
