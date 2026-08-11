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
import random
import time
from collections import deque
from collections.abc import Callable, Coroutine
from typing import Any
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

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

# Maximum WebSocket frame size for the Gateway connection (4 MB).
_MAX_WS_SIZE = 4 * 1024 * 1024

# Reconnect attempts must be paced even when a WebSocket closes cleanly.  Discord
# permits at most 1,000 IDENTIFY calls per 24 hours and resets the bot token when
# that global limit is exceeded, so an unbounded reconnect loop is destructive.
_RECONNECT_BASE_DELAY_SECONDS = 5.0
_RECONNECT_MAX_DELAY_SECONDS = 300.0
_RECONNECT_JITTER_RATIO = 0.2
_STABLE_CONNECTION_SECONDS = 60.0

# Discord resets a bot token after 1,000 IDENTIFY calls in 24 hours.  The
# authoritative /gateway/bot allowance is checked before every new session and
# a small reserve absorbs concurrent processes or an eventually-consistent
# counter.  The local fallback protects direct users of DiscordGatewayClient.
_IDENTIFY_SAFETY_RESERVE = 5
_LOCAL_IDENTIFY_LIMIT = 950
_IDENTIFY_WINDOW_SECONDS = 24 * 60 * 60

# Discord explicitly marks these close codes as non-reconnectable.  Retrying
# them cannot heal the connection and repeatedly consumes IDENTIFY calls.
_FATAL_CLOSE_CODES = frozenset({4004, 4010, 4011, 4012, 4013, 4014})

# These closes invalidate the old Gateway session.  Reconnecting with RESUME
# would only provoke INVALID_SESSION, so discard the cached session first.
# A *remote* 1000/1001 close doesn't itself invalidate Discord's session; only
# client-initiated 1000/1001 closes do, and disconnect() never reconnects.
_NON_RESUMABLE_CLOSE_CODES = frozenset({4003, 4007, 4009})

AsyncMessageCallback = Callable[[dict[str, Any]], Coroutine[Any, Any, None]]
GatewayInfoProvider = Callable[[], dict[str, Any]]


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
        gateway_info_provider: GatewayInfoProvider | None = None,
    ) -> None:
        if not bot_token.startswith("Bot "):
            bot_token = f"Bot {bot_token}"
        self._token = bot_token
        self._on_message = on_message
        self._gateway_url = gateway_url
        self._session_id_audit = session_id
        self._task_id_audit = task_id
        self._gateway_info_provider = gateway_info_provider

        # Runtime state
        self._ws: Any = None  # websockets.WebSocketClientProtocol
        self._heartbeat_task: asyncio.Task[None] | None = None
        self._sequence: int | None = None
        self._discord_session_id: str | None = None
        self._resume_gateway_url: str | None = None
        self._bot_user_id: str | None = None
        self._running: bool = False
        self._heartbeat_interval: float | None = None
        self._last_heartbeat_sent_at: float | None = None
        self._last_heartbeat_ack_at: float | None = None
        self._last_ready_at: float | None = None
        self._last_resume_sent_at: float | None = None
        self._last_resumed_at: float | None = None
        self._last_disconnect_at: float | None = None
        self._last_disconnect_error: str | None = None
        self._last_invalid_session_resumable: bool | None = None
        self._reconnect_count: int = 0
        self._resume_attempt_count: int = 0
        self._invalid_session_count: int = 0
        self._server_reconnect_count: int = 0
        self._consecutive_reconnect_failures: int = 0
        self._last_close_code: int | None = None
        self._last_close_reason: str | None = None
        self._terminal_close_code: int | None = None
        self._terminal_close_reason: str | None = None
        self._identify_count: int = 0
        self._last_identify_at: float | None = None
        self._identify_allowance_remaining: int | None = None
        self._identify_allowance_reset_at: float | None = None
        self._local_identify_times: deque[float] = deque()

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    @property
    def bot_user_id(self) -> str | None:
        """The Discord user ID of the connected bot, available after READY."""
        return self._bot_user_id

    def get_diagnostics(self) -> dict[str, Any]:
        """Return a redacted lifecycle snapshot for operator diagnostics."""
        self._ensure_diagnostic_state()
        now = time.time()

        def _age(ts: float | None) -> float | None:
            if ts is None:
                return None
            return max(0.0, round(now - ts, 3))

        heartbeat_task_active = self._heartbeat_task is not None and not self._heartbeat_task.done()
        heartbeat_ack_overdue = False
        if self._last_heartbeat_sent_at is not None:
            heartbeat_ack_overdue = (
                self._last_heartbeat_ack_at is None
                or self._last_heartbeat_ack_at < self._last_heartbeat_sent_at
            )

        return {
            "running": self._running,
            "connected": self._ws is not None,
            "heartbeat_task_active": heartbeat_task_active,
            "heartbeat_interval_seconds": self._heartbeat_interval,
            "heartbeat_ack_overdue": heartbeat_ack_overdue,
            "last_heartbeat_sent_age_seconds": _age(self._last_heartbeat_sent_at),
            "last_heartbeat_ack_age_seconds": _age(self._last_heartbeat_ack_at),
            "sequence": self._sequence,
            "discord_session_active": bool(self._discord_session_id),
            "resume_gateway_url_present": bool(self._resume_gateway_url),
            "bot_user_id": self._bot_user_id,
            "last_ready_age_seconds": _age(self._last_ready_at),
            "last_resume_sent_age_seconds": _age(self._last_resume_sent_at),
            "last_resumed_age_seconds": _age(self._last_resumed_at),
            "last_disconnect_age_seconds": _age(self._last_disconnect_at),
            "last_disconnect_error": self._last_disconnect_error,
            "last_invalid_session_resumable": self._last_invalid_session_resumable,
            "reconnect_count": self._reconnect_count,
            "resume_attempt_count": self._resume_attempt_count,
            "invalid_session_count": self._invalid_session_count,
            "server_reconnect_count": self._server_reconnect_count,
            "consecutive_reconnect_failures": self._consecutive_reconnect_failures,
            "last_close_code": self._last_close_code,
            "last_close_reason": self._last_close_reason,
            "terminal_close_code": self._terminal_close_code,
            "terminal_close_reason": self._terminal_close_reason,
            "identify_count": self._identify_count,
            "last_identify_age_seconds": _age(self._last_identify_at),
            "identify_allowance_remaining": self._identify_allowance_remaining,
            "identify_allowance_resets_in_seconds": (
                None
                if self._identify_allowance_reset_at is None
                else max(0.0, round(self._identify_allowance_reset_at - now, 3))
            ),
        }

    def _ensure_diagnostic_state(self) -> None:
        """Backfill lifecycle diagnostic fields for low-level test objects."""
        defaults: dict[str, Any] = {
            "_heartbeat_interval": None,
            "_last_heartbeat_sent_at": None,
            "_last_heartbeat_ack_at": None,
            "_last_ready_at": None,
            "_last_resume_sent_at": None,
            "_last_resumed_at": None,
            "_last_disconnect_at": None,
            "_last_disconnect_error": None,
            "_last_invalid_session_resumable": None,
            "_reconnect_count": 0,
            "_resume_attempt_count": 0,
            "_invalid_session_count": 0,
            "_server_reconnect_count": 0,
            "_consecutive_reconnect_failures": 0,
            "_last_close_code": None,
            "_last_close_reason": None,
            "_terminal_close_code": None,
            "_terminal_close_reason": None,
            "_identify_count": 0,
            "_last_identify_at": None,
            "_identify_allowance_remaining": None,
            "_identify_allowance_reset_at": None,
            "_local_identify_times": deque(),
        }
        for name, value in defaults.items():
            if not hasattr(self, name):
                setattr(self, name, value)

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

        url = (
            self._gateway_url_with_query(self._resume_gateway_url)
            if self._resume_gateway_url
            else self._gateway_url
        )
        logger.debug("Discord Gateway: connecting to %s", url)
        self._ws = await websockets.connect(url, max_size=_MAX_WS_SIZE)
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

        self._last_disconnect_at = time.time()
        self._emit_audit("discord.gateway.disconnect", "allow", {})

    async def run(self) -> None:
        """Connect and run the event receive loop until disconnected.

        Transient disconnects use bounded exponential backoff with jitter.
        Clean WebSocket closes are paced too: ``websockets`` ends its async
        iterator normally for close codes 1000 and 1001, so handling only the
        exception path would create a zero-delay reconnect loop.  Discord close
        codes documented as non-reconnectable stop this run rather than burning
        through the application's global IDENTIFY allowance.
        """
        if self._running:
            logger.warning("Discord Gateway run() ignored: client is already running")
            return

        self._running = True
        self._consecutive_reconnect_failures = 0
        self._terminal_close_code = None
        self._terminal_close_reason = None

        while self._running:
            connected_at: float | None = None
            close_code: int | None = None
            close_reason: str | None = None
            normal_close = False

            try:
                if not (self._discord_session_id and self._sequence is not None):
                    await self._wait_for_identify_allowance()
                await self.connect()
                connected_at = time.monotonic()
                await self._receive_loop()
                normal_close = True
                close_code, close_reason = self._close_details()
            except Exception as exc:
                if not self._running:
                    break
                close_code, close_reason = self._close_details(exc)
                if not close_reason:
                    close_reason = str(exc) or type(exc).__name__
            finally:
                await self._cleanup_connection()

            if not self._running:
                break

            self._last_close_code = close_code
            self._last_close_reason = close_reason

            if close_code in _FATAL_CLOSE_CODES:
                self._terminal_close_code = close_code
                self._terminal_close_reason = close_reason
                self._last_disconnect_at = time.time()
                self._last_disconnect_error = self._format_close_error(
                    close_code, close_reason, normal_close
                )
                self._emit_audit(
                    "discord.gateway.disconnect",
                    "error",
                    {
                        "error": self._last_disconnect_error,
                        "close_code": close_code,
                        "reconnect": False,
                    },
                )
                logger.error(
                    "Discord Gateway closed with non-reconnectable code %s (%s); "
                    "automatic reconnect stopped",
                    close_code,
                    close_reason or "no reason supplied",
                )
                self._running = False
                break

            # Preserve READY session state across remote 1000/1001 and
            # code-less disconnects so the next connection tries RESUME.  A
            # failed RESUME is safely resolved by Discord's INVALID_SESSION.
            if close_code in _NON_RESUMABLE_CLOSE_CODES:
                self._clear_resume_state()

            connected_seconds = time.monotonic() - connected_at if connected_at is not None else 0.0
            if connected_seconds >= _STABLE_CONNECTION_SECONDS:
                self._consecutive_reconnect_failures = 0

            self._consecutive_reconnect_failures += 1
            self._reconnect_count += 1
            self._last_disconnect_at = time.time()
            self._last_disconnect_error = self._format_close_error(
                close_code, close_reason, normal_close
            )
            delay = self._reconnect_delay()
            self._emit_audit(
                "discord.gateway.disconnect",
                "error",
                {
                    "error": self._last_disconnect_error,
                    "close_code": close_code,
                    "reconnect": True,
                    "reconnect_count": self._reconnect_count,
                    "delay_seconds": round(delay, 3),
                },
            )
            logger.warning(
                "Gateway disconnected: %s — reconnecting in %.1fs",
                self._last_disconnect_error,
                delay,
            )
            await asyncio.sleep(delay)

    async def _wait_for_identify_allowance(self) -> None:
        """Wait until a fresh IDENTIFY is safe under Discord's daily limit.

        ``GET /gateway/bot`` is authoritative and reflects the application's
        global allowance.  The provider is injected by :class:`DiscordChannel`
        so the request still passes through Missy's policy-enforced REST client.
        Direct users without a provider receive a conservative in-process
        rolling-window guard instead.
        """
        if self._gateway_info_provider is None:
            await self._wait_for_local_identify_allowance()
            return

        while True:
            info = await asyncio.to_thread(self._gateway_info_provider)
            gateway_url = info.get("url")
            if isinstance(gateway_url, str) and gateway_url:
                self._gateway_url = self._gateway_url_with_query(gateway_url)

            limit = info.get("session_start_limit")
            if not isinstance(limit, dict):
                raise RuntimeError("Discord /gateway/bot omitted session_start_limit")

            remaining = limit.get("remaining")
            reset_after_ms = limit.get("reset_after")
            if (
                not isinstance(remaining, int)
                or isinstance(remaining, bool)
                or not isinstance(reset_after_ms, (int, float))
                or isinstance(reset_after_ms, bool)
                or reset_after_ms < 0
            ):
                raise RuntimeError("Discord /gateway/bot returned invalid session_start_limit")

            reset_seconds = float(reset_after_ms) / 1000.0
            self._identify_allowance_remaining = remaining
            self._identify_allowance_reset_at = time.time() + reset_seconds
            if remaining > _IDENTIFY_SAFETY_RESERVE:
                return

            delay = max(1.0, reset_seconds) + random.uniform(1.0, 5.0)
            logger.error(
                "Discord IDENTIFY allowance is nearly exhausted (%d remaining); "
                "waiting %.1fs for the daily reset",
                remaining,
                delay,
            )
            self._emit_audit(
                "discord.gateway.identify_throttled",
                "error",
                {"remaining": remaining, "delay_seconds": round(delay, 3)},
            )
            await asyncio.sleep(delay)

    async def _wait_for_local_identify_allowance(self) -> None:
        """Conservatively guard IDENTIFY calls when no REST provider exists."""
        while True:
            now = time.monotonic()
            cutoff = now - _IDENTIFY_WINDOW_SECONDS
            while self._local_identify_times and self._local_identify_times[0] <= cutoff:
                self._local_identify_times.popleft()
            if len(self._local_identify_times) < _LOCAL_IDENTIFY_LIMIT:
                self._identify_allowance_remaining = _LOCAL_IDENTIFY_LIMIT - len(
                    self._local_identify_times
                )
                return

            delay = max(
                1.0,
                self._local_identify_times[0] + _IDENTIFY_WINDOW_SECONDS - now,
            )
            self._identify_allowance_reset_at = time.time() + delay
            logger.error(
                "Local Discord IDENTIFY guard reached %d calls; waiting %.1fs",
                _LOCAL_IDENTIFY_LIMIT,
                delay,
            )
            await asyncio.sleep(delay)

    @staticmethod
    def _gateway_url_with_query(url: str) -> str:
        """Preserve Discord's required API version and encoding when resuming."""
        parsed = urlsplit(url)
        query = dict(parse_qsl(parsed.query, keep_blank_values=True))
        query.setdefault("v", "10")
        query.setdefault("encoding", "json")
        return urlunsplit(
            (parsed.scheme, parsed.netloc, parsed.path, urlencode(query), parsed.fragment)
        )

    async def _cleanup_connection(self) -> None:
        """Release per-connection state without invalidating a resumable session."""
        heartbeat_task = self._heartbeat_task
        self._heartbeat_task = None
        if heartbeat_task is not None and heartbeat_task is not asyncio.current_task():
            heartbeat_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await heartbeat_task

        ws = self._ws
        self._ws = None
        if ws is not None:
            # 4000 preserves a valid Discord session, unlike a normal 1000/1001
            # close.  Calling close() on an already-closed websocket is safe.
            with contextlib.suppress(Exception):
                await ws.close(code=4000, reason="gateway reconnect")

        # Heartbeat ACK timestamps belong to one physical WebSocket only.  If
        # they leak into the next connection, its heartbeat loop can interpret
        # the previous socket's missing ACK as an immediate new failure.
        self._last_heartbeat_sent_at = None
        self._last_heartbeat_ack_at = None
        self._heartbeat_interval = None

    def _close_details(self, exc: Exception | None = None) -> tuple[int | None, str | None]:
        """Extract a Discord close code/reason across websockets versions."""
        sources: list[Any] = []
        if exc is not None:
            received = getattr(exc, "rcvd", None)
            if received is not None:
                sources.append(received)
            sources.append(exc)
        if self._ws is not None:
            sources.append(self._ws)

        code: int | None = None
        reason: str | None = None
        for source in sources:
            candidate = getattr(source, "code", None)
            if code is None and isinstance(candidate, int) and not isinstance(candidate, bool):
                code = candidate
            candidate_reason = getattr(source, "reason", None)
            if reason is None and isinstance(candidate_reason, str) and candidate_reason:
                reason = candidate_reason

            # Modern websockets connections expose close_code/close_reason;
            # ConnectionClosed exceptions expose the received Close frame.
            candidate = getattr(source, "close_code", None)
            if code is None and isinstance(candidate, int) and not isinstance(candidate, bool):
                code = candidate
            candidate_reason = getattr(source, "close_reason", None)
            if reason is None and isinstance(candidate_reason, str) and candidate_reason:
                reason = candidate_reason

        return code, reason

    def _clear_resume_state(self) -> None:
        """Forget a Discord session that cannot be resumed."""
        self._discord_session_id = None
        self._resume_gateway_url = None
        self._sequence = None

    def _reconnect_delay(self) -> float:
        """Return bounded exponential backoff plus positive jitter."""
        exponent = min(max(self._consecutive_reconnect_failures - 1, 0), 16)
        base = min(
            _RECONNECT_BASE_DELAY_SECONDS * (2**exponent),
            _RECONNECT_MAX_DELAY_SECONDS,
        )
        return base + random.uniform(0.0, base * _RECONNECT_JITTER_RATIO)

    @staticmethod
    def _format_close_error(
        close_code: int | None,
        close_reason: str | None,
        normal_close: bool,
    ) -> str:
        description = "Gateway closed cleanly" if normal_close else "Gateway connection failed"
        if close_code is not None:
            description += f" (code={close_code})"
        if close_reason:
            description += f": {close_reason}"
        return description

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
        self._ensure_diagnostic_state()
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
            self._last_heartbeat_ack_at = time.time()
            self._emit_audit(
                "discord.gateway.heartbeat_ack",
                "allow",
                {"seq": self._sequence},
            )
            logger.debug("Gateway: heartbeat acknowledged")

        elif op == _OP_RECONNECT:
            logger.info("Gateway: server requested reconnect")
            self._server_reconnect_count += 1
            self._emit_audit(
                "discord.gateway.reconnect_requested",
                "allow",
                {"server_reconnect_count": self._server_reconnect_count},
            )
            # A normal 1000/1001 close invalidates the session.  Use a
            # reconnectable application close code so the next connection can
            # send RESUME instead of consuming a new IDENTIFY.
            await self._ws.close(code=4000, reason="server requested reconnect")

        elif op == _OP_INVALID_SESSION:
            resumable: bool = bool(data)
            logger.warning("Gateway: invalid session (resumable=%s)", resumable)
            self._invalid_session_count += 1
            self._last_invalid_session_resumable = resumable
            if not resumable:
                self._discord_session_id = None
                self._resume_gateway_url = None
                self._sequence = None
            self._emit_audit(
                "discord.gateway.invalid_session",
                "error",
                {
                    "resumable": resumable,
                    "invalid_session_count": self._invalid_session_count,
                },
            )
            await asyncio.sleep(2)
            await self._ws.close(code=4000, reason="invalid session")

        else:
            logger.debug("Gateway: unhandled opcode %d", op)

    async def _handle_dispatch(self, event_name: str | None, data: Any) -> None:
        """Handle a DISPATCH (opcode 0) event."""
        self._ensure_diagnostic_state()
        if event_name == "READY":
            self._discord_session_id = data.get("session_id")
            self._resume_gateway_url = data.get("resume_gateway_url")
            bot_user = data.get("user", {})
            self._bot_user_id = str(bot_user.get("id", ""))
            self._last_ready_at = time.time()
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
            self._last_resumed_at = time.time()
            self._emit_audit(
                "discord.gateway.session_resumed",
                "allow",
                {"resume_attempt_count": self._resume_attempt_count},
            )
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
        self._ensure_diagnostic_state()
        self._heartbeat_interval = interval
        if self._heartbeat_task is not None:
            self._heartbeat_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._heartbeat_task

        self._last_heartbeat_sent_at = None
        self._last_heartbeat_ack_at = None
        self._heartbeat_task = asyncio.create_task(self._heartbeat_loop(interval))

    async def _heartbeat_loop(self, interval: float) -> None:
        """Send heartbeats on the given interval forever.

        Discord's Gateway protocol requires that if an ACK for the
        previous heartbeat hasn't arrived by the time the next one is
        due, the client must close the connection (non-1000 code) and
        reconnect rather than keep heartbeating -- otherwise a half-open
        TCP connection (sends succeed locally, nothing ever arrives back)
        leaves the client sitting in a zombie session indefinitely,
        appearing "connected" while receiving nothing, until the process
        is restarted. ``get_diagnostics()``'s ``heartbeat_ack_overdue``
        already computes this exact condition; this closes the loop by
        actually acting on it instead of only surfacing it as a metric.
        """
        # Jitter: wait a random fraction of the interval before the first beat.
        import secrets

        await asyncio.sleep(interval * secrets.SystemRandom().random())
        while True:
            if self._last_heartbeat_sent_at is not None and (
                self._last_heartbeat_ack_at is None
                or self._last_heartbeat_ack_at < self._last_heartbeat_sent_at
            ):
                logger.warning(
                    "Gateway: heartbeat ACK not received before next heartbeat "
                    "was due; closing connection to force a reconnect."
                )
                self._emit_audit("discord.gateway.heartbeat_ack_missed", "error", {})
                self._reconnect_count += 1
                self._last_disconnect_at = time.time()
                self._last_disconnect_error = "heartbeat ACK timeout"
                if self._ws is not None:
                    with contextlib.suppress(Exception):
                        await self._ws.close(code=4000, reason="heartbeat ack timeout")
                return
            await self._send_heartbeat()
            await asyncio.sleep(interval)

    async def _send_heartbeat(self) -> None:
        """Send a single heartbeat payload to the Gateway."""
        self._ensure_diagnostic_state()
        if self._ws is None:
            return
        payload = json.dumps({"op": _OP_HEARTBEAT, "d": self._sequence})
        try:
            await self._ws.send(payload)
            self._last_heartbeat_sent_at = time.time()
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
        now_monotonic = time.monotonic()
        self._local_identify_times.append(now_monotonic)
        self._identify_count += 1
        self._last_identify_at = time.time()
        if self._identify_allowance_remaining is not None:
            self._identify_allowance_remaining = max(0, self._identify_allowance_remaining - 1)
        self._emit_audit(
            "discord.gateway.identify_sent",
            "allow",
            {"identify_count": self._identify_count},
        )
        logger.debug("Gateway: IDENTIFY sent")

    async def _send_resume(self) -> None:
        """Send the RESUME payload to restore an existing session."""
        self._ensure_diagnostic_state()
        payload = {
            "op": _OP_RESUME,
            "d": {
                "token": self._token,
                "session_id": self._discord_session_id,
                "seq": self._sequence,
            },
        }
        await self._ws.send(json.dumps(payload))
        self._resume_attempt_count += 1
        self._last_resume_sent_at = time.time()
        self._emit_audit(
            "discord.gateway.resume_sent",
            "allow",
            {"seq": self._sequence, "resume_attempt_count": self._resume_attempt_count},
        )
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
