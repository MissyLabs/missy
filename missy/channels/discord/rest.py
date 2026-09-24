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
import re
import secrets
import threading
import time
from collections import OrderedDict, deque
from typing import Any

from missy.gateway.client import PolicyHTTPClient, create_client

logger = logging.getLogger(__name__)

#: Discord REST API base URL.
BASE = "https://discord.com/api/v10"


_MENTION_ID_RE = re.compile(r"<@!?(\d+)>|<@&(\d+)>|<#(\d+)>")

#: Discord snowflake IDs are 64-bit integers (up to 20 digits).
_SNOWFLAKE_RE = re.compile(r"^\d{1,20}$")


def _validate_snowflake(value: str, name: str = "id") -> str:
    """Validate that *value* is a valid Discord snowflake ID.

    Raises:
        ValueError: When the value is not a valid snowflake ID.
    """
    if not _SNOWFLAKE_RE.match(value):
        raise ValueError(
            f"Invalid Discord {name}: {value!r}. Expected a numeric snowflake ID (1-20 digits)."
        )
    return value


def _mask_mentions(s: str) -> str:
    """Redact snowflake IDs inside common mention tokens for safer logging."""
    return _MENTION_ID_RE.sub(lambda m: re.sub(r"\d+", "redacted", m.group(0)), s or "")


#: Longest ``Retry-After`` (seconds) a single call will sleep for (RATE-01).
#: Discord can answer a 429 with a very long Retry-After; sleeping through it
#: pins a worker thread (and previously the whole gateway event loop) for
#: minutes. Longer waits fail fast so the caller can surface "rate limited".
MAX_RETRY_AFTER_SECONDS = 30.0

#: Discord bans an IP (Cloudflare, ~1h) after 10,000 invalid (401/403/429)
#: requests in 10 minutes. Stop well before that (RATE-03).
INVALID_REQUEST_WINDOW_SECONDS = 600.0
INVALID_REQUEST_THRESHOLD = 1000

#: How long all REST calls are refused after an HTTP 401 (bad/revoked token):
#: retrying can never succeed and only burns the invalid-request budget.
UNAUTHORIZED_COOLDOWN_SECONDS = 600.0

_MAX_TRACKED_ROUTES = 1000


class DiscordRateLimitedError(RuntimeError):
    """Raised instead of sleeping when Discord asks for a wait above the cap,
    or while the invalid-request circuit is open."""


def _header(response: Any, name: str) -> str | None:
    headers = getattr(response, "headers", None)
    if headers is None:
        return None
    try:
        value = headers.get(name)
    except Exception:
        return None
    return value if isinstance(value, str) else None


def _header_float(response: Any, name: str) -> float | None:
    value = _header(response, name)
    if value is None:
        return None
    try:
        return float(value)
    except ValueError:
        return None


class _DiscordRateGovernor:
    """Process-wide Discord REST rate-limit state (RATE-01 / RATE-03).

    Shared by every :class:`DiscordRestClient` so all accounts/threads in the
    process see the same global pause, per-bucket exhaustion, and
    invalid-request budget. Thread-safe.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._global_until = 0.0
        self._circuit_until = 0.0
        self._circuit_reason = ""
        self._route_bucket: OrderedDict[str, str] = OrderedDict()
        self._bucket_reset: dict[str, tuple[int, float]] = {}
        self._invalid: deque[float] = deque()

    def reset(self) -> None:
        with self._lock:
            self.__init__()

    def wait_time(self, route: str) -> float:
        """Seconds to wait before *route* may be called (raises if circuit open)."""
        now = time.monotonic()
        with self._lock:
            if now < self._circuit_until:
                raise DiscordRateLimitedError(
                    f"Discord REST circuit open ({self._circuit_reason}); "
                    f"retry in {self._circuit_until - now:.0f}s"
                )
            wait = max(0.0, self._global_until - now)
            bucket = self._route_bucket.get(route)
            if bucket is not None:
                remaining, reset_at = self._bucket_reset.get(bucket, (1, 0.0))
                if remaining <= 0 and reset_at > now:
                    wait = max(wait, reset_at - now)
        return wait

    def before_request(self, route: str) -> None:
        wait = self.wait_time(route)
        if wait <= 0:
            return
        if wait > MAX_RETRY_AFTER_SECONDS:
            raise DiscordRateLimitedError(
                f"Discord rate limit for {route} resets in {wait:.0f}s (over the "
                f"{MAX_RETRY_AFTER_SECONDS:.0f}s cap); not waiting"
            )
        time.sleep(wait)

    def after_response(self, route: str, response: Any) -> None:
        status = getattr(response, "status_code", None)
        if not isinstance(status, int):
            return
        now = time.monotonic()
        opened: str | None = None
        with self._lock:
            bucket = _header(response, "X-RateLimit-Bucket")
            remaining = _header_float(response, "X-RateLimit-Remaining")
            reset_after = _header_float(response, "X-RateLimit-Reset-After")
            if bucket:
                self._route_bucket[route] = bucket
                self._route_bucket.move_to_end(route)
                while len(self._route_bucket) > _MAX_TRACKED_ROUTES:
                    old_route, old_bucket = self._route_bucket.popitem(last=False)
                    if old_bucket not in self._route_bucket.values():
                        self._bucket_reset.pop(old_bucket, None)
                if remaining is not None and reset_after is not None:
                    self._bucket_reset[bucket] = (int(remaining), now + reset_after)
            if status == 429:
                is_global = (_header(response, "X-RateLimit-Global") or "").lower() == "true" or (
                    _header(response, "X-RateLimit-Scope") == "global"
                )
                retry_after = _header_float(response, "Retry-After") or 1.0
                if is_global:
                    self._global_until = max(self._global_until, now + retry_after)
            if status in (401, 403, 429) and _header(response, "X-RateLimit-Scope") != "shared":
                self._invalid.append(now)
            while self._invalid and now - self._invalid[0] > INVALID_REQUEST_WINDOW_SECONDS:
                self._invalid.popleft()
            if status == 401 and now >= self._circuit_until:
                self._circuit_until = now + UNAUTHORIZED_COOLDOWN_SECONDS
                self._circuit_reason = "HTTP 401: bot token rejected"
                opened = self._circuit_reason
            elif len(self._invalid) >= INVALID_REQUEST_THRESHOLD and now >= self._circuit_until:
                self._circuit_until = self._invalid[0] + INVALID_REQUEST_WINDOW_SECONDS
                self._circuit_reason = (
                    f"{len(self._invalid)} invalid requests in "
                    f"{INVALID_REQUEST_WINDOW_SECONDS:.0f}s"
                )
                opened = self._circuit_reason
        if opened:
            logger.error("Discord REST circuit opened: %s", opened)
            try:
                from missy.core.events import AuditEvent, event_bus

                event_bus.publish(
                    AuditEvent.now(
                        session_id="discord",
                        task_id="rest",
                        event_type="discord.rest.circuit_open",
                        category="network",
                        result="deny",
                        detail={"reason": opened},
                    )
                )
            except Exception:
                logger.debug("Could not publish circuit_open audit event", exc_info=True)


#: The process-wide governor instance.
rate_governor = _DiscordRateGovernor()


def _route_key(method: str, url: str) -> str:
    return f"{method.upper()} {url.split('?', 1)[0]}"


def _retry_after_delay(response: Any) -> float | None:
    """Parse Retry-After; raise if it exceeds :data:`MAX_RETRY_AFTER_SECONDS`."""
    delay = _header_float(response, "Retry-After")
    if delay is None:
        return None
    if delay > MAX_RETRY_AFTER_SECONDS:
        raise DiscordRateLimitedError(
            f"Discord asked to retry after {delay:.0f}s (over the "
            f"{MAX_RETRY_AFTER_SECONDS:.0f}s cap); giving up on this request"
        )
    return delay


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
        http_client: PolicyHTTPClient | None = None,
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

    def _headers(self, extra: dict[str, str] | None = None) -> dict[str, str]:
        """Build request headers with the bot token injected."""
        hdrs: dict[str, str] = {
            "Authorization": self._token,
            "Content-Type": "application/json",
            "User-Agent": "MissyBot (https://github.com/missy, 0.1)",
        }
        if extra:
            hdrs.update(extra)
        return hdrs

    _RETRY_STATUSES = frozenset({429, 502, 503, 504})
    _RETRY_BACKOFFS = (1.0, 2.0, 4.0)

    def _request_with_retry(self, method: str, url: str, **kwargs: Any) -> Any:
        """Issue an HTTP request, retrying transient Discord statuses.

        Applies the same Retry-After-aware backoff strategy
        :meth:`send_message` uses to every other REST call. Previously only
        ``send_message`` retried on 429/502/503/504 -- every other method
        (``add_reaction``, ``get_guild_roles``, ``create_thread``, etc.)
        called ``response.raise_for_status()`` directly with no retry at
        all, so a single transient rate-limit or upstream hiccup on any of
        those routes failed the whole operation immediately instead of
        recovering the way a chat message send already did.

        Returns the final response (2xx, a non-retryable error status, or
        the last retryable-status response once retries are exhausted) so
        the caller's own ``response.raise_for_status()``/status-code checks
        behave exactly as if no retry had happened.
        """
        attempt_count = len(self._RETRY_BACKOFFS) + 1
        response = None
        route = _route_key(method, url)
        for attempt in range(attempt_count):
            rate_governor.before_request(route)
            response = getattr(self._http, method)(url, **kwargs)
            rate_governor.after_response(route, response)
            if response.status_code not in self._RETRY_STATUSES:
                return response
            if attempt >= len(self._RETRY_BACKOFFS):
                return response

            delay: float | None = None
            if response.status_code == 429:
                delay = _retry_after_delay(response)
            if delay is None:
                delay = self._RETRY_BACKOFFS[attempt]

            delay = float(delay) + secrets.SystemRandom().uniform(0.0, 0.25)
            logger.warning(
                "Discord %s %s transient HTTP %d; retrying in %.2fs (attempt %d/%d)",
                method.upper(),
                url,
                response.status_code,
                delay,
                attempt + 1,
                attempt_count,
            )
            time.sleep(delay)
        return response

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
        response = self._request_with_retry("get", url, headers=self._headers())
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
        response = self._request_with_retry("get", url, headers=self._headers())
        response.raise_for_status()
        return response.json()

    def send_message(
        self,
        channel_id: str,
        content: str,
        reply_to_message_id: str | None = None,
        mention_user_ids: list[str] | None = None,
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
        _validate_snowflake(channel_id, "channel_id")
        if reply_to_message_id is not None:
            _validate_snowflake(reply_to_message_id, "reply_to_message_id")
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

        def _log_final_failure(
            *, response: Any | None, exc: Exception | None, attempt_index: int
        ) -> None:
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

        route = _route_key("post", url)
        for attempt in range(attempt_count):
            response = None
            try:
                rate_governor.before_request(route)
                response = self._http.post(url, headers=self._headers(), json=body)
                rate_governor.after_response(route, response)

                if response.status_code in retry_statuses:
                    delay: float | None = None
                    if response.status_code == 429:
                        delay = _retry_after_delay(response)

                    # Exhaustion check must happen regardless of where
                    # *delay* came from. Previously this was nested inside
                    # `if delay is None:`, so a delay sourced from a real
                    # Retry-After header (the common case under sustained
                    # 429 rate-limiting) skipped it entirely even on the
                    # final allotted attempt -- the loop just slept and
                    # looped one more time, the for-loop then ended, and
                    # execution fell through to a bare, uninformative
                    # "failed without exception" RuntimeError instead of
                    # the real, logged httpx.HTTPStatusError every other
                    # exhaustion path produces (and instead of ever calling
                    # _log_final_failure at all).
                    if attempt >= len(backoffs):
                        _log_final_failure(response=response, exc=None, attempt_index=attempt_count)
                        response.raise_for_status()

                    if delay is None:
                        delay = backoffs[attempt]

                    delay = float(delay) + secrets.SystemRandom().uniform(0.0, 0.25)
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

            except DiscordRateLimitedError as exc:
                # Retrying can't help within the cap -- fail fast (RATE-01).
                _log_final_failure(response=response, exc=exc, attempt_index=attempt + 1)
                raise
            except Exception as exc:
                if attempt >= len(backoffs):
                    _log_final_failure(response=response, exc=exc, attempt_index=attempt_count)
                    raise
                delay = backoffs[attempt] + secrets.SystemRandom().uniform(0.0, 0.25)
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

        _validate_snowflake(channel_id, "channel_id")
        path = Path(file_path).expanduser()
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        mime = mimetypes.guess_type(str(path))[0] or "application/octet-stream"
        url = f"{BASE}/channels/{channel_id}/messages"
        headers = {k: v for k, v in self._headers().items() if k != "Content-Type"}

        with path.open("rb") as fh:
            files = {"file": (path.name, fh, mime)}
            data = {"content": caption} if caption else {}
            route = _route_key("post", url)
            rate_governor.before_request(route)
            response = self._http.post(url, headers=headers, files=files, data=data, timeout=60)
            rate_governor.after_response(route, response)
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

        _validate_snowflake(channel_id, "channel_id")
        _validate_snowflake(message_id, "message_id")
        encoded = quote(emoji, safe="")
        url = f"{BASE}/channels/{channel_id}/messages/{message_id}/reactions/{encoded}/@me"
        # Discord expects a PUT with empty body; returns 204 No Content.
        response = self._request_with_retry(
            "put",
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
        _validate_snowflake(channel_id, "channel_id")
        url = f"{BASE}/channels/{channel_id}/typing"
        route = _route_key("post", url)
        try:
            rate_governor.before_request(route)
            response = self._http.post(url, headers=self._headers())
            rate_governor.after_response(route, response)
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
        _validate_snowflake(channel_id, "channel_id")
        _validate_snowflake(message_id, "message_id")
        url = f"{BASE}/channels/{channel_id}/messages/{message_id}"
        try:
            response = self._request_with_retry(
                "delete",
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
        message_id: str | None = None,
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
        _validate_snowflake(channel_id, "channel_id")
        if message_id:
            _validate_snowflake(message_id, "message_id")
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
        response = self._request_with_retry("post", url, headers=self._headers(), json=body)
        response.raise_for_status()
        return response.json()

    def get_channel(self, channel_id: str) -> dict[str, Any]:
        """Fetch a channel object by ID.

        Args:
            channel_id: The channel snowflake ID.

        Returns:
            The Discord Channel object as a dict.
        """
        _validate_snowflake(channel_id, "channel_id")
        url = f"{BASE}/channels/{channel_id}"
        response = self._request_with_retry("get", url, headers=self._headers())
        response.raise_for_status()
        return response.json()

    def get_guild_roles(self, guild_id: str) -> list[dict[str, Any]]:
        """Fetch the list of roles defined in a guild.

        Used to resolve the role ID snowflakes carried on a message's
        ``member.roles`` field to the human-readable role names that
        ``DiscordGuildPolicy.allowed_roles`` is configured with (task
        #12/allowed_roles enforcement).

        Args:
            guild_id: The guild snowflake ID.

        Returns:
            A list of Discord Role objects (each with ``id``/``name``).
        """
        _validate_snowflake(guild_id, "guild_id")
        url = f"{BASE}/guilds/{guild_id}/roles"
        response = self._request_with_retry("get", url, headers=self._headers())
        response.raise_for_status()
        return response.json()

    def search_guild_members(
        self, guild_id: str, query: str, limit: int = 10
    ) -> list[dict[str, Any]]:
        """Search guild members by username/nickname prefix.

        Backs :class:`~missy.tools.builtin.discord_lookup_user.DiscordLookupUserTool`:
        resolving a display name to the real numeric snowflake ID Discord's
        ``<@ID>`` mention syntax requires (a bare ``@name`` in outbound
        message content never pings anyone). Only useful for someone who
        hasn't spoken recently in the channel -- the agent's own
        conversation context already carries real IDs for recent speakers,
        so this is the fallback path, not the common one.

        Args:
            guild_id: The guild snowflake ID to search within.
            query: Username/nickname prefix to match (Discord does a
                case-insensitive prefix match, not a substring search).
            limit: Max results (1-1000 per Discord's API; default 10).

        Returns:
            A list of Discord Guild Member objects (each with a nested
            ``user`` dict carrying ``id``/``username``/``global_name``).
        """
        _validate_snowflake(guild_id, "guild_id")
        if not query:
            raise ValueError("query must not be empty")
        limit = max(1, min(1000, limit))
        url = f"{BASE}/guilds/{guild_id}/members/search"
        params: dict[str, Any] = {"query": query, "limit": limit}
        response = self._request_with_retry("get", url, headers=self._headers(), params=params)
        response.raise_for_status()
        return response.json()

    def send_interaction_response(
        self,
        interaction_id: str,
        interaction_token: str,
        response_type: int,
        data: dict[str, Any] | None = None,
    ) -> None:
        """Send an initial response to a Discord interaction.

        Args:
            interaction_id: The interaction snowflake ID.
            interaction_token: The interaction token.
            response_type: The interaction callback type (e.g. 4 or 5).
            data: Optional response data payload.
        """
        _validate_snowflake(interaction_id, "interaction_id")
        url = f"{BASE}/interactions/{interaction_id}/{interaction_token}/callback"
        body: dict[str, Any] = {"type": response_type}
        if data is not None:
            body["data"] = data
        response = self._request_with_retry("post", url, headers=self._headers(), json=body)
        response.raise_for_status()

    def edit_interaction_response(
        self,
        application_id: str,
        interaction_token: str,
        content: str,
    ) -> dict[str, Any]:
        """Edit the original interaction response (for deferred replies).

        Args:
            application_id: The Discord application ID.
            interaction_token: The interaction token.
            content: The message content to set.

        Returns:
            The updated message object as a dict.
        """
        _validate_snowflake(application_id, "application_id")
        url = f"{BASE}/webhooks/{application_id}/{interaction_token}/messages/@original"
        response = self._request_with_retry(
            "patch",
            url,
            headers=self._headers(),
            json={"content": content[:2000]},
        )
        response.raise_for_status()
        return response.json()

    def get_channel_messages(
        self,
        channel_id: str,
        limit: int = 10,
        before: str | None = None,
    ) -> list[dict[str, Any]]:
        """Fetch recent messages from a Discord channel.

        Args:
            channel_id: The channel snowflake ID.
            limit: Number of messages to fetch (1-100, default 10).
            before: Optional message ID to fetch messages before.

        Returns:
            List of Discord Message objects (newest first).
        """
        _validate_snowflake(channel_id, "channel_id")
        if before:
            _validate_snowflake(before, "before")
        limit = max(1, min(100, limit))
        url = f"{BASE}/channels/{channel_id}/messages"
        params: dict[str, Any] = {"limit": limit}
        if before:
            params["before"] = before
        response = self._request_with_retry("get", url, headers=self._headers(), params=params)
        response.raise_for_status()
        return response.json()

    def download_attachment(self, url: str, timeout: int = 30) -> bytes:
        """Download a file from a Discord CDN URL.

        Args:
            url: The attachment URL (cdn.discordapp.com or media.discordapp.net).
            timeout: Request timeout in seconds.

        Returns:
            Raw file bytes.

        Raises:
            ValueError: If the URL is not a Discord CDN URL.
            httpx.HTTPStatusError: On non-2xx responses.
        """
        from urllib.parse import urlparse

        parsed = urlparse(url)
        if parsed.hostname not in (
            "cdn.discordapp.com",
            "media.discordapp.net",
        ):
            raise ValueError(
                f"Not a Discord CDN URL: {url!r}. "
                "Only cdn.discordapp.com and media.discordapp.net are allowed."
            )
        response = self._request_with_retry("get", url, timeout=timeout)
        response.raise_for_status()
        return response.content

    def register_slash_commands(
        self,
        application_id: str,
        commands: list[dict[str, Any]],
        guild_id: str | None = None,
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
        response = self._request_with_retry("put", url, headers=self._headers(), json=commands)
        response.raise_for_status()
        return response.json()
