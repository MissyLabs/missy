"""RATE-01 / RATE-03: Discord REST must not block the event loop, must cap
Retry-After, honor buckets/global limits, and stop before an IP ban."""

from __future__ import annotations

import asyncio
import contextlib
import time
from unittest.mock import MagicMock, patch

import pytest

from missy.channels.discord import rest as rest_mod
from missy.channels.discord.rest import (
    DiscordRateLimitedError,
    DiscordRestClient,
    rate_governor,
)


def _resp(status: int, headers: dict | None = None, body: dict | None = None):
    r = MagicMock()
    r.status_code = status
    r.headers = headers or {}
    r.json.return_value = body if body is not None else {"id": "1"}
    r.text = ""
    if status >= 400:
        import httpx

        r.raise_for_status.side_effect = httpx.HTTPStatusError(
            "err", request=MagicMock(), response=r
        )
    else:
        r.raise_for_status.return_value = None
    return r


def _client(http) -> DiscordRestClient:
    return DiscordRestClient(bot_token="t", http_client=http)


class TestRetryAfterCap:
    def test_huge_retry_after_fails_fast_without_sleeping(self):
        http = MagicMock()
        http.post.return_value = _resp(429, {"Retry-After": "3600"})
        with (
            patch.object(rest_mod.time, "sleep") as sleep,
            pytest.raises(DiscordRateLimitedError),
        ):
            _client(http).send_message("123", "hi")
        sleep.assert_not_called()

    def test_small_retry_after_is_honored(self):
        http = MagicMock()
        http.post.side_effect = [_resp(429, {"Retry-After": "0.5"}), _resp(200)]
        with patch.object(rest_mod.time, "sleep") as sleep:
            assert _client(http).send_message("123", "hi") == {"id": "1"}
        assert 0.5 <= sleep.call_args.args[0] <= 0.75

    def test_generic_request_path_also_capped(self):
        http = MagicMock()
        http.get.return_value = _resp(429, {"Retry-After": "999"})
        with patch.object(rest_mod.time, "sleep"), pytest.raises(DiscordRateLimitedError):
            _client(http).get_current_user()


class TestGovernor:
    def test_exhausted_bucket_prewaits(self):
        route = "POST https://discord.com/api/v10/channels/1/messages"
        rate_governor.after_response(
            route,
            _resp(
                200,
                {
                    "X-RateLimit-Bucket": "b1",
                    "X-RateLimit-Remaining": "0",
                    "X-RateLimit-Reset-After": "2.0",
                },
            ),
        )
        wait = rate_governor.wait_time(route)
        assert 1.5 < wait <= 2.0

    def test_global_429_pauses_every_route(self):
        rate_governor.after_response(
            "GET a", _resp(429, {"Retry-After": "5", "X-RateLimit-Global": "true"})
        )
        assert rate_governor.wait_time("POST something-else") > 4

    def test_401_opens_circuit_and_is_audited(self):
        with patch("missy.core.events.event_bus.publish") as publish:
            rate_governor.after_response("GET x", _resp(401))
        assert publish.call_args.args[0].event_type == "discord.rest.circuit_open"
        with pytest.raises(DiscordRateLimitedError, match="401"):
            rate_governor.before_request("GET y")

    def test_invalid_request_budget_opens_circuit(self, monkeypatch):
        monkeypatch.setattr(rest_mod, "INVALID_REQUEST_THRESHOLD", 5)
        for _ in range(5):
            rate_governor.after_response("GET x", _resp(403))
        with pytest.raises(DiscordRateLimitedError, match="invalid requests"):
            rate_governor.before_request("GET x")

    def test_shared_scope_429_not_counted_as_invalid(self, monkeypatch):
        monkeypatch.setattr(rest_mod, "INVALID_REQUEST_THRESHOLD", 2)
        for _ in range(3):
            rate_governor.after_response(
                "GET x", _resp(429, {"Retry-After": "0", "X-RateLimit-Scope": "shared"})
            )
        rate_governor.before_request("GET x")  # must not raise

    def test_mock_headers_are_ignored_safely(self):
        r = MagicMock()
        r.status_code = 200
        rate_governor.after_response("GET x", r)  # MagicMock headers -> no crash
        assert rate_governor.wait_time("GET x") == 0


class TestEventLoopNotBlocked:
    def test_slow_rest_call_does_not_stall_loop(self):
        from missy.channels.discord.channel import DiscordChannel

        channel = DiscordChannel.__new__(DiscordChannel)
        rest = MagicMock()

        def slow_typing(_channel_id):
            time.sleep(0.5)

        rest.trigger_typing.side_effect = slow_typing
        rest.send_message.return_value = {"id": "9"}
        channel._rest = rest

        async def scenario():
            stamps: list[float] = []

            async def heartbeat():
                for _ in range(10):
                    await asyncio.sleep(0.02)
                    stamps.append(time.monotonic())

            hb = asyncio.create_task(heartbeat())
            with contextlib.suppress(Exception):
                await channel.send_to("123", "hello")
            await hb
            return stamps

        start = time.monotonic()
        stamps = asyncio.run(scenario())
        rest.trigger_typing.assert_called_once()
        assert time.monotonic() - start >= 0.5  # the slow call really ran
        gaps = [b - a for a, b in zip([start, *stamps], stamps, strict=False)]
        # A blocked loop would show one ~0.5s gap; offloaded, ticks stay regular.
        assert max(gaps) < 0.3


class TestInteractionTokenExpiry:
    """Premortem: an expired interaction token (401) must not mute the bot."""

    def test_webhook_and_interaction_401s_do_not_open_circuit(self):
        base = "https://discord.com/api/v10"
        rate_governor.after_response(
            f"PATCH {base}/webhooks/123/tok/messages/@original", _resp(401)
        )
        rate_governor.after_response(f"POST {base}/interactions/9/tok/callback", _resp(401))
        rate_governor.before_request(f"POST {base}/channels/1/messages")  # must not raise

    def test_bot_route_401_still_opens_circuit(self):
        base = "https://discord.com/api/v10"
        with patch("missy.core.events.event_bus.publish"):
            rate_governor.after_response(f"GET {base}/users/@me", _resp(401))
        with pytest.raises(DiscordRateLimitedError):
            rate_governor.before_request(f"POST {base}/channels/1/messages")

    def test_edit_interaction_response_401_leaves_bot_usable(self):
        http = MagicMock()
        http.patch.return_value = _resp(401)
        client = _client(http)
        with contextlib.suppress(Exception):
            client.edit_interaction_response("123", "expired-token", "late answer")
        http.post.return_value = _resp(200)
        assert client.send_message("123", "hi") == {"id": "1"}
