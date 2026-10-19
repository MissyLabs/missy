"""openai-codex usage-limit 429s put the provider in a cooldown until reset.

Live validation: Codex answered HTTP 429 ``usage_limit_reached`` with
``resets_in_seconds`` ~3h. Missy ignored it and kept trying Codex first on
every turn (15 times over 3 hours), each attempt sleeping up to
``max_wait_seconds`` in the rate limiter before falling back.
"""

from __future__ import annotations

import json
import time
from unittest.mock import MagicMock, patch

import httpx
import pytest

from missy.config.settings import ProviderConfig
from missy.core.exceptions import ProviderError
from missy.providers.base import BaseProvider, CompletionResponse, Message
from missy.providers.codex_provider import CodexProvider, _usage_limit_reset_seconds
from missy.providers.health import ProviderFailureClass, classify_provider_error
from missy.providers.rate_limiter import RateLimiter
from missy.providers.registry import ProviderRegistry

THREE_HOURS = 3 * 3600


def _usage_limit_error(
    *, resets_in: float | None = THREE_HOURS, resets_at: float | None = None, headers=None
) -> httpx.HTTPStatusError:
    err: dict = {"type": "usage_limit_reached", "message": "The usage limit has been reached"}
    if resets_in is not None:
        err["resets_in_seconds"] = resets_in
    if resets_at is not None:
        err["resets_at"] = resets_at
    request = httpx.Request("POST", "https://chatgpt.com/backend-api/codex/responses")
    response = httpx.Response(
        429, request=request, headers=headers or {}, content=json.dumps({"error": err}).encode()
    )
    return httpx.HTTPStatusError("429 Too Many Requests", request=request, response=response)


def _plain_429(retry_after: str) -> httpx.HTTPStatusError:
    request = httpx.Request("POST", "https://chatgpt.com/backend-api/codex/responses")
    response = httpx.Response(
        429, request=request, headers={"retry-after": retry_after}, content=b"slow down"
    )
    return httpx.HTTPStatusError("429", request=request, response=response)


def _msgs() -> list[Message]:
    return [Message(role="user", content="hi")]


def _single() -> CodexProvider:
    return CodexProvider(ProviderConfig(name="openai-codex", model="gpt-5.2", api_key="tok"))


def _multi() -> CodexProvider:
    return CodexProvider(
        ProviderConfig(
            name="openai-codex",
            model="gpt-5.2",
            api_key=None,
            oauth_accounts=["work", "personal"],
            key_rotation_strategy="round_robin",
        )
    )


class _Fallback(BaseProvider):
    name = "fallback"

    def complete(self, messages, **kwargs):
        return CompletionResponse(
            content="fallback answer", model="m", provider="fallback", usage={}, raw={}
        )

    def is_available(self):
        return True


class TestParse:
    def test_resets_in_seconds(self):
        assert _usage_limit_reset_seconds(_usage_limit_error()) == THREE_HOURS

    def test_resets_at_epoch(self):
        now = 1_000_000.0
        exc = _usage_limit_error(resets_in=None, resets_at=now + 600)
        assert _usage_limit_reset_seconds(exc, now=now) == pytest.approx(600)

    def test_retry_after_header(self):
        exc = _usage_limit_error(resets_in=None, headers={"retry-after": "7200"})
        assert _usage_limit_reset_seconds(exc) == 7200

    def test_no_reset_info_uses_default(self):
        assert _usage_limit_reset_seconds(_usage_limit_error(resets_in=None)) > 0

    def test_absurd_reset_is_bounded(self):
        assert _usage_limit_reset_seconds(_usage_limit_error(resets_in=1e12)) <= 7 * 24 * 3600

    def test_plain_429_is_not_usage_limit(self):
        assert _usage_limit_reset_seconds(_plain_429("5")) is None


def _run_complete(provider, error, mock_cls):
    mock_cls.return_value.post.side_effect = error
    with pytest.raises(ProviderError) as info:
        provider.complete(_msgs())
    return info.value


class TestCooldown:
    def test_usage_limit_enters_cooldown_without_sleeping(self):
        provider = _single()
        with (
            patch("missy.providers.codex_provider.PolicyHTTPClient") as mock_cls,
            patch("missy.providers.codex_provider._load_oauth_token", return_value=None),
            patch.object(RateLimiter, "on_rate_limit_response") as limiter_sleep,
        ):
            err = _run_complete(provider, _usage_limit_error(), mock_cls)
            assert "usage limit" in str(err)
            assert classify_provider_error(err) == ProviderFailureClass.RATE_LIMIT
            limiter_sleep.assert_not_called()
            assert provider.cooldown_remaining() == pytest.approx(THREE_HOURS, abs=5)
            assert provider.is_available() is False

            # Next call fails fast without another HTTP request.
            with pytest.raises(ProviderError, match="usage limit"):
                provider.complete(_msgs())
            assert mock_cls.return_value.post.call_count == 1

    def test_cooldown_expires(self):
        provider = _single()
        with (
            patch("missy.providers.codex_provider.PolicyHTTPClient") as mock_cls,
            patch("missy.providers.codex_provider._load_oauth_token", return_value=None),
        ):
            _run_complete(provider, _usage_limit_error(resets_in=60), mock_cls)
        real = time.monotonic
        with patch(
            "missy.providers.codex_provider.time.monotonic", side_effect=lambda: real() + 61
        ):
            assert provider.cooldown_remaining() == 0.0
            assert provider.is_available() is True

    def test_long_plain_retry_after_also_cools_down(self):
        provider = _single()
        with (
            patch("missy.providers.codex_provider.PolicyHTTPClient") as mock_cls,
            patch("missy.providers.codex_provider._load_oauth_token", return_value=None),
            patch.object(RateLimiter, "on_rate_limit_response") as limiter_sleep,
        ):
            _run_complete(provider, _plain_429("3600"), mock_cls)
        limiter_sleep.assert_not_called()
        assert provider.cooldown_remaining() > 3000

    def test_short_plain_retry_after_keeps_limiter_backoff(self):
        provider = _multi()
        with (
            patch("missy.providers.codex_provider.PolicyHTTPClient") as mock_cls,
            patch(
                "missy.providers.codex_provider._load_oauth_token",
                side_effect=lambda force_refresh=False, account=None: f"{account}-tok",
            ),
            patch.object(provider, "_acquire_rate_limit"),
            patch.object(RateLimiter, "on_rate_limit_response") as limiter_sleep,
        ):
            _run_complete(provider, _plain_429("5"), mock_cls)
        limiter_sleep.assert_called_once_with(5.0)
        assert provider.cooldown_remaining() == 0.0

    def test_multi_account_only_limited_account_is_benched(self):
        provider = _multi()
        accounts_used: list[str] = []

        def post(url, json=None, headers=None, stream=False):
            accounts_used.append(provider._current_account_name())
            raise _usage_limit_error()

        with (
            patch("missy.providers.codex_provider.PolicyHTTPClient") as mock_cls,
            patch(
                "missy.providers.codex_provider._load_oauth_token",
                side_effect=lambda force_refresh=False, account=None: f"{account}-tok",
            ),
            patch.object(provider, "_acquire_rate_limit"),
        ):
            mock_cls.return_value.post.side_effect = post
            with pytest.raises(ProviderError):
                provider.complete(_msgs())
            # One account limited: the provider is still available and the
            # next call goes to the other account.
            assert provider.cooldown_remaining() == 0.0
            assert provider.is_available() is True
            with pytest.raises(ProviderError):
                provider.complete(_msgs())
            assert len(set(accounts_used)) == 2
            # Both limited now: provider cools down, further calls fail fast.
            assert provider.cooldown_remaining() > 0
            assert provider.is_available() is False
            with pytest.raises(ProviderError, match="usage limit"):
                provider.complete(_msgs())
            assert len(accounts_used) == 2


def _registry_with_cooling_codex() -> tuple[ProviderRegistry, CodexProvider]:
    codex = _single()
    codex._cooldown_until = time.monotonic() + THREE_HOURS
    registry = ProviderRegistry()
    registry.register("openai-codex", codex)
    registry.register("fallback", _Fallback())
    registry._default_name = "openai-codex"
    # Simulate a fresh-but-stale-in-meaning positive probe cached just
    # before the limit was hit.
    registry._availability_cache["openai-codex"] = (True, time.monotonic())
    registry._availability_cache["fallback"] = (True, time.monotonic())
    return registry, codex


class TestSelectionSkipsCoolingProvider:
    def test_select_weighted_skips_cooling_codex(self):
        registry, _ = _registry_with_cooling_codex()
        for _ in range(4):
            picked = registry.select_weighted()
            assert picked is not None and picked.name == "fallback"

    def test_get_available_excludes_cooling_codex(self):
        registry, _ = _registry_with_cooling_codex()
        assert [p.name for p in registry.get_available()] == ["fallback"]

    def test_vision_path_skips_cooling_codex(self):
        from missy.vision import provider_call

        registry, codex = _registry_with_cooling_codex()
        with (
            patch("missy.providers.registry.get_registry", return_value=registry),
            patch.object(provider_call, "_ollama_fallback_candidate", return_value=None),
            patch.object(codex, "complete", side_effect=AssertionError("codex called")),
        ):
            text, used = provider_call.analyze_image_with_provider_fallback("describe", "aGk=")
        assert used == "fallback"
        assert text == "fallback answer"

    def test_runtime_get_provider_goes_to_fallback(self):
        from missy.agent.runtime import AgentRuntime

        registry, _ = _registry_with_cooling_codex()
        runtime = MagicMock()
        runtime.config.provider = "openai-codex"
        with patch("missy.agent.runtime.get_registry", return_value=registry):
            provider = AgentRuntime._get_provider(runtime, "openai-codex")
        assert provider.name == "fallback"
