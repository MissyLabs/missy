"""Run-18 provider fail-closed and credential-isolation regressions."""

from __future__ import annotations

import threading
import time
from dataclasses import FrozenInstanceError, asdict
from unittest.mock import MagicMock, patch

import pytest

from missy.config.settings import ProviderConfig, get_default_config
from missy.core.exceptions import ProviderError
from missy.providers.anthropic_provider import AnthropicProvider
from missy.providers.openai_provider import OpenAIProvider
from missy.providers.registry import ProviderRegistry
from missy.providers.round_robin import RoundRobinAccounts


def test_prov_037_anthropic_transport_construction_fails_closed() -> None:
    provider = AnthropicProvider(ProviderConfig(name="anthropic", model="test", api_key="fake-key"))
    with (
        patch(
            "missy.providers.policy_http.build_policy_http_client",
            side_effect=RuntimeError("policy unavailable"),
        ),
        patch("missy.providers.anthropic_provider._anthropic_sdk.Anthropic") as sdk,
        pytest.raises(ProviderError, match="policy-aware transport"),
    ):
        provider._make_client()
    sdk.assert_not_called()
    assert not provider.is_available()


def test_prov_037_openai_transport_construction_fails_closed() -> None:
    provider = OpenAIProvider(ProviderConfig(name="openai", model="test", api_key="fake-key"))
    with (
        patch(
            "missy.providers.policy_http.build_policy_http_client",
            side_effect=RuntimeError("policy unavailable"),
        ),
        patch("missy.providers.openai_provider._openai_sdk.OpenAI") as sdk,
        pytest.raises(ProviderError, match="policy-aware transport"),
    ):
        provider._build_client("fake-key")
    sdk.assert_not_called()
    assert not provider.is_available()


def test_prov_042_public_account_state_cannot_mutate_or_reveal_credentials() -> None:
    rr = RoundRobinAccounts(
        ["secret-alpha", "secret-beta"],
        make_rate_limiter=object,
    )
    before = rr.accounts
    assert isinstance(before, tuple)
    assert "secret-" not in repr(before)
    assert not hasattr(before[0], "api_key")
    with pytest.raises(FrozenInstanceError):
        before[0].index = 99  # type: ignore[misc]
    assert [rr.select().index for _ in range(4)] == [0, 1, 0, 1]


def test_prov_041_registration_rejects_shadowing_without_stale_state() -> None:
    registry = ProviderRegistry()
    first = AnthropicProvider(ProviderConfig(name="anthropic", model="first", api_key="secret-a"))
    second = AnthropicProvider(ProviderConfig(name="anthropic", model="second", api_key="secret-b"))
    registry.register("primary", first)
    registry.set_default("primary")

    with pytest.raises(ValueError, match="already registered") as caught:
        registry.register("primary", second)

    assert registry.get("primary") is first
    assert registry.get_default_name() == "primary"
    assert "secret-" not in str(caught.value)
    registry.register("primary", first)


def test_prov_045_registry_construction_preserves_caller_config() -> None:
    config = get_default_config()
    config.providers = {
        "local": ProviderConfig(
            name="ollama",
            model="test",
            base_url="http://provider.example.test:11434",
        )
    }
    before = asdict(config)

    registry = ProviderRegistry.from_config(config)

    assert asdict(config) == before
    assert "provider.example.test" in registry.effective_provider_hosts


def test_prov_040_bulk_availability_has_deadline_and_skips_disabled() -> None:
    registry = ProviderRegistry()
    registry.AVAILABILITY_BULK_DEADLINE_SECONDS = 0.03
    fast = AnthropicProvider(ProviderConfig(name="anthropic", model="fast", api_key="a"))
    slow = AnthropicProvider(ProviderConfig(name="anthropic", model="slow", api_key="b"))
    disabled = AnthropicProvider(ProviderConfig(name="anthropic", model="disabled", api_key="c"))
    fast.is_available = MagicMock(return_value=True)
    slow.is_available = MagicMock(side_effect=lambda: (time.sleep(0.2), True)[1])
    disabled.is_available = MagicMock(return_value=True)
    registry.register("fast", fast)
    registry.register("slow", slow)
    registry.register("disabled", disabled)
    registry.set_enabled("disabled", False)

    started = time.monotonic()
    available = registry.get_available()

    assert time.monotonic() - started < 0.15
    assert available == [fast]
    disabled.is_available.assert_not_called()
    assert registry.availability_status()["slow"]["in_flight"]


def test_prov_040_concurrent_availability_probes_coalesce() -> None:
    registry = ProviderRegistry()
    entered = threading.Event()
    release = threading.Event()
    provider = AnthropicProvider(ProviderConfig(name="anthropic", model="coalesced", api_key="a"))

    def probe() -> bool:
        entered.set()
        release.wait(1)
        return True

    provider.is_available = MagicMock(side_effect=probe)
    registry.register("provider", provider)
    results = []
    threads = [
        threading.Thread(target=lambda: results.append(registry.get_available())) for _ in range(12)
    ]
    for thread in threads:
        thread.start()
    assert entered.wait(1)
    release.set()
    for thread in threads:
        thread.join(1)

    assert provider.is_available.call_count == 1
    assert results == [[provider]] * 12
