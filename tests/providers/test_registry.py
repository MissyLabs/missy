"""Tests for missy.providers.registry.ProviderRegistry."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from missy.config.settings import (
    FilesystemPolicy,
    MissyConfig,
    NetworkPolicy,
    PluginPolicy,
    ProviderConfig,
    ShellPolicy,
)
from missy.providers import registry as registry_module
from missy.providers.base import BaseProvider, CompletionResponse
from missy.providers.registry import (
    ProviderRegistry,
    get_registry,
    init_registry,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_provider(name: str = "fake", available: bool = True) -> BaseProvider:
    provider = MagicMock(spec=BaseProvider)
    provider.name = name
    provider.is_available.return_value = available
    provider.complete.return_value = CompletionResponse(
        content="reply",
        model="fake-model",
        provider=name,
        usage={"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        raw={},
    )
    return provider


def _make_config(providers: dict | None = None) -> MissyConfig:
    return MissyConfig(
        network=NetworkPolicy(),
        filesystem=FilesystemPolicy(),
        shell=ShellPolicy(),
        plugins=PluginPolicy(),
        providers=providers or {},
        workspace_path="/tmp",
        audit_log_path="/tmp/audit.log",
    )


@pytest.fixture(autouse=True)
def reset_singleton():
    original = registry_module._registry
    yield
    registry_module._registry = original


# ---------------------------------------------------------------------------
# ProviderRegistry.register and get
# ---------------------------------------------------------------------------


class TestRegisterAndGet:
    def test_register_then_get_returns_provider(self):
        registry = ProviderRegistry()
        provider = _make_provider("openai")
        registry.register("openai", provider)
        assert registry.get("openai") is provider

    def test_get_unknown_name_returns_none(self):
        registry = ProviderRegistry()
        assert registry.get("nonexistent") is None

    def test_register_replaces_existing(self):
        registry = ProviderRegistry()
        first = _make_provider("alpha")
        second = _make_provider("alpha")
        registry.register("alpha", first)
        registry.register("alpha", second)
        assert registry.get("alpha") is second


# ---------------------------------------------------------------------------
# ProviderRegistry.list_providers
# ---------------------------------------------------------------------------


class TestListProviders:
    def test_empty_registry(self):
        registry = ProviderRegistry()
        assert registry.list_providers() == []

    def test_single_provider(self):
        registry = ProviderRegistry()
        registry.register("anthropic", _make_provider("anthropic"))
        assert registry.list_providers() == ["anthropic"]

    def test_multiple_providers_sorted(self):
        registry = ProviderRegistry()
        for name in ["openai", "anthropic", "ollama"]:
            registry.register(name, _make_provider(name))
        assert registry.list_providers() == ["anthropic", "ollama", "openai"]


# ---------------------------------------------------------------------------
# ProviderRegistry.get_available
# ---------------------------------------------------------------------------


class TestGetAvailable:
    def test_all_available(self):
        registry = ProviderRegistry()
        for name in ["p1", "p2"]:
            registry.register(name, _make_provider(name, available=True))
        available = registry.get_available()
        assert len(available) == 2

    def test_none_available(self):
        registry = ProviderRegistry()
        for name in ["p1", "p2"]:
            registry.register(name, _make_provider(name, available=False))
        assert registry.get_available() == []

    def test_mixed_availability(self):
        registry = ProviderRegistry()
        registry.register("good", _make_provider("good", available=True))
        registry.register("bad", _make_provider("bad", available=False))
        available = registry.get_available()
        assert len(available) == 1
        assert available[0].name == "good"

    def test_exception_in_is_available_treated_as_unavailable(self):
        registry = ProviderRegistry()
        flaky = _make_provider("flaky")
        flaky.is_available.side_effect = RuntimeError("network error")
        registry.register("flaky", flaky)
        available = registry.get_available()
        assert available == []


# ---------------------------------------------------------------------------
# ProviderRegistry.from_config
# ---------------------------------------------------------------------------


class TestFromConfig:
    def test_from_config_skips_unknown_providers(self):
        config = _make_config(
            providers={
                "mystery": ProviderConfig(
                    name="unknown_provider_xyz",
                    model="some-model",
                    api_key="key",
                )
            }
        )
        registry = ProviderRegistry.from_config(config)
        assert registry.get("mystery") is None

    def test_from_config_registers_known_provider(self):
        config = _make_config(
            providers={
                "my_ollama": ProviderConfig(
                    name="ollama",
                    model="llama3.2",
                )
            }
        )
        registry = ProviderRegistry.from_config(config)
        provider = registry.get("my_ollama")
        assert provider is not None
        assert provider.name == "ollama"

    def test_from_config_empty_providers(self):
        config = _make_config(providers={})
        registry = ProviderRegistry.from_config(config)
        assert registry.list_providers() == []


# ---------------------------------------------------------------------------
# Singleton: init_registry / get_registry
# ---------------------------------------------------------------------------


class TestSingleton:
    def test_get_registry_before_init_raises(self):
        registry_module._registry = None
        with pytest.raises(RuntimeError, match="not been initialised"):
            get_registry()

    def test_init_registry_returns_provider_registry(self):
        config = _make_config()
        registry = init_registry(config)
        assert isinstance(registry, ProviderRegistry)

    def test_get_registry_after_init_returns_same_instance(self):
        config = _make_config()
        registry = init_registry(config)
        assert get_registry() is registry

    def test_second_init_replaces_registry(self):
        config = _make_config()
        first = init_registry(config)
        second = init_registry(config)
        assert get_registry() is second
        assert first is not second
