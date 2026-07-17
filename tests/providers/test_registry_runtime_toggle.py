"""Tests for ProviderRegistry runtime enable/disable (Web TUI provider toggle)."""

from __future__ import annotations

import pytest

from missy.providers.registry import ProviderRegistry


class FakeProvider:
    name = "fake"

    def __init__(self, available: bool = True):
        self._available = available

    def is_available(self) -> bool:
        return self._available


@pytest.fixture
def registry() -> ProviderRegistry:
    reg = ProviderRegistry()
    reg.register("alpha", FakeProvider())
    reg.register("beta", FakeProvider())
    return reg


class TestSetEnabled:
    def test_providers_enabled_by_default(self, registry):
        assert registry.is_enabled("alpha") is True
        assert registry.is_enabled("beta") is True

    def test_disable_excludes_from_available(self, registry):
        registry.set_enabled("beta", False)
        assert len(registry.get_available()) == 1
        assert registry.is_enabled("beta") is False

    def test_disabled_provider_stays_registered(self, registry):
        registry.set_enabled("beta", False)
        assert "beta" in registry.list_providers()
        assert registry.get("beta") is not None

    def test_reenable_restores_availability(self, registry):
        registry.set_enabled("beta", False)
        registry.set_enabled("beta", True)
        assert registry.is_enabled("beta") is True
        assert len(registry.get_available()) == 2

    def test_unknown_provider_raises(self, registry):
        with pytest.raises(ValueError, match="not registered"):
            registry.set_enabled("ghost", False)

    def test_cannot_disable_current_default(self, registry):
        registry.set_default("alpha")
        with pytest.raises(ValueError, match="current default"):
            registry.set_enabled("alpha", False)
        assert registry.is_enabled("alpha") is True

    def test_set_default_refuses_disabled_provider(self, registry):
        registry.set_enabled("beta", False)
        with pytest.raises(ValueError, match="disabled"):
            registry.set_default("beta")

    def test_disable_non_default_while_default_set(self, registry):
        registry.set_default("alpha")
        registry.set_enabled("beta", False)
        assert registry.get_default_name() == "alpha"
        assert registry.is_enabled("beta") is False
