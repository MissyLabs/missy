"""Tests for runtime provider hot-swap (Feature 4)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from missy.providers.registry import ProviderRegistry


class TestSetDefault:
    """Tests for ProviderRegistry.set_default / get_default_name."""

    def _make_registry_with_providers(self, *names: str) -> ProviderRegistry:
        registry = ProviderRegistry()
        for name in names:
            mock_provider = MagicMock()
            mock_provider.name = name
            mock_provider.is_available.return_value = True
            registry.register(name, mock_provider)
        return registry

    def test_set_default_valid(self):
        registry = self._make_registry_with_providers("anthropic", "ollama")
        registry.set_default("ollama")
        assert registry.get_default_name() == "ollama"

    def test_set_default_unknown_raises(self):
        registry = self._make_registry_with_providers("anthropic")
        with pytest.raises(ValueError, match="not registered"):
            registry.set_default("nonexistent")

    def test_set_default_unavailable_raises(self):
        registry = ProviderRegistry()
        mock_provider = MagicMock()
        mock_provider.name = "broken"
        mock_provider.is_available.return_value = False
        registry.register("broken", mock_provider)

        with pytest.raises(ValueError, match="not available"):
            registry.set_default("broken")

    def test_get_default_name_initially_none(self):
        registry = ProviderRegistry()
        assert registry.get_default_name() is None


class TestRuntimeSwitchProvider:
    """Tests for AgentRuntime.switch_provider."""

    def test_runtime_switch_provider(self):
        from missy.agent.runtime import AgentConfig, AgentRuntime

        # Mock the registry
        mock_registry = MagicMock()
        mock_provider = MagicMock()
        mock_provider.name = "ollama"
        mock_provider.is_available.return_value = True
        mock_registry.get.return_value = mock_provider

        config = AgentConfig(provider="anthropic")

        with patch("missy.agent.runtime.get_registry", return_value=mock_registry):
            agent = AgentRuntime(config)
            agent.switch_provider("ollama")

        assert config.provider == "ollama"
        mock_registry.set_default.assert_called_once_with("ollama")
