"""Tests for missy.vision.provider_call -- real multimodal image analysis
via whichever configured provider actually supports it.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from missy.core.exceptions import ProviderError
from missy.providers.base import CompletionResponse
from missy.vision.provider_call import analyze_image_with_provider_fallback


def _make_provider(name: str, *, response_text: str | None = None, error: str | None = None):
    provider = MagicMock()
    provider.name = name
    if error is not None:
        provider.complete.side_effect = ProviderError(error)
    else:
        provider.complete.return_value = CompletionResponse(
            content=response_text or f"{name} analysis",
            model="m",
            provider=name,
            usage={"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            raw={},
        )
    return provider


class TestAnalyzeImageWithProviderFallback:
    @patch("missy.providers.registry.get_registry")
    def test_uses_default_provider_when_available(self, mock_get_registry):
        anthropic = _make_provider("anthropic", response_text="A red mug on a desk.")
        registry = MagicMock()
        registry.get_available.return_value = [anthropic]
        registry.get_default_name.return_value = "anthropic"
        mock_get_registry.return_value = registry

        text, provider_name = analyze_image_with_provider_fallback("Describe this", "ZmFrZQ==")

        assert text == "A red mug on a desk."
        assert provider_name == "anthropic"

    @patch("missy.providers.registry.get_registry")
    def test_falls_back_to_next_candidate_on_failure(self, mock_get_registry):
        broken = _make_provider("openai-codex", error="rate limited")
        healthy = _make_provider("anthropic", response_text="Fallback analysis.")
        registry = MagicMock()
        registry.get_available.return_value = [broken, healthy]
        registry.get_default_name.return_value = "openai-codex"
        mock_get_registry.return_value = registry

        text, provider_name = analyze_image_with_provider_fallback("Describe this", "ZmFrZQ==")

        assert text == "Fallback analysis."
        assert provider_name == "anthropic"
        broken.complete.assert_called_once()
        healthy.complete.assert_called_once()

    @patch("missy.providers.registry.get_registry")
    def test_default_provider_tried_first_even_if_registered_later(self, mock_get_registry):
        first_registered = _make_provider("anthropic", response_text="from anthropic")
        default = _make_provider("openai", response_text="from openai")
        registry = MagicMock()
        # Registration order puts anthropic first, but openai is configured default.
        registry.get_available.return_value = [first_registered, default]
        registry.get_default_name.return_value = "openai"
        mock_get_registry.return_value = registry

        text, provider_name = analyze_image_with_provider_fallback("Describe this", "ZmFrZQ==")

        assert provider_name == "openai"
        default.complete.assert_called_once()
        first_registered.complete.assert_not_called()

    @patch("missy.providers.registry.get_registry")
    def test_acpx_excluded_from_candidates(self, mock_get_registry):
        acpx = _make_provider("acpx", response_text="should never be used")
        healthy = _make_provider("anthropic", response_text="real analysis")
        registry = MagicMock()
        registry.get_available.return_value = [acpx, healthy]
        registry.get_default_name.return_value = None
        mock_get_registry.return_value = registry

        text, provider_name = analyze_image_with_provider_fallback("Describe this", "ZmFrZQ==")

        assert provider_name == "anthropic"
        acpx.complete.assert_not_called()

    @patch("missy.providers.registry.get_registry")
    def test_no_available_providers_raises_clear_error(self, mock_get_registry):
        registry = MagicMock()
        registry.get_available.return_value = []
        registry.get_default_name.return_value = None
        mock_get_registry.return_value = registry

        with pytest.raises(ProviderError, match="No available provider"):
            analyze_image_with_provider_fallback("Describe this", "ZmFrZQ==")

    @patch("missy.providers.registry.get_registry")
    def test_only_acpx_available_raises_clear_error(self, mock_get_registry):
        acpx = _make_provider("acpx")
        registry = MagicMock()
        registry.get_available.return_value = [acpx]
        registry.get_default_name.return_value = "acpx"
        mock_get_registry.return_value = registry

        with pytest.raises(ProviderError, match="No available provider"):
            analyze_image_with_provider_fallback("Describe this", "ZmFrZQ==")
        acpx.complete.assert_not_called()

    @patch("missy.providers.registry.get_registry")
    def test_all_candidates_fail_raises_error_with_all_details(self, mock_get_registry):
        first = _make_provider("anthropic", error="auth failed")
        second = _make_provider("openai", error="timeout")
        registry = MagicMock()
        registry.get_available.return_value = [first, second]
        registry.get_default_name.return_value = None
        mock_get_registry.return_value = registry

        with pytest.raises(ProviderError) as exc_info:
            analyze_image_with_provider_fallback("Describe this", "ZmFrZQ==")
        message = str(exc_info.value)
        assert "anthropic" in message
        assert "auth failed" in message
        assert "openai" in message
        assert "timeout" in message

    @patch("missy.providers.registry.get_registry")
    def test_unexpected_exception_does_not_abort_fallback(self, mock_get_registry):
        """A non-ProviderError exception (e.g. a bug in one provider's
        own code) must not prevent trying the next candidate."""
        broken = MagicMock()
        broken.name = "anthropic"
        broken.complete.side_effect = RuntimeError("unexpected bug")
        healthy = _make_provider("openai", response_text="still works")
        registry = MagicMock()
        registry.get_available.return_value = [broken, healthy]
        registry.get_default_name.return_value = None
        mock_get_registry.return_value = registry

        text, provider_name = analyze_image_with_provider_fallback("Describe this", "ZmFrZQ==")
        assert provider_name == "openai"
