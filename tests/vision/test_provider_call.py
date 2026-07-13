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


# Every test in TestAnalyzeImageWithProviderFallback explicitly disables the
# Ollama last-resort fallback (tested separately in TestOllamaFallback) so
# these don't implicitly depend on whether a real Ollama instance happens to
# be reachable in the environment running the suite.
@patch("missy.vision.provider_call._ollama_fallback_candidate", return_value=None)
class TestAnalyzeImageWithProviderFallback:
    @patch("missy.providers.registry.get_registry")
    def test_uses_default_provider_when_available(self, mock_get_registry, _mock_ollama):
        anthropic = _make_provider("anthropic", response_text="A red mug on a desk.")
        registry = MagicMock()
        registry.get_available.return_value = [anthropic]
        registry.get_default_name.return_value = "anthropic"
        mock_get_registry.return_value = registry

        text, provider_name = analyze_image_with_provider_fallback("Describe this", "ZmFrZQ==")

        assert text == "A red mug on a desk."
        assert provider_name == "anthropic"

    @patch("missy.providers.registry.get_registry")
    def test_falls_back_to_next_candidate_on_failure(self, mock_get_registry, _mock_ollama):
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
    def test_default_provider_tried_first_even_if_registered_later(
        self, mock_get_registry, _mock_ollama
    ):
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
    def test_acpx_excluded_from_candidates(self, mock_get_registry, _mock_ollama):
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
    def test_no_available_providers_raises_clear_error(self, mock_get_registry, _mock_ollama):
        registry = MagicMock()
        registry.get_available.return_value = []
        registry.get_default_name.return_value = None
        mock_get_registry.return_value = registry

        with pytest.raises(ProviderError, match="No available provider"):
            analyze_image_with_provider_fallback("Describe this", "ZmFrZQ==")

    @patch("missy.providers.registry.get_registry")
    def test_only_acpx_available_raises_clear_error(self, mock_get_registry, _mock_ollama):
        acpx = _make_provider("acpx")
        registry = MagicMock()
        registry.get_available.return_value = [acpx]
        registry.get_default_name.return_value = "acpx"
        mock_get_registry.return_value = registry

        with pytest.raises(ProviderError, match="No available provider"):
            analyze_image_with_provider_fallback("Describe this", "ZmFrZQ==")
        acpx.complete.assert_not_called()

    @patch("missy.providers.registry.get_registry")
    def test_all_candidates_fail_raises_error_with_all_details(
        self, mock_get_registry, _mock_ollama
    ):
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
    def test_unexpected_exception_does_not_abort_fallback(self, mock_get_registry, _mock_ollama):
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


# ---------------------------------------------------------------------------
# Ollama last-resort vision fallback
#
# An operator may deliberately keep Ollama disabled for general chat
# (enabled: false) while still wanting a local vision-capable model as a
# safety net for when the actually-enabled provider(s) can't -- or
# currently don't -- handle images. _ollama_fallback_candidate() adds it
# back in specifically for vision, bypassing the "enabled" gate that
# normally keeps it out of ProviderRegistry entirely.
# ---------------------------------------------------------------------------


class TestOllamaFallbackCandidate:
    @patch("missy.providers.ollama_provider.OllamaProvider")
    @patch("missy.config.settings.load_config")
    def test_returns_provider_when_configured_and_reachable(
        self, mock_load_config, mock_ollama_cls
    ):
        from missy.vision.provider_call import _ollama_fallback_candidate

        ollama_cfg = MagicMock()
        mock_load_config.return_value.providers = {"ollama": ollama_cfg}
        instance = mock_ollama_cls.return_value
        instance.is_available.return_value = True

        result = _ollama_fallback_candidate(already_present_names=set())

        assert result is instance
        mock_ollama_cls.assert_called_once_with(ollama_cfg)

    def test_none_when_already_an_enabled_candidate(self):
        from missy.vision.provider_call import _ollama_fallback_candidate

        # No mocking needed -- "ollama" already present short-circuits
        # before load_config()/OllamaProvider are ever touched.
        assert _ollama_fallback_candidate(already_present_names={"ollama"}) is None

    @patch("missy.config.settings.load_config")
    def test_none_when_ollama_not_configured_at_all(self, mock_load_config):
        from missy.vision.provider_call import _ollama_fallback_candidate

        mock_load_config.return_value.providers = {}
        assert _ollama_fallback_candidate(already_present_names=set()) is None

    @patch("missy.providers.ollama_provider.OllamaProvider")
    @patch("missy.config.settings.load_config")
    def test_none_when_ollama_not_reachable(self, mock_load_config, mock_ollama_cls):
        from missy.vision.provider_call import _ollama_fallback_candidate

        mock_load_config.return_value.providers = {"ollama": MagicMock()}
        mock_ollama_cls.return_value.is_available.return_value = False

        assert _ollama_fallback_candidate(already_present_names=set()) is None

    @patch("missy.config.settings.load_config")
    def test_none_when_load_config_raises(self, mock_load_config):
        from missy.vision.provider_call import _ollama_fallback_candidate

        mock_load_config.side_effect = RuntimeError("no config file")
        assert _ollama_fallback_candidate(already_present_names=set()) is None


class TestOllamaFallbackEndToEnd:
    @patch("missy.vision.provider_call._ollama_fallback_candidate")
    @patch("missy.providers.registry.get_registry")
    def test_used_as_last_resort_when_only_enabled_provider_cannot_see_images(
        self, mock_get_registry, mock_ollama_fallback
    ):
        """The exact scenario the fallback exists for: openai-codex (or
        any single enabled provider) fails on this call, and a
        configured-but-disabled Ollama actually answers it."""
        broken = _make_provider("openai-codex", error="does not support image input")
        ollama = _make_provider("ollama", response_text="A handwritten note reading 'hello'.")
        mock_ollama_fallback.return_value = ollama

        registry = MagicMock()
        registry.get_available.return_value = [broken]
        registry.get_default_name.return_value = "openai-codex"
        mock_get_registry.return_value = registry

        text, provider_name = analyze_image_with_provider_fallback("Read the text", "ZmFrZQ==")

        assert provider_name == "ollama"
        assert "hello" in text
        broken.complete.assert_called_once()
        ollama.complete.assert_called_once()

    @patch("missy.vision.provider_call._ollama_fallback_candidate", return_value=None)
    @patch("missy.providers.registry.get_registry")
    def test_not_used_when_primary_provider_already_succeeds(
        self, mock_get_registry, mock_ollama_fallback
    ):
        """The common case: primary succeeds, fallback lookup is never
        even attempted (short-circuits before Ollama is touched)."""
        working = _make_provider("anthropic", response_text="Real analysis.")
        registry = MagicMock()
        registry.get_available.return_value = [working]
        registry.get_default_name.return_value = "anthropic"
        mock_get_registry.return_value = registry

        text, provider_name = analyze_image_with_provider_fallback("Describe this", "ZmFrZQ==")

        assert provider_name == "anthropic"
        # _ollama_fallback_candidate is still called once (to build the
        # candidate list up front) but Ollama's own complete() must never
        # be reached since the primary already succeeded.
        working.complete.assert_called_once()
