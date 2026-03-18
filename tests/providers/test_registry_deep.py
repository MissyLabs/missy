"""Deep tests for missy.providers.registry.

Covers ProviderRegistry, ModelRouter, and the module-level singleton with
45+ tests targeting all documented behaviours including edge cases.
"""

from __future__ import annotations

import threading
from unittest.mock import MagicMock, patch

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
    ModelRouter,
    ProviderRegistry,
    get_registry,
    init_registry,
)

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _make_provider(name: str = "fake", available: bool = True) -> MagicMock:
    """Return a mock BaseProvider with the given name and availability."""
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
    """Return a minimal MissyConfig suitable for unit tests."""
    return MissyConfig(
        network=NetworkPolicy(),
        filesystem=FilesystemPolicy(),
        shell=ShellPolicy(),
        plugins=PluginPolicy(),
        providers=providers or {},
        workspace_path="/tmp",
        audit_log_path="/tmp/audit.log",
    )


def _make_provider_config(
    name: str = "anthropic",
    model: str = "claude-sonnet-4-6",
    *,
    api_key: str | None = None,
    api_keys: list | None = None,
    enabled: bool = True,
    base_url: str | None = None,
    fast_model: str = "",
    premium_model: str = "",
) -> ProviderConfig:
    return ProviderConfig(
        name=name,
        model=model,
        api_key=api_key,
        api_keys=api_keys or [],
        enabled=enabled,
        base_url=base_url,
        fast_model=fast_model,
        premium_model=premium_model,
    )


# ---------------------------------------------------------------------------
# Autouse fixture: always reset the module-level singleton
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def reset_registry_singleton():
    """Restore the module-level _registry after every test."""
    original = registry_module._registry
    yield
    registry_module._registry = original


# ===========================================================================
# ProviderRegistry – basic register / get
# ===========================================================================


class TestEmptyRegistry:
    def test_get_on_empty_returns_none(self):
        registry = ProviderRegistry()
        assert registry.get("any") is None

    def test_list_providers_on_empty_is_empty(self):
        registry = ProviderRegistry()
        assert registry.list_providers() == []

    def test_get_available_on_empty_is_empty(self):
        registry = ProviderRegistry()
        assert registry.get_available() == []

    def test_get_default_name_initially_none(self):
        registry = ProviderRegistry()
        assert registry.get_default_name() is None


class TestRegisterAndGet:
    def test_register_then_get_returns_same_instance(self):
        registry = ProviderRegistry()
        provider = _make_provider("openai")
        registry.register("openai", provider)
        assert registry.get("openai") is provider

    def test_register_stores_with_config(self):
        registry = ProviderRegistry()
        provider = _make_provider("anthropic")
        cfg = _make_provider_config("anthropic")
        registry.register("anthropic", provider, config=cfg)
        assert registry.get("anthropic") is provider

    def test_register_replaces_existing(self):
        registry = ProviderRegistry()
        first = _make_provider("alpha")
        second = _make_provider("alpha")
        registry.register("alpha", first)
        registry.register("alpha", second)
        assert registry.get("alpha") is second

    def test_get_unknown_name_returns_none(self):
        registry = ProviderRegistry()
        registry.register("real", _make_provider())
        assert registry.get("ghost") is None


# ===========================================================================
# ProviderRegistry – list_providers
# ===========================================================================


class TestListProviders:
    def test_single_provider(self):
        registry = ProviderRegistry()
        registry.register("anthropic", _make_provider("anthropic"))
        assert registry.list_providers() == ["anthropic"]

    def test_multiple_providers_sorted_alphabetically(self):
        registry = ProviderRegistry()
        for name in ["openai", "anthropic", "ollama"]:
            registry.register(name, _make_provider(name))
        assert registry.list_providers() == ["anthropic", "ollama", "openai"]

    def test_list_providers_returns_new_list(self):
        registry = ProviderRegistry()
        registry.register("a", _make_provider())
        result = registry.list_providers()
        result.append("injected")
        assert "injected" not in registry.list_providers()


# ===========================================================================
# ProviderRegistry – get_available
# ===========================================================================


class TestGetAvailable:
    def test_all_available(self):
        registry = ProviderRegistry()
        for name in ["p1", "p2"]:
            registry.register(name, _make_provider(name, available=True))
        assert len(registry.get_available()) == 2

    def test_none_available(self):
        registry = ProviderRegistry()
        for name in ["p1", "p2"]:
            registry.register(name, _make_provider(name, available=False))
        assert registry.get_available() == []

    def test_mixed_availability_returns_only_available(self):
        registry = ProviderRegistry()
        registry.register("good", _make_provider("good", available=True))
        registry.register("bad", _make_provider("bad", available=False))
        available = registry.get_available()
        assert len(available) == 1
        assert available[0].name == "good"

    def test_is_available_exception_treated_as_unavailable(self):
        registry = ProviderRegistry()
        flaky = _make_provider("flaky")
        flaky.is_available.side_effect = RuntimeError("network error")
        registry.register("flaky", flaky)
        assert registry.get_available() == []

    def test_is_available_exception_does_not_affect_other_providers(self):
        registry = ProviderRegistry()
        flaky = _make_provider("flaky")
        flaky.is_available.side_effect = RuntimeError("boom")
        registry.register("flaky", flaky)
        registry.register("stable", _make_provider("stable", available=True))
        available = registry.get_available()
        assert len(available) == 1
        assert available[0].name == "stable"


# ===========================================================================
# ProviderRegistry – set_default / get_default_name
# ===========================================================================


class TestSetDefault:
    def test_set_default_valid_available_provider(self):
        registry = ProviderRegistry()
        provider = _make_provider("anthropic", available=True)
        registry.register("anthropic", provider)
        registry.set_default("anthropic")
        assert registry.get_default_name() == "anthropic"

    def test_set_default_unregistered_raises_value_error(self):
        registry = ProviderRegistry()
        with pytest.raises(ValueError, match="not registered"):
            registry.set_default("nonexistent")

    def test_set_default_unavailable_raises_value_error(self):
        registry = ProviderRegistry()
        provider = _make_provider("slow", available=False)
        registry.register("slow", provider)
        with pytest.raises(ValueError, match="not available"):
            registry.set_default("slow")

    def test_set_default_is_available_raises_non_value_error_wrapped(self):
        registry = ProviderRegistry()
        provider = _make_provider("boom")
        provider.is_available.side_effect = ConnectionError("timeout")
        registry.register("boom", provider)
        with pytest.raises(ValueError, match="availability check failed"):
            registry.set_default("boom")

    def test_set_default_is_available_raises_value_error_propagates_unchanged(self):
        """A ValueError from is_available propagates directly without wrapping."""
        registry = ProviderRegistry()
        provider = _make_provider("tricky")
        provider.is_available.side_effect = ValueError("original message")
        registry.register("tricky", provider)
        with pytest.raises(ValueError, match="original message"):
            registry.set_default("tricky")

    def test_get_default_name_returns_none_initially(self):
        registry = ProviderRegistry()
        assert registry.get_default_name() is None

    def test_get_default_name_after_set(self):
        registry = ProviderRegistry()
        registry.register("a", _make_provider("a", available=True))
        registry.set_default("a")
        assert registry.get_default_name() == "a"

    def test_set_default_can_be_changed(self):
        registry = ProviderRegistry()
        for name in ("a", "b"):
            registry.register(name, _make_provider(name, available=True))
        registry.set_default("a")
        registry.set_default("b")
        assert registry.get_default_name() == "b"


# ===========================================================================
# ProviderRegistry – rotate_key
# ===========================================================================


class TestRotateKey:
    def _registry_with_keys(self, keys: list[str]) -> tuple[ProviderRegistry, MagicMock]:
        registry = ProviderRegistry()
        provider = _make_provider("myp")
        # Give the provider an api_key attribute (not on spec by default)
        provider.api_key = keys[0] if keys else None
        cfg = _make_provider_config("myp", api_key=keys[0] if keys else None, api_keys=keys)
        registry.register("myp", provider, config=cfg)
        return registry, provider

    def test_rotate_key_round_robin_with_three_keys(self):
        keys = ["key-a", "key-b", "key-c"]
        registry, provider = self._registry_with_keys(keys)
        # Start at index 0 (key-a); first rotation → key-b
        registry.rotate_key("myp")
        assert provider.api_key == "key-b"
        registry.rotate_key("myp")
        assert provider.api_key == "key-c"
        # Wraps around
        registry.rotate_key("myp")
        assert provider.api_key == "key-a"

    def test_rotate_key_fewer_than_two_keys_skips(self):
        keys = ["only-key"]
        registry, provider = self._registry_with_keys(keys)
        provider.api_key = "only-key"
        registry.rotate_key("myp")
        # api_key must remain unchanged
        assert provider.api_key == "only-key"

    def test_rotate_key_zero_keys_skips(self):
        registry, provider = self._registry_with_keys([])
        original = provider.api_key
        registry.rotate_key("myp")
        assert provider.api_key == original

    def test_rotate_key_unknown_provider_warns(self, caplog):
        registry = ProviderRegistry()
        import logging

        with caplog.at_level(logging.WARNING, logger="missy.providers.registry"):
            registry.rotate_key("does-not-exist")
        assert any("not found" in r.message for r in caplog.records)

    def test_rotate_key_updates_api_key_attribute(self):
        """rotate_key sets provider.api_key when the attribute exists."""
        registry = ProviderRegistry()
        provider = _make_provider("r")
        provider.api_key = "first"
        cfg = _make_provider_config("r", api_keys=["first", "second"])
        registry.register("r", provider, config=cfg)
        registry.rotate_key("r")
        assert provider.api_key == "second"

    def test_rotate_key_updates_private_api_key_attribute_when_no_public(self):
        """rotate_key falls back to _api_key if api_key doesn't exist."""
        registry = ProviderRegistry()
        # Use a plain object that has _api_key but not api_key
        provider = MagicMock(spec=BaseProvider)
        provider.name = "priv"
        provider.is_available.return_value = True
        # Remove api_key from spec mock so hasattr returns False
        del provider.api_key
        provider._api_key = "first"
        cfg = _make_provider_config("priv", api_keys=["first", "second"])
        registry.register("priv", provider, config=cfg)
        registry.rotate_key("priv")
        assert provider._api_key == "second"

    def test_rotate_key_no_config_warns(self, caplog):
        """Provider registered without config treats rotate as unknown."""
        registry = ProviderRegistry()
        provider = _make_provider("no-cfg")
        # Register without config so _provider_configs has no entry
        registry._providers["no-cfg"] = provider
        import logging

        with caplog.at_level(logging.WARNING, logger="missy.providers.registry"):
            registry.rotate_key("no-cfg")
        assert any("not found" in r.message for r in caplog.records)


# ===========================================================================
# ProviderRegistry – from_config
# ===========================================================================


class TestFromConfig:
    def test_from_config_empty_providers(self):
        config = _make_config(providers={})
        registry = ProviderRegistry.from_config(config)
        assert registry.list_providers() == []

    def test_from_config_skips_disabled_providers(self):
        config = _make_config(
            providers={
                "disabled_one": _make_provider_config("anthropic", enabled=False),
            }
        )
        registry = ProviderRegistry.from_config(config)
        assert registry.get("disabled_one") is None

    def test_from_config_skips_unknown_provider_name(self):
        config = _make_config(
            providers={
                "mystery": _make_provider_config("totally_unknown_backend"),
            }
        )
        registry = ProviderRegistry.from_config(config)
        assert registry.get("mystery") is None

    def test_from_config_handles_provider_construction_failure(self):
        """When the provider class constructor raises, the key is skipped."""
        config = _make_config(
            providers={
                "bad": _make_provider_config("anthropic"),
            }
        )
        with patch(
            "missy.providers.registry._PROVIDER_CLASSES",
            {"anthropic": MagicMock(side_effect=RuntimeError("boom"))},
        ):
            registry = ProviderRegistry.from_config(config)
        assert registry.get("bad") is None

    def test_from_config_registers_known_ollama_provider(self):
        config = _make_config(
            providers={
                "local": _make_provider_config("ollama", model="llama3.2"),
            }
        )
        registry = ProviderRegistry.from_config(config)
        # OllamaProvider should be instantiated without network calls
        assert registry.get("local") is not None

    def test_from_config_auto_populates_provider_allowed_hosts_from_base_url(self):
        config = _make_config(
            providers={
                "custom": _make_provider_config(
                    "ollama",
                    model="llama3.2",
                    base_url="http://myhost.internal:11434",
                ),
            }
        )
        ProviderRegistry.from_config(config)
        assert "myhost.internal" in config.network.provider_allowed_hosts

    def test_from_config_does_not_duplicate_already_allowed_hosts(self):
        config = _make_config(
            providers={
                "local": _make_provider_config(
                    "ollama",
                    model="llama3.2",
                    base_url="http://myhost.internal:11434",
                ),
            }
        )
        config.network.provider_allowed_hosts.append("myhost.internal")
        ProviderRegistry.from_config(config)
        count = config.network.provider_allowed_hosts.count("myhost.internal")
        assert count == 1

    def test_from_config_skips_base_url_for_disabled_providers(self):
        config = _make_config(
            providers={
                "off": _make_provider_config(
                    "ollama",
                    model="llama3.2",
                    base_url="http://should-not-appear.test",
                    enabled=False,
                ),
            }
        )
        ProviderRegistry.from_config(config)
        assert "should-not-appear.test" not in config.network.provider_allowed_hosts

    def test_from_config_multiple_providers_registers_all_enabled(self):
        config = _make_config(
            providers={
                "local_a": _make_provider_config("ollama", model="llama3.2"),
                "local_b": _make_provider_config("ollama", model="mistral"),
                "disabled": _make_provider_config("ollama", model="phi3", enabled=False),
            }
        )
        registry = ProviderRegistry.from_config(config)
        names = registry.list_providers()
        assert "local_a" in names
        assert "local_b" in names
        assert "disabled" not in names


# ===========================================================================
# ModelRouter – score_complexity
# ===========================================================================


class TestModelRouterScoreComplexity:
    @pytest.fixture
    def router(self) -> ModelRouter:
        return ModelRouter()

    def test_short_prompt_with_fast_indicator_returns_fast(self, router):
        assert router.score_complexity("what is the time", tool_count=0) == "fast"

    def test_short_prompt_how_returns_fast(self, router):
        assert router.score_complexity("how do I do this", tool_count=0) == "fast"

    def test_fast_indicator_with_tools_does_not_force_fast(self, router):
        # tool_count > 0 breaks the fast-path condition
        result = router.score_complexity("what is this", tool_count=1)
        assert result != "fast"

    def test_premium_keyword_debug_returns_premium(self, router):
        assert router.score_complexity("debug this code for me") == "premium"

    def test_premium_keyword_architect_returns_premium(self, router):
        assert router.score_complexity("architect a new microservice") == "premium"

    def test_premium_keyword_refactor_returns_premium(self, router):
        assert router.score_complexity("refactor the entire module") == "premium"

    def test_premium_keyword_analyze_returns_premium(self, router):
        assert router.score_complexity("analyze the performance") == "premium"

    def test_premium_keyword_optimize_returns_premium(self, router):
        assert router.score_complexity("optimize the database queries") == "premium"

    def test_premium_keyword_complex_returns_premium(self, router):
        assert router.score_complexity("this is a complex task") == "premium"

    def test_long_prompt_above_500_chars_returns_premium(self, router):
        long_prompt = "a" * 501
        assert router.score_complexity(long_prompt) == "premium"

    def test_prompt_exactly_500_chars_is_primary(self, router):
        prompt = "b" * 500
        result = router.score_complexity(prompt)
        assert result == "primary"

    def test_many_tools_above_3_returns_premium(self, router):
        assert router.score_complexity("short prompt", tool_count=4) == "premium"

    def test_exactly_3_tools_is_primary(self, router):
        assert router.score_complexity("short neutral prompt", tool_count=3) == "primary"

    def test_long_history_above_10_returns_premium(self, router):
        assert router.score_complexity("short prompt", history_length=11) == "premium"

    def test_exactly_10_history_is_primary(self, router):
        assert router.score_complexity("short neutral prompt", history_length=10) == "primary"

    def test_neutral_prompt_returns_primary(self, router):
        assert router.score_complexity("please help me write a letter") == "primary"

    def test_premium_keyword_case_insensitive(self, router):
        # Keywords are matched after lowercasing
        assert router.score_complexity("DEBUG this issue") == "premium"

    def test_empty_prompt_returns_primary(self, router):
        result = router.score_complexity("")
        # No fast indicators, no premium signals → primary
        assert result == "primary"


# ===========================================================================
# ModelRouter – select_model
# ===========================================================================


class TestModelRouterSelectModel:
    @pytest.fixture
    def router(self) -> ModelRouter:
        return ModelRouter()

    def _cfg(
        self,
        model: str = "primary-model",
        fast_model: str = "",
        premium_model: str = "",
    ) -> ProviderConfig:
        return _make_provider_config(
            model=model,
            fast_model=fast_model,
            premium_model=premium_model,
        )

    def test_select_model_primary_tier_returns_model(self, router):
        cfg = self._cfg(model="gpt-4o")
        assert router.select_model(cfg, "primary") == "gpt-4o"

    def test_select_model_fast_tier_with_fast_model(self, router):
        cfg = self._cfg(model="big-model", fast_model="haiku")
        assert router.select_model(cfg, "fast") == "haiku"

    def test_select_model_fast_tier_without_fast_model_falls_back(self, router):
        cfg = self._cfg(model="big-model", fast_model="")
        assert router.select_model(cfg, "fast") == "big-model"

    def test_select_model_premium_tier_with_premium_model(self, router):
        cfg = self._cfg(model="sonnet", premium_model="opus")
        assert router.select_model(cfg, "premium") == "opus"

    def test_select_model_premium_tier_without_premium_model_falls_back(self, router):
        cfg = self._cfg(model="sonnet", premium_model="")
        assert router.select_model(cfg, "premium") == "sonnet"

    def test_select_model_unknown_tier_falls_back_to_model(self, router):
        cfg = self._cfg(model="fallback-model")
        assert router.select_model(cfg, "nonexistent_tier") == "fallback-model"


# ===========================================================================
# Module-level singleton: init_registry / get_registry
# ===========================================================================


class TestSingleton:
    def test_get_registry_before_init_raises_runtime_error(self):
        registry_module._registry = None
        with pytest.raises(RuntimeError, match="not been initialised"):
            get_registry()

    def test_init_registry_returns_provider_registry(self):
        config = _make_config()
        result = init_registry(config)
        assert isinstance(result, ProviderRegistry)

    def test_get_registry_after_init_returns_same_instance(self):
        config = _make_config()
        registry = init_registry(config)
        assert get_registry() is registry

    def test_init_registry_replaces_existing_singleton(self):
        config = _make_config()
        first = init_registry(config)
        second = init_registry(config)
        assert get_registry() is second
        assert first is not second

    def test_thread_safety_concurrent_init_and_get(self):
        """Concurrent init_registry and get_registry calls must not corrupt state."""
        config = _make_config()
        errors: list[Exception] = []
        results: list[ProviderRegistry] = []
        lock = threading.Lock()

        def worker_init():
            try:
                r = init_registry(config)
                with lock:
                    results.append(r)
            except Exception as exc:  # noqa: BLE001
                with lock:
                    errors.append(exc)

        def worker_get():
            try:
                # init may not have run yet; tolerate RuntimeError
                r = get_registry()
                with lock:
                    results.append(r)
            except RuntimeError:
                pass
            except Exception as exc:  # noqa: BLE001
                with lock:
                    errors.append(exc)

        threads = [threading.Thread(target=worker_init) for _ in range(4)]
        threads += [threading.Thread(target=worker_get) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=5)

        assert errors == [], f"Thread errors: {errors}"
        # After all workers complete, get_registry must return a valid instance
        assert isinstance(get_registry(), ProviderRegistry)

    def test_singleton_reset_via_monkeypatch(self, monkeypatch):
        """Verify monkeypatch can reset the singleton (fixture compatibility check)."""
        monkeypatch.setattr(registry_module, "_registry", None)
        with pytest.raises(RuntimeError):
            get_registry()
        config = _make_config()
        init_registry(config)
        assert isinstance(get_registry(), ProviderRegistry)
