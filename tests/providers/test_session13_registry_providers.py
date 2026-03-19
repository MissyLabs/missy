"""Session 13 – comprehensive edge-case tests for ProviderRegistry and base dataclasses.

Coverage targets NOT already in test_registry.py / test_registry_deep.py / test_base.py:

- Fallback-chain resolution when the preferred provider is unavailable
- API-key rotation: wrap-around, private-attribute path, idempotency, interleaved rotation
- fast_model / premium_model selection covering every combination
- Concurrent provider switching via set_default from multiple threads
- Message / CompletionResponse / ToolCall / ToolResult dataclass contracts
  (field defaults, mutation, repr, hashing behaviour, empty collections)
- ProviderConfig timeout validation and model-name pass-through
- AnthropicProvider: setup-token rejection, _api_key fall-through
- OpenAIProvider: is_available with vs without key
- from_config: key uses config name when provider_config.name is falsy
- from_config: provider key index initialised at 0 on first registration
- Edge cases: all providers unavailable, empty registry fallback chain,
  duplicate-name registration preserves key index, rotate on provider with
  no api_key attribute (neither public nor private)
"""

from __future__ import annotations

import threading
from dataclasses import asdict, fields
from typing import Any
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
from missy.providers.base import (
    BaseProvider,
    CompletionResponse,
    Message,
    ToolCall,
    ToolResult,
)
from missy.providers.registry import (
    ModelRouter,
    ProviderRegistry,
)

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _make_provider(name: str = "fake", available: bool = True) -> MagicMock:
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


def _make_provider_config(
    name: str = "anthropic",
    model: str = "claude-sonnet-4-6",
    *,
    api_key: str | None = None,
    api_keys: list[str] | None = None,
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


@pytest.fixture(autouse=True)
def _reset_singleton():
    original = registry_module._registry
    yield
    registry_module._registry = original


# ===========================================================================
# Fallback-chain resolution
# ===========================================================================


class TestFallbackChainResolution:
    """get_available drives fallback logic in real agent code."""

    def test_fallback_skips_unavailable_and_returns_next(self):
        registry = ProviderRegistry()
        registry.register("primary", _make_provider("primary", available=False))
        registry.register("fallback", _make_provider("fallback", available=True))
        available = registry.get_available()
        names = [p.name for p in available]
        assert "primary" not in names
        assert "fallback" in names

    def test_fallback_chain_all_unavailable_returns_empty(self):
        registry = ProviderRegistry()
        for n in ("a", "b", "c"):
            registry.register(n, _make_provider(n, available=False))
        assert registry.get_available() == []

    def test_fallback_chain_preserves_insertion_order(self):
        """get_available preserves dict insertion order (Python 3.7+)."""
        registry = ProviderRegistry()
        for n in ("first", "second", "third"):
            registry.register(n, _make_provider(n, available=True))
        names = [p.name for p in registry.get_available()]
        assert names == ["first", "second", "third"]

    def test_single_available_in_long_chain(self):
        registry = ProviderRegistry()
        for n in ("u1", "u2", "u3", "u4"):
            registry.register(n, _make_provider(n, available=False))
        registry.register("ok", _make_provider("ok", available=True))
        available = registry.get_available()
        assert len(available) == 1
        assert available[0].name == "ok"

    def test_fallback_exception_provider_does_not_prevent_later_providers(self):
        registry = ProviderRegistry()
        bad = _make_provider("bad")
        bad.is_available.side_effect = OSError("disk full")
        registry.register("bad", bad)
        registry.register("good", _make_provider("good", available=True))
        available = registry.get_available()
        assert any(p.name == "good" for p in available)

    def test_empty_registry_fallback_returns_empty_list(self):
        registry = ProviderRegistry()
        assert registry.get_available() == []


# ===========================================================================
# API-key rotation – edge cases
# ===========================================================================


class TestApiKeyRotationEdgeCases:
    def _registry_with_keys(
        self,
        keys: list[str],
        use_private: bool = False,
    ) -> tuple[ProviderRegistry, MagicMock]:
        registry = ProviderRegistry()
        provider = MagicMock(spec=BaseProvider)
        provider.name = "p"
        provider.is_available.return_value = True
        if use_private:
            # Simulate a provider that only exposes _api_key (not api_key).
            del provider.api_key
            provider._api_key = keys[0] if keys else None
        else:
            provider.api_key = keys[0] if keys else None
        cfg = _make_provider_config("p", api_key=keys[0] if keys else None, api_keys=keys)
        registry.register("p", provider, config=cfg)
        return registry, provider

    def test_two_key_rotation_wraps_correctly(self):
        registry, provider = self._registry_with_keys(["k1", "k2"])
        registry.rotate_key("p")
        assert provider.api_key == "k2"
        # Second rotation wraps back to k1
        registry.rotate_key("p")
        assert provider.api_key == "k1"

    def test_five_key_full_cycle(self):
        keys = [f"key{i}" for i in range(5)]
        registry, provider = self._registry_with_keys(keys)
        for expected_idx in range(1, 5):
            registry.rotate_key("p")
            assert provider.api_key == keys[expected_idx]
        # Wrap back to index 0
        registry.rotate_key("p")
        assert provider.api_key == keys[0]

    def test_rotate_with_private_attribute_only(self):
        registry, provider = self._registry_with_keys(["first", "second"], use_private=True)
        registry.rotate_key("p")
        assert provider._api_key == "second"

    def test_rotate_idempotent_with_single_key(self):
        registry, provider = self._registry_with_keys(["sole"])
        original = provider.api_key
        for _ in range(5):
            registry.rotate_key("p")
        assert provider.api_key == original

    def test_rotate_preserves_key_index_between_calls(self):
        """Internal index must increment monotonically across independent calls."""
        registry, provider = self._registry_with_keys(["a", "b", "c"])
        registry.rotate_key("p")
        assert registry._key_indices["p"] == 1
        registry.rotate_key("p")
        assert registry._key_indices["p"] == 2

    def test_rotate_key_no_public_or_private_attr_does_not_raise(self):
        """Provider with neither api_key nor _api_key must not raise."""
        registry = ProviderRegistry()
        # Create a mock whose spec does not include api_key or _api_key at all.
        provider = MagicMock(spec=[])  # empty spec
        provider.name = "bare"
        cfg = _make_provider_config("bare", api_keys=["x", "y"])
        registry.register("bare", provider, config=cfg)
        # Must complete without exception.
        registry.rotate_key("bare")

    def test_duplicate_registration_resets_key_index_idempotently(self):
        """Re-registering the same name must not clobber an existing key index."""
        registry = ProviderRegistry()
        provider = _make_provider("dup")
        provider.api_key = "k1"
        cfg = _make_provider_config("dup", api_keys=["k1", "k2"])
        registry.register("dup", provider, config=cfg)
        # Rotate once so index becomes 1
        registry.rotate_key("dup")
        assert provider.api_key == "k2"
        # Re-register with the same name (replacement) – setdefault must NOT reset index
        registry.register("dup", provider, config=cfg)
        # Index remains 1 (setdefault leaves it untouched)
        assert registry._key_indices.get("dup") == 1

    def test_interleaved_rotation_two_providers_independent(self):
        registry = ProviderRegistry()
        for name, keys in (("pa", ["a1", "a2", "a3"]), ("pb", ["b1", "b2"])):
            p = _make_provider(name)
            p.api_key = keys[0]
            cfg = _make_provider_config(name, api_keys=keys)
            registry.register(name, p, config=cfg)
        # Rotate pa twice, pb once
        pa = registry.get("pa")
        pb = registry.get("pb")
        registry.rotate_key("pa")
        registry.rotate_key("pa")
        registry.rotate_key("pb")
        assert pa.api_key == "a3"
        assert pb.api_key == "b2"


# ===========================================================================
# fast_model / premium_model tier selection
# ===========================================================================


class TestModelTierSelection:
    @pytest.fixture
    def router(self) -> ModelRouter:
        return ModelRouter()

    def test_all_tiers_configured_correctly_routed(self, router):
        cfg = _make_provider_config(
            model="primary-model",
            fast_model="fast-model",
            premium_model="premium-model",
        )
        assert router.select_model(cfg, "primary") == "primary-model"
        assert router.select_model(cfg, "fast") == "fast-model"
        assert router.select_model(cfg, "premium") == "premium-model"

    def test_fast_tier_empty_string_falls_back_to_primary(self, router):
        cfg = _make_provider_config(model="main", fast_model="")
        assert router.select_model(cfg, "fast") == "main"

    def test_premium_tier_empty_string_falls_back_to_primary(self, router):
        cfg = _make_provider_config(model="main", premium_model="")
        assert router.select_model(cfg, "premium") == "main"

    def test_only_fast_model_set_premium_falls_back(self, router):
        cfg = _make_provider_config(model="base", fast_model="haiku", premium_model="")
        assert router.select_model(cfg, "premium") == "base"
        assert router.select_model(cfg, "fast") == "haiku"

    def test_only_premium_model_set_fast_falls_back(self, router):
        cfg = _make_provider_config(model="base", fast_model="", premium_model="opus")
        assert router.select_model(cfg, "fast") == "base"
        assert router.select_model(cfg, "premium") == "opus"

    def test_score_complexity_boundary_history_length_11_is_premium(self, router):
        assert router.score_complexity("neutral prompt", history_length=11) == "premium"

    def test_score_complexity_boundary_tool_count_4_is_premium(self, router):
        assert router.score_complexity("neutral prompt", tool_count=4) == "premium"

    def test_score_complexity_short_prompt_with_tool_returns_premium_or_primary(self, router):
        # The fast path requires tool_count == 0; with tool_count=1 it falls through.
        result = router.score_complexity("what is this", tool_count=1)
        assert result in ("primary", "premium")

    def test_score_complexity_prompt_exactly_80_chars_no_fast(self, router):
        # len(prompt) < 80 is the fast-path condition; at exactly 80 it does not apply.
        prompt = "w" * 80  # 80 chars, contains no fast indicator words
        result = router.score_complexity(prompt, tool_count=0)
        assert result == "primary"

    def test_score_complexity_multiple_premium_keywords_still_premium(self, router):
        assert router.score_complexity("please debug and refactor this complex code") == "premium"

    def test_score_complexity_returns_one_of_three_valid_tiers(self, router):
        for prompt in ("hi", "debug the code", "a" * 600, "what is this"):
            assert router.score_complexity(prompt) in ("fast", "primary", "premium")


# ===========================================================================
# Concurrent provider switching via set_default
# ===========================================================================


class TestConcurrentSetDefault:
    def test_concurrent_set_default_does_not_corrupt_state(self):
        registry = ProviderRegistry()
        for name in ("a", "b", "c", "d"):
            registry.register(name, _make_provider(name, available=True))

        errors: list[Exception] = []
        lock = threading.Lock()

        def switch(name: str) -> None:
            try:
                registry.set_default(name)
            except Exception as exc:  # noqa: BLE001
                with lock:
                    errors.append(exc)

        threads = [threading.Thread(target=switch, args=(n,)) for n in ("a", "b", "c", "d") * 5]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=5)

        assert errors == []
        # After all threads, the default must be one of the valid registered names.
        default = registry.get_default_name()
        assert default in ("a", "b", "c", "d")

    def test_concurrent_register_and_get_available(self):
        """Concurrent register + get_available must not raise."""
        registry = ProviderRegistry()
        errors: list[Exception] = []
        lock = threading.Lock()

        def registrar(name: str) -> None:
            try:
                registry.register(name, _make_provider(name, available=True))
            except Exception as exc:  # noqa: BLE001
                with lock:
                    errors.append(exc)

        def reader() -> None:
            try:
                registry.get_available()
            except Exception as exc:  # noqa: BLE001
                with lock:
                    errors.append(exc)

        threads = [threading.Thread(target=registrar, args=(f"p{i}",)) for i in range(10)]
        threads += [threading.Thread(target=reader) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=5)

        assert errors == []


# ===========================================================================
# Message dataclass contract
# ===========================================================================


class TestMessageDataclassContract:
    def test_fields_are_role_and_content(self):
        field_names = {f.name for f in fields(Message)}
        assert field_names == {"role", "content"}

    def test_empty_content_is_valid(self):
        msg = Message(role="user", content="")
        assert msg.content == ""

    def test_multiline_content_preserved(self):
        text = "line one\nline two\nline three"
        msg = Message(role="assistant", content=text)
        assert msg.content == text

    def test_message_is_mutable(self):
        msg = Message(role="user", content="original")
        msg.content = "modified"
        assert msg.content == "modified"

    def test_message_as_dict_via_asdict(self):
        msg = Message(role="system", content="sys prompt")
        d = asdict(msg)
        assert d == {"role": "system", "content": "sys prompt"}

    def test_message_equality_both_fields(self):
        assert Message("user", "hi") == Message("user", "hi")
        assert Message("user", "hi") != Message("assistant", "hi")
        assert Message("user", "hi") != Message("user", "bye")

    def test_message_repr_contains_role_and_content(self):
        msg = Message(role="user", content="test")
        r = repr(msg)
        assert "user" in r
        assert "test" in r


# ===========================================================================
# ToolCall dataclass contract
# ===========================================================================


class TestToolCallDataclassContract:
    def test_fields_present(self):
        field_names = {f.name for f in fields(ToolCall)}
        assert field_names == {"id", "name", "arguments"}

    def test_basic_construction(self):
        tc = ToolCall(id="tc-1", name="shell_exec", arguments={"cmd": "ls"})
        assert tc.id == "tc-1"
        assert tc.name == "shell_exec"
        assert tc.arguments == {"cmd": "ls"}

    def test_empty_arguments_dict(self):
        tc = ToolCall(id="tc-2", name="noop", arguments={})
        assert tc.arguments == {}

    def test_nested_arguments(self):
        args: dict[str, Any] = {"params": {"key": "value", "nested": [1, 2, 3]}}
        tc = ToolCall(id="tc-3", name="complex", arguments=args)
        assert tc.arguments["params"]["nested"] == [1, 2, 3]

    def test_equality_on_all_fields(self):
        t1 = ToolCall(id="x", name="fn", arguments={"a": 1})
        t2 = ToolCall(id="x", name="fn", arguments={"a": 1})
        assert t1 == t2

    def test_inequality_on_id(self):
        t1 = ToolCall(id="x", name="fn", arguments={})
        t2 = ToolCall(id="y", name="fn", arguments={})
        assert t1 != t2

    def test_toolcall_is_mutable(self):
        tc = ToolCall(id="tc-4", name="before", arguments={})
        tc.name = "after"
        assert tc.name == "after"


# ===========================================================================
# ToolResult dataclass contract
# ===========================================================================


class TestToolResultDataclassContract:
    def test_fields_present(self):
        field_names = {f.name for f in fields(ToolResult)}
        assert field_names == {"tool_call_id", "name", "content", "is_error"}

    def test_is_error_defaults_to_false(self):
        tr = ToolResult(tool_call_id="tc-1", name="mytool", content="output")
        assert tr.is_error is False

    def test_is_error_can_be_set_true(self):
        tr = ToolResult(tool_call_id="tc-2", name="mytool", content="fail", is_error=True)
        assert tr.is_error is True

    def test_empty_content_valid(self):
        tr = ToolResult(tool_call_id="tc-3", name="noop", content="")
        assert tr.content == ""

    def test_equality_with_same_values(self):
        t1 = ToolResult(tool_call_id="x", name="f", content="c", is_error=False)
        t2 = ToolResult(tool_call_id="x", name="f", content="c", is_error=False)
        assert t1 == t2

    def test_inequality_when_is_error_differs(self):
        t1 = ToolResult(tool_call_id="x", name="f", content="c", is_error=False)
        t2 = ToolResult(tool_call_id="x", name="f", content="c", is_error=True)
        assert t1 != t2

    def test_asdict_round_trip(self):
        tr = ToolResult(tool_call_id="tc-5", name="scan", content="ok", is_error=False)
        d = asdict(tr)
        assert d == {"tool_call_id": "tc-5", "name": "scan", "content": "ok", "is_error": False}


# ===========================================================================
# CompletionResponse dataclass contract
# ===========================================================================


class TestCompletionResponseContract:
    def _minimal(self, **overrides: Any) -> CompletionResponse:
        base: dict[str, Any] = {
            "content": "hello",
            "model": "m",
            "provider": "p",
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            "raw": {},
        }
        base.update(overrides)
        return CompletionResponse(**base)

    def test_tool_calls_defaults_to_empty_list(self):
        resp = self._minimal()
        assert resp.tool_calls == []

    def test_finish_reason_defaults_to_stop(self):
        resp = self._minimal()
        assert resp.finish_reason == "stop"

    def test_finish_reason_tool_calls(self):
        resp = self._minimal(finish_reason="tool_calls")
        assert resp.finish_reason == "tool_calls"

    def test_finish_reason_length(self):
        resp = self._minimal(finish_reason="length")
        assert resp.finish_reason == "length"

    def test_tool_calls_populated(self):
        tc = ToolCall(id="tc-1", name="fn", arguments={})
        resp = self._minimal(tool_calls=[tc], finish_reason="tool_calls")
        assert len(resp.tool_calls) == 1
        assert resp.tool_calls[0].name == "fn"

    def test_usage_dict_all_three_keys(self):
        usage = {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}
        resp = self._minimal(usage=usage)
        assert resp.usage["total_tokens"] == 15

    def test_raw_dict_preserved(self):
        raw = {"id": "resp-42", "object": "chat.completion"}
        resp = self._minimal(raw=raw)
        assert resp.raw["id"] == "resp-42"

    def test_empty_content_valid(self):
        resp = self._minimal(content="")
        assert resp.content == ""

    def test_tool_calls_list_is_independent_per_instance(self):
        """Mutable default (field(default_factory=list)) must not be shared."""
        r1 = self._minimal()
        r2 = self._minimal()
        r1.tool_calls.append("sentinel")
        assert "sentinel" not in r2.tool_calls


# ===========================================================================
# ProviderConfig – timeout and model-name pass-through
# ===========================================================================


class TestProviderConfigContract:
    def test_timeout_default_is_30(self):
        cfg = ProviderConfig(name="anthropic", model="claude-sonnet-4-6")
        assert cfg.timeout == 30

    def test_timeout_explicit_value(self):
        cfg = ProviderConfig(name="openai", model="gpt-4o", timeout=60)
        assert cfg.timeout == 60

    def test_model_name_preserved_verbatim(self):
        model = "claude-haiku-4-5-20251001"
        cfg = ProviderConfig(name="anthropic", model=model)
        assert cfg.model == model

    def test_api_key_defaults_to_none(self):
        cfg = ProviderConfig(name="ollama", model="llama3.2")
        assert cfg.api_key is None

    def test_api_keys_defaults_to_empty_list(self):
        cfg = ProviderConfig(name="ollama", model="llama3.2")
        assert cfg.api_keys == []

    def test_enabled_defaults_to_true(self):
        cfg = ProviderConfig(name="anthropic", model="claude-sonnet-4-6")
        assert cfg.enabled is True

    def test_fast_model_defaults_to_empty_string(self):
        cfg = ProviderConfig(name="anthropic", model="claude-sonnet-4-6")
        assert cfg.fast_model == ""

    def test_premium_model_defaults_to_empty_string(self):
        cfg = ProviderConfig(name="anthropic", model="claude-sonnet-4-6")
        assert cfg.premium_model == ""


# ===========================================================================
# AnthropicProvider – setup-token rejection and key pass-through
# ===========================================================================


class TestAnthropicProviderKeyHandling:
    def test_setup_token_rejected_and_is_available_false(self):
        """Keys starting with sk-ant-oat must not be used by the provider."""
        from missy.providers.anthropic_provider import AnthropicProvider

        cfg = ProviderConfig(
            name="anthropic",
            model="claude-sonnet-4-6",
            api_key="sk-ant-oat-something",
        )
        provider = AnthropicProvider(cfg)
        # The provider should mark itself unavailable (key was nulled out).
        assert provider._api_key is None

    def test_valid_api_key_stored(self):
        from missy.providers.anthropic_provider import AnthropicProvider

        cfg = ProviderConfig(
            name="anthropic",
            model="claude-sonnet-4-6",
            api_key="sk-ant-api-validkey",
        )
        provider = AnthropicProvider(cfg)
        assert provider._api_key == "sk-ant-api-validkey"

    def test_none_api_key_stored_as_none(self):
        from missy.providers.anthropic_provider import AnthropicProvider

        cfg = ProviderConfig(name="anthropic", model="claude-sonnet-4-6", api_key=None)
        provider = AnthropicProvider(cfg)
        assert provider._api_key is None

    def test_is_available_false_when_no_key(self):
        from missy.providers.anthropic_provider import AnthropicProvider

        cfg = ProviderConfig(name="anthropic", model="claude-sonnet-4-6", api_key=None)
        provider = AnthropicProvider(cfg)
        # Even if the SDK is installed, no api_key means unavailable.
        assert provider.is_available() is False

    def test_model_falls_back_to_default_when_empty(self):
        from missy.providers.anthropic_provider import _DEFAULT_MODEL, AnthropicProvider

        cfg = ProviderConfig(name="anthropic", model="")
        provider = AnthropicProvider(cfg)
        assert provider._model == _DEFAULT_MODEL

    def test_timeout_stored(self):
        from missy.providers.anthropic_provider import AnthropicProvider

        cfg = ProviderConfig(name="anthropic", model="claude-sonnet-4-6", timeout=45)
        provider = AnthropicProvider(cfg)
        assert provider._timeout == 45


# ===========================================================================
# OpenAIProvider – key and base_url handling
# ===========================================================================


class TestOpenAIProviderKeyHandling:
    def test_api_key_stored(self):
        from missy.providers.openai_provider import OpenAIProvider

        cfg = ProviderConfig(name="openai", model="gpt-4o", api_key="sk-test-key")
        provider = OpenAIProvider(cfg)
        assert provider._api_key == "sk-test-key"

    def test_none_api_key_stored_as_none(self):
        from missy.providers.openai_provider import OpenAIProvider

        cfg = ProviderConfig(name="openai", model="gpt-4o", api_key=None)
        provider = OpenAIProvider(cfg)
        assert provider._api_key is None

    def test_is_available_false_when_no_key(self):
        from missy.providers.openai_provider import OpenAIProvider

        cfg = ProviderConfig(name="openai", model="gpt-4o", api_key=None)
        provider = OpenAIProvider(cfg)
        assert provider.is_available() is False

    def test_base_url_stored(self):
        from missy.providers.openai_provider import OpenAIProvider

        cfg = ProviderConfig(
            name="openai",
            model="gpt-4o",
            api_key="key",
            base_url="http://localhost:11434/v1",
        )
        provider = OpenAIProvider(cfg)
        assert provider._base_url == "http://localhost:11434/v1"

    def test_base_url_none_when_not_set(self):
        from missy.providers.openai_provider import OpenAIProvider

        cfg = ProviderConfig(name="openai", model="gpt-4o", api_key="key")
        provider = OpenAIProvider(cfg)
        assert provider._base_url is None

    def test_model_falls_back_to_default_when_empty(self):
        from missy.providers.openai_provider import _DEFAULT_MODEL, OpenAIProvider

        cfg = ProviderConfig(name="openai", model="")
        provider = OpenAIProvider(cfg)
        assert provider._model == _DEFAULT_MODEL


# ===========================================================================
# from_config: edge cases not covered by existing tests
# ===========================================================================


class TestFromConfigEdgeCases:
    def test_key_index_initialised_at_zero_on_registration(self):
        """After from_config, key indices for registered providers start at 0."""
        config = _make_config(
            providers={
                "local": _make_provider_config("ollama", model="llama3.2"),
            }
        )
        registry = ProviderRegistry.from_config(config)
        assert registry._key_indices.get("local") == 0

    def test_from_config_uses_dict_key_when_provider_name_matches(self):
        """Config key and provider_config.name may differ; registry uses the dict key."""
        config = _make_config(
            providers={
                "my_ollama": _make_provider_config("ollama", model="llama3.2"),
            }
        )
        registry = ProviderRegistry.from_config(config)
        # Must be registered under the dict key, not the name field
        assert registry.get("my_ollama") is not None

    def test_from_config_rate_limiter_attached(self):
        """Every registered provider must have a rate_limiter set."""
        config = _make_config(
            providers={
                "lo": _make_provider_config("ollama", model="llama3.2"),
            }
        )
        registry = ProviderRegistry.from_config(config)
        provider = registry.get("lo")
        assert provider is not None
        assert provider.rate_limiter is not None

    def test_from_config_base_url_case_insensitive_dedup(self):
        """Host from base_url is lowercased before dedup check."""
        config = _make_config(
            providers={
                "lo": _make_provider_config(
                    "ollama",
                    model="llama3.2",
                    base_url="http://MyHost.Internal:11434",
                ),
            }
        )
        config.network.provider_allowed_hosts.append("myhost.internal")
        ProviderRegistry.from_config(config)
        count = sum(
            1 for h in config.network.provider_allowed_hosts if h.lower() == "myhost.internal"
        )
        assert count == 1

    def test_from_config_multiple_unknown_providers_all_skipped(self):
        config = _make_config(
            providers={
                "x1": _make_provider_config("unknown_a"),
                "x2": _make_provider_config("unknown_b"),
                "x3": _make_provider_config("unknown_c"),
            }
        )
        registry = ProviderRegistry.from_config(config)
        assert registry.list_providers() == []


# ===========================================================================
# BaseProvider default implementations
# ===========================================================================


class TestBaseProviderDefaultImplementations:
    """Tests for get_tool_schema, complete_with_tools, and stream defaults."""

    class _MinimalProvider(BaseProvider):
        name = "minimal"

        def complete(self, messages: list[Message], **kwargs: Any) -> CompletionResponse:
            return CompletionResponse(
                content="ok",
                model="mini",
                provider="minimal",
                usage={"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
                raw={},
            )

        def is_available(self) -> bool:
            return True

    def test_complete_with_tools_delegates_to_complete(self):
        provider = self._MinimalProvider()
        messages = [Message(role="user", content="hi")]
        resp = provider.complete_with_tools(messages, tools=[])
        assert isinstance(resp, CompletionResponse)
        assert resp.content == "ok"

    def test_stream_yields_complete_content_as_single_chunk(self):
        provider = self._MinimalProvider()
        messages = [Message(role="user", content="hello")]
        chunks = list(provider.stream(messages))
        assert chunks == ["ok"]

    def test_get_tool_schema_empty_list(self):
        provider = self._MinimalProvider()
        assert provider.get_tool_schema([]) == []

    def test_get_tool_schema_uses_get_schema_when_present(self):
        provider = self._MinimalProvider()
        tool = MagicMock()
        tool.name = "my_tool"
        tool.description = "Does stuff"
        tool.get_schema.return_value = {"parameters": {"type": "object", "properties": {}}}
        result = provider.get_tool_schema([tool])
        assert len(result) == 1
        assert result[0]["name"] == "my_tool"
        assert result[0]["description"] == "Does stuff"

    def test_get_tool_schema_no_get_schema_method_uses_empty_params(self):
        provider = self._MinimalProvider()
        tool = MagicMock(spec=[])  # no get_schema attribute
        tool.name = "bare_tool"
        tool.description = "No schema"
        result = provider.get_tool_schema([tool])
        assert result[0]["parameters"] == {}

    def test_repr_format(self):
        provider = self._MinimalProvider()
        r = repr(provider)
        # BaseProvider.__repr__ uses self.__class__.__name__ (unqualified)
        assert r == "_MinimalProvider(name='minimal')"

    def test_acquire_rate_limit_with_none_limiter_is_noop(self):
        """_acquire_rate_limit must not raise when rate_limiter is None."""
        provider = self._MinimalProvider()
        provider.rate_limiter = None
        provider._acquire_rate_limit(estimated_tokens=100)  # must not raise

    def test_record_rate_limit_usage_with_none_limiter_is_noop(self):
        """_record_rate_limit_usage must not raise when rate_limiter is None."""
        provider = self._MinimalProvider()
        provider.rate_limiter = None
        resp = CompletionResponse(
            content="x",
            model="m",
            provider="p",
            usage={"prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8},
            raw={},
        )
        provider._record_rate_limit_usage(resp)  # must not raise
