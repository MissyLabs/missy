"""DATA-05: a config hot reload must not reset live provider state."""

from __future__ import annotations

import copy
from unittest.mock import patch

import pytest

from missy.providers import registry as registry_mod
from missy.providers.registry import get_registry, init_registry
from tests.providers.test_registry import _make_config


@pytest.fixture(autouse=True)
def _restore_registry():
    saved = registry_mod._registry
    registry_mod._registry = None
    yield
    registry_mod._registry = saved


def _two_providers():
    from missy.config.settings import ProviderConfig

    return _make_config(
        {
            "anthropic": ProviderConfig(name="anthropic", model="claude-test", api_key="k1"),
            "openai": ProviderConfig(name="openai", model="gpt-test", api_key="k2"),
        }
    )


def test_unchanged_provider_keeps_instance_and_limiter():
    cfg = _two_providers()
    first = init_registry(cfg)
    provider = first.get("anthropic")
    limiter = provider.rate_limiter
    reloaded = init_registry(copy.deepcopy(cfg))
    assert reloaded.get("anthropic") is provider
    assert reloaded.get("anthropic").rate_limiter is limiter


def test_changed_provider_is_rebuilt():
    cfg = _two_providers()
    first = init_registry(cfg)
    old = first.get("openai")
    changed = copy.deepcopy(cfg)
    changed.providers["openai"].model = "gpt-other"
    assert init_registry(changed).get("openai") is not old


def test_runtime_disable_and_switched_default_survive_reload():
    cfg = _two_providers()
    reg = init_registry(cfg)
    with patch.object(type(reg), "_availability_for", return_value=True):
        reg.set_default("anthropic")
    reg.set_enabled("openai", False)
    reloaded = init_registry(copy.deepcopy(cfg))
    assert reloaded.is_enabled("openai") is False
    assert reloaded.get_default_name() == "anthropic"


def test_hotreload_only_reseeds_default_when_config_changes():
    from missy.config import hotreload

    cfg = _two_providers()
    cfg.default_provider = "openai"
    init_registry(cfg)
    reg = get_registry()
    with patch.object(type(reg), "_availability_for", return_value=True):
        reg.set_default("anthropic")  # operator switch at runtime
        with (
            patch("missy.observability.otel.init_otel"),
            patch("missy.observability.audit_logger.init_audit_logger"),
        ):
            hotreload._apply_config(copy.deepcopy(cfg))  # unrelated reload
            assert get_registry().get_default_name() == "anthropic"
            changed = copy.deepcopy(cfg)
            changed.default_provider = "anthropic"
            hotreload._apply_config(changed)  # config default changed -> re-seeded
            assert get_registry().get_default_name() == "anthropic"
            back = copy.deepcopy(cfg)  # default_provider: openai again
            hotreload._apply_config(back)
            assert get_registry().get_default_name() == "openai"


def test_provider_limiter_uses_configured_rpm():
    """RATE-02: requests_per_minute from config is the effective limit."""
    from missy.config.settings import ProviderConfig

    cfg = _make_config(
        {
            "anthropic": ProviderConfig(
                name="anthropic", model="m", api_key="k", requests_per_minute=600
            )
        }
    )
    limiter = init_registry(cfg).get("anthropic").rate_limiter
    assert limiter.requests_per_minute == 600
