"""Hot reload must retire old Foundry credentials without touching other tools."""

from __future__ import annotations

from dataclasses import replace
from unittest.mock import patch

import pytest

from missy.config.hotreload import ConfigWatcher, _apply_config
from missy.config.settings import RepoevalFoundryConfig, get_default_config
from missy.tools.builtin.repoeval_tools import RepoevalFoundryTool
from missy.tools.registry import ToolRegistry


class Client:
    category = "tool"
    response_limit = 1024 * 1024

    def __init__(self):
        self.calls = []

    def get_limited(self, url, max_bytes, **kwargs):
        assert max_bytes == self.response_limit
        assert kwargs.get("follow_redirects") is False
        assert kwargs.get("headers", {}).get("Authorization") == "Bearer fixture-token"
        self.calls.append((url, max_bytes, kwargs))
        return self

    status_code = 200

    def json(self):
        return {"ok": True, "data": ["repo-a"]}


def _reload(config, registry, monkeypatch):
    monkeypatch.setattr("missy.tools.registry.get_tool_registry", lambda: registry)
    with (
        patch("missy.policy.engine.PolicyEngine"),
        patch("missy.providers.registry.ProviderRegistry.from_config"),
        patch("missy.policy.engine.init_policy_engine"),
        patch("missy.providers.registry.init_registry"),
        patch("missy.observability.otel.init_otel"),
        patch("missy.observability.audit_logger.init_audit_logger"),
    ):
        _apply_config(config)


def _setup(tmp_path):
    token = tmp_path / "token"
    token.write_text("fixture-token\n", encoding="ascii")
    token.chmod(0o600)
    foundry_config = RepoevalFoundryConfig(
        enabled=True,
        api_available=True,
        base_url="https://foundry.example/api",
        project_id="alpha",
        allowed_hosts=["foundry.example"],
        token_file=str(token),
    )
    config = get_default_config()
    config.repoeval_foundry = foundry_config
    client = Client()
    tool = RepoevalFoundryTool(
        base_url=foundry_config.base_url,
        project_id=foundry_config.project_id,
        allowed_hosts=foundry_config.allowed_hosts,
        token_file=foundry_config.token_file,
        api_available=True,
        http_client=client,
    )
    registry = ToolRegistry()
    registry.register(tool)
    assert tool.registration_ready
    assert tool.execute(action="list").success
    assert len(client.calls) == 1
    assert client.calls[0][1] == client.response_limit
    assert client.calls[0][2]["follow_redirects"] is False
    assert client.calls[0][2]["headers"]["Authorization"] == "Bearer fixture-token"
    return config, tool, client, registry


@pytest.mark.parametrize(
    "change",
    [
        {"enabled": False},
        {"api_available": False},
        {"project_id": "beta"},
        {"base_url": "https://foundry.example"},
        {"token_file": "/different/token"},
        {"allowed_hosts": ["foundry.example", "other.example"]},
    ],
)
def test_reload_revokes_old_client_without_request(tmp_path, monkeypatch, change):
    config, tool, client, registry = _setup(tmp_path)
    next_config = replace(config, repoeval_foundry=replace(config.repoeval_foundry, **change))
    _reload(next_config, registry, monkeypatch)

    assert registry.get("repoeval_foundry") is tool  # stale registration is harmless
    assert not tool.registration_ready
    assert not tool.execute(action="list").success
    assert not registry.get("repoeval_foundry").execute(action="list").success
    assert len(client.calls) == 1


def test_unchanged_reload_keeps_healthy_client(tmp_path, monkeypatch):
    config, tool, client, registry = _setup(tmp_path)
    _reload(
        replace(config, repoeval_foundry=replace(config.repoeval_foundry)), registry, monkeypatch
    )

    assert tool.registration_ready
    assert tool.execute(action="list").success
    assert len(client.calls) == 2


def test_watcher_reload_revokes_disabled_client(tmp_path, monkeypatch):
    config, tool, client, registry = _setup(tmp_path)
    disabled = replace(config, repoeval_foundry=replace(config.repoeval_foundry, enabled=False))
    watcher = ConfigWatcher(
        str(tmp_path / "config.yaml"), lambda cfg: _reload(cfg, registry, monkeypatch)
    )
    watcher._active_config = config
    monkeypatch.setattr(watcher, "_check_file_safety", lambda: True)
    monkeypatch.setattr("missy.config.settings.load_config", lambda _: disabled)

    watcher._do_reload()

    assert watcher._active_config is disabled
    assert not tool.execute(action="list").success
    assert len(client.calls) == 1
