"""Fail-closed registration checks for the optional RepoEval Foundry tool."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

import missy.tools.builtin as builtin
from missy.tools.registry import ToolRegistry


def _config(**overrides: object) -> SimpleNamespace:
    config = {
        "enabled": True,
        "api_available": True,
        "base_url": "https://foundry.example.test/api",
        "project_id": "sample-project",
        "allowed_hosts": ["foundry.example.test"],
    }
    config.update(overrides)
    return SimpleNamespace(**config)


def _registration(config: SimpleNamespace | None, monkeypatch: pytest.MonkeyPatch) -> ToolRegistry:
    # Isolate this opt-in gate from the unrelated builtins and their dependencies.
    monkeypatch.setattr(builtin, "_ALL_TOOL_CLASSES", [])
    registry = ToolRegistry()
    builtin.register_builtin_tools(registry, repoeval_foundry_config=config)
    return registry


def test_no_config_does_not_expose_foundry_tool(monkeypatch: pytest.MonkeyPatch) -> None:
    registry = _registration(None, monkeypatch)
    assert "repoeval_foundry" not in registry.list_tools()


@pytest.mark.parametrize(
    "overrides",
    [
        {"enabled": False},
        {"api_available": False},
        {"base_url": ""},
        {"base_url": "not-an-endpoint"},
        {"base_url": "https://foundry.example.test:invalid/api"},
        {"base_url": "https://other.example.test/api"},
        {"base_url": "https://user:pass@foundry.example.test/api"},
        {"base_url": "https://foundry.example.test/api?secret=oops"},
        {"project_id": " "},
        {"allowed_hosts": []},
        {"allowed_hosts": ["example.test"]},
        {"allowed_hosts": ["foundry.example.test:443"]},
    ],
)
def test_incomplete_or_mismatched_configuration_never_registers(
    overrides: dict[str, object], monkeypatch: pytest.MonkeyPatch
) -> None:
    registry = _registration(_config(**overrides), monkeypatch)
    assert "repoeval_foundry" not in registry.list_tools()


def test_complete_explicit_configuration_registers_with_protected_token(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    token = tmp_path / "token"
    token.write_text("test-only-token", encoding="ascii")
    token.chmod(0o600)
    registry = _registration(
        _config(
            allowed_hosts=["FOUNDRY.EXAMPLE.TEST"],
            token_file=str(token),
        ),
        monkeypatch,
    )
    tool = registry.get("repoeval_foundry")
    assert tool is not None
    assert tool.permissions.network
    assert tool.permissions.allowed_hosts == ["foundry.example.test"]
    assert tool._client.category == "tool"


def test_missing_token_never_registers(monkeypatch):
    registry = _registration(_config(), monkeypatch)
    assert "repoeval_foundry" not in registry.list_tools()


@pytest.mark.parametrize(
    "overrides",
    [
        {"enabled": False},
        {"enabled": "true"},
        {"api_available": False},
        {"api_available": "true"},
        {"project_id": "../other"},
        {"base_url": "https://foundry.example.test:invalid/api"},
        {"allowed_hosts": []},
    ],
)
def test_protected_token_cannot_override_other_gates(overrides, monkeypatch, tmp_path):
    token = tmp_path / "token"
    token.write_text("test-only-token", encoding="ascii")
    token.chmod(0o600)
    registry = _registration(_config(token_file=str(token), **overrides), monkeypatch)
    assert "repoeval_foundry" not in registry.list_tools()


def test_invalid_token_file_modes_and_symlinks_not_registered(monkeypatch, tmp_path):
    token = tmp_path / "token"
    token.write_text("test-only-token", encoding="ascii")
    token.chmod(0o644)
    assert (
        "repoeval_foundry"
        not in _registration(_config(token_file=str(token)), monkeypatch).list_tools()
    )
    token.chmod(0o600)
    link = tmp_path / "link"
    link.symlink_to(token)
    assert (
        "repoeval_foundry"
        not in _registration(_config(token_file=str(link)), monkeypatch).list_tools()
    )
