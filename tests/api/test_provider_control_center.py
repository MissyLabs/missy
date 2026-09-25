"""Focused tests for the provider control center's safe mutation surface."""

from __future__ import annotations

from unittest.mock import MagicMock

import yaml

from missy.agent.runtime import AgentConfig
from missy.api.operator_controls import execute_operator_control, list_operator_controls
from missy.api.server import _agent_settings_json, _provider_weight
from missy.api.web_console import render_page


def _config(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text(
        """
providers:
  openai:
    model: gpt-5.6-sol
    oauth_accounts: [primary, reserve]
max_sub_agents: 10
max_concurrent_agents: 3
""".lstrip(),
        encoding="utf-8",
    )
    return path


def _registry():
    registry = MagicMock()
    registry.list_providers.return_value = ["openai"]
    return registry


def test_controls_advertise_agent_and_account_configuration(tmp_path):
    controls = list_operator_controls(
        provider_registry=_registry(), config_path=str(_config(tmp_path))
    )["controls"]

    ids = {control["id"] for control in controls}
    assert "agent.set_field" in ids
    assert "provider.set_account_weights" in ids


def test_agent_setting_requires_value_bound_confirmation(tmp_path):
    path = _config(tmp_path)

    status, data, detail = execute_operator_control(
        "agent.set_field",
        {"field": "max_sub_agents", "value": 14, "confirm": "set-agent-field:max_sub_agents:10"},
        config_path=str(path),
    )

    assert status == 409
    assert data["confirmation"] == "set-agent-field:max_sub_agents:14"
    assert detail["reason"] == "confirmation_required"
    assert yaml.safe_load(path.read_text())["max_sub_agents"] == 10


def test_agent_setting_persists_after_exact_confirmation(tmp_path):
    path = _config(tmp_path)

    status, data, detail = execute_operator_control(
        "agent.set_field",
        {"field": "max_sub_agents", "value": 14, "confirm": "set-agent-field:max_sub_agents:14"},
        config_path=str(path),
    )

    assert status == 200
    assert data["value"] == 14
    assert detail["reason"] == "confirmed"
    assert yaml.safe_load(path.read_text())["max_sub_agents"] == 14


def test_account_weights_persist_without_exposing_credentials(tmp_path):
    path = _config(tmp_path)

    status, data, detail = execute_operator_control(
        "provider.set_account_weights",
        {
            "target": "openai",
            "value": [3, 1],
            "confirm": "set-account-weights:openai:3,1",
        },
        provider_registry=_registry(),
        config_path=str(path),
    )

    assert status == 200
    assert data["value"] == [3.0, 1.0]
    assert detail["reason"] == "confirmed"
    raw = yaml.safe_load(path.read_text())
    assert raw["providers"]["openai"]["account_weights"] == [3.0, 1.0]
    assert "api_key" not in data


def test_zero_provider_weight_is_preserved():
    config = MagicMock()
    config.weight = 0.0

    assert _provider_weight(config) == 0.0


def test_agent_settings_payload_uses_configured_values(tmp_path):
    path = _config(tmp_path)
    raw = yaml.safe_load(path.read_text())
    raw.update(
        temperature=1.3,
        max_sub_agent_depth=4,
        max_spend_usd=2.5,
        global_max_spend_usd=25,
        global_budget_period="monthly",
    )
    raw["features"] = {"model_routing_enabled": True}
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    runtime = MagicMock()
    runtime.config = AgentConfig()

    settings = _agent_settings_json(runtime, str(path))

    assert settings["temperature"] == 1.3
    assert settings["max_sub_agents"] == 10
    assert settings["max_sub_agent_depth"] == 4
    assert settings["max_spend_usd"] == 2.5
    assert settings["global_max_spend_usd"] == 25
    assert settings["global_budget_period"] == "monthly"
    assert settings["model_routing_enabled"] is True
    assert settings["editable"] is True
    assert settings["hot_reload"] is True


def test_provider_page_renders_complete_control_center():
    page = render_page("providers", csrf_token="token")

    assert "Provider control center" in page
    assert 'id="provider-kpis"' in page
    assert 'id="agent-settings"' in page
    assert 'id="balancing-board"' in page
    assert "/controls/agent.set_field" in page
    assert "/controls/provider.set_account_weights" in page
    assert "api_key_configured" in page
