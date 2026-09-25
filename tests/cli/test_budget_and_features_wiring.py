"""BUDGET-01 / GAP-02: global budget installed at every entry point; features:
config reaches AgentConfig."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner

from missy.agent.cost_tracker import BudgetExceededError
from missy.agent.global_budget import GlobalBudget, get_global_budget, init_global_budget
from missy.cli.main import _agent_feature_kwargs, _install_global_budget, cli
from missy.config.settings import FeaturesConfig, _parse_features, _parse_retention


class TestInstallGlobalBudget:
    def test_installs_configured_cap(self, tmp_path, monkeypatch):
        monkeypatch.setattr(
            "missy.agent.global_budget.DEFAULT_GLOBAL_BUDGET_PATH", str(tmp_path / "gb.json")
        )
        cfg = MagicMock(global_max_spend_usd=5.0, global_budget_period="daily")
        _install_global_budget(cfg)
        budget = get_global_budget()
        assert budget.enabled and budget.max_spend_usd == 5.0 and budget.period == "daily"

    def test_zero_installs_disabled_budget(self):
        init_global_budget(5.0)
        _install_global_budget(MagicMock(global_max_spend_usd=0.0))
        assert get_global_budget().enabled is False

    def test_mock_values_do_not_enable(self):
        _install_global_budget(MagicMock())  # attributes are MagicMocks
        assert get_global_budget().enabled is False

    def test_ask_installs_budget_before_runtime(self, tmp_path):
        cfg_path = tmp_path / "c.yaml"
        cfg_path.write_text("providers: {}\n")
        order: list[str] = []
        cfg = MagicMock(max_spend_usd=0.0, providers={"anthropic": MagicMock(model="m")})
        with (
            patch("missy.cli.main._load_subsystems", return_value=cfg),
            patch(
                "missy.cli.main._install_global_budget",
                side_effect=lambda *_a, **_k: order.append("budget"),
            ),
            patch(
                "missy.agent.runtime.AgentRuntime",
                side_effect=lambda *_a, **_k: order.append("runtime") or MagicMock(),
            ),
            patch("missy.agent.runtime.AgentConfig"),
            patch("missy.agent.hatching.HatchingManager") as hatch,
        ):
            hatch.return_value.needs_hatching.return_value = False
            CliRunner().invoke(cli, ["--config", str(cfg_path), "ask", "hi"])
        assert order[:2] == ["budget", "runtime"]

    @pytest.fixture(autouse=True)
    def _reset_active(self):
        yield
        init_global_budget(0.0)


class TestCorruptBudgetFailsClosed:
    def test_corrupt_file_is_quarantined_and_blocks_spend(self, tmp_path):
        path = tmp_path / "gb.json"
        path.write_text("{not json")
        budget = GlobalBudget(10.0, path=path)
        with (
            patch("missy.core.events.event_bus.publish") as publish,
            pytest.raises(BudgetExceededError),
        ):
            budget.check()
        assert (tmp_path / "gb.json.corrupt").exists()
        assert publish.call_args.args[0].event_type == "budget.global.corrupt"
        # Persisted: a fresh instance still sees the period as spent.
        assert GlobalBudget(10.0, path=path).total_spent() == pytest.approx(10.0)
        budget.reset()
        assert budget.total_spent() == 0.0

    def test_negative_spent_is_rejected(self, tmp_path):
        path = tmp_path / "gb.json"
        budget = GlobalBudget(10.0, path=path)
        budget.record(1.0)
        data = json.loads(path.read_text())
        data["spent"] = -100
        path.write_text(json.dumps(data))
        assert budget.total_spent() == pytest.approx(10.0)

    def test_missing_file_is_a_fresh_budget(self, tmp_path):
        assert GlobalBudget(10.0, path=tmp_path / "none.json").total_spent() == 0.0


class TestGatewayOrdering:
    def test_budget_installed_before_proactive_and_scheduler(self):
        import inspect

        from missy.cli import main

        src = inspect.getsource(main.gateway_start.callback)
        budget = src.index("_install_global_budget(cfg, announce=True)")
        assert budget < src.index("ProactiveManager(")
        assert budget < src.index("scheduler_manager.start()")


class TestFeatures:
    def test_parse_features(self):
        f = _parse_features({"model_routing_enabled": True, "condenser_min_messages": 12})
        assert f.model_routing_enabled is True
        assert f.graph_memory_enabled is False
        assert f.condenser_min_messages == 12

    def test_feature_kwargs_from_config(self):
        cfg = MagicMock()
        cfg.features = FeaturesConfig(model_routing_enabled=True, graph_memory_enabled=True)
        kwargs = _agent_feature_kwargs(cfg)
        assert kwargs["model_routing_enabled"] is True
        assert kwargs["graph_memory_enabled"] is True
        assert kwargs["condenser_min_messages"] == 30

    def test_feature_kwargs_ignore_mocks(self):
        assert _agent_feature_kwargs(MagicMock()) == {}

    def test_feature_kwargs_accepted_by_agent_config(self):
        from missy.agent.runtime import AgentConfig

        cfg = MagicMock()
        cfg.features = FeaturesConfig(
            model_routing_enabled=True,
            prompt_patch_proposals_enabled=True,
            condenser_pipeline_enabled=True,
            semantic_memory_enabled=True,
        )
        agent_cfg = AgentConfig(**_agent_feature_kwargs(cfg))
        assert agent_cfg.model_routing_enabled is True
        assert agent_cfg.condenser_pipeline_enabled is True

    def test_load_config_parses_features_and_retention(self, tmp_path):
        from missy.config.settings import load_config

        p = tmp_path / "config.yaml"
        p.write_text(
            "features:\n  model_routing_enabled: true\n"
            "retention:\n  captures_days: 5\n  memory_days: 90\n"
        )
        cfg = load_config(str(p))
        assert cfg.features.model_routing_enabled is True
        assert cfg.retention.captures_days == 5
        assert cfg.retention.memory_days == 90

    def test_parse_retention_defaults_and_negatives(self):
        r = _parse_retention({"checkpoints_days": -3})
        assert r.checkpoints_days == 0
        assert r.inbound_attachments_days == 0  # content pruners are opt-in
        assert r.enabled is True
