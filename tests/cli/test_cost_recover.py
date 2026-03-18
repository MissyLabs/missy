"""Tests for missy cost and missy recover CLI commands."""

from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner

from missy.cli.main import cli


@pytest.fixture
def runner():
    return CliRunner()


@pytest.fixture
def mock_config():
    cfg = MagicMock()
    cfg.max_spend_usd = 5.0
    cfg.audit_log_path = "/tmp/test_audit.jsonl"
    cfg.providers = {}
    return cfg


class TestCostCommand:
    def test_cost_shows_budget(self, runner, mock_config):
        with patch("missy.cli.main._load_subsystems", return_value=mock_config):
            result = runner.invoke(cli, ["cost"])
            assert result.exit_code == 0
            assert "$5.00" in result.output

    def test_cost_unlimited_budget(self, runner, mock_config):
        mock_config.max_spend_usd = 0.0
        with patch("missy.cli.main._load_subsystems", return_value=mock_config):
            result = runner.invoke(cli, ["cost"])
            assert result.exit_code == 0
            assert "unlimited" in result.output

    def test_cost_with_session(self, runner, mock_config, tmp_path):
        mock_store = MagicMock()
        mock_store.get_session_turns.return_value = [MagicMock(), MagicMock()]
        mock_store.get_session_costs.return_value = [
            {"model": "claude-sonnet-4", "cost_usd": 0.0045, "prompt_tokens": 500, "completion_tokens": 200},
            {"model": "claude-haiku-4", "cost_usd": 0.0003, "prompt_tokens": 100, "completion_tokens": 50},
        ]

        with (
            patch("missy.cli.main._load_subsystems", return_value=mock_config),
            patch("missy.memory.sqlite_store.SQLiteMemoryStore", return_value=mock_store),
        ):
            result = runner.invoke(cli, ["cost", "--session", "test-session"])
            assert result.exit_code == 0
            assert "test-session" in result.output
            assert "2" in result.output  # 2 API calls

    def test_cost_with_session_no_data(self, runner, mock_config, tmp_path):
        mock_store = MagicMock()
        mock_store.get_session_turns.return_value = []
        mock_store.get_session_costs.return_value = []

        with (
            patch("missy.cli.main._load_subsystems", return_value=mock_config),
            patch("missy.memory.sqlite_store.SQLiteMemoryStore", return_value=mock_store),
        ):
            result = runner.invoke(cli, ["cost", "--session", "no-such-session"])
            assert result.exit_code == 0
            assert "No cost records" in result.output

    def test_cost_help_exits_zero(self, runner):
        result = runner.invoke(cli, ["cost", "--help"])
        assert result.exit_code == 0


class TestRecoverCommand:
    def test_recover_no_checkpoints(self, runner):
        with patch("missy.agent.checkpoint.scan_for_recovery", return_value=[]):
            result = runner.invoke(cli, ["recover"])
            assert result.exit_code == 0
            assert "No incomplete checkpoints" in result.output

    def test_recover_shows_checkpoints(self, runner):
        from missy.agent.checkpoint import RecoveryResult

        results = [
            RecoveryResult(
                checkpoint_id="aaaabbbb-1234-5678-9abc-def012345678",
                session_id="sess-001",
                prompt="Summarize this document",
                action="resume",
                iteration=3,
            ),
            RecoveryResult(
                checkpoint_id="ccccdddd-1234-5678-9abc-def012345678",
                session_id="sess-002",
                prompt="Run analysis",
                action="restart",
                iteration=1,
            ),
        ]
        with patch("missy.agent.checkpoint.scan_for_recovery", return_value=results):
            result = runner.invoke(cli, ["recover"])
            assert result.exit_code == 0
            assert "2 checkpoint(s)" in result.output

    def test_recover_abandon_all(self, runner):
        mock_cm = MagicMock()
        mock_cm.abandon_old.return_value = 3
        with patch("missy.agent.checkpoint.CheckpointManager", return_value=mock_cm):
            result = runner.invoke(cli, ["recover", "--abandon-all"])
            assert result.exit_code == 0
            assert "Abandoned 3" in result.output
            mock_cm.abandon_old.assert_called_once_with(max_age_seconds=0)

    def test_recover_help_exits_zero(self, runner):
        result = runner.invoke(cli, ["recover", "--help"])
        assert result.exit_code == 0
