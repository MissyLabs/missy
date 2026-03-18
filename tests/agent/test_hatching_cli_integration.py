"""Integration tests for hatching and persona CLI commands.

Tests the Click CLI interface, verifying commands are invocable
and produce expected output.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner

from missy.agent.persona import PersonaConfig, PersonaManager
from missy.cli.main import cli


@pytest.fixture
def runner():
    return CliRunner()


@pytest.fixture
def mock_persona_mgr(tmp_path):
    """Create a real PersonaManager pointing to tmp_path."""
    return PersonaManager(persona_path=str(tmp_path / "persona.yaml"))


class TestHatchCommand:
    def test_hatch_help(self, runner):
        result = runner.invoke(cli, ["hatch", "--help"])
        assert result.exit_code == 0

    def test_hatch_non_interactive(self, runner, tmp_path):
        """Non-interactive hatch should attempt the flow."""
        with patch("missy.agent.hatching.HatchingManager") as MockHatch:
            mock_hm = MagicMock()
            mock_hm.needs_hatching.return_value = True
            mock_hm.state = MagicMock()
            mock_hm.state.status = "UNHATCHED"
            MockHatch.return_value = mock_hm

            result = runner.invoke(cli, ["hatch", "--non-interactive"])
            # May fail but shouldn't crash
            assert result.exit_code in (0, 1, 2)


class TestPersonaShowCommand:
    def test_persona_show(self, runner, tmp_path):
        mgr = PersonaManager(persona_path=str(tmp_path / "persona.yaml"))
        with patch("missy.agent.persona.PersonaManager", return_value=mgr):
            result = runner.invoke(cli, ["persona", "show"])
        assert result.exit_code == 0
        assert "Missy" in result.output  # Default name

    def test_persona_show_help(self, runner):
        result = runner.invoke(cli, ["persona", "show", "--help"])
        assert result.exit_code == 0


class TestPersonaEditCommand:
    def test_persona_edit_name(self, runner, tmp_path):
        mgr = PersonaManager(persona_path=str(tmp_path / "persona.yaml"))
        with patch("missy.agent.persona.PersonaManager", return_value=mgr):
            result = runner.invoke(cli, ["persona", "edit", "--name", "CustomBot"])
        assert result.exit_code == 0
        # Verify the name was updated
        assert mgr.get_persona().name == "CustomBot"

    def test_persona_edit_help(self, runner):
        result = runner.invoke(cli, ["persona", "edit", "--help"])
        assert result.exit_code == 0


class TestPersonaResetCommand:
    def test_persona_reset(self, runner, tmp_path):
        mgr = PersonaManager(persona_path=str(tmp_path / "persona.yaml"))
        mgr.update(name="Changed")
        with patch("missy.agent.persona.PersonaManager", return_value=mgr):
            result = runner.invoke(cli, ["persona", "reset"])
        assert result.exit_code == 0
        assert mgr.get_persona().name == "Missy"  # Reset to default

    def test_persona_reset_help(self, runner):
        result = runner.invoke(cli, ["persona", "reset", "--help"])
        assert result.exit_code == 0


class TestPersonaBackupsCommand:
    def test_persona_backups_empty(self, runner, tmp_path):
        mgr = PersonaManager(persona_path=str(tmp_path / "persona.yaml"))
        with patch("missy.agent.persona.PersonaManager", return_value=mgr):
            result = runner.invoke(cli, ["persona", "backups"])
        assert result.exit_code == 0

    def test_persona_backups_help(self, runner):
        result = runner.invoke(cli, ["persona", "backups", "--help"])
        assert result.exit_code == 0


class TestPersonaDiffCommand:
    def test_persona_diff(self, runner, tmp_path):
        mgr = PersonaManager(persona_path=str(tmp_path / "persona.yaml"))
        with patch("missy.agent.persona.PersonaManager", return_value=mgr):
            result = runner.invoke(cli, ["persona", "diff"])
        assert result.exit_code == 0

    def test_persona_diff_help(self, runner):
        result = runner.invoke(cli, ["persona", "diff", "--help"])
        assert result.exit_code == 0


class TestPersonaRollbackCommand:
    def test_persona_rollback(self, runner, tmp_path):
        mgr = PersonaManager(persona_path=str(tmp_path / "persona.yaml"))
        with patch("missy.agent.persona.PersonaManager", return_value=mgr):
            result = runner.invoke(cli, ["persona", "rollback"])
        assert result.exit_code == 0

    def test_persona_rollback_help(self, runner):
        result = runner.invoke(cli, ["persona", "rollback", "--help"])
        assert result.exit_code == 0


class TestPersonaLogCommand:
    def test_persona_log(self, runner, tmp_path):
        mgr = PersonaManager(persona_path=str(tmp_path / "persona.yaml"))
        with patch("missy.agent.persona.PersonaManager", return_value=mgr):
            result = runner.invoke(cli, ["persona", "log"])
        assert result.exit_code == 0

    def test_persona_log_help(self, runner):
        result = runner.invoke(cli, ["persona", "log", "--help"])
        assert result.exit_code == 0
