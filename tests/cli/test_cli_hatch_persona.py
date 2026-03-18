"""Tests for missy hatch and missy persona CLI commands."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner

from missy.agent.hatching import HatchingState, HatchingStatus
from missy.agent.persona import PersonaConfig
from missy.cli.main import cli


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def runner() -> CliRunner:
    return CliRunner(mix_stderr=False)


def _make_hatching_manager(
    *,
    is_hatched: bool = False,
    run_status: HatchingStatus = HatchingStatus.HATCHED,
    completed_at: str = "2026-03-18T12:00:00+00:00",
    steps_completed: list[str] | None = None,
    persona_generated: bool = True,
    memory_seeded: bool = True,
    error: str | None = None,
) -> MagicMock:
    """Return a configured HatchingManager mock."""
    mgr = MagicMock()
    mgr.is_hatched.return_value = is_hatched

    state = HatchingState(
        status=run_status,
        completed_at=completed_at,
        steps_completed=steps_completed or ["validate_environment", "finalize"],
        persona_generated=persona_generated,
        memory_seeded=memory_seeded,
        error=error,
    )
    mgr.get_state.return_value = state
    mgr.run_hatching.return_value = state
    return mgr


def _make_persona_manager(*, version: int = 1) -> MagicMock:
    """Return a configured PersonaManager mock with a default PersonaConfig."""
    mgr = MagicMock()
    mgr.get_persona.return_value = PersonaConfig(version=version)
    mgr.version = version
    return mgr


# ---------------------------------------------------------------------------
# missy hatch
# ---------------------------------------------------------------------------


class TestHatchCommand:
    def test_hatch_already_hatched(self, runner: CliRunner) -> None:
        """When is_hatched() returns True, prints already-hatched message and exits 0."""
        mock_mgr = _make_hatching_manager(is_hatched=True)

        with patch("missy.agent.hatching.HatchingManager", return_value=mock_mgr):
            result = runner.invoke(cli, ["hatch"])

        assert result.exit_code == 0
        assert "already hatched" in result.output.lower()

    def test_hatch_already_hatched_shows_completed_at(self, runner: CliRunner) -> None:
        """Already-hatched output includes the completion timestamp."""
        mock_mgr = _make_hatching_manager(
            is_hatched=True, completed_at="2026-03-18T12:00:00+00:00"
        )

        with patch("missy.agent.hatching.HatchingManager", return_value=mock_mgr):
            result = runner.invoke(cli, ["hatch"])

        assert result.exit_code == 0
        assert "2026-03-18T12:00:00+00:00" in result.output

    def test_hatch_success(self, runner: CliRunner) -> None:
        """When run_hatching() returns HATCHED status, prints success panel and exits 0."""
        mock_mgr = _make_hatching_manager(
            is_hatched=False,
            run_status=HatchingStatus.HATCHED,
            persona_generated=True,
            memory_seeded=True,
        )

        with patch("missy.agent.hatching.HatchingManager", return_value=mock_mgr):
            result = runner.invoke(cli, ["hatch"])

        assert result.exit_code == 0
        # The success panel message includes "hatched successfully"
        assert "hatched successfully" in result.output.lower()

    def test_hatch_success_shows_step_counts(self, runner: CliRunner) -> None:
        """Success output includes steps completed count."""
        steps = ["validate_environment", "initialize_config", "finalize"]
        mock_mgr = _make_hatching_manager(
            is_hatched=False,
            run_status=HatchingStatus.HATCHED,
            steps_completed=steps,
        )

        with patch("missy.agent.hatching.HatchingManager", return_value=mock_mgr):
            result = runner.invoke(cli, ["hatch"])

        assert result.exit_code == 0
        assert str(len(steps)) in result.output

    def test_hatch_failure(self, runner: CliRunner) -> None:
        """When run_hatching() returns FAILED status, prints error message and exits 1."""
        mock_mgr = _make_hatching_manager(
            is_hatched=False,
            run_status=HatchingStatus.FAILED,
            error="Step 'verify_providers' failed: RuntimeError: no key",
        )

        with patch("missy.agent.hatching.HatchingManager", return_value=mock_mgr):
            result = runner.invoke(cli, ["hatch"])

        assert result.exit_code == 1
        # _print_error writes to err_console (stderr); check there
        all_output = result.output + result.stderr
        assert "Step 'verify_providers' failed" in all_output

    def test_hatch_failure_includes_hint(self, runner: CliRunner) -> None:
        """Failure output includes the hint about --non-interactive."""
        mock_mgr = _make_hatching_manager(
            is_hatched=False,
            run_status=HatchingStatus.FAILED,
            error="something went wrong",
        )

        with patch("missy.agent.hatching.HatchingManager", return_value=mock_mgr):
            result = runner.invoke(cli, ["hatch"])

        assert result.exit_code == 1
        # _print_error writes to err_console (stderr); check there
        all_output = result.output + result.stderr
        assert "--non-interactive" in all_output

    def test_hatch_non_interactive_flag(self, runner: CliRunner) -> None:
        """--non-interactive passes interactive=False to run_hatching."""
        mock_mgr = _make_hatching_manager(
            is_hatched=False,
            run_status=HatchingStatus.HATCHED,
        )

        with patch("missy.agent.hatching.HatchingManager", return_value=mock_mgr):
            result = runner.invoke(cli, ["hatch", "--non-interactive"])

        assert result.exit_code == 0
        mock_mgr.run_hatching.assert_called_once_with(interactive=False)

    def test_hatch_interactive_by_default(self, runner: CliRunner) -> None:
        """Without --non-interactive flag, run_hatching is called with interactive=True."""
        mock_mgr = _make_hatching_manager(
            is_hatched=False,
            run_status=HatchingStatus.HATCHED,
        )

        with patch("missy.agent.hatching.HatchingManager", return_value=mock_mgr):
            result = runner.invoke(cli, ["hatch"])

        assert result.exit_code == 0
        mock_mgr.run_hatching.assert_called_once_with(interactive=True)


# ---------------------------------------------------------------------------
# missy persona show
# ---------------------------------------------------------------------------


class TestPersonaShowCommand:
    def test_persona_show_exit_code(self, runner: CliRunner) -> None:
        """persona show exits 0."""
        mock_mgr = _make_persona_manager()

        with patch("missy.agent.persona.PersonaManager", return_value=mock_mgr):
            result = runner.invoke(cli, ["persona", "show"])

        assert result.exit_code == 0

    def test_persona_show_default_name(self, runner: CliRunner) -> None:
        """persona show displays the default persona name 'Missy'."""
        mock_mgr = _make_persona_manager()

        with patch("missy.agent.persona.PersonaManager", return_value=mock_mgr):
            result = runner.invoke(cli, ["persona", "show"])

        assert result.exit_code == 0
        assert "Missy" in result.output

    def test_persona_show_displays_version(self, runner: CliRunner) -> None:
        """persona show includes the version number in the table title."""
        mock_mgr = _make_persona_manager(version=3)

        with patch("missy.agent.persona.PersonaManager", return_value=mock_mgr):
            result = runner.invoke(cli, ["persona", "show"])

        assert result.exit_code == 0
        assert "v3" in result.output

    def test_persona_show_displays_tone_fields(self, runner: CliRunner) -> None:
        """persona show renders tone values from the persona config."""
        mock_mgr = _make_persona_manager()
        # Default PersonaConfig has tone ["helpful", "direct", "technical"]
        with patch("missy.agent.persona.PersonaManager", return_value=mock_mgr):
            result = runner.invoke(cli, ["persona", "show"])

        assert result.exit_code == 0
        assert "helpful" in result.output

    def test_persona_show_displays_identity(self, runner: CliRunner) -> None:
        """persona show renders the identity description."""
        mock_mgr = _make_persona_manager()

        with patch("missy.agent.persona.PersonaManager", return_value=mock_mgr):
            result = runner.invoke(cli, ["persona", "show"])

        assert result.exit_code == 0
        # Default identity mentions "security-first"
        assert "security-first" in result.output

    def test_persona_show_table_fields(self, runner: CliRunner) -> None:
        """persona show renders the expected field labels in the table."""
        mock_mgr = _make_persona_manager()

        with patch("missy.agent.persona.PersonaManager", return_value=mock_mgr):
            result = runner.invoke(cli, ["persona", "show"])

        assert result.exit_code == 0
        for field_label in ("Name", "Tone", "Personality", "Identity"):
            assert field_label in result.output


# ---------------------------------------------------------------------------
# missy persona edit
# ---------------------------------------------------------------------------


class TestPersonaEditCommand:
    def test_persona_edit_name(self, runner: CliRunner) -> None:
        """--name updates the persona name field."""
        mock_mgr = _make_persona_manager(version=1)

        with patch("missy.agent.persona.PersonaManager", return_value=mock_mgr):
            result = runner.invoke(cli, ["persona", "edit", "--name", "NewName"])

        assert result.exit_code == 0
        mock_mgr.update.assert_called_once()
        call_kwargs = mock_mgr.update.call_args[1]
        assert call_kwargs.get("name") == "NewName"
        mock_mgr.save.assert_called_once()

    def test_persona_edit_name_shown_in_output(self, runner: CliRunner) -> None:
        """--name value appears in the confirmation output."""
        mock_mgr = _make_persona_manager(version=2)

        with patch("missy.agent.persona.PersonaManager", return_value=mock_mgr):
            result = runner.invoke(cli, ["persona", "edit", "--name", "Aria"])

        assert result.exit_code == 0
        assert "Aria" in result.output

    def test_persona_edit_tone(self, runner: CliRunner) -> None:
        """--tone parses comma-separated values into a list."""
        mock_mgr = _make_persona_manager()

        with patch("missy.agent.persona.PersonaManager", return_value=mock_mgr):
            result = runner.invoke(cli, ["persona", "edit", "--tone", "friendly,casual"])

        assert result.exit_code == 0
        call_kwargs = mock_mgr.update.call_args[1]
        assert call_kwargs.get("tone") == ["friendly", "casual"]

    def test_persona_edit_tone_strips_whitespace(self, runner: CliRunner) -> None:
        """--tone strips whitespace from each value."""
        mock_mgr = _make_persona_manager()

        with patch("missy.agent.persona.PersonaManager", return_value=mock_mgr):
            result = runner.invoke(cli, ["persona", "edit", "--tone", " direct , formal "])

        assert result.exit_code == 0
        call_kwargs = mock_mgr.update.call_args[1]
        assert call_kwargs.get("tone") == ["direct", "formal"]

    def test_persona_edit_identity(self, runner: CliRunner) -> None:
        """--identity updates the identity_description field."""
        mock_mgr = _make_persona_manager()
        new_desc = "A helpful Linux assistant focused on security."

        with patch("missy.agent.persona.PersonaManager", return_value=mock_mgr):
            result = runner.invoke(cli, ["persona", "edit", "--identity", new_desc])

        assert result.exit_code == 0
        call_kwargs = mock_mgr.update.call_args[1]
        assert call_kwargs.get("identity_description") == new_desc

    def test_persona_edit_multiple_fields(self, runner: CliRunner) -> None:
        """Multiple flags in one invocation pass all fields to update()."""
        mock_mgr = _make_persona_manager()

        with patch("missy.agent.persona.PersonaManager", return_value=mock_mgr):
            result = runner.invoke(
                cli,
                ["persona", "edit", "--name", "Nova", "--tone", "calm,clear"],
            )

        assert result.exit_code == 0
        call_kwargs = mock_mgr.update.call_args[1]
        assert call_kwargs.get("name") == "Nova"
        assert call_kwargs.get("tone") == ["calm", "clear"]

    def test_persona_edit_no_args(self, runner: CliRunner) -> None:
        """With no flags, prints 'No changes specified' and does not call update or save."""
        mock_mgr = _make_persona_manager()

        with patch("missy.agent.persona.PersonaManager", return_value=mock_mgr):
            result = runner.invoke(cli, ["persona", "edit"])

        assert result.exit_code == 0
        assert "No changes specified" in result.output
        mock_mgr.update.assert_not_called()
        mock_mgr.save.assert_not_called()

    def test_persona_edit_shows_updated_message(self, runner: CliRunner) -> None:
        """Successful edit prints 'Persona updated' with version number."""
        mock_mgr = _make_persona_manager(version=4)

        with patch("missy.agent.persona.PersonaManager", return_value=mock_mgr):
            result = runner.invoke(cli, ["persona", "edit", "--name", "Aria"])

        assert result.exit_code == 0
        assert "Persona updated" in result.output
        assert "v4" in result.output


# ---------------------------------------------------------------------------
# missy persona reset
# ---------------------------------------------------------------------------


class TestPersonaResetCommand:
    def test_persona_reset_calls_reset(self, runner: CliRunner) -> None:
        """persona reset calls reset() on PersonaManager."""
        mock_mgr = _make_persona_manager(version=5)

        with patch("missy.agent.persona.PersonaManager", return_value=mock_mgr):
            result = runner.invoke(cli, ["persona", "reset"])

        assert result.exit_code == 0
        mock_mgr.reset.assert_called_once()

    def test_persona_reset_shows_success_message(self, runner: CliRunner) -> None:
        """persona reset prints a success message with the version."""
        mock_mgr = _make_persona_manager(version=6)

        with patch("missy.agent.persona.PersonaManager", return_value=mock_mgr):
            result = runner.invoke(cli, ["persona", "reset"])

        assert result.exit_code == 0
        assert "reset" in result.output.lower()

    def test_persona_reset_shows_version(self, runner: CliRunner) -> None:
        """persona reset includes the version number in confirmation output."""
        mock_mgr = _make_persona_manager(version=7)

        with patch("missy.agent.persona.PersonaManager", return_value=mock_mgr):
            result = runner.invoke(cli, ["persona", "reset"])

        assert result.exit_code == 0
        assert "v7" in result.output
