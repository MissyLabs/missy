"""Regression tests for session-8 bug fixes in behavior.py and persona.py.

Covers:
- BehaviorLayer.get_response_guidelines(None) must not raise AttributeError.
- BehaviorLayer.get_response_guidelines({}) must return a valid string.
- BehaviorLayer.should_be_concise(None) must not raise; must return bool.
- BehaviorLayer.should_be_concise({}) must return False (all defaults are below threshold).
- PersonaManager._prune_backups() must not propagate OSError when unlink fails;
  a warning must be logged instead.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from unittest.mock import patch

import pytest
import yaml

from missy.agent.behavior import BehaviorLayer
from missy.agent.persona import PersonaConfig, PersonaManager


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def default_layer() -> BehaviorLayer:
    """BehaviorLayer with default persona (no arguments)."""
    return BehaviorLayer()


@pytest.fixture
def layer_with_persona() -> BehaviorLayer:
    """BehaviorLayer with an explicit, fully-populated PersonaConfig."""
    persona = PersonaConfig(
        name="Missy",
        tone=["warm", "direct"],
        identity_description="A helpful test assistant.",
        personality_traits=["curious", "pragmatic"],
        behavioral_tendencies=["asks clarifying questions"],
        response_style_rules=["be concise"],
        boundaries=["never expose secrets"],
    )
    return BehaviorLayer(persona)


@pytest.fixture
def persona_manager(tmp_path: Path) -> PersonaManager:
    """PersonaManager pointed at a temp directory."""
    persona_yaml = tmp_path / "persona.yaml"
    # Write a minimal valid persona file so the manager has something to back up.
    persona_yaml.write_text(
        yaml.dump({"version": 1, "name": "Missy"}),
        encoding="utf-8",
    )
    return PersonaManager(persona_path=persona_yaml)


# ---------------------------------------------------------------------------
# BehaviorLayer.get_response_guidelines — None and empty-dict inputs
# ---------------------------------------------------------------------------


class TestGetResponseGuidelinesNullSafety:
    """get_response_guidelines must handle None and empty dict without raising."""

    def test_none_context_does_not_raise(self, default_layer: BehaviorLayer) -> None:
        """Passing None must not raise AttributeError or any other exception."""
        # The fix: get_response_guidelines uses `ctx = context or {}` to guard
        # against None being passed where a dict is expected.
        result = default_layer.get_response_guidelines(None)  # type: ignore[arg-type]
        assert result is not None

    def test_none_context_returns_string(self, default_layer: BehaviorLayer) -> None:
        result = default_layer.get_response_guidelines(None)  # type: ignore[arg-type]
        assert isinstance(result, str)

    def test_none_context_with_persona_does_not_raise(
        self, layer_with_persona: BehaviorLayer
    ) -> None:
        """None context must also be safe when a persona is present."""
        result = layer_with_persona.get_response_guidelines(None)  # type: ignore[arg-type]
        assert isinstance(result, str)

    def test_empty_dict_returns_string(self, default_layer: BehaviorLayer) -> None:
        """An empty context dict should produce a valid (possibly empty) string."""
        result = default_layer.get_response_guidelines({})
        assert isinstance(result, str)

    def test_empty_dict_with_persona_includes_tendencies(
        self, layer_with_persona: BehaviorLayer
    ) -> None:
        """With an empty context, persona behavioral tendencies still appear."""
        result = layer_with_persona.get_response_guidelines({})
        # The fixture persona has a behavioral_tendency and a response_style_rule;
        # at least one of them should appear in the guidelines.
        assert "asks clarifying questions" in result or "be concise" in result

    def test_empty_dict_no_vision_mode_key_error(
        self, default_layer: BehaviorLayer
    ) -> None:
        """Missing 'vision_mode' key must not cause a KeyError."""
        # get_response_guidelines uses context.get("vision_mode", "") internally,
        # but after the None guard is applied the local `ctx` dict is used —
        # passing the raw `context` argument to .get() was the original bug.
        result = default_layer.get_response_guidelines({})
        assert isinstance(result, str)


# ---------------------------------------------------------------------------
# BehaviorLayer.should_be_concise — None and empty-dict inputs
# ---------------------------------------------------------------------------


class TestShouldBeConciseNullSafety:
    """should_be_concise must handle None context without raising."""

    def test_none_context_does_not_raise(self, default_layer: BehaviorLayer) -> None:
        """None must not propagate as an AttributeError inside the method."""
        result = default_layer.should_be_concise(None)
        assert result is not None

    def test_none_context_returns_bool(self, default_layer: BehaviorLayer) -> None:
        result = default_layer.should_be_concise(None)
        assert isinstance(result, bool)

    def test_none_context_returns_false(self, default_layer: BehaviorLayer) -> None:
        """With no context, all defaults (turn_count=0, urgency='low') mean
        conciseness is NOT triggered."""
        result = default_layer.should_be_concise(None)
        assert result is False

    def test_empty_dict_returns_bool(self, default_layer: BehaviorLayer) -> None:
        result = default_layer.should_be_concise({})
        assert isinstance(result, bool)

    def test_empty_dict_returns_false(self, default_layer: BehaviorLayer) -> None:
        """All defaults: turn_count=0 (<10), user_tone='' (not 'brief'),
        urgency='low' (not 'high') → False."""
        result = default_layer.should_be_concise({})
        assert result is False

    def test_high_turn_count_triggers_concise(self, default_layer: BehaviorLayer) -> None:
        """Sanity check: 10+ turns should return True."""
        result = default_layer.should_be_concise({"turn_count": 10})
        assert result is True

    def test_brief_tone_triggers_concise(self, default_layer: BehaviorLayer) -> None:
        result = default_layer.should_be_concise({"user_tone": "brief"})
        assert result is True

    def test_high_urgency_triggers_concise(self, default_layer: BehaviorLayer) -> None:
        result = default_layer.should_be_concise({"urgency": "high"})
        assert result is True


# ---------------------------------------------------------------------------
# PersonaManager._prune_backups — unlink failure is logged, not re-raised
# ---------------------------------------------------------------------------


class TestPruneBackupsUnlinkFailure:
    """_prune_backups must swallow OSError from unlink() and log a warning."""

    def _create_backup_files(self, backup_dir: Path, count: int) -> list[Path]:
        """Create *count* synthetic backup files in *backup_dir*.

        Files are stamped with distinct, monotonically increasing mtimes so
        that ``list_backups()`` (which sorts by mtime) returns them in the
        same order they were created — oldest first.
        """
        backup_dir.mkdir(parents=True, exist_ok=True)
        files: list[Path] = []
        for i in range(count):
            p = backup_dir / f"persona.yaml.20260101_00000{i}"
            p.write_text(f"version: {i + 1}\nname: Missy\n", encoding="utf-8")
            # Assign an explicit mtime so sort order is deterministic even
            # when files are created within the same clock tick.
            mtime = float(i * 1000)
            os.utime(str(p), (mtime, mtime))
            files.append(p)
        return files

    def test_unlink_failure_does_not_raise(
        self, persona_manager: PersonaManager, tmp_path: Path
    ) -> None:
        """When the oldest backup's unlink() raises OSError, _prune_backups
        must not propagate the exception."""
        backup_dir = persona_manager.backup_dir
        # Create _MAX_BACKUPS + 1 backups so pruning is triggered.
        self._create_backup_files(backup_dir, PersonaManager._MAX_BACKUPS + 1)
        # Identify the oldest via the same sort logic the manager uses.
        oldest = persona_manager.list_backups()[0]

        original_unlink = Path.unlink

        def failing_unlink(self_path: Path, missing_ok: bool = False) -> None:
            if self_path == oldest:
                raise OSError("Simulated permission denied")
            original_unlink(self_path, missing_ok=missing_ok)

        with patch.object(Path, "unlink", failing_unlink):
            # Must complete without raising.
            persona_manager._prune_backups()

    def test_unlink_failure_logs_warning(
        self,
        persona_manager: PersonaManager,
        tmp_path: Path,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """A warning must be emitted when backup deletion fails."""
        backup_dir = persona_manager.backup_dir
        self._create_backup_files(backup_dir, PersonaManager._MAX_BACKUPS + 1)
        oldest = persona_manager.list_backups()[0]

        original_unlink = Path.unlink

        def failing_unlink(self_path: Path, missing_ok: bool = False) -> None:
            if self_path == oldest:
                raise OSError("Simulated permission denied")
            original_unlink(self_path, missing_ok=missing_ok)

        # Use the root logger level so the warning propagates to caplog regardless
        # of whether the missy.agent.persona logger has been initialised yet.
        with caplog.at_level(logging.WARNING):
            with patch.object(Path, "unlink", failing_unlink):
                persona_manager._prune_backups()

        warning_messages = [r.message for r in caplog.records if r.levelno == logging.WARNING]
        assert any(
            "backup" in msg.lower() or oldest.name in msg
            for msg in warning_messages
        ), (
            f"Expected a warning mentioning the failed backup. Got: {warning_messages}"
        )

    def test_unlink_failure_leaves_other_backups_intact(
        self,
        persona_manager: PersonaManager,
        tmp_path: Path,
    ) -> None:
        """Even when the oldest backup's unlink() fails, the remaining
        backups beyond the limit should still be pruned where possible."""
        backup_dir = persona_manager.backup_dir
        # Create _MAX_BACKUPS + 2 backups; first one fails, second one should
        # be pruned successfully.
        self._create_backup_files(backup_dir, PersonaManager._MAX_BACKUPS + 2)
        sorted_backups = persona_manager.list_backups()
        oldest = sorted_backups[0]
        second_oldest = sorted_backups[1]

        original_unlink = Path.unlink

        def failing_unlink(self_path: Path, missing_ok: bool = False) -> None:
            if self_path == oldest:
                raise OSError("Simulated permission denied")
            original_unlink(self_path, missing_ok=missing_ok)

        with patch.object(Path, "unlink", failing_unlink):
            persona_manager._prune_backups()

        # The oldest survived (unlink failed), but the second-oldest should
        # have been removed.
        assert oldest.exists(), "Oldest backup should still exist after failed unlink"
        assert not second_oldest.exists(), (
            "Second-oldest backup should have been removed successfully"
        )

    def test_no_pruning_needed_does_not_call_unlink(
        self,
        persona_manager: PersonaManager,
    ) -> None:
        """When backup count is within the limit, no deletion should occur."""
        backup_dir = persona_manager.backup_dir
        # Create exactly _MAX_BACKUPS files — no pruning needed.
        backups = self._create_backup_files(backup_dir, PersonaManager._MAX_BACKUPS)

        with patch.object(Path, "unlink") as mock_unlink:
            persona_manager._prune_backups()
            mock_unlink.assert_not_called()

        # All backup files should still be present.
        for p in backups:
            assert p.exists()
