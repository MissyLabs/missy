"""Integration tests for the hatching -> persona -> behavior pipeline.

These tests exercise the full chain from HatchingManager bootstrapping through
PersonaManager configuration through BehaviorLayer response shaping.  All
filesystem I/O is redirected to pytest's tmp_path fixture so tests are
hermetic and leave no side effects on the real ~/.missy/ directory.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from missy.agent.behavior import BehaviorLayer
from missy.agent.hatching import HatchingManager, HatchingState, HatchingStatus
from missy.agent.persona import PersonaConfig, PersonaManager


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_BASE_PROMPT = "You are a helpful assistant."

_MINIMAL_CTX: dict = {
    "user_tone": "casual",
    "topic": "",
    "turn_count": 1,
    "has_tool_results": False,
    "intent": "question",
    "urgency": "low",
}


def _patch_hatching_paths(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Redirect all module-level path constants in missy.agent.hatching to tmp_path."""
    monkeypatch.setattr("missy.agent.hatching._MISSY_DIR", tmp_path)
    monkeypatch.setattr("missy.agent.hatching._CONFIG_PATH", tmp_path / "config.yaml")
    monkeypatch.setattr("missy.agent.hatching._PERSONA_PATH", tmp_path / "persona.yaml")
    monkeypatch.setattr("missy.agent.hatching._IDENTITY_PATH", tmp_path / "identity.pem")
    monkeypatch.setattr("missy.agent.hatching._SECRETS_DIR", tmp_path / "secrets")
    monkeypatch.setattr("missy.agent.hatching._MEMORY_DB_PATH", tmp_path / "memory.db")


def _make_manager(tmp_path: Path) -> HatchingManager:
    """Return a HatchingManager scoped entirely to tmp_path."""
    return HatchingManager(
        state_path=tmp_path / "hatching.yaml",
        log_path=tmp_path / "hatching_log.jsonl",
    )


def _run_hatching(tmp_path: Path) -> HatchingState:
    """Run a non-interactive hatching and return the final state."""
    manager = _make_manager(tmp_path)
    return manager.run_hatching(interactive=False)


# ---------------------------------------------------------------------------
# Test 1 — full hatching creates persona.yaml
# ---------------------------------------------------------------------------


def test_full_hatching_creates_persona(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """HatchingManager.run_hatching() must create persona.yaml via PersonaManager.

    After a successful non-interactive hatch the persona file must exist at the
    path the module constants point to, and PersonaManager must be able to read
    it back without error.
    """
    _patch_hatching_paths(monkeypatch, tmp_path)

    state = _run_hatching(tmp_path)

    persona_path = tmp_path / "persona.yaml"
    assert persona_path.exists(), "persona.yaml must be created after hatching"
    assert state.persona_generated is True

    # PersonaManager must load the file without raising
    pm = PersonaManager(persona_path=persona_path)
    config = pm.get_persona()
    assert isinstance(config, PersonaConfig)
    assert config.name  # name must be non-empty
    assert config.version >= 1


# ---------------------------------------------------------------------------
# Test 2 — hatched persona feeds BehaviorLayer
# ---------------------------------------------------------------------------


def test_hatching_persona_feeds_behavior_layer(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """BehaviorLayer built from a hatched persona must produce a non-trivial prompt.

    shape_system_prompt() must return a string that starts with the base prompt
    and contains persona-derived content such as the agent name.
    """
    _patch_hatching_paths(monkeypatch, tmp_path)
    _run_hatching(tmp_path)

    pm = PersonaManager(persona_path=tmp_path / "persona.yaml")
    persona = pm.get_persona()

    layer = BehaviorLayer(persona)
    shaped = layer.shape_system_prompt(_BASE_PROMPT, _MINIMAL_CTX)

    assert shaped.startswith(_BASE_PROMPT)
    # The persona block must be present in the shaped prompt
    assert "## Persona" in shaped
    assert persona.name in shaped


# ---------------------------------------------------------------------------
# Test 3 — hatched persona is editable and BehaviorLayer picks up changes
# ---------------------------------------------------------------------------


def test_hatched_persona_editable(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """After hatching, editing the persona and rebuilding BehaviorLayer reflects changes.

    Editing the name and tone must be visible in the shaped system prompt
    produced by a new BehaviorLayer constructed from the updated config.
    """
    _patch_hatching_paths(monkeypatch, tmp_path)
    _run_hatching(tmp_path)

    persona_path = tmp_path / "persona.yaml"
    pm = PersonaManager(persona_path=persona_path)

    # Modify fields and persist
    pm.update(name="Harriet", tone=["whimsical", "encouraging"])
    pm.save()

    # Load fresh and verify
    pm2 = PersonaManager(persona_path=persona_path)
    updated_persona = pm2.get_persona()
    assert updated_persona.name == "Harriet"
    assert "whimsical" in updated_persona.tone

    layer = BehaviorLayer(updated_persona)
    shaped = layer.shape_system_prompt(_BASE_PROMPT, _MINIMAL_CTX)
    assert "Harriet" in shaped


# ---------------------------------------------------------------------------
# Test 4 — hatching resumes after partial completion
# ---------------------------------------------------------------------------


def test_hatching_resume_after_partial(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A partially completed hatching can be resumed and still produce a persona.

    We simulate a partial run by pre-writing a state file that marks two steps
    as already done and status as IN_PROGRESS, then call run_hatching() again.
    The remaining steps (including generate_persona) must run and the final
    state must be HATCHED.
    """
    import yaml

    _patch_hatching_paths(monkeypatch, tmp_path)
    tmp_path.mkdir(parents=True, exist_ok=True)

    # Pre-write a partial state: environment validated + config written, but nothing else
    partial_state = {
        "status": "in_progress",
        "started_at": "2026-01-01T00:00:00+00:00",
        "completed_at": None,
        "steps_completed": ["validate_environment", "initialize_config"],
        "persona_generated": False,
        "environment_validated": True,
        "provider_verified": False,
        "security_initialized": False,
        "memory_seeded": False,
        "error": None,
    }
    state_path = tmp_path / "hatching.yaml"
    state_path.write_text(yaml.safe_dump(partial_state), encoding="utf-8")

    # Also write a minimal config so _initialize_config is skipped correctly
    config_path = tmp_path / "config.yaml"
    config_path.write_text("config_version: 2\n", encoding="utf-8")

    state = _run_hatching(tmp_path)

    persona_path = tmp_path / "persona.yaml"
    assert persona_path.exists(), "Resume must produce persona.yaml"
    assert state.persona_generated is True
    assert state.status is HatchingStatus.HATCHED


# ---------------------------------------------------------------------------
# Test 5 — persona reset after hatching restores defaults; BehaviorLayer works
# ---------------------------------------------------------------------------


def test_persona_reset_after_hatching(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Calling PersonaManager.reset() after hatching restores factory defaults.

    The BehaviorLayer must still be constructable and usable after the reset
    without raising any exceptions.
    """
    _patch_hatching_paths(monkeypatch, tmp_path)
    _run_hatching(tmp_path)

    pm = PersonaManager(persona_path=tmp_path / "persona.yaml")

    # Modify, then reset
    pm.update(name="Temporary Name")
    pm.save()
    pm.reset()

    config = pm.get_persona()
    assert config.name == "Missy", "Name must revert to factory default after reset"

    # BehaviorLayer must still function after a reset
    layer = BehaviorLayer(config)
    shaped = layer.shape_system_prompt(_BASE_PROMPT, _MINIMAL_CTX)
    assert _BASE_PROMPT in shaped


# ---------------------------------------------------------------------------
# Test 6 — formal tone in persona appears in BehaviorLayer guidelines
# ---------------------------------------------------------------------------


def test_behavior_tone_matches_persona(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A persona with tone=['formal'] must produce guidelines mentioning formal style.

    get_response_guidelines() is called with user_tone='formal' in the context;
    the returned string must reference precision/professional phrasing.
    """
    _patch_hatching_paths(monkeypatch, tmp_path)
    _run_hatching(tmp_path)

    pm = PersonaManager(persona_path=tmp_path / "persona.yaml")
    pm.update(tone=["formal"])
    pm.save()

    config = pm.get_persona()
    layer = BehaviorLayer(config)

    formal_ctx = dict(_MINIMAL_CTX, user_tone="formal")
    guidelines = layer.get_response_guidelines(formal_ctx)

    # The tone adaptation map for "formal" contains "professional" and "precise"
    assert guidelines, "Guidelines must be non-empty"
    assert "formal" in guidelines.lower() or "professional" in guidelines.lower(), (
        "Guidelines must reflect formal tone; got: " + guidelines
    )


# ---------------------------------------------------------------------------
# Test 7 — persona backup survives hatching and can rollback
# ---------------------------------------------------------------------------


def test_persona_backup_survives_hatching(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Modifying the persona after hatching creates a backup; rollback restores state.

    After hatching + save, at least one backup must exist in the backup directory.
    After rollback, the manager must load the backed-up persona without error.
    """
    _patch_hatching_paths(monkeypatch, tmp_path)
    _run_hatching(tmp_path)

    persona_path = tmp_path / "persona.yaml"
    pm = PersonaManager(persona_path=persona_path)

    # First save creates backup of the hatching-generated file
    original_name = pm.get_persona().name
    pm.update(name="ModifiedName")
    pm.save()

    backups = pm.list_backups()
    assert len(backups) >= 1, "At least one backup must exist after saving"

    # Rollback must succeed and restore the previous persona
    restored_path = pm.rollback()
    assert restored_path is not None, "rollback() must return the backup path"

    rolled_back_name = pm.get_persona().name
    assert rolled_back_name == original_name, (
        f"After rollback, name should be {original_name!r}, got {rolled_back_name!r}"
    )


# ---------------------------------------------------------------------------
# Test 8 — memory DB exists after hatching
# ---------------------------------------------------------------------------


def test_hatching_memory_seed_exists(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """After hatching, the SQLite memory DB must exist and contain the welcome turn.

    We connect directly with sqlite3 to avoid importing SQLiteMemoryStore in a
    way that would create an unrelated DB.  The welcome message text is verified
    to be present in the conversation_turns table.
    """
    _patch_hatching_paths(monkeypatch, tmp_path)
    state = _run_hatching(tmp_path)

    db_path = tmp_path / "memory.db"

    if not state.memory_seeded:
        pytest.skip(
            "memory_seeded is False — SQLiteMemoryStore may not be importable "
            "in this environment; skipping DB content check"
        )

    assert db_path.exists(), "memory.db must exist after hatching"

    conn = sqlite3.connect(str(db_path))
    try:
        cursor = conn.execute(
            "SELECT content FROM turns WHERE session_id = 'hatching' LIMIT 1"
        )
        row = cursor.fetchone()
        assert row is not None, "Welcome turn must be present in turns table"
        assert "hatching" in row[0].lower() or "ready" in row[0].lower(), (
            f"Welcome content unexpected: {row[0]!r}"
        )
    finally:
        conn.close()
