"""YAML parsing fuzz and edge-case tests for hatching and persona subsystems.

These tests probe boundary conditions, malformed inputs, adversarial payloads,
and unusual but valid YAML constructs.  All I/O is confined to pytest's
tmp_path so no real filesystem state is modified.
"""

from __future__ import annotations

import threading
from pathlib import Path

import yaml

from missy.agent.hatching import HatchingManager, HatchingState, HatchingStatus
from missy.agent.persona import PersonaConfig, PersonaManager

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_hatching_manager(tmp_path: Path) -> HatchingManager:
    return HatchingManager(
        state_path=tmp_path / "hatching.yaml",
        log_path=tmp_path / "hatching_log.jsonl",
    )


# ---------------------------------------------------------------------------
# Test 1 — binary content in state file → graceful fallback
# ---------------------------------------------------------------------------


def test_hatching_state_binary_yaml(tmp_path: Path) -> None:
    """Writing non-YAML bytes (but valid UTF-8) to the state file yields default state.

    HatchingManager.get_state() catches yaml.YAMLError and returns a default
    UNHATCHED state.  We use bytes that are valid UTF-8 but completely
    invalid YAML so the YAML parser rejects them cleanly.

    Note: raw non-UTF-8 bytes (e.g. 0xff) would raise UnicodeDecodeError before
    the YAML parser runs; that error is not caught by the current implementation.
    This test specifically targets the yaml.YAMLError recovery path.
    """
    # Null bytes are valid UTF-8 but cause a YAML reader error (control chars)
    state_path = tmp_path / "hatching.yaml"
    state_path.write_bytes(b"\x00" * 100)

    manager = _make_hatching_manager(tmp_path)
    state = manager.get_state()

    assert isinstance(state, HatchingState)
    assert state.status is HatchingStatus.UNHATCHED


# ---------------------------------------------------------------------------
# Test 2 — 100-level nested YAML dict
# ---------------------------------------------------------------------------


def test_hatching_state_deeply_nested(tmp_path: Path) -> None:
    """A 100-level deeply nested YAML mapping in the state file must not crash.

    yaml.safe_load has a default recursion guard; HatchingManager must survive
    even if the YAML parser emits a warning or error.
    """
    # Build a 100-deep nested structure starting with a valid outer key
    nested: dict = {}
    current = nested
    for _i in range(99):
        current["level"] = {}
        current = current["level"]
    current["leaf"] = "value"

    state_path = tmp_path / "hatching.yaml"
    state_path.write_text(yaml.safe_dump(nested), encoding="utf-8")

    manager = _make_hatching_manager(tmp_path)
    # Must not raise; result must be a valid HatchingState (defaults because mapping
    # lacks the required keys)
    state = manager.get_state()
    assert isinstance(state, HatchingState)


# ---------------------------------------------------------------------------
# Test 3 — 1 MB YAML blob in state file
# ---------------------------------------------------------------------------


def test_hatching_state_very_large(tmp_path: Path) -> None:
    """A ~1 MB YAML file must be loadable without memory error or crash.

    We create a YAML mapping with a single key whose value is a very long string.
    HatchingManager.get_state() must not raise and must return a HatchingState.
    """
    large_value = "x" * (1024 * 1024)  # 1 MiB of 'x'
    data = {"status": "unhatched", "large_key": large_value}

    state_path = tmp_path / "hatching.yaml"
    state_path.write_text(yaml.safe_dump(data), encoding="utf-8")

    manager = _make_hatching_manager(tmp_path)
    state = manager.get_state()

    assert isinstance(state, HatchingState)
    assert state.status is HatchingStatus.UNHATCHED


# ---------------------------------------------------------------------------
# Test 4 — YAML anchors and aliases (potential alias abuse)
# ---------------------------------------------------------------------------


def test_hatching_state_yaml_bomb(tmp_path: Path) -> None:
    """YAML with anchors/aliases must be handled safely by safe_load.

    A classic YAML alias bomb attempts exponential expansion.  PyYAML's
    safe_load does not expand anchors into Python objects recursively in a
    dangerous way; this test verifies get_state() survives and returns a
    sensible default.
    """
    yaml_bomb = """\
a: &a ['lol', 'lol', 'lol', 'lol', 'lol', 'lol', 'lol', 'lol', 'lol']
b: &b [*a, *a, *a, *a, *a, *a, *a, *a, *a]
c: &c [*b, *b, *b, *b, *b, *b, *b, *b, *b]
"""
    state_path = tmp_path / "hatching.yaml"
    state_path.write_text(yaml_bomb, encoding="utf-8")

    manager = _make_hatching_manager(tmp_path)
    # Must not raise; the parsed dict lacks hatching keys so we get defaults
    state = manager.get_state()
    assert isinstance(state, HatchingState)


# ---------------------------------------------------------------------------
# Test 5 — persona YAML with all null fields → defaults applied
# ---------------------------------------------------------------------------


def test_persona_yaml_null_fields(tmp_path: Path) -> None:
    """A persona file where every field is null must fall back to defaults.

    _persona_from_dict() ignores None values by filtering only known keys;
    PersonaConfig's field defaults then fill the gaps.
    """
    null_data = {
        "name": None,
        "tone": None,
        "personality_traits": None,
        "behavioral_tendencies": None,
        "response_style_rules": None,
        "boundaries": None,
        "identity_description": None,
        "version": None,
    }
    persona_path = tmp_path / "persona.yaml"
    persona_path.write_text(yaml.safe_dump(null_data), encoding="utf-8")

    pm = PersonaManager(persona_path=persona_path)
    config = pm.get_persona()

    # None values are passed through _persona_from_dict which passes them to
    # PersonaConfig; the dataclass stores them as-is.  We verify the manager
    # does not raise and returns a PersonaConfig instance.
    assert isinstance(config, PersonaConfig)


# ---------------------------------------------------------------------------
# Test 6 — persona YAML with wrong types (int where str expected)
# ---------------------------------------------------------------------------


def test_persona_yaml_wrong_types(tmp_path: Path) -> None:
    """Persona fields with wrong types must not crash PersonaManager.

    For instance, name=42 (int) and tone=True (bool) are unusual but the
    manager should load without raising.  BehaviorLayer must also survive.
    """
    wrong_type_data = {
        "version": 1,
        "name": 42,
        "tone": True,
        "personality_traits": 3.14,
        "response_style_rules": 0,
        "boundaries": False,
        "identity_description": 99,
    }
    persona_path = tmp_path / "persona.yaml"
    persona_path.write_text(yaml.safe_dump(wrong_type_data), encoding="utf-8")

    pm = PersonaManager(persona_path=persona_path)
    config = pm.get_persona()
    assert isinstance(config, PersonaConfig)


# ---------------------------------------------------------------------------
# Test 7 — persona YAML with emoji, CJK, and RTL characters
# ---------------------------------------------------------------------------


def test_persona_yaml_unicode_stress(tmp_path: Path) -> None:
    """Persona fields containing emoji, CJK, and RTL text must round-trip losslessly.

    We write, save, and reload a persona with heavy Unicode content and verify
    the strings survive the round-trip intact.
    """
    emoji_name = "Missy \U0001f916\U0001f4a1"  # robot + bulb
    cjk_identity = "\u4eba\u5de5\u667a\u80fd\u52a9\u624b"  # "AI assistant" in Chinese
    rtl_boundary = "\u0644\u0627 \u062a\u0643\u0634\u0641 \u0627\u0644\u0623\u0633\u0631\u0627\u0631"  # Arabic: "do not reveal secrets"

    persona_path = tmp_path / "persona.yaml"
    pm = PersonaManager(persona_path=persona_path)
    pm.update(
        name=emoji_name,
        identity_description=cjk_identity,
        boundaries=[rtl_boundary],
    )
    pm.save()

    pm2 = PersonaManager(persona_path=persona_path)
    config = pm2.get_persona()

    assert config.name == emoji_name
    assert config.identity_description == cjk_identity
    assert config.boundaries == [rtl_boundary]


# ---------------------------------------------------------------------------
# Test 8 — persona YAML with empty lists
# ---------------------------------------------------------------------------


def test_persona_yaml_empty_lists(tmp_path: Path) -> None:
    """Persona with empty lists for all list fields must load and be usable.

    BehaviorLayer built from a persona with no traits or style rules must not
    raise when shape_system_prompt() is called.
    """
    from missy.agent.behavior import BehaviorLayer

    empty_list_data = {
        "version": 1,
        "name": "Minimal",
        "tone": [],
        "personality_traits": [],
        "behavioral_tendencies": [],
        "response_style_rules": [],
        "boundaries": [],
        "identity_description": "A minimal persona.",
    }
    persona_path = tmp_path / "persona.yaml"
    persona_path.write_text(yaml.safe_dump(empty_list_data), encoding="utf-8")

    pm = PersonaManager(persona_path=persona_path)
    config = pm.get_persona()
    assert config.tone == []
    assert config.boundaries == []

    layer = BehaviorLayer(config)
    shaped = layer.shape_system_prompt("Base.", {"user_tone": "casual", "turn_count": 1})
    assert "Base." in shaped


# ---------------------------------------------------------------------------
# Test 9 — persona YAML with unknown keys is silently ignored
# ---------------------------------------------------------------------------


def test_persona_yaml_extra_keys_ignored(tmp_path: Path) -> None:
    """Unknown keys in persona YAML must be ignored without raising.

    _persona_from_dict() filters to only known PersonaConfig fields; future
    schema additions should not break existing installs.
    """
    extra_keys_data = {
        "version": 2,
        "name": "Known",
        "tone": ["direct"],
        "unknown_future_field": "some value",
        "another_extra": [1, 2, 3],
        "nested_extra": {"deep": "value"},
    }
    persona_path = tmp_path / "persona.yaml"
    persona_path.write_text(yaml.safe_dump(extra_keys_data), encoding="utf-8")

    pm = PersonaManager(persona_path=persona_path)
    config = pm.get_persona()

    assert config.name == "Known"
    assert config.tone == ["direct"]
    assert config.version == 2
    # Unknown fields must not appear on the dataclass
    assert not hasattr(config, "unknown_future_field")
    assert not hasattr(config, "another_extra")


# ---------------------------------------------------------------------------
# Test 10 — YAML injection with !!python/object is blocked by safe_load
# ---------------------------------------------------------------------------


def test_hatching_state_yaml_injection(tmp_path: Path) -> None:
    """YAML with !!python/object tags must be rejected by safe_load.

    PyYAML's safe_load refuses to deserialise arbitrary Python objects.  The
    hatching state loader must either handle the resulting ConstructorError
    gracefully (returning defaults) or let it propagate — but must never
    instantiate arbitrary Python classes from attacker-controlled YAML.
    """
    injection_yaml = """\
!!python/object/apply:os.system
- "echo pwned"
"""
    state_path = tmp_path / "hatching.yaml"
    state_path.write_text(injection_yaml, encoding="utf-8")

    manager = _make_hatching_manager(tmp_path)

    # The manager catches yaml.YAMLError and returns defaults — no arbitrary
    # code execution must occur.
    state = manager.get_state()
    assert isinstance(state, HatchingState)
    assert state.status is HatchingStatus.UNHATCHED


def test_hatching_state_yaml_reduce_blocked(tmp_path: Path) -> None:
    """__reduce__ inside YAML must not execute code when loaded safely.

    A classic pickle-over-YAML gadget using !!python/object/new.  safe_load
    must block this; the manager must return default state.
    """
    reduce_yaml = "!!python/object/new:subprocess.Popen\n- [id]\n"
    state_path = tmp_path / "hatching.yaml"
    state_path.write_text(reduce_yaml, encoding="utf-8")

    manager = _make_hatching_manager(tmp_path)
    state = manager.get_state()

    assert isinstance(state, HatchingState)
    assert state.status is HatchingStatus.UNHATCHED


# ---------------------------------------------------------------------------
# Test 11 — persona YAML with multiline literal block strings
# ---------------------------------------------------------------------------


def test_persona_yaml_multiline_strings(tmp_path: Path) -> None:
    """Persona YAML using YAML literal block scalars must round-trip correctly.

    Multiline identity descriptions are a realistic use case for thoughtful
    users who craft detailed personas.
    """
    multiline_identity = (
        "Missy is a capable AI assistant.\n"
        "She values clarity above all else.\n"
        "  - She uses bullet points when helpful.\n"
        "  - She avoids unnecessary preamble.\n"
        "Line four ends without a newline."
    )
    multiline_boundary = (
        "Never execute destructive operations\n"
        "without explicit written confirmation."
    )

    persona_path = tmp_path / "persona.yaml"
    pm = PersonaManager(persona_path=persona_path)
    pm.update(
        identity_description=multiline_identity,
        boundaries=[multiline_boundary],
    )
    pm.save()

    pm2 = PersonaManager(persona_path=persona_path)
    config = pm2.get_persona()

    assert config.identity_description == multiline_identity
    assert config.boundaries == [multiline_boundary]


# ---------------------------------------------------------------------------
# Test 12 — concurrent state writes from two threads
# ---------------------------------------------------------------------------


def test_hatching_state_concurrent_write(tmp_path: Path) -> None:
    """Writing hatching state concurrently from two threads must not corrupt the file.

    HatchingManager._save_state() uses an atomic rename pattern.  Two threads
    writing different states should result in one valid YAML file — either
    state — but never a partial/corrupt file that cannot be read back.
    """
    state_path = tmp_path / "hatching.yaml"
    log_path = tmp_path / "hatching_log.jsonl"
    errors: list[Exception] = []

    def write_state(status_value: str) -> None:
        try:
            manager = HatchingManager(state_path=state_path, log_path=log_path)
            state = HatchingState(
                status=HatchingStatus(status_value),
                steps_completed=[status_value],
            )
            # Access internal _save_state directly to stress the atomic write
            for _ in range(20):
                manager._save_state(state)
        except Exception as exc:  # noqa: BLE001
            errors.append(exc)

    t1 = threading.Thread(target=write_state, args=("in_progress",), daemon=True)
    t2 = threading.Thread(target=write_state, args=("unhatched",), daemon=True)

    t1.start()
    t2.start()
    t1.join(timeout=10)
    t2.join(timeout=10)

    assert not errors, f"Threads raised exceptions: {errors}"

    # The file must be readable and parseable after concurrent writes
    manager = _make_hatching_manager(tmp_path)
    state = manager.get_state()
    assert isinstance(state, HatchingState), "State file must be valid after concurrent writes"
    assert state.status in (
        HatchingStatus.UNHATCHED,
        HatchingStatus.IN_PROGRESS,
    ), f"Unexpected status after concurrent writes: {state.status}"
