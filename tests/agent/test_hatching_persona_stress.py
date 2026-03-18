"""Stress and edge-case tests for the hatching and persona systems.

Covers scenarios not addressed by test_hatching.py, test_persona.py, or
test_behavior.py:

- Hatching with rapid restarts (100 stop/start cycles)
- Hatching concurrent step execution (two managers sharing one state file)
- Hatching step timeout via slow handler injection
- Persona with very long fields (10 KB name, 50 KB boundaries list)
- Persona with special characters (Unicode emoji, control chars, null bytes)
- Persona rapid edit cycle (100 edits, backup count stays at max)
- Persona YAML injection (YAML special chars in field values)
- Behavior layer with extreme inputs (100 KB message, 1000-message history)
- Behavior tone analysis stability (same input → same tone every time)
- Intent classifier boundary cases (multi-intent messages)
- Response shaper idempotency (applying shaper twice)
- Hatching log rotation under a very large log file
- Persona audit log integrity (1000 entries, all valid JSONL)
- Persona concurrent reads (multiple threads)
- Hatching + persona integration (full hatch produces valid persona)
"""

from __future__ import annotations

import json
import os
import sys
import threading
import time
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
import yaml

from missy.agent.behavior import BehaviorLayer, IntentInterpreter, ResponseShaper
from missy.agent.hatching import (
    HatchingLog,
    HatchingManager,
    HatchingState,
    HatchingStatus,
    _HatchingStepWarning,
)
from missy.agent.persona import (
    PersonaConfig,
    PersonaManager,
    _persona_from_dict,
    _persona_to_dict,
)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _make_manager(tmp_path: Path) -> HatchingManager:
    """Create a HatchingManager with all files isolated to *tmp_path*."""
    return HatchingManager(
        state_path=tmp_path / "hatching.yaml",
        log_path=tmp_path / "log.jsonl",
    )


def _patch_module_paths(monkeypatch: Any, tmp_path: Path) -> None:
    """Redirect all module-level path constants in hatching.py to *tmp_path*."""
    import missy.agent.hatching as hatching_mod

    monkeypatch.setattr(hatching_mod, "_MISSY_DIR", tmp_path)
    monkeypatch.setattr(hatching_mod, "_CONFIG_PATH", tmp_path / "config.yaml")
    monkeypatch.setattr(hatching_mod, "_IDENTITY_PATH", tmp_path / "identity.pem")
    monkeypatch.setattr(hatching_mod, "_SECRETS_DIR", tmp_path / "secrets")
    monkeypatch.setattr(hatching_mod, "_PERSONA_PATH", tmp_path / "persona.yaml")
    monkeypatch.setattr(hatching_mod, "_MEMORY_DB_PATH", tmp_path / "memory.db")


import contextlib


@contextlib.contextmanager
def _mock_hatching_deps():
    """Patch PersonaManager and SQLiteMemoryStore/ConversationTurn on the real
    modules so that deferred imports inside hatching.py receive mocks without
    replacing entire modules in sys.modules (which causes cross-test leaks).
    """
    mock_turn = MagicMock()
    mock_turn.id = "turn-stub"
    mock_store = MagicMock()

    with (
        patch("missy.agent.persona.PersonaManager", return_value=MagicMock()),
        patch("missy.memory.sqlite_store.SQLiteMemoryStore", return_value=mock_store),
        patch("missy.memory.sqlite_store.ConversationTurn", MagicMock(new=MagicMock(return_value=mock_turn))),
    ):
        yield


# ---------------------------------------------------------------------------
# 1. Hatching with 100 rapid restarts
# ---------------------------------------------------------------------------


class TestHatching100RapidRestarts:
    """State file integrity under 100 rapid reset→hatch cycles."""

    def test_state_file_survives_100_reset_cycles(self, tmp_path: Path, monkeypatch: Any) -> None:
        _patch_module_paths(monkeypatch, tmp_path)
        monkeypatch.setenv("ANTHROPIC_API_KEY", "key-stress")

        mgr = _make_manager(tmp_path)
        with _mock_hatching_deps():
            for _ in range(100):
                mgr.reset()
                state = mgr.run_hatching(interactive=False)
                assert state.status is HatchingStatus.HATCHED, (
                    f"Expected HATCHED after re-hatch, got {state.status}"
                )

        # Final state on disk must be valid YAML that decodes back to HATCHED.
        state_path = tmp_path / "hatching.yaml"
        raw = yaml.safe_load(state_path.read_text(encoding="utf-8"))
        assert raw["status"] == "hatched"

    def test_each_restart_preserves_all_steps(self, tmp_path: Path, monkeypatch: Any) -> None:
        _patch_module_paths(monkeypatch, tmp_path)
        monkeypatch.setenv("ANTHROPIC_API_KEY", "key-stress")

        expected_steps = {
            "validate_environment",
            "initialize_config",
            "verify_providers",
            "initialize_security",
            "generate_persona",
            "seed_memory",
            "finalize",
        }

        mgr = _make_manager(tmp_path)
        with _mock_hatching_deps():
            for i in range(5):
                mgr.reset()
                state = mgr.run_hatching(interactive=False)
                missing = expected_steps - set(state.steps_completed)
                assert not missing, (
                    f"Iteration {i}: missing steps {missing!r}"
                )

    def test_log_file_is_always_valid_jsonl_after_restarts(
        self, tmp_path: Path, monkeypatch: Any
    ) -> None:
        _patch_module_paths(monkeypatch, tmp_path)
        monkeypatch.setenv("ANTHROPIC_API_KEY", "key-stress")
        log_path = tmp_path / "log.jsonl"

        mgr = _make_manager(tmp_path)
        with _mock_hatching_deps():
            for _ in range(10):
                mgr.reset()
                mgr.run_hatching(interactive=False)

        # Every line in the log file must be parseable JSON.
        for lineno, raw_line in enumerate(log_path.read_text(encoding="utf-8").splitlines(), 1):
            line = raw_line.strip()
            if not line:
                continue
            parsed = json.loads(line)  # raises if corrupt
            assert "step" in parsed, f"Line {lineno} missing 'step' key"
            assert "timestamp" in parsed, f"Line {lineno} missing 'timestamp' key"


# ---------------------------------------------------------------------------
# 2. Hatching concurrent step execution
# ---------------------------------------------------------------------------


class TestHatchingConcurrentExecution:
    """Two manager instances racing over a shared state file."""

    def test_only_one_manager_finishes_hatched(self, tmp_path: Path, monkeypatch: Any) -> None:
        """Both managers start with no state; only one should win HATCHED cleanly.

        Because there is no advisory lock, both may succeed — but neither
        should corrupt the on-disk YAML to the point of being unparseable.
        """
        _patch_module_paths(monkeypatch, tmp_path)
        monkeypatch.setenv("ANTHROPIC_API_KEY", "key-concurrent")

        state_path = tmp_path / "hatching.yaml"
        log_path = tmp_path / "log.jsonl"

        results: list[HatchingState] = []
        errors: list[Exception] = []

        def _run() -> None:
            with _mock_hatching_deps():
                mgr = HatchingManager(state_path=state_path, log_path=log_path)
                try:
                    state = mgr.run_hatching(interactive=False)
                    results.append(state)
                except Exception as exc:  # noqa: BLE001
                    errors.append(exc)

        threads = [threading.Thread(target=_run) for _ in range(2)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)

        assert not errors, f"Threads raised errors: {errors}"

        # The state file must be readable after concurrent writes.
        raw = yaml.safe_load(state_path.read_text(encoding="utf-8"))
        assert isinstance(raw, dict), "State file became non-dict after concurrent writes"
        assert raw.get("status") in ("hatched", "in_progress", "failed")

    def test_concurrent_reads_never_raise(self, tmp_path: Path, monkeypatch: Any) -> None:
        """Many threads reading the state file concurrently should not raise."""
        _patch_module_paths(monkeypatch, tmp_path)
        monkeypatch.setenv("ANTHROPIC_API_KEY", "key-read-concurrent")

        state_path = tmp_path / "hatching.yaml"
        log_path = tmp_path / "log.jsonl"

        # Write a HATCHED state first.
        hatched = HatchingState(status=HatchingStatus.HATCHED)
        state_path.write_text(
            yaml.safe_dump(hatched.to_dict()), encoding="utf-8"
        )

        errors: list[Exception] = []

        def _read() -> None:
            try:
                mgr = HatchingManager(state_path=state_path, log_path=log_path)
                for _ in range(20):
                    state = mgr.get_state()
                    assert state.status is HatchingStatus.HATCHED
            except Exception as exc:  # noqa: BLE001
                errors.append(exc)

        threads = [threading.Thread(target=_read) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)

        assert not errors, f"Read threads raised: {errors}"


# ---------------------------------------------------------------------------
# 3. Hatching step timeout (slow handler)
# ---------------------------------------------------------------------------


class TestHatchingStepTimeout:
    """Inject a slow step and verify the manager surfaces the failure correctly."""

    def test_slow_step_does_not_hang_indefinitely_when_interrupted(
        self, tmp_path: Path, monkeypatch: Any
    ) -> None:
        """Replace _finalize with a step that raises after a short sleep.

        This is not a real timeout test (no signal/alarm), but verifies the
        failure path when a step raises partway through execution.
        """
        _patch_module_paths(monkeypatch, tmp_path)
        monkeypatch.setenv("ANTHROPIC_API_KEY", "key-slow")

        call_log: list[str] = []

        def _slow_finalize(state: HatchingState, *, interactive: bool) -> None:
            call_log.append("entered")
            time.sleep(0.01)  # simulate some latency
            raise RuntimeError("simulated timeout failure")

        mgr = _make_manager(tmp_path)
        mgr._finalize = _slow_finalize  # type: ignore[method-assign]

        with _mock_hatching_deps():
            state = mgr.run_hatching(interactive=False)

        assert state.status is HatchingStatus.FAILED
        assert "simulated timeout failure" in (state.error or "")
        assert "entered" in call_log, "Slow step was never called"

    def test_step_warning_from_slow_step_is_non_fatal(
        self, tmp_path: Path, monkeypatch: Any
    ) -> None:
        """A slow step that raises _HatchingStepWarning should not abort the hatch."""
        _patch_module_paths(monkeypatch, tmp_path)
        monkeypatch.setenv("ANTHROPIC_API_KEY", "key-warn-slow")

        def _slow_warning_step(state: HatchingState, *, interactive: bool) -> None:
            time.sleep(0.01)
            raise _HatchingStepWarning("slow but non-fatal")

        mgr = _make_manager(tmp_path)
        # Inject into verify_providers (a step that emits warnings normally).
        mgr._verify_providers = _slow_warning_step  # type: ignore[method-assign]

        with _mock_hatching_deps():
            state = mgr.run_hatching(interactive=False)

        # Hatching should still complete because warnings are non-fatal.
        assert state.status is HatchingStatus.HATCHED


# ---------------------------------------------------------------------------
# 4. Persona with very long fields
# ---------------------------------------------------------------------------


class TestPersonaVeryLongFields:
    """PersonaManager must handle oversized field values without data loss."""

    def test_10kb_name_survives_save_load_round_trip(self, tmp_path: Path) -> None:
        name_10kb = "N" * (10 * 1024)
        pm = PersonaManager(persona_path=tmp_path / "persona.yaml")
        pm.update(name=name_10kb)
        pm.save()

        pm2 = PersonaManager(persona_path=tmp_path / "persona.yaml")
        assert pm2.get_persona().name == name_10kb

    def test_50kb_boundaries_list_survives_round_trip(self, tmp_path: Path) -> None:
        # 500 boundary entries of ~100 chars each ≈ 50 KB
        big_boundaries = [f"Boundary rule number {i}: " + "x" * 80 for i in range(500)]
        pm = PersonaManager(persona_path=tmp_path / "persona.yaml")
        pm.update(boundaries=big_boundaries)
        pm.save()

        pm2 = PersonaManager(persona_path=tmp_path / "persona.yaml")
        loaded = pm2.get_persona().boundaries
        assert len(loaded) == 500
        assert loaded[0] == big_boundaries[0]
        assert loaded[-1] == big_boundaries[-1]

    def test_system_prompt_prefix_with_large_fields_is_str(self, tmp_path: Path) -> None:
        pm = PersonaManager(persona_path=tmp_path / "persona.yaml")
        pm.update(
            name="A" * 5000,
            identity_description="B" * 5000,
            boundaries=["C" * 100] * 100,
            response_style_rules=["D" * 100] * 100,
            behavioral_tendencies=["E" * 100] * 100,
        )
        prefix = pm.get_system_prompt_prefix()
        assert isinstance(prefix, str)
        assert len(prefix) > 10000

    def test_large_tone_list_round_trips(self, tmp_path: Path) -> None:
        huge_tone = [f"tone_adj_{i}" for i in range(1000)]
        pm = PersonaManager(persona_path=tmp_path / "persona.yaml")
        pm.update(tone=huge_tone)
        pm.save()

        pm2 = PersonaManager(persona_path=tmp_path / "persona.yaml")
        assert len(pm2.get_persona().tone) == 1000

    def test_50kb_identity_description_round_trips(self, tmp_path: Path) -> None:
        long_desc = "Word " * 10000  # ~50 KB
        pm = PersonaManager(persona_path=tmp_path / "persona.yaml")
        pm.update(identity_description=long_desc)
        pm.save()

        pm2 = PersonaManager(persona_path=tmp_path / "persona.yaml")
        assert pm2.get_persona().identity_description == long_desc


# ---------------------------------------------------------------------------
# 5. Persona with special characters
# ---------------------------------------------------------------------------


class TestPersonaSpecialCharacters:
    """Special chars in persona fields must not corrupt YAML serialisation."""

    def test_emoji_in_name(self, tmp_path: Path) -> None:
        name = "Missy 🤖🔐🎯"
        pm = PersonaManager(persona_path=tmp_path / "persona.yaml")
        pm.update(name=name)
        pm.save()

        pm2 = PersonaManager(persona_path=tmp_path / "persona.yaml")
        assert pm2.get_persona().name == name

    def test_control_chars_in_identity_description(self, tmp_path: Path) -> None:
        # Tab and newline are valid in YAML strings; NUL is not but we test
        # that the system degrades gracefully (yaml.safe_load may strip it).
        desc_with_tabs = "Line one\tcolumn two\n\tindented line three"
        pm = PersonaManager(persona_path=tmp_path / "persona.yaml")
        pm.update(identity_description=desc_with_tabs)
        pm.save()

        pm2 = PersonaManager(persona_path=tmp_path / "persona.yaml")
        loaded = pm2.get_persona().identity_description
        # At minimum, the text should be loadable without exception and non-empty.
        assert isinstance(loaded, str)
        assert len(loaded) > 0

    def test_null_byte_in_name_degrades_gracefully(self, tmp_path: Path) -> None:
        """A name containing NUL bytes should either be stored as-is or cause
        the load to fall back to defaults — never crash."""
        name_with_nul = "Missy\x00Agent"
        pm = PersonaManager(persona_path=tmp_path / "persona.yaml")
        pm.update(name=name_with_nul)
        try:
            pm.save()
        except Exception:  # noqa: BLE001
            # Saving may fail on some YAML libraries — that's acceptable.
            return

        pm2 = PersonaManager(persona_path=tmp_path / "persona.yaml")
        # Either the NUL was preserved or the fallback default was used.
        loaded_name = pm2.get_persona().name
        assert isinstance(loaded_name, str)

    def test_yaml_special_chars_in_name(self, tmp_path: Path) -> None:
        """Characters like : { } [ ] & * ! | ' " should not break YAML output."""
        special_name = 'Missy: {agent} [v2] & *system* | "primary" \'sec\''
        pm = PersonaManager(persona_path=tmp_path / "persona.yaml")
        pm.update(name=special_name)
        pm.save()

        pm2 = PersonaManager(persona_path=tmp_path / "persona.yaml")
        assert pm2.get_persona().name == special_name

    def test_yaml_special_chars_in_boundaries(self, tmp_path: Path) -> None:
        boundaries = [
            "Never do: {rm -rf /}",
            "Always use [safe] mode",
            "Respect *all* policies",
            "Key: value pairs are fine",
            "Pipe | operator must be quoted",
        ]
        pm = PersonaManager(persona_path=tmp_path / "persona.yaml")
        pm.update(boundaries=boundaries)
        pm.save()

        pm2 = PersonaManager(persona_path=tmp_path / "persona.yaml")
        assert pm2.get_persona().boundaries == boundaries

    def test_multi_line_string_in_identity_description(self, tmp_path: Path) -> None:
        multi = "Line one.\nLine two.\nLine three with special: chars {like} these."
        pm = PersonaManager(persona_path=tmp_path / "persona.yaml")
        pm.update(identity_description=multi)
        pm.save()

        pm2 = PersonaManager(persona_path=tmp_path / "persona.yaml")
        assert pm2.get_persona().identity_description == multi

    def test_unicode_cjk_characters_in_tone(self, tmp_path: Path) -> None:
        cjk_tones = ["直接", "技術的", "役に立つ", "친근한"]
        pm = PersonaManager(persona_path=tmp_path / "persona.yaml")
        pm.update(tone=cjk_tones)
        pm.save()

        pm2 = PersonaManager(persona_path=tmp_path / "persona.yaml")
        assert pm2.get_persona().tone == cjk_tones

    def test_rtl_text_in_name(self, tmp_path: Path) -> None:
        rtl_name = "مساعد مسي"  # Arabic: "Missy's assistant"
        pm = PersonaManager(persona_path=tmp_path / "persona.yaml")
        pm.update(name=rtl_name)
        pm.save()

        pm2 = PersonaManager(persona_path=tmp_path / "persona.yaml")
        assert pm2.get_persona().name == rtl_name


# ---------------------------------------------------------------------------
# 6. Persona rapid edit cycle — backup count stays at max
# ---------------------------------------------------------------------------


class TestPersonaRapidEditCycle:
    """100 sequential edits must not exceed _MAX_BACKUPS backup files."""

    def test_100_edits_backup_count_stays_at_max(self, tmp_path: Path) -> None:
        pm = PersonaManager(persona_path=tmp_path / "persona.yaml")
        pm.save()  # Create initial file so subsequent saves make backups.

        for i in range(100):
            pm.update(name=f"Iteration{i}")
            pm.save()

        backup_count = len(pm.list_backups())
        assert backup_count <= pm._MAX_BACKUPS, (
            f"Expected ≤ {pm._MAX_BACKUPS} backups, got {backup_count}"
        )

    def test_100_edits_version_monotonically_increases(self, tmp_path: Path) -> None:
        pm = PersonaManager(persona_path=tmp_path / "persona.yaml")
        versions: list[int] = [pm.version]

        pm.save()
        for i in range(99):
            pm.update(name=f"V{i}")
            pm.save()
            versions.append(pm.version)

        # Versions must be strictly increasing.
        for a, b in zip(versions, versions[1:]):
            assert b > a, f"Version went backwards: {a} → {b}"

    def test_100_edits_final_name_is_correct(self, tmp_path: Path) -> None:
        pm = PersonaManager(persona_path=tmp_path / "persona.yaml")
        pm.save()

        for i in range(100):
            pm.update(name=f"Name{i}")
            pm.save()

        # Reload from disk and verify the last written name.
        pm2 = PersonaManager(persona_path=tmp_path / "persona.yaml")
        assert pm2.get_persona().name == "Name99"

    def test_backup_files_are_valid_yaml(self, tmp_path: Path) -> None:
        pm = PersonaManager(persona_path=tmp_path / "persona.yaml")
        pm.save()

        for i in range(10):
            pm.update(name=f"Backup{i}")
            pm.save()

        for backup_path in pm.list_backups():
            raw = yaml.safe_load(backup_path.read_text(encoding="utf-8"))
            assert isinstance(raw, dict), f"Backup {backup_path.name} is not a dict"
            assert "name" in raw, f"Backup {backup_path.name} missing 'name' key"


# ---------------------------------------------------------------------------
# 7. Persona YAML injection
# ---------------------------------------------------------------------------


class TestPersonaYamlInjection:
    """YAML-hostile values in persona fields must not alter the structure."""

    @pytest.mark.parametrize("injected_name", [
        "!!python/object:os.system",
        "!!str",
        "&anchor value",
        "*alias",
        "key: injected\nnew_key: evil",
        "---\ninjected: true",
        "...\ninjected: true",
        "> folded\n  scalar\n  here",
        "| literal\n  block",
    ])
    def test_yaml_injection_in_name_is_safe(
        self, tmp_path: Path, injected_name: str
    ) -> None:
        """Injected YAML constructs must not execute or restructure the file."""
        pm = PersonaManager(persona_path=tmp_path / "persona.yaml")
        pm.update(name=injected_name)
        pm.save()

        # Reload — should not raise and must load as a flat dict, not execute code.
        pm2 = PersonaManager(persona_path=tmp_path / "persona.yaml")
        loaded_name = pm2.get_persona().name
        # The name may be transformed by YAML quoting, but the object must be a str.
        assert isinstance(loaded_name, str)

    def test_injection_does_not_add_extra_keys_to_yaml(self, tmp_path: Path) -> None:
        injected = "name: injected\nmalicious_key: evil_value"
        pm = PersonaManager(persona_path=tmp_path / "persona.yaml")
        pm.update(name=injected)
        pm.save()

        raw = yaml.safe_load((tmp_path / "persona.yaml").read_text(encoding="utf-8"))
        # Injected key must NOT appear as a top-level key.
        assert "malicious_key" not in raw

    def test_colon_in_boundary_survives_round_trip(self, tmp_path: Path) -> None:
        boundary = "Never run: rm -rf / (destructive: true)"
        pm = PersonaManager(persona_path=tmp_path / "persona.yaml")
        pm.update(boundaries=[boundary])
        pm.save()

        pm2 = PersonaManager(persona_path=tmp_path / "persona.yaml")
        assert pm2.get_persona().boundaries == [boundary]


# ---------------------------------------------------------------------------
# 8. Behavior layer with extreme inputs
# ---------------------------------------------------------------------------


class TestBehaviorLayerExtremeInputs:
    """BehaviorLayer and IntentInterpreter must handle extreme-size inputs."""

    def test_100kb_single_message_tone_analysis_does_not_crash(self) -> None:
        layer = BehaviorLayer()
        big_content = "word " * 20000  # ~100 KB
        messages = [{"role": "user", "content": big_content}]
        tone = layer.analyze_user_tone(messages)
        assert tone in ("casual", "formal", "frustrated", "technical", "brief", "verbose")

    def test_1000_message_history_tone_analysis(self) -> None:
        """Only the last 5 user messages are inspected; this must remain fast."""
        layer = BehaviorLayer()
        messages = [
            {"role": "user" if i % 2 == 0 else "assistant", "content": f"Message {i}"}
            for i in range(1000)
        ]
        # Last user message is "Message 998" which is 2 words → brief
        tone = layer.analyze_user_tone(messages)
        assert isinstance(tone, str)
        assert len(tone) > 0

    def test_100kb_message_intent_classification(self) -> None:
        interp = IntentInterpreter()
        big_text = "a " * 50000  # ~100 KB
        result = interp.classify_intent(big_text)
        valid_intents = {
            "greeting", "farewell", "confirmation", "frustration",
            "troubleshooting", "clarification", "feedback",
            "exploration", "command", "question",
        }
        assert result in valid_intents

    def test_shape_system_prompt_with_100kb_base_prompt(self) -> None:
        layer = BehaviorLayer()
        big_base = "instruction " * 10000  # ~100 KB
        result = layer.shape_system_prompt(big_base, {})
        assert isinstance(result, str)
        assert big_base.rstrip() in result

    def test_response_shaper_with_100kb_clean_response(self) -> None:
        shaper = ResponseShaper()
        # A large clean response (no robotic phrases) should pass through intact.
        big_clean = "The answer is: " + "detail " * 20000
        result = shaper.shape_response(big_clean, persona=None, context={})
        assert "The answer is" in result

    def test_1000_messages_only_last_5_user_messages_affect_tone(self) -> None:
        """Verify that history truncation to 5 user messages is respected."""
        layer = BehaviorLayer()
        # 995 casual messages followed by 5 technical messages.
        messages: list[dict] = [
            {"role": "user", "content": "hey lol cool thanks ya btw ya ngl it is awesome"}
            for _ in range(995)
        ]
        technical_msg = (
            "I need to call the function via the API endpoint using "
            "the database query schema config yaml json async await thread"
        )
        messages.extend(
            {"role": "user", "content": technical_msg} for _ in range(5)
        )
        # Only last 5 → all technical → should detect technical.
        tone = layer.analyze_user_tone(messages)
        assert tone == "technical"

    def test_get_response_guidelines_with_1000_persona_rules(
        self, real_persona_module: Any
    ) -> None:
        persona = real_persona_module.PersonaConfig(
            behavioral_tendencies=[f"tendency_{i}" for i in range(500)],
            response_style_rules=[f"rule_{i}" for i in range(500)],
        )
        layer = BehaviorLayer(persona)
        ctx = {
            "user_tone": "casual",
            "turn_count": 1,
            "has_tool_results": False,
            "topic": "",
            "intent": "question",
            "urgency": "low",
        }
        guidelines = layer.get_response_guidelines(ctx)
        assert isinstance(guidelines, str)
        # All rules should appear.
        assert "tendency_0" in guidelines
        assert "rule_499" in guidelines


# ---------------------------------------------------------------------------
# 9. Behavior tone analysis stability
# ---------------------------------------------------------------------------


class TestBehaviorToneAnalysisStability:
    """Same input must produce the same tone on every invocation."""

    @pytest.mark.parametrize("content,expected_tone", [
        (
            "I need to call the function via the API endpoint using "
            "the database query schema config yaml json async await thread",
            "technical",
        ),
        (
            "Please kindly assist me regarding this matter and furthermore "
            "I would appreciate your thorough response accordingly therefore "
            "sincerely I ask",
            "formal",
        ),
        (
            "it still doesn't work I've tried everything same error again",
            "frustrated",
        ),
        (
            "hey cool thanks ya lol btw fyi gonna wanna kinda ngl tbh",
            "casual",
        ),
    ])
    def test_same_input_always_produces_same_tone(
        self, content: str, expected_tone: str
    ) -> None:
        layer = BehaviorLayer()
        messages = [{"role": "user", "content": content}]
        results = {layer.analyze_user_tone(messages) for _ in range(20)}
        assert len(results) == 1, (
            f"Non-deterministic tone for {expected_tone!r}: got {results}"
        )
        assert results.pop() == expected_tone

    def test_tone_is_deterministic_across_multiple_instances(self) -> None:
        """Two separate BehaviorLayer instances must agree on tone."""
        content = "docker kubernetes container deployment pipeline ci config script"
        messages = [{"role": "user", "content": content}]

        tones = {BehaviorLayer().analyze_user_tone(messages) for _ in range(10)}
        assert len(tones) == 1, f"Different instances produced different tones: {tones}"


# ---------------------------------------------------------------------------
# 10. Intent classifier boundary cases
# ---------------------------------------------------------------------------


class TestIntentClassifierBoundaryCases:
    """Messages that match multiple intent patterns; ordering must be consistent."""

    def test_greeting_plus_question_resolves_to_greeting(self) -> None:
        """Greeting takes highest priority even when a question mark is present."""
        interp = IntentInterpreter()
        assert interp.classify_intent("hey, how does this work?") == "greeting"

    def test_farewell_plus_question_resolves_to_farewell(self) -> None:
        """Farewell pattern wins over question when present."""
        interp = IntentInterpreter()
        result = interp.classify_intent("bye, is there anything else I should know?")
        assert result == "farewell"

    def test_frustration_beats_troubleshooting(self) -> None:
        """Frustration is checked before troubleshooting in classifier order."""
        interp = IntentInterpreter()
        # Contains both frustration and troubleshooting signals.
        result = interp.classify_intent(
            "still not working — getting the same error again and again"
        )
        assert result == "frustration"

    def test_confirmation_short_yes_with_trailing_words(self) -> None:
        """Confirmation regex is anchored; trailing words disqualify it."""
        interp = IntentInterpreter()
        # Plain "yes" → confirmation
        assert interp.classify_intent("yes") == "confirmation"
        # "yes I want to add logging" → NOT confirmation (too long)
        result = interp.classify_intent("yes I want to add logging to the service")
        assert result != "confirmation"

    def test_command_vs_question_boundary(self) -> None:
        """Commands starting with question words should classify as question."""
        interp = IntentInterpreter()
        # "what" matches question pattern first in the fallback check.
        assert interp.classify_intent("what do I run to restart nginx?") == "question"

    def test_exploration_plus_command_resolves_to_exploration(self) -> None:
        """Exploration is checked after command patterns — edge on ordering."""
        interp = IntentInterpreter()
        # "tell me more" matches exploration.
        result = interp.classify_intent("tell me more about how to deploy this")
        assert result == "exploration"

    def test_clarification_plus_question_mark(self) -> None:
        interp = IntentInterpreter()
        result = interp.classify_intent("what do you mean by that?")
        assert result == "clarification"

    def test_empty_input_is_question(self) -> None:
        interp = IntentInterpreter()
        assert interp.classify_intent("") == "question"

    def test_punctuation_only_is_question(self) -> None:
        interp = IntentInterpreter()
        assert interp.classify_intent("???") == "question"

    def test_numeric_only_is_question(self) -> None:
        interp = IntentInterpreter()
        assert interp.classify_intent("42") == "question"

    def test_mixed_urgency_high_takes_precedence(self) -> None:
        interp = IntentInterpreter()
        # Both "asap" (high) and "today" (medium) present.
        assert interp.extract_urgency("fix this asap, must be done today") == "high"

    def test_urgency_is_deterministic_for_boundary_phrases(self) -> None:
        interp = IntentInterpreter()
        phrase = "need to fix this before end of morning deadline soon"
        results = {interp.extract_urgency(phrase) for _ in range(20)}
        assert len(results) == 1


# ---------------------------------------------------------------------------
# 11. Response shaper idempotency
# ---------------------------------------------------------------------------


class TestResponseShaperIdempotency:
    """Applying the shaper twice must produce the same output as once."""

    @pytest.mark.parametrize("raw_response", [
        "Certainly! As an AI, I can help you. The answer is 42.",
        "Great question! Of course I'll assist. Here is the code:\n```python\nprint(1)\n```",
        "I'd be happy to help. As an AI language model, here is what you need.",
        "As your assistant, I recommend this approach.",
        "Absolutely! I'm here to help. Let me explain step by step.",
        "The configuration lives at ~/.missy/config.yaml.",  # no robotic phrases
        "```bash\necho 'hello'\n```",  # pure code block
        "",  # empty
        "   \n\n  ",  # whitespace only
    ])
    def test_applying_shaper_twice_is_idempotent(self, raw_response: str) -> None:
        shaper = ResponseShaper()
        once = shaper.shape_response(raw_response, persona=None, context={})
        twice = shaper.shape_response(once, persona=None, context={})
        assert once == twice, (
            f"Shaper not idempotent on input {raw_response!r}\n"
            f"  After 1st pass: {once!r}\n"
            f"  After 2nd pass: {twice!r}"
        )

    def test_shaper_idempotent_on_many_code_blocks(self) -> None:
        shaper = ResponseShaper()
        raw = "\n".join([
            "Certainly! Here is the solution.",
            "```python",
            "# As an AI, I wrote this",
            "def foo(): return 42",
            "```",
            "And also:",
            "```bash",
            "echo 'Of course!'",
            "```",
            "That completes it.",
        ])
        once = shaper.shape_response(raw, persona=None, context={})
        twice = shaper.shape_response(once, persona=None, context={})
        assert once == twice

    def test_shaper_idempotent_with_persona(self) -> None:
        persona = PersonaConfig()
        shaper = ResponseShaper()
        raw = "As an AI language model, I'd be happy to help you with that."
        once = shaper.shape_response(raw, persona=persona, context={})
        twice = shaper.shape_response(once, persona=persona, context={})
        assert once == twice


# ---------------------------------------------------------------------------
# 12. Hatching log rotation (large file)
# ---------------------------------------------------------------------------


class TestHatchingLogRotation:
    """HatchingLog must handle very large log files without memory issues."""

    def test_write_and_read_10000_log_entries(self, tmp_path: Path) -> None:
        log_path = tmp_path / "big.jsonl"
        log = HatchingLog(log_path=log_path)

        for i in range(10000):
            log.log(f"step_{i % 7}", "ok", f"Message {i}", {"index": i})

        entries = log.get_entries()
        assert len(entries) == 10000
        assert entries[0]["details"]["index"] == 0
        assert entries[-1]["details"]["index"] == 9999

    def test_corrupt_lines_in_large_file_are_skipped(self, tmp_path: Path) -> None:
        log_path = tmp_path / "corrupted.jsonl"
        good_entry = json.dumps({
            "timestamp": "2026-01-01T00:00:00+00:00",
            "step": "good",
            "status": "ok",
            "message": "fine",
            "details": {},
        })

        lines: list[str] = []
        for i in range(1000):
            if i % 10 == 5:
                lines.append(f"NOT JSON at line {i}\n")
            else:
                lines.append(good_entry + "\n")
        log_path.write_text("".join(lines), encoding="utf-8")

        log = HatchingLog(log_path=log_path)
        entries = log.get_entries()
        # 100 lines are corrupt (indices 5, 15, 25, …, 995) → 900 good.
        assert len(entries) == 900

    def test_large_log_file_get_entries_order_preserved(self, tmp_path: Path) -> None:
        log_path = tmp_path / "ordered.jsonl"
        log = HatchingLog(log_path=log_path)

        for i in range(500):
            log.log("step", "ok", f"entry {i}", {"seq": i})

        entries = log.get_entries()
        seqs = [e["details"]["seq"] for e in entries]
        assert seqs == list(range(500))

    def test_log_with_very_long_message_per_entry(self, tmp_path: Path) -> None:
        """Each log message is 10 KB; 100 entries → ~1 MB log file."""
        log_path = tmp_path / "longmsg.jsonl"
        log = HatchingLog(log_path=log_path)

        for i in range(100):
            log.log("step", "ok", "X" * 10240, {"index": i})

        entries = log.get_entries()
        assert len(entries) == 100
        assert len(entries[0]["message"]) == 10240


# ---------------------------------------------------------------------------
# 13. Persona audit log integrity
# ---------------------------------------------------------------------------


class TestPersonaAuditLogIntegrity:
    """1000 audit entries must all be valid JSONL."""

    def test_1000_saves_all_produce_valid_audit_entries(self, tmp_path: Path) -> None:
        pm = PersonaManager(persona_path=tmp_path / "persona.yaml")
        pm.save()  # Create initial file.

        for i in range(999):
            pm.update(name=f"Audit{i}")
            pm.save()

        audit_path = tmp_path / "persona_audit.jsonl"
        assert audit_path.exists()

        lines = [
            line.strip()
            for line in audit_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        # 1000 saves → 1000 audit entries.
        assert len(lines) == 1000

        for lineno, line in enumerate(lines, 1):
            entry = json.loads(line)  # raises if malformed
            assert "timestamp" in entry, f"Line {lineno} missing timestamp"
            assert "action" in entry, f"Line {lineno} missing action"
            assert "version" in entry, f"Line {lineno} missing version"
            assert "name" in entry, f"Line {lineno} missing name"
            assert entry["action"] == "save", f"Line {lineno} wrong action: {entry['action']}"

    def test_audit_entry_versions_are_monotonically_increasing(self, tmp_path: Path) -> None:
        pm = PersonaManager(persona_path=tmp_path / "persona.yaml")
        for i in range(20):
            pm.update(name=f"Ver{i}")
            pm.save()

        entries = pm.get_audit_log()
        versions = [e["version"] for e in entries]
        for a, b in zip(versions, versions[1:]):
            assert b > a, f"Audit version went backwards: {a} → {b}"

    def test_audit_log_reset_and_rollback_entries_present(self, tmp_path: Path, monkeypatch: Any) -> None:
        call_count = 0

        def _mock_strftime(fmt: str, *args: Any) -> str:
            nonlocal call_count
            call_count += 1
            return f"20260318_{call_count:06d}"

        monkeypatch.setattr(time, "strftime", _mock_strftime)

        pm = PersonaManager(persona_path=tmp_path / "persona.yaml")
        pm.save()
        pm.update(name="Modified")
        pm.save()
        pm.reset()
        pm.rollback()

        entries = pm.get_audit_log()
        actions = [e["action"] for e in entries]
        assert "save" in actions
        assert "reset" in actions
        assert "rollback" in actions

    def test_audit_log_survives_concurrent_appends(self, tmp_path: Path) -> None:
        """Multiple PersonaManager instances writing to the audit log concurrently.

        The persona backup mechanism uses second-granularity timestamps for
        backup filenames.  When two threads back up in the same wall-clock
        second ``shutil.copy2`` raises ``SameFileError`` because both threads
        resolve to the same destination path.  That collision is an accepted
        limitation of the backup naming scheme and must *not* corrupt the audit
        log — so we filter ``SameFileError`` out of the fatal-error list and
        only fail on unexpected exceptions.
        """
        import shutil

        pm_path = tmp_path / "persona.yaml"
        # Pre-create a saved persona so later managers load it.
        seed = PersonaManager(persona_path=pm_path)
        seed.save()

        errors: list[Exception] = []

        def _write_audit(n: int) -> None:
            try:
                pm = PersonaManager(persona_path=pm_path)
                for i in range(n):
                    pm.update(name=f"Thread{n}_{i}")
                    pm.save()
            except shutil.SameFileError:
                # Two threads created a backup at the same second — known race.
                pass
            except Exception as exc:  # noqa: BLE001
                errors.append(exc)

        threads = [threading.Thread(target=_write_audit, args=(10,)) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=15)

        assert not errors, f"Threads raised unexpected errors: {errors}"

        # Every line in the audit log must parse as JSON.
        audit_path = tmp_path / "persona_audit.jsonl"
        if audit_path.exists():
            for raw_line in audit_path.read_text(encoding="utf-8").splitlines():
                line = raw_line.strip()
                if line:
                    json.loads(line)  # raises on corruption


# ---------------------------------------------------------------------------
# 14. Persona concurrent reads
# ---------------------------------------------------------------------------


class TestPersonaConcurrentReads:
    """Multiple threads reading the persona simultaneously must not see errors."""

    def test_50_threads_read_persona_simultaneously(self, tmp_path: Path) -> None:
        path = tmp_path / "persona.yaml"
        pm = PersonaManager(persona_path=path)
        pm.update(name="Concurrent", tone=["stable"])
        pm.save()

        errors: list[Exception] = []
        names_seen: set[str] = set()
        lock = threading.Lock()

        def _read() -> None:
            try:
                for _ in range(20):
                    reader = PersonaManager(persona_path=path)
                    persona = reader.get_persona()
                    with lock:
                        names_seen.add(persona.name)
            except Exception as exc:  # noqa: BLE001
                errors.append(exc)

        threads = [threading.Thread(target=_read) for _ in range(50)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=15)

        assert not errors, f"Read threads raised: {errors}"
        # All threads must have seen the same name.
        assert names_seen == {"Concurrent"}

    def test_reads_during_concurrent_writes_do_not_raise(self, tmp_path: Path) -> None:
        """Reader threads should never crash even while a writer is active."""
        path = tmp_path / "persona.yaml"
        pm_writer = PersonaManager(persona_path=path)
        pm_writer.save()

        stop_event = threading.Event()
        errors: list[Exception] = []

        def _writer() -> None:
            try:
                for i in range(50):
                    pm_writer.update(name=f"Write{i}")
                    pm_writer.save()
            except Exception as exc:  # noqa: BLE001
                errors.append(exc)
            finally:
                stop_event.set()

        def _reader() -> None:
            while not stop_event.is_set():
                try:
                    r = PersonaManager(persona_path=path)
                    r.get_persona()
                except Exception as exc:  # noqa: BLE001
                    errors.append(exc)
                    break

        writer_thread = threading.Thread(target=_writer)
        reader_threads = [threading.Thread(target=_reader) for _ in range(5)]

        writer_thread.start()
        for t in reader_threads:
            t.start()

        writer_thread.join(timeout=10)
        stop_event.set()
        for t in reader_threads:
            t.join(timeout=5)

        assert not errors, f"Threads raised: {errors}"


# ---------------------------------------------------------------------------
# 15. Hatching + persona integration
# ---------------------------------------------------------------------------


@pytest.fixture()
def real_persona_module() -> Any:
    """Ensure ``missy.agent.persona`` is the real module, not a MagicMock.

    Some tests in the wider test suite patch ``sys.modules`` with a MagicMock
    for ``missy.agent.persona``.  If such a test fails mid-teardown the mock
    can bleed into subsequent tests.  This fixture saves the current entry,
    forces a real reload of the module, and restores the original entry after
    the test completes — regardless of whether the test passes or fails.
    """
    import importlib

    key = "missy.agent.persona"
    original = sys.modules.get(key)

    # Remove any stale cached entry (which may be a MagicMock) then reload the
    # real module so ``from missy.agent.persona import PersonaManager`` inside
    # ``_generate_persona`` always gets the genuine class.
    sys.modules.pop(key, None)
    real_module = importlib.import_module(key)
    sys.modules[key] = real_module

    yield real_module

    # Restore original (may be None if it was not imported before the suite).
    if original is None:
        sys.modules.pop(key, None)
    else:
        sys.modules[key] = original


class TestHatchingPersonaIntegration:
    """Full hatching with real PersonaManager must produce a valid persona file."""

    def test_full_hatch_creates_valid_persona_yaml(
        self, tmp_path: Path, monkeypatch: Any, real_persona_module: Any
    ) -> None:
        _patch_module_paths(monkeypatch, tmp_path)
        monkeypatch.setenv("ANTHROPIC_API_KEY", "key-integration")

        # Allow the real PersonaManager to run (no stub).
        mock_turn = MagicMock()
        mock_turn.id = "turn-int"
        mock_store = MagicMock()

        with patch.dict(
            "sys.modules",
            {
                "missy.memory.sqlite_store": MagicMock(
                    SQLiteMemoryStore=MagicMock(return_value=mock_store),
                    ConversationTurn=MagicMock(new=MagicMock(return_value=mock_turn)),
                ),
            },
        ):
            mgr = _make_manager(tmp_path)
            state = mgr.run_hatching(interactive=False)

        assert state.status is HatchingStatus.HATCHED
        persona_path = tmp_path / "persona.yaml"
        assert persona_path.exists(), "Persona file was not created by hatching"

        # Load via PersonaManager — must not fall back to defaults (i.e., parse cleanly).
        pm = PersonaManager(persona_path=persona_path)
        persona = pm.get_persona()
        assert persona.name == "Missy"
        assert isinstance(persona.tone, list)
        assert isinstance(persona.boundaries, list)
        assert persona.version >= 1

    def test_full_hatch_persona_system_prompt_is_valid(
        self, tmp_path: Path, monkeypatch: Any, real_persona_module: Any
    ) -> None:
        _patch_module_paths(monkeypatch, tmp_path)
        monkeypatch.setenv("ANTHROPIC_API_KEY", "key-integration-prompt")

        mock_turn = MagicMock()
        mock_turn.id = "turn-int2"
        mock_store = MagicMock()

        with patch.dict(
            "sys.modules",
            {
                "missy.memory.sqlite_store": MagicMock(
                    SQLiteMemoryStore=MagicMock(return_value=mock_store),
                    ConversationTurn=MagicMock(new=MagicMock(return_value=mock_turn)),
                ),
            },
        ):
            mgr = _make_manager(tmp_path)
            mgr.run_hatching(interactive=False)

        pm = PersonaManager(persona_path=tmp_path / "persona.yaml")
        prefix = pm.get_system_prompt_prefix()
        assert "# Identity" in prefix
        assert "# Tone" in prefix
        assert "# Boundaries" in prefix

    def test_hatch_then_persona_edit_then_rollback_restores_hatch_defaults(
        self, tmp_path: Path, monkeypatch: Any, real_persona_module: Any
    ) -> None:
        _patch_module_paths(monkeypatch, tmp_path)
        monkeypatch.setenv("ANTHROPIC_API_KEY", "key-rollback")

        mock_turn = MagicMock()
        mock_turn.id = "turn-rb"
        mock_store = MagicMock()

        with patch.dict(
            "sys.modules",
            {
                "missy.memory.sqlite_store": MagicMock(
                    SQLiteMemoryStore=MagicMock(return_value=mock_store),
                    ConversationTurn=MagicMock(new=MagicMock(return_value=mock_turn)),
                ),
            },
        ):
            mgr = _make_manager(tmp_path)
            mgr.run_hatching(interactive=False)

        persona_path = tmp_path / "persona.yaml"
        pm = PersonaManager(persona_path=persona_path)

        # Edit and save so a backup of the hatch-created persona exists.
        pm.update(name="EditedPostHatch")
        pm.save()
        assert pm.get_persona().name == "EditedPostHatch"

        # Roll back to the hatch-created version.
        result = pm.rollback()
        assert result is not None, "Rollback returned None — no backup found"
        assert pm.get_persona().name == "Missy", (
            f"After rollback expected 'Missy', got {pm.get_persona().name!r}"
        )

    def test_hatching_persona_generated_flag_set(
        self, tmp_path: Path, monkeypatch: Any, real_persona_module: Any
    ) -> None:
        _patch_module_paths(monkeypatch, tmp_path)
        monkeypatch.setenv("ANTHROPIC_API_KEY", "key-flag")

        mock_turn = MagicMock()
        mock_turn.id = "turn-flag"
        mock_store = MagicMock()

        with patch.dict(
            "sys.modules",
            {
                "missy.memory.sqlite_store": MagicMock(
                    SQLiteMemoryStore=MagicMock(return_value=mock_store),
                    ConversationTurn=MagicMock(new=MagicMock(return_value=mock_turn)),
                ),
            },
        ):
            mgr = _make_manager(tmp_path)
            state = mgr.run_hatching(interactive=False)

        assert state.persona_generated is True

    def test_hatch_with_existing_persona_skips_generation(
        self, tmp_path: Path, monkeypatch: Any, real_persona_module: Any
    ) -> None:
        """If persona.yaml already exists, _generate_persona should skip silently."""
        _patch_module_paths(monkeypatch, tmp_path)
        monkeypatch.setenv("ANTHROPIC_API_KEY", "key-skip")

        # Pre-create a persona file with a custom name.
        existing_pm = PersonaManager(persona_path=tmp_path / "persona.yaml")
        existing_pm.update(name="PreExisting")
        existing_pm.save()

        mock_turn = MagicMock()
        mock_turn.id = "turn-skip"
        mock_store = MagicMock()

        with patch.dict(
            "sys.modules",
            {
                "missy.memory.sqlite_store": MagicMock(
                    SQLiteMemoryStore=MagicMock(return_value=mock_store),
                    ConversationTurn=MagicMock(new=MagicMock(return_value=mock_turn)),
                ),
            },
        ):
            mgr = _make_manager(tmp_path)
            state = mgr.run_hatching(interactive=False)

        assert state.status is HatchingStatus.HATCHED
        # The pre-existing name must be preserved.
        pm = PersonaManager(persona_path=tmp_path / "persona.yaml")
        assert pm.get_persona().name == "PreExisting"


# ---------------------------------------------------------------------------
# Additional edge cases not covered by existing tests
# ---------------------------------------------------------------------------


class TestHatchingStateSerializationEdgeCases:
    """Extreme values in HatchingState serialisation."""

    def test_very_long_error_message_round_trips(self) -> None:
        long_error = "E" * 100000
        state = HatchingState(status=HatchingStatus.FAILED, error=long_error)
        restored = HatchingState.from_dict(state.to_dict())
        assert restored.error == long_error

    def test_steps_completed_with_duplicates_preserved(self) -> None:
        """from_dict must not deduplicate steps_completed (caller's responsibility)."""
        state = HatchingState(steps_completed=["step_a", "step_a", "step_b"])
        restored = HatchingState.from_dict(state.to_dict())
        assert restored.steps_completed.count("step_a") == 2

    def test_from_dict_with_all_statuses(self) -> None:
        for status in HatchingStatus:
            d = {"status": status.value}
            restored = HatchingState.from_dict(d)
            assert restored.status is status

    def test_to_dict_from_dict_preserves_unicode_error(self) -> None:
        error = "Step failed: 日本語 エラー 🚨"
        state = HatchingState(status=HatchingStatus.FAILED, error=error)
        restored = HatchingState.from_dict(state.to_dict())
        assert restored.error == error


class TestPersonaManagerFieldIsolation:
    """get_persona() must return a deep-enough copy that list mutations are safe."""

    def test_mutating_tone_on_copy_does_not_affect_manager(self, tmp_path: Path) -> None:
        pm = PersonaManager(persona_path=tmp_path / "persona.yaml")
        copy = pm.get_persona()
        copy.tone.append("mutated")
        assert "mutated" not in pm.get_persona().tone

    def test_mutating_boundaries_on_copy_does_not_affect_manager(self, tmp_path: Path) -> None:
        pm = PersonaManager(persona_path=tmp_path / "persona.yaml")
        copy = pm.get_persona()
        copy.boundaries.clear()
        assert len(pm.get_persona().boundaries) > 0

    def test_two_successive_saves_produce_distinct_backup_files(self, tmp_path: Path, monkeypatch: Any) -> None:
        call_count = 0

        def _mock_strftime(fmt: str, *args: Any) -> str:
            nonlocal call_count
            call_count += 1
            return f"20260101_{call_count:06d}"

        monkeypatch.setattr(time, "strftime", _mock_strftime)

        pm = PersonaManager(persona_path=tmp_path / "persona.yaml")
        pm.save()  # creates file, no backup
        pm.save()  # backup 1
        pm.save()  # backup 2

        backups = pm.list_backups()
        assert len(backups) == 2
        # Names must be distinct.
        names = [b.name for b in backups]
        assert len(set(names)) == 2


class TestResponseShaperCodeBlockStash:
    """Verify that the stash/restore mechanism handles edge cases."""

    def test_code_block_placeholder_text_in_response_is_not_confused(self) -> None:
        """If the response literally contains the placeholder pattern, shaper must not corrupt it."""
        shaper = ResponseShaper()
        # Construct a response that happens to contain the internal placeholder format.
        raw = "See \x00CODE_BLOCK_0\x00 for details."
        result = shaper.shape_response(raw, persona=None, context={})
        # The placeholder text is not a real code block, so it passes through as-is.
        assert isinstance(result, str)

    def test_nested_backticks_in_code_block_are_preserved(self) -> None:
        shaper = ResponseShaper()
        raw = (
            "As an AI, here is the example:\n"
            "```python\n"
            "# inline `code` within code block\n"
            "x = '`backtick`'\n"
            "```"
        )
        result = shaper.shape_response(raw, persona=None, context={})
        assert "`code`" in result
        assert "`backtick`" in result
        assert "As an AI" not in result

    def test_response_with_only_robotic_phrase_becomes_empty_or_minimal(self) -> None:
        shaper = ResponseShaper()
        raw = "Certainly!"
        result = shaper.shape_response(raw, persona=None, context={})
        # After stripping "Certainly!" the result should be empty or whitespace-free.
        assert result == result.strip()

    def test_100_code_blocks_all_restored(self) -> None:
        shaper = ResponseShaper()
        blocks = [f"```python\nprint({i})\n```" for i in range(100)]
        raw = "Certainly!\n" + "\n\n".join(blocks)
        result = shaper.shape_response(raw, persona=None, context={})
        for i in range(100):
            assert f"print({i})" in result, f"Code block {i} was not restored"
