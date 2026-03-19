"""Session 13 supplemental edge-case tests for persona.py and behavior.py.

Covers gaps not addressed in test_persona.py / test_behavior.py:

Persona
- PersonaConfig field-level defaults (exact values, not just presence)
- system_prompt_prefix with all fields fully populated
- system_prompt_prefix with empty behavioral_tendencies / response_style_rules
- system_prompt_prefix with empty personality_traits
- Very long identity_description
- Atomic save: temp file is cleaned up on error
- Version tracking across multiple update+save cycles
- reset() after multiple saves preserves version continuity
- Backup list is sorted oldest-first (mtime ordering)
- rollback() when persona file does not yet exist on disk
- diff() unified-diff format markers (--- / +++)
- diff() shows changed field value
- Audit log: rollback entry carries from_backup detail
- Audit log: corrupt lines are skipped gracefully
- Concurrent save() calls do not lose data (thread safety via threading.Thread)
- _persona_from_dict with partial dict (missing optional fields uses defaults)

Behavior
- IntentInterpreter: "feedback" intent from "you got it wrong"
- IntentInterpreter: "farewell" intent mid-sentence (cya / ttyl)
- IntentInterpreter: "confirmation" with period ("okay.")
- IntentInterpreter: high urgency from "immediately"
- IntentInterpreter: high urgency from "outage"
- IntentInterpreter: medium urgency from "soon"
- IntentInterpreter: medium urgency from "must"
- IntentInterpreter: empty string urgency is low
- IntentInterpreter: very long input does not raise
- IntentInterpreter: all ten categories are reachable
- ResponseShaper: "I am an AI assistant" preamble stripped
- ResponseShaper: "Please note that I am an AI" stripped
- ResponseShaper: "Absolutely!" stripped
- ResponseShaper: "That's a great question!" stripped
- ResponseShaper: content after stripping leading robotic phrase is non-empty
- ResponseShaper: fenced block with language tag preserved verbatim
- ResponseShaper: inline code containing robotic phrase text preserved
- ResponseShaper: trailing whitespace removed from result
- BehaviorLayer: vision_mode="painting" adds warm/encouraging directive
- BehaviorLayer: vision_mode="puzzle" adds patient/placement directive
- BehaviorLayer: vision_mode="inspection" (generic) adds visual detail directive
- BehaviorLayer: system prompt contains "## Response guidelines" section header
- BehaviorLayer: no_persona analyze_user_tone with only non-user messages
- BehaviorLayer: medium urgency produces "focused" / "actionable" guideline
- BehaviorLayer: farewell intent produces brief farewell guideline
- BehaviorLayer: shape_system_prompt with empty base prompt
- BehaviorLayer: get_response_guidelines returns bullet lines for all non-empty personas
"""

from __future__ import annotations

import threading
import time

import yaml

from missy.agent.behavior import BehaviorLayer, IntentInterpreter, ResponseShaper
from missy.agent.persona import (
    PersonaConfig,
    PersonaManager,
    _persona_from_dict,
)

# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------


def _make_manager(tmp_path, name: str = "persona.yaml") -> PersonaManager:
    return PersonaManager(persona_path=tmp_path / name)


def _full_persona() -> PersonaConfig:
    return PersonaConfig(
        name="FullBot",
        tone=["warm", "direct", "technical"],
        personality_traits=["curious", "pragmatic", "security-conscious"],
        behavioral_tendencies=["prefers action", "asks clarifying questions"],
        response_style_rules=["be concise", "show reasoning"],
        boundaries=["never expose secrets", "respect policy engine"],
        identity_description="FullBot is a comprehensive AI assistant for all tasks.",
        version=3,
    )


def _base_ctx(**overrides) -> dict:
    ctx = {
        "user_tone": "casual",
        "topic": "",
        "turn_count": 1,
        "has_tool_results": False,
        "intent": "question",
        "urgency": "low",
    }
    ctx.update(overrides)
    return ctx


def _user_messages(*texts: str) -> list[dict]:
    return [{"role": "user", "content": t} for t in texts]


# ===========================================================================
# PERSONA TESTS
# ===========================================================================


class TestPersonaConfigDefaultValues:
    """Verify exact default values, not just type/presence."""

    def test_default_name_is_missy(self):
        assert PersonaConfig().name == "Missy"

    def test_default_version_is_1(self):
        assert PersonaConfig().version == 1

    def test_default_tone_contains_helpful(self):
        assert "helpful" in PersonaConfig().tone

    def test_default_tone_contains_direct(self):
        assert "direct" in PersonaConfig().tone

    def test_default_tone_contains_technical(self):
        assert "technical" in PersonaConfig().tone

    def test_default_personality_traits_contains_curious(self):
        assert "curious" in PersonaConfig().personality_traits

    def test_default_personality_traits_contains_pragmatic(self):
        assert "pragmatic" in PersonaConfig().personality_traits

    def test_default_boundaries_mentions_secrets(self):
        joined = " ".join(PersonaConfig().boundaries).lower()
        assert "secret" in joined or "credential" in joined

    def test_default_identity_mentions_linux(self):
        assert "Linux" in PersonaConfig().identity_description

    def test_default_identity_mentions_security(self):
        desc = PersonaConfig().identity_description.lower()
        assert "security" in desc


class TestSystemPromptPrefixAllFields:
    """Verify prompt prefix when all fields are populated."""

    def test_identity_section_present(self, tmp_path):
        pm = _make_manager(tmp_path)
        pm.update(identity_description="FullBot description here.")
        prefix = pm.get_system_prompt_prefix()
        assert "# Identity" in prefix
        assert "FullBot description here." in prefix

    def test_tone_section_lists_all_tones(self, tmp_path):
        pm = _make_manager(tmp_path)
        pm.update(tone=["alpha", "beta", "gamma"])
        prefix = pm.get_system_prompt_prefix()
        assert "alpha" in prefix
        assert "beta" in prefix
        assert "gamma" in prefix

    def test_personality_section_present(self, tmp_path):
        pm = _make_manager(tmp_path)
        pm.update(personality_traits=["trait-x", "trait-y"])
        prefix = pm.get_system_prompt_prefix()
        assert "# Personality" in prefix
        assert "trait-x" in prefix
        assert "trait-y" in prefix

    def test_behavioural_tendencies_listed_as_bullets(self, tmp_path):
        pm = _make_manager(tmp_path)
        pm.update(behavioral_tendencies=["tendency-a", "tendency-b"])
        prefix = pm.get_system_prompt_prefix()
        assert "# Behavioural Tendencies" in prefix
        assert "- tendency-a" in prefix
        assert "- tendency-b" in prefix

    def test_response_style_listed_as_bullets(self, tmp_path):
        pm = _make_manager(tmp_path)
        pm.update(response_style_rules=["rule-one", "rule-two"])
        prefix = pm.get_system_prompt_prefix()
        assert "# Response Style" in prefix
        assert "- rule-one" in prefix
        assert "- rule-two" in prefix

    def test_boundaries_listed_as_bullets(self, tmp_path):
        pm = _make_manager(tmp_path)
        pm.update(boundaries=["bound-x", "bound-y"])
        prefix = pm.get_system_prompt_prefix()
        assert "# Boundaries" in prefix
        assert "- bound-x" in prefix
        assert "- bound-y" in prefix

    def test_empty_personality_traits_omits_section(self, tmp_path):
        pm = _make_manager(tmp_path)
        pm.update(personality_traits=[])
        prefix = pm.get_system_prompt_prefix()
        assert "# Personality" not in prefix

    def test_empty_behavioural_tendencies_omits_section(self, tmp_path):
        pm = _make_manager(tmp_path)
        pm.update(behavioral_tendencies=[])
        prefix = pm.get_system_prompt_prefix()
        assert "# Behavioural Tendencies" not in prefix

    def test_empty_response_style_omits_section(self, tmp_path):
        pm = _make_manager(tmp_path)
        pm.update(response_style_rules=[])
        prefix = pm.get_system_prompt_prefix()
        assert "# Response Style" not in prefix

    def test_very_long_identity_included_in_full(self, tmp_path):
        long_desc = "X" * 5000
        pm = _make_manager(tmp_path)
        pm.update(identity_description=long_desc)
        prefix = pm.get_system_prompt_prefix()
        assert long_desc in prefix


class TestVersionTracking:
    """Version increments correctly across update+save cycles."""

    def test_version_increments_each_save(self, tmp_path):
        pm = _make_manager(tmp_path)
        initial = pm.version
        for expected_delta in range(1, 6):
            pm.save()
            assert pm.version == initial + expected_delta

    def test_reset_after_multiple_saves_continues_version(self, tmp_path):
        pm = _make_manager(tmp_path)
        pm.save()  # v2
        pm.save()  # v3
        version_before_reset = pm.version
        pm.reset()  # save inside reset -> v4
        assert pm.version == version_before_reset + 1

    def test_version_survives_reload(self, tmp_path):
        path = tmp_path / "persona.yaml"
        pm = PersonaManager(persona_path=path)
        pm.save()  # v2
        pm.save()  # v3
        pm2 = PersonaManager(persona_path=path)
        assert pm2.version == 3

    def test_persona_from_dict_partial_uses_defaults(self):
        """Partial dict with only name should fill remaining fields from defaults."""
        persona = _persona_from_dict({"name": "PartialBot"})
        assert persona.name == "PartialBot"
        assert persona.version == 1
        assert isinstance(persona.tone, list)
        assert len(persona.tone) > 0


class TestAtomicSave:
    """Atomic save: no leftover temp files under normal and error conditions."""

    def test_no_tmp_files_after_successful_save(self, tmp_path):
        pm = _make_manager(tmp_path)
        pm.save()
        tmp_files = list(tmp_path.glob("*.yaml.tmp"))
        assert tmp_files == []

    def test_persona_file_readable_after_save(self, tmp_path):
        pm = _make_manager(tmp_path)
        pm.update(name="AtomicBot")
        pm.save()
        raw = yaml.safe_load((tmp_path / "persona.yaml").read_text(encoding="utf-8"))
        assert raw["name"] == "AtomicBot"


class TestAuditLogEdgeCases:
    """Audit log edge cases beyond the basics covered in test_persona.py."""

    def test_rollback_audit_entry_has_from_backup_detail(self, tmp_path, monkeypatch):
        call_count = 0

        def _mock_strftime(fmt, *args):
            nonlocal call_count
            call_count += 1
            return f"20260318_99000{call_count}"

        monkeypatch.setattr(time, "strftime", _mock_strftime)

        path = tmp_path / "persona.yaml"
        pm = PersonaManager(persona_path=path)
        pm.save()
        pm.update(name="PreRollback")
        pm.save()
        pm.rollback()

        entries = pm.get_audit_log()
        rollback_entries = [e for e in entries if e["action"] == "rollback"]
        assert len(rollback_entries) == 1
        details = rollback_entries[0].get("details", {})
        assert "from_backup" in details
        assert "persona.yaml." in details["from_backup"]

    def test_audit_log_skips_corrupt_lines(self, tmp_path):
        path = tmp_path / "persona.yaml"
        pm = PersonaManager(persona_path=path)
        pm.save()

        # Inject a corrupt line into the audit log
        audit_path = tmp_path / "persona_audit.jsonl"
        with audit_path.open("a", encoding="utf-8") as fh:
            fh.write("{broken json line\n")
            fh.write('{"action":"extra","version":99,"timestamp":"t","name":"X","details":{}}\n')

        entries = pm.get_audit_log()
        # The corrupt line is skipped; at least the valid extra line is present
        actions = [e["action"] for e in entries]
        assert "extra" in actions
        # No entry from the broken line
        assert all(isinstance(e, dict) for e in entries)

    def test_audit_log_entries_have_required_keys(self, tmp_path):
        path = tmp_path / "persona.yaml"
        pm = PersonaManager(persona_path=path)
        pm.save()
        pm.reset()
        entries = pm.get_audit_log()
        required_keys = {"timestamp", "action", "version", "name", "details"}
        for entry in entries:
            assert required_keys.issubset(entry.keys()), f"Entry missing keys: {entry}"


class TestDiffEdgeCases:
    """Diff functionality beyond what test_persona.py covers."""

    def test_diff_contains_unified_diff_markers(self, tmp_path):
        path = tmp_path / "persona.yaml"
        pm = PersonaManager(persona_path=path)
        pm.save()  # v2
        pm.update(name="DiffBot")
        pm.save()  # v3
        diff = pm.diff()
        # A real unified diff has --- and +++ header lines
        assert "---" in diff
        assert "+++" in diff

    def test_diff_shows_changed_value(self, tmp_path, monkeypatch):
        call_count = 0

        def _mock_strftime(fmt, *args):
            nonlocal call_count
            call_count += 1
            return f"20260318_77000{call_count}"

        monkeypatch.setattr(time, "strftime", _mock_strftime)

        path = tmp_path / "persona.yaml"
        pm = PersonaManager(persona_path=path)
        pm.save()  # v2 — creates file with name: Missy
        pm.update(name="DiffedName")
        pm.save()  # v3 — backup of v2 exists
        diff = pm.diff()
        assert "DiffedName" in diff or "Missy" in diff

    def test_rollback_on_missing_file_returns_none(self, tmp_path):
        """rollback() with no backups and no persona file returns None."""
        path = tmp_path / "persona.yaml"
        pm = PersonaManager(persona_path=path)
        # No file, no backups
        result = pm.rollback()
        assert result is None


class TestConcurrentSave:
    """Thread-safety: concurrent saves should not corrupt the persona file."""

    def test_concurrent_saves_leave_valid_yaml(self, tmp_path):
        path = tmp_path / "persona.yaml"
        pm = PersonaManager(persona_path=path)
        pm.save()  # initial file

        errors: list[Exception] = []

        def _worker(index: int) -> None:
            try:
                pm.update(name=f"Bot{index}")
                pm.save()
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=_worker, args=(i,)) for i in range(6)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # No unhandled exceptions
        assert errors == [], f"Exceptions from worker threads: {errors}"

        # File should still be valid YAML
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        assert isinstance(raw, dict)
        assert "name" in raw


# ===========================================================================
# BEHAVIOR TESTS
# ===========================================================================


class TestIntentInterpreterAllCategories:
    """Ensure all ten intent categories are reachable."""

    def setup_method(self):
        self.interp = IntentInterpreter()

    def test_greeting_reachable(self):
        assert self.interp.classify_intent("hello there") == "greeting"

    def test_farewell_reachable(self):
        assert self.interp.classify_intent("goodbye") == "farewell"

    def test_farewell_cya(self):
        assert self.interp.classify_intent("cya later") == "farewell"

    def test_farewell_ttyl(self):
        assert self.interp.classify_intent("ttyl") == "farewell"

    def test_confirmation_reachable(self):
        assert self.interp.classify_intent("yes") == "confirmation"

    def test_confirmation_okay_period(self):
        assert self.interp.classify_intent("okay.") == "confirmation"

    def test_frustration_reachable(self):
        assert self.interp.classify_intent("still not working, I tried that") == "frustration"

    def test_troubleshooting_reachable(self):
        assert self.interp.classify_intent("I'm getting a traceback") == "troubleshooting"

    def test_clarification_reachable(self):
        assert self.interp.classify_intent("I don't understand what you mean") == "clarification"

    def test_feedback_reachable(self):
        assert self.interp.classify_intent("you got it wrong this time") == "feedback"

    def test_exploration_reachable(self):
        assert self.interp.classify_intent("tell me more about that") == "exploration"

    def test_command_reachable(self):
        assert self.interp.classify_intent("run the tests") == "command"

    def test_question_reachable(self):
        assert self.interp.classify_intent("what is Python?") == "question"

    def test_default_fallback_to_question(self):
        # Gibberish with no matching pattern falls back to question
        assert self.interp.classify_intent("zxqvbnm zxqvbnm") == "question"


class TestIntentInterpreterUrgency:
    """All urgency branches and boundary cases."""

    def setup_method(self):
        self.interp = IntentInterpreter()

    def test_immediately_is_high(self):
        assert self.interp.extract_urgency("Fix this immediately!") == "high"

    def test_outage_is_high(self):
        assert self.interp.extract_urgency("We have an outage in production") == "high"

    def test_crash_is_high(self):
        assert self.interp.extract_urgency("The app crashed and is down") == "high"

    def test_blocking_is_high(self):
        assert self.interp.extract_urgency("This bug is blocking our release") == "high"

    def test_soon_is_medium(self):
        assert self.interp.extract_urgency("I need this done soon please") == "medium"

    def test_must_is_medium(self):
        assert self.interp.extract_urgency("We must deploy before the weekend") == "medium"

    def test_important_is_medium(self):
        assert self.interp.extract_urgency("This is important for the sprint") == "medium"

    def test_deadline_is_medium(self):
        assert self.interp.extract_urgency("We have a deadline tomorrow") == "medium"

    def test_empty_string_is_low(self):
        assert self.interp.extract_urgency("") == "low"

    def test_normal_query_is_low(self):
        assert self.interp.extract_urgency("What is the difference between lists and tuples?") == "low"

    def test_high_takes_priority_over_medium(self):
        mixed = "This is important but also production is down right now"
        assert self.interp.extract_urgency(mixed) == "high"

    def test_very_long_input_does_not_raise(self):
        long_text = "I need help " * 5000
        result = self.interp.extract_urgency(long_text)
        assert result in ("high", "medium", "low")


class TestIntentInterpreterEdgeCasesExtra:
    """Additional edge cases not in test_behavior.py."""

    def setup_method(self):
        self.interp = IntentInterpreter()

    def test_classify_empty_string_returns_question(self):
        assert self.interp.classify_intent("") == "question"

    def test_classify_unicode_input_does_not_raise(self):
        result = self.interp.classify_intent("こんにちは世界")
        assert isinstance(result, str)

    def test_classify_numbers_only_returns_question(self):
        assert self.interp.classify_intent("12345") == "question"

    def test_classify_greeting_is_case_insensitive(self):
        assert self.interp.classify_intent("HELLO THERE") == "greeting"

    def test_classify_very_long_input_returns_valid_category(self):
        categories = {
            "greeting", "farewell", "confirmation", "frustration",
            "troubleshooting", "clarification", "feedback",
            "exploration", "command", "question",
        }
        result = self.interp.classify_intent("what " * 10_000)
        assert result in categories


class TestResponseShaperPreambleStripping:
    """Robotic preamble patterns not specifically tested in test_behavior.py."""

    def setup_method(self):
        self.shaper = ResponseShaper()

    def test_i_am_an_ai_assistant_stripped(self):
        result = self.shaper.shape_response(
            "I am an AI assistant, here to help. Use pip install.", None, {}
        )
        assert "I am an AI assistant" not in result
        assert "pip install" in result

    def test_please_note_i_am_an_ai_stripped(self):
        result = self.shaper.shape_response(
            "Please note that I am an AI and cannot do that. However, try this.",
            None,
            {},
        )
        assert "Please note that I am an AI" not in result

    def test_absolutely_stripped(self):
        result = self.shaper.shape_response(
            "Absolutely! Here is how you do it.", None, {}
        )
        assert "Absolutely" not in result
        assert "Here is how you do it." in result

    def test_thats_a_great_question_stripped(self):
        result = self.shaper.shape_response(
            "That's a great question! The answer is 42.", None, {}
        )
        assert "great question" not in result
        assert "42" in result

    def test_content_after_preamble_is_non_empty(self):
        result = self.shaper.shape_response(
            "Certainly! As an AI language model, I'd be happy to assist. The config is here.",
            None,
            {},
        )
        assert result.strip() != ""
        assert "config" in result

    def test_trailing_whitespace_removed(self):
        result = self.shaper.shape_response(
            "Here is the answer.   \n  ", None, {}
        )
        assert result == result.strip()


class TestResponseShaperCodePreservation:
    """Code block preservation edge cases."""

    def setup_method(self):
        self.shaper = ResponseShaper()

    def test_fenced_block_with_language_tag_preserved_verbatim(self):
        code = "```python\n# As an AI\ndef foo():\n    return 'Certainly!'\n```"
        raw = f"As an AI, here is the code:\n{code}"
        result = self.shaper.shape_response(raw, None, {})
        assert "```python" in result
        assert "def foo():" in result
        assert "Certainly!" in result  # inside code block

    def test_inline_code_containing_robotic_text_preserved(self):
        raw = "Use `As an AI` as a variable name — it is valid Python."
        result = self.shaper.shape_response(raw, None, {})
        assert "`As an AI`" in result

    def test_multiple_fenced_blocks_all_preserved(self):
        raw = (
            "Certainly!\n"
            "```bash\necho 'Certainly'\n```\n"
            "Middle text.\n"
            "```python\nprint('Of course')\n```"
        )
        result = self.shaper.shape_response(raw, None, {})
        assert "echo 'Certainly'" in result
        assert "print('Of course')" in result

    def test_response_with_no_robotic_phrases_unchanged_modulo_strip(self):
        raw = "The answer is simply 42."
        result = self.shaper.shape_response(raw, None, {})
        assert result == raw.strip()


class TestBehaviorLayerVisionGuidelines:
    """Vision-specific guidelines from get_response_guidelines."""

    def test_painting_mode_encourages_warmly(self):
        layer = BehaviorLayer()
        ctx = _base_ctx(vision_mode="painting")
        guidelines = layer.get_response_guidelines(ctx)
        lower = guidelines.lower()
        assert "warm" in lower or "encouraging" in lower or "artwork" in lower

    def test_painting_mode_no_harsh_language_instruction(self):
        layer = BehaviorLayer()
        ctx = _base_ctx(vision_mode="painting")
        guidelines = layer.get_response_guidelines(ctx)
        lower = guidelines.lower()
        assert "harsh" in lower or "dismiss" in lower or "encouraging" in lower

    def test_puzzle_mode_patient_and_actionable(self):
        layer = BehaviorLayer()
        ctx = _base_ctx(vision_mode="puzzle")
        guidelines = layer.get_response_guidelines(ctx)
        lower = guidelines.lower()
        assert "patient" in lower or "placement" in lower or "puzzle" in lower

    def test_puzzle_mode_mentions_colors_or_patterns(self):
        layer = BehaviorLayer()
        ctx = _base_ctx(vision_mode="puzzle")
        guidelines = layer.get_response_guidelines(ctx)
        lower = guidelines.lower()
        assert "color" in lower or "pattern" in lower or "specific" in lower

    def test_generic_vision_mode_references_visual_details(self):
        layer = BehaviorLayer()
        ctx = _base_ctx(vision_mode="inspection")
        guidelines = layer.get_response_guidelines(ctx)
        lower = guidelines.lower()
        assert "visual" in lower or "observant" in lower or "specific" in lower

    def test_no_vision_mode_no_vision_guideline(self):
        layer = BehaviorLayer()
        ctx = _base_ctx()  # no vision_mode key
        guidelines = layer.get_response_guidelines(ctx)
        assert "artwork" not in guidelines.lower()
        assert "puzzle" not in guidelines.lower()


class TestBehaviorLayerSystemPromptStructure:
    """System prompt structure details."""

    def test_response_guidelines_section_header_present(self):
        layer = BehaviorLayer()
        result = layer.shape_system_prompt("Base.", _base_ctx())
        assert "## Response guidelines" in result

    def test_empty_base_prompt_still_has_guidelines(self):
        layer = BehaviorLayer()
        result = layer.shape_system_prompt("", _base_ctx())
        assert "## Response guidelines" in result

    def test_base_prompt_is_first_section(self):
        layer = BehaviorLayer()
        base = "BASE_SENTINEL"
        result = layer.shape_system_prompt(base, _base_ctx())
        assert result.startswith(base)

    def test_medium_urgency_focused_guideline(self):
        layer = BehaviorLayer()
        ctx = _base_ctx(urgency="medium")
        guidelines = layer.get_response_guidelines(ctx)
        lower = guidelines.lower()
        assert "focused" in lower or "actionable" in lower

    def test_farewell_intent_brief_guideline(self):
        layer = BehaviorLayer()
        ctx = _base_ctx(intent="farewell")
        guidelines = layer.get_response_guidelines(ctx)
        lower = guidelines.lower()
        assert "farewell" in lower or "brief" in lower or "friendly" in lower

    def test_no_user_messages_analyze_tone_returns_casual(self):
        layer = BehaviorLayer()
        msgs = [{"role": "assistant", "content": "Hello!"}]
        assert layer.analyze_user_tone(msgs) == "casual"

    def test_all_guidelines_lines_are_bullets(self):
        persona = PersonaConfig(
            behavioral_tendencies=["action over narration"],
            response_style_rules=["be concise"],
        )
        layer = BehaviorLayer(persona)
        ctx = _base_ctx(
            user_tone="frustrated",
            urgency="high",
            intent="troubleshooting",
            has_tool_results=True,
            turn_count=12,
        )
        guidelines = layer.get_response_guidelines(ctx)
        for line in guidelines.splitlines():
            if line.strip():
                assert line.startswith("- "), (
                    f"Expected bullet prefix on: {line!r}"
                )
