"""Tests for missy.agent.behavior — humanistic behavior layer."""

from __future__ import annotations

import pytest

from missy.agent.behavior import BehaviorLayer, IntentInterpreter, ResponseShaper
from missy.agent.persona import PersonaConfig


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------


def _make_persona(
    name: str = "Missy",
    tone: str | list[str] = "warm",
    behavioral_tendencies: list[str] | None = None,
    response_style_rules: list[str] | None = None,
    boundaries: list[str] | None = None,
    identity_description: str = "A helpful assistant.",
) -> PersonaConfig:
    return PersonaConfig(
        name=name,
        tone=[tone] if isinstance(tone, str) else tone,
        identity_description=identity_description,
        behavioral_tendencies=behavioral_tendencies or [],
        response_style_rules=response_style_rules or [],
        boundaries=boundaries or [],
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


# ---------------------------------------------------------------------------
# BehaviorLayer — shape_system_prompt
# ---------------------------------------------------------------------------


class TestShapeSystemPromptWithPersona:
    def test_includes_persona_name(self):
        persona = _make_persona(name="Missy")
        layer = BehaviorLayer(persona)
        result = layer.shape_system_prompt("You are helpful.", _base_ctx())
        assert "Missy" in result

    def test_includes_persona_identity_description(self):
        persona = _make_persona(identity_description="A security-focused AI.")
        layer = BehaviorLayer(persona)
        result = layer.shape_system_prompt("Base.", _base_ctx())
        assert "A security-focused AI." in result

    def test_includes_persona_tone(self):
        persona = _make_persona(tone="warm")
        layer = BehaviorLayer(persona)
        result = layer.shape_system_prompt("Base.", _base_ctx())
        assert "warm" in result

    def test_base_prompt_always_present(self):
        persona = _make_persona()
        layer = BehaviorLayer(persona)
        base = "You are a helpful assistant."
        result = layer.shape_system_prompt(base, _base_ctx())
        assert base in result

    def test_persona_boundaries_included(self):
        persona = _make_persona(boundaries=["Never expose secrets"])
        layer = BehaviorLayer(persona)
        result = layer.shape_system_prompt("Base.", _base_ctx())
        assert "Never expose secrets" in result

    def test_persona_behavioral_tendencies_in_guidelines(self):
        tendency = "prefers action over narration"
        persona = _make_persona(behavioral_tendencies=[tendency])
        layer = BehaviorLayer(persona)
        result = layer.shape_system_prompt("Base.", _base_ctx())
        assert tendency in result

    def test_persona_response_style_rules_in_guidelines(self):
        rule = "Use bullet points for lists"
        persona = _make_persona(response_style_rules=[rule])
        layer = BehaviorLayer(persona)
        result = layer.shape_system_prompt("Base.", _base_ctx())
        assert rule in result


class TestShapeSystemPromptWithoutPersona:
    def test_base_prompt_preserved(self):
        layer = BehaviorLayer()
        base = "You are helpful."
        result = layer.shape_system_prompt(base, _base_ctx())
        assert base in result

    def test_no_persona_section_when_no_persona(self):
        layer = BehaviorLayer()
        result = layer.shape_system_prompt("Base.", _base_ctx())
        assert "## Persona" not in result

    def test_none_context_does_not_raise(self):
        layer = BehaviorLayer()
        result = layer.shape_system_prompt("Base.", None)
        assert "Base." in result

    def test_returns_string(self):
        layer = BehaviorLayer()
        result = layer.shape_system_prompt("Base.", _base_ctx())
        assert isinstance(result, str)


# ---------------------------------------------------------------------------
# BehaviorLayer — analyze_user_tone
# ---------------------------------------------------------------------------


class TestAnalyzeUserToneCasual:
    def test_hey_is_casual(self):
        layer = BehaviorLayer()
        msgs = _user_messages(
            "hey can you help me with something cool"
            " thx btw ya ngl tbh fyi"
        )
        assert layer.analyze_user_tone(msgs) == "casual"

    def test_lol_signals_casual(self):
        layer = BehaviorLayer()
        # Message must exceed 8 words so the "brief" branch is not triggered first.
        msgs = _user_messages("lol that was so funny haha cool thanks man ngl")
        assert layer.analyze_user_tone(msgs) == "casual"

    def test_cool_signals_casual(self):
        layer = BehaviorLayer()
        # Message must exceed 8 words so the "brief" branch is not triggered first.
        msgs = _user_messages("cool awesome thanks ya ngl tbh imo it was great")
        assert layer.analyze_user_tone(msgs) == "casual"


class TestAnalyzeUserToneFormal:
    def test_please_kindly_is_formal(self):
        layer = BehaviorLayer()
        # Long messages to avoid triggering "brief" tone
        msgs = _user_messages(
            "Please kindly assist me regarding this matter and furthermore "
            "I would appreciate your thorough response accordingly therefore "
            "sincerely I ask"
        )
        assert layer.analyze_user_tone(msgs) == "formal"

    def test_would_you_is_formal(self):
        layer = BehaviorLayer()
        msgs = _user_messages(
            "Would you kindly help me please regarding this furthermore "
            "therefore accordingly henceforth I appreciate your assistance"
        )
        assert layer.analyze_user_tone(msgs) == "formal"


class TestAnalyzeUserToneFrustrated:
    def test_wrong_triggers_frustrated(self):
        layer = BehaviorLayer()
        msgs = _user_messages("that is wrong it still doesn't work")
        assert layer.analyze_user_tone(msgs) == "frustrated"

    def test_broken_triggers_frustrated(self):
        layer = BehaviorLayer()
        msgs = _user_messages("it's broken again ugh why is this useless")
        assert layer.analyze_user_tone(msgs) == "frustrated"

    def test_doesnt_work_triggers_frustrated(self):
        layer = BehaviorLayer()
        msgs = _user_messages("it doesn't work I've tried everything still nothing works")
        assert layer.analyze_user_tone(msgs) == "frustrated"

    def test_frustration_pattern_overrides_other_tones(self):
        # Even a formal-sounding message with frustration keywords should
        # return "frustrated" since frustration takes priority.
        layer = BehaviorLayer()
        msgs = _user_messages("Please note this still doesn't work regardless of my attempts")
        assert layer.analyze_user_tone(msgs) == "frustrated"


class TestAnalyzeUserToneTechnical:
    def test_function_api_database_is_technical(self):
        layer = BehaviorLayer()
        msgs = _user_messages(
            "I need to call the function via the API endpoint using "
            "the database query schema config yaml json async await thread"
        )
        assert layer.analyze_user_tone(msgs) == "technical"

    def test_docker_kubernetes_is_technical(self):
        layer = BehaviorLayer()
        msgs = _user_messages(
            "How do I deploy a docker container to kubernetes with ssl tls "
            "auth oauth pipeline ci config script module import library"
        )
        assert layer.analyze_user_tone(msgs) == "technical"


class TestAnalyzeUserToneEdgeCases:
    def test_empty_messages_returns_casual(self):
        layer = BehaviorLayer()
        assert layer.analyze_user_tone([]) == "casual"

    def test_no_user_messages_returns_casual(self):
        layer = BehaviorLayer()
        msgs = [{"role": "assistant", "content": "Hello there!"}]
        assert layer.analyze_user_tone(msgs) == "casual"

    def test_short_messages_return_brief(self):
        layer = BehaviorLayer()
        # Single short message (< 8 words) with no tone signals
        msgs = _user_messages("fix this")
        assert layer.analyze_user_tone(msgs) == "brief"

    def test_long_messages_return_verbose(self):
        layer = BehaviorLayer()
        # > 40 words per message and no strong tone signals
        long_text = " ".join(["word"] * 50)
        msgs = _user_messages(long_text)
        assert layer.analyze_user_tone(msgs) == "verbose"

    def test_only_assistant_messages_ignored(self):
        layer = BehaviorLayer()
        msgs = [
            {"role": "assistant", "content": "I can help."},
            {"role": "assistant", "content": "Sure thing."},
        ]
        assert layer.analyze_user_tone(msgs) == "casual"

    def test_no_matching_tone_signals_returns_casual(self):
        layer = BehaviorLayer()
        # Message long enough to avoid "brief" but no keyword matches
        msgs = _user_messages(
            "The rain in spain falls mainly on the plain every single year"
        )
        assert layer.analyze_user_tone(msgs) == "casual"


# ---------------------------------------------------------------------------
# BehaviorLayer — get_response_guidelines
# ---------------------------------------------------------------------------


class TestGetResponseGuidelines:
    def test_returns_non_empty_string(self):
        layer = BehaviorLayer()
        result = layer.get_response_guidelines(_base_ctx(user_tone="casual"))
        assert isinstance(result, str)
        assert len(result) > 0

    def test_high_urgency_included(self):
        layer = BehaviorLayer()
        result = layer.get_response_guidelines(_base_ctx(urgency="high"))
        assert "time pressure" in result.lower() or "preamble" in result.lower()

    def test_tool_results_guidance_included(self):
        layer = BehaviorLayer()
        result = layer.get_response_guidelines(_base_ctx(has_tool_results=True))
        assert "tool" in result.lower()

    def test_frustration_intent_guidance(self):
        layer = BehaviorLayer()
        result = layer.get_response_guidelines(_base_ctx(intent="frustration"))
        assert "empathy" in result.lower() or "acknowledge" in result.lower()

    def test_greeting_intent_guidance(self):
        layer = BehaviorLayer()
        result = layer.get_response_guidelines(_base_ctx(intent="greeting"))
        assert "warmly" in result.lower() or "greeting" in result.lower()

    def test_exploration_intent_guidance(self):
        layer = BehaviorLayer()
        result = layer.get_response_guidelines(_base_ctx(intent="exploration"))
        assert "ideas" in result.lower() or "suggestions" in result.lower() or "proactive" in result.lower()

    def test_code_topic_includes_snippet_guidance(self):
        layer = BehaviorLayer()
        result = layer.get_response_guidelines(_base_ctx(topic="write a function"))
        assert "code" in result.lower() or "snippet" in result.lower()

    def test_long_conversation_concise_hint(self):
        layer = BehaviorLayer()
        result = layer.get_response_guidelines(_base_ctx(turn_count=15))
        assert "concise" in result.lower() or "long" in result.lower()

    def test_all_lines_prefixed_with_bullet(self):
        layer = BehaviorLayer()
        result = layer.get_response_guidelines(_base_ctx(user_tone="casual"))
        for line in result.splitlines():
            if line.strip():
                assert line.startswith("- "), f"Expected bullet prefix on: {line!r}"


# ---------------------------------------------------------------------------
# BehaviorLayer — should_be_concise
# ---------------------------------------------------------------------------


class TestShouldBeConcise:
    def test_high_turn_count_is_concise(self):
        layer = BehaviorLayer()
        assert layer.should_be_concise({"turn_count": 10}) is True

    def test_above_threshold_is_concise(self):
        layer = BehaviorLayer()
        assert layer.should_be_concise({"turn_count": 25}) is True

    def test_below_threshold_is_not_concise(self):
        layer = BehaviorLayer()
        assert layer.should_be_concise({"turn_count": 5}) is False

    def test_brief_tone_is_concise(self):
        layer = BehaviorLayer()
        assert layer.should_be_concise({"user_tone": "brief", "turn_count": 1}) is True

    def test_high_urgency_is_concise(self):
        layer = BehaviorLayer()
        assert layer.should_be_concise({"urgency": "high", "turn_count": 1}) is True

    def test_normal_context_not_concise(self):
        layer = BehaviorLayer()
        assert layer.should_be_concise({"user_tone": "casual", "turn_count": 3, "urgency": "low"}) is False

    def test_empty_context_not_concise(self):
        layer = BehaviorLayer()
        assert layer.should_be_concise({}) is False


# ---------------------------------------------------------------------------
# BehaviorLayer — get_tone_adaptation
# ---------------------------------------------------------------------------


class TestGetToneAdaptation:
    def test_casual_returns_non_empty(self):
        layer = BehaviorLayer()
        assert layer.get_tone_adaptation("casual") != ""

    def test_formal_returns_non_empty(self):
        layer = BehaviorLayer()
        assert layer.get_tone_adaptation("formal") != ""

    def test_frustrated_returns_non_empty(self):
        layer = BehaviorLayer()
        assert layer.get_tone_adaptation("frustrated") != ""

    def test_technical_returns_non_empty(self):
        layer = BehaviorLayer()
        assert layer.get_tone_adaptation("technical") != ""

    def test_brief_returns_non_empty(self):
        layer = BehaviorLayer()
        assert layer.get_tone_adaptation("brief") != ""

    def test_verbose_returns_non_empty(self):
        layer = BehaviorLayer()
        assert layer.get_tone_adaptation("verbose") != ""

    def test_unknown_tone_returns_empty_string(self):
        layer = BehaviorLayer()
        assert layer.get_tone_adaptation("nonexistent_tone") == ""

    def test_returns_string_type(self):
        layer = BehaviorLayer()
        result = layer.get_tone_adaptation("casual")
        assert isinstance(result, str)


# ---------------------------------------------------------------------------
# IntentInterpreter — classify_intent
# ---------------------------------------------------------------------------


class TestClassifyIntentGreeting:
    def test_hello_is_greeting(self):
        interpreter = IntentInterpreter()
        assert interpreter.classify_intent("hello") == "greeting"

    def test_hi_there_is_greeting(self):
        interpreter = IntentInterpreter()
        assert interpreter.classify_intent("hi there") == "greeting"

    def test_hey_is_greeting(self):
        interpreter = IntentInterpreter()
        assert interpreter.classify_intent("hey!") == "greeting"

    def test_good_morning_is_greeting(self):
        interpreter = IntentInterpreter()
        assert interpreter.classify_intent("good morning") == "greeting"

    def test_howdy_is_greeting(self):
        interpreter = IntentInterpreter()
        assert interpreter.classify_intent("howdy partner") == "greeting"


class TestClassifyIntentCommand:
    def test_run_this_is_command(self):
        interpreter = IntentInterpreter()
        assert interpreter.classify_intent("run this script") == "command"

    def test_execute_is_command(self):
        interpreter = IntentInterpreter()
        assert interpreter.classify_intent("execute the deployment") == "command"

    def test_delete_is_command(self):
        interpreter = IntentInterpreter()
        assert interpreter.classify_intent("delete the old files") == "command"

    def test_create_is_command(self):
        interpreter = IntentInterpreter()
        assert interpreter.classify_intent("create a new config file") == "command"

    def test_install_is_command(self):
        interpreter = IntentInterpreter()
        assert interpreter.classify_intent("install the package") == "command"


class TestClassifyIntentQuestion:
    def test_what_is_is_question(self):
        interpreter = IntentInterpreter()
        assert interpreter.classify_intent("what is the best practice?") == "question"

    def test_how_do_i_is_question(self):
        interpreter = IntentInterpreter()
        assert interpreter.classify_intent("how do I restart nginx?") == "question"

    def test_trailing_question_mark_is_question(self):
        interpreter = IntentInterpreter()
        assert interpreter.classify_intent("can you help me?") == "question"

    def test_is_it_possible_is_question(self):
        interpreter = IntentInterpreter()
        assert interpreter.classify_intent("is it possible to do this?") == "question"


class TestClassifyIntentFrustration:
    def test_still_not_working_is_frustration(self):
        interpreter = IntentInterpreter()
        assert interpreter.classify_intent("still not working, I tried that already") == "frustration"

    def test_useless_is_frustration(self):
        interpreter = IntentInterpreter()
        assert interpreter.classify_intent("this is useless nothing works") == "frustration"

    def test_same_error_is_frustration(self):
        interpreter = IntentInterpreter()
        assert interpreter.classify_intent("same error again I already tried this") == "frustration"


class TestClassifyIntentFarewell:
    def test_bye_is_farewell(self):
        interpreter = IntentInterpreter()
        assert interpreter.classify_intent("bye") == "farewell"

    def test_see_you_is_farewell(self):
        interpreter = IntentInterpreter()
        assert interpreter.classify_intent("see you later") == "farewell"

    def test_goodbye_is_farewell(self):
        interpreter = IntentInterpreter()
        assert interpreter.classify_intent("goodbye!") == "farewell"


class TestClassifyIntentClarification:
    def test_what_do_you_mean_is_clarification(self):
        interpreter = IntentInterpreter()
        assert interpreter.classify_intent("what do you mean by that?") == "clarification"

    def test_can_you_clarify_is_clarification(self):
        interpreter = IntentInterpreter()
        assert interpreter.classify_intent("can you clarify what you mean?") == "clarification"


class TestClassifyIntentExploration:
    def test_tell_me_more_is_exploration(self):
        interpreter = IntentInterpreter()
        assert interpreter.classify_intent("tell me more about that topic") == "exploration"

    def test_suggest_is_exploration(self):
        interpreter = IntentInterpreter()
        assert interpreter.classify_intent("could you suggest some options?") == "exploration"


class TestClassifyIntentFeedback:
    def test_thats_wrong_is_feedback(self):
        interpreter = IntentInterpreter()
        assert interpreter.classify_intent("that's wrong, I needed something different") == "feedback"

    def test_not_quite_is_feedback(self):
        interpreter = IntentInterpreter()
        assert interpreter.classify_intent("not quite what I was looking for") == "feedback"


# ---------------------------------------------------------------------------
# IntentInterpreter — extract_urgency
# ---------------------------------------------------------------------------


class TestExtractUrgency:
    def test_asap_is_high(self):
        interpreter = IntentInterpreter()
        assert interpreter.extract_urgency("I need this fixed asap") == "high"

    def test_production_down_is_high(self):
        interpreter = IntentInterpreter()
        assert interpreter.extract_urgency("production down urgent outage") == "high"

    def test_critical_is_high(self):
        interpreter = IntentInterpreter()
        assert interpreter.extract_urgency("critical issue, system crash") == "high"

    def test_not_working_is_high(self):
        interpreter = IntentInterpreter()
        assert interpreter.extract_urgency("it's not working right now") == "high"

    def test_today_is_medium(self):
        interpreter = IntentInterpreter()
        assert interpreter.extract_urgency("I need this done today please") == "medium"

    def test_deadline_is_medium(self):
        interpreter = IntentInterpreter()
        assert interpreter.extract_urgency("there's a deadline for this") == "medium"

    def test_normal_text_is_low(self):
        interpreter = IntentInterpreter()
        assert interpreter.extract_urgency("how do I configure the yaml file?") == "low"

    def test_empty_string_is_low(self):
        interpreter = IntentInterpreter()
        assert interpreter.extract_urgency("") == "low"

    def test_casual_chat_is_low(self):
        interpreter = IntentInterpreter()
        assert interpreter.extract_urgency("hey, what's the weather like?") == "low"


# ---------------------------------------------------------------------------
# ResponseShaper — shape_response
# ---------------------------------------------------------------------------


class TestShapeResponseRemovesRoboticPhrases:
    def test_removes_as_an_ai(self):
        shaper = ResponseShaper()
        result = shaper.shape_response(
            "As an AI, I can help you with that.", persona=None, context={}
        )
        assert "As an AI" not in result
        assert "I can help you with that." in result or result.strip() != ""

    def test_removes_as_a_large_language_model(self):
        shaper = ResponseShaper()
        result = shaper.shape_response(
            "As a large language model, here is the answer.", persona=None, context={}
        )
        assert "large language model" not in result

    def test_removes_certainly(self):
        shaper = ResponseShaper()
        result = shaper.shape_response(
            "Certainly! Here is what you need.", persona=None, context={}
        )
        assert "Certainly" not in result
        assert "Here is what you need." in result

    def test_removes_of_course(self):
        shaper = ResponseShaper()
        result = shaper.shape_response(
            "Of course! I'll help you with that.", persona=None, context={}
        )
        assert "Of course" not in result

    def test_removes_great_question(self):
        shaper = ResponseShaper()
        result = shaper.shape_response(
            "Great question! The answer is 42.", persona=None, context={}
        )
        assert "Great question" not in result
        assert "42" in result

    def test_removes_id_be_happy_to(self):
        shaper = ResponseShaper()
        result = shaper.shape_response(
            "I'd be happy to help you with that.", persona=None, context={}
        )
        assert "I'd be happy to" not in result

    def test_removes_i_am_here_to_help(self):
        shaper = ResponseShaper()
        result = shaper.shape_response(
            "I'm here to help. The answer is simple.", persona=None, context={}
        )
        assert "I'm here to help" not in result

    def test_removes_as_your_assistant(self):
        shaper = ResponseShaper()
        result = shaper.shape_response(
            "As your assistant, I recommend this approach.", persona=None, context={}
        )
        assert "As your assistant" not in result

    def test_multiple_robotic_phrases_removed(self):
        shaper = ResponseShaper()
        raw = (
            "Certainly! As an AI language model, I'd be happy to help you. "
            "Great question! The answer involves configuration."
        )
        result = shaper.shape_response(raw, persona=None, context={})
        assert "Certainly" not in result
        assert "As an AI language model" not in result
        assert "Great question" not in result
        assert "configuration" in result


class TestShapeResponsePreservesCodeBlocks:
    def test_fenced_code_block_preserved(self):
        shaper = ResponseShaper()
        raw = (
            "As an AI, here is the code:\n"
            "```python\n"
            "def hello():\n"
            "    print('As an AI, hello')\n"
            "```\n"
            "That's the function."
        )
        result = shaper.shape_response(raw, persona=None, context={})
        assert "As an AI, hello" in result
        assert "```python" in result
        assert "def hello():" in result

    def test_inline_code_preserved(self):
        shaper = ResponseShaper()
        raw = "As an AI, use `As an AI trick` to do it."
        result = shaper.shape_response(raw, persona=None, context={})
        assert "`As an AI trick`" in result

    def test_multiple_code_blocks_all_preserved(self):
        shaper = ResponseShaper()
        raw = (
            "Certainly!\n"
            "```bash\necho 'Certainly block one'\n```\n"
            "And also:\n"
            "```yaml\nkey: Certainly block two\n```"
        )
        result = shaper.shape_response(raw, persona=None, context={})
        assert "Certainly block one" in result
        assert "Certainly block two" in result
        assert "```bash" in result
        assert "```yaml" in result


class TestShapeResponseEdgeCases:
    def test_empty_string_returns_empty(self):
        shaper = ResponseShaper()
        assert shaper.shape_response("", persona=None, context={}) == ""

    def test_clean_response_unchanged(self):
        shaper = ResponseShaper()
        raw = "The configuration file lives at ~/.missy/config.yaml."
        result = shaper.shape_response(raw, persona=None, context={})
        assert "~/.missy/config.yaml" in result

    def test_works_with_empty_persona(self):
        shaper = ResponseShaper()
        persona = PersonaConfig(
            name="",
            tone=[],
            identity_description="",
            behavioral_tendencies=[],
            response_style_rules=[],
            boundaries=[],
        )
        raw = "As an AI, here is the answer."
        result = shaper.shape_response(raw, persona=persona, context={})
        assert "As an AI" not in result

    def test_result_is_stripped(self):
        shaper = ResponseShaper()
        raw = "  \n\nAs an AI, hello.\n\n  "
        result = shaper.shape_response(raw, persona=None, context={})
        assert result == result.strip()

    def test_multiple_blank_lines_collapsed(self):
        shaper = ResponseShaper()
        raw = "First paragraph.\n\n\n\nSecond paragraph."
        result = shaper.shape_response(raw, persona=None, context={})
        assert "\n\n\n" not in result


# ---------------------------------------------------------------------------
# ResponseShaper — detect_robotic_patterns
# ---------------------------------------------------------------------------


class TestDetectRoboticPatterns:
    def test_detects_as_an_ai(self):
        shaper = ResponseShaper()
        found = shaper.detect_robotic_patterns("As an AI, I can assist.")
        assert len(found) > 0
        assert any("AI" in phrase for phrase in found)

    def test_detects_certainly(self):
        shaper = ResponseShaper()
        found = shaper.detect_robotic_patterns("Certainly! Here is the answer.")
        assert len(found) > 0

    def test_detects_great_question(self):
        shaper = ResponseShaper()
        found = shaper.detect_robotic_patterns("Great question! Let me explain.")
        assert len(found) > 0

    def test_detects_id_be_happy_to(self):
        shaper = ResponseShaper()
        found = shaper.detect_robotic_patterns("I'd be happy to help you with that.")
        assert len(found) > 0

    def test_clean_text_returns_empty_list(self):
        shaper = ResponseShaper()
        found = shaper.detect_robotic_patterns("The config lives at ~/.missy/config.yaml.")
        assert found == []

    def test_returns_list_type(self):
        shaper = ResponseShaper()
        found = shaper.detect_robotic_patterns("Hello world.")
        assert isinstance(found, list)

    def test_multiple_patterns_all_detected(self):
        shaper = ResponseShaper()
        text = (
            "Certainly! As an AI, I'd be happy to help. "
            "Great question! Of course I can assist."
        )
        found = shaper.detect_robotic_patterns(text)
        assert len(found) >= 3

    def test_does_not_modify_input(self):
        shaper = ResponseShaper()
        original = "As an AI language model, I can help."
        shaper.detect_robotic_patterns(original)
        assert original == "As an AI language model, I can help."
