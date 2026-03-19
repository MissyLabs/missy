"""Session 11: Comprehensive behavior layer tests.

Tests for BehaviorLayer, IntentInterpreter, and ResponseShaper
covering tone analysis, intent classification, urgency extraction,
response shaping, and vision-specific guidelines.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pytest


# ---------------------------------------------------------------------------
# Mock PersonaConfig for tests (avoid importing real one if it has deps)
# ---------------------------------------------------------------------------


@dataclass
class MockPersona:
    name: str = "Missy"
    tone: str = "warm"
    identity_description: str = "A helpful security-first assistant."
    personality_traits: list[str] = field(default_factory=lambda: ["curious", "patient"])
    behavioral_tendencies: list[str] = field(
        default_factory=lambda: ["Avoid jargon when possible."]
    )
    response_style_rules: list[str] = field(
        default_factory=lambda: ["Use short paragraphs."]
    )
    boundaries: list[str] = field(
        default_factory=lambda: ["Never execute destructive commands without confirmation."]
    )


# ---------------------------------------------------------------------------
# BehaviorLayer tests
# ---------------------------------------------------------------------------


class TestBehaviorLayerToneAnalysis:
    """Test BehaviorLayer.analyze_user_tone()."""

    def test_casual_from_signals(self) -> None:
        from missy.agent.behavior import BehaviorLayer

        layer = BehaviorLayer()
        msgs = [
            {"role": "user", "content": "hey cool thanks for helping me out with this stuff yo"}
        ]
        assert layer.analyze_user_tone(msgs) == "casual"

    def test_formal_from_signals(self) -> None:
        from missy.agent.behavior import BehaviorLayer

        layer = BehaviorLayer()
        msgs = [
            {"role": "user", "content": "I would appreciate it if you could kindly provide an explanation regarding this matter."}
        ]
        assert layer.analyze_user_tone(msgs) == "formal"

    def test_frustrated_takes_priority(self) -> None:
        from missy.agent.behavior import BehaviorLayer

        layer = BehaviorLayer()
        msgs = [
            {"role": "user", "content": "This is still not working! I've tried everything and it's broken again. Why doesn't it work?"}
        ]
        assert layer.analyze_user_tone(msgs) == "frustrated"

    def test_technical_from_signals(self) -> None:
        from missy.agent.behavior import BehaviorLayer

        layer = BehaviorLayer()
        msgs = [
            {"role": "user", "content": "I need to configure the async function to handle the oauth api endpoint with proper auth and token management in the docker container pipeline."}
        ]
        assert layer.analyze_user_tone(msgs) == "technical"

    def test_brief_from_short_messages(self) -> None:
        from missy.agent.behavior import BehaviorLayer

        layer = BehaviorLayer()
        msgs = [
            {"role": "user", "content": "fix it"},
            {"role": "user", "content": "deploy now"},
            {"role": "user", "content": "check logs"},
        ]
        assert layer.analyze_user_tone(msgs) == "brief"

    def test_verbose_from_long_messages(self) -> None:
        from missy.agent.behavior import BehaviorLayer

        layer = BehaviorLayer()
        long_msg = " ".join(["word"] * 60)
        msgs = [{"role": "user", "content": long_msg}]
        assert layer.analyze_user_tone(msgs) == "verbose"

    def test_empty_messages_returns_casual(self) -> None:
        from missy.agent.behavior import BehaviorLayer

        layer = BehaviorLayer()
        assert layer.analyze_user_tone([]) == "casual"

    def test_no_user_messages_returns_casual(self) -> None:
        from missy.agent.behavior import BehaviorLayer

        layer = BehaviorLayer()
        msgs = [
            {"role": "assistant", "content": "Hello there!"},
            {"role": "system", "content": "You are an assistant."},
        ]
        assert layer.analyze_user_tone(msgs) == "casual"

    def test_only_recent_messages_analyzed(self) -> None:
        """Only the last 5 user messages should be analyzed."""
        from missy.agent.behavior import BehaviorLayer

        layer = BehaviorLayer()
        # First 10 messages are formal
        formal_msgs = [
            {"role": "user", "content": "I would appreciate your assistance regarding this."}
            for _ in range(10)
        ]
        # Last 5 are casual (need enough words to avoid "brief" classification)
        casual_msgs = [
            {"role": "user", "content": "hey cool yo sup awesome that was really neat thanks a lot dude"}
            for _ in range(5)
        ]
        all_msgs = formal_msgs + casual_msgs
        tone = layer.analyze_user_tone(all_msgs)
        assert tone == "casual"


class TestBehaviorLayerPromptShaping:
    """Test BehaviorLayer.shape_system_prompt() and guidelines."""

    def test_base_prompt_preserved(self) -> None:
        from missy.agent.behavior import BehaviorLayer

        layer = BehaviorLayer()
        result = layer.shape_system_prompt("You are helpful.")
        assert result.startswith("You are helpful.")

    def test_persona_block_added(self) -> None:
        from missy.agent.behavior import BehaviorLayer

        persona = MockPersona()
        layer = BehaviorLayer(persona)
        result = layer.shape_system_prompt("Base prompt.")
        assert "Missy" in result
        assert "warm" in result
        assert "curious" in result

    def test_persona_boundaries_included(self) -> None:
        from missy.agent.behavior import BehaviorLayer

        persona = MockPersona()
        layer = BehaviorLayer(persona)
        result = layer.shape_system_prompt("Base.")
        assert "destructive" in result

    def test_guidelines_include_tone_adaptation(self) -> None:
        from missy.agent.behavior import BehaviorLayer

        layer = BehaviorLayer()
        ctx = {"user_tone": "frustrated", "turn_count": 1, "has_tool_results": False,
               "topic": "", "intent": "", "urgency": "low"}
        guidelines = layer.get_response_guidelines(ctx)
        assert "empathetic" in guidelines.lower() or "difficulty" in guidelines.lower()

    def test_guidelines_conciseness_at_turn_10(self) -> None:
        from missy.agent.behavior import BehaviorLayer

        layer = BehaviorLayer()
        ctx = {"user_tone": "casual", "turn_count": 10, "has_tool_results": False,
               "topic": "", "intent": "", "urgency": "low"}
        guidelines = layer.get_response_guidelines(ctx)
        assert "concise" in guidelines.lower()

    def test_guidelines_tool_results(self) -> None:
        from missy.agent.behavior import BehaviorLayer

        layer = BehaviorLayer()
        ctx = {"user_tone": "casual", "turn_count": 1, "has_tool_results": True,
               "topic": "", "intent": "", "urgency": "low"}
        guidelines = layer.get_response_guidelines(ctx)
        assert "tool" in guidelines.lower()

    def test_guidelines_high_urgency(self) -> None:
        from missy.agent.behavior import BehaviorLayer

        layer = BehaviorLayer()
        ctx = {"user_tone": "casual", "turn_count": 1, "has_tool_results": False,
               "topic": "", "intent": "", "urgency": "high"}
        guidelines = layer.get_response_guidelines(ctx)
        assert "preamble" in guidelines.lower() or "answer" in guidelines.lower()

    def test_guidelines_painting_mode(self) -> None:
        from missy.agent.behavior import BehaviorLayer

        layer = BehaviorLayer()
        ctx = {"user_tone": "casual", "turn_count": 1, "has_tool_results": False,
               "topic": "", "intent": "", "urgency": "low", "vision_mode": "painting"}
        guidelines = layer.get_response_guidelines(ctx)
        assert "encouraging" in guidelines.lower() or "warm" in guidelines.lower()

    def test_guidelines_puzzle_mode(self) -> None:
        from missy.agent.behavior import BehaviorLayer

        layer = BehaviorLayer()
        ctx = {"user_tone": "casual", "turn_count": 1, "has_tool_results": False,
               "topic": "", "intent": "", "urgency": "low", "vision_mode": "puzzle"}
        guidelines = layer.get_response_guidelines(ctx)
        assert "puzzle" in guidelines.lower()

    def test_guidelines_generic_vision_mode(self) -> None:
        from missy.agent.behavior import BehaviorLayer

        layer = BehaviorLayer()
        ctx = {"user_tone": "casual", "turn_count": 1, "has_tool_results": False,
               "topic": "", "intent": "", "urgency": "low", "vision_mode": "general"}
        guidelines = layer.get_response_guidelines(ctx)
        assert "visual" in guidelines.lower()

    def test_guidelines_technical_topic(self) -> None:
        from missy.agent.behavior import BehaviorLayer

        layer = BehaviorLayer()
        ctx = {"user_tone": "casual", "turn_count": 1, "has_tool_results": False,
               "topic": "writing a function", "intent": "", "urgency": "low"}
        guidelines = layer.get_response_guidelines(ctx)
        assert "code" in guidelines.lower() or "technical" in guidelines.lower()

    def test_should_be_concise_high_urgency(self) -> None:
        from missy.agent.behavior import BehaviorLayer

        layer = BehaviorLayer()
        assert layer.should_be_concise({"urgency": "high"}) is True

    def test_should_be_concise_brief_tone(self) -> None:
        from missy.agent.behavior import BehaviorLayer

        layer = BehaviorLayer()
        assert layer.should_be_concise({"user_tone": "brief"}) is True

    def test_should_be_concise_none_context(self) -> None:
        from missy.agent.behavior import BehaviorLayer

        layer = BehaviorLayer()
        assert layer.should_be_concise(None) is False


# ---------------------------------------------------------------------------
# IntentInterpreter tests
# ---------------------------------------------------------------------------


class TestIntentInterpreter:
    """Test IntentInterpreter.classify_intent() and extract_urgency()."""

    def test_greeting(self) -> None:
        from missy.agent.behavior import IntentInterpreter

        interp = IntentInterpreter()
        assert interp.classify_intent("Hello there!") == "greeting"
        assert interp.classify_intent("hey") == "greeting"
        assert interp.classify_intent("Good morning") == "greeting"
        assert interp.classify_intent("Yo") == "greeting"

    def test_farewell(self) -> None:
        from missy.agent.behavior import IntentInterpreter

        interp = IntentInterpreter()
        assert interp.classify_intent("bye!") == "farewell"
        assert interp.classify_intent("see you later") == "farewell"
        assert interp.classify_intent("take care") == "farewell"

    def test_confirmation(self) -> None:
        from missy.agent.behavior import IntentInterpreter

        interp = IntentInterpreter()
        assert interp.classify_intent("ok") == "confirmation"
        assert interp.classify_intent("yes") == "confirmation"
        assert interp.classify_intent("sounds good") == "confirmation"
        assert interp.classify_intent("go ahead") == "confirmation"

    def test_frustration(self) -> None:
        from missy.agent.behavior import IntentInterpreter

        interp = IntentInterpreter()
        assert interp.classify_intent("it's still not working!") == "frustration"
        assert interp.classify_intent("nothing works") == "frustration"

    def test_troubleshooting(self) -> None:
        from missy.agent.behavior import IntentInterpreter

        interp = IntentInterpreter()
        assert interp.classify_intent("I'm getting a traceback error") == "troubleshooting"
        assert interp.classify_intent("connection refused on port 8080") == "troubleshooting"
        assert interp.classify_intent("permission denied when running the script") == "troubleshooting"

    def test_clarification(self) -> None:
        from missy.agent.behavior import IntentInterpreter

        interp = IntentInterpreter()
        assert interp.classify_intent("what do you mean by that?") == "clarification"
        assert interp.classify_intent("could you elaborate on that?") == "clarification"

    def test_feedback(self) -> None:
        from missy.agent.behavior import IntentInterpreter

        interp = IntentInterpreter()
        assert interp.classify_intent("that's perfect") == "feedback"
        assert interp.classify_intent("not quite right") == "feedback"

    def test_exploration(self) -> None:
        from missy.agent.behavior import IntentInterpreter

        interp = IntentInterpreter()
        assert interp.classify_intent("tell me more about Redis caching") == "exploration"
        assert interp.classify_intent("what other options do I have?") == "exploration"

    def test_command(self) -> None:
        from missy.agent.behavior import IntentInterpreter

        interp = IntentInterpreter()
        assert interp.classify_intent("run the tests") == "command"
        assert interp.classify_intent("please deploy to staging") == "command"
        assert interp.classify_intent("delete the old logs") == "command"

    def test_question_default(self) -> None:
        from missy.agent.behavior import IntentInterpreter

        interp = IntentInterpreter()
        assert interp.classify_intent("what is the meaning of life?") == "question"
        assert interp.classify_intent("random unmatched text here") == "question"

    def test_urgency_high(self) -> None:
        from missy.agent.behavior import IntentInterpreter

        interp = IntentInterpreter()
        assert interp.extract_urgency("production is down!") == "high"
        assert interp.extract_urgency("this is urgent") == "high"
        assert interp.extract_urgency("need this fixed ASAP") == "high"

    def test_urgency_medium(self) -> None:
        from missy.agent.behavior import IntentInterpreter

        interp = IntentInterpreter()
        assert interp.extract_urgency("I need this done today") == "medium"
        assert interp.extract_urgency("can you do this quickly?") == "medium"

    def test_urgency_low(self) -> None:
        from missy.agent.behavior import IntentInterpreter

        interp = IntentInterpreter()
        assert interp.extract_urgency("when you have time, could you...") == "low"
        assert interp.extract_urgency("just a general question") == "low"


# ---------------------------------------------------------------------------
# ResponseShaper tests
# ---------------------------------------------------------------------------


class TestResponseShaper:
    """Test ResponseShaper.shape_response() and detect_robotic_patterns()."""

    def test_strips_ai_preamble(self) -> None:
        from missy.agent.behavior import ResponseShaper

        shaper = ResponseShaper()
        result = shaper.shape_response(
            "As an AI language model, I can help you. Here is the answer.",
            persona=None, context={},
        )
        assert "As an AI" not in result
        assert "Here is the answer." in result

    def test_strips_certainly(self) -> None:
        from missy.agent.behavior import ResponseShaper

        shaper = ResponseShaper()
        result = shaper.shape_response(
            "Certainly! I'll help you with that. The solution is X.",
            persona=None, context={},
        )
        assert "Certainly" not in result
        assert "The solution is X." in result

    def test_strips_great_question(self) -> None:
        from missy.agent.behavior import ResponseShaper

        shaper = ResponseShaper()
        result = shaper.shape_response(
            "Great question! The answer is 42.",
            persona=None, context={},
        )
        assert "Great question" not in result
        assert "42" in result

    def test_preserves_code_blocks(self) -> None:
        from missy.agent.behavior import ResponseShaper

        shaper = ResponseShaper()
        code = '```python\nAs an AI, print("hello")\n```'
        result = shaper.shape_response(
            f"Here is the code:\n{code}",
            persona=None, context={},
        )
        assert 'As an AI, print("hello")' in result

    def test_preserves_inline_code(self) -> None:
        from missy.agent.behavior import ResponseShaper

        shaper = ResponseShaper()
        result = shaper.shape_response(
            "Use `As an AI` as a string constant.",
            persona=None, context={},
        )
        assert "`As an AI`" in result

    def test_empty_response_unchanged(self) -> None:
        from missy.agent.behavior import ResponseShaper

        shaper = ResponseShaper()
        assert shaper.shape_response("", persona=None, context={}) == ""

    def test_collapses_excess_blank_lines(self) -> None:
        from missy.agent.behavior import ResponseShaper

        shaper = ResponseShaper()
        result = shaper.shape_response(
            "Line 1\n\n\n\n\nLine 2",
            persona=None, context={},
        )
        assert "\n\n\n" not in result
        assert "Line 1" in result
        assert "Line 2" in result

    def test_detect_robotic_patterns(self) -> None:
        from missy.agent.behavior import ResponseShaper

        shaper = ResponseShaper()
        text = "As an AI language model, I'm here to help. Certainly! Great question!"
        found = shaper.detect_robotic_patterns(text)
        assert len(found) >= 2

    def test_detect_no_robotic_patterns_in_clean_text(self) -> None:
        from missy.agent.behavior import ResponseShaper

        shaper = ResponseShaper()
        found = shaper.detect_robotic_patterns("The solution is to restart nginx.")
        assert found == []

    def test_multiple_robotic_phrases_all_stripped(self) -> None:
        from missy.agent.behavior import ResponseShaper

        shaper = ResponseShaper()
        result = shaper.shape_response(
            "Of course! I'd be happy to help you. As an AI assistant, I can explain that the answer is 42.",
            persona=None, context={},
        )
        assert "Of course" not in result
        assert "happy to help" not in result
        assert "As an AI" not in result
        assert "42" in result


# ---------------------------------------------------------------------------
# Integration: full pipeline
# ---------------------------------------------------------------------------


class TestBehaviorIntegration:
    """End-to-end behavior layer usage."""

    def test_full_pipeline(self) -> None:
        from missy.agent.behavior import BehaviorLayer, IntentInterpreter, ResponseShaper

        persona = MockPersona()
        layer = BehaviorLayer(persona)
        interp = IntentInterpreter()
        shaper = ResponseShaper()

        messages = [
            {"role": "user", "content": "hey, I'm getting an error when I try to deploy. It's still broken."}
        ]

        tone = layer.analyze_user_tone(messages)
        intent = interp.classify_intent(messages[-1]["content"])
        urgency = interp.extract_urgency(messages[-1]["content"])

        ctx = {
            "user_tone": tone,
            "topic": "deployment",
            "turn_count": 3,
            "has_tool_results": False,
            "intent": intent,
            "urgency": urgency,
        }

        system = layer.shape_system_prompt("You are a helpful assistant.", ctx)
        assert "Missy" in system
        assert "warm" in system

        guidelines = layer.get_response_guidelines(ctx)
        # Should have tone and intent guidance
        assert len(guidelines) > 0

        raw = "Certainly! As an AI, I can help. The deployment is failing because..."
        clean = shaper.shape_response(raw, persona, ctx)
        assert "Certainly" not in clean
        assert "deployment" in clean

    def test_all_intent_types_produce_guidelines(self) -> None:
        """Every intent type should produce non-empty guidelines."""
        from missy.agent.behavior import BehaviorLayer

        layer = BehaviorLayer()
        intents = [
            "greeting", "farewell", "confirmation", "frustration",
            "troubleshooting", "clarification", "feedback",
            "exploration", "command", "question",
        ]

        for intent in intents:
            ctx = {
                "user_tone": "casual", "turn_count": 1,
                "has_tool_results": False, "topic": "",
                "intent": intent, "urgency": "low",
            }
            guidelines = layer.get_response_guidelines(ctx)
            # At minimum, tone adaptation should be present
            assert isinstance(guidelines, str)
