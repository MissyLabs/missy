"""End-to-end tests for the persona → behavior → response pipeline.

Validates that:
- Persona influences system prompt generation
- Behavior layer adapts to user tone and intent
- Response shaper strips robotic artifacts
- The full pipeline works together
- Edge cases (missing persona, empty messages) are handled
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from missy.agent.behavior import BehaviorLayer, IntentInterpreter, ResponseShaper
from missy.agent.persona import PersonaConfig, PersonaManager


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def persona():
    return PersonaConfig(
        name="TestBot",
        tone=["direct", "technical"],
        identity_description="A security-focused Linux assistant.",
        personality_traits=["precise", "efficient"],
        behavioral_tendencies=["prefers code examples", "avoids small talk"],
        response_style_rules=["use bullet points", "be concise"],
        boundaries=["no financial advice", "no medical advice"],
    )


@pytest.fixture
def behavior(persona):
    return BehaviorLayer(persona)


@pytest.fixture
def interpreter():
    return IntentInterpreter()


@pytest.fixture
def shaper():
    return ResponseShaper()


# ---------------------------------------------------------------------------
# Full pipeline tests
# ---------------------------------------------------------------------------


class TestFullPipeline:
    """Test the complete persona → behavior → response pipeline."""

    def test_technical_question_pipeline(self, behavior, interpreter, shaper, persona):
        """Technical question should get technical treatment."""
        user_msg = "How do I configure nginx reverse proxy with SSL termination?"
        messages = [{"role": "user", "content": user_msg}]

        # Step 1: Analyze tone
        tone = behavior.analyze_user_tone(messages)

        # Step 2: Classify intent
        intent = interpreter.classify_intent(user_msg)
        urgency = interpreter.extract_urgency(user_msg)

        # Step 3: Build context
        ctx = {
            "user_tone": tone,
            "turn_count": 1,
            "has_tool_results": False,
            "topic": "nginx",
            "intent": intent,
            "urgency": urgency,
        }

        # Step 4: Shape system prompt
        base = "You are a helpful assistant."
        shaped = behavior.shape_system_prompt(base, ctx)
        assert "You are a helpful assistant." in shaped
        assert "TestBot" in shaped  # persona name injected
        assert "security" in shaped.lower()  # identity description

        # Step 5: Simulate raw response and shape it
        raw = "Certainly! I'd be happy to help you with that. Here's how to configure nginx..."
        clean = shaper.shape_response(raw, persona, ctx)
        assert "Certainly" not in clean  # robotic phrase stripped
        assert "nginx" in clean  # content preserved

    def test_urgent_troubleshooting_pipeline(self, behavior, interpreter, shaper, persona):
        """Urgent troubleshooting should get prioritized treatment."""
        user_msg = "Production is down! Getting connection refused on port 443 immediately"
        messages = [{"role": "user", "content": user_msg}]

        tone = behavior.analyze_user_tone(messages)
        intent = interpreter.classify_intent(user_msg)
        urgency = interpreter.extract_urgency(user_msg)

        assert urgency == "high"
        assert intent in ("troubleshooting", "command")

        ctx = {
            "user_tone": tone,
            "turn_count": 1,
            "has_tool_results": False,
            "topic": "ssl",
            "intent": intent,
            "urgency": urgency,
        }

        guidelines = behavior.get_response_guidelines(ctx)
        assert isinstance(guidelines, str)
        assert len(guidelines) > 0

    def test_casual_greeting_pipeline(self, behavior, interpreter, shaper, persona):
        """Casual greeting should be handled naturally."""
        user_msg = "hey there!"
        messages = [{"role": "user", "content": user_msg}]

        tone = behavior.analyze_user_tone(messages)
        intent = interpreter.classify_intent(user_msg)
        urgency = interpreter.extract_urgency(user_msg)

        assert intent == "greeting"
        assert urgency == "low"

        ctx = {
            "user_tone": tone,
            "turn_count": 1,
            "has_tool_results": False,
            "topic": "",
            "intent": intent,
            "urgency": urgency,
        }

        shaped = behavior.shape_system_prompt("Base prompt.", ctx)
        assert "Base prompt." in shaped

    def test_multi_turn_context(self, behavior, interpreter, shaper, persona):
        """Multiple turns should influence tone analysis."""
        messages = [
            {"role": "user", "content": "I need help with the docker config for the kubernetes deployment pipeline"},
            {"role": "assistant", "content": "Here's the Dockerfile..."},
            {"role": "user", "content": "The container keeps crashing with OOM errors on the async worker thread"},
        ]

        tone = behavior.analyze_user_tone(messages)
        assert tone == "technical"

    def test_frustrated_user_pipeline(self, behavior, interpreter, shaper, persona):
        """Frustrated user should trigger empathetic response guidelines."""
        user_msg = "ugh this still doesn't work, I already tried that suggestion and it's broken"
        messages = [{"role": "user", "content": user_msg}]

        tone = behavior.analyze_user_tone(messages)
        intent = interpreter.classify_intent(user_msg)

        assert intent == "frustration"

        ctx = {
            "user_tone": tone,
            "turn_count": 5,
            "has_tool_results": False,
            "topic": "debugging",
            "intent": intent,
            "urgency": "medium",
        }

        guidelines = behavior.get_response_guidelines(ctx)
        assert isinstance(guidelines, str)

    def test_confirmation_intent(self, behavior, interpreter, shaper, persona):
        """Short confirmations should be classified correctly."""
        for msg in ["ok", "yes", "sounds good", "go ahead", "approved"]:
            intent = interpreter.classify_intent(msg)
            assert intent == "confirmation", f"Expected 'confirmation' for '{msg}', got '{intent}'"

    def test_response_shaping_preserves_code(self, shaper, persona):
        """Code blocks must not be modified by response shaping."""
        raw = """As an AI, here's the fix:

```python
def configure_ssl():
    # Set up SSL context
    ctx = ssl.create_default_context()
    return ctx
```

This should work."""

        clean = shaper.shape_response(raw, persona, {})
        assert "As an AI" not in clean
        assert "```python" in clean
        assert "def configure_ssl():" in clean
        assert "ssl.create_default_context()" in clean

    def test_pipeline_with_no_persona(self, interpreter, shaper):
        """Pipeline works without a persona."""
        layer = BehaviorLayer(None)
        ctx = {
            "user_tone": "casual",
            "turn_count": 1,
            "has_tool_results": False,
            "topic": "",
            "intent": "general",
            "urgency": "low",
        }
        shaped = layer.shape_system_prompt("Base.", ctx)
        assert "Base." in shaped

        clean = shaper.shape_response("Absolutely! Here you go.", None, ctx)
        assert isinstance(clean, str)


class TestPersonaManagerIntegration:
    """Test PersonaManager with the behavior layer."""

    def test_default_persona_loads(self, tmp_path):
        """Default persona should work with behavior layer."""
        mgr = PersonaManager(persona_path=str(tmp_path / "persona.yaml"))
        persona = mgr.get_persona()
        assert persona is not None
        assert persona.name == "Missy"

        layer = BehaviorLayer(persona)
        shaped = layer.shape_system_prompt("Base prompt.", {})
        assert "Missy" in shaped

    def test_edited_persona_reflected(self, tmp_path):
        """Persona edits should be reflected in behavior."""
        mgr = PersonaManager(persona_path=str(tmp_path / "persona.yaml"))
        mgr.update(name="CustomBot")
        persona = mgr.get_persona()
        assert persona.name == "CustomBot"

        layer = BehaviorLayer(persona)
        shaped = layer.shape_system_prompt("Base.", {})
        assert "CustomBot" in shaped

    def test_reset_persona(self, tmp_path):
        mgr = PersonaManager(persona_path=str(tmp_path / "persona.yaml"))
        mgr.update(name="Custom")
        mgr.reset()
        persona = mgr.get_persona()
        assert persona.name == "Missy"  # Back to default


class TestBehaviorEdgeCases:
    """Edge cases in the behavior pipeline."""

    def test_empty_messages_list(self, behavior):
        tone = behavior.analyze_user_tone([])
        assert isinstance(tone, str)

    def test_assistant_only_messages(self, behavior):
        messages = [
            {"role": "assistant", "content": "I can help with that."},
            {"role": "assistant", "content": "Here's more info."},
        ]
        tone = behavior.analyze_user_tone(messages)
        assert isinstance(tone, str)

    def test_non_dict_messages_ignored(self, behavior):
        messages = ["not a dict", None, 42, {"role": "user", "content": "hi"}]
        tone = behavior.analyze_user_tone(messages)
        assert isinstance(tone, str)

    def test_messages_without_content(self, behavior):
        messages = [{"role": "user"}, {"role": "user", "content": ""}]
        tone = behavior.analyze_user_tone(messages)
        assert isinstance(tone, str)

    def test_very_many_messages(self, behavior):
        """Should handle large message histories without issue."""
        messages = [
            {"role": "user", "content": f"Message number {i} about docker kubernetes config"}
            for i in range(100)
        ]
        tone = behavior.analyze_user_tone(messages)
        assert isinstance(tone, str)

    def test_tool_results_in_context(self, behavior, persona):
        """Has_tool_results should influence guidelines."""
        ctx_with = {
            "user_tone": "technical",
            "turn_count": 3,
            "has_tool_results": True,
            "topic": "deployment",
            "intent": "command",
            "urgency": "low",
        }
        ctx_without = {**ctx_with, "has_tool_results": False}

        g1 = behavior.get_response_guidelines(ctx_with)
        g2 = behavior.get_response_guidelines(ctx_without)
        # Both should produce valid guidelines
        assert isinstance(g1, str) and len(g1) > 0
        assert isinstance(g2, str) and len(g2) > 0
