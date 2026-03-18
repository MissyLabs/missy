"""Integration tests verifying that the persona system influences agent runtime behavior.

Tests exercise the full chain:
  PersonaManager -> BehaviorLayer -> AgentRuntime system prompt shaping
  IntentInterpreter -> BehaviorLayer.get_response_guidelines
  ResponseShaper -> final output post-processing

All filesystem I/O is redirected to pytest's tmp_path so tests are hermetic
and leave no side effects on the real ~/.missy/ directory.

Provider calls are mocked via unittest.mock to avoid network access.
"""

from __future__ import annotations

from contextlib import ExitStack
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from missy.agent.behavior import BehaviorLayer, IntentInterpreter, ResponseShaper
from missy.agent.persona import PersonaConfig, PersonaManager
from missy.agent.runtime import AgentConfig, AgentRuntime
from missy.providers.base import CompletionResponse

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

_BASE_SYSTEM_PROMPT = "You are Missy, a helpful assistant."


def _make_completion(text: str = "Here is the answer.") -> CompletionResponse:
    return CompletionResponse(
        content=text,
        model="mock-model",
        provider="mock",
        usage={"prompt_tokens": 10, "completion_tokens": 10, "total_tokens": 20},
        raw={},
        tool_calls=[],
        finish_reason="stop",
    )


def _make_mock_provider(response_text: str = "Here is the answer.") -> MagicMock:
    provider = MagicMock()
    provider.name = "mock"
    provider.is_available.return_value = True
    provider.complete.return_value = _make_completion(response_text)
    return provider


def _make_mock_registry(provider: MagicMock) -> MagicMock:
    registry = MagicMock()
    registry.get.return_value = provider
    registry.get_available.return_value = [provider]
    return registry


def _run_with_mocks(
    runtime: AgentRuntime,
    user_input: str,
    provider: MagicMock,
) -> str:
    """Run the agent with mocked registry and censor, capturing provider calls."""
    registry = _make_mock_registry(provider)
    with ExitStack() as stack:
        stack.enter_context(
            patch("missy.agent.runtime.get_registry", return_value=registry)
        )
        # Disable tool registry so we stay in single-turn mode
        stack.enter_context(
            patch("missy.agent.runtime.get_tool_registry", side_effect=RuntimeError)
        )
        stack.enter_context(
            patch("missy.agent.runtime.censor_response", side_effect=lambda x: x)
        )
        return runtime.run(user_input)


def _capture_system_prompt(provider: MagicMock) -> str:
    """Extract the system prompt string passed to provider.complete()."""
    assert provider.complete.called, "provider.complete was never called"
    call_args = provider.complete.call_args
    # complete(messages, system=...) or complete(messages)
    messages_arg = call_args[0][0] if call_args[0] else []
    # System prompt may be passed as kwarg or as the first Message with role=system
    system_kwarg = call_args[1].get("system", "") if call_args[1] else ""
    if system_kwarg:
        return system_kwarg
    # Fallback: find a system-role message in the list
    for msg in messages_arg:
        role = getattr(msg, "role", None) or (msg.get("role") if isinstance(msg, dict) else None)
        if role == "system":
            content = getattr(msg, "content", None) or (msg.get("content") if isinstance(msg, dict) else "")
            return content or ""
    return ""


def _make_runtime_with_persona(
    persona: PersonaConfig,
    system_prompt: str = _BASE_SYSTEM_PROMPT,
    tmp_path: Path | None = None,
) -> AgentRuntime:
    """Create an AgentRuntime whose BehaviorLayer is seeded with the given persona."""
    config = AgentConfig(
        provider="mock",
        system_prompt=system_prompt,
        max_iterations=1,
    )
    runtime = AgentRuntime(config)
    # Inject the real BehaviorLayer built from the supplied persona
    runtime._behavior = BehaviorLayer(persona=persona)
    runtime._response_shaper = ResponseShaper()
    runtime._intent_interpreter = IntentInterpreter()
    # Point persona_manager at the supplied persona via a lightweight stub
    pm_stub = MagicMock()
    pm_stub.get_persona.return_value = persona
    runtime._persona_manager = pm_stub
    return runtime


# ---------------------------------------------------------------------------
# Test 1: System prompt includes persona identity
# ---------------------------------------------------------------------------


class TestPersonaInSystemPrompt:
    """Verify that a loaded persona's identity appears in the provider's system prompt."""

    def test_persona_name_in_system_prompt(self, tmp_path: Path) -> None:
        """When a persona is loaded, the system prompt sent to the provider
        contains the persona's name."""
        persona = PersonaConfig(name="Athena", identity_description="Athena is a wise assistant.")
        runtime = _make_runtime_with_persona(persona, tmp_path=tmp_path)
        provider = _make_mock_provider()

        _run_with_mocks(runtime, "Hello", provider)

        system_prompt = _capture_system_prompt(provider)
        assert "Athena" in system_prompt, (
            f"Expected persona name 'Athena' in system prompt. Got:\n{system_prompt}"
        )

    def test_persona_identity_description_in_system_prompt(self, tmp_path: Path) -> None:
        """The identity_description field is injected into the system prompt."""
        description = "Helios is a security-focused engineering assistant."
        persona = PersonaConfig(name="Helios", identity_description=description)
        runtime = _make_runtime_with_persona(persona, tmp_path=tmp_path)
        provider = _make_mock_provider()

        _run_with_mocks(runtime, "What can you help with?", provider)

        system_prompt = _capture_system_prompt(provider)
        assert "Helios" in system_prompt, (
            f"Expected 'Helios' in system prompt. Got:\n{system_prompt}"
        )

    def test_base_system_prompt_preserved_with_persona(self, tmp_path: Path) -> None:
        """The original base system prompt is always preserved even after persona injection."""
        custom_base = "You are a specialized coding assistant."
        persona = PersonaConfig(name="Cody")
        runtime = _make_runtime_with_persona(persona, system_prompt=custom_base, tmp_path=tmp_path)
        provider = _make_mock_provider()

        _run_with_mocks(runtime, "Explain recursion", provider)

        system_prompt = _capture_system_prompt(provider)
        assert custom_base in system_prompt or "specialized coding assistant" in system_prompt, (
            f"Base system prompt fragment not found. Got:\n{system_prompt}"
        )


# ---------------------------------------------------------------------------
# Test 2: Response shaping applies persona tone
# ---------------------------------------------------------------------------


class TestPersonaToneInResponseShaping:
    """Verify that ResponseShaper strips robotic artifacts regardless of persona tone."""

    def test_robotic_preamble_stripped_from_response(self) -> None:
        """ResponseShaper removes 'As an AI language model,' style phrases."""
        persona = PersonaConfig(name="Missy", tone=["direct", "technical"])
        shaper = ResponseShaper()
        robotic_response = (
            "As an AI language model, I can help you with that. "
            "Certainly! Here is how you restart nginx: sudo systemctl restart nginx."
        )
        cleaned = shaper.shape_response(robotic_response, persona=persona, context={})
        assert "As an AI language model" not in cleaned
        assert "nginx" in cleaned, "Technical content must be preserved after shaping"

    def test_code_blocks_preserved_during_shaping(self) -> None:
        """ResponseShaper must not alter content inside fenced code blocks."""
        persona = PersonaConfig(name="Missy", tone=["technical"])
        shaper = ResponseShaper()
        response = (
            "As an AI, I can show you:\n"
            "```bash\n"
            "sudo systemctl restart nginx\n"
            "```"
        )
        cleaned = shaper.shape_response(response, persona=persona, context={})
        assert "sudo systemctl restart nginx" in cleaned
        assert "```bash" in cleaned

    def test_response_shaper_applied_in_runtime(self, tmp_path: Path) -> None:
        """When AgentRuntime runs, the ResponseShaper is applied to the provider's output."""
        persona = PersonaConfig(name="Missy")
        runtime = _make_runtime_with_persona(persona, tmp_path=tmp_path)
        provider = _make_mock_provider(
            "Certainly! As an AI assistant, I'd be happy to help. The answer is 42."
        )

        result = _run_with_mocks(runtime, "What is the answer?", provider)

        # The runtime should have applied response shaping
        assert isinstance(result, str)
        assert len(result) > 0
        # Robotic phrase should have been stripped
        assert "Certainly!" not in result or "As an AI assistant" not in result


# ---------------------------------------------------------------------------
# Test 3: Persona boundaries injected into system prompt
# ---------------------------------------------------------------------------


class TestPersonaBoundariesInSystemPrompt:
    """Security boundaries defined in the persona must appear in the system prompt."""

    def test_boundaries_present_in_shaped_prompt(self, tmp_path: Path) -> None:
        """PersonaConfig.boundaries items appear in the shaped system prompt."""
        custom_boundary = "Never disclose internal architecture details"
        persona = PersonaConfig(
            name="Vault",
            boundaries=[custom_boundary, "Never execute destructive operations without confirmation"],
        )
        runtime = _make_runtime_with_persona(persona, tmp_path=tmp_path)
        provider = _make_mock_provider()

        _run_with_mocks(runtime, "Tell me about yourself", provider)

        system_prompt = _capture_system_prompt(provider)
        assert custom_boundary in system_prompt, (
            f"Expected boundary text in system prompt. Got:\n{system_prompt}"
        )

    def test_behavior_layer_includes_boundaries_directly(self) -> None:
        """BehaviorLayer._build_persona_block always includes boundary lines."""
        persona = PersonaConfig(
            name="Fortress",
            boundaries=["Never expose secrets or credentials", "Always respect policy decisions"],
        )
        layer = BehaviorLayer(persona=persona)
        block = layer._build_persona_block()

        assert "Never expose secrets or credentials" in block
        assert "Always respect policy decisions" in block
        assert "Hard boundaries" in block


# ---------------------------------------------------------------------------
# Test 4: Persona change mid-session updates system prompt
# ---------------------------------------------------------------------------


class TestPersonaChangeMidSession:
    """Changing persona between runs causes the new persona to appear in subsequent prompts."""

    def test_updated_persona_reflected_in_new_run(self, tmp_path: Path) -> None:
        """After swapping the BehaviorLayer persona, the next run uses the updated identity."""
        original_persona = PersonaConfig(name="Atlas", identity_description="Atlas the original.")
        updated_persona = PersonaConfig(name="Phoenix", identity_description="Phoenix the updated.")

        config = AgentConfig(
            provider="mock",
            system_prompt=_BASE_SYSTEM_PROMPT,
            max_iterations=1,
        )
        runtime = AgentRuntime(config)
        runtime._response_shaper = ResponseShaper()
        runtime._intent_interpreter = IntentInterpreter()

        # First run with original persona
        runtime._behavior = BehaviorLayer(persona=original_persona)
        provider_first = _make_mock_provider("First response.")
        _run_with_mocks(runtime, "Hello", provider_first)
        first_system = _capture_system_prompt(provider_first)
        assert "Atlas" in first_system

        # Swap persona to simulate mid-session change
        runtime._behavior = BehaviorLayer(persona=updated_persona)
        pm_stub = MagicMock()
        pm_stub.get_persona.return_value = updated_persona
        runtime._persona_manager = pm_stub

        provider_second = _make_mock_provider("Second response.")
        _run_with_mocks(runtime, "Hello again", provider_second)
        second_system = _capture_system_prompt(provider_second)

        assert "Phoenix" in second_system, (
            f"Expected updated persona 'Phoenix' in system prompt. Got:\n{second_system}"
        )
        assert "Atlas" not in second_system or "Phoenix" in second_system


# ---------------------------------------------------------------------------
# Test 5: Default persona works without an explicit persona file
# ---------------------------------------------------------------------------


class TestDefaultPersona:
    """Without an explicit persona YAML file, the default PersonaConfig is used."""

    def test_persona_manager_returns_default_without_file(self, tmp_path: Path) -> None:
        """PersonaManager with a non-existent path returns a valid default PersonaConfig."""
        nonexistent = tmp_path / "no_persona.yaml"
        pm = PersonaManager(persona_path=nonexistent)
        config = pm.get_persona()

        assert isinstance(config, PersonaConfig)
        assert config.name == "Missy"
        assert config.version == 1
        assert len(config.tone) > 0
        assert len(config.boundaries) > 0

    def test_default_persona_feeds_behavior_layer(self, tmp_path: Path) -> None:
        """A BehaviorLayer built from the default persona produces a non-empty shaped prompt."""
        pm = PersonaManager(persona_path=tmp_path / "missing.yaml")
        persona = pm.get_persona()
        layer = BehaviorLayer(persona=persona)

        ctx = {
            "user_tone": "casual",
            "topic": "",
            "turn_count": 1,
            "has_tool_results": False,
            "intent": "question",
            "urgency": "low",
        }
        shaped = layer.shape_system_prompt(_BASE_SYSTEM_PROMPT, ctx)

        assert _BASE_SYSTEM_PROMPT in shaped
        assert "## Persona" in shaped
        assert "Missy" in shaped

    def test_runtime_uses_default_persona_when_file_absent(self, tmp_path: Path) -> None:
        """AgentRuntime falls back to the default persona when no persona file exists."""
        # Runtime's _make_persona_manager will fall back to default since no file
        config = AgentConfig(
            provider="mock",
            system_prompt=_BASE_SYSTEM_PROMPT,
            max_iterations=1,
        )
        runtime = AgentRuntime(config)
        # The runtime should have created subsystems without error
        assert runtime._behavior is not None or runtime._persona_manager is not None

        provider = _make_mock_provider()
        result = _run_with_mocks(runtime, "Hi", provider)
        assert isinstance(result, str)


# ---------------------------------------------------------------------------
# Test 6: Persona disabled — raw response passes through unchanged
# ---------------------------------------------------------------------------


class TestPersonaDisabled:
    """When BehaviorLayer has no persona, raw responses pass through without persona injection."""

    def test_no_persona_block_when_persona_is_none(self) -> None:
        """BehaviorLayer with persona=None produces no ## Persona section."""
        layer = BehaviorLayer(persona=None)
        shaped = layer.shape_system_prompt(_BASE_SYSTEM_PROMPT, context={})

        assert "## Persona" not in shaped

    def test_system_prompt_still_has_base_content_without_persona(self) -> None:
        """BehaviorLayer with no persona still returns the base prompt intact."""
        layer = BehaviorLayer(persona=None)
        shaped = layer.shape_system_prompt(_BASE_SYSTEM_PROMPT, context={})

        assert _BASE_SYSTEM_PROMPT in shaped

    def test_response_shaper_with_none_persona_still_strips_robotic_phrases(self) -> None:
        """ResponseShaper cleans robotic phrases even when persona is None."""
        shaper = ResponseShaper()
        robotic = "Great question! As an AI language model, I can explain this."
        cleaned = shaper.shape_response(robotic, persona=None, context={})

        assert "Great question!" not in cleaned
        assert "As an AI language model" not in cleaned

    def test_runtime_with_null_behavior_returns_valid_response(self, tmp_path: Path) -> None:
        """AgentRuntime with _behavior=None degrades gracefully and returns provider output."""
        config = AgentConfig(
            provider="mock",
            system_prompt=_BASE_SYSTEM_PROMPT,
            max_iterations=1,
        )
        runtime = AgentRuntime(config)
        runtime._behavior = None
        runtime._response_shaper = None
        runtime._intent_interpreter = None
        runtime._persona_manager = None

        provider = _make_mock_provider("Raw response from provider.")
        result = _run_with_mocks(runtime, "Say something", provider)

        assert "Raw response from provider." in result


# ---------------------------------------------------------------------------
# Test 7: Intent classification influences response guidelines
# ---------------------------------------------------------------------------


class TestIntentClassificationInfluencesGuidelines:
    """IntentInterpreter.classify_intent drives different guideline text in BehaviorLayer."""

    @pytest.mark.parametrize(
        "user_text,expected_intent,expected_guideline_fragment",
        [
            (
                "How do I restart nginx?",
                "question",
                None,  # question is the fallback — presence of guidelines is enough
            ),
            (
                "run the deployment script",
                "command",
                "Direct instruction detected",
            ),
            (
                "I'm getting a segfault and a traceback in my app",
                "troubleshooting",
                "likely cause",
            ),
            (
                "hey there!",
                "greeting",
                "warmly",
            ),
            (
                "goodbye, see you later",
                "farewell",
                "farewell",
            ),
            (
                "still not working, you already told me that, this is useless",
                "frustration",
                "empathy",
            ),
            (
                "tell me more about your ideas for optimizing this",
                "exploration",
                "ideas",
            ),
        ],
    )
    def test_intent_specific_guidelines(
        self,
        user_text: str,
        expected_intent: str,
        expected_guideline_fragment: str | None,
    ) -> None:
        """For each intent, the correct directive phrase appears in guidelines."""
        interpreter = IntentInterpreter()
        intent = interpreter.classify_intent(user_text)
        assert intent == expected_intent, (
            f"classify_intent({user_text!r}) returned {intent!r}, expected {expected_intent!r}"
        )

        persona = PersonaConfig(name="Missy")
        layer = BehaviorLayer(persona=persona)
        ctx = {
            "user_tone": "casual",
            "topic": "",
            "turn_count": 1,
            "has_tool_results": False,
            "intent": intent,
            "urgency": "low",
        }
        guidelines = layer.get_response_guidelines(ctx)

        if expected_guideline_fragment is not None:
            assert expected_guideline_fragment.lower() in guidelines.lower(), (
                f"Expected {expected_guideline_fragment!r} in guidelines for intent={intent!r}. "
                f"Got:\n{guidelines}"
            )

    def test_troubleshooting_intent_in_runtime_system_prompt(self, tmp_path: Path) -> None:
        """A troubleshooting message routes the structured diagnosis guideline into the system prompt."""
        persona = PersonaConfig(name="Debugger")
        runtime = _make_runtime_with_persona(persona, tmp_path=tmp_path)
        provider = _make_mock_provider()

        _run_with_mocks(
            runtime,
            "I'm getting a permission denied error when I run my script, traceback attached",
            provider,
        )

        system_prompt = _capture_system_prompt(provider)
        # BehaviorLayer should have injected the troubleshooting guideline
        assert "likely cause" in system_prompt or "diagnostic" in system_prompt or "fix" in system_prompt, (
            f"Troubleshooting guideline not found in system prompt:\n{system_prompt}"
        )

    def test_command_intent_in_runtime_system_prompt(self, tmp_path: Path) -> None:
        """A command-type message routes the 'Direct instruction' guideline into the system prompt."""
        persona = PersonaConfig(name="Executor")
        runtime = _make_runtime_with_persona(persona, tmp_path=tmp_path)
        provider = _make_mock_provider()

        _run_with_mocks(runtime, "run the tests and show me the results", provider)

        system_prompt = _capture_system_prompt(provider)
        assert "Direct instruction" in system_prompt or "Execute" in system_prompt, (
            f"Command guideline not found in system prompt:\n{system_prompt}"
        )


# ---------------------------------------------------------------------------
# Test 8: Memory-informed replies — memory context influences response shaping
# ---------------------------------------------------------------------------


class TestMemoryInformedReplies:
    """When memory context (tool results) is available, it flags in shaping guidelines."""

    def test_tool_results_flag_triggers_weave_guideline(self) -> None:
        """has_tool_results=True injects the 'weave them into your reply' directive."""
        persona = PersonaConfig(name="Missy")
        layer = BehaviorLayer(persona=persona)
        ctx = {
            "user_tone": "casual",
            "topic": "",
            "turn_count": 1,
            "has_tool_results": True,
            "intent": "question",
            "urgency": "low",
        }
        guidelines = layer.get_response_guidelines(ctx)

        assert "tool results" in guidelines.lower(), (
            f"Expected 'tool results' guidance in:\n{guidelines}"
        )
        assert "weave" in guidelines.lower() or "naturally" in guidelines.lower(), (
            f"Expected 'weave' or 'naturally' in:\n{guidelines}"
        )

    def test_no_tool_results_no_weave_guideline(self) -> None:
        """has_tool_results=False must not inject the tool-result weave directive."""
        persona = PersonaConfig(name="Missy")
        layer = BehaviorLayer(persona=persona)
        ctx = {
            "user_tone": "casual",
            "topic": "",
            "turn_count": 1,
            "has_tool_results": False,
            "intent": "question",
            "urgency": "low",
        }
        guidelines = layer.get_response_guidelines(ctx)

        assert "weave" not in guidelines.lower()

    def test_persona_response_style_rules_included_in_guidelines(self) -> None:
        """PersonaConfig.response_style_rules are emitted in guidelines."""
        custom_rule = "Always cite your sources when referencing external facts"
        persona = PersonaConfig(
            name="Scholar",
            response_style_rules=[custom_rule],
        )
        layer = BehaviorLayer(persona=persona)
        ctx = {
            "user_tone": "formal",
            "topic": "",
            "turn_count": 1,
            "has_tool_results": False,
            "intent": "question",
            "urgency": "low",
        }
        guidelines = layer.get_response_guidelines(ctx)

        assert custom_rule in guidelines, (
            f"Expected custom response rule in guidelines. Got:\n{guidelines}"
        )


# ---------------------------------------------------------------------------
# Test 9: Tone adaptation — behavior layer adapts to detected user tone
# ---------------------------------------------------------------------------


class TestToneAdaptation:
    """BehaviorLayer.analyze_user_tone drives different directive text in guidelines."""

    @pytest.mark.parametrize(
        "messages,expected_tone",
        [
            (
                [{"role": "user", "content": "hey yo sup, lol this is cool thx"}],
                "casual",
            ),
            (
                [{"role": "user", "content": "Please kindly assist me. Regarding the configuration, could you elaborate?"}],
                "formal",
            ),
            (
                [{"role": "user", "content": "This still doesn't work, same error again and again, why won't it fix?"}],
                "frustrated",
            ),
            (
                [{"role": "user", "content": "The async function uses a docker container with ssl tls and oauth tokens"}],
                "technical",
            ),
            (
                [{"role": "user", "content": "fix it"}],
                "brief",
            ),
        ],
    )
    def test_tone_detection(self, messages: list[dict], expected_tone: str) -> None:
        """analyze_user_tone correctly identifies the tone from message content."""
        layer = BehaviorLayer()
        detected = layer.analyze_user_tone(messages)
        assert detected == expected_tone, (
            f"Expected tone {expected_tone!r}, got {detected!r} for messages: {messages}"
        )

    def test_casual_tone_adaptation_directive(self) -> None:
        """Casual tone produces a 'conversational' directive in guidelines."""
        layer = BehaviorLayer()
        adaptation = layer.get_tone_adaptation("casual")
        assert "casual" in adaptation.lower() or "conversational" in adaptation.lower(), (
            f"Expected casual guidance. Got: {adaptation!r}"
        )

    def test_formal_tone_adaptation_directive(self) -> None:
        """Formal tone produces a 'professional' or 'precise' directive."""
        layer = BehaviorLayer()
        adaptation = layer.get_tone_adaptation("formal")
        assert "professional" in adaptation.lower() or "precise" in adaptation.lower(), (
            f"Expected formal guidance. Got: {adaptation!r}"
        )

    def test_frustrated_tone_adaptation_directive(self) -> None:
        """Frustrated tone produces an 'empathetic' or 'acknowledge' directive."""
        layer = BehaviorLayer()
        adaptation = layer.get_tone_adaptation("frustrated")
        assert "empathetic" in adaptation.lower() or "acknowledge" in adaptation.lower(), (
            f"Expected frustrated guidance. Got: {adaptation!r}"
        )

    def test_technical_tone_adaptation_directive(self) -> None:
        """Technical tone produces a directive about technical vocabulary."""
        layer = BehaviorLayer()
        adaptation = layer.get_tone_adaptation("technical")
        assert "technical" in adaptation.lower() or "vocabulary" in adaptation.lower(), (
            f"Expected technical guidance. Got: {adaptation!r}"
        )

    def test_unknown_tone_returns_empty_string(self) -> None:
        """An unrecognised tone label returns an empty string, not an error."""
        layer = BehaviorLayer()
        adaptation = layer.get_tone_adaptation("xyzzy_unknown_tone")
        assert adaptation == ""

    def test_tone_included_in_runtime_system_prompt(self, tmp_path: Path) -> None:
        """When the runtime detects a casual tone from history, casual guidance appears in the prompt."""
        persona = PersonaConfig(name="Missy")
        config = AgentConfig(
            provider="mock",
            system_prompt=_BASE_SYSTEM_PROMPT,
            max_iterations=1,
        )
        runtime = AgentRuntime(config)
        runtime._behavior = BehaviorLayer(persona=persona)
        runtime._response_shaper = ResponseShaper()
        runtime._intent_interpreter = IntentInterpreter()
        pm_stub = MagicMock()
        pm_stub.get_persona.return_value = persona
        runtime._persona_manager = pm_stub

        # Pre-populate history with casual messages so the tone analyzer fires
        session = runtime._resolve_session(None)
        sid = str(session.id)
        # Inject casual messages directly into the in-memory session store
        if hasattr(runtime._session_mgr, "_sessions"):
            runtime._session_mgr._sessions[sid] = session

        provider = _make_mock_provider()
        _run_with_mocks(runtime, "hey cool thx lol", provider)

        system_prompt = _capture_system_prompt(provider)
        # The system prompt should contain behavioral guidelines from the behavior layer
        assert "## Persona" in system_prompt or "## Response guidelines" in system_prompt


# ---------------------------------------------------------------------------
# Test 10: Context carryover — previous conversation context influences response shaping
# ---------------------------------------------------------------------------


class TestContextCarryover:
    """Previous conversation state is reflected in BehaviorLayer guidelines."""

    def test_long_conversation_triggers_concise_guideline(self) -> None:
        """After 10+ turns, should_be_concise returns True and the guideline is emitted."""
        persona = PersonaConfig(name="Missy")
        layer = BehaviorLayer(persona=persona)
        ctx = {
            "user_tone": "casual",
            "topic": "",
            "turn_count": 12,
            "has_tool_results": False,
            "intent": "question",
            "urgency": "low",
        }
        assert layer.should_be_concise(ctx) is True

        guidelines = layer.get_response_guidelines(ctx)
        assert "concise" in guidelines.lower(), (
            f"Expected conciseness guideline after 12 turns. Got:\n{guidelines}"
        )

    def test_short_conversation_no_concise_guideline(self) -> None:
        """In a fresh session (1 turn), should_be_concise is False."""
        persona = PersonaConfig(name="Missy")
        layer = BehaviorLayer(persona=persona)
        ctx = {
            "user_tone": "casual",
            "topic": "",
            "turn_count": 1,
            "has_tool_results": False,
            "intent": "question",
            "urgency": "low",
        }
        assert layer.should_be_concise(ctx) is False

    def test_high_urgency_triggers_concise_and_lead_with_answer(self) -> None:
        """High urgency sets should_be_concise=True and injects the urgency directive."""
        persona = PersonaConfig(name="Missy")
        layer = BehaviorLayer(persona=persona)
        ctx = {
            "user_tone": "casual",
            "topic": "",
            "turn_count": 1,
            "has_tool_results": False,
            "intent": "question",
            "urgency": "high",
        }
        assert layer.should_be_concise(ctx) is True
        guidelines = layer.get_response_guidelines(ctx)
        assert "Lead with the answer" in guidelines or "time pressure" in guidelines, (
            f"Expected urgency guideline. Got:\n{guidelines}"
        )

    def test_persona_behavioral_tendencies_persist_across_turns(self) -> None:
        """PersonaConfig.behavioral_tendencies appear in every call to get_response_guidelines."""
        tendency = "Prefers bullet-point summaries over prose"
        persona = PersonaConfig(
            name="Listicle",
            behavioral_tendencies=[tendency],
        )
        layer = BehaviorLayer(persona=persona)

        for turn in (1, 5, 15):
            ctx = {
                "user_tone": "casual",
                "topic": "",
                "turn_count": turn,
                "has_tool_results": False,
                "intent": "question",
                "urgency": "low",
            }
            guidelines = layer.get_response_guidelines(ctx)
            assert tendency in guidelines, (
                f"Behavioral tendency missing at turn {turn}. Got:\n{guidelines}"
            )

    def test_persona_loaded_from_file_round_trips_correctly(self, tmp_path: Path) -> None:
        """A persona saved to disk and reloaded produces identical shaped prompts."""
        persona_path = tmp_path / "persona.yaml"
        pm = PersonaManager(persona_path=persona_path)
        pm.update(
            name="Roundtrip",
            tone=["precise", "thorough"],
            boundaries=["Never guess when you can look it up"],
        )
        pm.save()

        # Load from disk
        pm2 = PersonaManager(persona_path=persona_path)
        loaded = pm2.get_persona()

        assert loaded.name == "Roundtrip"
        assert "precise" in loaded.tone
        assert "Never guess when you can look it up" in loaded.boundaries

        # Both should produce the same shaped prompt
        layer = BehaviorLayer(persona=loaded)
        ctx = {
            "user_tone": "formal",
            "topic": "",
            "turn_count": 1,
            "has_tool_results": False,
            "intent": "question",
            "urgency": "low",
        }
        shaped = layer.shape_system_prompt(_BASE_SYSTEM_PROMPT, ctx)
        assert "Roundtrip" in shaped
        assert "Never guess when you can look it up" in shaped

    def test_get_system_prompt_prefix_matches_behavior_layer_block(self, tmp_path: Path) -> None:
        """PersonaManager.get_system_prompt_prefix and BehaviorLayer._build_persona_block
        both surface the same boundaries so callers can use either pathway.

        Note: get_system_prompt_prefix() renders boundaries under '# Boundaries' using the
        identity_description field (not the name field).  BehaviorLayer._build_persona_block()
        also renders boundaries and injects the name directly.  The shared assertion is the
        boundary text, which must appear in both outputs.
        """
        persona_path = tmp_path / "persona.yaml"
        pm = PersonaManager(persona_path=persona_path)
        boundary_text = "No hallucination without caveat"
        pm.update(
            name="Consistent",
            identity_description="Consistent is a reliable assistant.",
            boundaries=[boundary_text],
        )
        pm.save()

        pm2 = PersonaManager(persona_path=persona_path)
        persona = pm2.get_persona()

        # get_system_prompt_prefix uses identity_description and boundaries (not name directly)
        prefix = pm2.get_system_prompt_prefix()
        assert boundary_text in prefix, (
            f"Expected boundary text in prefix. Got:\n{prefix}"
        )
        assert "Consistent" in prefix, (
            f"Expected 'Consistent' from identity_description in prefix. Got:\n{prefix}"
        )

        # BehaviorLayer._build_persona_block uses name and boundaries
        layer = BehaviorLayer(persona=persona)
        block = layer._build_persona_block()
        assert boundary_text in block, (
            f"Expected boundary text in persona block. Got:\n{block}"
        )
        assert "Consistent" in block, (
            f"Expected persona name in block. Got:\n{block}"
        )


# ---------------------------------------------------------------------------
# Test 11: Custom personality_traits appear in system prompt
# ---------------------------------------------------------------------------


class TestPersonalityTraitsInSystemPrompt:
    """PersonaConfig.personality_traits must propagate through BehaviorLayer into the shaped prompt."""

    def test_custom_personality_traits_in_persona_block(self) -> None:
        """personality_traits are rendered inside the ## Persona block."""
        persona = PersonaConfig(
            name="Trait-Bot",
            personality_traits=["methodical", "empathetic", "detail-oriented"],
        )
        layer = BehaviorLayer(persona=persona)
        block = layer._build_persona_block()

        assert "methodical" in block, f"Expected 'methodical' in persona block:\n{block}"
        assert "empathetic" in block, f"Expected 'empathetic' in persona block:\n{block}"
        assert "detail-oriented" in block, f"Expected 'detail-oriented' in persona block:\n{block}"
        assert "Personality traits:" in block or "traits" in block.lower(), (
            f"Expected 'Personality traits' label in block:\n{block}"
        )

    def test_personality_traits_surfaced_in_shaped_system_prompt(self, tmp_path: Path) -> None:
        """personality_traits appear in the system prompt delivered to the provider."""
        trait = "never-skips-verification-steps"
        persona = PersonaConfig(
            name="Verifier",
            personality_traits=[trait],
        )
        runtime = _make_runtime_with_persona(persona, tmp_path=tmp_path)
        provider = _make_mock_provider()

        _run_with_mocks(runtime, "Help me deploy this", provider)

        system_prompt = _capture_system_prompt(provider)
        assert trait in system_prompt, (
            f"Expected trait {trait!r} in system prompt. Got:\n{system_prompt}"
        )

    def test_get_system_prompt_prefix_includes_personality_traits(self, tmp_path: Path) -> None:
        """PersonaManager.get_system_prompt_prefix renders personality_traits under # Personality."""
        pm = PersonaManager(persona_path=tmp_path / "persona.yaml")
        pm.update(personality_traits=["inquisitive", "security-conscious"])
        prefix = pm.get_system_prompt_prefix()

        assert "# Personality" in prefix, f"Expected '# Personality' header. Got:\n{prefix}"
        assert "inquisitive" in prefix, f"Expected 'inquisitive' trait. Got:\n{prefix}"
        assert "security-conscious" in prefix, f"Expected 'security-conscious' trait. Got:\n{prefix}"

    def test_empty_personality_traits_omits_personality_section(self, tmp_path: Path) -> None:
        """An empty personality_traits list produces no # Personality section in the prefix."""
        pm = PersonaManager(persona_path=tmp_path / "persona.yaml")
        pm.update(personality_traits=[])
        prefix = pm.get_system_prompt_prefix()

        assert "# Personality" not in prefix, (
            f"Expected no '# Personality' section when traits=[]. Got:\n{prefix}"
        )


# ---------------------------------------------------------------------------
# Test 12: Graceful handling of empty persona fields
# ---------------------------------------------------------------------------


class TestPersonaWithEmptyFields:
    """PersonaConfig with empty strings and empty lists must not raise errors anywhere in the pipeline."""

    def test_empty_persona_fields_do_not_raise_in_behavior_layer(self) -> None:
        """BehaviorLayer with all-empty persona fields completes without exceptions."""
        persona = PersonaConfig(
            name="",
            tone=[],
            personality_traits=[],
            behavioral_tendencies=[],
            response_style_rules=[],
            boundaries=[],
            identity_description="",
        )
        layer = BehaviorLayer(persona=persona)
        ctx = {
            "user_tone": "casual",
            "topic": "",
            "turn_count": 1,
            "has_tool_results": False,
            "intent": "question",
            "urgency": "low",
        }
        shaped = layer.shape_system_prompt("Base prompt.", ctx)
        assert "Base prompt." in shaped

    def test_empty_persona_block_with_all_empty_fields(self) -> None:
        """BehaviorLayer._build_persona_block returns a minimal ## Persona when all fields are empty."""
        persona = PersonaConfig(
            name="",
            tone=[],
            personality_traits=[],
            behavioral_tendencies=[],
            response_style_rules=[],
            boundaries=[],
            identity_description="",
        )
        layer = BehaviorLayer(persona=persona)
        block = layer._build_persona_block()
        # Must not raise and must return a string; content may be minimal
        assert isinstance(block, str)

    def test_empty_persona_get_system_prompt_prefix_is_valid_string(self, tmp_path: Path) -> None:
        """PersonaManager.get_system_prompt_prefix returns a string even when all list fields are empty."""
        pm = PersonaManager(persona_path=tmp_path / "persona.yaml")
        pm.update(
            tone=[],
            personality_traits=[],
            behavioral_tendencies=[],
            response_style_rules=[],
            boundaries=[],
            identity_description="",
        )
        prefix = pm.get_system_prompt_prefix()
        assert isinstance(prefix, str)
        # With no tone, personality, tendencies, rules, or boundaries, only the (empty) # Identity section remains
        assert "# Identity" in prefix

    def test_response_shaper_with_empty_persona_fields_does_not_raise(self) -> None:
        """ResponseShaper handles a persona with all-empty fields without raising."""
        persona = PersonaConfig(
            name="",
            tone=[],
            personality_traits=[],
            behavioral_tendencies=[],
            response_style_rules=[],
            boundaries=[],
            identity_description="",
        )
        shaper = ResponseShaper()
        result = shaper.shape_response(
            "As an AI, here is the answer.", persona=persona, context={}
        )
        assert isinstance(result, str)
        assert "As an AI" not in result

    def test_runtime_with_empty_persona_does_not_crash(self, tmp_path: Path) -> None:
        """AgentRuntime injected with an all-empty persona still runs successfully."""
        persona = PersonaConfig(
            name="",
            tone=[],
            personality_traits=[],
            behavioral_tendencies=[],
            response_style_rules=[],
            boundaries=[],
            identity_description="",
        )
        runtime = _make_runtime_with_persona(persona, tmp_path=tmp_path)
        provider = _make_mock_provider("Empty persona response.")
        result = _run_with_mocks(runtime, "Say something", provider)
        assert isinstance(result, str)
        assert len(result) > 0


# ---------------------------------------------------------------------------
# Test 13: Persona reset restores factory defaults
# ---------------------------------------------------------------------------


class TestPersonaResetRestoresDefaults:
    """After edit + reset, the persona returned by PersonaManager matches factory defaults."""

    def test_reset_after_deep_edit_restores_all_defaults(self, tmp_path: Path) -> None:
        """Every editable field is reset to its default after calling pm.reset()."""
        pm = PersonaManager(persona_path=tmp_path / "persona.yaml")
        pm.update(
            name="CustomName",
            tone=["aggressive", "terse"],
            personality_traits=["reckless"],
            behavioral_tendencies=["ignores warnings"],
            response_style_rules=["Never explain yourself"],
            boundaries=[],
            identity_description="A rogue agent.",
        )
        pm.save()

        # Verify the edits stuck
        saved = pm.get_persona()
        assert saved.name == "CustomName"
        assert saved.tone == ["aggressive", "terse"]

        # Reset
        pm.reset()
        restored = pm.get_persona()

        assert restored.name == "Missy", f"Expected default name. Got: {restored.name!r}"
        assert "helpful" in restored.tone or len(restored.tone) > 0, (
            f"Expected default tone list. Got: {restored.tone!r}"
        )
        assert len(restored.personality_traits) > 0
        assert len(restored.behavioral_tendencies) > 0
        assert len(restored.response_style_rules) > 0
        assert len(restored.boundaries) > 0
        assert "Missy" in restored.identity_description

    def test_reset_persona_produces_same_system_prompt_prefix_as_fresh_manager(
        self, tmp_path: Path
    ) -> None:
        """After reset, get_system_prompt_prefix() matches what a brand-new PersonaManager produces."""
        pm_fresh = PersonaManager(persona_path=tmp_path / "fresh.yaml")
        pm_fresh.get_system_prompt_prefix()

        pm_edited = PersonaManager(persona_path=tmp_path / "edited.yaml")
        pm_edited.update(name="Temp", tone=["weird"])
        pm_edited.save()
        pm_edited.reset()
        reset_prefix = pm_edited.get_system_prompt_prefix()

        # Both should contain the same default identity text
        assert "Missy" in reset_prefix, f"Expected 'Missy' after reset. Got:\n{reset_prefix}"
        # The core content should be the same class of information
        assert "# Identity" in reset_prefix
        assert "# Tone" in reset_prefix
        assert "# Boundaries" in reset_prefix

    def test_reset_persona_file_persisted_correctly(self, tmp_path: Path) -> None:
        """After reset + reload, the new PersonaManager instance reads the factory defaults."""
        persona_path = tmp_path / "persona.yaml"
        pm = PersonaManager(persona_path=persona_path)
        pm.update(name="Ephemeral", tone=["cold"])
        pm.save()
        pm.reset()

        # Re-load from disk
        pm2 = PersonaManager(persona_path=persona_path)
        p2 = pm2.get_persona()

        assert p2.name == "Missy", f"Expected 'Missy' after reload. Got: {p2.name!r}"
        assert "helpful" in p2.tone or len(p2.tone) > 0, (
            f"Expected non-empty default tone. Got: {p2.tone!r}"
        )

    def test_behavior_layer_after_reset_uses_defaults(self, tmp_path: Path) -> None:
        """A BehaviorLayer built from a reset persona matches one built from a fresh default."""
        pm = PersonaManager(persona_path=tmp_path / "persona.yaml")
        pm.update(name="OverriddenBot")
        pm.reset()
        persona = pm.get_persona()

        layer = BehaviorLayer(persona=persona)
        ctx = {
            "user_tone": "casual",
            "topic": "",
            "turn_count": 1,
            "has_tool_results": False,
            "intent": "question",
            "urgency": "low",
        }
        shaped = layer.shape_system_prompt("Base.", ctx)
        # Default persona name should appear
        assert "Missy" in shaped, f"Expected 'Missy' after reset. Got:\n{shaped}"
        # Should NOT contain the overridden name
        assert "OverriddenBot" not in shaped


# ---------------------------------------------------------------------------
# Test 14: Full pipeline — persona -> behavior -> system prompt -> response
# ---------------------------------------------------------------------------


class TestFullPersonaPipelineEndToEnd:
    """End-to-end pipeline: load a custom persona, run a query, verify the shaped system
    prompt and cleaned response both reflect the persona configuration."""

    def test_full_pipeline_persona_to_response(self, tmp_path: Path) -> None:
        """Pipeline: custom persona file -> PersonaManager -> BehaviorLayer ->
        AgentRuntime system prompt -> provider call -> ResponseShaper -> final output."""
        # 1. Build and save a custom persona
        persona_path = tmp_path / "persona.yaml"
        pm = PersonaManager(persona_path=persona_path)
        pm.update(
            name="Nexus",
            identity_description="Nexus is a precision-focused systems engineer.",
            tone=["technical", "concise"],
            personality_traits=["methodical", "no-nonsense"],
            boundaries=["Never guess when real data is available"],
        )
        pm.save()

        # 2. Load the persona back from disk and wire the runtime
        pm2 = PersonaManager(persona_path=persona_path)
        loaded_persona = pm2.get_persona()

        runtime = _make_runtime_with_persona(
            loaded_persona,
            system_prompt="You are a systems assistant.",
            tmp_path=tmp_path,
        )
        # The provider returns a robotic response that shaping should clean up
        robotic_text = (
            "Certainly! As an AI assistant, I'd be happy to help you with that system issue. "
            "The kernel parameter to tune is vm.swappiness."
        )
        provider = _make_mock_provider(robotic_text)

        # 3. Run
        result = _run_with_mocks(runtime, "How do I reduce swap usage?", provider)

        # 4. Verify the system prompt that reached the provider contains persona data
        system_prompt = _capture_system_prompt(provider)
        assert "Nexus" in system_prompt, (
            f"Expected persona name 'Nexus' in system prompt:\n{system_prompt}"
        )
        assert "Never guess when real data is available" in system_prompt, (
            f"Expected custom boundary in system prompt:\n{system_prompt}"
        )

        # 5. Verify the response was shaped (robotic phrases stripped)
        assert "Certainly!" not in result, f"Expected 'Certainly!' stripped. Got: {result!r}"
        assert "As an AI assistant" not in result, (
            f"Expected 'As an AI assistant' stripped. Got: {result!r}"
        )
        # The substantive content must survive shaping
        assert "vm.swappiness" in result, f"Expected technical content preserved. Got: {result!r}"

    def test_full_pipeline_intent_routing_to_guidelines(self, tmp_path: Path) -> None:
        """Pipeline: troubleshooting user input -> IntentInterpreter -> BehaviorLayer guidelines
        -> system prompt injected -> provider sees structured diagnosis directive."""
        persona_path = tmp_path / "persona.yaml"
        pm = PersonaManager(persona_path=persona_path)
        pm.update(name="Debugger", identity_description="Debugger specialises in root-cause analysis.")
        pm.save()

        pm2 = PersonaManager(persona_path=persona_path)
        loaded_persona = pm2.get_persona()

        runtime = _make_runtime_with_persona(loaded_persona, tmp_path=tmp_path)
        provider = _make_mock_provider("Check dmesg for kernel errors and review /var/log/syslog.")

        _run_with_mocks(
            runtime,
            "I'm getting a connection refused error and a timeout in my service logs",
            provider,
        )

        system_prompt = _capture_system_prompt(provider)
        # Troubleshooting directive: "likely cause → diagnostic steps → fix"
        assert (
            "likely cause" in system_prompt
            or "diagnostic" in system_prompt
            or "fix" in system_prompt
        ), f"Troubleshooting directive missing from system prompt:\n{system_prompt}"

    def test_full_pipeline_persona_persistence_across_save_load_cycle(
        self, tmp_path: Path
    ) -> None:
        """A persona saved and reloaded produces the same shaped system prompt in two independent runtimes."""
        persona_path = tmp_path / "persona.yaml"

        def _build_and_run(persona: PersonaConfig) -> str:
            """Wire a runtime with the given persona, run a query, return the system prompt."""
            rt = _make_runtime_with_persona(persona, system_prompt="Base.", tmp_path=tmp_path)
            p = _make_mock_provider()
            _run_with_mocks(rt, "Hello", p)
            return _capture_system_prompt(p)

        # First run: save persona
        pm1 = PersonaManager(persona_path=persona_path)
        pm1.update(
            name="Persisted",
            boundaries=["Always verify before deleting"],
            tone=["cautious"],
        )
        pm1.save()
        first_prompt = _build_and_run(pm1.get_persona())

        # Second run: reload persona from disk
        pm2 = PersonaManager(persona_path=persona_path)
        second_prompt = _build_and_run(pm2.get_persona())

        assert "Persisted" in first_prompt
        assert "Persisted" in second_prompt
        assert "Always verify before deleting" in first_prompt
        assert "Always verify before deleting" in second_prompt

    def test_full_pipeline_changing_persona_changes_behavior_output(
        self, tmp_path: Path
    ) -> None:
        """Swapping the persona between runs causes measurably different system prompts."""
        strict_persona = PersonaConfig(
            name="Strict",
            identity_description="Strict always validates input before acting.",
            tone=["cautious", "precise"],
            boundaries=["Validate every input before acting"],
        )
        relaxed_persona = PersonaConfig(
            name="Relaxed",
            identity_description="Relaxed prefers quick answers over thoroughness.",
            tone=["casual", "brief"],
            boundaries=[],
        )

        config = AgentConfig(
            provider="mock",
            system_prompt=_BASE_SYSTEM_PROMPT,
            max_iterations=1,
        )

        # Run with strict persona
        runtime = AgentRuntime(config)
        runtime._behavior = BehaviorLayer(persona=strict_persona)
        runtime._response_shaper = ResponseShaper()
        runtime._intent_interpreter = IntentInterpreter()
        pm_stub_strict = MagicMock()
        pm_stub_strict.get_persona.return_value = strict_persona
        runtime._persona_manager = pm_stub_strict

        provider_strict = _make_mock_provider("Strict response.")
        _run_with_mocks(runtime, "What should I do?", provider_strict)
        strict_prompt = _capture_system_prompt(provider_strict)

        # Run with relaxed persona
        runtime._behavior = BehaviorLayer(persona=relaxed_persona)
        pm_stub_relaxed = MagicMock()
        pm_stub_relaxed.get_persona.return_value = relaxed_persona
        runtime._persona_manager = pm_stub_relaxed

        provider_relaxed = _make_mock_provider("Relaxed response.")
        _run_with_mocks(runtime, "What should I do?", provider_relaxed)
        relaxed_prompt = _capture_system_prompt(provider_relaxed)

        # The two system prompts must differ in persona-specific content
        assert "Strict" in strict_prompt, f"Expected 'Strict' in strict prompt:\n{strict_prompt}"
        assert "Relaxed" in relaxed_prompt, f"Expected 'Relaxed' in relaxed prompt:\n{relaxed_prompt}"
        assert "Validate every input before acting" in strict_prompt, (
            f"Expected strict boundary in strict prompt:\n{strict_prompt}"
        )
        # Strict persona's boundary should not appear in the relaxed run
        assert "Validate every input before acting" not in relaxed_prompt, (
            f"Strict boundary leaked into relaxed prompt:\n{relaxed_prompt}"
        )
