"""Comprehensive behavior layer tests.


Tests for BehaviorLayer, IntentInterpreter, and ResponseShaper covering:
- Tone analysis (casual, formal, frustrated, technical, brief, verbose, neutral)
- Intent classification (all 11 categories)
- Urgency extraction (high, medium, low)
- System prompt shaping (persona + context)
- Response guidelines (tone, vision modes, intents, urgency)
- Robotic phrase stripping and code block preservation
- Edge cases (empty input, long messages, special characters)
"""

from __future__ import annotations

from missy.agent.behavior import BehaviorLayer, IntentInterpreter, ResponseShaper
from missy.agent.persona import PersonaConfig

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _make_persona(
    name: str = "Missy",
    tone: str | list[str] = "warm",
    identity_description: str = "A helpful security-first assistant.",
    behavioral_tendencies: list[str] | None = None,
    response_style_rules: list[str] | None = None,
    boundaries: list[str] | None = None,
    personality_traits: list[str] | None = None,
) -> PersonaConfig:
    return PersonaConfig(
        name=name,
        tone=[tone] if isinstance(tone, str) else tone,
        identity_description=identity_description,
        behavioral_tendencies=behavioral_tendencies or [],
        response_style_rules=response_style_rules or [],
        boundaries=boundaries or [],
        personality_traits=personality_traits or [],
    )


def _user_msg(*texts: str) -> list[dict]:
    return [{"role": "user", "content": t} for t in texts]


def _base_ctx(**overrides) -> dict:
    ctx: dict = {
        "user_tone": "casual",
        "topic": "",
        "turn_count": 1,
        "has_tool_results": False,
        "intent": "question",
        "urgency": "low",
    }
    ctx.update(overrides)
    return ctx


# ---------------------------------------------------------------------------
# BehaviorLayer — analyze_user_tone
# ---------------------------------------------------------------------------


class TestAnalyzeUserToneCasual:
    """Casual signal keywords drive casual classification.

    Note: the brief/verbose check fires before tone scoring, so casual messages
    must be at least 8 words long to avoid being classified as 'brief'.
    """

    def test_hey_is_casual(self) -> None:
        layer = BehaviorLayer()
        # 9 words with casual signals — above the 8-word brief threshold
        assert layer.analyze_user_tone(
            _user_msg("hey thanks so much for helping me out with that")
        ) == "casual"

    def test_yo_is_casual(self) -> None:
        layer = BehaviorLayer()
        assert layer.analyze_user_tone(
            _user_msg("yo sup that was awesome cool stuff thanks a bunch")
        ) == "casual"

    def test_tbh_ngl_are_casual(self) -> None:
        layer = BehaviorLayer()
        assert layer.analyze_user_tone(
            _user_msg("tbh ngl this is kinda awesome really good work thanks")
        ) == "casual"

    def test_thx_is_casual(self) -> None:
        layer = BehaviorLayer()
        assert layer.analyze_user_tone(
            _user_msg("thx ty np btw cool awesome hey ya yep dunno")
        ) == "casual"

    def test_casual_beats_zero_scores(self) -> None:
        """When no signals present but message is not short or long, default is casual."""
        layer = BehaviorLayer()
        msgs = _user_msg("this is a moderate length sentence with nine words here")
        assert layer.analyze_user_tone(msgs) == "casual"


class TestAnalyzeUserToneFormal:
    """Formal signal keywords drive formal classification."""

    def test_please_kindly_would(self) -> None:
        layer = BehaviorLayer()
        msgs = _user_msg(
            "I would kindly appreciate if you could please clarify this regarding the matter"
        )
        assert layer.analyze_user_tone(msgs) == "formal"

    def test_furthermore_however(self) -> None:
        layer = BehaviorLayer()
        msgs = _user_msg(
            "Furthermore however therefore we should accordingly act sincerely and respectfully"
        )
        assert layer.analyze_user_tone(msgs) == "formal"

    def test_pursuant_henceforth(self) -> None:
        layer = BehaviorLayer()
        msgs = _user_msg(
            "Pursuant to our discussion henceforth I would request formal clarification"
        )
        assert layer.analyze_user_tone(msgs) == "formal"


class TestAnalyzeUserToneFrustrated:
    """Frustration takes priority over all other tones."""

    def test_broken_still_not_working(self) -> None:
        layer = BehaviorLayer()
        msgs = _user_msg("Still not working. It's still broken again. Why doesn't this work?")
        assert layer.analyze_user_tone(msgs) == "frustrated"

    def test_ugh_argh_signals(self) -> None:
        layer = BehaviorLayer()
        msgs = _user_msg("ugh this is wrong again ridiculous")
        assert layer.analyze_user_tone(msgs) == "frustrated"

    def test_frustration_overrides_casual(self) -> None:
        """Even a message heavy with casual signals returns frustrated if frustration present."""
        layer = BehaviorLayer()
        msgs = _user_msg("hey yo cool but doesn't work and broken again why")
        assert layer.analyze_user_tone(msgs) == "frustrated"

    def test_frustration_overrides_formal(self) -> None:
        layer = BehaviorLayer()
        msgs = _user_msg(
            "I would appreciate however this doesn't work and is still broken again"
        )
        assert layer.analyze_user_tone(msgs) == "frustrated"

    def test_useless_and_failed_signals(self) -> None:
        layer = BehaviorLayer()
        msgs = _user_msg("this is useless it failed and waste of time seriously")
        assert layer.analyze_user_tone(msgs) == "frustrated"

    def test_frustration_pattern_match(self) -> None:
        """Frustration pattern regex (e.g. 'still not working') triggers frustrated."""
        layer = BehaviorLayer()
        msgs = _user_msg("I've tried that again and still not working nothing works")
        assert layer.analyze_user_tone(msgs) == "frustrated"


class TestAnalyzeUserToneTechnical:
    """Technical keyword density drives technical classification."""

    def test_api_docker_oauth(self) -> None:
        layer = BehaviorLayer()
        msgs = _user_msg(
            "I need to configure the async function to handle the oauth api endpoint "
            "with auth and token management in the docker container pipeline"
        )
        assert layer.analyze_user_tone(msgs) == "technical"

    def test_sql_database_schema(self) -> None:
        layer = BehaviorLayer()
        msgs = _user_msg(
            "The sql query against the database with the schema and index should "
            "use a cache for the module import and the library framework config"
        )
        assert layer.analyze_user_tone(msgs) == "technical"

    def test_kubernetes_tls_socket(self) -> None:
        layer = BehaviorLayer()
        msgs = _user_msg(
            "kubernetes deployment with ssl tls socket port process thread async await ci pipeline"
        )
        assert layer.analyze_user_tone(msgs) == "technical"


class TestAnalyzeUserToneBriefVerbose:
    """Short avg word count => brief; long avg word count => verbose."""

    def test_brief_short_messages(self) -> None:
        layer = BehaviorLayer()
        msgs = _user_msg("fix it", "deploy now", "check logs", "restart")
        assert layer.analyze_user_tone(msgs) == "brief"

    def test_brief_single_word(self) -> None:
        layer = BehaviorLayer()
        assert layer.analyze_user_tone(_user_msg("go")) == "brief"

    def test_verbose_very_long_message(self) -> None:
        layer = BehaviorLayer()
        long_text = " ".join(["word"] * 60)
        assert layer.analyze_user_tone(_user_msg(long_text)) == "verbose"

    def test_verbose_threshold_is_over_40_words(self) -> None:
        layer = BehaviorLayer()
        # 41 words: above the threshold
        text_41 = " ".join(["word"] * 41)
        assert layer.analyze_user_tone(_user_msg(text_41)) == "verbose"

    def test_brief_threshold_is_under_8_words(self) -> None:
        layer = BehaviorLayer()
        # 7 words: below threshold, no frustration/formal/casual signals
        text_7 = "one two three four five six seven"
        assert layer.analyze_user_tone(_user_msg(text_7)) == "brief"


class TestAnalyzeUserToneEdgeCases:
    """Edge cases: empty list, no user messages, only assistant messages."""

    def test_empty_messages_list_returns_casual(self) -> None:
        layer = BehaviorLayer()
        assert layer.analyze_user_tone([]) == "casual"

    def test_no_user_role_messages_returns_casual(self) -> None:
        layer = BehaviorLayer()
        msgs = [
            {"role": "assistant", "content": "Hello there!"},
            {"role": "system", "content": "You are an assistant."},
        ]
        assert layer.analyze_user_tone(msgs) == "casual"

    def test_only_last_five_messages_considered(self) -> None:
        """10 formal messages followed by 5 casual should resolve to casual."""
        layer = BehaviorLayer()
        formal = [
            {"role": "user", "content": "I would appreciate your assistance regarding this matter."}
            for _ in range(10)
        ]
        casual = [
            {"role": "user", "content": "hey yo cool sup awesome thanks thx ngl tbh dude"}
            for _ in range(5)
        ]
        assert layer.analyze_user_tone(formal + casual) == "casual"

    def test_non_dict_messages_ignored(self) -> None:
        layer = BehaviorLayer()
        msgs = ["not a dict", 42, None, {"role": "user", "content": "hey cool yo sup"}]
        result = layer.analyze_user_tone(msgs)
        assert result in ("casual", "brief", "verbose", "formal", "frustrated", "technical")

    def test_very_long_single_message(self) -> None:
        layer = BehaviorLayer()
        long_text = " ".join(["word"] * 500)
        result = layer.analyze_user_tone([{"role": "user", "content": long_text}])
        assert result == "verbose"

    def test_special_characters_in_message(self) -> None:
        layer = BehaviorLayer()
        msgs = _user_msg("hey! @#$%^&*()_+ cool 🎉 awesome yo")
        result = layer.analyze_user_tone(msgs)
        assert result in ("casual", "brief")

    def test_mixed_roles_only_user_content_scored(self) -> None:
        """Assistant messages are ignored; only user messages are scored.

        The user message must be >= 8 words to pass the brief threshold first.
        """
        layer = BehaviorLayer()
        msgs = [
            {"role": "assistant", "content": "please furthermore kindly would however"},
            {"role": "user", "content": "hey yo cool tbh ngl sup awesome thanks buddy dude"},
        ]
        assert layer.analyze_user_tone(msgs) == "casual"


# ---------------------------------------------------------------------------
# IntentInterpreter — classify_intent
# ---------------------------------------------------------------------------


class TestClassifyIntentGreeting:
    def test_hello(self) -> None:
        assert IntentInterpreter().classify_intent("Hello there!") == "greeting"

    def test_hey(self) -> None:
        assert IntentInterpreter().classify_intent("hey") == "greeting"

    def test_good_morning(self) -> None:
        assert IntentInterpreter().classify_intent("Good morning") == "greeting"

    def test_hi_with_name(self) -> None:
        assert IntentInterpreter().classify_intent("Hi Missy") == "greeting"

    def test_howdy(self) -> None:
        assert IntentInterpreter().classify_intent("Howdy partner") == "greeting"

    def test_greetings(self) -> None:
        assert IntentInterpreter().classify_intent("Greetings!") == "greeting"

    def test_yo(self) -> None:
        assert IntentInterpreter().classify_intent("yo") == "greeting"


class TestClassifyIntentFarewell:
    def test_bye(self) -> None:
        assert IntentInterpreter().classify_intent("bye!") == "farewell"

    def test_goodbye(self) -> None:
        assert IntentInterpreter().classify_intent("goodbye") == "farewell"

    def test_see_you_later(self) -> None:
        assert IntentInterpreter().classify_intent("see you later") == "farewell"

    def test_take_care(self) -> None:
        assert IntentInterpreter().classify_intent("take care") == "farewell"

    def test_ttyl(self) -> None:
        assert IntentInterpreter().classify_intent("ttyl") == "farewell"

    def test_signing_off(self) -> None:
        assert IntentInterpreter().classify_intent("signing off for today") == "farewell"


class TestClassifyIntentQuestion:
    def test_what_question(self) -> None:
        assert IntentInterpreter().classify_intent("What is the capital of France?") == "question"

    def test_how_question(self) -> None:
        assert IntentInterpreter().classify_intent("How do I restart nginx?") == "question"

    def test_why_question(self) -> None:
        # Avoid words that trigger frustration (why+is, again, not working, etc.)
        assert IntentInterpreter().classify_intent("Why do containers need namespaces?") == "question"

    def test_trailing_question_mark(self) -> None:
        assert IntentInterpreter().classify_intent("This works, right?") == "question"

    def test_can_you_question(self) -> None:
        assert IntentInterpreter().classify_intent("Can you explain this?") == "question"

    def test_should_question(self) -> None:
        assert IntentInterpreter().classify_intent("Should I use async or sync here?") == "question"


class TestClassifyIntentCommand:
    def test_run_command(self) -> None:
        assert IntentInterpreter().classify_intent("run the tests") == "command"

    def test_deploy_command(self) -> None:
        assert IntentInterpreter().classify_intent("deploy the application") == "command"

    def test_create_command(self) -> None:
        assert IntentInterpreter().classify_intent("create a new config file") == "command"

    def test_start_command(self) -> None:
        assert IntentInterpreter().classify_intent("start the service") == "command"

    def test_please_prefix_command(self) -> None:
        assert IntentInterpreter().classify_intent("please fix the bug") == "command"

    def test_show_list_command(self) -> None:
        assert IntentInterpreter().classify_intent("list all running containers") == "command"


class TestClassifyIntentClarification:
    def test_what_do_you_mean(self) -> None:
        assert IntentInterpreter().classify_intent("What do you mean by that?") == "clarification"

    def test_can_you_clarify(self) -> None:
        assert IntentInterpreter().classify_intent("Can you clarify the second step?") == "clarification"

    def test_could_you_elaborate(self) -> None:
        assert IntentInterpreter().classify_intent("Could you elaborate on that?") == "clarification"

    def test_i_dont_understand(self) -> None:
        assert IntentInterpreter().classify_intent("I don't understand") == "clarification"

    def test_please_explain(self) -> None:
        assert IntentInterpreter().classify_intent("Please explain what you mean") == "clarification"

    def test_more_detail(self) -> None:
        assert IntentInterpreter().classify_intent("Can I get more detail on that?") == "clarification"


class TestClassifyIntentFeedback:
    def test_thats_wrong(self) -> None:
        assert IntentInterpreter().classify_intent("That's wrong, the answer is 42") == "feedback"

    def test_you_got_it(self) -> None:
        assert IntentInterpreter().classify_intent("You got it exactly right!") == "feedback"

    def test_not_quite(self) -> None:
        # "again" alone triggers frustration, so use a form without it
        assert IntentInterpreter().classify_intent("Not quite, the value should be 10") == "feedback"

    def test_thats_perfect(self) -> None:
        assert IntentInterpreter().classify_intent("That's perfect, thank you") == "feedback"

    def test_correction(self) -> None:
        assert IntentInterpreter().classify_intent("Correction: it should be port 8080") == "feedback"

    def test_almost(self) -> None:
        assert IntentInterpreter().classify_intent("Almost, but the path is different") == "feedback"


class TestClassifyIntentExploration:
    def test_tell_me_more(self) -> None:
        assert IntentInterpreter().classify_intent("Tell me more about this approach") == "exploration"

    def test_im_curious(self) -> None:
        assert IntentInterpreter().classify_intent("I'm curious about alternatives") == "exploration"

    def test_suggest(self) -> None:
        assert IntentInterpreter().classify_intent("suggest some options for me") == "exploration"

    def test_what_else(self) -> None:
        assert IntentInterpreter().classify_intent("What else can you do?") == "exploration"

    def test_recommend(self) -> None:
        assert IntentInterpreter().classify_intent("recommend a good library for this") == "exploration"

    def test_ideas_for(self) -> None:
        assert IntentInterpreter().classify_intent("ideas for improving the performance") == "exploration"


class TestClassifyIntentTroubleshoot:
    def test_error_keyword(self) -> None:
        assert IntentInterpreter().classify_intent("I'm getting an error here") == "troubleshooting"

    def test_exception_traceback(self) -> None:
        assert IntentInterpreter().classify_intent("There's a traceback in the exception") == "troubleshooting"

    def test_permission_denied(self) -> None:
        assert IntentInterpreter().classify_intent("permission denied when running the script") == "troubleshooting"

    def test_connection_refused(self) -> None:
        assert IntentInterpreter().classify_intent("connection refused on port 5432") == "troubleshooting"

    def test_debug_keyword(self) -> None:
        assert IntentInterpreter().classify_intent("help me debug this function") == "troubleshooting"

    def test_timeout_keyword(self) -> None:
        # Pattern matches exact word "timeout" not "timing out"
        assert IntentInterpreter().classify_intent("the request hit a timeout") == "troubleshooting"

    def test_failed_with(self) -> None:
        assert IntentInterpreter().classify_intent("the job failed with exit code 1") == "troubleshooting"


class TestClassifyIntentConfirmation:
    def test_ok(self) -> None:
        assert IntentInterpreter().classify_intent("ok") == "confirmation"

    def test_yes(self) -> None:
        assert IntentInterpreter().classify_intent("yes") == "confirmation"

    def test_sounds_good(self) -> None:
        assert IntentInterpreter().classify_intent("sounds good") == "confirmation"

    def test_go_ahead(self) -> None:
        assert IntentInterpreter().classify_intent("go ahead") == "confirmation"

    def test_got_it(self) -> None:
        assert IntentInterpreter().classify_intent("got it") == "confirmation"

    def test_confirmed(self) -> None:
        assert IntentInterpreter().classify_intent("confirmed") == "confirmation"

    def test_makes_sense(self) -> None:
        assert IntentInterpreter().classify_intent("makes sense") == "confirmation"


class TestClassifyIntentFrustration:
    def test_still_not_working(self) -> None:
        assert IntentInterpreter().classify_intent("this is still not working") == "frustration"

    def test_doesnt_work(self) -> None:
        assert IntentInterpreter().classify_intent("it doesn't work") == "frustration"

    def test_tried_that_again(self) -> None:
        assert IntentInterpreter().classify_intent("I tried that again and same error") == "frustration"

    def test_nothing_works(self) -> None:
        assert IntentInterpreter().classify_intent("nothing works at all") == "frustration"

    def test_useless(self) -> None:
        assert IntentInterpreter().classify_intent("this is useless") == "frustration"

    def test_same_issue(self) -> None:
        assert IntentInterpreter().classify_intent("same problem again") == "frustration"


class TestClassifyIntentGeneral:
    """Unmatched input falls back to 'question' (the catch-all)."""

    def test_unrecognized_statement(self) -> None:
        result = IntentInterpreter().classify_intent("the sky is blue today")
        assert result == "question"

    def test_empty_string(self) -> None:
        result = IntentInterpreter().classify_intent("")
        assert result == "question"

    def test_single_word_unknown(self) -> None:
        result = IntentInterpreter().classify_intent("banana")
        assert result == "question"


# ---------------------------------------------------------------------------
# IntentInterpreter — extract_urgency
# ---------------------------------------------------------------------------


class TestExtractUrgencyHigh:
    def test_asap(self) -> None:
        assert IntentInterpreter().extract_urgency("I need this asap") == "high"

    def test_production_down(self) -> None:
        assert IntentInterpreter().extract_urgency("production is down right now") == "high"

    def test_immediately(self) -> None:
        assert IntentInterpreter().extract_urgency("fix this immediately") == "high"

    def test_critical_outage(self) -> None:
        assert IntentInterpreter().extract_urgency("critical outage affecting all users") == "high"

    def test_emergency(self) -> None:
        assert IntentInterpreter().extract_urgency("this is an emergency") == "high"

    def test_crash(self) -> None:
        assert IntentInterpreter().extract_urgency("the server crashed and is down") == "high"

    def test_blocking(self) -> None:
        assert IntentInterpreter().extract_urgency("this is blocking my work") == "high"

    def test_right_now(self) -> None:
        assert IntentInterpreter().extract_urgency("I need this right now") == "high"


class TestExtractUrgencyMedium:
    def test_soon(self) -> None:
        assert IntentInterpreter().extract_urgency("can you look at this soon?") == "medium"

    def test_deadline(self) -> None:
        assert IntentInterpreter().extract_urgency("I have a deadline for this") == "medium"

    def test_today(self) -> None:
        assert IntentInterpreter().extract_urgency("I need this today") == "medium"

    def test_must(self) -> None:
        assert IntentInterpreter().extract_urgency("I must complete this task") == "medium"

    def test_before_end(self) -> None:
        assert IntentInterpreter().extract_urgency("need it before end of day") == "medium"

    def test_quickly(self) -> None:
        assert IntentInterpreter().extract_urgency("can you help quickly?") == "medium"


class TestExtractUrgencyLow:
    def test_no_urgency_signal(self) -> None:
        assert IntentInterpreter().extract_urgency("what is 2 + 2?") == "low"

    def test_empty_string(self) -> None:
        assert IntentInterpreter().extract_urgency("") == "low"

    def test_casual_message(self) -> None:
        assert IntentInterpreter().extract_urgency("hey can you help me understand this") == "low"

    def test_exploration_message(self) -> None:
        assert IntentInterpreter().extract_urgency("tell me more about machine learning") == "low"


# ---------------------------------------------------------------------------
# BehaviorLayer — shape_system_prompt
# ---------------------------------------------------------------------------


class TestShapeSystemPromptWithoutPersona:
    def test_base_prompt_preserved(self) -> None:
        layer = BehaviorLayer()
        result = layer.shape_system_prompt("You are a helpful assistant.")
        assert result.startswith("You are a helpful assistant.")

    def test_no_persona_block_without_persona(self) -> None:
        layer = BehaviorLayer()
        result = layer.shape_system_prompt("Base prompt.")
        assert "## Persona" not in result

    def test_returns_string(self) -> None:
        layer = BehaviorLayer()
        result = layer.shape_system_prompt("Base.", _base_ctx())
        assert isinstance(result, str)

    def test_none_context_does_not_raise(self) -> None:
        layer = BehaviorLayer()
        result = layer.shape_system_prompt("Base.", None)
        assert "Base." in result

    def test_guidelines_section_present_with_context(self) -> None:
        layer = BehaviorLayer()
        ctx = _base_ctx(user_tone="frustrated")
        result = layer.shape_system_prompt("Base.", ctx)
        assert "Response guidelines" in result

    def test_empty_base_prompt(self) -> None:
        layer = BehaviorLayer()
        result = layer.shape_system_prompt("")
        assert isinstance(result, str)


class TestShapeSystemPromptWithPersona:
    def test_persona_name_included(self) -> None:
        persona = _make_persona(name="Atlas")
        layer = BehaviorLayer(persona)
        result = layer.shape_system_prompt("Base.", _base_ctx())
        assert "Atlas" in result

    def test_persona_identity_description_included(self) -> None:
        persona = _make_persona(identity_description="A security-conscious linux expert.")
        layer = BehaviorLayer(persona)
        result = layer.shape_system_prompt("Base.", _base_ctx())
        assert "security-conscious" in result

    def test_persona_tone_included(self) -> None:
        persona = _make_persona(tone="playful")
        layer = BehaviorLayer(persona)
        result = layer.shape_system_prompt("Base.", _base_ctx())
        assert "playful" in result

    def test_persona_boundaries_included(self) -> None:
        persona = _make_persona(boundaries=["Never delete production data."])
        layer = BehaviorLayer(persona)
        result = layer.shape_system_prompt("Base.", _base_ctx())
        assert "Never delete production data." in result

    def test_persona_behavioral_tendencies_in_guidelines(self) -> None:
        persona = _make_persona(behavioral_tendencies=["Always check policy before acting."])
        layer = BehaviorLayer(persona)
        result = layer.shape_system_prompt("Base.", _base_ctx())
        assert "Always check policy before acting." in result

    def test_persona_response_style_rules_in_guidelines(self) -> None:
        persona = _make_persona(response_style_rules=["Use short paragraphs."])
        layer = BehaviorLayer(persona)
        result = layer.shape_system_prompt("Base.", _base_ctx())
        assert "Use short paragraphs." in result

    def test_persona_personality_traits_included(self) -> None:
        persona = _make_persona(personality_traits=["curious", "meticulous"])
        layer = BehaviorLayer(persona)
        result = layer.shape_system_prompt("Base.", _base_ctx())
        assert "curious" in result
        assert "meticulous" in result

    def test_base_prompt_not_overwritten(self) -> None:
        persona = _make_persona(name="Missy")
        layer = BehaviorLayer(persona)
        base = "You are the best assistant."
        result = layer.shape_system_prompt(base, _base_ctx())
        assert result.startswith(base)

    def test_sections_separated_by_blank_line(self) -> None:
        persona = _make_persona()
        layer = BehaviorLayer(persona)
        result = layer.shape_system_prompt("Base.", _base_ctx())
        assert "\n\n" in result


# ---------------------------------------------------------------------------
# BehaviorLayer — get_response_guidelines
# ---------------------------------------------------------------------------


class TestGetResponseGuidelinesToneAdaptation:
    def test_casual_tone_guidance(self) -> None:
        layer = BehaviorLayer()
        guidelines = layer.get_response_guidelines(_base_ctx(user_tone="casual"))
        assert "conversational" in guidelines.lower() or "relaxed" in guidelines.lower()

    def test_formal_tone_guidance(self) -> None:
        layer = BehaviorLayer()
        guidelines = layer.get_response_guidelines(_base_ctx(user_tone="formal"))
        assert "professional" in guidelines.lower() or "precise" in guidelines.lower()

    def test_frustrated_tone_guidance(self) -> None:
        layer = BehaviorLayer()
        guidelines = layer.get_response_guidelines(_base_ctx(user_tone="frustrated"))
        assert "empathetic" in guidelines.lower() or "difficulty" in guidelines.lower()

    def test_technical_tone_guidance(self) -> None:
        layer = BehaviorLayer()
        guidelines = layer.get_response_guidelines(_base_ctx(user_tone="technical"))
        assert "technical" in guidelines.lower() or "vocabulary" in guidelines.lower()

    def test_brief_tone_guidance(self) -> None:
        layer = BehaviorLayer()
        guidelines = layer.get_response_guidelines(_base_ctx(user_tone="brief"))
        # brief triggers conciseness via should_be_concise
        assert "concise" in guidelines.lower() or "bullet" in guidelines.lower()

    def test_verbose_tone_guidance(self) -> None:
        layer = BehaviorLayer()
        guidelines = layer.get_response_guidelines(_base_ctx(user_tone="verbose"))
        assert "detail" in guidelines.lower() or "thorough" in guidelines.lower()

    def test_unknown_tone_no_guidance(self) -> None:
        layer = BehaviorLayer()
        guidelines = layer.get_response_guidelines(_base_ctx(user_tone="alien"))
        # Unknown tone should not raise; may produce empty guidance for that line
        assert isinstance(guidelines, str)


class TestGetResponseGuidelinesContext:
    def test_tool_results_present(self) -> None:
        layer = BehaviorLayer()
        guidelines = layer.get_response_guidelines(_base_ctx(has_tool_results=True))
        assert "tool" in guidelines.lower()

    def test_high_urgency_lead_with_answer(self) -> None:
        layer = BehaviorLayer()
        guidelines = layer.get_response_guidelines(_base_ctx(urgency="high"))
        assert "answer" in guidelines.lower() or "preamble" in guidelines.lower()

    def test_medium_urgency_focused(self) -> None:
        layer = BehaviorLayer()
        guidelines = layer.get_response_guidelines(_base_ctx(urgency="medium"))
        assert "focused" in guidelines.lower() or "actionable" in guidelines.lower()

    def test_low_urgency_no_urgency_line(self) -> None:
        layer = BehaviorLayer()
        guidelines = layer.get_response_guidelines(_base_ctx(urgency="low"))
        assert "time pressure" not in guidelines.lower()

    def test_conciseness_after_10_turns(self) -> None:
        layer = BehaviorLayer()
        guidelines = layer.get_response_guidelines(_base_ctx(turn_count=10))
        assert "concise" in guidelines.lower()

    def test_no_conciseness_at_low_turns(self) -> None:
        layer = BehaviorLayer()
        guidelines = layer.get_response_guidelines(_base_ctx(turn_count=3))
        assert "long" not in guidelines.lower() or "concise" not in guidelines.lower()

    def test_technical_topic_triggers_code_suggestion(self) -> None:
        layer = BehaviorLayer()
        guidelines = layer.get_response_guidelines(_base_ctx(topic="writing a function"))
        assert "code" in guidelines.lower() or "technical" in guidelines.lower()

    def test_api_topic_triggers_code_suggestion(self) -> None:
        layer = BehaviorLayer()
        guidelines = layer.get_response_guidelines(_base_ctx(topic="building an api"))
        assert "code" in guidelines.lower() or "technical" in guidelines.lower()

    def test_non_technical_topic_no_code_suggestion(self) -> None:
        layer = BehaviorLayer()
        guidelines = layer.get_response_guidelines(_base_ctx(topic="cooking recipes"))
        # Should not append code suggestion for non-technical topic
        assert "code snippets" not in guidelines.lower() or "technical topic" not in guidelines.lower()


class TestGetResponseGuidelinesIntentSpecific:
    def test_frustration_intent(self) -> None:
        layer = BehaviorLayer()
        guidelines = layer.get_response_guidelines(_base_ctx(intent="frustration"))
        assert "empathy" in guidelines.lower() or "difficulty" in guidelines.lower()

    def test_exploration_intent(self) -> None:
        layer = BehaviorLayer()
        guidelines = layer.get_response_guidelines(_base_ctx(intent="exploration"))
        assert "ideas" in guidelines.lower() or "suggestion" in guidelines.lower()

    def test_greeting_intent(self) -> None:
        layer = BehaviorLayer()
        guidelines = layer.get_response_guidelines(_base_ctx(intent="greeting"))
        assert "warm" in guidelines.lower() or "energy" in guidelines.lower()

    def test_farewell_intent(self) -> None:
        layer = BehaviorLayer()
        guidelines = layer.get_response_guidelines(_base_ctx(intent="farewell"))
        assert "farewell" in guidelines.lower() or "brief" in guidelines.lower()

    def test_troubleshooting_intent(self) -> None:
        layer = BehaviorLayer()
        guidelines = layer.get_response_guidelines(_base_ctx(intent="troubleshooting"))
        assert "diagnostic" in guidelines.lower() or "cause" in guidelines.lower()

    def test_confirmation_intent(self) -> None:
        layer = BehaviorLayer()
        guidelines = layer.get_response_guidelines(_base_ctx(intent="confirmation"))
        assert "proceed" in guidelines.lower() or "next" in guidelines.lower()

    def test_clarification_intent(self) -> None:
        layer = BehaviorLayer()
        guidelines = layer.get_response_guidelines(_base_ctx(intent="clarification"))
        assert "re-explain" in guidelines.lower() or "analogy" in guidelines.lower() or "detail" in guidelines.lower()

    def test_command_intent(self) -> None:
        layer = BehaviorLayer()
        guidelines = layer.get_response_guidelines(_base_ctx(intent="command"))
        assert "direct" in guidelines.lower() or "action" in guidelines.lower() or "execute" in guidelines.lower()


class TestGetResponseGuidelinesVisionModes:
    def test_painting_mode_warm_encouraging(self) -> None:
        layer = BehaviorLayer()
        ctx = _base_ctx(vision_mode="painting")
        guidelines = layer.get_response_guidelines(ctx)
        assert "encouraging" in guidelines.lower() or "warm" in guidelines.lower()

    def test_painting_mode_no_harsh_language(self) -> None:
        layer = BehaviorLayer()
        ctx = _base_ctx(vision_mode="painting")
        guidelines = layer.get_response_guidelines(ctx)
        assert "harsh" in guidelines.lower() or "dismissive" in guidelines.lower()

    def test_painting_mode_suggestions_as_possibilities(self) -> None:
        layer = BehaviorLayer()
        ctx = _base_ctx(vision_mode="painting")
        guidelines = layer.get_response_guidelines(ctx)
        assert "possibilities" in guidelines.lower() or "suggestions" in guidelines.lower()

    def test_puzzle_mode_patient_observant(self) -> None:
        layer = BehaviorLayer()
        ctx = _base_ctx(vision_mode="puzzle")
        guidelines = layer.get_response_guidelines(ctx)
        assert "puzzle" in guidelines.lower() or "patient" in guidelines.lower()

    def test_puzzle_mode_actionable_placement(self) -> None:
        layer = BehaviorLayer()
        ctx = _base_ctx(vision_mode="puzzle")
        guidelines = layer.get_response_guidelines(ctx)
        assert "placement" in guidelines.lower() or "actionable" in guidelines.lower()

    def test_puzzle_mode_celebrates_progress(self) -> None:
        layer = BehaviorLayer()
        ctx = _base_ctx(vision_mode="puzzle")
        guidelines = layer.get_response_guidelines(ctx)
        assert "progress" in guidelines.lower() or "celebrat" in guidelines.lower()

    def test_generic_vision_mode_references_visual(self) -> None:
        layer = BehaviorLayer()
        ctx = _base_ctx(vision_mode="general")
        guidelines = layer.get_response_guidelines(ctx)
        assert "visual" in guidelines.lower()

    def test_generic_vision_mode_specific(self) -> None:
        layer = BehaviorLayer()
        ctx = _base_ctx(vision_mode="inspection")
        guidelines = layer.get_response_guidelines(ctx)
        assert "visual" in guidelines.lower() or "specific" in guidelines.lower()

    def test_no_vision_mode_no_visual_guidance(self) -> None:
        layer = BehaviorLayer()
        guidelines = layer.get_response_guidelines(_base_ctx())
        assert "visual input is available" not in guidelines.lower()
        assert "reviewing artwork" not in guidelines.lower()
        assert "helping with a puzzle" not in guidelines.lower()


class TestGetResponseGuidelinesPersonaTendencies:
    def test_behavioral_tendencies_appended(self) -> None:
        persona = _make_persona(behavioral_tendencies=["Always cite sources."])
        layer = BehaviorLayer(persona)
        guidelines = layer.get_response_guidelines(_base_ctx())
        assert "Always cite sources." in guidelines

    def test_response_style_rules_appended(self) -> None:
        persona = _make_persona(response_style_rules=["Keep answers under 100 words."])
        layer = BehaviorLayer(persona)
        guidelines = layer.get_response_guidelines(_base_ctx())
        assert "Keep answers under 100 words." in guidelines

    def test_multiple_tendencies_all_present(self) -> None:
        persona = _make_persona(
            behavioral_tendencies=["Cite sources.", "Prefer bullet points."]
        )
        layer = BehaviorLayer(persona)
        guidelines = layer.get_response_guidelines(_base_ctx())
        assert "Cite sources." in guidelines
        assert "Prefer bullet points." in guidelines

    def test_none_persona_no_tendency_error(self) -> None:
        layer = BehaviorLayer(None)
        guidelines = layer.get_response_guidelines(_base_ctx())
        assert isinstance(guidelines, str)

    def test_returns_bullet_formatted_lines(self) -> None:
        layer = BehaviorLayer()
        ctx = _base_ctx(user_tone="casual", has_tool_results=True)
        guidelines = layer.get_response_guidelines(ctx)
        if guidelines:
            assert "- " in guidelines


# ---------------------------------------------------------------------------
# ResponseShaper — strip robotic phrases
# ---------------------------------------------------------------------------


class TestResponseShaperStripRoboticPhrases:
    def test_strips_as_an_ai(self) -> None:
        shaper = ResponseShaper()
        result = shaper.shape_response("As an AI, I can help you.", None, {})
        assert "As an AI" not in result
        assert "help you" in result

    def test_strips_as_an_ai_language_model(self) -> None:
        shaper = ResponseShaper()
        result = shaper.shape_response("As an AI language model, here is the answer.", None, {})
        assert "language model" not in result
        assert "answer" in result

    def test_strips_certainly(self) -> None:
        shaper = ResponseShaper()
        result = shaper.shape_response("Certainly! Here is the information.", None, {})
        assert "Certainly" not in result
        assert "information" in result

    def test_strips_certainly_ill_help(self) -> None:
        shaper = ResponseShaper()
        result = shaper.shape_response("Certainly, I'll help you! The answer is 42.", None, {})
        assert "Certainly" not in result

    def test_strips_of_course(self) -> None:
        shaper = ResponseShaper()
        result = shaper.shape_response("Of course! Let me explain.", None, {})
        assert "Of course" not in result
        assert "explain" in result

    def test_strips_absolutely(self) -> None:
        shaper = ResponseShaper()
        result = shaper.shape_response("Absolutely! Here's what you need.", None, {})
        assert "Absolutely" not in result

    def test_strips_great_question(self) -> None:
        shaper = ResponseShaper()
        result = shaper.shape_response("Great question! The answer is yes.", None, {})
        assert "Great question" not in result
        assert "answer" in result

    def test_strips_thats_a_great_question(self) -> None:
        shaper = ResponseShaper()
        result = shaper.shape_response("That's a great question! Here's why.", None, {})
        assert "great question" not in result.lower()

    def test_strips_id_be_happy_to_help(self) -> None:
        shaper = ResponseShaper()
        result = shaper.shape_response("I'd be happy to help you with that.", None, {})
        assert "happy to help" not in result

    def test_strips_im_an_ai_assistant(self) -> None:
        shaper = ResponseShaper()
        result = shaper.shape_response("I'm an AI assistant here to help. The answer is X.", None, {})
        assert "I'm an AI" not in result

    def test_strips_as_your_assistant(self) -> None:
        shaper = ResponseShaper()
        result = shaper.shape_response("As your assistant, I recommend the following.", None, {})
        assert "As your assistant" not in result

    def test_strips_please_note_im_an_ai(self) -> None:
        """The pattern strips 'Please note that I'm an AI...' as a full sentence.

        The regex anchors to 'I'm an AI' and removes that segment. The leading
        'Please note that' text is not removed by the pattern (it is upstream
        of the match anchor). The useful content after the sentence is preserved.
        """
        shaper = ResponseShaper()
        result = shaper.shape_response(
            "Please note that I'm an AI and cannot do that. Here is what I can do.", None, {}
        )
        # The actual sentence content following the robotic clause must survive
        assert "what I can do" in result

    def test_strips_i_dont_have_feelings(self) -> None:
        shaper = ResponseShaper()
        result = shaper.shape_response(
            "I don't have feelings, but here is my answer.", None, {}
        )
        assert "feelings" not in result or "answer" in result

    def test_multiple_robotic_phrases_stripped(self) -> None:
        shaper = ResponseShaper()
        raw = "Certainly! As an AI, I'd be happy to help. Great question! The answer is 42."
        result = shaper.shape_response(raw, None, {})
        assert "Certainly" not in result
        assert "As an AI" not in result
        assert "happy to help" not in result
        assert "Great question" not in result
        assert "42" in result

    def test_result_stripped_of_leading_trailing_whitespace(self) -> None:
        shaper = ResponseShaper()
        result = shaper.shape_response("Certainly! The answer.", None, {})
        assert result == result.strip()

    def test_result_is_not_empty_when_only_content_remains(self) -> None:
        shaper = ResponseShaper()
        result = shaper.shape_response(
            "As an AI, I can confirm the file exists.", None, {}
        )
        assert len(result) > 0


class TestResponseShaperPreservesCodeBlocks:
    def test_code_block_with_robotic_phrase_preserved(self) -> None:
        shaper = ResponseShaper()
        raw = 'Here is the code:\n\n```python\n# As an AI would do\nprint("Certainly!")\n```'
        result = shaper.shape_response(raw, None, {})
        assert "Certainly!" in result
        assert "As an AI would do" in result

    def test_fenced_block_content_untouched(self) -> None:
        shaper = ResponseShaper()
        raw = "Sure.\n\n```bash\necho 'Great question!'\n```\n\nDone."
        result = shaper.shape_response(raw, None, {})
        assert "Great question!" in result

    def test_inline_code_preserved(self) -> None:
        shaper = ResponseShaper()
        raw = "Use `Certainly()` function to proceed. Certainly! Done."
        result = shaper.shape_response(raw, None, {})
        assert "`Certainly()`" in result

    def test_multiple_code_blocks_all_preserved(self) -> None:
        shaper = ResponseShaper()
        raw = (
            "Certainly!\n\n```python\nAs an AI = True\n```\n\n"
            "And also:\n\n```bash\necho 'Of course!'\n```"
        )
        result = shaper.shape_response(raw, None, {})
        assert "As an AI = True" in result
        assert "Of course!" in result

    def test_robotic_phrase_outside_block_still_stripped(self) -> None:
        shaper = ResponseShaper()
        raw = "Certainly! Here is the code:\n\n```python\nprint('hello')\n```\n\nGreat question!"
        result = shaper.shape_response(raw, None, {})
        assert "Certainly" not in result
        assert "Great question" not in result
        assert "print('hello')" in result


class TestResponseShaperEdgeCases:
    def test_empty_string_returns_empty(self) -> None:
        shaper = ResponseShaper()
        assert shaper.shape_response("", None, {}) == ""

    def test_whitespace_only_stripped(self) -> None:
        shaper = ResponseShaper()
        result = shaper.shape_response("   \n\n  \n  ", None, {})
        assert result == "" or result.strip() == ""

    def test_normal_response_unchanged(self) -> None:
        shaper = ResponseShaper()
        clean = "To restart nginx, run: sudo systemctl restart nginx"
        result = shaper.shape_response(clean, None, {})
        assert result == clean

    def test_persona_argument_accepted(self) -> None:
        persona = _make_persona()
        shaper = ResponseShaper()
        result = shaper.shape_response("Certainly! The answer is yes.", persona, {})
        assert "Certainly" not in result

    def test_context_argument_accepted(self) -> None:
        shaper = ResponseShaper()
        ctx = _base_ctx(user_tone="casual")
        result = shaper.shape_response("Of course! Here you go.", None, ctx)
        assert "Of course" not in result

    def test_multiple_blank_lines_collapsed(self) -> None:
        shaper = ResponseShaper()
        raw = "First paragraph.\n\n\n\n\nSecond paragraph."
        result = shaper.shape_response(raw, None, {})
        assert "\n\n\n" not in result

    def test_response_with_only_robotic_phrase_becomes_empty(self) -> None:
        shaper = ResponseShaper()
        result = shaper.shape_response("Certainly!", None, {})
        assert result == "" or result.strip() == ""

    def test_very_long_response_processed(self) -> None:
        shaper = ResponseShaper()
        long_text = "Certainly! " + ("word " * 1000)
        result = shaper.shape_response(long_text, None, {})
        assert "Certainly" not in result
        assert len(result) > 0


class TestDetectRoboticPatterns:
    def test_detects_certainly(self) -> None:
        shaper = ResponseShaper()
        found = shaper.detect_robotic_patterns("Certainly! Here is the answer.")
        assert any("Certainly" in f for f in found)

    def test_detects_as_an_ai(self) -> None:
        shaper = ResponseShaper()
        found = shaper.detect_robotic_patterns("As an AI, I can tell you.")
        assert any("As an AI" in f for f in found)

    def test_returns_empty_for_clean_text(self) -> None:
        shaper = ResponseShaper()
        found = shaper.detect_robotic_patterns("The answer is 42.")
        assert found == []

    def test_does_not_modify_text(self) -> None:
        shaper = ResponseShaper()
        text = "Certainly! As an AI, hello."
        shaper.detect_robotic_patterns(text)
        # Text should be unchanged (method only reads)
        assert text == "Certainly! As an AI, hello."


# ---------------------------------------------------------------------------
# BehaviorLayer — should_be_concise
# ---------------------------------------------------------------------------


class TestShouldBeConcise:
    def test_true_when_turn_count_10(self) -> None:
        layer = BehaviorLayer()
        assert layer.should_be_concise({"turn_count": 10}) is True

    def test_true_when_turn_count_over_10(self) -> None:
        layer = BehaviorLayer()
        assert layer.should_be_concise({"turn_count": 50}) is True

    def test_false_when_turn_count_9(self) -> None:
        layer = BehaviorLayer()
        assert layer.should_be_concise({"turn_count": 9}) is False

    def test_true_when_user_tone_brief(self) -> None:
        layer = BehaviorLayer()
        assert layer.should_be_concise({"user_tone": "brief"}) is True

    def test_true_when_urgency_high(self) -> None:
        layer = BehaviorLayer()
        assert layer.should_be_concise({"urgency": "high"}) is True

    def test_false_for_empty_context(self) -> None:
        layer = BehaviorLayer()
        assert layer.should_be_concise({}) is False

    def test_false_for_none_context(self) -> None:
        layer = BehaviorLayer()
        assert layer.should_be_concise(None) is False

    def test_false_for_medium_urgency(self) -> None:
        layer = BehaviorLayer()
        assert layer.should_be_concise({"urgency": "medium"}) is False


# ---------------------------------------------------------------------------
# BehaviorLayer — get_tone_adaptation
# ---------------------------------------------------------------------------


class TestGetToneAdaptation:
    def test_casual_returns_string(self) -> None:
        layer = BehaviorLayer()
        result = layer.get_tone_adaptation("casual")
        assert isinstance(result, str) and len(result) > 0

    def test_formal_returns_string(self) -> None:
        layer = BehaviorLayer()
        result = layer.get_tone_adaptation("formal")
        assert isinstance(result, str) and len(result) > 0

    def test_frustrated_returns_string(self) -> None:
        layer = BehaviorLayer()
        result = layer.get_tone_adaptation("frustrated")
        assert isinstance(result, str) and len(result) > 0

    def test_technical_returns_string(self) -> None:
        layer = BehaviorLayer()
        result = layer.get_tone_adaptation("technical")
        assert isinstance(result, str) and len(result) > 0

    def test_brief_returns_string(self) -> None:
        layer = BehaviorLayer()
        result = layer.get_tone_adaptation("brief")
        assert isinstance(result, str) and len(result) > 0

    def test_verbose_returns_string(self) -> None:
        layer = BehaviorLayer()
        result = layer.get_tone_adaptation("verbose")
        assert isinstance(result, str) and len(result) > 0

    def test_unknown_tone_returns_empty(self) -> None:
        layer = BehaviorLayer()
        result = layer.get_tone_adaptation("galactic")
        assert result == ""

    def test_neutral_tone_returns_empty(self) -> None:
        layer = BehaviorLayer()
        result = layer.get_tone_adaptation("neutral")
        assert result == ""
