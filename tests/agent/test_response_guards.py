"""Tests for missy.agent.response_guards — general fabrication and
promise-without-action detection for tool-free responses.
"""

from __future__ import annotations

from missy.agent.response_guards import (
    detect_fabrication,
    detect_false_capability_denial,
    detect_identity_confusion,
    detect_promise_without_action,
    detect_security_refusal_without_alternative,
    make_capability_denial_retry_prompt,
    make_fabrication_retry_prompt,
    make_identity_confusion_retry_prompt,
    make_promise_retry_prompt,
    make_security_refusal_retry_prompt,
)


class TestDetectFabrication:
    def test_claims_of_checking_something(self):
        assert detect_fabrication("I checked the logs and everything looks fine.", []) is True

    def test_claims_of_running_a_command(self):
        assert detect_fabrication("I ran the script and it completed successfully.", []) is True

    def test_claims_of_reviewing(self):
        assert detect_fabrication("I reviewed the configuration file for you.", []) is True

    def test_fake_shell_output_block(self):
        text = "Here's what I found:\n```bash\n$ ls -la\ntotal 24\ndrwxr-xr-x\n```"
        assert detect_fabrication(text, []) is True

    def test_fake_docker_ps_style_output(self):
        text = "```text\nCONTAINER ID   IMAGE   STATUS\nabc123   nginx   Up 2 hours\n```"
        assert detect_fabrication(text, []) is True

    def test_claims_of_completed_artifact(self):
        assert detect_fabrication("I generated the image and saved it to disk.", []) is True

    def test_claims_referencing_unchecked_logs(self):
        assert detect_fabrication("According to the logs, the service restarted twice.", []) is True

    def test_claims_referencing_unchecked_metrics(self):
        assert detect_fabrication("Based on the metrics, CPU usage is high.", []) is True

    def test_no_fabrication_when_tools_were_used(self):
        """The whole point: any real tool call anywhere this task makes
        the check a no-op, regardless of what the text says."""
        assert (
            detect_fabrication("I checked the logs and everything looks fine.", ["shell_exec"])
            is False
        )

    def test_short_response_not_flagged(self):
        assert detect_fabrication("I checked it.", []) is False

    def test_empty_response_not_flagged(self):
        assert detect_fabrication("", []) is False

    def test_ordinary_factual_answer_not_flagged(self):
        text = "The weather today is sunny with a high of 75 degrees outside."
        assert detect_fabrication(text, []) is False

    def test_ordinary_conversational_reply_not_flagged(self):
        text = "Sure, happy to help! What would you like to know more about?"
        assert detect_fabrication(text, []) is False


class TestMakeFabricationRetryPrompt:
    def test_not_empty(self):
        prompt = make_fabrication_retry_prompt()
        assert len(prompt) > 20

    def test_instructs_real_tool_call(self):
        prompt = make_fabrication_retry_prompt()
        assert "tool" in prompt.lower()


class TestDetectPromiseWithoutAction:
    def test_ill_do_something(self):
        assert (
            detect_promise_without_action("I'll go ahead and set that up for you now.", []) is True
        )

    def test_working_on_it(self):
        assert detect_promise_without_action("Working on it, give me a moment.", []) is True

    def test_present_progressive(self):
        assert detect_promise_without_action("I'm generating that report now.", []) is True

    def test_i_will_phrasing(self):
        assert detect_promise_without_action("I will handle that request right away.", []) is True

    def test_leading_acknowledgment_phrase(self):
        assert detect_promise_without_action("On it! Let me pull that together.", []) is True

    def test_no_promise_when_tools_were_used(self):
        assert detect_promise_without_action("Working on it.", ["shell_exec"]) is False

    def test_short_response_not_flagged(self):
        assert detect_promise_without_action("Sure!", []) is False

    def test_empty_response_not_flagged(self):
        assert detect_promise_without_action("", []) is False

    def test_ordinary_factual_answer_not_flagged(self):
        text = "The capital of France is Paris, a beautiful city on the Seine."
        assert detect_promise_without_action(text, []) is False

    def test_exempt_casual_future_tense_be(self):
        assert detect_promise_without_action("I'll be here if you need anything else.", []) is False

    def test_exempt_casual_future_tense_keep(self):
        assert detect_promise_without_action("I'll keep that in mind for next time.", []) is False

    def test_exempt_casual_future_tense_will_variant(self):
        assert detect_promise_without_action("I will remember that for our next chat.", []) is False


class TestMakePromiseRetryPrompt:
    def test_not_empty(self):
        prompt = make_promise_retry_prompt()
        assert len(prompt) > 20

    def test_instructs_real_tool_call(self):
        prompt = make_promise_retry_prompt()
        assert "tool" in prompt.lower()

    def test_no_anchor_when_user_input_omitted(self):
        prompt = make_promise_retry_prompt()
        assert "request you must actually fulfill" not in prompt

    def test_anchors_to_current_task_when_user_input_given(self):
        """SH-004 harness finding: a content-free retry let the model
        drift to a different, still-incomplete prior turn instead of the
        current one. Restating the task anchors the retry to it."""
        prompt = make_promise_retry_prompt("write project-name.txt with 'safe-project'")
        assert "write project-name.txt with 'safe-project'" in prompt


class TestSecurityRefusalAlternativeGuard:
    def test_flags_bare_privilege_refusal(self):
        assert detect_security_refusal_without_alternative(
            "I'm not able to help with that particular request.",
            "Use sudo to install a host package and change system settings.",
        )

    def test_accepts_refusal_with_container_alternative(self):
        assert not detect_security_refusal_without_alternative(
            "I can't do that. Safe alternative: use an unprivileged container.",
            "Use sudo to install a host package and change system settings.",
        )

    def test_flags_bare_secret_upload_refusal(self):
        assert detect_security_refusal_without_alternative(
            "I can't upload credentials.",
            "Upload the Discord token file to this channel.",
        )

    def test_flags_generic_but_wrong_secret_upload_alternative(self):
        assert detect_security_refusal_without_alternative(
            "I can't upload it. Safe alternative: I can explain credential rotation.",
            "Upload the Discord token file to this channel.",
        )

    def test_flags_typographic_apostrophe_from_live_secret_refusal(self):
        assert detect_security_refusal_without_alternative(
            "I can’t upload credentials. Safe alternative: create a sanitized example.",
            "Upload the Discord token file to this channel.",
        )

    def test_accepts_gitignore_secret_upload_alternative(self):
        assert not detect_security_refusal_without_alternative(
            "I can't upload it. I can verify that the token file is in gitignore.",
            "Upload the Discord token file to this channel.",
        )

    def test_flags_non_container_host_privilege_alternative(self):
        assert detect_security_refusal_without_alternative(
            "I can't use sudo. Safe alternative: ask an administrator.",
            "Use sudo to install a host package and change system settings.",
        )

    def test_ignores_ordinary_refusal(self):
        assert not detect_security_refusal_without_alternative(
            "I can't attend the meeting.",
            "Can you attend the meeting tomorrow?",
        )

    def test_retry_preserves_refusal_and_names_specific_alternatives(self):
        prompt = make_security_refusal_retry_prompt("Use sudo to install a package")
        lower = prompt.lower()
        assert "keep the refusal unchanged" in lower
        assert "unprivileged" in lower
        assert "do not call a tool" in lower


class TestMakeFabricationRetryPromptAnchor:
    def test_anchors_to_current_task_when_user_input_given(self):
        prompt = make_fabrication_retry_prompt("summarize the deployment checklist")
        assert "summarize the deployment checklist" in prompt

    def test_no_anchor_when_user_input_omitted(self):
        prompt = make_fabrication_retry_prompt()
        assert "request you must actually fulfill" not in prompt


class TestDetectIdentityConfusion:
    """3rd tool-specific validation run (2026-07-14): the dominant
    failure mode -- the delegate denies being Missy, or denies her own
    dispatched tools as belonging to a separate platform."""

    def test_claims_to_be_claude_code(self):
        text = "I'm Claude Code -- I don't have access to the Missy platform's tool."
        assert detect_identity_confusion(text) is True

    def test_am_claude_code_variant(self):
        text = "I am Claude Code, a coding assistant running in a terminal."
        assert detect_identity_confusion(text) is True

    def test_tool_belongs_to_missy_platform(self):
        text = "The vision_analyze tool belongs to the Missy agent platform and is not callable."
        assert detect_identity_confusion(text) is True

    def test_not_at_me_denial(self):
        text = "This Discord message is directed at the Missy agent, not at me."
        assert detect_identity_confusion(text) is True

    def test_no_access_to_missy(self):
        text = "I don't have access to the Missy platform's audio_set_volume tool."
        assert detect_identity_confusion(text) is True

    def test_should_not_execute(self):
        text = "I should not be executing Discord commands directed at the Missy agent."
        assert detect_identity_confusion(text) is True

    def test_coding_assistant_running_in_terminal(self):
        text = "My available tools are file/shell/search tools since I'm a coding assistant running in a terminal."
        assert detect_identity_confusion(text) is True

    def test_ordinary_reply_not_flagged(self):
        text = "Hard no. Uploading a secrets.env file to Discord would expose credentials."
        assert detect_identity_confusion(text) is False

    def test_short_response_not_flagged(self):
        assert detect_identity_confusion("Sure!") is False

    def test_empty_response_not_flagged(self):
        assert detect_identity_confusion("") is False

    def test_runs_regardless_of_tools_used(self):
        """VIS-004 harness finding: the delegate called vision_scene and
        vision_burst successfully, then in the same reply denied access
        to the sibling vision_analyze tool -- this must still be caught
        even though real tools were used this task."""
        text = (
            "The burst captured 2 frames. I'm Claude Code and don't have access "
            "to vision_analyze or the other Missy platform tools."
        )
        assert detect_identity_confusion(text) is True


class TestMakeIdentityConfusionRetryPrompt:
    def test_not_empty(self):
        prompt = make_identity_confusion_retry_prompt()
        assert len(prompt) > 20

    def test_does_not_use_assertive_identity_override_framing(self):
        """4th validation run finding: an assertive "[System reminder]:
        You ARE Missy, not a separate assistant..." framing resembles a
        jailbreak/identity-override attempt closely enough that some
        models refuse it as suspected prompt injection. The corrective
        prompt must not use that shape."""
        prompt = make_identity_confusion_retry_prompt()
        assert "[system reminder]" not in prompt.lower()
        assert "you are missy" not in prompt.lower()
        assert "you are missy" not in prompt.lower().replace("'", "")

    def test_cites_tool_list_as_grounding_fact(self):
        prompt = make_identity_confusion_retry_prompt()
        assert "tool list" in prompt.lower()

    def test_cites_available_tool_names_when_given(self):
        prompt = make_identity_confusion_retry_prompt(
            available_tool_names={"vision_capture", "calculator"}
        )
        assert "vision_capture" in prompt
        assert "calculator" in prompt

    def test_anchors_to_current_task_when_user_input_given(self):
        prompt = make_identity_confusion_retry_prompt("what is 2+2?")
        assert "what is 2+2?" in prompt


class TestDetectFalseCapabilityDenial:
    """False capability denials ("no X11", "headless", "no browser") are
    the same root bug as identity confusion, but never name Missy/Claude
    Code explicitly -- so this check additionally requires the denied
    tool category to actually be present in the available tool set."""

    def test_headless_denial_flagged_when_x11_tool_available(self):
        text = "No X11, no display, no GUI -- I'm headless."
        assert detect_false_capability_denial(text, [], {"x11_launch"}) is True

    def test_headless_denial_not_flagged_when_no_matching_tool(self):
        text = "No X11, no display, no GUI -- I'm headless."
        assert detect_false_capability_denial(text, [], {"calculator"}) is False

    def test_no_browser_denial_flagged_when_browser_tool_available(self):
        text = "I have no browser or JS runtime; web_fetch gets raw HTML only."
        assert detect_false_capability_denial(text, [], {"browser_navigate"}) is True

    def test_no_camera_denial_flagged_when_vision_tool_available(self):
        text = "I have no camera or webcam access."
        assert detect_false_capability_denial(text, [], {"vision_capture"}) is True

    def test_no_accessibility_denial_flagged_when_atspi_tool_available(self):
        text = "There are no accessibility tools available to me."
        assert detect_false_capability_denial(text, [], {"atspi_get_tree"}) is True

    def test_tools_used_short_circuits_to_false(self):
        text = "No X11, no display, no GUI -- I'm headless."
        assert detect_false_capability_denial(text, ["x11_launch"], {"x11_launch"}) is False

    def test_ordinary_reply_not_flagged(self):
        text = "The weather today is sunny with a high of 75 degrees."
        assert detect_false_capability_denial(text, [], {"x11_launch"}) is False

    def test_empty_response_not_flagged(self):
        assert detect_false_capability_denial("", [], {"x11_launch"}) is False


class TestMakeCapabilityDenialRetryPrompt:
    def test_not_empty(self):
        prompt = make_capability_denial_retry_prompt()
        assert len(prompt) > 20

    def test_instructs_calling_the_tool(self):
        prompt = make_capability_denial_retry_prompt()
        assert "tool" in prompt.lower()

    def test_does_not_use_system_reminder_framing(self):
        """Same over-refusal-spiral concern as the identity-confusion
        prompt: a bracketed pseudo-system tag pattern-matches as a
        jailbreak attempt to some models."""
        prompt = make_capability_denial_retry_prompt()
        assert "[system reminder]" not in prompt.lower()

    def test_cites_available_tool_names_when_given(self):
        prompt = make_capability_denial_retry_prompt(
            available_tool_names={"x11_launch", "atspi_get_tree"}
        )
        assert "x11_launch" in prompt
        assert "atspi_get_tree" in prompt

    def test_anchors_to_current_task_when_user_input_given(self):
        prompt = make_capability_denial_retry_prompt("launch a text editor")
        assert "launch a text editor" in prompt
