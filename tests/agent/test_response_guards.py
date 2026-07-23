"""Tests for missy.agent.response_guards — general fabrication and
promise-without-action detection for tool-free responses.
"""

from __future__ import annotations

import pytest

from missy.agent.response_guards import (
    detect_explicit_tool_requests,
    detect_fabrication,
    detect_false_capability_denial,
    detect_identity_confusion,
    detect_promise_without_action,
    detect_security_refusal_without_alternative,
    find_unmet_desktop_requests,
    find_unmet_video_generation_request,
    find_unmet_web_requests,
    find_unreported_calculator_expressions,
    find_unverified_desktop_action,
    find_unverified_filesystem_action,
    is_security_refusal,
    make_calculator_completeness_retry_prompt,
    make_capability_denial_retry_prompt,
    make_desktop_request_retry_prompt,
    make_desktop_verification_retry_prompt,
    make_explicit_tool_request_retry_prompt,
    make_fabrication_retry_prompt,
    make_filesystem_verification_retry_prompt,
    make_identity_confusion_retry_prompt,
    make_promise_retry_prompt,
    make_security_refusal_retry_prompt,
    make_video_generation_error_report_prompt,
    make_video_generation_retry_prompt,
    make_web_request_retry_prompt,
    terminal_parameter_errors_are_reported,
)


class TestVideoGenerationGuards:
    def test_render_request_requires_video_generate(self):
        prompt = "Generate a video of a sunset using the SVD backend."
        assert find_unmet_video_generation_request(prompt, [], {"video_generate"}) == [
            "video_generate"
        ]
        assert find_unmet_video_generation_request(
            prompt, ["video_generate"], {"video_generate"}
        ) == []

    def test_animation_request_requires_video_generate(self):
        assert find_unmet_video_generation_request(
            "Animate this image into a short video.", [], {"video_generate"}
        ) == ["video_generate"]

    def test_retry_prompt_requires_exact_tool(self):
        retry = make_video_generation_retry_prompt("Generate an SVD video")
        assert "video_generate" in retry
        assert "video_storyboard" in retry
        assert "Generate an SVD video" in retry

    def test_reported_parameter_constraint_is_terminal(self):
        errors = [
            "video_generate: audio_prompt and audio_path are mutually exclusive; pass only one."
        ]
        reply = "I can't apply both because those parameters are mutually exclusive."
        assert terminal_parameter_errors_are_reported(errors, reply)

    def test_video_timeout_is_terminal_once_reported(self):
        assert terminal_parameter_errors_are_reported(
            ["video_generate: generation timed out after 15 seconds"],
            "The generation timed out.",
        )

    def test_other_operational_failure_is_not_terminal(self):
        assert not terminal_parameter_errors_are_reported(
            ["shell_exec: command timed out after 15 seconds"],
            "The command timed out.",
        )

    def test_terminal_error_prompt_forbids_retry_and_preserves_evidence(self):
        retry = make_video_generation_error_report_prompt(
            ["video_generate: missing models/checkpoints/svd_xt.safetensors"],
            "Animate the test image with SVD.",
        )
        assert "Do not retry" in retry
        assert "svd_xt.safetensors" in retry
        assert "Animate the test image with SVD." in retry


class TestFilesystemVerificationGuard:
    def test_latest_write_requires_later_observation(self):
        assert find_unverified_filesystem_action(["file_write", "file_write"]) == "file_write"

    def test_listing_after_mutation_satisfies_guard(self):
        assert find_unverified_filesystem_action(["file_write", "list_files"]) is None

    def test_read_after_mutation_satisfies_guard(self):
        assert find_unverified_filesystem_action(["file_delete", "file_read"]) is None

    def test_observation_before_latest_mutation_does_not_satisfy_guard(self):
        assert find_unverified_filesystem_action(["list_files", "file_write"]) == "file_write"

    def test_retry_prompt_anchors_original_request(self):
        prompt = make_filesystem_verification_retry_prompt(
            "file_write", "Create README.md and src/main.txt"
        )
        assert "list_files" in prompt
        assert "Create README.md and src/main.txt" in prompt


class TestWebRequestGuard:
    def test_metadata_request_requires_navigation_url_read_and_close(self):
        prompt = "Open this local webpage in the browser and report the current URL and page title."
        assert find_unmet_web_requests(prompt, [], None) == [
            "browser_close",
            "browser_get_url",
            "browser_navigate",
        ]

    def test_complete_form_workflow_is_satisfied(self):
        prompt = "Open the local form page, fill fake name/email data, submit, and report text."
        used = [
            "browser_navigate",
            "browser_fill",
            "browser_fill",
            "browser_click",
            "browser_wait",
            "browser_get_content",
            "browser_close",
        ]
        assert find_unmet_web_requests(prompt, used, set(used)) == []

    def test_name_and_email_require_two_successful_fill_calls(self):
        prompt = "Open the form page, fill fake name/email data, and submit."
        used = ["browser_navigate", "browser_fill", "browser_click", "browser_close"]
        assert "browser_fill" in find_unmet_web_requests(prompt, used, set(used))

    def test_form_confirmation_steps_must_follow_submit_in_one_session(self):
        prompt = "Open the form page, fill fake name/email data, submit, and report text."
        used = [
            "browser_navigate",
            "browser_fill",
            "browser_fill",
            "browser_click",
            "browser_get_content",
            "browser_close",
            "browser_navigate",
            "browser_wait",
            "browser_close",
        ]
        missing = find_unmet_web_requests(prompt, used, set(used))
        assert missing == [
            "browser_click",
            "browser_close",
            "browser_fill",
            "browser_get_content",
            "browser_wait",
        ]

    def test_raw_fetch_request_requires_web_fetch(self):
        prompt = "Fetch this local test URL: http://127.0.0.1/page"
        assert find_unmet_web_requests(prompt, [], {"web_fetch"}) == ["web_fetch"]

    def test_accessible_submit_button_is_not_a_browser_form_request(self):
        prompt = "Click the accessible button named Submit."
        assert find_unmet_web_requests(prompt, [], None) == []

    def test_desktop_screenshot_is_not_a_browser_request(self):
        prompt = "Open a text editor, type hello, and take a screenshot."
        assert find_unmet_web_requests(prompt, [], None) == []

    def test_close_before_later_browser_action_is_not_terminal(self):
        prompt = "Open this page in the browser and report the page title."
        used = ["browser_navigate", "browser_close", "browser_get_url"]
        assert find_unmet_web_requests(prompt, used, set(used)) == ["browser_close"]

    def test_retry_prompt_names_tools_and_original_request(self):
        prompt = make_web_request_retry_prompt(
            ["browser_get_url", "browser_close"], "Open the page"
        )
        assert "browser_get_url" in prompt
        assert "browser_close" in prompt
        assert "Open the page" in prompt


class TestCalculatorCompletenessGuard:
    def test_finds_an_earlier_omitted_error(self):
        observations = [
            ("abs(-5)", "Unsupported expression construct: Call", True),
            ("x + 1", "Unsupported expression construct: Name", True),
        ]
        reply = "`x + 1` failed: Unsupported expression construct: Name"
        assert find_unreported_calculator_expressions(observations, reply) == ["abs(-5)"]

    def test_accepts_every_expression_and_observed_outcome(self):
        observations = [
            ("1 << 8", "256", False),
            ("1 / 0", "division by zero", True),
        ]
        reply = "`1 << 8` = 256; `1 / 0` failed with division by zero."
        assert find_unreported_calculator_expressions(observations, reply) == []

    def test_empty_input_is_bound_to_its_error_marker(self):
        observations = [("", "expression must not be empty", True)]
        assert find_unreported_calculator_expressions(observations, "No value was returned") == [
            "<empty or whitespace-only expression>"
        ]
        assert (
            find_unreported_calculator_expressions(
                observations, "The empty input failed: expression must not be empty."
            )
            == []
        )

    def test_retry_prompt_includes_all_grounded_outcomes_and_request(self):
        observations = [
            ("2 ** 100000", "Exponent 100000 exceeds the maximum allowed value of 1000.", True),
            ("2 ** 10", "1024", False),
        ]
        prompt = make_calculator_completeness_retry_prompt(
            observations, ["2 ** 100000"], "Calculate both with your calculator."
        )
        assert "2 ** 100000" in prompt
        assert "2 ** 10" in prompt
        assert "Exponent 100000" in prompt
        assert "1024" in prompt
        assert "Calculate both with your calculator." in prompt


class TestExplicitToolRequestGuard:
    @pytest.mark.parametrize(
        "prompt",
        [
            "Use your calculator to evaluate 2 + 2.",
            "Call the `calculator` tool for 2 + 2.",
            "Invoke calculator now.",
            "Compute this using the calculator tool.",
            "Calculate 2 + 2 with your calculator.",
        ],
    )
    def test_detects_explicit_available_tool(self, prompt):
        assert detect_explicit_tool_requests(prompt, {"calculator", "file_read"}) == {"calculator"}

    @pytest.mark.parametrize(
        "prompt",
        [
            "Do not use calculator; answer conceptually.",
            "Never call the calculator tool.",
            "Explain what calculator does.",
            "Is calculator available?",
            "Answer without using calculator.",
        ],
    )
    def test_does_not_turn_negation_or_mentions_into_requirement(self, prompt):
        assert detect_explicit_tool_requests(prompt, {"calculator"}) == frozenset()

    def test_only_returns_tools_available_this_turn(self):
        assert detect_explicit_tool_requests("Use shell_exec now", {"calculator"}) == frozenset()

    def test_retry_prompt_names_tool_and_anchors_original_request(self):
        prompt = make_explicit_tool_request_retry_prompt(
            {"calculator"}, "Use your calculator for 2 + 2"
        )
        assert "calculator" in prompt
        assert "Call the named tool now" in prompt
        assert "Use your calculator for 2 + 2" in prompt


class TestDesktopActionVerificationGuard:
    def test_returns_latest_action_when_no_later_observation_exists(self):
        assert (
            find_unverified_desktop_action(
                ["x11_read_screen", "atspi_get_tree", "atspi_click", "x11_click"]
            )
            == "x11_click"
        )

    def test_later_screen_or_tree_observation_satisfies_gate(self):
        assert find_unverified_desktop_action(["x11_click", "x11_read_screen"]) is None
        assert find_unverified_desktop_action(["atspi_set_value", "atspi_get_text"]) is None

    def test_observation_before_action_does_not_verify_action(self):
        assert find_unverified_desktop_action(["x11_read_screen", "x11_key"]) == "x11_key"

    def test_non_desktop_tools_do_not_trigger(self):
        assert find_unverified_desktop_action(["calculator", "file_read"]) is None

    def test_failed_desktop_action_is_excluded_by_successful_call_filter(self):
        # Runtime supplies only successful tool names; a failed action must be
        # handled by DONE criteria rather than demanding verification of an
        # input event that never occurred.
        assert find_unverified_desktop_action(["atspi_get_tree"]) is None

    def test_retry_prompt_names_action_and_requires_later_observation(self):
        prompt = make_desktop_verification_retry_prompt("x11_click", "Click the Run Test button")
        assert "x11_click" in prompt
        assert "successful input event is not proof" in prompt.lower()
        assert "Click the Run Test button" in prompt


class TestDesktopRequestExecutionGuard:
    def test_keyboard_shortcut_requires_current_x11_key(self):
        prompt = "Use keyboard shortcuts to select all text in the focused editor and copy it."
        assert find_unmet_desktop_requests(prompt, []) == ["keyboard shortcut"]
        assert find_unmet_desktop_requests(prompt, ["x11_key"]) == []

    def test_screen_read_requires_current_x11_read_screen(self):
        prompt = "Read the current desktop screen and describe what application is visible."
        assert find_unmet_desktop_requests(prompt, []) == ["screen read"]
        assert find_unmet_desktop_requests(prompt, ["x11_read_screen"]) == []

    def test_compound_desktop_prompt_tracks_each_operation(self):
        prompt = "Open a text editor, type hello, and take a screenshot."
        assert find_unmet_desktop_requests(prompt, ["desktop_launch_app"]) == [
            "desktop typing",
            "desktop screenshot",
        ]

    def test_browser_screenshot_does_not_require_x11_screenshot(self):
        prompt = (
            "Open the local dashboard page, take a screenshot, and upload it to Discord. "
            "URL: http://127.0.0.1/dashboard.html"
        )
        assert find_unmet_desktop_requests(prompt, ["browser_screenshot"]) == []

    def test_unavailable_tool_family_is_not_required(self):
        assert (
            find_unmet_desktop_requests("Read the current desktop screen.", [], {"calculator"})
            == []
        )

    def test_retry_prompt_reanchors_original_request(self):
        prompt = make_desktop_request_retry_prompt(["screen read"], "Read the screen")
        assert "screen read" in prompt
        assert "earlier channel turn" in prompt
        assert "Read the screen" in prompt


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

    def test_non_string_test_doubles_fail_open_without_crashing(self):
        assert not detect_security_refusal_without_alternative(
            object(),  # type: ignore[arg-type]
            object(),  # type: ignore[arg-type]
        )

    def test_recognizes_high_risk_refusal_even_with_valid_alternative(self):
        assert is_security_refusal(
            "I can't disable host security. I can use an unprivileged disposable container.",
            "Use shell_exec to disable host security.",
        )

    def test_ordinary_limitation_is_not_a_security_refusal(self):
        assert not is_security_refusal(
            "I can't calculate that right now.",
            "Use calculator for 2 + 2.",
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
