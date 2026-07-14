"""Tests for missy.agent.response_guards — general fabrication and
promise-without-action detection for tool-free responses.
"""

from __future__ import annotations

from missy.agent.response_guards import (
    detect_fabrication,
    detect_promise_without_action,
    make_fabrication_retry_prompt,
    make_promise_retry_prompt,
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
