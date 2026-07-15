"""Tests for missy.agent.done_criteria — DONE criteria engine."""

from __future__ import annotations

from missy.agent.done_criteria import (
    DoneCriteria,
    is_compound_task,
    is_observation_task,
    make_done_prompt,
    make_no_fabricated_observation_prompt,
    make_verification_prompt,
)


class TestDoneCriteria:
    def test_empty_conditions_not_met(self):
        dc = DoneCriteria()
        assert dc.all_met is False

    def test_all_verified(self):
        dc = DoneCriteria(conditions=["a", "b"], verified=[True, True])
        assert dc.all_met is True

    def test_not_all_verified(self):
        dc = DoneCriteria(conditions=["a", "b"], verified=[True, False])
        assert dc.all_met is False

    def test_no_verified_flags(self):
        dc = DoneCriteria(conditions=["a", "b"], verified=[])
        # all([]) is True in Python, and conditions is non-empty, so all_met is True
        # This is a known edge case; in practice verified is populated
        assert dc.all_met is True

    def test_pending_returns_unverified(self):
        dc = DoneCriteria(conditions=["a", "b", "c"], verified=[True, False, True])
        assert dc.pending == ["b"]

    def test_pending_all_done(self):
        dc = DoneCriteria(conditions=["a"], verified=[True])
        assert dc.pending == []

    def test_pending_all_pending(self):
        dc = DoneCriteria(conditions=["a", "b"], verified=[False, False])
        assert dc.pending == ["a", "b"]

    def test_pending_with_short_verified_list(self):
        dc = DoneCriteria(conditions=["a", "b", "c"], verified=[True])
        # strict=False in zip means extra conditions are skipped by pending
        pending = dc.pending
        assert "a" not in pending  # first one is verified


class TestIsCompoundTask:
    def test_then_connective(self):
        assert is_compound_task("Search for the file then delete it") is True

    def test_and_then(self):
        assert is_compound_task("Read the file and then process it") is True

    def test_after_that(self):
        assert is_compound_task("Do X after that do Y") is True

    def test_followed_by(self):
        assert is_compound_task("Step A followed by step B") is True

    def test_subsequently(self):
        assert is_compound_task("Do A subsequently do B") is True

    def test_numbered_list(self):
        assert is_compound_task("1. Do A\n2. Do B") is True

    def test_numbered_parenthesis(self):
        assert is_compound_task("1) Step one\n2) Step two") is True

    def test_bullet_dash(self):
        assert is_compound_task("- First item\n- Second item") is True

    def test_bullet_asterisk(self):
        assert is_compound_task("* First item\n* Second item") is True

    def test_ordinal_first(self):
        assert is_compound_task("First do A, then B") is True

    def test_ordinal_finally(self):
        assert is_compound_task("Do A, finally clean up") is True

    def test_simple_prompt_not_compound(self):
        assert is_compound_task("What is the weather?") is False

    def test_empty_string(self):
        assert is_compound_task("") is False


class TestPromptGenerators:
    def test_done_prompt_not_empty(self):
        prompt = make_done_prompt()
        assert "DONE" in prompt
        assert len(prompt) > 20

    def test_verification_prompt_not_empty(self):
        prompt = make_verification_prompt()
        assert "complete" in prompt.lower() or "tool output" in prompt.lower()
        assert len(prompt) > 20

    def test_verification_prompt_forbids_placeholder_values(self):
        """DISC-CMD-004 harness finding (2026-07-14): genuine, audit-
        verified list_files/file_read/file_write calls were made, but the
        reported directory listing was entirely invented (fabricated
        filenames that don't exist). Unlike the zero-tool-call guards in
        response_guards.py, this can't be caught deterministically since
        real tools WERE used -- the verification prompt is strengthened
        to explicitly forbid substituting example-looking values for the
        actual tool output."""
        prompt = make_verification_prompt()
        assert "copy" in prompt.lower() or "exactly" in prompt.lower()
        assert "placeholder" in prompt.lower() or "invent" in prompt.lower()


class TestIsObservationTask:
    """FX-round2-F4: detects requests implying a real vision/memory
    observation, so a zero-tool-call response to one can be caught as a
    fabrication rather than accepted at face value."""

    def test_vision_capture_request(self):
        assert is_observation_task("Capture a burst of frames and compare them") is True

    def test_vision_photo_request(self):
        assert is_observation_task("Take a picture with the webcam") is True

    def test_vision_look_request(self):
        assert is_observation_task("Look at what's on the desk right now") is True

    def test_memory_search_request(self):
        assert is_observation_task("Is there a deployment checklist in memory?") is True

    def test_memory_recall_request(self):
        assert is_observation_task("Do you remember what I told you yesterday?") is True

    def test_unrelated_request_not_observation(self):
        assert is_observation_task("What is 2 plus 2?") is False

    def test_empty_string(self):
        assert is_observation_task("") is False

    def test_case_insensitive(self):
        assert is_observation_task("CAPTURE a PHOTO of the room") is True


class TestNoFabricatedObservationPrompt:
    def test_prompt_not_empty(self):
        prompt = make_no_fabricated_observation_prompt()
        assert len(prompt) > 20

    def test_prompt_mentions_tool_call(self):
        prompt = make_no_fabricated_observation_prompt()
        assert "tool call" in prompt.lower()

    def test_prompt_instructs_honesty_on_failure(self):
        prompt = make_no_fabricated_observation_prompt()
        assert "say so" in prompt.lower()
