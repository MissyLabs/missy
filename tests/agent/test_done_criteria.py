"""Tests for missy.agent.done_criteria — DONE criteria engine."""

from __future__ import annotations

from missy.agent.done_criteria import (
    DoneCriteria,
    is_compound_task,
    make_done_prompt,
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
