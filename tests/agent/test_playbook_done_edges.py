"""Edge-case tests for missy.agent.playbook and missy.agent.done_criteria.

These tests focus on boundary conditions and behaviours not covered by the
existing test_playbook.py and test_done_criteria.py suites.
"""

from __future__ import annotations

import json
import os
import threading

import pytest

from missy.agent.done_criteria import (
    DoneCriteria,
    _COMPOUND_PATTERNS,
    is_compound_task,
    make_done_prompt,
    make_verification_prompt,
)
from missy.agent.playbook import Playbook, PlaybookEntry, _compute_pattern_id


# ---------------------------------------------------------------------------
# Playbook edge cases
# ---------------------------------------------------------------------------


class TestPlaybookStoreAndRetrieve:
    """Basic store/retrieve path, making sure the fundamentals work."""

    def test_record_stores_entry(self, tmp_path):
        pb = Playbook(store_path=str(tmp_path / "pb.json"))
        entry = pb.record("net", "fetch page", ["http_get"], "use GET")
        relevant = pb.get_relevant("net")
        assert len(relevant) == 1
        assert relevant[0].pattern_id == entry.pattern_id

    def test_record_returns_same_object_on_second_call(self, tmp_path):
        """record() updates in-place; both returned entries share a pattern_id."""
        pb = Playbook(store_path=str(tmp_path / "pb.json"))
        e1 = pb.record("file", "read cfg", ["file_read"], "h1")
        e2 = pb.record("file", "read cfg v2", ["file_read"], "h2")
        # Must be the same logical record
        assert e1.pattern_id == e2.pattern_id
        assert e2.success_count == 2

    def test_get_relevant_no_match_returns_empty_list(self, tmp_path):
        pb = Playbook(store_path=str(tmp_path / "pb.json"))
        pb.record("shell", "run tests", ["shell_exec"], "pytest")
        result = pb.get_relevant("nonexistent_type")
        assert result == []

    def test_get_relevant_top_k_zero_returns_empty(self, tmp_path):
        pb = Playbook(store_path=str(tmp_path / "pb.json"))
        pb.record("shell", "deploy", ["shell_exec"], "hint")
        assert pb.get_relevant("shell", top_k=0) == []

    def test_get_relevant_top_k_larger_than_entries_returns_all(self, tmp_path):
        pb = Playbook(store_path=str(tmp_path / "pb.json"))
        pb.record("shell", "task a", ["shell_exec"], "h1")
        pb.record("shell", "task b", ["shell_exec", "file_write"], "h2")
        result = pb.get_relevant("shell", top_k=100)
        assert len(result) == 2

    def test_get_relevant_sorted_descending_by_success_count(self, tmp_path):
        pb = Playbook(store_path=str(tmp_path / "pb.json"))
        # Pattern A: 1 success
        pb.record("shell", "a", ["shell_exec"], "h")
        # Pattern B: 3 successes
        for _ in range(3):
            pb.record("shell", "b", ["shell_exec", "file_write"], "h")
        # Pattern C: 2 successes
        pb.record("shell", "c", ["shell_exec", "file_read"], "h")
        pb.record("shell", "c", ["shell_exec", "file_read"], "h")

        results = pb.get_relevant("shell")
        counts = [e.success_count for e in results]
        assert counts == sorted(counts, reverse=True)


class TestPlaybookAutoPromotion:
    """Patterns crossing the success threshold become promotable."""

    def test_pattern_promotable_at_exactly_threshold(self, tmp_path):
        pb = Playbook(store_path=str(tmp_path / "pb.json"))
        for _ in range(3):
            pb.record("shell", "deploy", ["shell_exec"], "hint")
        promotable = pb.get_promotable(threshold=3)
        assert len(promotable) == 1
        assert promotable[0].success_count == 3

    def test_pattern_not_promotable_one_below_threshold(self, tmp_path):
        pb = Playbook(store_path=str(tmp_path / "pb.json"))
        for _ in range(2):
            pb.record("shell", "deploy", ["shell_exec"], "hint")
        assert pb.get_promotable(threshold=3) == []

    def test_promotion_at_threshold_one(self, tmp_path):
        """threshold=1 means any recorded pattern is immediately promotable."""
        pb = Playbook(store_path=str(tmp_path / "pb.json"))
        pb.record("file", "read cfg", ["file_read"], "hint")
        promotable = pb.get_promotable(threshold=1)
        assert len(promotable) == 1

    def test_already_promoted_excluded_from_promotable(self, tmp_path):
        pb = Playbook(store_path=str(tmp_path / "pb.json"))
        for _ in range(3):
            pb.record("shell", "deploy", ["shell_exec"], "hint")
        pid = pb.get_promotable()[0].pattern_id
        pb.mark_promoted(pid)
        # Must not appear again even with additional successes
        pb.record("shell", "deploy", ["shell_exec"], "hint")
        assert pb.get_promotable(threshold=3) == []

    def test_multiple_patterns_only_eligible_promoted(self, tmp_path):
        pb = Playbook(store_path=str(tmp_path / "pb.json"))
        for _ in range(5):
            pb.record("shell", "high", ["shell_exec"], "h")
        pb.record("file", "low", ["file_read"], "h")  # count=1
        promotable = pb.get_promotable(threshold=3)
        assert len(promotable) == 1
        assert promotable[0].task_type == "shell"


class TestPlaybookDeduplication:
    """Same logical pattern always maps to the same entry regardless of arg order."""

    def test_tool_sequence_order_irrelevant_for_deduplication(self, tmp_path):
        pb = Playbook(store_path=str(tmp_path / "pb.json"))
        pb.record("shell", "d1", ["z_tool", "a_tool", "m_tool"], "h1")
        pb.record("shell", "d2", ["a_tool", "m_tool", "z_tool"], "h2")
        result = pb.get_relevant("shell")
        assert len(result) == 1
        assert result[0].success_count == 2

    def test_different_task_type_same_tools_different_pattern(self, tmp_path):
        pb = Playbook(store_path=str(tmp_path / "pb.json"))
        pb.record("typeA", "desc", ["tool_x"], "h")
        pb.record("typeB", "desc", ["tool_x"], "h")
        assert len(pb.get_relevant("typeA")) == 1
        assert len(pb.get_relevant("typeB")) == 1

    def test_empty_tool_sequence_is_valid_and_deduped(self, tmp_path):
        """An empty tool list is an edge case but must not raise."""
        pb = Playbook(store_path=str(tmp_path / "pb.json"))
        e1 = pb.record("simple", "no tools needed", [], "hint")
        e2 = pb.record("simple", "still no tools", [], "hint2")
        assert e1.pattern_id == e2.pattern_id
        assert e2.success_count == 2

    def test_single_tool_sequence_deduplicated(self, tmp_path):
        pb = Playbook(store_path=str(tmp_path / "pb.json"))
        for _ in range(4):
            pb.record("shell", "run linter", ["ruff"], "ruff check .")
        result = pb.get_relevant("shell")
        assert len(result) == 1
        assert result[0].success_count == 4


class TestPlaybookPersistence:
    """JSON round-trip fidelity."""

    def test_persistence_preserves_success_count(self, tmp_path):
        path = str(tmp_path / "pb.json")
        pb1 = Playbook(store_path=path)
        for _ in range(5):
            pb1.record("shell", "deploy", ["shell_exec"], "hint")

        pb2 = Playbook(store_path=path)
        result = pb2.get_relevant("shell")
        assert result[0].success_count == 5

    def test_persistence_preserves_promoted_flag(self, tmp_path):
        path = str(tmp_path / "pb.json")
        pb1 = Playbook(store_path=path)
        for _ in range(3):
            pb1.record("shell", "deploy", ["shell_exec"], "hint")
        pid = pb1.get_promotable()[0].pattern_id
        pb1.mark_promoted(pid)

        pb2 = Playbook(store_path=path)
        result = pb2.get_relevant("shell")
        assert result[0].promoted is True

    def test_persistence_preserves_multiple_task_types(self, tmp_path):
        path = str(tmp_path / "pb.json")
        pb1 = Playbook(store_path=path)
        pb1.record("shell", "s-task", ["shell_exec"], "hs")
        pb1.record("file", "f-task", ["file_read"], "hf")
        pb1.record("net", "n-task", ["http_get"], "hn")

        pb2 = Playbook(store_path=path)
        assert len(pb2.get_relevant("shell")) == 1
        assert len(pb2.get_relevant("file")) == 1
        assert len(pb2.get_relevant("net")) == 1

    def test_save_creates_parent_directory(self, tmp_path):
        nested = tmp_path / "a" / "b" / "c" / "pb.json"
        pb = Playbook(store_path=str(nested))
        pb.record("shell", "task", ["shell_exec"], "hint")
        assert nested.exists()

    def test_json_file_is_valid_list(self, tmp_path):
        path = tmp_path / "pb.json"
        pb = Playbook(store_path=str(path))
        pb.record("shell", "task", ["shell_exec"], "hint")
        with open(path) as fh:
            data = json.load(fh)
        assert isinstance(data, list)
        assert len(data) == 1
        assert data[0]["task_type"] == "shell"


class TestPlaybookCorruptedFile:
    """Load must survive corrupted or malformed JSON without raising."""

    def test_corrupted_json_results_in_empty_playbook(self, tmp_path):
        path = tmp_path / "pb.json"
        path.write_text("{ this is not valid json !!!")
        # Must not raise; entries should be empty
        pb = Playbook(store_path=str(path))
        assert pb.get_relevant("anything") == []

    def test_json_array_with_missing_fields_does_not_crash(self, tmp_path):
        path = tmp_path / "pb.json"
        # Write a list entry that lacks required fields
        path.write_text('[{"task_type": "shell"}]')
        pb = Playbook(store_path=str(path))
        # Either loads nothing or raises internally — must not propagate
        # The result could be empty (if load silently swallows) or partial
        # The important invariant: no exception escapes
        _ = pb.get_relevant("shell")

    def test_empty_json_file_treated_as_empty_playbook(self, tmp_path):
        path = tmp_path / "pb.json"
        path.write_text("")
        pb = Playbook(store_path=str(path))
        assert pb.get_relevant("anything") == []

    def test_json_object_instead_of_array(self, tmp_path):
        """A JSON object at top level (not a list) must not crash load."""
        path = tmp_path / "pb.json"
        path.write_text('{"key": "value"}')
        pb = Playbook(store_path=str(path))
        assert pb.get_relevant("anything") == []


class TestPlaybookEmptyOperations:
    """Operations on a brand-new empty playbook must be safe."""

    def test_get_relevant_on_empty_playbook(self, tmp_path):
        pb = Playbook(store_path=str(tmp_path / "pb.json"))
        assert pb.get_relevant("shell") == []

    def test_get_promotable_on_empty_playbook(self, tmp_path):
        pb = Playbook(store_path=str(tmp_path / "pb.json"))
        assert pb.get_promotable() == []

    def test_mark_promoted_unknown_key_raises(self, tmp_path):
        pb = Playbook(store_path=str(tmp_path / "pb.json"))
        with pytest.raises(KeyError):
            pb.mark_promoted("ghost_id")

    def test_save_empty_playbook_writes_empty_array(self, tmp_path):
        path = tmp_path / "pb.json"
        pb = Playbook(store_path=str(path))
        pb.save()
        with open(path) as fh:
            data = json.load(fh)
        assert data == []


class TestPatternIdProperties:
    """_compute_pattern_id contract."""

    def test_pattern_id_is_16_hex_chars(self):
        pid = _compute_pattern_id("shell", ["tool_a", "tool_b"])
        assert len(pid) == 16
        assert all(c in "0123456789abcdef" for c in pid)

    def test_empty_tool_sequence_produces_stable_id(self):
        id1 = _compute_pattern_id("simple", [])
        id2 = _compute_pattern_id("simple", [])
        assert id1 == id2

    def test_single_tool_produces_stable_id(self):
        id1 = _compute_pattern_id("shell", ["only_tool"])
        id2 = _compute_pattern_id("shell", ["only_tool"])
        assert id1 == id2

    def test_different_task_types_differ(self):
        id1 = _compute_pattern_id("typeA", ["t"])
        id2 = _compute_pattern_id("typeB", ["t"])
        assert id1 != id2

    def test_different_tools_differ(self):
        id1 = _compute_pattern_id("shell", ["tool_a"])
        id2 = _compute_pattern_id("shell", ["tool_b"])
        assert id1 != id2


class TestPlaybookConcurrentReadsDuringWrite:
    """Concurrent reads must not conflict with concurrent writes."""

    def test_concurrent_reads_and_writes_no_exception(self, tmp_path):
        pb = Playbook(store_path=str(tmp_path / "pb.json"))
        errors: list[Exception] = []

        def writer(i: int) -> None:
            try:
                pb.record("shell", f"task{i}", ["shell_exec"], f"h{i}")
            except Exception as exc:
                errors.append(exc)

        def reader() -> None:
            try:
                pb.get_relevant("shell")
            except Exception as exc:
                errors.append(exc)

        threads = []
        for i in range(10):
            threads.append(threading.Thread(target=writer, args=(i,)))
            threads.append(threading.Thread(target=reader))
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors


# ---------------------------------------------------------------------------
# DoneCriteria edge cases
# ---------------------------------------------------------------------------


class TestDoneCriteriaAllMet:
    """Boundary conditions for the all_met property."""

    def test_single_condition_verified_true(self):
        dc = DoneCriteria(conditions=["task done"], verified=[True])
        assert dc.all_met is True

    def test_single_condition_verified_false(self):
        dc = DoneCriteria(conditions=["task done"], verified=[False])
        assert dc.all_met is False

    def test_empty_conditions_always_false(self):
        dc = DoneCriteria(conditions=[], verified=[])
        assert dc.all_met is False

    def test_empty_conditions_with_stray_true_in_verified(self):
        """Conditions list is empty; all_met must still return False."""
        dc = DoneCriteria(conditions=[], verified=[True])
        assert dc.all_met is False

    def test_all_false_verified(self):
        dc = DoneCriteria(conditions=["a", "b", "c"], verified=[False, False, False])
        assert dc.all_met is False

    def test_mutating_verified_after_construction(self):
        dc = DoneCriteria(conditions=["x", "y"], verified=[False, False])
        assert dc.all_met is False
        dc.verified[0] = True
        dc.verified[1] = True
        assert dc.all_met is True


class TestDoneCriteriaPending:
    """Boundary conditions for the pending property."""

    def test_pending_empty_when_all_verified(self):
        dc = DoneCriteria(conditions=["a", "b"], verified=[True, True])
        assert dc.pending == []

    def test_pending_all_when_none_verified(self):
        dc = DoneCriteria(conditions=["a", "b", "c"], verified=[False, False, False])
        assert dc.pending == ["a", "b", "c"]

    def test_pending_empty_conditions(self):
        dc = DoneCriteria(conditions=[], verified=[])
        assert dc.pending == []

    def test_pending_verified_longer_than_conditions(self):
        """Extra verified entries beyond conditions length are silently ignored."""
        dc = DoneCriteria(conditions=["a"], verified=[False, True, True])
        # zip(strict=False) stops at the shorter sequence
        assert dc.pending == ["a"]

    def test_pending_verified_shorter_than_conditions(self):
        """Conditions without a corresponding verified entry are not returned."""
        dc = DoneCriteria(conditions=["a", "b", "c"], verified=[True])
        # zip stops at verified length; only "a" is zipped; it's True so not pending
        pending = dc.pending
        # "b" and "c" fall outside zip range — not in pending
        assert "a" not in pending
        # "b" and "c" are absent because zip stopped early
        assert "b" not in pending
        assert "c" not in pending

    def test_pending_single_unverified_condition(self):
        dc = DoneCriteria(conditions=["only one"], verified=[False])
        assert dc.pending == ["only one"]

    def test_pending_order_preserved(self):
        dc = DoneCriteria(
            conditions=["first", "second", "third"],
            verified=[False, True, False],
        )
        assert dc.pending == ["first", "third"]

    def test_appending_condition_after_construction(self):
        dc = DoneCriteria(conditions=["a"], verified=[False])
        dc.conditions.append("b")
        dc.verified.append(True)
        # "a" is still unverified, "b" is verified
        assert dc.pending == ["a"]


class TestIsCompoundTaskEdges:
    """Edge inputs for is_compound_task."""

    def test_whitespace_only_string(self):
        assert is_compound_task("   \t\n  ") is False

    def test_empty_string(self):
        assert is_compound_task("") is False

    def test_case_insensitive_connective_upper(self):
        assert is_compound_task("Do A AND THEN do B") is True

    def test_case_insensitive_ordinal_upper(self):
        assert is_compound_task("FIRST do A, FINALLY do B") is True

    def test_ordinal_second_triggers(self):
        assert is_compound_task("Second, update the config") is True

    def test_ordinal_third_triggers(self):
        assert is_compound_task("Third step is to restart the service") is True

    def test_ordinal_lastly_triggers(self):
        assert is_compound_task("Lastly, send the report") is True

    def test_bullet_star_in_sentence_position(self):
        # Bullet pattern requires start of line or after whitespace indent
        assert is_compound_task("* Do thing A\n* Do thing B") is True

    def test_numbered_list_with_spaces(self):
        assert is_compound_task("  1. First step\n  2. Second step") is True

    def test_simple_question_not_compound(self):
        assert is_compound_task("What is 2 + 2?") is False

    def test_single_word_not_compound(self):
        assert is_compound_task("Help") is False

    def test_then_as_part_of_longer_word_does_not_trigger(self):
        # "then" surrounded by non-word chars should still match \b boundary
        result = is_compound_task("thenthere")
        # "then" is inside a longer word — \b boundary should prevent a match
        assert result is False

    def test_prompt_with_only_numbers_not_compound(self):
        assert is_compound_task("12345") is False

    def test_multiline_prompt_with_connective(self):
        prompt = "Read the file.\nThen write the output to disk."
        assert is_compound_task(prompt) is True


class TestDonePromptContent:
    """make_done_prompt and make_verification_prompt content contracts."""

    def test_done_prompt_contains_done_keyword(self):
        prompt = make_done_prompt()
        assert "DONE" in prompt

    def test_done_prompt_mentions_verifiable(self):
        prompt = make_done_prompt()
        assert "verifiable" in prompt.lower() or "verify" in prompt.lower()

    def test_done_prompt_is_a_non_empty_string(self):
        prompt = make_done_prompt()
        assert isinstance(prompt, str)
        assert len(prompt.strip()) > 0

    def test_verification_prompt_mentions_tool_output(self):
        prompt = make_verification_prompt()
        assert "tool output" in prompt.lower()

    def test_verification_prompt_mentions_complete(self):
        prompt = make_verification_prompt()
        assert "complete" in prompt.lower()

    def test_verification_prompt_is_a_non_empty_string(self):
        prompt = make_verification_prompt()
        assert isinstance(prompt, str)
        assert len(prompt.strip()) > 0

    def test_done_prompt_different_from_verification_prompt(self):
        """The two prompts must be distinct instructions."""
        assert make_done_prompt() != make_verification_prompt()

    def test_make_verification_prompt_is_pure(self):
        """Calling make_verification_prompt multiple times returns equal strings."""
        p1 = make_verification_prompt()
        p2 = make_verification_prompt()
        assert p1 == p2

    def test_make_done_prompt_is_pure(self):
        p1 = make_done_prompt()
        p2 = make_done_prompt()
        assert p1 == p2


class TestDoneCriteriaWithMultipleRounds:
    """Simulate multi-round tool call scenarios."""

    def test_conditions_partially_verified_across_rounds(self):
        """Simulate verifying conditions one by one across tool rounds."""
        dc = DoneCriteria(
            conditions=["file created", "content written", "permissions set"],
            verified=[False, False, False],
        )
        assert dc.pending == ["file created", "content written", "permissions set"]
        assert dc.all_met is False

        # Round 1: file created
        dc.verified[0] = True
        assert dc.pending == ["content written", "permissions set"]
        assert dc.all_met is False

        # Round 2: content written
        dc.verified[1] = True
        assert dc.pending == ["permissions set"]
        assert dc.all_met is False

        # Round 3: permissions set
        dc.verified[2] = True
        assert dc.pending == []
        assert dc.all_met is True

    def test_empty_tool_results_scenario_criteria_not_met(self):
        """No tool results means no conditions can be verified yet."""
        dc = DoneCriteria(
            conditions=["resource exists", "data valid"],
            verified=[False, False],
        )
        # Before any tool runs, nothing is done
        assert dc.all_met is False
        assert len(dc.pending) == 2

    def test_criteria_with_no_conditions_used(self):
        """A task with no explicit criteria is never 'all_met'."""
        dc = DoneCriteria()
        # all_met requires non-empty conditions
        assert dc.all_met is False
        assert dc.pending == []

    def test_single_step_task_done_immediately(self):
        dc = DoneCriteria(conditions=["output printed"], verified=[True])
        assert dc.all_met is True
        assert dc.pending == []

    def test_re_marking_already_verified_condition(self):
        """Setting an already-True verified flag to True again is idempotent."""
        dc = DoneCriteria(conditions=["done"], verified=[True])
        dc.verified[0] = True  # set again
        assert dc.all_met is True
