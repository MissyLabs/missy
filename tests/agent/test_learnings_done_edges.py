"""Session 15 — comprehensive edge-case and boundary tests for learnings and done_criteria.

Covers missy.agent.learnings and missy.agent.done_criteria with focus on:
  - TaskLearning dataclass invariants and field types
  - extract_task_type priority ordering and boundary tool combinations
  - extract_outcome keyword precedence and edge text
  - extract_learnings integration: lesson format, approach cap, type/outcome/lesson consistency
  - DoneCriteria properties: all_met and pending with every alignment permutation
  - is_compound_task pattern coverage, case sensitivity, whitespace, unicode
  - make_done_prompt and make_verification_prompt content contracts
  - Cross-module integration: learning extraction feeding done-criteria workflows

Tests in this file are distinct from those already exercised in:
  - tests/agent/test_learnings.py
  - tests/agent/test_done_criteria.py
"""

from __future__ import annotations

import threading
from datetime import UTC, datetime

from missy.agent.done_criteria import (
    _COMPOUND_PATTERNS,
    DoneCriteria,
    is_compound_task,
    make_done_prompt,
    make_verification_prompt,
)
from missy.agent.learnings import (
    TaskLearning,
    extract_learnings,
    extract_outcome,
    extract_task_type,
)

# ===========================================================================
# TaskLearning — dataclass field contracts
# ===========================================================================


class TestTaskLearningFieldContracts:
    """Verify that TaskLearning stores and exposes every field correctly."""

    def test_task_type_stored_verbatim(self):
        tl = TaskLearning(task_type="shell+web", approach=[], outcome="success", lesson="ok")
        assert tl.task_type == "shell+web"

    def test_approach_stored_as_list(self):
        tools = ["tool_a", "tool_b"]
        tl = TaskLearning(task_type="file", approach=tools, outcome="partial", lesson="x")
        assert isinstance(tl.approach, list)
        assert tl.approach == tools

    def test_outcome_stored_verbatim(self):
        for outcome in ("success", "failure", "partial"):
            tl = TaskLearning(task_type="chat", approach=[], outcome=outcome, lesson="l")
            assert tl.outcome == outcome

    def test_lesson_stored_verbatim(self):
        lesson = "shell: shell_exec → file_write succeeded"
        tl = TaskLearning(task_type="shell+file", approach=[], outcome="success", lesson=lesson)
        assert tl.lesson == lesson

    def test_timestamp_auto_set_is_iso8601_utc(self):
        before = datetime.now(UTC).isoformat()
        tl = TaskLearning(task_type="chat", approach=[], outcome="partial", lesson="x")
        after = datetime.now(UTC).isoformat()
        assert tl.timestamp >= before
        assert tl.timestamp <= after

    def test_timestamp_contains_T_separator(self):
        tl = TaskLearning(task_type="chat", approach=[], outcome="partial", lesson="x")
        assert "T" in tl.timestamp

    def test_explicit_timestamp_not_overwritten(self):
        ts = "2000-06-15T12:00:00+00:00"
        tl = TaskLearning(task_type="chat", approach=[], outcome="partial", lesson="x", timestamp=ts)
        assert tl.timestamp == ts

    def test_empty_approach_list_allowed(self):
        tl = TaskLearning(task_type="chat", approach=[], outcome="partial", lesson="x")
        assert tl.approach == []

    def test_approach_list_is_not_mutated_externally(self):
        tools = ["shell_exec"]
        tl = TaskLearning(task_type="shell", approach=tools, outcome="success", lesson="l")
        tools.append("extra")
        # The dataclass stores a reference; document the expected behaviour
        # (no defensive copy) — but the object itself stays consistent.
        assert tl.approach is tools

    def test_task_learning_is_dataclass_instances_are_equal_when_fields_match(self):
        ts = "2026-01-01T00:00:00+00:00"
        tl1 = TaskLearning("chat", [], "partial", "x", timestamp=ts)
        tl2 = TaskLearning("chat", [], "partial", "x", timestamp=ts)
        assert tl1 == tl2

    def test_task_learning_inequality_on_different_task_type(self):
        ts = "2026-01-01T00:00:00+00:00"
        tl1 = TaskLearning("chat", [], "partial", "x", timestamp=ts)
        tl2 = TaskLearning("shell", [], "partial", "x", timestamp=ts)
        assert tl1 != tl2


# ===========================================================================
# extract_task_type — exhaustive priority and combination tests
# ===========================================================================


class TestExtractTaskTypePriority:
    """Verify every branch of the priority ladder in extract_task_type."""

    def test_shell_web_requires_both_exactly(self):
        assert extract_task_type(["shell_exec", "web_fetch"]) == "shell+web"

    def test_shell_web_with_additional_tools_still_shell_web(self):
        assert extract_task_type(["shell_exec", "web_fetch", "file_read", "calculator"]) == "shell+web"

    def test_shell_file_requires_both_and_no_web(self):
        assert extract_task_type(["shell_exec", "file_write"]) == "shell+file"

    def test_shell_file_with_file_read_still_shell_file(self):
        assert extract_task_type(["shell_exec", "file_write", "file_read"]) == "shell+file"

    def test_shell_only_no_web_no_file(self):
        assert extract_task_type(["shell_exec", "calculator"]) == "shell"

    def test_web_only_no_shell_no_file(self):
        assert extract_task_type(["web_fetch", "json_parser"]) == "web"

    def test_file_read_alone_maps_to_file(self):
        assert extract_task_type(["file_read"]) == "file"

    def test_file_write_alone_maps_to_file(self):
        assert extract_task_type(["file_write"]) == "file"

    def test_file_read_and_file_write_maps_to_file(self):
        assert extract_task_type(["file_read", "file_write"]) == "file"

    def test_chat_fallback_single_unknown(self):
        assert extract_task_type(["summarizer"]) == "chat"

    def test_chat_fallback_multiple_unknowns(self):
        assert extract_task_type(["rag_search", "calculator", "formatter"]) == "chat"

    def test_empty_list_returns_chat(self):
        assert extract_task_type([]) == "chat"

    def test_tool_name_case_sensitive_no_match(self):
        # Tool names are checked as exact strings; uppercase variants do NOT match.
        assert extract_task_type(["SHELL_EXEC"]) == "chat"
        assert extract_task_type(["Web_Fetch"]) == "chat"

    def test_duplicate_tool_names_handled(self):
        # Set logic means duplicates collapse; shell+web still detected.
        assert extract_task_type(["shell_exec", "shell_exec", "web_fetch"]) == "shell+web"

    def test_single_web_fetch_not_shell(self):
        result = extract_task_type(["web_fetch"])
        assert result == "web"
        assert result != "shell"

    def test_return_type_is_str(self):
        assert isinstance(extract_task_type([]), str)
        assert isinstance(extract_task_type(["shell_exec"]), str)


# ===========================================================================
# extract_outcome — keyword boundaries and precedence
# ===========================================================================


class TestExtractOutcomeBoundaries:
    """Thorough boundary tests for extract_outcome keyword matching."""

    # --- success keywords ---

    def test_successfully_mid_sentence(self):
        assert extract_outcome("I have successfully deployed the app.") == "success"

    def test_completed_standalone(self):
        assert extract_outcome("completed") == "success"

    def test_done_as_single_word(self):
        assert extract_outcome("Done.") == "success"

    def test_finished_in_past_tense(self):
        assert extract_outcome("The migration has finished.") == "success"

    def test_worked_as_past_tense(self):
        assert extract_outcome("The patch worked.") == "success"

    def test_success_keyword_uppercase(self):
        assert extract_outcome("SUCCESSFULLY APPLIED.") == "success"

    def test_success_keyword_mixed_case(self):
        assert extract_outcome("Task Completed Successfully") == "success"

    # --- failure keywords ---

    def test_failed_at_start(self):
        assert extract_outcome("Failed to connect to the server.") == "failure"

    def test_error_colon(self):
        assert extract_outcome("Error: permission denied.") == "failure"

    def test_unable_to(self):
        assert extract_outcome("I am unable to proceed without more information.") == "failure"

    def test_couldnt_contraction(self):
        assert extract_outcome("I couldn't locate the file.") == "failure"

    def test_cannot_joined(self):
        assert extract_outcome("The system cannot handle this request.") == "failure"

    def test_failure_keyword_uppercase(self):
        assert extract_outcome("FAILED TO START") == "failure"

    # --- success vs failure precedence: success wins when both present ---

    def test_both_success_and_failure_keywords_success_wins(self):
        # "successfully" appears before "failed" — success check runs first.
        result = extract_outcome("I successfully retried after it failed.")
        assert result == "success"

    def test_failure_word_in_success_context_still_success(self):
        # "completed" matches before "error" is evaluated.
        result = extract_outcome("Completed the task despite the error in the logs.")
        assert result == "success"

    # --- partial fallback ---

    def test_neutral_statement_returns_partial(self):
        assert extract_outcome("The analysis shows three possible routes.") == "partial"

    def test_numeric_only_response_returns_partial(self):
        assert extract_outcome("42") == "partial"

    def test_whitespace_only_returns_partial(self):
        assert extract_outcome("   \n\t  ") == "partial"

    def test_empty_string_returns_partial(self):
        assert extract_outcome("") == "partial"

    def test_return_value_is_one_of_three_valid_strings(self):
        valid = {"success", "failure", "partial"}
        for text in ("all done!", "error occurred", "reviewing now"):
            assert extract_outcome(text) in valid


# ===========================================================================
# extract_learnings — integration of type + outcome + lesson construction
# ===========================================================================


class TestExtractLearningsIntegration:
    """Verify the full extract_learnings pipeline produces consistent results."""

    def test_lesson_contains_task_type_for_success(self):
        tl = extract_learnings(["shell_exec"], "Successfully ran the command.", "run cmd")
        assert tl.task_type in tl.lesson

    def test_lesson_contains_task_type_for_failure(self):
        tl = extract_learnings(["web_fetch"], "Failed to fetch the page.", "get url")
        assert tl.task_type in tl.lesson

    def test_lesson_contains_task_type_for_partial(self):
        tl = extract_learnings(["file_read"], "I read three lines of the file.", "read file")
        assert tl.task_type in tl.lesson

    def test_lesson_success_contains_succeeded_word(self):
        tl = extract_learnings(["shell_exec", "file_write"], "Task completed.", "create file")
        assert "succeeded" in tl.lesson

    def test_lesson_failure_contains_outcome_word(self):
        tl = extract_learnings(["web_fetch"], "Error: timeout", "fetch url")
        assert "failure" in tl.lesson

    def test_lesson_partial_contains_outcome_word(self):
        tl = extract_learnings(["file_read"], "Here is a summary of the content.", "summarize")
        assert "partial" in tl.lesson

    def test_approach_uses_arrow_separator_in_lesson(self):
        tl = extract_learnings(
            ["shell_exec", "file_write"],
            "Successfully done.",
            "create config",
        )
        assert "→" in tl.lesson

    def test_approach_single_tool_no_arrow(self):
        tl = extract_learnings(["shell_exec"], "Task completed.", "run one cmd")
        assert "→" not in tl.lesson

    def test_approach_exactly_five_tools(self):
        tools = ["t1", "t2", "t3", "t4", "t5"]
        tl = extract_learnings(tools, "Finished successfully.", "five steps")
        assert tl.approach == tools

    def test_approach_six_tools_capped_at_five(self):
        tools = ["t1", "t2", "t3", "t4", "t5", "t6"]
        tl = extract_learnings(tools, "Done.", "many steps")
        assert len(tl.approach) == 5
        assert tl.approach == tools[:5]

    def test_approach_ten_tools_capped_at_five(self):
        tools = [f"tool_{i}" for i in range(10)]
        tl = extract_learnings(tools, "Finished.", "ten steps")
        assert len(tl.approach) == 5

    def test_empty_tools_approach_is_direct_response(self):
        tl = extract_learnings([], "The answer is here.", "simple question")
        assert tl.approach == ["direct_response"]

    def test_empty_tools_task_type_is_chat(self):
        tl = extract_learnings([], "Here is the analysis.", "what is X?")
        assert tl.task_type == "chat"

    def test_timestamp_set_automatically(self):
        tl = extract_learnings(["shell_exec"], "Done.", "cmd")
        assert tl.timestamp  # non-empty

    def test_returned_object_is_task_learning_instance(self):
        tl = extract_learnings(["file_read"], "Done.", "read file")
        assert isinstance(tl, TaskLearning)

    def test_prompt_parameter_does_not_affect_type_or_outcome(self):
        """The prompt parameter is accepted but currently unused for type/outcome."""
        tl1 = extract_learnings(["shell_exec"], "Completed.", "prompt alpha")
        tl2 = extract_learnings(["shell_exec"], "Completed.", "prompt beta")
        assert tl1.task_type == tl2.task_type
        assert tl1.outcome == tl2.outcome

    def test_extract_learnings_with_single_web_tool_partial(self):
        tl = extract_learnings(["web_fetch"], "The page was retrieved.", "browse X")
        assert tl.task_type == "web"
        assert tl.outcome == "partial"
        assert "partial" in tl.lesson


# ===========================================================================
# DoneCriteria — all_met property exhaustive tests
# ===========================================================================


class TestDoneCriteriaAllMet:
    """Comprehensive tests for the all_met property."""

    def test_no_conditions_no_verified_is_false(self):
        dc = DoneCriteria()
        assert dc.all_met is False

    def test_one_condition_one_true_verified_is_true(self):
        dc = DoneCriteria(conditions=["done"], verified=[True])
        assert dc.all_met is True

    def test_one_condition_one_false_verified_is_false(self):
        dc = DoneCriteria(conditions=["done"], verified=[False])
        assert dc.all_met is False

    def test_two_conditions_both_true_is_true(self):
        dc = DoneCriteria(conditions=["a", "b"], verified=[True, True])
        assert dc.all_met is True

    def test_two_conditions_first_false_second_true_is_false(self):
        dc = DoneCriteria(conditions=["a", "b"], verified=[False, True])
        assert dc.all_met is False

    def test_two_conditions_first_true_second_false_is_false(self):
        dc = DoneCriteria(conditions=["a", "b"], verified=[True, False])
        assert dc.all_met is False

    def test_five_conditions_all_true_is_true(self):
        dc = DoneCriteria(conditions=list("abcde"), verified=[True] * 5)
        assert dc.all_met is True

    def test_five_conditions_last_false_is_false(self):
        dc = DoneCriteria(conditions=list("abcde"), verified=[True, True, True, True, False])
        assert dc.all_met is False

    def test_conditions_present_verified_empty_is_true(self):
        """all([]) is True; conditions non-empty; so all_met is True — known edge case."""
        dc = DoneCriteria(conditions=["task1"], verified=[])
        assert dc.all_met is True

    def test_return_type_is_bool(self):
        dc = DoneCriteria(conditions=["a"], verified=[True])
        result = dc.all_met
        assert type(result) is bool  # noqa: E721

    def test_mutating_verified_after_creation_affects_all_met(self):
        """verified is a mutable list; changes are reflected immediately."""
        dc = DoneCriteria(conditions=["a", "b"], verified=[True, False])
        assert dc.all_met is False
        dc.verified[1] = True
        assert dc.all_met is True

    def test_mutating_conditions_after_creation_affects_all_met(self):
        dc = DoneCriteria(conditions=[], verified=[])
        assert dc.all_met is False
        dc.conditions.append("new condition")
        # verified still empty → all([]) == True, conditions non-empty → all_met True
        assert dc.all_met is True


# ===========================================================================
# DoneCriteria — pending property exhaustive tests
# ===========================================================================


class TestDoneCriteriaPending:
    """Comprehensive tests for the pending property."""

    def test_empty_conditions_returns_empty_pending(self):
        dc = DoneCriteria()
        assert dc.pending == []

    def test_all_verified_returns_empty_pending(self):
        dc = DoneCriteria(conditions=["a", "b", "c"], verified=[True, True, True])
        assert dc.pending == []

    def test_none_verified_returns_all_conditions(self):
        dc = DoneCriteria(conditions=["x", "y"], verified=[False, False])
        assert dc.pending == ["x", "y"]

    def test_first_verified_second_not_returns_second(self):
        dc = DoneCriteria(conditions=["step1", "step2"], verified=[True, False])
        assert dc.pending == ["step2"]

    def test_first_not_verified_second_verified_returns_first(self):
        dc = DoneCriteria(conditions=["step1", "step2"], verified=[False, True])
        assert dc.pending == ["step1"]

    def test_middle_condition_not_verified(self):
        dc = DoneCriteria(conditions=["a", "b", "c"], verified=[True, False, True])
        assert dc.pending == ["b"]

    def test_five_conditions_alternating_verified(self):
        dc = DoneCriteria(
            conditions=["c0", "c1", "c2", "c3", "c4"],
            verified=[True, False, True, False, True],
        )
        assert dc.pending == ["c1", "c3"]

    def test_pending_returns_list_type(self):
        dc = DoneCriteria(conditions=["a"], verified=[False])
        assert isinstance(dc.pending, list)

    def test_pending_with_verified_shorter_than_conditions(self):
        """zip strict=False stops at shorter sequence; unmatched conditions excluded."""
        dc = DoneCriteria(conditions=["a", "b", "c"], verified=[True])
        # only first pair is consumed; "b" and "c" have no verified entry → not in pending
        assert "a" not in dc.pending

    def test_pending_with_verified_longer_than_conditions(self):
        """Extra verified entries are ignored by zip."""
        dc = DoneCriteria(conditions=["a"], verified=[False, True, True])
        assert dc.pending == ["a"]

    def test_pending_reflects_mutation_of_verified(self):
        dc = DoneCriteria(conditions=["a", "b"], verified=[False, False])
        assert "a" in dc.pending
        dc.verified[0] = True
        assert "a" not in dc.pending

    def test_pending_order_matches_conditions_order(self):
        conditions = ["first", "second", "third"]
        dc = DoneCriteria(conditions=conditions, verified=[False, False, False])
        assert dc.pending == conditions


# ===========================================================================
# DoneCriteria — construction and mutation
# ===========================================================================


class TestDoneCriteriaConstruction:
    def test_default_construction_empty_lists(self):
        dc = DoneCriteria()
        assert dc.conditions == []
        assert dc.verified == []

    def test_conditions_stored_correctly(self):
        conds = ["step A", "step B"]
        dc = DoneCriteria(conditions=conds, verified=[])
        assert dc.conditions == conds

    def test_verified_stored_correctly(self):
        vrf = [True, False]
        dc = DoneCriteria(conditions=["x", "y"], verified=vrf)
        assert dc.verified == vrf

    def test_append_condition_after_creation(self):
        dc = DoneCriteria()
        dc.conditions.append("new step")
        assert "new step" in dc.conditions

    def test_set_verified_flag_directly(self):
        dc = DoneCriteria(conditions=["task"], verified=[False])
        dc.verified[0] = True
        assert dc.verified[0] is True


# ===========================================================================
# is_compound_task — pattern coverage
# ===========================================================================


class TestIsCompoundTaskPatterns:
    """Detailed tests for every pattern in _COMPOUND_PATTERNS."""

    # --- sequential connectives ---

    def test_then_in_sentence(self):
        assert is_compound_task("Open the file then read it") is True

    def test_and_then_phrase(self):
        assert is_compound_task("Download the data and then process it") is True

    def test_after_that_phrase(self):
        assert is_compound_task("Submit the form after that check the confirmation") is True

    def test_followed_by_phrase(self):
        assert is_compound_task("Run the tests followed by deployment") is True

    def test_subsequently_word(self):
        assert is_compound_task("Start the server subsequently connect the client") is True

    def test_connective_case_insensitive_upper(self):
        assert is_compound_task("Do X THEN do Y") is True

    def test_connective_case_insensitive_mixed(self):
        assert is_compound_task("Do X And Then do Y") is True

    # --- numbered lists ---

    def test_numbered_dot_format(self):
        assert is_compound_task("1. Install\n2. Configure") is True

    def test_numbered_paren_format(self):
        assert is_compound_task("1) Install\n2) Configure") is True

    def test_single_numbered_item_matches(self):
        assert is_compound_task("1. Just one step") is True

    # --- bullet lists ---

    def test_dash_bullet(self):
        assert is_compound_task("- Install the package\n- Run the tests") is True

    def test_asterisk_bullet(self):
        assert is_compound_task("* Step one\n* Step two") is True

    def test_dash_with_leading_spaces(self):
        assert is_compound_task("  - Indented item") is True

    # --- ordinal words ---

    def test_ordinal_first(self):
        assert is_compound_task("First clone the repo") is True

    def test_ordinal_second(self):
        assert is_compound_task("Second install dependencies") is True

    def test_ordinal_third(self):
        assert is_compound_task("Third run tests") is True

    def test_ordinal_finally(self):
        assert is_compound_task("Finally deploy") is True

    def test_ordinal_lastly(self):
        assert is_compound_task("Lastly clean up temp files") is True

    def test_ordinal_case_insensitive(self):
        assert is_compound_task("FIRST do A then B") is True

    # --- non-compound prompts ---

    def test_simple_question(self):
        assert is_compound_task("What time is it?") is False

    def test_single_imperative(self):
        assert is_compound_task("Deploy the application") is False

    def test_empty_string(self):
        assert is_compound_task("") is False

    def test_whitespace_only(self):
        assert is_compound_task("   \n\t  ") is False

    def test_word_then_in_another_context(self):
        # "then" as connective is matched regardless of meaning context
        assert is_compound_task("If it rains then bring an umbrella") is True

    def test_no_pattern_in_plain_prose(self):
        prose = (
            "The system processes requests by looking up the configuration "
            "and returning the appropriate response to the caller."
        )
        assert is_compound_task(prose) is False

    def test_return_type_is_bool(self):
        result = is_compound_task("plain text")
        assert type(result) is bool  # noqa: E721

    # --- unicode / special characters ---

    def test_unicode_then_equivalent_not_matched(self):
        """Non-ASCII connectives are not matched by the ASCII patterns."""
        assert is_compound_task("Do A ＴＨＥＮdo B") is False

    def test_prompt_with_only_numbers_no_list_format(self):
        assert is_compound_task("Use port 8080 for connections") is False


# ===========================================================================
# _COMPOUND_PATTERNS module constant
# ===========================================================================


class TestCompoundPatternsConstant:
    def test_patterns_is_nonempty_list(self):
        assert isinstance(_COMPOUND_PATTERNS, list)
        assert len(_COMPOUND_PATTERNS) > 0

    def test_all_entries_are_compiled_regex(self):
        import re

        for p in _COMPOUND_PATTERNS:
            assert hasattr(p, "search"), f"Entry is not a compiled regex: {p!r}"
            assert isinstance(p, type(re.compile("")))

    def test_four_patterns_total(self):
        assert len(_COMPOUND_PATTERNS) == 4


# ===========================================================================
# make_done_prompt and make_verification_prompt — content contracts
# ===========================================================================


class TestMakeDonePromptContracts:
    def test_returns_string(self):
        assert isinstance(make_done_prompt(), str)

    def test_not_empty(self):
        assert len(make_done_prompt()) > 0

    def test_contains_done_keyword(self):
        assert "DONE" in make_done_prompt()

    def test_contains_conditions_word(self):
        assert "conditions" in make_done_prompt().lower()

    def test_instructs_to_define(self):
        assert "define" in make_done_prompt().lower()

    def test_contains_verifiable_or_specific(self):
        prompt = make_done_prompt().lower()
        assert "verifiable" in prompt or "specific" in prompt

    def test_returns_same_string_each_call(self):
        """Deterministic — no randomness or state."""
        assert make_done_prompt() == make_done_prompt()

    def test_minimum_length(self):
        assert len(make_done_prompt()) >= 50


class TestMakeVerificationPromptContracts:
    def test_returns_string(self):
        assert isinstance(make_verification_prompt(), str)

    def test_not_empty(self):
        assert len(make_verification_prompt()) > 0

    def test_contains_tool_output_reference(self):
        vp = make_verification_prompt().lower()
        assert "tool output" in vp or "review" in vp

    def test_contains_complete_or_completion_reference(self):
        vp = make_verification_prompt().lower()
        assert "complete" in vp

    def test_instructs_to_retry_on_failure(self):
        vp = make_verification_prompt().lower()
        assert "retry" in vp or "again" in vp or "fix" in vp

    def test_does_not_contain_done_keyword(self):
        """Verification prompt is distinct from the DONE definition prompt."""
        assert "DONE" not in make_verification_prompt()

    def test_returns_same_string_each_call(self):
        assert make_verification_prompt() == make_verification_prompt()

    def test_minimum_length(self):
        assert len(make_verification_prompt()) >= 50


# ===========================================================================
# extract_task_type — thread-safety (pure function, no shared state)
# ===========================================================================


class TestExtractTaskTypeThreadSafety:
    """extract_task_type is a pure function; concurrent calls must not interfere."""

    def test_concurrent_calls_return_consistent_results(self):
        results = {}
        errors = []

        def worker(idx: int) -> None:
            try:
                tools = ["shell_exec", "web_fetch"] if idx % 2 == 0 else ["file_read"]
                expected = "shell+web" if idx % 2 == 0 else "file"
                result = extract_task_type(tools)
                results[idx] = (result, expected)
            except Exception as exc:  # noqa: BLE001
                errors.append(exc)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors, f"Thread errors: {errors}"
        for idx, (result, expected) in results.items():
            assert result == expected, f"Thread {idx}: got {result!r}, expected {expected!r}"


# ===========================================================================
# extract_learnings — concurrent calls produce independent TaskLearning objects
# ===========================================================================


class TestExtractLearningsThreadSafety:
    def test_concurrent_extract_learnings_no_cross_contamination(self):
        learnings = []
        lock = threading.Lock()
        errors = []

        def worker(tools: list[str], response: str, prompt: str) -> None:
            try:
                tl = extract_learnings(tools, response, prompt)
                with lock:
                    learnings.append(tl)
            except Exception as exc:  # noqa: BLE001
                errors.append(exc)

        threads = []
        for i in range(10):
            if i % 2 == 0:
                t = threading.Thread(
                    target=worker,
                    args=(["shell_exec"], "Successfully done.", f"prompt {i}"),
                )
            else:
                t = threading.Thread(
                    target=worker,
                    args=(["web_fetch"], "Error: timeout", f"prompt {i}"),
                )
            threads.append(t)

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors
        assert len(learnings) == 10
        shell_results = [lr for lr in learnings if lr.task_type == "shell"]
        web_results = [lr for lr in learnings if lr.task_type == "web"]
        assert len(shell_results) == 5
        assert len(web_results) == 5


# ===========================================================================
# Cross-module integration: learnings output consumed by DoneCriteria workflow
# ===========================================================================


class TestLearningsAndDoneCriteriaIntegration:
    """Simulate how learning extraction and done-criteria interact in practice."""

    def test_successful_learning_maps_to_met_done_criteria(self):
        """A success outcome corresponds to a fully-met DoneCriteria."""
        tl = extract_learnings(
            ["shell_exec", "file_write"],
            "Successfully created and deployed the config.",
            "create and deploy config",
        )
        assert tl.outcome == "success"
        # Operator builds DoneCriteria from task outcomes
        dc = DoneCriteria(
            conditions=["config created", "config deployed"],
            verified=[tl.outcome == "success", tl.outcome == "success"],
        )
        assert dc.all_met is True
        assert dc.pending == []

    def test_failure_outcome_leaves_conditions_pending(self):
        tl = extract_learnings(
            ["web_fetch"],
            "Failed to retrieve the data.",
            "fetch and parse data",
        )
        assert tl.outcome == "failure"
        dc = DoneCriteria(
            conditions=["data fetched", "data parsed"],
            verified=[tl.outcome == "success", False],
        )
        assert dc.all_met is False
        assert "data fetched" in dc.pending

    def test_compound_task_detection_then_done_criteria_setup(self):
        """is_compound_task detects a multi-step prompt; DoneCriteria tracks each step."""
        prompt = "Download the file then parse it and then generate a report"
        assert is_compound_task(prompt) is True
        # Two steps + one synthesis step
        dc = DoneCriteria(
            conditions=["file downloaded", "file parsed", "report generated"],
            verified=[False, False, False],
        )
        assert dc.all_met is False
        assert len(dc.pending) == 3
        # Simulate step 1 completion
        dc.verified[0] = True
        assert dc.pending == ["file parsed", "report generated"]
        # Simulate steps 2 and 3
        dc.verified[1] = True
        dc.verified[2] = True
        assert dc.all_met is True
        assert dc.pending == []

    def test_make_done_prompt_used_for_compound_task(self):
        """make_done_prompt should be invoked when is_compound_task returns True."""
        prompt = "First clean up old logs then archive the data"
        assert is_compound_task(prompt) is True
        done_prompt = make_done_prompt()
        assert "DONE" in done_prompt
        assert len(done_prompt) > 0

    def test_make_verification_prompt_after_partial_learning(self):
        """After a partial outcome, verification prompt should be injected."""
        tl = extract_learnings(["file_read"], "I read the data.", "analyze file")
        assert tl.outcome == "partial"
        vp = make_verification_prompt()
        assert len(vp) > 0
        assert "complete" in vp.lower()

    def test_extract_learnings_lesson_derivable_from_done_criteria_state(self):
        """The lesson format from extract_learnings aligns with task_type description."""
        tools = ["shell_exec", "file_write"]
        response = "Task completed."
        tl = extract_learnings(tools, response, "create and write file")
        # lesson should reference the task type
        assert "shell+file" in tl.lesson
        # done criteria built from outcome
        dc = DoneCriteria(
            conditions=["file written"],
            verified=[tl.outcome == "success"],
        )
        assert dc.all_met is True

    def test_chat_type_learning_with_simple_done_criteria(self):
        """Pure chat responses (no tools) generate chat-type learnings and trivial criteria."""
        tl = extract_learnings([], "The answer is 42.", "what is 6 times 7?")
        assert tl.task_type == "chat"
        assert tl.approach == ["direct_response"]
        # A direct-response task has a single condition: response delivered
        dc = DoneCriteria(conditions=["response delivered"], verified=[True])
        assert dc.all_met is True

    def test_is_compound_returns_false_for_simple_learn_task(self):
        """Non-compound prompts should not trigger DONE prompt overhead."""
        simple_prompt = "List all files in the current directory"
        assert is_compound_task(simple_prompt) is False
        # Still safe to extract learnings from it
        tl = extract_learnings(["shell_exec"], "Finished successfully.", simple_prompt)
        assert tl.task_type == "shell"
        assert tl.outcome == "success"
