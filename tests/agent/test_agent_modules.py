"""Comprehensive tests for agent modules with 0% coverage.

Modules covered:
  - missy.agent.learnings
  - missy.agent.prompt_patches
  - missy.agent.sub_agent
  - missy.agent.watchdog
  - missy.agent.heartbeat
  - missy.agent.approval
"""

from __future__ import annotations

import contextlib
import json
import threading
import time
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import pytest

# ---------------------------------------------------------------------------
# missy.agent.learnings
# ---------------------------------------------------------------------------


class TestExtractTaskType:
    """Tests for learnings.extract_task_type."""

    def test_shell_and_web(self):
        from missy.agent.learnings import extract_task_type

        result = extract_task_type(["shell_exec", "web_fetch"])
        assert result == "shell+web"

    def test_shell_and_file(self):
        from missy.agent.learnings import extract_task_type

        result = extract_task_type(["shell_exec", "file_write"])
        assert result == "shell+file"

    def test_shell_only(self):
        from missy.agent.learnings import extract_task_type

        result = extract_task_type(["shell_exec"])
        assert result == "shell"

    def test_web_only(self):
        from missy.agent.learnings import extract_task_type

        result = extract_task_type(["web_fetch"])
        assert result == "web"

    def test_file_read(self):
        from missy.agent.learnings import extract_task_type

        result = extract_task_type(["file_read"])
        assert result == "file"

    def test_file_write(self):
        from missy.agent.learnings import extract_task_type

        result = extract_task_type(["file_write"])
        assert result == "file"

    def test_chat_fallback_empty(self):
        from missy.agent.learnings import extract_task_type

        result = extract_task_type([])
        assert result == "chat"

    def test_chat_fallback_unknown_tool(self):
        from missy.agent.learnings import extract_task_type

        result = extract_task_type(["some_other_tool"])
        assert result == "chat"

    def test_shell_web_priority_over_shell_file(self):
        """shell+web should take precedence when all three tools are present."""
        from missy.agent.learnings import extract_task_type

        result = extract_task_type(["shell_exec", "web_fetch", "file_write"])
        assert result == "shell+web"


class TestExtractOutcome:
    """Tests for learnings.extract_outcome."""

    def test_success_keyword_successfully(self):
        from missy.agent.learnings import extract_outcome

        assert extract_outcome("Successfully created the file.") == "success"

    def test_success_keyword_completed(self):
        from missy.agent.learnings import extract_outcome

        assert extract_outcome("Task completed without issues.") == "success"

    def test_success_keyword_done(self):
        from missy.agent.learnings import extract_outcome

        assert extract_outcome("Done.") == "success"

    def test_success_keyword_finished(self):
        from missy.agent.learnings import extract_outcome

        assert extract_outcome("Finished the job.") == "success"

    def test_success_keyword_worked(self):
        from missy.agent.learnings import extract_outcome

        assert extract_outcome("It worked perfectly.") == "success"

    def test_failure_keyword_failed(self):
        from missy.agent.learnings import extract_outcome

        assert extract_outcome("The command failed.") == "failure"

    def test_failure_keyword_error(self):
        from missy.agent.learnings import extract_outcome

        assert extract_outcome("An error occurred.") == "failure"

    def test_failure_keyword_unable(self):
        from missy.agent.learnings import extract_outcome

        assert extract_outcome("Unable to process request.") == "failure"

    def test_failure_keyword_couldnt(self):
        from missy.agent.learnings import extract_outcome

        assert extract_outcome("I couldn't do that.") == "failure"

    def test_failure_keyword_cannot(self):
        from missy.agent.learnings import extract_outcome

        assert extract_outcome("I cannot complete this.") == "failure"

    def test_partial_fallback(self):
        from missy.agent.learnings import extract_outcome

        assert extract_outcome("Here is some output.") == "partial"

    def test_case_insensitive(self):
        from missy.agent.learnings import extract_outcome

        assert extract_outcome("SUCCESSFULLY processed.") == "success"


class TestExtractLearnings:
    """Tests for learnings.extract_learnings."""

    def test_success_lesson_format(self):
        from missy.agent.learnings import extract_learnings

        learning = extract_learnings(
            tool_names_used=["shell_exec"],
            final_response="Successfully ran the command.",
            prompt="Run a command",
        )
        assert learning.task_type == "shell"
        assert learning.outcome == "success"
        assert "shell" in learning.lesson
        assert "shell_exec" in learning.lesson
        assert "succeeded" in learning.lesson

    def test_failure_lesson_format(self):
        from missy.agent.learnings import extract_learnings

        learning = extract_learnings(
            tool_names_used=["web_fetch"],
            final_response="An error occurred fetching the page.",
            prompt="Fetch a page",
        )
        assert learning.task_type == "web"
        assert learning.outcome == "failure"
        assert "failure" in learning.lesson

    def test_approach_capped_at_five(self):
        from missy.agent.learnings import extract_learnings

        tools = ["t1", "t2", "t3", "t4", "t5", "t6", "t7"]
        learning = extract_learnings(
            tool_names_used=tools,
            final_response="Done.",
            prompt="Many tools",
        )
        assert len(learning.approach) == 5
        assert learning.approach == tools[:5]

    def test_no_tools_uses_direct_response(self):
        from missy.agent.learnings import extract_learnings

        learning = extract_learnings(
            tool_names_used=[],
            final_response="Here is the answer.",
            prompt="What is 2+2?",
        )
        assert learning.approach == ["direct_response"]

    def test_timestamp_is_set(self):
        from missy.agent.learnings import extract_learnings

        learning = extract_learnings(
            tool_names_used=[],
            final_response="Done.",
            prompt="test",
        )
        assert learning.timestamp != ""

    def test_returns_task_learning_instance(self):
        from missy.agent.learnings import TaskLearning, extract_learnings

        learning = extract_learnings(
            tool_names_used=["file_read"],
            final_response="Completed.",
            prompt="Read a file",
        )
        assert isinstance(learning, TaskLearning)


class TestTaskLearning:
    """Tests for the TaskLearning dataclass."""

    def test_timestamp_auto_set(self):
        from missy.agent.learnings import TaskLearning

        t = TaskLearning(task_type="shell", approach=["shell_exec"], outcome="success", lesson="ok")
        assert t.timestamp != ""

    def test_explicit_timestamp_preserved(self):
        from missy.agent.learnings import TaskLearning

        ts = "2025-01-01T00:00:00+00:00"
        t = TaskLearning(
            task_type="chat", approach=[], outcome="partial", lesson="no-op", timestamp=ts
        )
        assert t.timestamp == ts


# ---------------------------------------------------------------------------
# missy.agent.prompt_patches
# ---------------------------------------------------------------------------


@pytest.fixture
def patches_path(tmp_path) -> Path:
    return tmp_path / "patches.json"


@pytest.fixture
def patch_manager(patches_path):
    from missy.agent.prompt_patches import PromptPatchManager

    return PromptPatchManager(store_path=str(patches_path))


class TestPromptPatch:
    """Tests for the PromptPatch dataclass."""

    def test_created_at_auto_set(self):
        from missy.agent.prompt_patches import PatchStatus, PatchType, PromptPatch

        p = PromptPatch(
            id="abc12345",
            patch_type=PatchType.TOOL_USAGE_HINT,
            content="hint",
            confidence=0.5,
        )
        assert p.created_at != ""
        assert p.status == PatchStatus.PROPOSED

    def test_success_rate_zero_when_no_applications(self):
        from missy.agent.prompt_patches import PatchType, PromptPatch

        p = PromptPatch(id="x", patch_type=PatchType.TOOL_USAGE_HINT, content="c", confidence=0.5)
        assert p.success_rate == 0.0

    def test_success_rate_calculation(self):
        from missy.agent.prompt_patches import PatchStatus, PatchType, PromptPatch

        p = PromptPatch(
            id="x",
            patch_type=PatchType.TOOL_USAGE_HINT,
            content="c",
            confidence=0.5,
            status=PatchStatus.APPROVED,
            applications=10,
            successes=7,
        )
        assert p.success_rate == pytest.approx(0.7)

    def test_is_expired_false_when_fewer_than_five_applications(self):
        from missy.agent.prompt_patches import PatchStatus, PatchType, PromptPatch

        p = PromptPatch(
            id="x",
            patch_type=PatchType.ERROR_AVOIDANCE,
            content="c",
            confidence=0.5,
            status=PatchStatus.APPROVED,
            applications=4,
            successes=0,
        )
        assert p.is_expired is False

    def test_is_expired_true_when_low_success_rate(self):
        from missy.agent.prompt_patches import PatchStatus, PatchType, PromptPatch

        p = PromptPatch(
            id="x",
            patch_type=PatchType.ERROR_AVOIDANCE,
            content="c",
            confidence=0.5,
            status=PatchStatus.APPROVED,
            applications=10,
            successes=3,  # 30% < 40%
        )
        assert p.is_expired is True

    def test_is_expired_false_when_good_success_rate(self):
        from missy.agent.prompt_patches import PatchStatus, PatchType, PromptPatch

        p = PromptPatch(
            id="x",
            patch_type=PatchType.ERROR_AVOIDANCE,
            content="c",
            confidence=0.5,
            status=PatchStatus.APPROVED,
            applications=10,
            successes=5,  # 50% >= 40%
        )
        assert p.is_expired is False


class TestPromptPatchManagerPropose:
    """Tests for PromptPatchManager.propose."""

    def test_propose_creates_patch(self, patch_manager):
        from missy.agent.prompt_patches import PatchStatus, PatchType

        patch = patch_manager.propose(PatchType.ERROR_AVOIDANCE, "Avoid errors.")
        assert patch is not None
        assert patch.content == "Avoid errors."
        assert patch.status == PatchStatus.PROPOSED

    def test_propose_auto_approves_high_confidence_tool_hint(self, patch_manager):
        from missy.agent.prompt_patches import PatchStatus, PatchType

        patch = patch_manager.propose(
            PatchType.TOOL_USAGE_HINT, "Always verify paths.", confidence=0.9
        )
        assert patch.status == PatchStatus.APPROVED

    def test_propose_auto_approves_high_confidence_domain_knowledge(self, patch_manager):
        from missy.agent.prompt_patches import PatchStatus, PatchType

        patch = patch_manager.propose(
            PatchType.DOMAIN_KNOWLEDGE, "Python uses indentation.", confidence=0.85
        )
        assert patch.status == PatchStatus.APPROVED

    def test_propose_auto_approves_high_confidence_style_preference(self, patch_manager):
        from missy.agent.prompt_patches import PatchStatus, PatchType

        patch = patch_manager.propose(
            PatchType.STYLE_PREFERENCE, "Use concise responses.", confidence=0.8
        )
        assert patch.status == PatchStatus.APPROVED

    def test_propose_does_not_auto_approve_low_confidence(self, patch_manager):
        from missy.agent.prompt_patches import PatchStatus, PatchType

        patch = patch_manager.propose(PatchType.TOOL_USAGE_HINT, "hint", confidence=0.7)
        assert patch.status == PatchStatus.PROPOSED

    def test_propose_does_not_auto_approve_error_avoidance_type(self, patch_manager):
        from missy.agent.prompt_patches import PatchStatus, PatchType

        patch = patch_manager.propose(PatchType.ERROR_AVOIDANCE, "avoid errors", confidence=0.95)
        assert patch.status == PatchStatus.PROPOSED

    def test_propose_returns_none_when_at_capacity(self, patch_manager):
        from missy.agent.prompt_patches import PatchType

        for i in range(patch_manager.MAX_PATCHES):
            patch_manager.propose(PatchType.ERROR_AVOIDANCE, f"patch {i}")
        result = patch_manager.propose(PatchType.ERROR_AVOIDANCE, "one too many")
        assert result is None

    def test_propose_persists_to_disk(self, patch_manager, patches_path):
        from missy.agent.prompt_patches import PatchType

        patch_manager.propose(PatchType.TOOL_USAGE_HINT, "saved hint")
        data = json.loads(patches_path.read_text())
        assert len(data) == 1
        assert data[0]["content"] == "saved hint"


class TestPromptPatchManagerApproveReject:
    """Tests for approve and reject operations."""

    def test_approve_existing_patch(self, patch_manager):
        from missy.agent.prompt_patches import PatchStatus, PatchType

        patch = patch_manager.propose(PatchType.WORKFLOW_PATTERN, "workflow hint", confidence=0.5)
        result = patch_manager.approve(patch.id)
        assert result is True
        assert patch.status == PatchStatus.APPROVED

    def test_approve_nonexistent_patch(self, patch_manager):
        result = patch_manager.approve("nonexistent")
        assert result is False

    def test_reject_existing_patch(self, patch_manager):
        from missy.agent.prompt_patches import PatchStatus, PatchType

        patch = patch_manager.propose(PatchType.WORKFLOW_PATTERN, "hint", confidence=0.5)
        result = patch_manager.reject(patch.id)
        assert result is True
        assert patch.status == PatchStatus.REJECTED

    def test_reject_nonexistent_patch(self, patch_manager):
        result = patch_manager.reject("nonexistent")
        assert result is False


class TestPromptPatchManagerQueries:
    """Tests for list/query operations on PromptPatchManager."""

    def test_list_proposed_returns_only_proposed(self, patch_manager):
        from missy.agent.prompt_patches import PatchType

        p1 = patch_manager.propose(PatchType.ERROR_AVOIDANCE, "p1", confidence=0.5)
        p2 = patch_manager.propose(PatchType.ERROR_AVOIDANCE, "p2", confidence=0.5)
        patch_manager.approve(p1.id)

        proposed = patch_manager.list_proposed()
        ids = [p.id for p in proposed]
        assert p1.id not in ids
        assert p2.id in ids

    def test_list_all_returns_everything(self, patch_manager):
        from missy.agent.prompt_patches import PatchType

        patch_manager.propose(PatchType.ERROR_AVOIDANCE, "p1", confidence=0.5)
        patch_manager.propose(PatchType.STYLE_PREFERENCE, "p2", confidence=0.9)
        all_patches = patch_manager.list_all()
        assert len(all_patches) == 2

    def test_get_active_patches_returns_approved_non_expired(self, patch_manager):
        from missy.agent.prompt_patches import PatchStatus, PatchType, PromptPatch

        # Manually inject an approved non-expired patch
        p = PromptPatch(
            id="active1",
            patch_type=PatchType.TOOL_USAGE_HINT,
            content="active hint",
            confidence=0.8,
            status=PatchStatus.APPROVED,
            applications=2,
            successes=2,
        )
        patch_manager._patches.append(p)

        active = patch_manager.get_active_patches()
        assert any(ap.id == "active1" for ap in active)

    def test_get_active_patches_excludes_proposed(self, patch_manager):
        from missy.agent.prompt_patches import PatchType

        patch_manager.propose(PatchType.ERROR_AVOIDANCE, "not approved", confidence=0.5)
        active = patch_manager.get_active_patches()
        assert active == []

    def test_get_active_patches_marks_expired(self, patch_manager):
        from missy.agent.prompt_patches import PatchStatus, PatchType, PromptPatch

        p = PromptPatch(
            id="expiring1",
            patch_type=PatchType.ERROR_AVOIDANCE,
            content="bad patch",
            confidence=0.5,
            status=PatchStatus.APPROVED,
            applications=10,
            successes=2,  # 20% < 40% threshold
        )
        patch_manager._patches.append(p)

        active = patch_manager.get_active_patches()
        expired_patch = next(pp for pp in patch_manager._patches if pp.id == "expiring1")
        assert expired_patch.status == PatchStatus.EXPIRED
        assert all(ap.id != "expiring1" for ap in active)


class TestPromptPatchManagerRecordOutcome:
    """Tests for record_outcome method."""

    def test_record_success_increments_applications_and_successes(self, patch_manager):
        from missy.agent.prompt_patches import PatchStatus, PatchType, PromptPatch

        p = PromptPatch(
            id="track1",
            patch_type=PatchType.TOOL_USAGE_HINT,
            content="track",
            confidence=0.8,
            status=PatchStatus.APPROVED,
        )
        patch_manager._patches.append(p)
        patch_manager.record_outcome(success=True)

        assert p.applications == 1
        assert p.successes == 1

    def test_record_failure_increments_applications_only(self, patch_manager):
        from missy.agent.prompt_patches import PatchStatus, PatchType, PromptPatch

        p = PromptPatch(
            id="track2",
            patch_type=PatchType.TOOL_USAGE_HINT,
            content="track",
            confidence=0.8,
            status=PatchStatus.APPROVED,
        )
        patch_manager._patches.append(p)
        patch_manager.record_outcome(success=False)

        assert p.applications == 1
        assert p.successes == 0

    def test_record_outcome_ignores_proposed_patches(self, patch_manager):
        from missy.agent.prompt_patches import PatchType

        p = patch_manager.propose(PatchType.ERROR_AVOIDANCE, "proposed only", confidence=0.5)
        patch_manager.record_outcome(success=True)

        assert p.applications == 0
        assert p.successes == 0


class TestPromptPatchManagerBuildPrompt:
    """Tests for build_patch_prompt method."""

    def test_empty_when_no_active_patches(self, patch_manager):
        result = patch_manager.build_patch_prompt()
        assert result == ""

    def test_contains_active_patch_content(self, patch_manager):
        from missy.agent.prompt_patches import PatchStatus, PatchType, PromptPatch

        p = PromptPatch(
            id="prm1",
            patch_type=PatchType.TOOL_USAGE_HINT,
            content="verify paths",
            confidence=0.9,
            status=PatchStatus.APPROVED,
        )
        patch_manager._patches.append(p)

        result = patch_manager.build_patch_prompt()
        assert "verify paths" in result
        assert "tool_usage_hint" in result
        assert "Active Prompt Guidance" in result

    def test_contains_header_section(self, patch_manager):
        from missy.agent.prompt_patches import PatchStatus, PatchType, PromptPatch

        p = PromptPatch(
            id="prm2",
            patch_type=PatchType.STYLE_PREFERENCE,
            content="be concise",
            confidence=0.9,
            status=PatchStatus.APPROVED,
        )
        patch_manager._patches.append(p)

        result = patch_manager.build_patch_prompt()
        assert result.startswith("\n## Active Prompt Guidance")


class TestPromptPatchManagerPersistence:
    """Tests for load/save round-trip."""

    def test_load_from_existing_file(self, tmp_path):
        from missy.agent.prompt_patches import PatchStatus, PatchType, PromptPatchManager

        store = tmp_path / "patches.json"
        data = [
            {
                "id": "abc12345",
                "patch_type": "tool_usage_hint",
                "content": "loaded hint",
                "confidence": 0.8,
                "status": "approved",
                "applications": 3,
                "successes": 3,
                "created_at": "2025-01-01T00:00:00+00:00",
            }
        ]
        store.write_text(json.dumps(data))

        mgr = PromptPatchManager(store_path=str(store))
        patches = mgr.list_all()
        assert len(patches) == 1
        assert patches[0].id == "abc12345"
        assert patches[0].patch_type == PatchType.TOOL_USAGE_HINT
        assert patches[0].status == PatchStatus.APPROVED

    def test_load_returns_empty_list_when_file_missing(self, tmp_path):
        from missy.agent.prompt_patches import PromptPatchManager

        mgr = PromptPatchManager(store_path=str(tmp_path / "nonexistent.json"))
        assert mgr.list_all() == []

    def test_load_returns_empty_list_when_file_malformed(self, tmp_path):
        from missy.agent.prompt_patches import PromptPatchManager

        store = tmp_path / "patches.json"
        store.write_text("not valid json {{")

        mgr = PromptPatchManager(store_path=str(store))
        assert mgr.list_all() == []


# ---------------------------------------------------------------------------
# missy.agent.sub_agent
# ---------------------------------------------------------------------------


class TestParseSubtasks:
    """Tests for sub_agent.parse_subtasks."""

    def test_numbered_list_parsed(self):
        from missy.agent.sub_agent import parse_subtasks

        tasks = parse_subtasks("1. Search the web\n2. Summarise results\n3. Write report")
        assert len(tasks) == 3
        assert tasks[0].description == "Search the web"
        assert tasks[1].description == "Summarise results"
        assert tasks[2].description == "Write report"
        assert tasks[0].id == 0
        assert tasks[1].id == 1
        assert tasks[2].id == 2

    def test_numbered_list_with_parentheses(self):
        from missy.agent.sub_agent import parse_subtasks

        tasks = parse_subtasks("1) Do this\n2) Do that")
        assert len(tasks) == 2
        assert tasks[0].description == "Do this"

    def test_sequential_connective_then(self):
        from missy.agent.sub_agent import parse_subtasks

        tasks = parse_subtasks("Search the web then summarise the results")
        assert len(tasks) == 2
        assert tasks[1].depends_on == [0]

    def test_sequential_connective_and_then(self):
        from missy.agent.sub_agent import parse_subtasks

        tasks = parse_subtasks("Fetch data and then process it")
        assert len(tasks) == 2
        assert tasks[1].depends_on == [0]

    def test_sequential_connective_finally(self):
        from missy.agent.sub_agent import parse_subtasks

        tasks = parse_subtasks("Do step one finally do step two")
        assert len(tasks) == 2

    def test_fallback_single_task(self):
        from missy.agent.sub_agent import parse_subtasks

        tasks = parse_subtasks("Just do this one thing")
        assert len(tasks) == 1
        assert tasks[0].id == 0
        assert tasks[0].description == "Just do this one thing"

    def test_sequential_first_task_has_no_dependencies(self):
        from missy.agent.sub_agent import parse_subtasks

        tasks = parse_subtasks("Step one then step two")
        assert tasks[0].depends_on == []


class TestSubAgentRunner:
    """Tests for SubAgentRunner."""

    def _make_runner(self, reply="done"):
        mock_runtime = MagicMock()
        mock_runtime.run.return_value = reply
        factory = Mock(return_value=mock_runtime)

        from missy.agent.sub_agent import SubAgentRunner

        runner = SubAgentRunner(runtime_factory=factory)
        return runner, factory, mock_runtime

    def test_run_subtask_returns_result(self):
        from missy.agent.sub_agent import SubTask

        runner, factory, runtime = self._make_runner(reply="task result")
        task = SubTask(id=0, description="do something")

        result = runner.run_subtask(task)
        assert result == "task result"
        assert task.result == "task result"

    def test_run_subtask_prepends_context(self):
        from missy.agent.sub_agent import SubTask

        runner, factory, runtime = self._make_runner()
        task = SubTask(id=1, description="step two")

        runner.run_subtask(task, context="prior result")
        call_prompt = runtime.run.call_args[0][0]
        assert "Context: prior result" in call_prompt
        assert "Task: step two" in call_prompt

    def test_run_subtask_handles_exception(self):
        from missy.agent.sub_agent import SubTask

        mock_runtime = MagicMock()
        mock_runtime.run.side_effect = RuntimeError("boom")
        factory = Mock(return_value=mock_runtime)

        from missy.agent.sub_agent import SubAgentRunner

        runner = SubAgentRunner(runtime_factory=factory)
        task = SubTask(id=0, description="failing task")

        result = runner.run_subtask(task)
        assert "[Error in subtask 0: boom]" in result
        assert task.error == "boom"

    def test_run_all_executes_all_tasks(self):
        from missy.agent.sub_agent import SubTask

        runner, factory, runtime = self._make_runner(reply="ok")

        tasks = [SubTask(id=0, description="a"), SubTask(id=1, description="b")]
        results = runner.run_all(tasks)
        assert len(results) == 2
        assert runtime.run.call_count == 2

    def test_run_all_caps_at_max_total(self):
        from missy.agent.sub_agent import SubTask

        runner, factory, runtime = self._make_runner(reply="ok")

        tasks = [SubTask(id=i, description=f"task {i}") for i in range(15)]
        results = runner.run_all(tasks, max_total=5)
        assert len(results) == 5
        assert runtime.run.call_count == 5

    def test_run_all_accumulates_dependency_context(self):
        from missy.agent.sub_agent import SubTask

        mock_runtime = MagicMock()
        call_prompts = []

        def capture_run(prompt):
            call_prompts.append(prompt)
            return "result"

        mock_runtime.run.side_effect = capture_run
        factory = Mock(return_value=mock_runtime)

        from missy.agent.sub_agent import SubAgentRunner

        runner = SubAgentRunner(runtime_factory=factory)

        tasks = [
            SubTask(id=0, description="first"),
            SubTask(id=1, description="second", depends_on=[0]),
        ]
        runner.run_all(tasks)

        # Second task should have context from first
        assert "Result of step 0:" in call_prompts[1]

    def test_run_all_returns_results_in_order(self):
        from missy.agent.sub_agent import SubTask

        counter = {"n": 0}

        def incremental_run(prompt):
            counter["n"] += 1
            return f"result_{counter['n']}"

        mock_runtime = MagicMock()
        mock_runtime.run.side_effect = incremental_run
        factory = Mock(return_value=mock_runtime)

        from missy.agent.sub_agent import SubAgentRunner

        runner = SubAgentRunner(runtime_factory=factory)
        tasks = [SubTask(id=i, description=f"t{i}") for i in range(3)]
        results = runner.run_all(tasks)
        assert results == ["result_1", "result_2", "result_3"]

    def test_subtask_constants(self):
        from missy.agent.sub_agent import MAX_CONCURRENT, MAX_SUB_AGENTS

        assert MAX_SUB_AGENTS == 10
        assert MAX_CONCURRENT == 3


# ---------------------------------------------------------------------------
# missy.agent.watchdog
# ---------------------------------------------------------------------------


class TestSubsystemHealth:
    """Tests for the SubsystemHealth dataclass."""

    def test_defaults(self):
        from missy.agent.watchdog import SubsystemHealth

        h = SubsystemHealth(name="test")
        assert h.healthy is True
        assert h.consecutive_failures == 0
        assert h.last_checked == 0.0
        assert h.last_error == ""


class TestWatchdog:
    """Tests for the Watchdog class."""

    def test_register_adds_check(self):
        from missy.agent.watchdog import Watchdog

        wd = Watchdog()
        wd.register("db", lambda: True)
        assert "db" in wd._checks
        assert "db" in wd._health

    def test_get_report_empty_before_any_checks(self):
        from missy.agent.watchdog import Watchdog

        wd = Watchdog()
        assert wd.get_report() == {}

    def test_get_report_after_register(self):
        from missy.agent.watchdog import Watchdog

        wd = Watchdog()
        wd.register("svc", lambda: True)
        report = wd.get_report()
        assert "svc" in report
        assert report["svc"]["healthy"] is True
        assert report["svc"]["consecutive_failures"] == 0

    def test_check_all_healthy_check(self):
        from missy.agent.watchdog import Watchdog

        wd = Watchdog()
        wd.register("ok_svc", lambda: True)

        with patch("missy.core.events.event_bus"):
            wd._check_all()

        assert wd._health["ok_svc"].healthy is True
        assert wd._health["ok_svc"].consecutive_failures == 0

    def test_check_all_failing_check_increments_failures(self):
        from missy.agent.watchdog import Watchdog

        wd = Watchdog()
        wd.register("bad_svc", lambda: False)

        with patch("missy.core.events.event_bus"):
            wd._check_all()

        assert wd._health["bad_svc"].healthy is False
        assert wd._health["bad_svc"].consecutive_failures == 1

    def test_check_all_exception_marks_unhealthy(self):
        from missy.agent.watchdog import Watchdog

        wd = Watchdog()
        wd.register("crash_svc", lambda: (_ for _ in ()).throw(RuntimeError("crash")))

        with patch("missy.core.events.event_bus"):
            wd._check_all()

        assert wd._health["crash_svc"].healthy is False
        assert "crash" in wd._health["crash_svc"].last_error

    def test_check_all_recovery_resets_state(self):
        from missy.agent.watchdog import SubsystemHealth, Watchdog

        wd = Watchdog()
        wd.register("recovering_svc", lambda: True)
        # Pre-set as unhealthy
        wd._health["recovering_svc"] = SubsystemHealth(
            name="recovering_svc", healthy=False, consecutive_failures=3
        )

        with patch("missy.core.events.event_bus"):
            wd._check_all()

        assert wd._health["recovering_svc"].healthy is True
        assert wd._health["recovering_svc"].consecutive_failures == 0

    def test_start_creates_daemon_thread(self):
        from missy.agent.watchdog import Watchdog

        wd = Watchdog(check_interval=9999)
        wd.start()
        try:
            assert wd._thread is not None
            assert wd._thread.is_alive()
            assert wd._thread.daemon is True
        finally:
            wd.stop()

    def test_stop_terminates_thread(self):
        from missy.agent.watchdog import Watchdog

        wd = Watchdog(check_interval=9999)
        wd.start()
        wd.stop()
        assert not wd._thread.is_alive()

    def test_multiple_checks_tracked_independently(self):
        from missy.agent.watchdog import Watchdog

        wd = Watchdog()
        wd.register("good", lambda: True)
        wd.register("bad", lambda: False)

        with patch("missy.core.events.event_bus"):
            wd._check_all()

        assert wd._health["good"].healthy is True
        assert wd._health["bad"].healthy is False

    def test_event_bus_publish_called_per_check(self):
        from missy.agent.watchdog import Watchdog

        wd = Watchdog()
        wd.register("svc1", lambda: True)
        wd.register("svc2", lambda: True)

        mock_bus = MagicMock()
        with patch("missy.core.events.event_bus", mock_bus):
            wd._check_all()

        assert mock_bus.publish.call_count == 2

    def test_consecutive_failures_accumulate_across_checks(self):
        from missy.agent.watchdog import Watchdog

        wd = Watchdog(failure_threshold=3)
        wd.register("flaky", lambda: False)

        with patch("missy.core.events.event_bus"):
            wd._check_all()
            wd._check_all()
            wd._check_all()

        assert wd._health["flaky"].consecutive_failures == 3


# ---------------------------------------------------------------------------
# missy.agent.heartbeat
# ---------------------------------------------------------------------------


class TestHeartbeatRunner:
    """Tests for HeartbeatRunner."""

    def _make_runner(self, tmp_path, agent_fn=None, active_hours="", report_fn=None):
        from missy.agent.heartbeat import HeartbeatRunner

        if agent_fn is None:
            agent_fn = Mock(return_value="heartbeat ok")

        return HeartbeatRunner(
            agent_run_fn=agent_fn,
            interval_seconds=9999,
            workspace=str(tmp_path),
            active_hours=active_hours,
            report_fn=report_fn,
        )

    def test_fire_skips_when_ok_file_present(self, tmp_path):
        ok_file = tmp_path / "HEARTBEAT_OK"
        ok_file.touch()

        agent_fn = Mock()
        runner = self._make_runner(tmp_path, agent_fn=agent_fn)
        runner._fire()

        agent_fn.assert_not_called()
        assert runner.metrics["skips"] == 1
        assert not ok_file.exists()

    def test_fire_skips_when_no_heartbeat_md(self, tmp_path):
        agent_fn = Mock()
        runner = self._make_runner(tmp_path, agent_fn=agent_fn)
        runner._fire()

        agent_fn.assert_not_called()
        assert runner.metrics["skips"] == 1

    def test_fire_runs_agent_with_checklist_content(self, tmp_path):
        checklist = tmp_path / "HEARTBEAT.md"
        checklist.write_text("- [ ] Check disk space\n- [ ] Check memory")

        agent_fn = Mock(return_value="all good")
        runner = self._make_runner(tmp_path, agent_fn=agent_fn)
        runner._fire()

        agent_fn.assert_called_once()
        call_prompt = agent_fn.call_args[0][0]
        assert "[HEARTBEAT CHECK]" in call_prompt
        assert "Check disk space" in call_prompt
        assert runner.metrics["runs"] == 1

    def test_fire_calls_report_fn_with_result(self, tmp_path):
        checklist = tmp_path / "HEARTBEAT.md"
        checklist.write_text("tasks here")

        report_fn = Mock()
        runner = self._make_runner(tmp_path, report_fn=report_fn)
        runner._fire()

        report_fn.assert_called_once_with("heartbeat ok")

    def test_fire_handles_agent_exception_gracefully(self, tmp_path):
        checklist = tmp_path / "HEARTBEAT.md"
        checklist.write_text("tasks")

        agent_fn = Mock(side_effect=RuntimeError("agent crashed"))
        runner = self._make_runner(tmp_path, agent_fn=agent_fn)
        runner._fire()  # Should not raise

        assert runner.metrics["runs"] == 0

    def test_fire_skips_when_outside_active_hours(self, tmp_path):
        checklist = tmp_path / "HEARTBEAT.md"
        checklist.write_text("tasks")

        agent_fn = Mock()
        runner = self._make_runner(tmp_path, agent_fn=agent_fn, active_hours="00:00-00:01")

        with patch.object(runner, "_in_active_hours", return_value=False):
            runner._fire()

        agent_fn.assert_not_called()
        assert runner.metrics["skips"] == 1

    def test_metrics_initial_state(self, tmp_path):
        runner = self._make_runner(tmp_path)
        assert runner.metrics == {"runs": 0, "skips": 0}

    def test_start_creates_daemon_thread(self, tmp_path):
        runner = self._make_runner(tmp_path)
        runner.start()
        try:
            assert runner._thread is not None
            assert runner._thread.is_alive()
            assert runner._thread.daemon is True
        finally:
            runner.stop()

    def test_stop_terminates_thread(self, tmp_path):
        runner = self._make_runner(tmp_path)
        runner.start()
        runner.stop()
        assert not runner._thread.is_alive()


class TestHeartbeatActiveHours:
    """Tests for _in_active_hours logic."""

    def _make_runner(self, active_hours, tmp_path):
        from missy.agent.heartbeat import HeartbeatRunner

        return HeartbeatRunner(
            agent_run_fn=Mock(),
            interval_seconds=9999,
            workspace=str(tmp_path),
            active_hours=active_hours,
        )

    def test_returns_true_when_no_active_hours(self, tmp_path):
        runner = self._make_runner("", tmp_path)
        assert runner._in_active_hours() is True

    def test_returns_true_when_inside_window(self, tmp_path):
        from datetime import datetime

        runner = self._make_runner("00:00-23:59", tmp_path)
        fake_now = datetime(2025, 1, 1, 12, 0, 0)
        with patch("datetime.datetime") as mock_dt:
            mock_dt.now.return_value = fake_now
            result = runner._in_active_hours()
        assert result is True

    def test_returns_false_when_outside_window(self, tmp_path):
        from datetime import datetime

        runner = self._make_runner("09:00-17:00", tmp_path)
        fake_now = datetime(2025, 1, 1, 20, 0, 0)
        with patch("datetime.datetime") as mock_dt:
            mock_dt.now.return_value = fake_now
            result = runner._in_active_hours()
        assert result is False

    def test_returns_true_for_invalid_format(self, tmp_path):
        runner = self._make_runner("not-a-time", tmp_path)
        assert runner._in_active_hours() is True


# ---------------------------------------------------------------------------
# missy.agent.approval
# ---------------------------------------------------------------------------


class TestPendingApproval:
    """Tests for PendingApproval."""

    def test_approve_sets_approved_true(self):
        from missy.agent.approval import PendingApproval

        pa = PendingApproval("delete files", "cleanup", timeout=1.0)
        pa.approve()
        assert pa._approved is True

    def test_deny_sets_approved_false(self):
        from missy.agent.approval import PendingApproval

        pa = PendingApproval("delete files", "cleanup", timeout=1.0)
        pa.deny()
        assert pa._approved is False

    def test_wait_returns_true_when_approved(self):
        from missy.agent.approval import PendingApproval

        pa = PendingApproval("action", "reason", timeout=2.0)
        pa.approve()
        assert pa.wait() is True

    def test_wait_raises_approval_denied_when_denied(self):
        from missy.agent.approval import ApprovalDenied, PendingApproval

        pa = PendingApproval("action", "reason", timeout=2.0)
        pa.deny()
        with pytest.raises(ApprovalDenied, match="Action denied"):
            pa.wait()

    def test_wait_raises_approval_timeout_when_no_response(self):
        from missy.agent.approval import ApprovalTimeout, PendingApproval

        pa = PendingApproval("action", "reason", timeout=0.05)
        with pytest.raises(ApprovalTimeout, match="Approval timed out"):
            pa.wait()


class TestApprovalGate:
    """Tests for ApprovalGate."""

    def test_request_sends_message_via_send_fn(self):
        from missy.agent.approval import ApprovalGate

        send_fn = Mock()
        gate = ApprovalGate(send_fn=send_fn, default_timeout=0.05)

        # Run request in a thread so it doesn't block the test
        def auto_approve():
            # Give the request time to register the pending approval
            time.sleep(0.02)
            # Approve the first pending item we find
            with gate._lock:
                for approval_id in list(gate._pending.keys()):
                    gate._pending[approval_id].approve()
                    break

        t = threading.Thread(target=auto_approve, daemon=True)
        t.start()
        gate.request("do something", risk="low")
        t.join(timeout=1.0)

        send_fn.assert_called_once()
        call_msg = send_fn.call_args[0][0]
        assert "Approval Required" in call_msg
        assert "do something" in call_msg

    def test_request_includes_reason_in_message(self):
        from missy.agent.approval import ApprovalGate

        send_fn = Mock()
        gate = ApprovalGate(send_fn=send_fn, default_timeout=0.05)

        def auto_approve():
            time.sleep(0.02)
            with gate._lock:
                for approval_id in list(gate._pending.keys()):
                    gate._pending[approval_id].approve()
                    break

        t = threading.Thread(target=auto_approve, daemon=True)
        t.start()
        gate.request("action", reason="important reason", risk="high")
        t.join(timeout=1.0)

        call_msg = send_fn.call_args[0][0]
        assert "important reason" in call_msg

    def test_request_raises_timeout_when_no_response(self):
        from missy.agent.approval import ApprovalGate, ApprovalTimeout

        gate = ApprovalGate(default_timeout=0.05)
        with pytest.raises(ApprovalTimeout):
            gate.request("risky action")

    def test_request_raises_approval_denied_when_denied(self):
        from missy.agent.approval import ApprovalDenied, ApprovalGate

        gate = ApprovalGate(default_timeout=1.0)

        def auto_deny():
            time.sleep(0.02)
            with gate._lock:
                for approval_id in list(gate._pending.keys()):
                    gate._pending[approval_id].deny()
                    break

        t = threading.Thread(target=auto_deny, daemon=True)
        t.start()
        with pytest.raises(ApprovalDenied):
            gate.request("risky action")
        t.join(timeout=1.0)

    def test_request_cleans_up_pending_after_completion(self):
        from missy.agent.approval import ApprovalGate

        gate = ApprovalGate(default_timeout=0.05)
        with contextlib.suppress(Exception):
            gate.request("action")
        assert len(gate._pending) == 0

    def test_handle_response_approves_matching_id(self):
        from missy.agent.approval import ApprovalGate, PendingApproval

        gate = ApprovalGate(default_timeout=10.0)
        pa = PendingApproval("action", "reason", timeout=10.0)
        gate._pending["abcd1234"] = pa

        result = gate.handle_response("approve abcd1234")
        assert result is True
        assert pa._approved is True

    def test_handle_response_denies_matching_id(self):
        from missy.agent.approval import ApprovalGate, PendingApproval

        gate = ApprovalGate(default_timeout=10.0)
        pa = PendingApproval("action", "reason", timeout=10.0)
        gate._pending["abcd1234"] = pa

        result = gate.handle_response("deny abcd1234")
        assert result is True
        assert pa._approved is False

    def test_handle_response_returns_false_for_unknown_id(self):
        from missy.agent.approval import ApprovalGate

        gate = ApprovalGate(default_timeout=10.0)
        result = gate.handle_response("approve unknownid")
        assert result is False

    def test_handle_response_is_case_insensitive(self):
        from missy.agent.approval import ApprovalGate, PendingApproval

        gate = ApprovalGate(default_timeout=10.0)
        pa = PendingApproval("action", "reason", timeout=10.0)
        gate._pending["abcd1234"] = pa

        result = gate.handle_response("APPROVE ABCD1234")
        assert result is True

    def test_list_pending_returns_all_pending(self):
        from missy.agent.approval import ApprovalGate, PendingApproval

        gate = ApprovalGate(default_timeout=10.0)
        gate._pending["id1"] = PendingApproval("action one", "reason one", timeout=10.0)
        gate._pending["id2"] = PendingApproval("action two", "reason two", timeout=10.0)

        pending = gate.list_pending()
        assert len(pending) == 2
        ids = {item["id"] for item in pending}
        assert "id1" in ids
        assert "id2" in ids

    def test_list_pending_returns_empty_when_none(self):
        from missy.agent.approval import ApprovalGate

        gate = ApprovalGate(default_timeout=10.0)
        assert gate.list_pending() == []

    def test_request_works_without_send_fn(self):
        from missy.agent.approval import ApprovalGate, ApprovalTimeout

        gate = ApprovalGate(send_fn=None, default_timeout=0.05)
        with pytest.raises(ApprovalTimeout):
            gate.request("action without notifier")

    def test_send_fn_exception_does_not_crash_request(self):
        from missy.agent.approval import ApprovalGate, ApprovalTimeout

        def crashing_send(msg):
            raise ConnectionError("cannot send")

        gate = ApprovalGate(send_fn=crashing_send, default_timeout=0.05)
        with pytest.raises(ApprovalTimeout):
            gate.request("action")  # Should timeout, not crash on send failure
