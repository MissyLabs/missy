"""Tests for missy.agent.learnings — cross-task learning extraction."""

from __future__ import annotations

from missy.agent.learnings import (
    TaskLearning,
    extract_learnings,
    extract_outcome,
    extract_task_type,
)


class TestTaskLearning:
    def test_auto_timestamp(self):
        tl = TaskLearning(
            task_type="shell", approach=["shell_exec"], outcome="success", lesson="ok"
        )
        assert tl.timestamp  # auto-populated

    def test_explicit_timestamp(self):
        tl = TaskLearning(
            task_type="chat",
            approach=[],
            outcome="partial",
            lesson="x",
            timestamp="2026-01-01T00:00:00",
        )
        assert tl.timestamp == "2026-01-01T00:00:00"


class TestExtractTaskType:
    def test_shell_and_web(self):
        assert extract_task_type(["shell_exec", "web_fetch"]) == "shell+web"

    def test_shell_and_file(self):
        assert extract_task_type(["shell_exec", "file_write"]) == "shell+file"

    def test_shell_only(self):
        assert extract_task_type(["shell_exec"]) == "shell"

    def test_web_only(self):
        assert extract_task_type(["web_fetch"]) == "web"

    def test_file_read(self):
        assert extract_task_type(["file_read"]) == "file"

    def test_file_write(self):
        assert extract_task_type(["file_write"]) == "file"

    def test_chat_fallback(self):
        assert extract_task_type([]) == "chat"

    def test_unknown_tool(self):
        assert extract_task_type(["calculator"]) == "chat"

    def test_priority_shell_web_over_file(self):
        assert extract_task_type(["shell_exec", "web_fetch", "file_write"]) == "shell+web"

    def test_priority_shell_file_over_shell_alone(self):
        assert extract_task_type(["shell_exec", "file_write", "calculator"]) == "shell+file"


class TestExtractOutcome:
    def test_success_keywords(self):
        assert extract_outcome("Successfully created the file.") == "success"
        assert extract_outcome("Task completed.") == "success"
        assert extract_outcome("The process is done now.") == "success"
        assert extract_outcome("Everything finished cleanly.") == "success"
        assert extract_outcome("That worked!") == "success"

    def test_failure_keywords(self):
        assert extract_outcome("The command failed with exit code 1.") == "failure"
        assert extract_outcome("Error: file not found.") == "failure"
        assert extract_outcome("I was unable to complete the task.") == "failure"
        assert extract_outcome("I couldn't find the file.") == "failure"
        assert extract_outcome("Cannot proceed without permissions.") == "failure"

    def test_partial_fallback(self):
        assert extract_outcome("I reviewed the code.") == "partial"
        assert extract_outcome("Here is the analysis.") == "partial"

    def test_case_insensitive(self):
        assert extract_outcome("SUCCESSFULLY DONE") == "success"
        assert extract_outcome("FAILED TO CONNECT") == "failure"

    def test_empty_string(self):
        assert extract_outcome("") == "partial"


class TestExtractLearnings:
    def test_success_learning(self):
        tl = extract_learnings(
            tool_names_used=["shell_exec", "file_write"],
            final_response="Successfully created the config file.",
            prompt="Create a config file",
        )
        assert tl.task_type == "shell+file"
        assert tl.outcome == "success"
        assert "succeeded" in tl.lesson
        assert tl.approach == ["shell_exec", "file_write"]

    def test_failure_learning(self):
        tl = extract_learnings(
            tool_names_used=["web_fetch"],
            final_response="Failed to fetch the URL due to timeout.",
            prompt="Get the page",
        )
        assert tl.task_type == "web"
        assert tl.outcome == "failure"
        assert "failure" in tl.lesson

    def test_no_tools_direct_response(self):
        tl = extract_learnings(
            tool_names_used=[],
            final_response="The answer is 42.",
            prompt="What is the meaning of life?",
        )
        assert tl.task_type == "chat"
        assert tl.approach == ["direct_response"]

    def test_approach_capped_at_five(self):
        tools = [f"tool_{i}" for i in range(10)]
        tl = extract_learnings(tools, "Done successfully.", "Do many things")
        assert len(tl.approach) == 5

    def test_partial_outcome(self):
        tl = extract_learnings(
            ["file_read"],
            "I read the file and here is the content.",
            "Read this file",
        )
        assert tl.outcome == "partial"
        assert "partial" in tl.lesson
