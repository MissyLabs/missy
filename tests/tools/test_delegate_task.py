"""Tests for DelegateTaskTool — SR-4.2 sub-agent delegation tool."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from missy.agent.sub_agent import MAX_SUB_AGENT_DEPTH
from missy.tools.builtin.delegate_task import DelegateTaskTool


@pytest.fixture
def tool():
    return DelegateTaskTool()


class TestNoRuntimeContext:
    def test_missing_runtime_is_rejected(self, tool):
        r = tool.execute(prompt="1. a\n2. b", _runtime=None, _session_id="s", _depth=0)
        assert not r.success
        assert "runtime context" in r.error


class TestDepthLimit:
    def test_at_max_depth_is_rejected(self, tool):
        r = tool.execute(prompt="1. a", _runtime=MagicMock(), _session_id="s", _depth=MAX_SUB_AGENT_DEPTH)
        assert not r.success
        assert "Delegation depth limit" in r.error

    def test_beyond_max_depth_is_rejected(self, tool):
        r = tool.execute(
            prompt="1. a", _runtime=MagicMock(), _session_id="s", _depth=MAX_SUB_AGENT_DEPTH + 5
        )
        assert not r.success

    def test_below_max_depth_is_allowed_through(self, tool):
        runtime = MagicMock()
        runtime.run.return_value = "ok"
        r = tool.execute(prompt="1. a", _runtime=runtime, _session_id="s", _depth=0)
        assert r.success


class TestPromptValidation:
    def test_empty_prompt_rejected(self, tool):
        r = tool.execute(prompt="", _runtime=MagicMock(), _session_id="s", _depth=0)
        assert not r.success
        assert "prompt is required" in r.error

    def test_whitespace_only_prompt_rejected(self, tool):
        r = tool.execute(prompt="   ", _runtime=MagicMock(), _session_id="s", _depth=0)
        assert not r.success


class TestDispatch:
    def test_single_subtask_success(self, tool):
        runtime = MagicMock()
        runtime.run.return_value = "the answer"
        r = tool.execute(prompt="do the thing", _runtime=runtime, _session_id="sess-1", _depth=0)
        assert r.success
        assert "the answer" in r.output

    def test_multi_step_success_includes_all_steps(self, tool):
        runtime = MagicMock()
        runtime.run.side_effect = ["result a", "result b"]
        r = tool.execute(prompt="1. do a\n2. do b", _runtime=runtime, _session_id="sess-1", _depth=0)
        assert r.success
        assert "result a" in r.output
        assert "result b" in r.output

    def test_depth_incremented_for_children(self, tool):
        """The sub-agent's own delegate_task calls (if any) must see
        depth+1, not the same depth -- otherwise the depth guard never
        actually bounds recursion."""
        runtime = MagicMock()
        runtime.run.return_value = "ok"
        tool.execute(prompt="1. a", _runtime=runtime, _session_id="sess-1", _depth=0)
        assert runtime.run.call_args.kwargs["_delegation_depth"] == 1

    def test_session_id_forwarded_for_budget_aggregation(self, tool):
        runtime = MagicMock()
        runtime.run.return_value = "ok"
        tool.execute(prompt="1. a", _runtime=runtime, _session_id="parent-session", _depth=0)
        assert runtime.run.call_args.kwargs["session_id"] == "parent-session"

    def test_one_step_failure_marks_whole_result_as_error(self, tool):
        runtime = MagicMock()
        runtime.run.side_effect = RuntimeError("boom")
        r = tool.execute(prompt="1. a\n2. b", _runtime=runtime, _session_id="sess-1", _depth=0)
        assert not r.success
        assert "failed" in r.error.lower()
        # Partial output is still surfaced, not swallowed.
        assert "boom" in r.output

    def test_more_than_max_sub_agents_subtasks_does_not_crash(self, tool):
        """Regression: SubAgentRunner.run_all() truncates its own local
        copy of *subtasks* to MAX_SUB_AGENTS (10) when the caller passes
        more, but never mutates the caller's list or returns the
        truncated one. DelegateTaskTool.execute() still held the full,
        untruncated `subtasks` list from parse_subtasks() and zipped it
        against `results` (sized to the truncated count) with
        strict=True -- for any prompt with more than 10 numbered steps,
        that raised an unhandled ValueError and crashed tool execution
        instead of returning a ToolResult(success=False, ...).
        """
        from missy.agent.sub_agent import MAX_SUB_AGENTS

        runtime = MagicMock()
        runtime.run.return_value = "ok"
        prompt = "\n".join(f"{i}. step {i}" for i in range(1, MAX_SUB_AGENTS + 6))
        r = tool.execute(prompt=prompt, _runtime=runtime, _session_id="sess-1", _depth=0)
        assert r.success
        assert r.output.count("Step ") == MAX_SUB_AGENTS


class TestSchema:
    def test_schema_declares_prompt_required(self, tool):
        schema = tool.get_schema()
        assert schema["name"] == "delegate_task"
        assert "prompt" in schema["parameters"]["required"]
