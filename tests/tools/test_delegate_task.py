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
        r = tool.execute(
            prompt="1. a", _runtime=MagicMock(), _session_id="s", _depth=MAX_SUB_AGENT_DEPTH
        )
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
        r = tool.execute(
            prompt="1. do a\n2. do b", _runtime=runtime, _session_id="sess-1", _depth=0
        )
        assert r.success
        assert "result a" in r.output
        assert "result b" in r.output

    def test_explicit_named_agents_are_generated_with_dependencies(self, tool):
        runtime = MagicMock()
        runtime.run.side_effect = ["research", "summary"]
        result = tool.execute(
            agents=[
                {"name": "Researcher", "task": "collect evidence"},
                {"name": "Writer", "task": "write summary", "depends_on": [0]},
            ],
            _runtime=runtime,
            _session_id="sess-1",
            _depth=0,
        )

        assert result.success
        assert "Agent Researcher (subagent-" in result.output
        assert "Agent Writer (subagent-" in result.output
        assert "Result of step 0: research" in runtime.run.call_args_list[1].args[0]

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


class TestAdvancedOrchestration:
    def test_diverse_strategy_runs_specialists_then_synthesizes(self, tool):
        runtime = MagicMock()

        def _run(prompt, session_id="", _delegation_depth=0):
            if "Lead Synthesizer" in prompt:
                assert all(
                    f"Result of step {index}: specialist finding" in prompt for index in range(4)
                )
                return "integrated final answer"
            return "specialist finding"

        runtime.run.side_effect = _run
        result = tool.execute(
            prompt="Develop an advanced launch strategy",
            strategy="diverse",
            _runtime=runtime,
            _session_id="sess-1",
            _depth=0,
        )

        assert result.success
        assert runtime.run.call_count == 5
        assert "strategy=diverse" in result.output
        assert "succeeded=5" in result.output
        assert "Final synthesis" in result.output
        assert "integrated final answer" in result.output

    def test_explicit_role_focus_criteria_and_tools_reach_agent(self, tool):
        runtime = MagicMock()
        runtime.run.return_value = "reviewed"

        result = tool.execute(
            agents=[
                {
                    "name": "Security Reviewer",
                    "role": "adversarial reviewer",
                    "focus": "authentication boundaries",
                    "success_criteria": "list exploitable gaps",
                    "tool_hints": ["file_read"],
                    "task": "review the design",
                }
            ],
            _runtime=runtime,
            _session_id="sess-1",
            _depth=0,
        )

        assert result.success
        prompt = runtime.run.call_args.args[0]
        assert "Security Reviewer — adversarial reviewer" in prompt
        assert "Focus: authentication boundaries" in prompt
        assert "Success criteria: list exploitable gaps" in prompt
        assert "Suggested tools: file_read" in prompt

    def test_explicit_agents_forward_independent_providers(self, tool):
        runtime = MagicMock()
        runtime.run.return_value = "ok"

        result = tool.execute(
            agents=[
                {"name": "Delegate", "task": "implement", "provider": "acpx"},
                {"name": "Reviewer", "task": "review", "provider": "openai"},
            ],
            _runtime=runtime,
            _session_id="sess-1",
            _depth=0,
        )

        assert result.success
        assert "[provider=acpx]" in result.output
        assert "[provider=openai]" in result.output
        providers = {call.kwargs["_provider"] for call in runtime.run.call_args_list}
        assert providers == {"acpx", "openai"}

    @pytest.mark.parametrize("provider", ["", "   ", 7, ["openai"]])
    def test_invalid_explicit_provider_is_rejected(self, tool, provider):
        runtime = MagicMock()

        result = tool.execute(
            agents=[{"task": "work", "provider": provider}],
            _runtime=runtime,
            _session_id="sess-1",
            _depth=0,
        )

        assert not result.success
        assert "provider must be a non-empty string" in result.error
        runtime.run.assert_not_called()

    @pytest.mark.parametrize(
        "dependencies, expected",
        [([3], "missing dependencies"), ([0], "depend on itself")],
    )
    def test_invalid_explicit_dependencies_are_rejected(self, tool, dependencies, expected):
        runtime = MagicMock()
        result = tool.execute(
            agents=[{"task": "unsafe plan", "depends_on": dependencies}],
            _runtime=runtime,
            _session_id="sess-1",
            _depth=0,
        )

        assert not result.success
        assert expected in result.error
        runtime.run.assert_not_called()

    def test_explicit_dependency_cycle_is_rejected(self, tool):
        runtime = MagicMock()
        result = tool.execute(
            agents=[
                {"task": "a", "depends_on": [1]},
                {"task": "b", "depends_on": [0]},
            ],
            _runtime=runtime,
            _session_id="sess-1",
            _depth=0,
        )

        assert not result.success
        assert "dependency cycle" in result.error
        runtime.run.assert_not_called()

    def test_skip_dependents_reports_partial_failure(self, tool):
        runtime = MagicMock()
        runtime.run.side_effect = RuntimeError("boom")
        result = tool.execute(
            agents=[
                {"task": "first"},
                {"task": "second", "depends_on": [0]},
            ],
            failure_policy="skip_dependents",
            _runtime=runtime,
            _session_id="sess-1",
            _depth=0,
        )

        assert not result.success
        assert runtime.run.call_count == 1
        assert "failed=1; skipped=1" in result.output
        assert "[SKIPPED]" in result.output


class TestSchema:
    def test_schema_allows_prompt_or_explicit_agents(self, tool):
        schema = tool.get_schema()
        assert schema["name"] == "delegate_task"
        assert "prompt" not in schema["parameters"]["required"]
        assert "agents" in schema["parameters"]["properties"]

    def test_description_encourages_batching_into_one_call(self, tool):
        """Regression test for the 5th tool-specific validation run's
        TDEEP-015 finding: 6 independent subtasks ran in a perfectly
        serial 3-second chain instead of exercising MAX_CONCURRENT's
        real 3-way parallelism, because the calling model issued 6
        separate single-subtask delegate_task calls rather than one call
        listing all 6. The tool description must explicitly steer the
        model toward batching independent subtasks into one call."""
        schema = tool.get_schema()
        assert "one call" in schema["description"].lower()
        assert "concurrent" in schema["parameters"]["properties"]["prompt"]["description"].lower()

    def test_schema_accepts_explicit_agent_definitions(self, tool):
        agents = tool.get_schema()["parameters"]["properties"]["agents"]
        assert agents["type"] == "array"
        assert agents["items"]["properties"]["depends_on"]["items"]["type"] == "integer"
        assert agents["items"]["properties"]["provider"]["type"] == "string"

    def test_schema_exposes_advanced_strategy_and_failure_control(self, tool):
        properties = tool.get_schema()["parameters"]["properties"]
        assert properties["strategy"]["enum"] == ["decompose", "diverse"]
        assert properties["failure_policy"]["enum"] == [
            "continue",
            "skip_dependents",
            "fail_fast",
        ]
        agent_properties = properties["agents"]["items"]["properties"]
        assert {"role", "focus", "success_criteria", "tool_hints"} <= set(agent_properties)
