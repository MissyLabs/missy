"""Tests for the speculative DAG executor (F02)."""

from __future__ import annotations

import time

from missy.planning.executor import NodeState, PlanExecutor
from missy.planning.plan import Plan


class TestDataFlowAndOrdering:
    def test_output_flows_between_nodes(self, registry) -> None:
        plan = Plan.from_dict(
            {
                "nodes": [
                    {"id": "x", "tool": "upper", "args": {"text": "hello"}},
                    {"id": "y", "tool": "upper", "args": {"text": "world"}},
                    {
                        "id": "z",
                        "tool": "concat",
                        "args": {"a": "${x.output}", "b": "${y.output}"},
                        "depends_on": ["x", "y"],
                    },
                ]
            }
        )
        result = PlanExecutor(registry).execute(plan)
        assert result.success
        assert result.outputs["z"] == "HELLO|WORLD"

    def test_execution_respects_dependency_order(self, registry) -> None:
        plan = Plan.from_dict(
            {
                "nodes": [
                    {"id": "a", "tool": "echo", "args": {"value": "1"}},
                    {
                        "id": "b",
                        "tool": "echo",
                        "args": {"value": "${a.output}"},
                        "depends_on": ["a"],
                    },
                ]
            }
        )
        PlanExecutor(registry).execute(plan)
        names = [c[0] for c in registry.calls]
        # 'a' before 'b' — and b received a's resolved output.
        assert names == ["echo", "echo"]
        assert registry.calls[1][1]["value"] == "1"


class TestSpeculativeParallelism:
    def test_independent_nodes_run_concurrently(self, registry) -> None:
        plan = Plan.from_dict(
            {"nodes": [{"id": x, "tool": "slow", "args": {"v": x}} for x in "abc"]}
        )
        t0 = time.time()
        result = PlanExecutor(registry, max_concurrent=3).execute(plan)
        elapsed = time.time() - t0
        assert result.success
        # 3 x 50ms nodes concurrently should be well under the 150ms serial time.
        assert elapsed < 0.13
        assert registry.max_active >= 2  # genuinely overlapped

    def test_max_concurrent_is_respected(self, registry) -> None:
        plan = Plan.from_dict(
            {"nodes": [{"id": str(i), "tool": "slow", "args": {"v": i}} for i in range(6)]}
        )
        PlanExecutor(registry, max_concurrent=2).execute(plan)
        assert registry.max_active <= 2


class TestFailureHandling:
    def test_failed_tool_marks_node_failed(self, registry) -> None:
        plan = Plan.from_dict({"nodes": [{"id": "a", "tool": "fail"}]})
        result = PlanExecutor(registry).execute(plan)
        assert not result.success
        assert result.executions["a"].state is NodeState.FAILED
        assert "tool failed" in result.executions["a"].reason

    def test_raising_tool_is_contained(self, registry) -> None:
        plan = Plan.from_dict({"nodes": [{"id": "a", "tool": "raise"}]})
        result = PlanExecutor(registry).execute(plan)
        assert result.executions["a"].state is NodeState.FAILED
        assert "raised" in result.executions["a"].reason

    def test_dependents_of_failure_are_skipped(self, registry) -> None:
        plan = Plan.from_dict(
            {
                "nodes": [
                    {"id": "a", "tool": "fail"},
                    {
                        "id": "b",
                        "tool": "upper",
                        "args": {"text": "${a.output}"},
                        "depends_on": ["a"],
                    },
                    {
                        "id": "c",
                        "tool": "upper",
                        "args": {"text": "${b.output}"},
                        "depends_on": ["b"],
                    },
                ]
            }
        )
        result = PlanExecutor(registry).execute(plan)
        assert result.executions["a"].state is NodeState.FAILED
        assert result.executions["b"].state is NodeState.SKIPPED
        assert result.executions["c"].state is NodeState.SKIPPED
        assert set(result.failed_nodes()) == {"a", "b", "c"}

    def test_independent_branch_survives_sibling_failure(self, registry) -> None:
        plan = Plan.from_dict(
            {
                "nodes": [
                    {"id": "bad", "tool": "fail"},
                    {"id": "good", "tool": "upper", "args": {"text": "ok"}},
                ]
            }
        )
        result = PlanExecutor(registry).execute(plan)
        assert not result.success  # overall not fully successful
        assert result.executions["good"].state is NodeState.DONE
        assert result.executions["good"].output == "OK"


class TestConditions:
    def test_postcondition_violation_fails_node(self, registry) -> None:
        plan = Plan.from_dict(
            {
                "nodes": [
                    {
                        "id": "a",
                        "tool": "upper",
                        "args": {"text": "hi"},
                        "postconditions": [{"kind": "output_equals", "value": "WRONG"}],
                    }
                ]
            }
        )
        result = PlanExecutor(registry).execute(plan)
        assert result.executions["a"].state is NodeState.FAILED
        assert "postcondition" in result.executions["a"].reason

    def test_satisfied_postcondition_passes(self, registry) -> None:
        plan = Plan.from_dict(
            {
                "nodes": [
                    {
                        "id": "a",
                        "tool": "upper",
                        "args": {"text": "hi"},
                        "postconditions": [{"kind": "output_equals", "value": "HI"}],
                    }
                ]
            }
        )
        assert PlanExecutor(registry).execute(plan).success

    def test_failed_precondition_skips_without_running(self, registry) -> None:
        # 'b' requires a's output to contain "ZZZ" (it won't), so b never runs.
        plan = Plan.from_dict(
            {
                "nodes": [
                    {"id": "a", "tool": "upper", "args": {"text": "hi"}},
                    {
                        "id": "b",
                        "tool": "upper",
                        "args": {"text": "x"},
                        "depends_on": ["a"],
                        "preconditions": [{"kind": "output_contains", "value": "ZZZ", "node": "a"}],
                    },
                ]
            }
        )
        result = PlanExecutor(registry).execute(plan)
        assert result.executions["a"].state is NodeState.DONE
        assert result.executions["b"].state is NodeState.SKIPPED
        assert "precondition" in result.executions["b"].reason
        # b's tool was never invoked
        assert [c for c in registry.calls if c[1].get("text") == "x"] == []


class TestResume:
    def test_resume_state_skips_completed_nodes(self, registry) -> None:
        plan = Plan.from_dict(
            {
                "nodes": [
                    {"id": "x", "tool": "upper", "args": {"text": "hello"}},
                    {"id": "y", "tool": "upper", "args": {"text": "world"}},
                    {
                        "id": "z",
                        "tool": "concat",
                        "args": {"a": "${x.output}", "b": "${y.output}"},
                        "depends_on": ["x", "y"],
                    },
                ]
            }
        )
        result = PlanExecutor(registry).execute(plan, resume_state={"x": "HELLO", "y": "WORLD"})
        assert result.success
        assert result.outputs["z"] == "HELLO|WORLD"
        # only the concat ran; the two 'upper' nodes were resumed.
        assert [c[0] for c in registry.calls] == ["concat"]
        assert result.executions["x"].reason == "resumed"

    def test_result_outputs_and_order_exposed(self, registry) -> None:
        plan = Plan.from_dict({"nodes": [{"id": "a", "tool": "upper", "args": {"text": "z"}}]})
        result = PlanExecutor(registry).execute(plan)
        assert result.order == ["a"]
        assert result.outputs == {"a": "Z"}
