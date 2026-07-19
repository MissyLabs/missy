"""Tests for the plan compiler (F02)."""

from __future__ import annotations

import pytest

from missy.planning.compiler import PlanCompiler
from missy.planning.plan import PlanValidationError


class TestCompileDict:
    def test_valid_plan_compiles(self, registry) -> None:
        plan = PlanCompiler(registry).compile_dict(
            {
                "goal": "g",
                "nodes": [
                    {"id": "a", "tool": "upper", "args": {"text": "x"}},
                    {
                        "id": "b",
                        "tool": "concat",
                        "args": {"a": "${a.output}", "b": "y"},
                        "depends_on": ["a"],
                    },
                ],
            }
        )
        assert plan.goal == "g"
        assert len(plan.nodes) == 2

    def test_unregistered_tool_rejected(self, registry) -> None:
        with pytest.raises(PlanValidationError, match="unregistered tool"):
            PlanCompiler(registry).compile_dict({"nodes": [{"id": "a", "tool": "no_such_tool"}]})

    def test_tool_verification_can_be_disabled(self, registry) -> None:
        # verify_tools=False allows a tool the registry doesn't know.
        plan = PlanCompiler(registry).compile_dict(
            {"nodes": [{"id": "a", "tool": "future_tool"}]}, verify_tools=False
        )
        assert plan.node("a").tool == "future_tool"

    def test_structural_errors_still_raised(self, registry) -> None:
        with pytest.raises(PlanValidationError, match="cycle"):
            PlanCompiler(registry).compile_dict(
                {
                    "nodes": [
                        {"id": "a", "tool": "upper", "depends_on": ["b"]},
                        {"id": "b", "tool": "upper", "depends_on": ["a"]},
                    ]
                }
            )

    def test_no_registry_skips_tool_verification(self) -> None:
        plan = PlanCompiler().compile_dict({"nodes": [{"id": "a", "tool": "anything"}]})
        assert plan.node("a").tool == "anything"


class TestCompileWithPlanner:
    def test_planner_output_is_compiled(self, registry) -> None:
        def planner(task, tools):
            assert "upper" in tools  # available tools were passed through
            return {"nodes": [{"id": "a", "tool": "upper", "args": {"text": task}}]}

        plan = PlanCompiler(registry).compile("shout", planner=planner)
        assert plan.goal == "shout"  # task became the goal
        assert plan.node("a").args["text"] == "shout"

    def test_planner_must_return_dict(self, registry) -> None:
        with pytest.raises(PlanValidationError, match="must return a dict"):
            PlanCompiler(registry).compile("t", planner=lambda task, tools: ["nope"])

    def test_planner_invalid_plan_rejected(self, registry) -> None:
        with pytest.raises(PlanValidationError, match="unregistered tool"):
            PlanCompiler(registry).compile(
                "t", planner=lambda task, tools: {"nodes": [{"id": "a", "tool": "ghost"}]}
            )


class TestSchema:
    def test_plan_schema_is_wellformed(self) -> None:
        schema = PlanCompiler.plan_schema()
        assert schema["type"] == "object"
        assert "nodes" in schema["properties"]
        node_schema = schema["properties"]["nodes"]["items"]
        assert set(node_schema["required"]) == {"id", "tool"}

    def test_available_tools_from_registry(self, registry) -> None:
        assert "upper" in PlanCompiler(registry).available_tools()

    def test_available_tools_empty_without_registry(self) -> None:
        assert PlanCompiler().available_tools() == []
