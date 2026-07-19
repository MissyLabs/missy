"""Tests for the plan data model, validation, and ref resolution (F02)."""

from __future__ import annotations

import pytest

from missy.planning.plan import (
    Condition,
    Plan,
    PlanValidationError,
    ToolNode,
    resolve_args,
)


class TestConstruction:
    def test_from_dict_builds_nodes_and_conditions(self) -> None:
        plan = Plan.from_dict(
            {
                "goal": "demo",
                "nodes": [
                    {
                        "id": "a",
                        "tool": "upper",
                        "args": {"text": "hi"},
                        "postconditions": [{"kind": "success"}],
                    },
                    {
                        "id": "b",
                        "tool": "concat",
                        "args": {"a": "${a.output}", "b": "x"},
                        "depends_on": ["a"],
                    },
                ],
            }
        )
        assert plan.goal == "demo"
        assert plan.node("a").postconditions[0].kind == "success"
        assert plan.node("b").tool == "concat"

    def test_condition_from_dict_defaults(self) -> None:
        c = Condition.from_dict({"kind": "output_contains", "value": "x"})
        assert c.value == "x"
        assert c.node is None
        assert c.negate is False


class TestDependencies:
    def test_arg_references_imply_dependency(self) -> None:
        node = ToolNode(id="b", tool="t", args={"x": "${a.output}", "y": "lit"})
        assert node.referenced_nodes() == {"a"}
        assert "a" in node.all_dependencies()

    def test_explicit_and_implicit_deps_merge(self) -> None:
        node = ToolNode(id="c", tool="t", args={"x": "${a.output}"}, depends_on=["b"])
        assert node.all_dependencies() == {"a", "b"}

    def test_dependents_of(self) -> None:
        plan = Plan.from_dict(
            {
                "nodes": [
                    {"id": "a", "tool": "t"},
                    {"id": "b", "tool": "t", "depends_on": ["a"]},
                    {"id": "c", "tool": "t", "args": {"v": "${a.output}"}},
                ]
            }
        )
        assert set(plan.dependents_of("a")) == {"b", "c"}


class TestValidation:
    def test_duplicate_ids_rejected(self) -> None:
        with pytest.raises(PlanValidationError, match="duplicate"):
            Plan.from_dict({"nodes": [{"id": "a", "tool": "t"}, {"id": "a", "tool": "t"}]})

    def test_dangling_dependency_rejected(self) -> None:
        with pytest.raises(PlanValidationError, match="unknown node"):
            Plan.from_dict({"nodes": [{"id": "a", "tool": "t", "depends_on": ["z"]}]})

    def test_condition_targeting_unknown_node_rejected(self) -> None:
        with pytest.raises(PlanValidationError, match="unknown node"):
            Plan.from_dict(
                {
                    "nodes": [
                        {
                            "id": "a",
                            "tool": "t",
                            "preconditions": [{"kind": "success", "node": "ghost"}],
                        }
                    ]
                }
            )

    def test_cycle_rejected(self) -> None:
        with pytest.raises(PlanValidationError, match="cycle"):
            Plan.from_dict(
                {
                    "nodes": [
                        {"id": "a", "tool": "t", "depends_on": ["b"]},
                        {"id": "b", "tool": "t", "depends_on": ["a"]},
                    ]
                }
            )

    def test_self_cycle_rejected(self) -> None:
        with pytest.raises(PlanValidationError, match="cycle"):
            Plan.from_dict({"nodes": [{"id": "a", "tool": "t", "depends_on": ["a"]}]})


class TestTopologicalOrder:
    def test_order_respects_dependencies(self) -> None:
        plan = Plan.from_dict(
            {
                "nodes": [
                    {"id": "c", "tool": "t", "depends_on": ["a", "b"]},
                    {"id": "a", "tool": "t"},
                    {"id": "b", "tool": "t", "depends_on": ["a"]},
                ]
            }
        )
        order = plan.topological_order()
        assert order.index("a") < order.index("b") < order.index("c")

    def test_independent_nodes_all_present(self) -> None:
        plan = Plan.from_dict({"nodes": [{"id": x, "tool": "t"} for x in ("a", "b", "c")]})
        assert set(plan.topological_order()) == {"a", "b", "c"}


class TestResolveArgs:
    def test_full_reference_preserves_type(self) -> None:
        node = ToolNode(id="b", tool="t", args={"n": "${a.output}"})
        resolved = resolve_args(node, {"a": 42})
        assert resolved["n"] == 42  # int preserved, not "42"

    def test_embedded_reference_stringified(self) -> None:
        node = ToolNode(id="b", tool="t", args={"msg": "value is ${a.output}!"})
        resolved = resolve_args(node, {"a": 7})
        assert resolved["msg"] == "value is 7!"

    def test_dotted_dict_path(self) -> None:
        node = ToolNode(id="b", tool="t", args={"v": "${a.output.name}"})
        resolved = resolve_args(node, {"a": {"name": "missy"}})
        assert resolved["v"] == "missy"

    def test_node_and_node_output_equivalent(self) -> None:
        node = ToolNode(id="b", tool="t", args={"x": "${a}", "y": "${a.output}"})
        resolved = resolve_args(node, {"a": "same"})
        assert resolved["x"] == resolved["y"] == "same"

    def test_missing_reference_left_literal(self) -> None:
        node = ToolNode(id="b", tool="t", args={"v": "${missing.output}"})
        resolved = resolve_args(node, {})
        assert resolved["v"] == "${missing.output}"

    def test_non_string_args_untouched(self) -> None:
        node = ToolNode(id="b", tool="t", args={"count": 5, "flag": True})
        resolved = resolve_args(node, {})
        assert resolved == {"count": 5, "flag": True}
