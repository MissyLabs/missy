"""Tests for typed condition evaluation (F02)."""

from __future__ import annotations

import pytest

from missy.planning.conditions import ConditionChecker
from missy.planning.plan import Condition
from missy.tools.base import ToolResult


@pytest.fixture
def checker() -> ConditionChecker:
    return ConditionChecker()


def _ok(output):
    return ToolResult(success=True, output=output)


class TestKinds:
    def test_success(self, checker) -> None:
        assert checker.check(Condition(kind="success"), self_result=_ok("x"))[0]
        bad = ToolResult(success=False, output=None, error="e")
        assert not checker.check(Condition(kind="success"), self_result=bad)[0]

    def test_output_not_empty(self, checker) -> None:
        assert checker.check(Condition(kind="output_not_empty"), self_result=_ok("x"))[0]
        for empty in ("", [], {}, None):
            ok, _ = checker.check(Condition(kind="output_not_empty"), self_result=_ok(empty))
            assert not ok

    def test_output_equals(self, checker) -> None:
        c = Condition(kind="output_equals", value=42)
        assert checker.check(c, self_result=_ok(42))[0]
        assert not checker.check(c, self_result=_ok(43))[0]

    def test_output_contains(self, checker) -> None:
        c = Condition(kind="output_contains", value="ell")
        assert checker.check(c, self_result=_ok("hello"))[0]
        assert not checker.check(c, self_result=_ok("world"))[0]

    def test_output_contains_coerces_non_containers(self, checker) -> None:
        # numeric output: falls back to string containment
        c = Condition(kind="output_contains", value="23")
        assert checker.check(c, self_result=_ok(1234))[0]

    def test_output_matches_regex(self, checker) -> None:
        c = Condition(kind="output_matches", value=r"\d{3}-\d{4}")
        assert checker.check(c, self_result=_ok("call 555-1234 now"))[0]
        assert not checker.check(c, self_result=_ok("no number"))[0]

    def test_output_is_number(self, checker) -> None:
        assert checker.check(Condition(kind="output_is_number"), self_result=_ok(3.14))[0]
        assert not checker.check(Condition(kind="output_is_number"), self_result=_ok("3"))[0]
        # bool is not treated as a number
        assert not checker.check(Condition(kind="output_is_number"), self_result=_ok(True))[0]

    def test_unknown_kind_raises(self, checker) -> None:
        with pytest.raises(ValueError, match="unknown condition kind"):
            checker.check(Condition(kind="frobnicate"), self_result=_ok("x"))


class TestNegateAndTargets:
    def test_negate_inverts(self, checker) -> None:
        c = Condition(kind="output_equals", value="x", negate=True)
        assert checker.check(c, self_result=_ok("y"))[0]
        assert not checker.check(c, self_result=_ok("x"))[0]

    def test_targets_named_node(self, checker) -> None:
        c = Condition(kind="output_equals", value="up", node="a")
        results = {"a": _ok("up")}
        assert checker.check(c, results=results)[0]

    def test_missing_target_result_fails_with_reason(self, checker) -> None:
        c = Condition(kind="success", node="a")
        ok, reason = checker.check(c, results={})
        assert not ok
        assert "no result available" in reason


class TestCheckAll:
    def test_returns_first_failure(self, checker) -> None:
        conds = [
            Condition(kind="success"),
            Condition(kind="output_equals", value="nope", description="second"),
            Condition(kind="output_not_empty"),
        ]
        ok, reason = checker.check_all(conds, self_result=_ok("actual"))
        assert not ok
        assert "second" in reason

    def test_all_pass(self, checker) -> None:
        conds = [Condition(kind="success"), Condition(kind="output_not_empty")]
        ok, reason = checker.check_all(conds, self_result=_ok("x"))
        assert ok and reason == "ok"

    def test_empty_list_passes(self, checker) -> None:
        assert checker.check_all([], self_result=_ok("x")) == (True, "ok")
