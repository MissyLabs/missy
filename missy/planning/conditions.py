"""Typed condition evaluation for the planning kernel (F02).

A :class:`~missy.planning.plan.Condition` is a small declarative assertion
about a node's result — the *verifier* half of the neuro-symbolic loop. The
executor checks a node's pre-conditions before running it and its
post-conditions after, using :class:`ConditionChecker`. Every check returns a
``(ok, reason)`` pair so failures carry an explanation into the audit/result.
"""

from __future__ import annotations

import re
from typing import Any

from missy.planning.plan import Condition


class ConditionChecker:
    """Evaluates :class:`Condition`s against tool results.

    A "result" here is a :class:`~missy.tools.base.ToolResult`-like object with
    ``success`` and ``output`` attributes (duck-typed so tests can pass simple
    stand-ins).
    """

    def check(
        self,
        cond: Condition,
        *,
        self_result: Any = None,
        results: dict[str, Any] | None = None,
    ) -> tuple[bool, str]:
        """Return ``(ok, reason)`` for one condition.

        Args:
            cond: The condition to evaluate.
            self_result: The owning node's own result (used when
                ``cond.node`` is ``None``).
            results: Mapping of node id → result, for conditions that target a
                named upstream node.
        """
        results = results or {}
        target = self_result if cond.node is None else results.get(cond.node)
        if target is None:
            return (False, self._label(cond, f"no result available for node {cond.node!r}"))

        ok = self._evaluate(cond, target)
        if cond.negate:
            ok = not ok
        if ok:
            return (True, self._label(cond, "ok"))
        return (False, self._label(cond, "assertion failed"))

    def check_all(
        self,
        conditions: list[Condition],
        *,
        self_result: Any = None,
        results: dict[str, Any] | None = None,
    ) -> tuple[bool, str]:
        """Check a list; return the first failure, else ``(True, "ok")``."""
        for cond in conditions:
            ok, reason = self.check(cond, self_result=self_result, results=results)
            if not ok:
                return (False, reason)
        return (True, "ok")

    # -- individual kinds -------------------------------------------------
    def _evaluate(self, cond: Condition, target: Any) -> bool:
        output = getattr(target, "output", target)
        success = bool(getattr(target, "success", True))
        kind = cond.kind
        if kind == "success":
            return success
        if kind == "output_not_empty":
            return output is not None and output != "" and output != [] and output != {}
        if kind == "output_equals":
            return output == cond.value
        if kind == "output_contains":
            try:
                return cond.value in output
            except TypeError:
                return str(cond.value) in str(output)
        if kind == "output_matches":
            return re.search(str(cond.value), str(output)) is not None
        if kind == "output_is_number":
            return isinstance(output, (int, float)) and not isinstance(output, bool)
        raise ValueError(f"unknown condition kind {kind!r}")

    @staticmethod
    def _label(cond: Condition, suffix: str) -> str:
        base = cond.description or f"{cond.kind}"
        if cond.node:
            base = f"{base} @{cond.node}"
        if cond.negate:
            base = f"not({base})"
        return f"{base}: {suffix}"
