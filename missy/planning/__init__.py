"""Neuro-symbolic planning kernel (F02): typed tool-call DAG + verified execution.

A symbolic planner + verifier that sits above the LLM. A task is compiled into
a **typed tool-call DAG** — nodes are tool invocations with declared pre/post
conditions, edges are data/ordering dependencies — which is then executed with
speculative parallelism (independent branches run concurrently), each node's
post-condition checked before its dependents run. This turns tool use from an
autoregressive guess into a checked, parallelizable, resumable plan.

Public surface::

    from missy.planning import PlanCompiler, PlanExecutor, Plan

    plan = PlanCompiler(registry).compile_dict({
        "goal": "fetch then summarize",
        "nodes": [
            {"id": "a", "tool": "fetch", "args": {"url": "..."}},
            {"id": "b", "tool": "summarize", "args": {"text": "${a.output}"},
             "depends_on": ["a"],
             "postconditions": [{"kind": "output_not_empty"}]},
        ],
    })
    result = PlanExecutor(registry).execute(plan)
"""

from __future__ import annotations

from missy.planning.compiler import PlanCompiler, Planner
from missy.planning.conditions import ConditionChecker
from missy.planning.executor import (
    MAX_CONCURRENT,
    NodeExecution,
    NodeState,
    PlanExecutor,
    PlanResult,
)
from missy.planning.plan import (
    Condition,
    Plan,
    PlanValidationError,
    ToolNode,
    resolve_args,
)

__all__ = [
    "MAX_CONCURRENT",
    "Condition",
    "ConditionChecker",
    "NodeExecution",
    "NodeState",
    "Plan",
    "PlanCompiler",
    "PlanExecutor",
    "PlanResult",
    "PlanValidationError",
    "Planner",
    "ToolNode",
    "resolve_args",
]
