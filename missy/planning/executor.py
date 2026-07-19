"""Speculative DAG executor for the planning kernel (F02).

:class:`PlanExecutor` runs a validated :class:`~missy.planning.plan.Plan` over
the real :class:`~missy.tools.registry.ToolRegistry`. It executes in
dependency order, running *independent* ready nodes concurrently (speculative
parallelism, capped like :class:`~missy.agent.sub_agent.SubAgentRunner`), and
verifies each node's post-conditions before its dependents become eligible —
turning tool use from an autoregressive guess into a checked, parallelizable
plan. A node whose pre-conditions fail, whose tool errors, or whose
post-conditions fail is marked accordingly and its downstream dependents are
skipped as unreachable rather than run on bad inputs.

Partial progress can be seeded via ``resume_state`` (a mapping of already-
completed node id → output), so a plan interrupted mid-run resumes without
re-executing finished nodes — the hook the checkpoint layer uses.
"""

from __future__ import annotations

import logging
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from missy.planning.conditions import ConditionChecker
from missy.planning.plan import Plan, ToolNode, resolve_args

logger = logging.getLogger(__name__)

# Matches SubAgentRunner's default fan-out cap.
MAX_CONCURRENT = 3


class NodeState(StrEnum):
    PENDING = "pending"
    DONE = "done"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class NodeExecution:
    """Outcome of a single node."""

    node_id: str
    state: NodeState
    output: Any = None
    success: bool = False
    reason: str = ""
    resolved_args: dict[str, Any] = field(default_factory=dict)


@dataclass
class PlanResult:
    """Aggregate outcome of executing a plan."""

    success: bool
    executions: dict[str, NodeExecution]
    order: list[str] = field(default_factory=list)

    @property
    def outputs(self) -> dict[str, Any]:
        return {nid: ex.output for nid, ex in self.executions.items() if ex.state is NodeState.DONE}

    def failed_nodes(self) -> list[str]:
        return [
            nid
            for nid, ex in self.executions.items()
            if ex.state in (NodeState.FAILED, NodeState.SKIPPED)
        ]


class PlanExecutor:
    """Executes a :class:`Plan` over a tool registry with verification."""

    def __init__(
        self,
        registry: Any,
        *,
        max_concurrent: int = MAX_CONCURRENT,
        checker: ConditionChecker | None = None,
    ) -> None:
        self._registry = registry
        self._max_concurrent = max(1, max_concurrent)
        self._checker = checker or ConditionChecker()

    def execute(
        self,
        plan: Plan,
        *,
        session_id: str = "",
        task_id: str = "",
        resume_state: dict[str, Any] | None = None,
    ) -> PlanResult:
        """Run ``plan`` and return a :class:`PlanResult`.

        Args:
            plan: A validated plan (``validate()`` is re-run defensively).
            session_id: Forwarded to ``registry.execute`` for audit/policy.
            task_id: Forwarded to ``registry.execute``.
            resume_state: Optional ``{node_id: output}`` for nodes already
                completed in a prior run; these are treated as ``DONE`` and
                not re-executed.
        """
        plan.validate()
        order = plan.topological_order()
        executions: dict[str, NodeExecution] = {}
        outputs: dict[str, Any] = {}
        lock = threading.Lock()

        # Seed resumed nodes as completed.
        for nid, out in (resume_state or {}).items():
            if nid in plan._by_id:
                executions[nid] = NodeExecution(
                    node_id=nid,
                    state=NodeState.DONE,
                    output=out,
                    success=True,
                    reason="resumed",
                )
                outputs[nid] = out

        def dep_state(nid: str) -> NodeState | None:
            ex = executions.get(nid)
            return ex.state if ex else None

        def ready_nodes() -> list[ToolNode]:
            ready: list[ToolNode] = []
            for n in plan.nodes:
                if n.id in executions:
                    continue
                dep_states = [dep_state(d) for d in n.all_dependencies()]
                if any(s is None for s in dep_states):
                    continue  # a dependency hasn't run yet
                if all(s is NodeState.DONE for s in dep_states):
                    ready.append(n)
                else:
                    # An upstream node failed/was skipped → this is unreachable.
                    executions[n.id] = NodeExecution(
                        node_id=n.id,
                        state=NodeState.SKIPPED,
                        reason="upstream dependency did not succeed",
                    )
            return ready

        # Rounds of speculative parallel execution until nothing is ready.
        while True:
            batch = ready_nodes()
            if not batch:
                break
            with ThreadPoolExecutor(max_workers=self._max_concurrent) as pool:
                futures = {
                    pool.submit(
                        self._run_node,
                        n,
                        outputs,
                        executions,
                        lock,
                        session_id,
                        task_id,
                    ): n
                    for n in batch
                }
                for fut in futures:
                    ex = fut.result()
                    with lock:
                        executions[ex.node_id] = ex
                        if ex.state is NodeState.DONE:
                            outputs[ex.node_id] = ex.output

        # Any node never reached (e.g. stranded behind a skip) is skipped.
        for n in plan.nodes:
            executions.setdefault(
                n.id,
                NodeExecution(node_id=n.id, state=NodeState.SKIPPED, reason="not reached"),
            )

        success = all(ex.state is NodeState.DONE for ex in executions.values())
        return PlanResult(success=success, executions=executions, order=order)

    def _run_node(
        self,
        node: ToolNode,
        outputs: dict[str, Any],
        executions: dict[str, NodeExecution],
        lock: threading.Lock,
        session_id: str,
        task_id: str,
    ) -> NodeExecution:
        # Pre-conditions: checked against a snapshot of results-so-far.
        with lock:
            snapshot = dict(outputs)
            result_snapshot = dict(executions)
        ok, reason = self._checker.check_all(
            node.preconditions, self_result=None, results=result_snapshot
        )
        if not ok:
            return NodeExecution(
                node_id=node.id,
                state=NodeState.SKIPPED,
                reason=f"precondition failed: {reason}",
            )

        resolved = resolve_args(node, snapshot)
        try:
            result = self._registry.execute(
                node.tool, session_id=session_id, task_id=task_id, **resolved
            )
        except Exception as exc:  # a raising tool must not crash the whole plan
            return NodeExecution(
                node_id=node.id,
                state=NodeState.FAILED,
                reason=f"tool raised: {exc}",
                resolved_args=resolved,
            )

        if not getattr(result, "success", False):
            return NodeExecution(
                node_id=node.id,
                state=NodeState.FAILED,
                output=getattr(result, "output", None),
                reason=f"tool failed: {getattr(result, 'error', 'unknown error')}",
                resolved_args=resolved,
            )

        # Post-conditions: verify the node's own result before dependents run.
        ok, reason = self._checker.check_all(node.postconditions, self_result=result)
        if not ok:
            return NodeExecution(
                node_id=node.id,
                state=NodeState.FAILED,
                output=getattr(result, "output", None),
                reason=f"postcondition failed: {reason}",
                resolved_args=resolved,
            )

        return NodeExecution(
            node_id=node.id,
            state=NodeState.DONE,
            output=getattr(result, "output", None),
            success=True,
            reason="ok",
            resolved_args=resolved,
        )
