"""Typed tool-call DAG data model for the planning kernel (F02).

A :class:`Plan` is a directed acyclic graph whose nodes are tool invocations
(:class:`ToolNode`) with declared pre/post-conditions and whose edges are
explicit data/ordering dependencies. This module owns the *structure*: node
definitions, condition specs, DAG validation, topological ordering, and
resolution of ``${node_id...}`` argument references against upstream results.
Execution lives in :mod:`missy.planning.executor`; condition evaluation in
:mod:`missy.planning.conditions`.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

# ``${node_id}`` or ``${node_id.output}`` / ``${node_id.output.key}`` — a
# reference to an upstream node's result, substituted at execution time.
_REF_RE = re.compile(r"\$\{([^}]+)\}")


class PlanValidationError(ValueError):
    """Raised when a :class:`Plan` is structurally invalid (cycle, bad ref…)."""


@dataclass
class Condition:
    """A typed pre- or post-condition assertion.

    Attributes:
        kind: One of ``success``, ``output_contains``, ``output_equals``,
            ``output_not_empty``, ``output_matches`` (regex),
            ``output_is_number``. Evaluated by
            :class:`~missy.planning.conditions.ConditionChecker`.
        value: Comparison operand (substring, expected value, regex…), if the
            kind needs one.
        node: Which node's result to assert on. ``None`` means "this node's
            own result" — the natural target for a post-condition. A
            pre-condition typically names an upstream dependency.
        negate: Invert the assertion.
        description: Human-readable label used in failure reasons.
    """

    kind: str
    value: Any = None
    node: str | None = None
    negate: bool = False
    description: str = ""

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Condition:
        return cls(
            kind=str(data["kind"]),
            value=data.get("value"),
            node=data.get("node"),
            negate=bool(data.get("negate", False)),
            description=str(data.get("description", "")),
        )


@dataclass
class ToolNode:
    """A single tool invocation in the plan DAG.

    Attributes:
        id: Unique node identifier within the plan.
        tool: Registered tool name to invoke.
        args: Keyword arguments for the tool. String values may embed
            ``${dep_id.output}`` references resolved from upstream results.
        depends_on: Node ids this node must run after. References inside
            ``args`` also imply a dependency, but listing it here is explicit
            and lets a node order after another without consuming its output.
        preconditions: Assertions checked *before* running; a failure blocks
            this node (and its dependents) without executing the tool.
        postconditions: Assertions checked *after* running against this
            node's own result; a failure marks the node failed.
    """

    id: str
    tool: str
    args: dict[str, Any] = field(default_factory=dict)
    depends_on: list[str] = field(default_factory=list)
    preconditions: list[Condition] = field(default_factory=list)
    postconditions: list[Condition] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ToolNode:
        return cls(
            id=str(data["id"]),
            tool=str(data["tool"]),
            args=dict(data.get("args", {})),
            depends_on=[str(d) for d in data.get("depends_on", [])],
            preconditions=[Condition.from_dict(c) for c in data.get("preconditions", [])],
            postconditions=[Condition.from_dict(c) for c in data.get("postconditions", [])],
        )

    def referenced_nodes(self) -> set[str]:
        """Return node ids referenced by ``${...}`` placeholders in ``args``."""
        refs: set[str] = set()
        for val in self.args.values():
            for m in _REF_RE.finditer(val) if isinstance(val, str) else []:
                refs.add(m.group(1).split(".", 1)[0])
        return refs

    def all_dependencies(self) -> set[str]:
        """Explicit ``depends_on`` plus any nodes referenced in ``args``."""
        return set(self.depends_on) | self.referenced_nodes()


@dataclass
class Plan:
    """A validated DAG of :class:`ToolNode`s."""

    nodes: list[ToolNode] = field(default_factory=list)
    goal: str = ""

    def __post_init__(self) -> None:
        self._by_id = {n.id: n for n in self.nodes}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Plan:
        plan = cls(
            nodes=[ToolNode.from_dict(n) for n in data.get("nodes", [])],
            goal=str(data.get("goal", "")),
        )
        plan.validate()
        return plan

    def node(self, node_id: str) -> ToolNode:
        return self._by_id[node_id]

    def dependents_of(self, node_id: str) -> list[str]:
        """Node ids that depend (directly) on ``node_id``."""
        return [n.id for n in self.nodes if node_id in n.all_dependencies()]

    def validate(self) -> None:
        """Check unique ids, resolvable dependencies, and acyclicity.

        Raises:
            PlanValidationError: on duplicate ids, dangling dependency/
                reference/condition target, or a cycle.
        """
        ids = [n.id for n in self.nodes]
        if len(ids) != len(set(ids)):
            raise PlanValidationError("duplicate node ids in plan")
        id_set = set(ids)
        for n in self.nodes:
            for dep in n.all_dependencies():
                if dep not in id_set:
                    raise PlanValidationError(f"node {n.id!r} depends on unknown node {dep!r}")
            for cond in list(n.preconditions) + list(n.postconditions):
                if cond.node is not None and cond.node not in id_set:
                    raise PlanValidationError(
                        f"node {n.id!r} condition targets unknown node {cond.node!r}"
                    )
        self.topological_order()  # raises on cycle

    def topological_order(self) -> list[str]:
        """Return node ids in a valid execution order (Kahn's algorithm).

        Raises:
            PlanValidationError: if the graph contains a cycle.
        """
        deps = {n.id: set(n.all_dependencies()) for n in self.nodes}
        order: list[str] = []
        ready = sorted([nid for nid, d in deps.items() if not d])
        remaining = dict(deps)
        while ready:
            nid = ready.pop(0)
            order.append(nid)
            remaining.pop(nid, None)
            newly: list[str] = []
            for other, d in remaining.items():
                if nid in d:
                    d.discard(nid)
                    if not d:
                        newly.append(other)
            ready.extend(sorted(newly))
            ready.sort()
        if remaining:
            raise PlanValidationError(f"plan contains a cycle among nodes {sorted(remaining)}")
        return order


def resolve_args(node: ToolNode, outputs: dict[str, Any]) -> dict[str, Any]:
    """Substitute ``${dep[.output[.key]]}`` references in ``node.args``.

    Args:
        node: The node whose args to resolve.
        outputs: Mapping of completed node id → that node's tool output.

    Returns:
        A new args dict with references replaced. A reference that is the
        entire string value is replaced with the referenced object (preserving
        type); an embedded reference is stringified in place. Unknown or
        not-yet-available references are left literal.
    """
    resolved: dict[str, Any] = {}
    for key, val in node.args.items():
        if not isinstance(val, str):
            resolved[key] = val
            continue
        full = _REF_RE.fullmatch(val)
        if full:
            obj = _lookup(full.group(1), outputs)
            resolved[key] = obj if obj is not _MISSING else val
        else:
            resolved[key] = _REF_RE.sub(
                lambda m: _stringify(_lookup(m.group(1), outputs), m.group(0)), val
            )
    return resolved


_MISSING = object()


def _lookup(ref: str, outputs: dict[str, Any]) -> Any:
    parts = ref.split(".")
    node_id = parts[0]
    if node_id not in outputs:
        return _MISSING
    cur: Any = outputs[node_id]
    # ``node`` and ``node.output`` both mean the output object itself.
    for part in parts[1:]:
        if part == "output":
            continue
        if isinstance(cur, dict) and part in cur:
            cur = cur[part]
        else:
            return _MISSING
    return cur


def _stringify(obj: Any, original: str) -> str:
    return original if obj is _MISSING else str(obj)
