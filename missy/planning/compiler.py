"""Plan compilation for the planning kernel (F02).

:class:`PlanCompiler` turns a task into a validated :class:`Plan`. The DAG
itself is proposed in a constrained JSON schema — either by an injected
``planner`` callable (an LLM in production, ideally wrapped by F09's
``StructuredOutputRunner`` so the schema is enforced with retry) or supplied
directly as a dict (the deterministic path used by callers that already know
the graph). Either way the compiler is the single place that validates the
proposal against the available tool set and the DAG invariants before it can
be executed, so a malformed or hallucinated plan is rejected up front rather
than partway through execution.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from missy.planning.plan import Plan, PlanValidationError

# A planner takes (task, tool_names) and returns a plan dict in the schema
# below. In production this is an LLM call; in tests it's a lambda.
Planner = Callable[[str, list[str]], dict[str, Any]]


class PlanCompiler:
    """Compiles and validates tool-call DAGs."""

    def __init__(self, registry: Any = None) -> None:
        self._registry = registry

    def available_tools(self) -> list[str]:
        if self._registry is None:
            return []
        # ToolRegistry exposes list_tools(); fall back to _tools keys.
        if hasattr(self._registry, "list_tools"):
            return list(self._registry.list_tools())
        return list(getattr(self._registry, "_tools", {}).keys())

    def compile_dict(self, data: dict[str, Any], *, verify_tools: bool = True) -> Plan:
        """Build and validate a :class:`Plan` from a plan dict.

        Args:
            data: A plan mapping (``{"goal": ..., "nodes": [...]}``).
            verify_tools: When ``True`` and a registry is set, every node's
                ``tool`` must be a registered tool name.

        Raises:
            PlanValidationError: on structural problems or an unknown tool.
        """
        plan = Plan.from_dict(data)  # runs structural validation
        if verify_tools and self._registry is not None:
            known = set(self.available_tools())
            for n in plan.nodes:
                if n.tool not in known:
                    raise PlanValidationError(
                        f"node {n.id!r} references unregistered tool {n.tool!r}"
                    )
        return plan

    def compile(self, task: str, *, planner: Planner, verify_tools: bool = True) -> Plan:
        """Ask ``planner`` for a DAG for ``task`` and compile it.

        Raises:
            PlanValidationError: if the planner's output is not a valid plan.
        """
        proposal = planner(task, self.available_tools())
        if not isinstance(proposal, dict):
            raise PlanValidationError(f"planner must return a dict, got {type(proposal).__name__}")
        proposal.setdefault("goal", task)
        return self.compile_dict(proposal, verify_tools=verify_tools)

    @staticmethod
    def plan_schema() -> dict[str, Any]:
        """Return the JSON schema a structured-output planner should target."""
        condition = {
            "type": "object",
            "properties": {
                "kind": {
                    "type": "string",
                    "enum": [
                        "success",
                        "output_contains",
                        "output_equals",
                        "output_not_empty",
                        "output_matches",
                        "output_is_number",
                    ],
                },
                "value": {},
                "node": {"type": ["string", "null"]},
                "negate": {"type": "boolean"},
                "description": {"type": "string"},
            },
            "required": ["kind"],
        }
        node = {
            "type": "object",
            "properties": {
                "id": {"type": "string"},
                "tool": {"type": "string"},
                "args": {"type": "object"},
                "depends_on": {"type": "array", "items": {"type": "string"}},
                "preconditions": {"type": "array", "items": condition},
                "postconditions": {"type": "array", "items": condition},
            },
            "required": ["id", "tool"],
        }
        return {
            "type": "object",
            "properties": {
                "goal": {"type": "string"},
                "nodes": {"type": "array", "items": node},
            },
            "required": ["nodes"],
        }
