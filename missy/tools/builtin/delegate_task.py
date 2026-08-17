"""Built-in tool: decompose a compound task into sub-agent delegated steps.

SR-4.2: dispatches through :class:`~missy.agent.sub_agent.SubAgentRunner`,
which reuses the *calling* :class:`~missy.agent.runtime.AgentRuntime`
instance and session -- sub-tasks are bound by the exact same policy,
capability_mode, and per-session budget cap as the call that invoked
``delegate_task``, not a separately-authorized context. Recursion is
bounded by :data:`~missy.agent.sub_agent.MAX_SUB_AGENT_DEPTH`.
"""

from __future__ import annotations

import logging

from missy.tools.base import BaseTool, ToolPermissions, ToolResult

logger = logging.getLogger(__name__)


class DelegateTaskTool(BaseTool):
    """Decompose a compound task into sub-agent steps and run them."""

    name = "delegate_task"
    description = (
        "Decompose a compound, multi-step task into sub-agent calls and run "
        "them, respecting any sequential dependencies you describe (e.g. "
        "numbered steps or 'first...then...') and running independent steps "
        "concurrently (up to 3 at a time). Each step runs as a full agent "
        "turn with the same tools and permissions available to you right "
        "now -- it does not grant any additional capability. Use this for "
        "genuinely decomposable work (e.g. 'research X, then Y, then "
        "combine results'), not for a single atomic action. IMPORTANT: "
        "list ALL independent subtasks in ONE call's prompt (as a numbered "
        "list) -- that is what actually gets them run concurrently. Calling "
        "this tool once per subtask instead forces them to run one at a "
        "time, since each tool call in your own turn is handled before the "
        "next begins."
    )
    permissions = ToolPermissions()
    parameters = {
        "prompt": {
            "type": "string",
            "description": (
                "The compound task to decompose, as a numbered list or a "
                "sequence using connectives like 'then'/'first'/'finally'. "
                "Put every independent subtask in this single string (e.g. "
                "'1. ... 2. ... 3. ...') rather than making separate calls "
                "to this tool per subtask, so independent ones actually run "
                "concurrently instead of serially."
            ),
            "required": False,
        },
        "agents": {
            "type": "array",
            "description": (
                "Optional explicit agent definitions. Each item has a name, task, "
                "and optional depends_on list of zero-based agent indexes. Use this "
                "when distinct roles or dependencies matter; otherwise prompt is "
                "automatically decomposed. At most 10 agents are created."
            ),
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "task": {"type": "string"},
                    "depends_on": {"type": "array", "items": {"type": "integer"}},
                },
                "required": ["task"],
            },
            "required": False,
        },
    }

    def execute(
        self,
        *,
        prompt: str = "",
        _runtime: object | None = None,
        _session_id: str = "",
        _depth: int = 0,
        _parent_agent_id: str = "",
        _parent_task_id: str = "",
        agents: list[dict] | None = None,
        **_kwargs,
    ) -> ToolResult:
        from missy.agent.sub_agent import (
            MAX_SUB_AGENT_DEPTH,
            MAX_SUB_AGENTS,
            SubAgentRunner,
            parse_subtasks,
        )

        if _runtime is None:
            return ToolResult(
                success=False,
                output="",
                error="delegate_task requires runtime context and cannot be called directly.",
            )

        if _depth >= MAX_SUB_AGENT_DEPTH:
            return ToolResult(
                success=False,
                output="",
                error=(
                    f"Delegation depth limit ({MAX_SUB_AGENT_DEPTH}) reached; "
                    "a sub-agent cannot delegate further. Complete this step "
                    "directly instead."
                ),
            )

        if (not prompt or not prompt.strip()) and not agents:
            return ToolResult(
                success=False,
                output="",
                error="prompt is required unless explicit agents are provided.",
            )

        if agents:
            subtasks = []
            for index, spec in enumerate(agents[:MAX_SUB_AGENTS]):
                if not isinstance(spec, dict) or not str(spec.get("task") or "").strip():
                    return ToolResult(
                        success=False,
                        output="",
                        error=f"agents[{index}].task must be a non-empty string.",
                    )
                raw_dependencies = spec.get("depends_on") or []
                if not isinstance(raw_dependencies, list) or any(
                    isinstance(dep, bool) or not isinstance(dep, int) for dep in raw_dependencies
                ):
                    return ToolResult(
                        success=False,
                        output="",
                        error=f"agents[{index}].depends_on must be a list of integer indexes.",
                    )
                valid_dependencies = sorted(
                    {dep for dep in raw_dependencies if 0 <= dep < len(agents) and dep != index}
                )
                from missy.agent.sub_agent import SubTask

                subtasks.append(
                    SubTask(
                        id=index,
                        description=str(spec["task"]).strip(),
                        name=str(spec.get("name") or "").strip()[:80],
                        depends_on=valid_dependencies,
                    )
                )
        else:
            subtasks = parse_subtasks(prompt)
        # SubAgentRunner.run_all() truncates its own local copy to
        # MAX_SUB_AGENTS when the caller passes more subtasks than that --
        # it never mutates *this* list, so `results` ends up shorter than
        # `subtasks` for any prompt with more than MAX_SUB_AGENTS steps.
        # Truncate the same way here so the two stay the same length for
        # the zip(..., strict=True) below, instead of that raising an
        # unhandled ValueError and crashing tool execution.
        if len(subtasks) > MAX_SUB_AGENTS:
            subtasks = subtasks[:MAX_SUB_AGENTS]
        logger.info(
            "Agent %s task %s generated %d sub-agent(s) at depth %d",
            _parent_agent_id or "unknown",
            _parent_task_id or "unknown",
            len(subtasks),
            _depth + 1,
        )
        runner = SubAgentRunner(
            runtime=_runtime,
            session_id=_session_id,
            depth=_depth + 1,
            parent_agent_id=_parent_agent_id,
            parent_task_id=_parent_task_id,
        )
        results = runner.run_all(subtasks)

        lines = [
            f"Step {t.id} — Agent {t.name} ({t.agent_id})"
            f"{' [FAILED]' if t.error else ''}: {t.description}\n{r}"
            for t, r in zip(subtasks, results, strict=True)
        ]
        any_failed = any(t.error for t in subtasks)
        return ToolResult(
            success=not any_failed,
            output="\n\n".join(lines),
            error="One or more sub-agent steps failed; see output above." if any_failed else None,
        )
