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
        "concurrently. Each step runs as a full agent turn with the same "
        "tools and permissions available to you right now -- it does not "
        "grant any additional capability. Use this for genuinely "
        "decomposable work (e.g. 'research X, then Y, then combine "
        "results'), not for a single atomic action."
    )
    permissions = ToolPermissions()
    parameters = {
        "prompt": {
            "type": "string",
            "description": (
                "The compound task to decompose, as a numbered list or a "
                "sequence using connectives like 'then'/'first'/'finally'."
            ),
            "required": True,
        },
    }

    def execute(
        self,
        *,
        prompt: str,
        _runtime: object | None = None,
        _session_id: str = "",
        _depth: int = 0,
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

        if not prompt or not prompt.strip():
            return ToolResult(success=False, output="", error="prompt is required.")

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
        runner = SubAgentRunner(runtime=_runtime, session_id=_session_id, depth=_depth + 1)
        results = runner.run_all(subtasks)

        lines = [
            f"Step {t.id}{' [FAILED]' if t.error else ''}: {t.description}\n{r}"
            for t, r in zip(subtasks, results, strict=True)
        ]
        any_failed = any(t.error for t in subtasks)
        return ToolResult(
            success=not any_failed,
            output="\n\n".join(lines),
            error="One or more sub-agent steps failed; see output above." if any_failed else None,
        )
