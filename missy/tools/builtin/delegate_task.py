"""Built-in tool: orchestrate decomposed or diversified sub-agent work.

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
        "next begins. For advanced, ambiguous, or high-impact requests, set "
        "strategy='diverse' to run complementary requirements, architecture, "
        "risk, and verification specialists in parallel and have a lead agent "
        "synthesize their findings into one final answer."
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
                "Optional explicit agent definitions. Each item has a task and may "
                "define a name, role, focus, success criteria, provider registry key, tool hints, and "
                "depends_on list of zero-based agent indexes. Use complementary "
                "roles instead of giving every agent the same generic assignment. "
                "At most 10 agents are allowed."
            ),
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "task": {"type": "string"},
                    "role": {"type": "string"},
                    "focus": {"type": "string"},
                    "success_criteria": {"type": "string"},
                    "provider": {
                        "type": "string",
                        "description": (
                            "Optional configured provider registry key for this agent "
                            "(for example 'acpx' or 'openai'). Omit to inherit the "
                            "parent runtime's provider."
                        ),
                    },
                    "tool_hints": {"type": "array", "items": {"type": "string"}},
                    "depends_on": {"type": "array", "items": {"type": "integer"}},
                },
                "required": ["task"],
            },
            "required": False,
        },
        "strategy": {
            "type": "string",
            "enum": ["decompose", "diverse"],
            "description": (
                "decompose parses concrete workflow steps; diverse assigns the same "
                "advanced request to complementary specialists and optionally adds a "
                "final synthesis agent. Defaults to decompose."
            ),
            "required": False,
        },
        "include_synthesis": {
            "type": "boolean",
            "description": (
                "When strategy=diverse, add a lead synthesis agent that depends on all "
                "specialists. Defaults to true."
            ),
            "required": False,
        },
        "failure_policy": {
            "type": "string",
            "enum": ["continue", "skip_dependents", "fail_fast"],
            "description": (
                "How to handle a failed agent: continue passes the failure to dependent "
                "agents for partial recovery; skip_dependents blocks unsafe downstream "
                "work; fail_fast stops all remaining work. Defaults to continue."
            ),
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
        _parent_provider: str = "",
        agents: list[dict] | None = None,
        strategy: str = "decompose",
        include_synthesis: bool = True,
        failure_policy: str = "continue",
        **_kwargs,
    ) -> ToolResult:
        from missy.agent.sub_agent import (
            MAX_SUB_AGENT_DEPTH,
            MAX_SUB_AGENTS,
            DelegationPlanError,
            SubAgentRunner,
            SubTask,
            build_diverse_subtasks,
            parse_subtasks,
            validate_subtasks,
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

        if strategy not in {"decompose", "diverse"}:
            return ToolResult(
                success=False,
                output="",
                error="strategy must be either 'decompose' or 'diverse'.",
            )
        if failure_policy not in {"continue", "skip_dependents", "fail_fast"}:
            return ToolResult(
                success=False,
                output="",
                error=(
                    "failure_policy must be one of 'continue', 'skip_dependents', or "
                    "'fail_fast' (which stops work after the current parallel wave)."
                ),
            )

        if agents:
            if len(agents) > MAX_SUB_AGENTS:
                return ToolResult(
                    success=False,
                    output="",
                    error=f"At most {MAX_SUB_AGENTS} explicit agents are allowed.",
                )
            subtasks = []
            for index, spec in enumerate(agents):
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
                raw_tool_hints = spec.get("tool_hints") or []
                if not isinstance(raw_tool_hints, list) or any(
                    not isinstance(hint, str) for hint in raw_tool_hints
                ):
                    return ToolResult(
                        success=False,
                        output="",
                        error=f"agents[{index}].tool_hints must be a list of strings.",
                    )
                raw_provider = spec.get("provider")
                if raw_provider is not None and (
                    not isinstance(raw_provider, str) or not raw_provider.strip()
                ):
                    return ToolResult(
                        success=False,
                        output="",
                        error=f"agents[{index}].provider must be a non-empty string when set.",
                    )

                subtasks.append(
                    SubTask(
                        id=index,
                        description=str(spec["task"]).strip(),
                        name=str(spec.get("name") or "").strip()[:80],
                        role=str(spec.get("role") or "").strip()[:240],
                        focus=str(spec.get("focus") or "").strip()[:1_000],
                        success_criteria=str(spec.get("success_criteria") or "").strip()[:1_000],
                        provider=str(raw_provider or "").strip()[:128],
                        tool_hints=[hint.strip()[:80] for hint in raw_tool_hints if hint.strip()],
                        depends_on=sorted(set(raw_dependencies)),
                    )
                )
            strategy_used = "explicit"
        elif strategy == "diverse":
            try:
                subtasks = build_diverse_subtasks(
                    prompt,
                    include_synthesis=bool(include_synthesis),
                )
            except DelegationPlanError as exc:
                return ToolResult(success=False, output="", error=str(exc))
            strategy_used = "diverse"
        else:
            subtasks = parse_subtasks(prompt)
            strategy_used = "decompose"
        # SubAgentRunner.run_all() truncates its own local copy to
        # MAX_SUB_AGENTS when the caller passes more subtasks than that --
        # it never mutates *this* list, so `results` ends up shorter than
        # `subtasks` for any prompt with more than MAX_SUB_AGENTS steps.
        # Truncate the same way here so the two stay the same length for
        # the zip(..., strict=True) below, instead of that raising an
        # unhandled ValueError and crashing tool execution.
        if len(subtasks) > MAX_SUB_AGENTS:
            subtasks = subtasks[:MAX_SUB_AGENTS]
        try:
            validate_subtasks(subtasks)
        except DelegationPlanError as exc:
            return ToolResult(success=False, output="", error=f"Invalid delegation plan: {exc}")
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
            failure_policy=failure_policy,
            default_provider=_parent_provider,
        )
        results = runner.run_all(subtasks)

        succeeded = sum(t.status == "complete" for t in subtasks)
        failed = sum(t.status == "error" for t in subtasks)
        skipped = sum(t.status == "skipped" for t in subtasks)
        summary = (
            f"Delegation summary — strategy={strategy_used}; agents={len(subtasks)}; "
            f"succeeded={succeeded}; failed={failed}; skipped={skipped}."
        )
        synthesis = next((task for task in subtasks if task.is_synthesis), None)
        synthesis_section = ""
        if synthesis is not None and synthesis.result:
            synthesis_section = (
                f"\n\nFinal synthesis — Agent {synthesis.name} "
                f"({synthesis.agent_id}):\n{synthesis.result}"
            )
        lines = []
        for task, result in zip(subtasks, results, strict=True):
            displayed_result = (
                "[Final synthesis shown above.]" if task.is_synthesis and task.result else result
            )
            lines.append(
                f"Step {task.id} — Agent {task.name} ({task.agent_id})"
                f"{' [provider=' + task.provider + ']' if task.provider else ''}"
                f"{' [FAILED]' if task.status == 'error' else ''}"
                f"{' [SKIPPED]' if task.status == 'skipped' else ''}"
                f"{' — ' + task.role if task.role else ''}: {task.description}\n"
                f"{displayed_result}"
            )
        any_failed = failed > 0 or skipped > 0
        return ToolResult(
            success=not any_failed,
            output=summary + synthesis_section + "\n\nAgent results:\n\n" + "\n\n".join(lines),
            error="One or more sub-agent steps failed; see output above." if any_failed else None,
        )
