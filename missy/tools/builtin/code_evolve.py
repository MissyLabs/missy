"""Code self-evolution tool for the Missy agent.

Allows the agent to propose, review, and reject modifications to its own
source code — all inline during a conversation. Every mutation goes
through the :class:`~missy.agent.code_evolution.CodeEvolutionManager`
lifecycle so that changes are validated, tested, committed, and
reversible.

Inline lifecycle available to the agent::

    1. Agent reads source via file_read to understand the code
    2. Agent calls code_evolve(action="propose", ...) to propose a fix
    3. Agent stops. Approving and applying a proposal requires an
       authenticated human operator running
       ``missy evolve approve <id>`` and ``missy evolve apply <id>``
       from a terminal session on the host, or the equivalent
       authenticated Web TUI control when available.

SR-1.2/1.3: a model, Discord user, scheduled job, or tool call must
never approve its own code change. ``approve``, ``apply``, and
``rollback`` are deliberately **not** exposed through this agent-facing
tool — they mutate Missy's own source and restart the process, and
``CodeEvolutionManager.approve()``/``apply()``/``rollback()`` perform no
authentication of their own (they trust every caller). The only
legitimate callers are the ``missy evolve`` CLI commands
(``missy/cli/main.py``), which run under an interactive human operator's
own shell session on the host — the same trust boundary every other
local-first, single-operator Missy control surface relies on. If a Web
API surface for evolve approval is ever added, it must reuse the API's
existing authenticated-session boundary, not this tool.

Actions available to the agent:
    ``propose``
        Create a new evolution proposal with a single file diff.
    ``propose_multi``
        Create a proposal spanning multiple files.
    ``list``
        List all evolution proposals and their statuses.
    ``show``
        Show full details of a specific proposal.
    ``reject``
        Reject a proposal (safe: narrows scope, does not mutate source).

Actions that require the human-operator CLI (not available here):
    ``approve``, ``apply``, ``rollback``
"""

from __future__ import annotations

import json
import logging
from typing import Any

from missy.tools.base import BaseTool, ToolPermissions, ToolResult

logger = logging.getLogger(__name__)

# SR-1.2/1.3: actions that mutate Missy's own source and/or restart the
# process must go through an authenticated human operator (the `missy
# evolve` CLI), never through this agent-facing tool. Keep this in sync
# with the operator-only commands documented in missy/cli/main.py.
_HUMAN_OPERATOR_ONLY_ACTIONS: frozenset[str] = frozenset({"approve", "apply", "rollback"})


class CodeEvolveTool(BaseTool):
    """Agent-facing tool for proposing self-modification changes.

    Wraps the read-only and proposal-creation surface of
    :class:`~missy.agent.code_evolution.CodeEvolutionManager`. Approving,
    applying, and rolling back proposals is intentionally **not**
    reachable from here — see the module docstring and SR-1.2/1.3.
    """

    name = "code_evolve"
    description = (
        "Propose and review modifications to Missy's own source code. "
        "Use file_read first to understand the code, then propose a "
        "change with exact original_code and proposed_code. "
        "Actions: propose, propose_multi, list, show, reject. "
        "Approving, applying, and rolling back a proposal requires an "
        "authenticated human operator running `missy evolve approve/"
        "apply/rollback <id>` from a terminal on the host -- this tool "
        "cannot perform those actions and will refuse if asked. Never "
        "suggest writing directly to source files, editing the "
        "evolutions store, or any other route that bypasses that "
        "requirement; report the limitation and stop."
    )
    permissions = ToolPermissions(
        filesystem_read=True,
        filesystem_write=True,
    )

    parameters = {
        "action": {
            "type": "string",
            "description": (
                "Action to perform: propose, propose_multi, list, show, "
                "reject. (approve/apply/rollback require the human-operator "
                "`missy evolve` CLI and are not available here.)"
            ),
            "enum": [
                "propose",
                "propose_multi",
                "list",
                "show",
                "reject",
                "approve",
                "apply",
                "rollback",
            ],
            "required": True,
        },
        "title": {
            "type": "string",
            "description": "One-line summary of the proposed change (propose/propose_multi).",
            "required": False,
        },
        "description": {
            "type": "string",
            "description": "Detailed rationale for the change (propose/propose_multi).",
            "required": False,
        },
        "file_path": {
            "type": "string",
            "description": "Relative path from repo root to the file to modify (propose).",
            "required": False,
        },
        "original_code": {
            "type": "string",
            "description": "Exact text to replace in the file (propose).",
            "required": False,
        },
        "proposed_code": {
            "type": "string",
            "description": "Replacement text (propose).",
            "required": False,
        },
        "diffs": {
            "type": "string",
            "description": (
                "JSON array of diffs for propose_multi. Each element: "
                '{"file_path": "...", "original_code": "...", '
                '"proposed_code": "...", "description": "..."}'
            ),
            "required": False,
        },
        "trigger": {
            "type": "string",
            "description": (
                "What caused this proposal: repeated_error, user_request, "
                "learning, performance, security (default: user_request)."
            ),
            "required": False,
        },
        "trigger_detail": {
            "type": "string",
            "description": "Extra context for the trigger (error message, etc.).",
            "required": False,
        },
        "confidence": {
            "type": "number",
            "description": "Agent confidence 0.0-1.0 that this change is correct (default: 0.5).",
            "required": False,
        },
        "error_pattern": {
            "type": "string",
            "description": "Recurring error pattern if triggered by an error.",
            "required": False,
        },
        "proposal_id": {
            "type": "string",
            "description": "Proposal ID for show/reject actions.",
            "required": False,
        },
    }

    def execute(self, **kwargs: Any) -> ToolResult:
        """Dispatch to the appropriate action handler."""
        action = kwargs.get("action", "")

        if action in _HUMAN_OPERATOR_ONLY_ACTIONS:
            proposal_id = kwargs.get("proposal_id", "")
            id_arg = f" {proposal_id}" if proposal_id else " <id>"
            return ToolResult(
                success=False,
                output=None,
                error=(
                    f"'{action}' requires an authenticated human operator (SR-1.2/1.3: "
                    "a model must never approve or apply its own code change). "
                    f"Ask the operator to run `missy evolve {action}{id_arg}` from a "
                    "terminal on the host, or use the Web TUI evolve controls if "
                    "available. There is no way to perform this action from within "
                    "the conversation -- do not write to source files, the "
                    "evolutions store, or any other file as a substitute."
                ),
            )

        from missy.agent.code_evolution import CodeEvolutionManager

        try:
            mgr = CodeEvolutionManager()
        except Exception as exc:
            return ToolResult(
                success=False,
                output=None,
                error=f"Failed to initialize CodeEvolutionManager: {exc}",
            )

        dispatch = {
            "propose": self._propose,
            "propose_multi": self._propose_multi,
            "list": self._list,
            "show": self._show,
            "reject": self._reject,
        }

        handler = dispatch.get(action)
        if handler is None:
            return ToolResult(
                success=False,
                output=None,
                error=(
                    f"Unknown action: {action!r}. "
                    "Use: propose, propose_multi, list, show, reject "
                    "(approve/apply/rollback require the `missy evolve` CLI)"
                ),
            )

        return handler(mgr, kwargs)

    # ------------------------------------------------------------------
    # propose
    # ------------------------------------------------------------------

    def _propose(self, mgr, kwargs: dict) -> ToolResult:
        required = ("title", "description", "file_path", "original_code", "proposed_code")
        missing = [k for k in required if not kwargs.get(k)]
        if missing:
            return ToolResult(
                success=False,
                output=None,
                error=f"Missing required fields for propose: {', '.join(missing)}",
            )

        try:
            prop = mgr.propose(
                title=kwargs["title"],
                description=kwargs["description"],
                file_path=kwargs["file_path"],
                original_code=kwargs["original_code"],
                proposed_code=kwargs["proposed_code"],
                trigger=kwargs.get("trigger", "user_request"),
                trigger_detail=kwargs.get("trigger_detail", ""),
                confidence=max(0.0, min(1.0, float(kwargs.get("confidence", 0.5)))),
                error_pattern=kwargs.get("error_pattern", ""),
            )
            return ToolResult(
                success=True,
                output=(
                    f"Evolution proposed: {prop.id}\n"
                    f"Title: {prop.title}\n"
                    f"Status: {prop.status.value}\n"
                    f"File: {prop.diffs[0].file_path}\n\n"
                    "This proposal is pending. Ask an operator to review and run "
                    f"`missy evolve approve {prop.id}` then `missy evolve apply {prop.id}` "
                    "from a terminal on the host -- approval and apply are not "
                    "available from this conversation."
                ),
            )
        except ValueError as exc:
            return ToolResult(success=False, output=None, error=str(exc))

    # ------------------------------------------------------------------
    # propose_multi
    # ------------------------------------------------------------------

    def _propose_multi(self, mgr, kwargs: dict) -> ToolResult:
        from missy.agent.code_evolution import FileDiff

        required = ("title", "description", "diffs")
        missing = [k for k in required if not kwargs.get(k)]
        if missing:
            return ToolResult(
                success=False,
                output=None,
                error=f"Missing required fields for propose_multi: {', '.join(missing)}",
            )

        try:
            raw_diffs = json.loads(kwargs["diffs"])
            diffs = [FileDiff(**d) for d in raw_diffs]
        except (json.JSONDecodeError, TypeError) as exc:
            return ToolResult(
                success=False,
                output=None,
                error=f"Invalid diffs JSON: {exc}",
            )

        try:
            prop = mgr.propose_multi(
                title=kwargs["title"],
                description=kwargs["description"],
                diffs=diffs,
                trigger=kwargs.get("trigger", "user_request"),
                trigger_detail=kwargs.get("trigger_detail", ""),
                confidence=max(0.0, min(1.0, float(kwargs.get("confidence", 0.5)))),
                error_pattern=kwargs.get("error_pattern", ""),
            )
            files = [d.file_path for d in prop.diffs]
            return ToolResult(
                success=True,
                output=(
                    f"Multi-file evolution proposed: {prop.id}\n"
                    f"Title: {prop.title}\n"
                    f"Files: {', '.join(files)}\n"
                    f"Status: {prop.status.value}\n\n"
                    "This proposal is pending. Ask an operator to review and run "
                    f"`missy evolve approve {prop.id}` from a terminal on the host -- "
                    "approval is not available from this conversation."
                ),
            )
        except ValueError as exc:
            return ToolResult(success=False, output=None, error=str(exc))

    # ------------------------------------------------------------------
    # list
    # ------------------------------------------------------------------

    def _list(self, mgr, _kwargs: dict) -> ToolResult:
        proposals = mgr.list_all()
        if not proposals:
            return ToolResult(success=True, output="No evolution proposals.")

        lines = [f"{'ID':8} {'Status':12} {'Trigger':16} {'Confidence':>10}  Title"]
        lines.append("-" * 80)
        for p in proposals:
            lines.append(
                f"{p.id:8} {p.status.value:12} {p.trigger.value:16} "
                f"{p.confidence:>10.0%}  {p.title[:40]}"
            )
        return ToolResult(success=True, output="\n".join(lines))

    # ------------------------------------------------------------------
    # show
    # ------------------------------------------------------------------

    def _show(self, mgr, kwargs: dict) -> ToolResult:
        proposal_id = kwargs.get("proposal_id", "")
        if not proposal_id:
            return ToolResult(success=False, output=None, error="proposal_id is required for show.")

        prop = mgr.get(proposal_id)
        if not prop:
            return ToolResult(
                success=False,
                output=None,
                error=f"Proposal {proposal_id!r} not found.",
            )

        lines = [
            f"ID: {prop.id}",
            f"Title: {prop.title}",
            f"Status: {prop.status.value}",
            f"Trigger: {prop.trigger.value}",
            f"Confidence: {prop.confidence:.0%}",
            f"Created: {prop.created_at}",
            f"Resolved: {prop.resolved_at or '—'}",
            f"Commit: {prop.git_commit_sha or '—'}",
            f"\nDescription:\n{prop.description}",
        ]

        if prop.diffs:
            lines.append(f"\nDiffs ({len(prop.diffs)}):")
            for i, d in enumerate(prop.diffs, 1):
                lines.append(f"\n--- Diff {i}: {d.file_path} ---")
                if d.description:
                    lines.append(f"Why: {d.description}")
                lines.append(f"- {d.original_code}")
                lines.append(f"+ {d.proposed_code}")

        if prop.error_pattern:
            lines.append(f"\nError pattern: {prop.error_pattern}")

        if prop.test_output:
            lines.append(f"\nTest output (last 500 chars):\n{prop.test_output[-500:]}")

        return ToolResult(success=True, output="\n".join(lines))

    # ------------------------------------------------------------------
    # reject
    # ------------------------------------------------------------------

    def _reject(self, mgr, kwargs: dict) -> ToolResult:
        proposal_id = kwargs.get("proposal_id", "")
        if not proposal_id:
            return ToolResult(
                success=False, output=None, error="proposal_id is required for reject."
            )

        if mgr.reject(proposal_id):
            return ToolResult(
                success=True,
                output=f"Proposal {proposal_id} rejected.",
            )
        return ToolResult(
            success=False,
            output=None,
            error=f"Proposal {proposal_id!r} not found or not in a rejectable status.",
        )
