"""Code self-evolution tool for the Missy agent.

Allows the agent to propose, review, approve, apply, and roll back
modifications to its own source code — all inline during a conversation.

Every mutation goes through the
:class:`~missy.agent.code_evolution.CodeEvolutionManager` lifecycle so
that changes are validated, tested, committed, and reversible.

Full inline lifecycle::

    1. Agent reads source via file_read to understand the code
    2. Agent calls code_evolve(action="propose", ...) to propose a fix
    3. Agent calls code_evolve(action="approve", ...) to approve it
       (human confirmation via ApprovalGate)
    4. Agent calls code_evolve(action="apply", ...) to apply it
       (runs tests, commits, restarts on success)

Actions:
    ``propose``
        Create a new evolution proposal with a single file diff.
    ``propose_multi``
        Create a proposal spanning multiple files.
    ``list``
        List all evolution proposals and their statuses.
    ``show``
        Show full details of a specific proposal.
    ``approve``
        Approve a proposal (requests human confirmation via ApprovalGate).
    ``reject``
        Reject a proposal.
    ``apply``
        Apply an approved proposal (runs tests, commits on success,
        restarts the process to load changes).
    ``rollback``
        Revert a previously applied evolution via ``git revert``.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from missy.tools.base import BaseTool, ToolPermissions, ToolResult

logger = logging.getLogger(__name__)


class CodeEvolveTool(BaseTool):
    """Agent-facing tool for self-modification proposals.

    This tool wraps :class:`~missy.agent.code_evolution.CodeEvolutionManager`
    and exposes its full lifecycle to the model as a single unified tool
    with an ``action`` parameter.  The agent can drive the complete
    propose → approve → apply → rollback workflow without leaving the
    conversation.
    """

    name = "code_evolve"
    description = (
        "Propose, approve, apply, and roll back modifications to Missy's "
        "own source code. Use file_read first to understand the code, then "
        "propose a change with exact original_code and proposed_code. "
        "Actions: propose, propose_multi, list, show, approve, reject, "
        "apply, rollback. The approve action requires human confirmation. "
        "The apply action runs the full test suite and commits on success."
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
                "approve, reject, apply, rollback"
            ),
            "enum": [
                "propose", "propose_multi", "list", "show",
                "approve", "reject", "apply", "rollback",
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
            "description": "Proposal ID for show/approve/reject/apply/rollback actions.",
            "required": False,
        },
    }

    def execute(self, **kwargs: Any) -> ToolResult:
        """Dispatch to the appropriate action handler."""
        from missy.agent.code_evolution import CodeEvolutionManager

        action = kwargs.get("action", "")

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
            "approve": self._approve,
            "reject": self._reject,
            "apply": self._apply,
            "rollback": self._rollback,
        }

        handler = dispatch.get(action)
        if handler is None:
            return ToolResult(
                success=False,
                output=None,
                error=(
                    f"Unknown action: {action!r}. "
                    "Use: propose, propose_multi, list, show, "
                    "approve, reject, apply, rollback"
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
                confidence=float(kwargs.get("confidence", 0.5)),
                error_pattern=kwargs.get("error_pattern", ""),
            )
            return ToolResult(
                success=True,
                output=(
                    f"Evolution proposed: {prop.id}\n"
                    f"Title: {prop.title}\n"
                    f"Status: {prop.status.value}\n"
                    f"File: {prop.diffs[0].file_path}\n\n"
                    f"Next: call code_evolve(action='approve', proposal_id='{prop.id}') "
                    "to approve, then code_evolve(action='apply', ...) to apply."
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
                confidence=float(kwargs.get("confidence", 0.5)),
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
                    f"Next: call code_evolve(action='approve', proposal_id='{prop.id}') "
                    "to approve."
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
            return ToolResult(
                success=False, output=None, error="proposal_id is required for show."
            )

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
    # approve
    # ------------------------------------------------------------------

    def _approve(self, mgr, kwargs: dict) -> ToolResult:
        proposal_id = kwargs.get("proposal_id", "")
        if not proposal_id:
            return ToolResult(
                success=False, output=None, error="proposal_id is required for approve."
            )

        prop = mgr.get(proposal_id)
        if not prop:
            return ToolResult(
                success=False,
                output=None,
                error=f"Proposal {proposal_id!r} not found.",
            )

        if prop.status.value not in ("proposed",):
            return ToolResult(
                success=False,
                output=None,
                error=(
                    f"Proposal {proposal_id} is '{prop.status.value}', "
                    "only 'proposed' proposals can be approved."
                ),
            )

        if mgr.approve(proposal_id):
            return ToolResult(
                success=True,
                output=(
                    f"Proposal {proposal_id} approved.\n\n"
                    f"Next: call code_evolve(action='apply', proposal_id='{proposal_id}') "
                    "to apply (runs tests, commits, restarts)."
                ),
            )
        return ToolResult(
            success=False,
            output=None,
            error=f"Failed to approve proposal {proposal_id}.",
        )

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

    # ------------------------------------------------------------------
    # apply
    # ------------------------------------------------------------------

    def _apply(self, mgr, kwargs: dict) -> ToolResult:
        proposal_id = kwargs.get("proposal_id", "")
        if not proposal_id:
            return ToolResult(
                success=False, output=None, error="proposal_id is required for apply."
            )

        prop = mgr.get(proposal_id)
        if not prop:
            return ToolResult(
                success=False,
                output=None,
                error=f"Proposal {proposal_id!r} not found.",
            )

        if prop.status != "approved":
            return ToolResult(
                success=False,
                output=None,
                error=(
                    f"Proposal {proposal_id} is '{prop.status.value}'. "
                    "Approve it first with code_evolve(action='approve', "
                    f"proposal_id='{proposal_id}')."
                ),
            )

        try:
            result = mgr.apply(proposal_id)
            if result["success"]:
                import contextlib

                from missy.agent.code_evolution import restart_process

                msg = (
                    result["message"]
                    + "\n\nRestarting process to load evolved code..."
                )

                with contextlib.suppress(SystemExit):
                    restart_process()
                return ToolResult(success=True, output=msg)
            return ToolResult(
                success=False,
                output=result["message"],
                error=result["message"],
            )
        except ValueError as exc:
            return ToolResult(success=False, output=None, error=str(exc))

    # ------------------------------------------------------------------
    # rollback
    # ------------------------------------------------------------------

    def _rollback(self, mgr, kwargs: dict) -> ToolResult:
        proposal_id = kwargs.get("proposal_id", "")
        if not proposal_id:
            return ToolResult(
                success=False, output=None, error="proposal_id is required for rollback."
            )

        result = mgr.rollback(proposal_id)
        if result["success"]:
            import contextlib

            from missy.agent.code_evolution import restart_process

            msg = result["message"] + "\n\nRestarting to load reverted code..."
            with contextlib.suppress(SystemExit):
                restart_process()
            return ToolResult(success=True, output=msg)
        return ToolResult(
            success=False,
            output=None,
            error=result["message"],
        )
