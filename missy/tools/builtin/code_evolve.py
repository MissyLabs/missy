"""Code self-evolution tool for the Missy agent.

Allows the agent to propose, inspect, and (with approval) apply
modifications to its own source code.  Every mutation goes through the
:class:`~missy.agent.code_evolution.CodeEvolutionManager` lifecycle so
that changes are validated, tested, committed, and reversible.

Actions:
    ``propose``
        Create a new evolution proposal with a code diff.
    ``propose_multi``
        Create a proposal spanning multiple files.
    ``list``
        List all evolution proposals.
    ``show``
        Show full details of a specific proposal.
    ``apply``
        Apply an *approved* proposal (runs tests, commits on success).

The ``apply`` action is gated behind the
:class:`~missy.agent.approval.ApprovalGate` — the agent cannot
silently rewrite itself.
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
    and exposes its operations to the model as a single unified tool with
    an ``action`` parameter.
    """

    name = "code_evolve"
    description = (
        "Propose and manage modifications to Missy's own source code. "
        "Actions: propose (single file), propose_multi (multiple files), "
        "list, show, apply. Applied changes require prior human approval "
        "and pass the full test suite."
    )
    permissions = ToolPermissions(
        filesystem_read=True,
        filesystem_write=True,
    )

    parameters = {
        "action": {
            "type": "string",
            "description": (
                "Action to perform: propose, propose_multi, list, show, apply"
            ),
            "enum": ["propose", "propose_multi", "list", "show", "apply"],
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
            "description": "Proposal ID for show/apply actions.",
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

        if action == "propose":
            return self._propose(mgr, kwargs)
        elif action == "propose_multi":
            return self._propose_multi(mgr, kwargs)
        elif action == "list":
            return self._list(mgr)
        elif action == "show":
            return self._show(mgr, kwargs)
        elif action == "apply":
            return self._apply(mgr, kwargs)
        else:
            return ToolResult(
                success=False,
                output=None,
                error=f"Unknown action: {action!r}. Use: propose, propose_multi, list, show, apply",
            )

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
                    "Use `missy evolve approve " + prop.id + "` to approve, "
                    "then `missy evolve apply " + prop.id + "` to apply."
                ),
            )
        except ValueError as exc:
            return ToolResult(success=False, output=None, error=str(exc))

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
                    f"Status: {prop.status.value}"
                ),
            )
        except ValueError as exc:
            return ToolResult(success=False, output=None, error=str(exc))

    def _list(self, mgr) -> ToolResult:
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
                    "It must be approved first via `missy evolve approve`."
                ),
            )

        try:
            result = mgr.apply(proposal_id)
            return ToolResult(
                success=result["success"],
                output=result["message"],
                error=None if result["success"] else result["message"],
            )
        except ValueError as exc:
            return ToolResult(success=False, output=None, error=str(exc))
