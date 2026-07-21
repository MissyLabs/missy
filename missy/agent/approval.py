"""Approval gate: pause execution and wait for user confirmation.

Usage::

    gate = ApprovalGate(channel_send_fn)
    gate.request("delete all files in /tmp/work", risk="high")
    # Blocks until approved or timeout
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable

logger = logging.getLogger(__name__)


class ApprovalTimeout(Exception):
    pass


class ApprovalDenied(Exception):
    pass


class PendingApproval:
    def __init__(self, action: str, reason: str, timeout: float):
        self.action = action
        self.reason = reason
        self.timeout = timeout
        self._event = threading.Event()
        self._approved: bool | None = None

    def approve(self) -> None:
        self._approved = True
        self._event.set()

    def deny(self) -> None:
        self._approved = False
        self._event.set()

    def wait(self) -> bool:
        self._event.wait(timeout=self.timeout)
        if self._approved is None:
            raise ApprovalTimeout(f"Approval timed out after {self.timeout}s for: {self.action}")
        if not self._approved:
            raise ApprovalDenied(f"Action denied: {self.action}")
        return True


class ApprovalGate:
    """Manages pending approval requests.

    Args:
        send_fn: Callable that sends a message to the user (e.g. channel.send).
        default_timeout: Seconds to wait for approval (default 60).
    """

    def __init__(self, send_fn: Callable[[str], None] | None = None, default_timeout: float = 60.0):
        self._send = send_fn
        self._timeout = default_timeout
        self._pending: dict[str, PendingApproval] = {}
        self._lock = threading.Lock()

    def request(self, action: str, reason: str = "", risk: str = "medium") -> None:
        """Request approval for an action. Blocks until approved, denied, or timeout.

        Args:
            action: Short description of the action.
            reason: Why approval is required.
            risk: "low" | "medium" | "high"

        Raises:
            ApprovalTimeout: If no response within timeout.
            ApprovalDenied: If the user denies the action.
        """
        pending = PendingApproval(action, reason, self._timeout)
        import uuid

        approval_id = str(uuid.uuid4())[:8]

        with self._lock:
            self._pending[approval_id] = pending

        msg = f"⚠️ **Approval Required** [ID: {approval_id}]\nAction: {action}\nRisk: {risk}\n"
        if reason:
            msg += f"Reason: {reason}\n"
        msg += (
            f"\nReply `approve {approval_id}` or `deny {approval_id}` within {int(self._timeout)}s."
        )

        if self._send:
            try:
                self._send(msg)
            except Exception as exc:
                logger.warning("ApprovalGate: could not send approval prompt: %s", exc)

        try:
            pending.wait()
        finally:
            with self._lock:
                self._pending.pop(approval_id, None)

    def handle_response(self, text: str) -> bool:
        """Process an approval/deny response from a user message.

        Returns True if the message was handled as an approval command.
        """
        text = text.strip().lower()
        with self._lock:
            for approval_id, pending in list(self._pending.items()):
                if text.startswith(f"approve {approval_id}") or text == f"approve {approval_id}":
                    pending.approve()
                    return True
                if text.startswith(f"deny {approval_id}") or text == f"deny {approval_id}":
                    pending.deny()
                    return True
        return False

    def list_pending(self) -> list[dict]:
        with self._lock:
            return [
                {"id": k, "action": v.action, "reason": v.reason} for k, v in self._pending.items()
            ]

    def approve_by_id(self, approval_id: str) -> bool:
        """Approve the pending request identified by *approval_id*.

        Args:
            approval_id: The ID returned by :meth:`list_pending`.

        Returns:
            ``True`` if a matching pending request was found and
            approved, ``False`` if no request with that ID is pending
            (already resolved, timed out, or never existed).
        """
        with self._lock:
            pending = self._pending.get(approval_id)
            if pending is None:
                return False
            pending.approve()
            return True

    def deny_by_id(self, approval_id: str) -> bool:
        """Deny the pending request identified by *approval_id*.

        Args:
            approval_id: The ID returned by :meth:`list_pending`.

        Returns:
            ``True`` if a matching pending request was found and
            denied, ``False`` if no request with that ID is pending.
        """
        with self._lock:
            pending = self._pending.get(approval_id)
            if pending is None:
                return False
            pending.deny()
            return True


# ---------------------------------------------------------------------------
# Shared gate accessor
# ---------------------------------------------------------------------------
#
# Built-in tools (obs_tools.py, desktop_tools.py, ...) that need human
# confirmation for a destructive/public-facing action have no constructor
# injection point today -- register_builtin_tools() instantiates every
# tool class with zero arguments (see missy/tools/builtin/__init__.py).
# Rather than threading an ApprovalGate through that registration path (a
# much larger, riskier plumbing change), such tools look it up here at
# execute() time. gateway_start() registers the same shared gate it
# already builds for ProactiveManager/the Web API here too, so an
# interactive Discord/Web session can actually approve these prompts; a
# process with no gate registered (plain `missy ask`, most tests) sees
# `get_shared_approval_gate()` return None, and each such tool must treat
# that as "fail closed" -- the same posture McpManager already uses for
# `requires_approval` MCP tools when no gate is configured.

_shared_gate: ApprovalGate | None = None
_shared_gate_lock = threading.Lock()


def set_shared_approval_gate(gate: ApprovalGate | None) -> None:
    """Register *gate* as the process-wide shared approval gate.

    Called once by ``gateway_start()`` with the same gate wired into
    ``ProactiveManager``/the Web API, so built-in tools without their own
    constructor-injected gate can still route through real interactive
    confirmation. Passing ``None`` clears it (e.g. on shutdown).
    """
    global _shared_gate
    with _shared_gate_lock:
        _shared_gate = gate


def get_shared_approval_gate() -> ApprovalGate | None:
    """Return the process-wide shared approval gate, or ``None`` if unset."""
    with _shared_gate_lock:
        return _shared_gate
