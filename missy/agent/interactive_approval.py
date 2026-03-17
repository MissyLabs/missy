"""Interactive approval TUI for policy-denied operations.

When a network request or tool call is blocked by policy, this module
surfaces the decision in the terminal for real-time operator approval.
Approved decisions can be remembered for the duration of the session.

Usage::

    approval = InteractiveApproval()
    allowed = approval.prompt_user("network_request", "GET https://example.com")
    if allowed:
        # proceed with the request
        ...
"""

from __future__ import annotations

import hashlib
import logging
import sys
import threading

logger = logging.getLogger(__name__)


class InteractiveApproval:
    """Terminal-based interactive approval for policy-denied operations.

    Thread-safe.  When stdin is not a TTY (e.g. running as a service or
    in a pipeline), the safe default is to deny all requests without
    prompting.

    The ``_remembered`` dict stores session-scoped "allow always"
    decisions keyed by a hash of ``action + detail``.
    """

    def __init__(self) -> None:
        self._remembered: dict[str, bool] = {}
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def check_remembered(self, action: str, detail: str) -> bool | None:
        """Check whether a previous "allow always" decision exists.

        Args:
            action: The action category (e.g. ``"network_request"``).
            detail: Specific detail string (e.g. URL or command).

        Returns:
            ``True`` if previously allowed always, ``False`` if
            previously denied always, or ``None`` if no decision
            was recorded.
        """
        key = self._make_key(action, detail)
        with self._lock:
            return self._remembered.get(key)

    def prompt_user(self, action: str, detail: str) -> bool:
        """Display a Rich-formatted prompt asking the operator to approve.

        Responses:
            - ``y`` — allow this one time
            - ``n`` — deny
            - ``a`` — allow always (remembered for this session)

        When stdin is not a TTY, returns ``False`` (deny) immediately.

        Args:
            action: The action category.
            detail: Human-readable detail about the blocked operation.

        Returns:
            ``True`` if the operator approves, ``False`` otherwise.
        """
        # Check remembered decisions first.
        remembered = self.check_remembered(action, detail)
        if remembered is not None:
            return remembered

        # Non-interactive fallback: always deny.
        if not self._is_tty():
            logger.debug(
                "Non-interactive terminal; auto-denying %s: %s",
                action,
                detail,
            )
            return False

        return self._do_prompt(action, detail)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _do_prompt(self, action: str, detail: str) -> bool:
        """Render the prompt and read operator input."""
        try:
            from rich.console import Console
            from rich.panel import Panel

            console = Console(stderr=True)
            console.print()
            console.print(
                Panel(
                    f"[bold yellow]Action:[/] {action}\n"
                    f"[bold yellow]Detail:[/] {detail}\n\n"
                    "[dim](y) allow once  (n) deny  (a) allow always[/]",
                    title="[bold red]Policy Denied — Approval Required[/]",
                    border_style="red",
                )
            )
            response = console.input("[bold]Allow? [y/n/a]: [/]").strip().lower()
        except (ImportError, EOFError, KeyboardInterrupt):
            return False

        if response == "y":
            return True
        elif response == "a":
            key = self._make_key(action, detail)
            with self._lock:
                self._remembered[key] = True
            return True
        else:
            # Anything other than y/a is a deny.
            return False

    @staticmethod
    def _make_key(action: str, detail: str) -> str:
        """Create a stable hash key for an action+detail pair."""
        raw = f"{action}:{detail}"
        return hashlib.sha256(raw.encode()).hexdigest()

    @staticmethod
    def _is_tty() -> bool:
        """Return ``True`` if stdin is connected to a terminal."""
        try:
            return sys.stdin.isatty()
        except Exception:
            return False
