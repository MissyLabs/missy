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
    decisions keyed by a hash of ``session_id + action + detail``. A
    single ``AgentRuntime`` (and therefore a single ``InteractiveApproval``
    instance) is shared across every Discord user/Web API session it
    serves -- without the ``session_id`` component, an operator's one-time
    "allow always" response to one user's blocked request would silently
    and permanently auto-approve that exact same action/detail for every
    *other* user of the same running process too, contradicting this
    class's own "remembered for the duration of the session" contract.
    ``session_id`` defaults to ``""`` for the single-operator interactive
    CLI case (``missy run``/``missy ask``), where there is genuinely only
    one session to begin with.
    """

    def __init__(self) -> None:
        self._remembered: dict[str, bool] = {}
        self._lock = threading.Lock()
        # Serializes _do_prompt() calls across concurrent callers (e.g.
        # SubAgentRunner's ThreadPoolExecutor running independent subtasks
        # concurrently against the same shared AgentRuntime/session). Without
        # this, two threads hitting a policy denial at once could both reach
        # _do_prompt() at the same moment: their Rich panels interleave on
        # stderr and both block on console.input() reading the same stdin,
        # so the operator's single typed response races between the two
        # prompts and can resolve the wrong one. Deliberately a *different*
        # lock than self._lock (which only guards the _remembered dict) --
        # holding this one across the blocking I/O in _do_prompt() while
        # self._lock is also acquired inside it (the "allow always" branch)
        # would deadlock if they were the same lock.
        self._prompt_lock = threading.Lock()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def check_remembered(self, action: str, detail: str, session_id: str = "") -> bool | None:
        """Check whether a previous "allow always" decision exists.

        Args:
            action: The action category (e.g. ``"network_request"``).
            detail: Specific detail string (e.g. URL or command).
            session_id: The session this decision is scoped to. Defaults
                to ``""`` for the single-operator interactive CLI case.

        Returns:
            ``True`` if previously allowed always, ``False`` if
            previously denied always, or ``None`` if no decision
            was recorded.
        """
        key = self._make_key(action, detail, session_id)
        with self._lock:
            return self._remembered.get(key)

    def prompt_user(self, action: str, detail: str, session_id: str = "") -> bool:
        """Display a Rich-formatted prompt asking the operator to approve.

        Responses:
            - ``y`` — allow this one time
            - ``n`` — deny
            - ``a`` — allow always (remembered for this session)

        When stdin is not a TTY, returns ``False`` (deny) immediately.

        Args:
            action: The action category.
            detail: Human-readable detail about the blocked operation.
            session_id: The session this decision is scoped to. Defaults
                to ``""`` for the single-operator interactive CLI case;
                callers serving multiple concurrent sessions/users from
                one shared instance (e.g. a Discord bot's per-guild
                sessions) must pass their real session id so "allow
                always" doesn't leak across unrelated users.

        Returns:
            ``True`` if the operator approves, ``False`` otherwise.
        """
        # Check remembered decisions first.
        remembered = self.check_remembered(action, detail, session_id)
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

        with self._prompt_lock:
            # Re-check: another thread may have already prompted for this
            # exact action/detail/session (and possibly recorded an "allow
            # always" decision) while this call was waiting for the lock --
            # in which case reuse that answer instead of showing the
            # operator a redundant, confusing duplicate prompt.
            remembered = self.check_remembered(action, detail, session_id)
            if remembered is not None:
                return remembered
            return self._do_prompt(action, detail, session_id)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _do_prompt(self, action: str, detail: str, session_id: str = "") -> bool:
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
            key = self._make_key(action, detail, session_id)
            with self._lock:
                self._remembered[key] = True
            return True
        else:
            # Anything other than y/a is a deny.
            return False

    @staticmethod
    def _make_key(action: str, detail: str, session_id: str = "") -> str:
        """Create a stable hash key for a session+action+detail triple."""
        raw = f"{session_id}:{action}:{detail}"
        return hashlib.sha256(raw.encode()).hexdigest()

    @staticmethod
    def _is_tty() -> bool:
        """Return ``True`` if stdin is connected to a terminal."""
        try:
            return sys.stdin.isatty()
        except Exception:
            return False
