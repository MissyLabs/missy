"""Shell policy enforcement for the Missy framework.

Shell command execution is evaluated against a :class:`ShellPolicy` instance.
When ``policy.enabled`` is ``False`` every command is denied regardless of the
allow-list.  When enabled, the first token of the command (the program name)
must begin with an entry in ``policy.allowed_commands``.

Every check emits an :class:`~missy.core.events.AuditEvent`.

Example::

    from missy.config.settings import ShellPolicy
    from missy.policy.shell import ShellPolicyEngine

    policy = ShellPolicy(enabled=True, allowed_commands=["git", "ls"])
    engine = ShellPolicyEngine(policy)
    engine.check_command("git status")   # -> True
    engine.check_command("rm -rf /")     # -> raises PolicyViolationError
"""

from __future__ import annotations

import logging
import shlex
from typing import Optional

from missy.config.settings import ShellPolicy
from missy.core.events import AuditEvent, event_bus
from missy.core.exceptions import PolicyViolationError

logger = logging.getLogger(__name__)


class ShellPolicyEngine:
    """Evaluates shell commands against a :class:`ShellPolicy`.

    Args:
        policy: The shell policy to enforce.
    """

    def __init__(self, policy: ShellPolicy) -> None:
        self._policy = policy

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def check_command(
        self,
        command: str,
        session_id: str = "",
        task_id: str = "",
    ) -> bool:
        """Return ``True`` if executing *command* is permitted by policy.

        Evaluation order:

        1. If ``policy.enabled`` is ``False``, deny unconditionally.
        2. Parse the leading token (program name) from *command* using
           POSIX shell splitting rules.  An empty or whitespace-only command
           string is always denied.
        3. Check whether the program name starts with any entry in
           ``policy.allowed_commands``.  This permits both bare names
           (``"git"``) and path-qualified names (``"/usr/bin/git"``).

        Args:
            command: The shell command string to evaluate.
            session_id: Optional identifier of the calling session.
            task_id: Optional identifier of the calling task.

        Returns:
            ``True`` when the command is explicitly allowed.

        Raises:
            PolicyViolationError: When the command is denied.
        """
        # Step 1 – shell execution globally disabled.
        if not self._policy.enabled:
            self._emit_event(command, "deny", "shell_disabled", session_id, task_id)
            raise PolicyViolationError(
                "Shell execution is disabled by policy.",
                category="shell",
                detail="ShellPolicy.enabled is False; no shell commands are permitted.",
            )

        # Step 2 – extract program token.
        program = self._extract_program(command)
        if program is None:
            self._emit_event(command, "deny", None, session_id, task_id)
            raise PolicyViolationError(
                "Shell command denied: empty or unparseable command.",
                category="shell",
                detail=f"Could not determine the program name from command {command!r}.",
            )

        # Step 3 – allow-list check.
        # Empty allowed_commands means nothing is allowed (deny-all).
        rule = self._match_allowed(program)
        if rule:
            self._emit_event(command, "allow", rule, session_id, task_id)
            return True

        self._emit_event(command, "deny", None, session_id, task_id)
        raise PolicyViolationError(
            f"Shell command denied: {program!r} is not in the allowed commands list.",
            category="shell",
            detail=(
                f"Program {program!r} (from command {command!r}) did not match "
                "any allowed_commands entry."
            ),
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_program(command: str) -> Optional[str]:
        """Return the program token from a shell command string.

        Uses :func:`shlex.split` for POSIX-correct tokenisation.  Returns
        ``None`` if the command is empty, contains only whitespace, or cannot
        be parsed (e.g. unmatched quotes).

        Args:
            command: Raw shell command string.

        Returns:
            The first token (program name), or ``None``.
        """
        if not command or not command.strip():
            return None
        try:
            tokens = shlex.split(command)
        except ValueError:
            # Malformed quoting – deny.
            logger.debug("ShellPolicyEngine: could not parse command %r", command)
            return None
        return tokens[0] if tokens else None

    def _match_allowed(self, program: str) -> Optional[str]:
        """Return the first ``allowed_commands`` entry matched by *program*.

        Matching is performed as a prefix check on the *basename* of the
        program so that ``/usr/bin/git`` matches the entry ``"git"``, and the
        entry ``"git"`` does not accidentally match ``"gitk"`` (because the
        comparison is against the full basename or path, not a substring).

        Specifically, the program matches an entry when:

        * ``program`` equals ``entry`` exactly, or
        * ``program`` ends with ``"/" + entry`` (path-qualified match), or
        * ``entry`` ends with ``"/" + basename(program)`` (entry is qualified).

        Args:
            program: The program name or path extracted from the command.

        Returns:
            The matching entry string prefixed with ``"cmd:"``, or ``None``.
        """
        import os.path

        program_basename = os.path.basename(program)

        for entry in self._policy.allowed_commands:
            entry_basename = os.path.basename(entry)
            # Match if basenames agree.
            if program_basename == entry_basename:
                return f"cmd:{entry}"
            # Also allow: configured entry is a prefix of the program token
            # for commands like "git" matching "git-commit" style sub-commands
            # is intentionally NOT done here – we match exact basename only
            # to avoid unintended over-permissioning.

        return None

    def _emit_event(
        self,
        command: str,
        result: str,
        rule: Optional[str],
        session_id: str,
        task_id: str,
    ) -> None:
        """Publish a shell audit event.

        Args:
            command: The original command string.
            result: ``"allow"`` or ``"deny"``.
            rule: The matching policy rule, or ``None``.
            session_id: Calling session identifier.
            task_id: Calling task identifier.
        """
        event = AuditEvent.now(
            session_id=session_id,
            task_id=task_id,
            event_type="shell_check",
            category="shell",
            result=result,  # type: ignore[arg-type]
            policy_rule=rule,
            detail={"command": command},
        )
        event_bus.publish(event)
