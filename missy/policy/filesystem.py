"""Filesystem policy enforcement for the Missy framework.

Read and write access to the local filesystem is evaluated against a
:class:`FilesystemPolicy` instance.  Symlinks are always resolved before the
path is compared against the allow-lists, preventing traversal attacks.

Every check emits an :class:`~missy.core.events.AuditEvent` regardless of
the outcome.

Example::

    from missy.config.settings import FilesystemPolicy
    from missy.policy.filesystem import FilesystemPolicyEngine

    policy = FilesystemPolicy(
        allowed_read_paths=["/home/user/workspace"],
        allowed_write_paths=["/home/user/workspace/output"],
    )
    engine = FilesystemPolicyEngine(policy)
    engine.check_read("/home/user/workspace/notes.txt")    # -> True
    engine.check_write("/etc/passwd")                      # -> raises PolicyViolationError
"""

from __future__ import annotations

import logging
from pathlib import Path

from missy.config.settings import FilesystemPolicy
from missy.core.events import AuditEvent, event_bus
from missy.core.exceptions import PolicyViolationError

logger = logging.getLogger(__name__)


class FilesystemPolicyEngine:
    """Evaluates filesystem access requests against a :class:`FilesystemPolicy`.

    Args:
        policy: The filesystem policy to enforce.
    """

    def __init__(self, policy: FilesystemPolicy) -> None:
        self._policy = policy

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def check_write(
        self,
        path: str | Path,
        session_id: str = "",
        task_id: str = "",
    ) -> bool:
        """Return ``True`` if writing to *path* is permitted by policy.

        The path is resolved to an absolute path (resolving any symlink
        components that already exist) before comparison.  This prevents
        symlink-traversal attacks where an attacker plants a symlink inside an
        allowed directory that points outside it.

        Args:
            path: The filesystem path the caller intends to write.
            session_id: Optional identifier of the calling session.
            task_id: Optional identifier of the calling task.

        Returns:
            ``True`` when the path is within an allowed write directory.

        Raises:
            PolicyViolationError: When the path is not permitted.
        """
        resolved = self._resolve(path)
        rule = self._is_under_path(resolved, self._policy.allowed_write_paths)
        if rule:
            self._emit_event("write", str(resolved), "allow", rule, session_id, task_id)
            return True

        self._emit_event("write", str(resolved), "deny", None, session_id, task_id)
        raise PolicyViolationError(
            f"Filesystem write denied: {str(path)!r} is not within an allowed write path.",
            category="filesystem",
            detail=(
                f"Resolved path {str(resolved)!r} is not under any allowed_write_paths entry."
            ),
        )

    def check_read(
        self,
        path: str | Path,
        session_id: str = "",
        task_id: str = "",
    ) -> bool:
        """Return ``True`` if reading from *path* is permitted by policy.

        Like :meth:`check_write`, the path is resolved before comparison.

        Args:
            path: The filesystem path the caller intends to read.
            session_id: Optional identifier of the calling session.
            task_id: Optional identifier of the calling task.

        Returns:
            ``True`` when the path is within an allowed read directory.

        Raises:
            PolicyViolationError: When the path is not permitted.
        """
        resolved = self._resolve(path)
        rule = self._is_under_path(resolved, self._policy.allowed_read_paths)
        if rule:
            self._emit_event("read", str(resolved), "allow", rule, session_id, task_id)
            return True

        self._emit_event("read", str(resolved), "deny", None, session_id, task_id)
        raise PolicyViolationError(
            f"Filesystem read denied: {str(path)!r} is not within an allowed read path.",
            category="filesystem",
            detail=(
                f"Resolved path {str(resolved)!r} is not under any allowed_read_paths entry."
            ),
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve(path: str | Path) -> Path:
        """Return an absolute, symlink-resolved :class:`Path`.

        Uses :meth:`Path.resolve` with ``strict=False`` so that paths
        referring to not-yet-existing files (e.g. a file about to be created)
        are still resolved without raising :exc:`FileNotFoundError`.

        Args:
            path: An absolute or relative path.

        Returns:
            An absolute :class:`Path` with symlinks resolved as far as
            possible.
        """
        return Path(path).resolve(strict=False)

    def _is_under_path(
        self,
        path: Path,
        allowed_paths: list[str],
    ) -> str | None:
        """Return the first entry in *allowed_paths* that contains *path*.

        The comparison checks that *path* is the allowed directory itself or
        is strictly nested inside it.  Trailing slashes on configured entries
        are stripped so that ``"/tmp/"`` and ``"/tmp"`` are treated
        equivalently.

        Args:
            path: An absolute, already-resolved :class:`Path`.
            allowed_paths: List of allowed directory path strings from the
                policy configuration.

        Returns:
            The matching entry string, or ``None`` if no entry matched.
        """
        for entry in allowed_paths:
            allowed = Path(entry).resolve(strict=False)
            # Path.is_relative_to is available from Python 3.9+; the project
            # requires 3.11, so this is always safe.
            try:
                if path == allowed or path.is_relative_to(allowed):
                    return str(allowed)
            except ValueError:
                # is_relative_to raises ValueError for paths on different
                # drives (Windows); harmless to skip on POSIX.
                continue
        return None

    def _emit_event(
        self,
        operation: str,
        path_str: str,
        result: str,
        rule: str | None,
        session_id: str,
        task_id: str,
    ) -> None:
        """Publish a filesystem audit event.

        Args:
            operation: ``"read"`` or ``"write"``.
            path_str: String representation of the resolved path.
            result: ``"allow"`` or ``"deny"``.
            rule: The matching allowed path, or ``None``.
            session_id: Calling session identifier.
            task_id: Calling task identifier.
        """
        event = AuditEvent.now(
            session_id=session_id,
            task_id=task_id,
            event_type=f"filesystem_{operation}",
            category="filesystem",
            result=result,  # type: ignore[arg-type]
            policy_rule=rule,
            detail={"path": path_str, "operation": operation},
        )
        event_bus.publish(event)
