"""Tool registry for the Missy framework.

The :class:`ToolRegistry` manages a named collection of :class:`~.base.BaseTool`
instances and acts as the single execution gateway: policy checks are
performed here before any tool's :meth:`~.base.BaseTool.execute` method is
called, and audit events are emitted for every invocation.

Example::

    from missy.tools.registry import init_tool_registry, get_tool_registry
    from missy.tools.builtin.calculator import CalculatorTool

    registry = init_tool_registry()
    registry.register(CalculatorTool())
    result = registry.execute("calculator", expression="2 + 2")
"""

from __future__ import annotations

import logging
import threading
from typing import Optional

from missy.core.events import AuditEvent, event_bus
from missy.core.exceptions import PolicyViolationError, ProviderError
from missy.policy.engine import get_policy_engine

from .base import BaseTool, ToolResult

logger = logging.getLogger(__name__)


class ToolRegistry:
    """Registry that manages and executes :class:`~.base.BaseTool` instances.

    Before executing a tool, the registry validates the tool's declared
    :class:`~.base.ToolPermissions` against the active
    :class:`~missy.policy.engine.PolicyEngine`.  Audit events are emitted
    for every invocation attempt regardless of outcome.
    """

    def __init__(self) -> None:
        self._tools: dict[str, BaseTool] = {}

    # ------------------------------------------------------------------
    # Mutation
    # ------------------------------------------------------------------

    def register(self, tool: BaseTool) -> None:
        """Add *tool* to the registry keyed by :attr:`~.base.BaseTool.name`.

        A previous registration under the same name is silently replaced.

        Args:
            tool: The tool instance to register.
        """
        self._tools[tool.name] = tool
        logger.debug("Registered tool %r (%s).", tool.name, type(tool).__name__)

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def get(self, name: str) -> Optional[BaseTool]:
        """Return the tool registered under *name*, or ``None``.

        Args:
            name: Registry key to look up.

        Returns:
            The :class:`~.base.BaseTool` instance, or ``None``.
        """
        return self._tools.get(name)

    def list_tools(self) -> list[str]:
        """Return a sorted list of all registered tool names.

        Returns:
            A new list of string keys in ascending alphabetical order.
        """
        return sorted(self._tools)

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    def execute(
        self,
        name: str,
        session_id: str = "",
        task_id: str = "",
        **kwargs,
    ) -> ToolResult:
        """Execute the named tool after verifying permissions.

        Policy checks are performed in this order:

        1. Network policy - if ``tool.permissions.network`` is ``True``, each
           host in ``tool.permissions.allowed_hosts`` is checked.
        2. Filesystem read policy - if ``tool.permissions.filesystem_read``
           is ``True``, each path in ``tool.permissions.allowed_paths`` is
           checked.
        3. Filesystem write policy - if ``tool.permissions.filesystem_write``
           is ``True``, each path in ``tool.permissions.allowed_paths`` is
           checked.
        4. Shell policy - if ``tool.permissions.shell`` is ``True``, the
           engine's shell policy is consulted.

        An audit event is emitted after every execution attempt.

        Args:
            name: Registry key of the tool to run.
            session_id: Forwarded to audit events and policy checks.
            task_id: Forwarded to audit events and policy checks.
            **kwargs: Keyword arguments forwarded verbatim to the tool's
                :meth:`~.base.BaseTool.execute` method.

        Returns:
            A :class:`~.base.ToolResult` from the tool, or a failure result
            when a policy check or unexpected exception occurs.

        Raises:
            KeyError: When *name* is not registered.
        """
        tool = self._tools.get(name)
        if tool is None:
            raise KeyError(f"No tool registered under the name {name!r}.")

        # Policy checks - failures surface as ToolResult(success=False)
        # rather than raised exceptions so the agent can handle them gracefully.
        try:
            self._check_permissions(tool, session_id, task_id, kwargs)
        except PolicyViolationError as exc:
            logger.warning("Policy denied execution of tool %r: %s", name, exc)
            self._emit_event(name, session_id, task_id, "deny", str(exc))
            return ToolResult(success=False, output=None, error=str(exc))

        # Strip registry-internal keys that tools don't accept.
        tool_kwargs = {k: v for k, v in kwargs.items()
                       if k not in ("session_id", "task_id")}
        try:
            result = tool.execute(**tool_kwargs)
        except Exception as exc:
            logger.exception("Tool %r raised an unhandled exception.", name)
            self._emit_event(name, session_id, task_id, "error", str(exc))
            return ToolResult(success=False, output=None, error=str(exc))

        event_result = "allow" if result.success else "error"
        self._emit_event(name, session_id, task_id, event_result, result.error or "")
        return result

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _check_permissions(
        self,
        tool: BaseTool,
        session_id: str,
        task_id: str,
        kwargs: dict | None = None,
    ) -> None:
        """Run policy engine checks for the tool's declared permissions.

        Args:
            tool: The tool whose permissions should be validated.
            session_id: Forwarded to the policy engine.
            task_id: Forwarded to the policy engine.

        Raises:
            PolicyViolationError: When any required permission is denied.
        """
        # Policy engine may not be initialised in all test contexts; skip
        # gracefully if it is not available.
        try:
            engine = get_policy_engine()
        except RuntimeError:
            logger.debug(
                "PolicyEngine not initialised; skipping permission checks for tool %r.",
                tool.name,
            )
            return

        perms = tool.permissions

        if perms.network:
            for host in perms.allowed_hosts:
                engine.check_network(host, session_id=session_id, task_id=task_id)

        if perms.filesystem_read:
            for path in perms.allowed_paths:
                engine.check_read(path, session_id=session_id, task_id=task_id)

        if perms.filesystem_write:
            for path in perms.allowed_paths:
                engine.check_write(path, session_id=session_id, task_id=task_id)

        if perms.shell:
            # Pass the actual command so the policy engine can check it.
            command = (kwargs or {}).get("command", "shell")
            engine.check_shell(command, session_id=session_id, task_id=task_id)

    def _emit_event(
        self,
        tool_name: str,
        session_id: str,
        task_id: str,
        result: str,
        detail_msg: str,
    ) -> None:
        """Publish a tool audit event to the global event bus.

        Args:
            tool_name: The name of the tool being executed.
            session_id: Calling session identifier.
            task_id: Calling task identifier.
            result: One of ``"allow"``, ``"deny"``, or ``"error"``.
            detail_msg: Human-readable description.
        """
        try:
            event = AuditEvent.now(
                session_id=session_id,
                task_id=task_id,
                event_type="tool_execute",
                category="plugin",
                result=result,  # type: ignore[arg-type]
                detail={"tool": tool_name, "message": detail_msg},
            )
            event_bus.publish(event)
        except Exception:
            logger.exception("Failed to emit audit event for tool %r", tool_name)


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_registry: Optional[ToolRegistry] = None
_lock: threading.Lock = threading.Lock()


def init_tool_registry() -> ToolRegistry:
    """Create and install a fresh process-level :class:`ToolRegistry`.

    Subsequent calls replace the existing registry atomically.

    Returns:
        The newly installed :class:`ToolRegistry` (empty; register tools
        separately).
    """
    global _registry
    registry = ToolRegistry()
    with _lock:
        _registry = registry
    return registry


def get_tool_registry() -> ToolRegistry:
    """Return the process-level :class:`ToolRegistry`.

    Returns:
        The currently installed :class:`ToolRegistry`.

    Raises:
        RuntimeError: When :func:`init_tool_registry` has not yet been called.
    """
    with _lock:
        registry = _registry
    if registry is None:
        raise RuntimeError(
            "ToolRegistry has not been initialised. "
            "Call missy.tools.registry.init_tool_registry() first."
        )
    return registry
