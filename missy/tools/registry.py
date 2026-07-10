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

from missy.core.events import AuditEvent, event_bus
from missy.core.exceptions import PolicyViolationError
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
        self._disabled: set[str] = set()

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

    def get(self, name: str) -> BaseTool | None:
        """Return the tool registered under *name*, or ``None``.

        Args:
            name: Registry key to look up.

        Returns:
            The :class:`~.base.BaseTool` instance, or ``None``.
        """
        return self._tools.get(name)

    def list_tools(self) -> list[str]:
        """Return a sorted list of all registered tool names.

        Includes disabled tools; use :meth:`is_enabled` to check state.

        Returns:
            A new list of string keys in ascending alphabetical order.
        """
        return sorted(self._tools)

    def list_disabled_tools(self) -> list[str]:
        """Return a sorted list of tool names currently disabled by an operator.

        Returns:
            A new list of string keys in ascending alphabetical order.
        """
        return sorted(self._disabled)

    def is_enabled(self, name: str) -> bool:
        """Return whether *name* is currently enabled for exposure/execution.

        Unregistered names are reported as enabled (there is nothing to
        disable); callers should check :meth:`get` separately if they need
        to distinguish "unknown" from "enabled".

        Args:
            name: Registry key to check.

        Returns:
            ``False`` if an operator has disabled the tool, ``True`` otherwise.
        """
        return name not in self._disabled

    # ------------------------------------------------------------------
    # Operator enable/disable
    # ------------------------------------------------------------------

    def disable(self, name: str) -> None:
        """Mark a registered tool as disabled.

        Disabled tools are excluded from the tool schemas exposed to the
        model (see :meth:`~missy.agent.runtime.AgentRuntime._get_tools`) and
        :meth:`execute` refuses to run them, so this is a defense-in-depth
        control on both the "can the model call it" and "will it actually
        run" axes.

        Args:
            name: Registry key of the tool to disable.

        Raises:
            KeyError: When *name* is not registered.
        """
        if name not in self._tools:
            raise KeyError(f"No tool registered under the name {name!r}.")
        self._disabled.add(name)
        logger.info("Tool %r disabled by operator.", name)

    def enable(self, name: str) -> None:
        """Clear a previous :meth:`disable` for *name*.

        Args:
            name: Registry key of the tool to re-enable.

        Raises:
            KeyError: When *name* is not registered.
        """
        if name not in self._tools:
            raise KeyError(f"No tool registered under the name {name!r}.")
        self._disabled.discard(name)
        logger.info("Tool %r re-enabled by operator.", name)

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    def execute(
        self,
        tool_name: str,
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
            KeyError: When *tool_name* is not registered.
        """
        tool = self._tools.get(tool_name)
        if tool is None:
            raise KeyError(f"No tool registered under the name {tool_name!r}.")

        if tool_name in self._disabled:
            logger.warning("Execution denied for disabled tool %r.", tool_name)
            self._emit_event(tool_name, session_id, task_id, "deny", "Tool disabled by operator")
            return ToolResult(
                success=False,
                output=None,
                error=f"Tool {tool_name!r} is disabled by operator.",
            )

        # Policy checks - failures surface as ToolResult(success=False)
        # rather than raised exceptions so the agent can handle them gracefully.
        try:
            self._check_permissions(tool, session_id, task_id, kwargs)
        except PolicyViolationError as exc:
            logger.warning("Policy denied execution of tool %r: %s", tool_name, exc)
            self._emit_event(tool_name, session_id, task_id, "deny", str(exc))
            return ToolResult(success=False, output=None, error=str(exc))

        # Strip registry-internal keys that tools don't accept.
        tool_kwargs = {k: v for k, v in kwargs.items() if k not in ("session_id", "task_id")}
        try:
            result = tool.execute(**tool_kwargs)
        except Exception as exc:
            logger.exception("Tool %r raised an unhandled exception.", tool_name)
            self._emit_event(tool_name, session_id, task_id, "error", str(exc))
            return ToolResult(success=False, output=None, error=str(exc))

        event_result = "allow" if result.success else "error"
        self._emit_event(tool_name, session_id, task_id, event_result, result.error or "")
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
        perms = tool.permissions

        # If the tool declares no permissions, no policy checks are needed.
        needs_policy = (
            perms.network or perms.shell or perms.filesystem_read or perms.filesystem_write
        )
        if not needs_policy:
            return

        # Policy engine must be initialised — fail closed if it is not.
        try:
            engine = get_policy_engine()
        except RuntimeError as err:
            logger.warning(
                "PolicyEngine not initialised; DENYING tool %r (fail-closed).",
                tool.name,
            )
            raise PolicyViolationError(
                f"Tool {tool.name!r} denied: policy engine not initialised.",
                category="security",
                detail="Policy engine must be initialised before tool execution.",
            ) from err

        # SR-1.4/SR-1.5/SR-1.6: a tool that overrides resolve_network_hosts /
        # resolve_filesystem_targets / resolve_shell_command is declaring the
        # real operation it performs, rather than relying on the registry's
        # generic kwarg-name heuristic (or, for network, the registry's
        # static-only allowed_hosts check) -- several first-party tools'
        # actual kwargs/targets don't match those defaults, so enforcement
        # was silently skipped instead of failing closed.
        _kw = kwargs or {}

        if perms.network:
            for host in perms.allowed_hosts:
                engine.check_network(host, session_id=session_id, task_id=task_id)
            if type(tool).resolve_network_hosts is not BaseTool.resolve_network_hosts:
                for host in tool.resolve_network_hosts(_kw):
                    engine.check_network(host, session_id=session_id, task_id=task_id)

        resolves_fs_targets = (
            type(tool).resolve_filesystem_targets is not BaseTool.resolve_filesystem_targets
        )
        resolved_read_paths: list[str] = []
        resolved_write_paths: list[str] = []
        if resolves_fs_targets and (perms.filesystem_read or perms.filesystem_write):
            resolved_read_paths, resolved_write_paths = tool.resolve_filesystem_targets(_kw)

        if perms.filesystem_read:
            # Check statically declared allowed_paths …
            for path in perms.allowed_paths:
                engine.check_read(path, session_id=session_id, task_id=task_id)
            # … and the actual path from tool kwargs (H2 fix: enforce policy
            # on the real target path, not just the tool's static declarations).
            if resolves_fs_targets:
                for path in resolved_read_paths:
                    engine.check_read(path, session_id=session_id, task_id=task_id)
            else:
                # Generic heuristic for tools without an explicit resolver:
                # check common path parameter names for defense-in-depth.
                for path_key in ("path", "file_path", "target", "destination"):
                    actual_path = _kw.get(path_key)
                    if actual_path:
                        engine.check_read(actual_path, session_id=session_id, task_id=task_id)

        if perms.filesystem_write:
            for path in perms.allowed_paths:
                engine.check_write(path, session_id=session_id, task_id=task_id)
            if resolves_fs_targets:
                for path in resolved_write_paths:
                    engine.check_write(path, session_id=session_id, task_id=task_id)
            else:
                for path_key in ("path", "file_path", "target", "destination"):
                    actual_path = _kw.get(path_key)
                    if actual_path:
                        engine.check_write(actual_path, session_id=session_id, task_id=task_id)

        if perms.shell:
            if type(tool).resolve_shell_command is not BaseTool.resolve_shell_command:
                # Tool declares the real host command it invokes — use that
                # instead of guessing from a generic "command" kwarg that may
                # not exist, or may refer to something other than the host
                # program actually executed (e.g. a sandboxed guest command).
                resolved_command = tool.resolve_shell_command(_kw)
                command = resolved_command if resolved_command is not None else "shell"
            else:
                # Pass the actual command so the policy engine can check it.
                command = _kw.get("command", "shell")
            # Some tools may pass command as a list; convert to string for policy check.
            if isinstance(command, list):
                command = " ".join(str(c) for c in command)
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
            # Redact potential secrets from audit event detail messages.
            safe_msg = detail_msg
            try:
                from missy.security.censor import censor_response

                safe_msg = censor_response(detail_msg)
            except Exception:
                logger.debug("Censor import failed; using raw detail message")
            event = AuditEvent.now(
                session_id=session_id,
                task_id=task_id,
                event_type="tool_execute",
                category="plugin",
                result=result,  # type: ignore[arg-type]
                detail={"tool": tool_name, "message": safe_msg},
            )
            event_bus.publish(event)
        except Exception:
            logger.exception("Failed to emit audit event for tool %r", tool_name)


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_registry: ToolRegistry | None = None
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
