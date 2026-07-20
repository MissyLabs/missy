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

import copy
import logging
import queue
import threading

from missy.core.events import AuditEvent, event_bus
from missy.core.exceptions import PolicyViolationError
from missy.policy.engine import get_policy_engine

from .base import BaseTool, ToolResult

logger = logging.getLogger(__name__)

_GENERIC_TOOL_ERROR = "Tool execution failed; sensitive details were withheld."
_RESOLVER_TIMEOUT_SECONDS = 0.5
_MAX_RESOLVED_TARGETS = 64


def _safe_text(value: object) -> str:
    """Return bounded censored text, failing closed on censor malfunction."""
    try:
        from missy.security.censor import censor_response

        result = censor_response(str(value))
        if not isinstance(result, str):
            raise TypeError("censor returned a non-string")
        return result[:2_000]
    except Exception:
        return _GENERIC_TOOL_ERROR


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
        self._lock = threading.RLock()
        self._retired = False
        self._active_calls = 0

    # ------------------------------------------------------------------
    # Mutation
    # ------------------------------------------------------------------

    def register(self, tool: BaseTool) -> None:
        """Add *tool* to the registry keyed by :attr:`~.base.BaseTool.name`.

        A previous registration under the same name is rejected. Re-registering
        the same object is an idempotent no-op.

        Args:
            tool: The tool instance to register.
        """
        if not isinstance(tool, BaseTool):
            raise TypeError("tool must be a BaseTool instance")
        if not isinstance(tool.name, str) or not tool.name:
            raise ValueError("tool name must be a non-empty string")
        with self._lock:
            if self._retired:
                raise RuntimeError("Cannot register tools on a retired registry.")
            existing = self._tools.get(tool.name)
            if existing is tool:
                return
            if existing is not None:
                raise ValueError(f"Tool {tool.name!r} is already registered")
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
        with self._lock:
            return self._tools.get(name)

    def list_tools(self) -> list[str]:
        """Return a sorted list of all registered tool names.

        Includes disabled tools; use :meth:`is_enabled` to check state.

        Returns:
            A new list of string keys in ascending alphabetical order.
        """
        with self._lock:
            return sorted(self._tools)

    def list_disabled_tools(self) -> list[str]:
        """Return a sorted list of tool names currently disabled by an operator.

        Returns:
            A new list of string keys in ascending alphabetical order.
        """
        with self._lock:
            return sorted(self._disabled)

    def is_enabled(self, name: str) -> bool:
        """Return whether *name* is currently enabled for exposure/execution.

        Unregistered names are never reported as enabled.

        Args:
            name: Registry key to check.

        Returns:
            ``False`` if an operator has disabled the tool, ``True`` otherwise.
        """
        with self._lock:
            return name in self._tools and name not in self._disabled

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
        with self._lock:
            if self._retired:
                raise RuntimeError("Cannot disable tools on a retired registry.")
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
        with self._lock:
            if self._retired:
                raise RuntimeError("Cannot enable tools on a retired registry.")
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
        /,
        session_id: str = "",
        task_id: str = "",
        **kwargs,
    ) -> ToolResult:
        """Execute through this registry unless it has been superseded."""
        with self._lock:
            if self._retired:
                return ToolResult(
                    success=False,
                    output=None,
                    error="Tool registry was replaced; retry against the current registry.",
                )
            self._active_calls += 1
        try:
            return self._execute_active(
                tool_name,
                session_id=session_id,
                task_id=task_id,
                **kwargs,
            )
        finally:
            with self._lock:
                self._active_calls -= 1

    def retire(self) -> None:
        """Revoke new execution and mutation while allowing active calls to finish."""
        with self._lock:
            self._retired = True

    def _execute_active(
        self,
        tool_name: str,
        /,
        session_id: str = "",
        task_id: str = "",
        **kwargs,
    ) -> ToolResult:
        """Execute the named tool after verifying permissions.

        ``tool_name`` is positional-only (the ``/`` above) so that a tool
        whose own parameter schema happens to define an argument literally
        named ``tool_name`` -- e.g. ``SelfCreateTool``, whose ``tool_name``
        is the name of the *new* tool being proposed, unrelated to which
        tool is being invoked here -- can never collide with it. Before
        this, ``registry.execute(tool_call.name, **tool_args)`` with
        ``tool_args = {"tool_name": ..., ...}`` raised
        ``TypeError: execute() got multiple values for argument
        'tool_name'`` on every single call, since Python's own argument
        binding conflates the caller's positional ``tool_name`` value with
        the tool's own keyword-supplied ``tool_name`` before this method
        body ever runs -- making ``self_create_tool`` completely
        uncallable for its ``create``/``delete`` actions (which require
        ``tool_name``) regardless of any policy or permission check.
        ``session_id``/``task_id`` remain keyword-assignable as before;
        no current tool defines a parameter with either name, and the one
        real caller (``AgentRuntime._execute_tool()``) always passes them
        as explicit keywords.

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
        with self._lock:
            tool = self._tools.get(tool_name)
            disabled = tool_name in self._disabled
        if tool is None:
            raise KeyError(f"No tool registered under the name {tool_name!r}.")

        if disabled:
            logger.warning("Execution denied for disabled tool %r.", tool_name)
            self._emit_event(tool_name, session_id, task_id, "deny", "Tool disabled by operator")
            return ToolResult(
                success=False,
                output=None,
                error=f"Tool {tool_name!r} is disabled by operator.",
            )

        # The policy resolvers, audit, and implementation must consume one
        # immutable snapshot. Mutable provider objects cannot swap a checked
        # benign target for a different target after authorization.
        try:
            argument_snapshot = {
                key: value if key.startswith("_") else copy.deepcopy(value)
                for key, value in kwargs.items()
            }
        except Exception:
            error = "Tool arguments could not be safely snapshotted."
            self._emit_event(tool_name, session_id, task_id, "error", error)
            return ToolResult(success=False, output=None, error=error)

        # Policy checks - failures surface as ToolResult(success=False)
        # rather than raised exceptions so the agent can handle them gracefully.
        try:
            self._check_permissions(tool, session_id, task_id, argument_snapshot)
        except PolicyViolationError as exc:
            logger.warning("Policy denied execution of tool %r: %s", tool_name, exc)
            safe_error = _safe_text(exc)
            self._emit_event(tool_name, session_id, task_id, "deny", safe_error)
            return ToolResult(success=False, output=None, error=safe_error, policy_denied=True)
        except Exception as exc:
            safe_error = _safe_text(exc)
            logger.warning("Policy resolution failed for tool %r; execution denied.", tool_name)
            self._emit_event(tool_name, session_id, task_id, "error", safe_error)
            return ToolResult(success=False, output=None, error=safe_error)

        # Strip registry-internal keys that tools don't accept.
        tool_kwargs = {
            k: v for k, v in argument_snapshot.items() if k not in ("session_id", "task_id")
        }
        try:
            result = tool.execute(**tool_kwargs)
        except Exception as exc:
            safe_error = _safe_text(exc)
            logger.error("Tool %r raised an exception; details withheld.", tool_name)
            self._emit_event(tool_name, session_id, task_id, "error", safe_error)
            return ToolResult(success=False, output=None, error=safe_error)
        if not isinstance(result, ToolResult):
            error = "Tool returned an invalid result type."
            self._emit_event(tool_name, session_id, task_id, "error", error)
            return ToolResult(success=False, output=None, error=error)

        event_result = "allow" if result.success else "error"
        if not result.success and result.error:
            result = ToolResult(
                success=False,
                output=None,
                error=_safe_text(result.error),
                policy_denied=result.policy_denied,
            )
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
                hosts = _validate_target_list(
                    _call_policy_resolver(tool.resolve_network_hosts, _kw),
                    kind="network host",
                )
                for host in hosts:
                    engine.check_network(host, session_id=session_id, task_id=task_id)

        resolves_fs_targets = (
            type(tool).resolve_filesystem_targets is not BaseTool.resolve_filesystem_targets
        )
        resolved_read_paths: list[str] = []
        resolved_write_paths: list[str] = []
        if resolves_fs_targets and (perms.filesystem_read or perms.filesystem_write):
            resolved = _call_policy_resolver(tool.resolve_filesystem_targets, _kw)
            if not isinstance(resolved, (tuple, list)) or len(resolved) != 2:
                raise ValueError(
                    "Filesystem resolver must return a (read_paths, write_paths) pair."
                )
            resolved_read_paths = _validate_target_list(resolved[0], kind="read path")
            resolved_write_paths = _validate_target_list(resolved[1], kind="write path")

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
                resolved_command = _call_policy_resolver(tool.resolve_shell_command, _kw)
                if resolved_command is not None and not isinstance(
                    resolved_command, (str, list, tuple)
                ):
                    raise ValueError("Shell resolver must return text, an argv list, or None.")
                if isinstance(resolved_command, (list, tuple)):
                    if not 1 <= len(resolved_command) <= 256 or any(
                        not _safe_resolver_text(item) for item in resolved_command
                    ):
                        raise ValueError("Shell resolver returned invalid or unbounded argv.")
                    resolved_command = list(resolved_command)
                elif isinstance(resolved_command, str) and not _safe_resolver_text(
                    resolved_command
                ):
                    raise ValueError("Shell resolver returned invalid or unbounded command text.")
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
            safe_msg = _safe_text(detail_msg) if detail_msg else ""
            event = AuditEvent.now(
                session_id=session_id,
                task_id=task_id,
                event_type="tool_execute",
                category="tool",
                result=result,  # type: ignore[arg-type]
                detail={"tool": tool_name, "message": safe_msg},
            )
            event_bus.publish(event)
        except Exception:
            logger.exception("Failed to emit audit event for tool %r", tool_name)


def _safe_resolver_text(value: object) -> bool:
    return (
        isinstance(value, str)
        and 0 < len(value) <= 4096
        and all(char.isprintable() and char not in "\r\n\x00" for char in value)
    )


def _validate_target_list(value: object, *, kind: str) -> list[str]:
    if not isinstance(value, (list, tuple)) or len(value) > _MAX_RESOLVED_TARGETS:
        raise ValueError(f"Policy resolver returned invalid or unbounded {kind} targets.")
    targets = list(value)
    if any(not _safe_resolver_text(item) for item in targets) or len(set(targets)) != len(targets):
        raise ValueError(f"Policy resolver returned invalid or duplicate {kind} targets.")
    return targets


def _call_policy_resolver(resolver: object, kwargs: dict) -> object:
    """Run one resolver against an isolated snapshot with a hard caller deadline."""
    resolver_args = {
        key: copy.deepcopy(value) for key, value in kwargs.items() if not key.startswith("_")
    }
    before = copy.deepcopy(resolver_args)
    completed: queue.Queue[tuple[bool, object, bool]] = queue.Queue(maxsize=1)

    def invoke() -> None:
        try:
            value = resolver(resolver_args)  # type: ignore[operator]
            try:
                mutated = resolver_args != before
            except Exception:
                mutated = True
            completed.put((True, value, mutated))
        except BaseException as exc:  # noqa: BLE001 - marshal to caller
            completed.put((False, exc, False))

    threading.Thread(target=invoke, daemon=True, name="missy-policy-resolver").start()
    try:
        ok, value, mutated = completed.get(timeout=_RESOLVER_TIMEOUT_SECONDS)
    except queue.Empty as exc:
        raise TimeoutError("Policy resolver exceeded its bounded execution deadline.") from exc
    if not ok:
        if isinstance(value, Exception):
            raise value
        raise RuntimeError("Policy resolver terminated abnormally.")
    if mutated:
        raise ValueError("Policy resolver mutated its argument snapshot.")
    return value


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
        previous = _registry
        _registry = registry
    if previous is not None:
        previous.retire()
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
