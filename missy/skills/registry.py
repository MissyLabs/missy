"""Skill registry for the Missy framework.

:class:`SkillRegistry` manages a named collection of :class:`~.base.BaseSkill`
instances.  Every execution attempt — successful or otherwise — emits an audit
event through :data:`~missy.core.events.event_bus`.

Example::

    from missy.skills.registry import init_skill_registry, get_skill_registry
    from missy.skills.base import BaseSkill, SkillPermissions, SkillResult

    class EchoSkill(BaseSkill):
        name = "echo"
        description = "Return the input text."
        permissions = SkillPermissions()

        def execute(self, *, text: str = "") -> SkillResult:
            return SkillResult(success=True, output=text)

    registry = init_skill_registry(
        permission_authorizer=lambda name, permissions: True,
    )
    registry.register(EchoSkill())
    result = registry.execute("echo", text="hello")
"""

from __future__ import annotations

import copy
import logging
import re
import threading
from collections.abc import Callable
from typing import Any

from missy.core.events import AuditEvent, event_bus

from .base import BaseSkill, SkillPermissions, SkillResult

logger = logging.getLogger(__name__)

_SKILL_NAME_RE = re.compile(r"^[a-z][a-z0-9_]{0,63}$")
_VERSION_RE = re.compile(r"^(?:0|[1-9][0-9]*)\.(?:0|[1-9][0-9]*)\.(?:0|[1-9][0-9]*)$")
_MAX_DESCRIPTION_CHARS = 500
_GENERIC_SKILL_ERROR = "Skill execution failed; sensitive details were withheld."
PermissionAuthorizer = Callable[[str, SkillPermissions], bool]


def _safe_text(value: object) -> str:
    """Return bounded censored text, failing closed if censoring malfunctions."""
    try:
        from missy.security.censor import censor_response

        result = censor_response(str(value))
        if not isinstance(result, str):
            raise TypeError("censor returned a non-string")
        return result[:2_000]
    except Exception:
        return _GENERIC_SKILL_ERROR


def _validate_metadata(skill: BaseSkill) -> None:
    name = getattr(skill, "name", None)
    if not isinstance(name, str) or not _SKILL_NAME_RE.fullmatch(name):
        raise ValueError("Skill name must match ^[a-z][a-z0-9_]{0,63}$ (lowercase ASCII identity)")
    description = getattr(skill, "description", None)
    if (
        not isinstance(description, str)
        or len(description) > _MAX_DESCRIPTION_CHARS
        or any(
            ord(ch) < 32 or ch in "\x7f\u202a\u202b\u202d\u202e\u2066\u2067\u2068\u2069"
            for ch in description
        )
    ):
        raise ValueError("Skill description must be bounded, single-line, inert text")
    version = getattr(skill, "version", None)
    if not isinstance(version, str) or not _VERSION_RE.fullmatch(version):
        raise ValueError("Skill version must be a canonical MAJOR.MINOR.PATCH value")
    permissions = getattr(skill, "permissions", None)
    if not isinstance(permissions, SkillPermissions):
        raise TypeError("Skill permissions must be a SkillPermissions value")


class SkillRegistry:
    """Registry that manages and executes :class:`~.base.BaseSkill` instances.

    Audit events with category ``"skill"`` are emitted for every execution
    attempt so that skill invocations appear in the unified audit trail. A
    registry without an explicit permission authorizer denies execution.
    """

    def __init__(self, permission_authorizer: PermissionAuthorizer | None = None) -> None:
        self._skills: dict[str, BaseSkill] = {}
        self._active: dict[str, int] = {}
        self._lock = threading.RLock()
        self._execution_local = threading.local()
        self._permission_authorizer = permission_authorizer

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register(self, skill: BaseSkill) -> None:
        """Add *skill* to the registry keyed by :attr:`~.base.BaseSkill.name`.

        A previous registration under the same name is rejected. Re-registering
        the same object is an idempotent no-op.

        Args:
            skill: The skill instance to register.
        """
        if not isinstance(skill, BaseSkill):
            raise TypeError("skill must be a BaseSkill instance")
        _validate_metadata(skill)
        with self._lock:
            existing = self._skills.get(skill.name)
            if existing is skill:
                return
            if existing is not None:
                raise ValueError(f"Skill {skill.name!r} is already registered")
            self._skills[skill.name] = skill
        logger.debug("Registered skill %r (%s).", skill.name, type(skill).__name__)

    def unregister(self, name: str) -> None:
        """Remove an idle skill atomically; active executions must quiesce first."""
        with self._lock:
            if name not in self._skills:
                raise KeyError(f"No skill registered under the name {name!r}.")
            if self._active.get(name, 0):
                raise RuntimeError(f"Skill {name!r} has active executions")
            del self._skills[name]

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def get(self, name: str) -> BaseSkill | None:
        """Return the skill registered under *name*, or ``None``.

        Args:
            name: Registry key to look up.

        Returns:
            The :class:`~.base.BaseSkill` instance, or ``None`` if not found.
        """
        with self._lock:
            return self._skills.get(name)

    def list_skills(self) -> list[str]:
        """Return a sorted list of all registered skill names.

        Returns:
            A new list of string keys in ascending alphabetical order.
        """
        with self._lock:
            return sorted(self._skills)

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    def execute(
        self,
        name: str,
        session_id: str = "",
        task_id: str = "",
        **kwargs,
    ) -> SkillResult:
        """Execute the named skill and emit audit events.

        Args:
            name: Registry key of the skill to run.
            session_id: Forwarded to audit events.
            task_id: Forwarded to audit events.
            **kwargs: Keyword arguments forwarded verbatim to the skill's
                :meth:`~.base.BaseSkill.execute` method.

        Returns:
            A :class:`~.base.SkillResult` from the skill, or a failure result
            when the skill is not found or raises an unhandled exception.
        """
        stack = list(getattr(self._execution_local, "stack", ()))
        if name in stack or len(stack) >= 16:
            error_msg = f"Recursive skill execution denied for {name!r}."
            self._emit_event(name, session_id, task_id, "error", error_msg)
            return SkillResult(success=False, output=None, error=error_msg)
        stack.append(name)
        self._execution_local.stack = stack

        with self._lock:
            skill = self._skills.get(name)
            if skill is not None:
                self._active[name] = self._active.get(name, 0) + 1
        if skill is None:
            error_msg = f"No skill registered under the name {name!r}."
            logger.warning(error_msg)
            self._emit_event(
                skill_name=name,
                session_id=session_id,
                task_id=task_id,
                result="error",
                detail_msg=error_msg,
            )
            stack.pop()
            self._execution_local.stack = stack
            return SkillResult(success=False, output=None, error=error_msg)

        try:
            if not self._permission_allowed(name, skill.permissions):
                error_msg = "Skill execution denied by permission policy."
                self._emit_event(name, session_id, task_id, "deny", error_msg, skill=skill)
                return SkillResult(success=False, output=None, error=error_msg)
            try:
                call_kwargs: dict[str, Any] = copy.deepcopy(kwargs)
            except Exception:
                error_msg = "Skill arguments could not be safely snapshotted."
                self._emit_event(name, session_id, task_id, "error", error_msg, skill=skill)
                return SkillResult(success=False, output=None, error=error_msg)
            skill_result = skill.execute(**call_kwargs)
            if not isinstance(skill_result, SkillResult):
                raise TypeError("Skill returned an invalid result type")
        except Exception as exc:
            error_msg = _safe_text(exc)
            logger.error("Skill %r raised an exception; details withheld.", name)
            self._emit_event(
                skill_name=name,
                session_id=session_id,
                task_id=task_id,
                result="error",
                detail_msg=error_msg,
            )
            return SkillResult(success=False, output=None, error=error_msg)
        finally:
            stack.pop()
            self._execution_local.stack = stack
            with self._lock:
                remaining = self._active.get(name, 1) - 1
                if remaining > 0:
                    self._active[name] = remaining
                else:
                    self._active.pop(name, None)

        event_result = "allow" if skill_result.success else "error"
        if not skill_result.success:
            # Failure output is not a second public error channel. Built-ins
            # may use it internally for diagnostics, but registry publication
            # normalises it away and censors the bounded error string.
            skill_result = SkillResult(
                success=False,
                output=None,
                error=_safe_text(skill_result.error or "Skill returned failure."),
            )
        self._emit_event(
            skill_name=name,
            session_id=session_id,
            task_id=task_id,
            result=event_result,
            detail_msg=skill_result.error or "",
            skill=skill,
        )
        return skill_result

    def _permission_allowed(self, name: str, permissions: SkillPermissions) -> bool:
        """Fail closed unless an explicit authorizer returns literal ``True``."""
        authorizer = self._permission_authorizer
        if authorizer is None:
            return False
        try:
            return authorizer(name, permissions) is True
        except Exception:
            logger.exception("Skill permission authorizer failed for %r", name)
            return False

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _emit_event(
        self,
        skill_name: str,
        session_id: str,
        task_id: str,
        result: str,
        detail_msg: str,
        skill: BaseSkill | None = None,
    ) -> None:
        """Publish a skill audit event to the global event bus.

        Args:
            skill_name: The name of the skill being executed.
            session_id: Calling session identifier.
            task_id: Calling task identifier.
            result: One of ``"allow"``, ``"deny"``, or ``"error"``.
            detail_msg: Human-readable description.
        """
        try:
            safe_msg = _safe_text(detail_msg) if detail_msg else ""
            event = AuditEvent.now(
                session_id=session_id,
                task_id=task_id,
                event_type="skill.execute",
                category="skill",
                result=result,  # type: ignore[arg-type]
                detail={
                    "skill": skill_name,
                    "message": safe_msg,
                    "subsystem": "skill",
                    "implementation": type(skill).__qualname__ if skill is not None else None,
                    "version": getattr(skill, "version", None),
                    "origin": getattr(type(skill), "__module__", None)
                    if skill is not None
                    else None,
                },
            )
            event_bus.publish(event)
        except Exception:
            logger.exception("Failed to emit audit event for skill %r", skill_name)


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_registry: SkillRegistry | None = None
_lock: threading.Lock = threading.Lock()


def init_skill_registry(
    permission_authorizer: PermissionAuthorizer | None = None,
) -> SkillRegistry:
    """Create and install a fresh process-level :class:`SkillRegistry`.

    Subsequent calls replace the existing registry atomically under a lock.

    Returns:
        The newly installed :class:`SkillRegistry` (empty; register skills
        separately via :meth:`~SkillRegistry.register`).
    """
    global _registry
    registry = SkillRegistry(permission_authorizer=permission_authorizer)
    with _lock:
        _registry = registry
    return registry


def get_skill_registry() -> SkillRegistry:
    """Return the process-level :class:`SkillRegistry`.

    Returns:
        The currently installed :class:`SkillRegistry`.

    Raises:
        RuntimeError: When :func:`init_skill_registry` has not yet been called.
    """
    with _lock:
        registry = _registry
    if registry is None:
        raise RuntimeError(
            "SkillRegistry has not been initialised. "
            "Call missy.skills.registry.init_skill_registry() first."
        )
    return registry
