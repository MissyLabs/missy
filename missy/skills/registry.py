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

    registry = init_skill_registry()
    registry.register(EchoSkill())
    result = registry.execute("echo", text="hello")
"""

from __future__ import annotations

import logging
import threading
from typing import Optional

from missy.core.events import AuditEvent, event_bus

from .base import BaseSkill, SkillResult

logger = logging.getLogger(__name__)


class SkillRegistry:
    """Registry that manages and executes :class:`~.base.BaseSkill` instances.

    Audit events with category ``"plugin"`` are emitted for every execution
    attempt so that skill invocations appear in the unified audit trail.
    """

    def __init__(self) -> None:
        self._skills: dict[str, BaseSkill] = {}

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register(self, skill: BaseSkill) -> None:
        """Add *skill* to the registry keyed by :attr:`~.base.BaseSkill.name`.

        A previous registration under the same name is silently replaced.

        Args:
            skill: The skill instance to register.
        """
        self._skills[skill.name] = skill
        logger.debug("Registered skill %r (%s).", skill.name, type(skill).__name__)

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def get(self, name: str) -> Optional[BaseSkill]:
        """Return the skill registered under *name*, or ``None``.

        Args:
            name: Registry key to look up.

        Returns:
            The :class:`~.base.BaseSkill` instance, or ``None`` if not found.
        """
        return self._skills.get(name)

    def list_skills(self) -> list[str]:
        """Return a sorted list of all registered skill names.

        Returns:
            A new list of string keys in ascending alphabetical order.
        """
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
        skill = self._skills.get(name)
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
            return SkillResult(success=False, output=None, error=error_msg)

        try:
            skill_result = skill.execute(**kwargs)
        except Exception as exc:
            error_msg = str(exc)
            logger.exception("Skill %r raised an unhandled exception.", name)
            self._emit_event(
                skill_name=name,
                session_id=session_id,
                task_id=task_id,
                result="error",
                detail_msg=error_msg,
            )
            return SkillResult(success=False, output=None, error=error_msg)

        event_result = "allow" if skill_result.success else "error"
        self._emit_event(
            skill_name=name,
            session_id=session_id,
            task_id=task_id,
            result=event_result,
            detail_msg=skill_result.error or "",
        )
        return skill_result

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
            event = AuditEvent.now(
                session_id=session_id,
                task_id=task_id,
                event_type="skill.execute",
                category="plugin",
                result=result,  # type: ignore[arg-type]
                detail={"skill": skill_name, "message": detail_msg},
            )
            event_bus.publish(event)
        except Exception:
            logger.exception("Failed to emit audit event for skill %r", skill_name)


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_registry: Optional[SkillRegistry] = None
_lock: threading.Lock = threading.Lock()


def init_skill_registry() -> SkillRegistry:
    """Create and install a fresh process-level :class:`SkillRegistry`.

    Subsequent calls replace the existing registry atomically under a lock.

    Returns:
        The newly installed :class:`SkillRegistry` (empty; register skills
        separately via :meth:`~SkillRegistry.register`).
    """
    global _registry
    registry = SkillRegistry()
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
