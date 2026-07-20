"""Base classes for Missy skills.

Skills are lightweight, in-process callable units that the agent can invoke
directly.  Unlike plugins, skills do not require explicit configuration to be
enabled — they are registered and executed through the
:class:`~missy.skills.registry.SkillRegistry`.

Example::

    from missy.skills.base import BaseSkill, SkillPermissions, SkillResult

    class EchoSkill(BaseSkill):
        name = "echo"
        description = "Return the input text unchanged."
        permissions = SkillPermissions()

        def execute(self, *, text: str = "") -> SkillResult:
            return SkillResult(success=True, output=text)
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class SkillPermissions:
    """Declares the resources a skill requires at execution time.

    All permissions default to ``False`` (deny everything).  A skill that
    needs outbound network access must declare ``network=True``, and so on.

    Attributes:
        network: Skill may make outbound network requests.
        filesystem_read: Skill may read from the filesystem.
        filesystem_write: Skill may write to the filesystem.
        shell: Skill may execute shell commands.
    """

    network: bool = False
    filesystem_read: bool = False
    filesystem_write: bool = False
    shell: bool = False


@dataclass
class SkillResult:
    """Encapsulates the outcome of a skill execution.

    Attributes:
        success: ``True`` when the skill executed without error.
        output: The skill's return value; may be any JSON-serialisable type.
        error: Human-readable error description; empty string on success.
    """

    success: bool
    output: Any
    error: str = ""

    _MAX_SERIALIZED_BYTES = 1_000_000

    def __post_init__(self) -> None:
        if not isinstance(self.success, bool):
            raise TypeError("SkillResult.success must be a bool")
        if not isinstance(self.error, str):
            raise TypeError("SkillResult.error must be a string")
        if self.success and self.error:
            raise ValueError("A successful SkillResult cannot contain an error")
        try:
            encoded = json.dumps(
                {"output": self.output, "error": self.error},
                ensure_ascii=False,
                allow_nan=False,
            ).encode("utf-8")
        except (TypeError, ValueError, RecursionError) as exc:
            raise ValueError("SkillResult must be JSON serializable") from exc
        if len(encoded) > self._MAX_SERIALIZED_BYTES:
            raise ValueError(
                f"SkillResult exceeds the {self._MAX_SERIALIZED_BYTES}-byte serialized limit"
            )


class BaseSkill(ABC):
    """Abstract base for all Missy skills.

    Concrete subclasses must declare :attr:`name`, :attr:`description`, and
    :attr:`permissions` as class attributes, and implement :meth:`execute`.

    Class attributes:
        name: Unique registry key for the skill (e.g. ``"echo"``).
        description: One-line description shown in help text.
        version: Semantic version string.  Defaults to ``"0.1.0"``.
        permissions: :class:`SkillPermissions` instance declaring required
            resources.
    """

    name: str
    description: str
    version: str = "0.1.0"
    # SkillPermissions is immutable, so a shared deny-all default is safe and
    # remains a real permission value for subclasses that omit the attribute.
    permissions: SkillPermissions = SkillPermissions()

    @abstractmethod
    def execute(self, **kwargs: Any) -> SkillResult:
        """Run the skill with the supplied keyword arguments.

        Args:
            **kwargs: Skill-specific parameters.

        Returns:
            A :class:`SkillResult` describing the outcome.
        """
        ...

    def get_help(self) -> str:
        """Return a one-line help string for this skill.

        Returns:
            A string of the form ``"<name> v<version>: <description>"``.
        """
        return f"{self.name} v{self.version}: {self.description}"
