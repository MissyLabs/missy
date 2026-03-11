"""Abstract base classes for Missy tool implementations.

A *tool* is a discrete capability that the agent runtime can invoke.
Each tool declares the permissions it requires via :class:`ToolPermissions`,
which the :class:`~missy.tools.registry.ToolRegistry` validates against the
active policy engine before execution.

Example::

    from missy.tools.base import BaseTool, ToolPermissions, ToolResult

    class EchoTool(BaseTool):
        name = "echo"
        description = "Return the input string unchanged"
        permissions = ToolPermissions()

        def execute(self, *, text: str) -> ToolResult:
            return ToolResult(success=True, output=text)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class ToolPermissions:
    """Declares the permissions a tool requires to operate.

    The registry checks these flags against the active policy engine before
    allowing a tool to execute.  All fields default to ``False`` / empty
    list, so tools that need no elevated access require no explicit
    configuration.

    Attributes:
        network: Tool may make outbound network requests.
        filesystem_read: Tool may read from the filesystem.
        filesystem_write: Tool may write to the filesystem.
        shell: Tool may execute shell commands.
        allowed_paths: Specific filesystem paths the tool is permitted to
            access (read or write, depending on the above flags).
        allowed_hosts: Specific hostnames or IPs the tool is permitted to
            contact.
    """

    network: bool = False
    filesystem_read: bool = False
    filesystem_write: bool = False
    shell: bool = False
    allowed_paths: list[str] = field(default_factory=list)
    allowed_hosts: list[str] = field(default_factory=list)


@dataclass
class ToolResult:
    """The outcome of a single tool execution.

    Attributes:
        success: ``True`` when the tool completed without error.
        output: The primary result value.  Type varies by tool.
        error: Human-readable error description when ``success`` is
            ``False``.  ``None`` on success.
    """

    success: bool
    output: Any
    error: Optional[str] = None


class BaseTool(ABC):
    """Abstract base for all Missy tool implementations.

    Subclasses must set the class-level attributes :attr:`name`,
    :attr:`description`, and :attr:`permissions`, and implement
    :meth:`execute`.

    Attributes:
        name: Short unique identifier used to look up the tool in the
            registry (e.g. ``"calculator"``).
        description: One-sentence human-readable description included in
            the tool schema presented to the model.
        permissions: Permission flags required by this tool.
    """

    name: str
    description: str
    permissions: ToolPermissions

    @abstractmethod
    def execute(self, **kwargs: Any) -> ToolResult:
        """Run the tool with the given keyword arguments.

        Args:
            **kwargs: Tool-specific parameters.

        Returns:
            A :class:`ToolResult` describing success or failure.
        """
        ...

    def get_schema(self) -> dict[str, Any]:
        """Return a JSON Schema dict describing the tool's parameters.

        The default implementation returns a minimal schema with ``name``
        and ``description`` but no typed parameter definitions.  Override
        this method to return a more precise schema for function-calling
        integrations.

        Returns:
            A ``dict`` conforming to a subset of JSON Schema / OpenAI
            function-calling format::

                {
                    "name": "calculator",
                    "description": "...",
                    "parameters": {
                        "type": "object",
                        "properties": {...},
                        "required": [...],
                    },
                }
        """
        return {
            "name": self.name,
            "description": self.description,
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        }

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name!r})"
