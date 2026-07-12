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
from typing import Any


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
        policy_denied: ``True`` when ``success is False`` specifically
            because the policy engine raised ``PolicyViolationError`` (see
            :meth:`missy.tools.registry.ToolRegistry.execute`), as opposed
            to the tool itself failing internally. Lets callers apply a
            harsher trust-score penalty for policy violations than for
            ordinary tool failures without re-parsing ``error`` text.
    """

    success: bool
    output: Any
    error: str | None = None
    policy_denied: bool = False


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

    def resolve_shell_command(self, kwargs: dict[str, Any]) -> str | list[str] | None:
        """Return the actual host command this invocation will execute.

        The registry's default heuristic reads a ``command`` kwarg to decide
        what to check against the shell allow-list. That heuristic is wrong
        for tools whose real host-level program is fixed (or otherwise not
        carried verbatim in a ``command`` kwarg) — e.g. a tool that always
        shells out to a specific binary regardless of what the model passes
        as an argument, or one where a ``command`` kwarg describes something
        *other* than the host program invoked (such as a command run inside
        a sandboxed guest). Overriding this method lets such a tool declare
        the real target explicitly instead of letting an unrelated or
        missing kwarg silently skip enforcement.

        Returning ``None`` (the default) tells the registry to fall back to
        its generic ``command`` kwarg heuristic, preserving existing
        behaviour for tools that don't need this.
        """
        return None

    def resolve_network_hosts(self, kwargs: dict[str, Any]) -> list[str]:
        """Return additional hostnames this invocation will contact.

        Unlike the filesystem/shell checks, the registry has no kwarg-name
        heuristic for network targets at all by default — it only checks
        ``permissions.allowed_hosts`` (a static declaration). A tool whose
        real destination is only known at call time (e.g. a URL kwarg, or
        a browser-automation tool that hands the URL to something other
        than :class:`~missy.gateway.client.PolicyHTTPClient`) must
        override this method, or the network policy engine never sees the
        real destination and the declared ``network=True`` permission
        enforces nothing.

        Returning ``[]`` (the default) means no additional hosts beyond
        ``permissions.allowed_hosts`` are checked, preserving existing
        behaviour for tools that don't need this.
        """
        return []

    def resolve_filesystem_targets(self, kwargs: dict[str, Any]) -> tuple[list[str], list[str]]:
        """Return the ``(read_paths, write_paths)`` this invocation will touch.

        The registry's default heuristic reads ``path``/``file_path``/
        ``target``/``destination`` kwargs to find the real filesystem target
        to check. Tools that carry their target under a different kwarg name
        must override this method, or the filesystem policy engine never
        sees the real path and the declared permission enforces nothing.

        Returning ``([], [])`` (the default) tells the registry to fall back
        to its generic-name heuristic, preserving existing behaviour for
        tools that don't need this.
        """
        return ([], [])

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
        # Build properties from the class-level `parameters` dict if present.
        props: dict = {}
        required: list = []
        raw_params = getattr(self, "parameters", {}) or {}
        for param_name, param_def in raw_params.items():
            props[param_name] = {k: v for k, v in param_def.items() if k != "required"}
            if param_def.get("required", False):
                required.append(param_name)
        return {
            "name": self.name,
            "description": self.description,
            "parameters": {
                "type": "object",
                "properties": props,
                "required": required,
            },
        }

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name!r})"
