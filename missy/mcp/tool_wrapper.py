"""Wraps a connected MCP tool as a real BaseTool for ToolRegistry registration.

SR-4.7: this is what makes "register tools through the reference monitor"
literally true for MCP tools -- each wrapper is a real
:class:`~missy.tools.base.BaseTool` subclass instance, registered into the
same :class:`~missy.tools.registry.ToolRegistry` every built-in tool goes
through, so dispatch is gated by the same permission check and produces the
same ``tool_execute`` audit event. :class:`~missy.mcp.manager.McpManager`
layers its own additional, MCP-specific checks (digest re-verification,
annotation-driven approval) on top in :meth:`McpManager.call_tool`, since
those concerns have no equivalent in the generic registry.
"""

from __future__ import annotations

from typing import Any

from missy.mcp.annotations import ToolAnnotation
from missy.tools.base import BaseTool, ToolPermissions, ToolResult

#: Prefixes McpManager.call_tool() uses for a blocked/denied outcome --
#: used here only to decide ToolResult.success, not to alter the message.
_BLOCKED_PREFIXES = ("[MCP BLOCKED]", "[MCP DENIED]", "[MCP error]")


class McpToolWrapper(BaseTool):
    """Adapts one connected MCP tool to Missy's BaseTool interface.

    Args:
        manager: The owning :class:`~missy.mcp.manager.McpManager`, used to
            actually dispatch the call (including its digest/approval
            checks).
        namespaced_name: The ``server__tool`` name this wrapper registers
            under.
        description: Human-readable description from the MCP tool manifest.
        input_schema: The MCP tool's ``inputSchema`` (standard JSON Schema
            ``{"type": "object", "properties": {...}, "required": [...]}``)
            -- already in the shape :meth:`get_schema` needs to return, so
            it is used directly rather than round-tripped through Missy's
            simplified ``parameters`` dict format.
        annotation: The tool's :class:`~missy.mcp.annotations.ToolAnnotation`,
            used to derive a coarse :class:`ToolPermissions` declaration.
            Network/filesystem targets are NOT concretely resolvable for an
            arbitrary third-party MCP tool (it runs as its own external
            process, not through Missy's PolicyHTTPClient/filesystem
            layer), so this declaration is necessarily coarse -- it signals
            intent to the policy engine but does not itself constrain which
            host or path the external MCP server process touches. The
            digest pin (server identity/behavior integrity) and the
            approval gate (human sign-off for destructive tools) are the
            concrete, enforceable controls for MCP; see
            :meth:`~missy.mcp.manager.McpManager.call_tool`.
    """

    def __init__(
        self,
        manager: Any,
        namespaced_name: str,
        description: str,
        input_schema: dict,
        annotation: ToolAnnotation,
    ) -> None:
        self.name = namespaced_name
        self.description = description or f"MCP tool {namespaced_name}"
        self._manager = manager
        self._input_schema = input_schema if isinstance(input_schema, dict) else {}
        self.permissions = ToolPermissions(
            network=annotation.network_access,
            filesystem_read=annotation.filesystem_access,
            filesystem_write=annotation.filesystem_access and annotation.mutating,
        )

    def get_schema(self) -> dict[str, Any]:
        params = self._input_schema or {"type": "object", "properties": {}, "required": []}
        return {"name": self.name, "description": self.description, "parameters": params}

    def execute(self, **kwargs: Any) -> ToolResult:
        content = self._manager.call_tool(self.name, kwargs)
        is_blocked = isinstance(content, str) and content.startswith(_BLOCKED_PREFIXES)
        return ToolResult(
            success=not is_blocked,
            output=content,
            error=content if is_blocked else None,
        )
