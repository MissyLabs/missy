"""Model Context Protocol (MCP) integration for Missy."""

from missy.mcp.annotations import BUILTIN_ANNOTATIONS, AnnotationRegistry, ToolAnnotation
from missy.mcp.client import McpClient
from missy.mcp.manager import McpManager

__all__ = [
    "AnnotationRegistry",
    "BUILTIN_ANNOTATIONS",
    "McpClient",
    "McpManager",
    "ToolAnnotation",
]
