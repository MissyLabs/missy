"""Tests for missy.mcp.tool_wrapper.McpToolWrapper — SR-4.7.

McpToolWrapper is what makes "register MCP tools through the reference
monitor" literally true: each wrapper is a real BaseTool the ToolRegistry
can register, permission-check, and audit exactly like any built-in tool.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from missy.mcp.annotations import ToolAnnotation
from missy.mcp.tool_wrapper import McpToolWrapper
from missy.tools.base import BaseTool


class TestConstruction:
    def test_is_a_base_tool(self):
        wrapper = McpToolWrapper(MagicMock(), "srv__tool", "desc", {}, ToolAnnotation())
        assert isinstance(wrapper, BaseTool)

    def test_name_is_namespaced(self):
        wrapper = McpToolWrapper(MagicMock(), "srv__read_file", "desc", {}, ToolAnnotation())
        assert wrapper.name == "srv__read_file"

    def test_empty_description_gets_fallback(self):
        wrapper = McpToolWrapper(MagicMock(), "srv__tool", "", {}, ToolAnnotation())
        assert "srv__tool" in wrapper.description


class TestPermissionDerivation:
    """Coarse ToolPermissions derived from the tool's annotation -- signals
    intent to the policy engine even though exact network/filesystem
    targets aren't concretely resolvable for an arbitrary MCP server."""

    def test_network_access_annotation_sets_network_permission(self):
        ann = ToolAnnotation(network_access=True)
        wrapper = McpToolWrapper(MagicMock(), "srv__fetch", "d", {}, ann)
        assert wrapper.permissions.network is True

    def test_filesystem_access_annotation_sets_read_permission(self):
        ann = ToolAnnotation(filesystem_access=True, mutating=False)
        wrapper = McpToolWrapper(MagicMock(), "srv__read", "d", {}, ann)
        assert wrapper.permissions.filesystem_read is True
        assert wrapper.permissions.filesystem_write is False

    def test_filesystem_access_plus_mutating_sets_write_permission(self):
        ann = ToolAnnotation(filesystem_access=True, mutating=True)
        wrapper = McpToolWrapper(MagicMock(), "srv__write", "d", {}, ann)
        assert wrapper.permissions.filesystem_write is True

    def test_default_annotation_grants_no_permissions(self):
        wrapper = McpToolWrapper(MagicMock(), "srv__noop", "d", {}, ToolAnnotation())
        assert wrapper.permissions.network is False
        assert wrapper.permissions.filesystem_read is False
        assert wrapper.permissions.filesystem_write is False


class TestSchema:
    def test_input_schema_used_directly(self):
        schema = {
            "type": "object",
            "properties": {"path": {"type": "string"}},
            "required": ["path"],
        }
        wrapper = McpToolWrapper(MagicMock(), "srv__read", "Reads a file", schema, ToolAnnotation())
        result = wrapper.get_schema()
        assert result["name"] == "srv__read"
        assert result["description"] == "Reads a file"
        assert result["parameters"] == schema

    def test_missing_input_schema_gets_empty_object_schema(self):
        wrapper = McpToolWrapper(MagicMock(), "srv__noop", "d", {}, ToolAnnotation())
        result = wrapper.get_schema()
        assert result["parameters"]["type"] == "object"

    def test_non_dict_input_schema_falls_back_to_empty(self):
        wrapper = McpToolWrapper(MagicMock(), "srv__weird", "d", "not-a-dict", ToolAnnotation())
        result = wrapper.get_schema()
        assert result["parameters"]["type"] == "object"


class TestExecute:
    def test_success_result(self):
        manager = MagicMock()
        manager.call_tool.return_value = "real output"
        wrapper = McpToolWrapper(manager, "srv__tool", "d", {}, ToolAnnotation())

        result = wrapper.execute(arg="value")

        assert result.success is True
        assert result.output == "real output"
        assert result.error is None
        manager.call_tool.assert_called_once_with("srv__tool", {"arg": "value"})

    @pytest.mark.parametrize(
        "prefix",
        ["[MCP BLOCKED]", "[MCP DENIED]", "[MCP error]"],
    )
    def test_blocked_prefixes_map_to_failure(self, prefix):
        manager = MagicMock()
        manager.call_tool.return_value = f"{prefix} something went wrong"
        wrapper = McpToolWrapper(manager, "srv__tool", "d", {}, ToolAnnotation())

        result = wrapper.execute()

        assert result.success is False
        assert result.error == f"{prefix} something went wrong"

    def test_ordinary_content_starting_differently_is_success(self):
        manager = MagicMock()
        manager.call_tool.return_value = "Result: [MCP something]"  # doesn't start with a blocked prefix
        wrapper = McpToolWrapper(manager, "srv__tool", "d", {}, ToolAnnotation())

        result = wrapper.execute()

        assert result.success is True
