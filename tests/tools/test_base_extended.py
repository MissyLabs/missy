"""Tests for missy.tools.base covering uncovered get_schema and repr paths."""

from __future__ import annotations

from missy.tools.base import BaseTool, ToolPermissions, ToolResult


class EchoTool(BaseTool):
    name = "echo"
    description = "Echo text"
    permissions = ToolPermissions()

    def execute(self, *, text: str = "", **kwargs):
        return ToolResult(success=True, output=text)


class ToolWithParams(BaseTool):
    name = "paramtool"
    description = "Tool with declared parameters"
    permissions = ToolPermissions(network=True)
    parameters = {
        "url": {"type": "string", "description": "URL to fetch", "required": True},
        "timeout": {"type": "integer", "description": "Timeout in seconds"},
    }

    def execute(self, **kwargs):
        return ToolResult(success=True, output="ok")


class TestToolPermissions:
    def test_default_all_false(self):
        p = ToolPermissions()
        assert p.network is False
        assert p.filesystem_read is False
        assert p.filesystem_write is False
        assert p.shell is False
        assert p.allowed_paths == []
        assert p.allowed_hosts == []


class TestToolResult:
    def test_default_error_none(self):
        r = ToolResult(success=True, output="hi")
        assert r.error is None

    def test_error_result(self):
        r = ToolResult(success=False, output=None, error="bad")
        assert r.error == "bad"


class TestGetSchema:
    def test_basic_schema(self):
        tool = EchoTool()
        schema = tool.get_schema()
        assert schema["name"] == "echo"
        assert schema["description"] == "Echo text"
        assert schema["parameters"]["type"] == "object"

    def test_schema_with_parameters_attribute(self):
        tool = ToolWithParams()
        schema = tool.get_schema()
        assert schema["name"] == "paramtool"
        props = schema["parameters"]["properties"]
        assert "url" in props
        assert "timeout" in props
        assert props["url"]["type"] == "string"
        # "required" key should be stripped from property def
        assert "required" not in props["url"]
        # Required list should contain url
        assert "url" in schema["parameters"]["required"]
        # Timeout is not required
        assert "timeout" not in schema["parameters"]["required"]

    def test_schema_no_parameters_attribute(self):
        tool = EchoTool()
        schema = tool.get_schema()
        assert schema["parameters"]["properties"] == {}
        assert schema["parameters"]["required"] == []


class TestRepr:
    def test_repr_includes_name(self):
        tool = EchoTool()
        assert "echo" in repr(tool)
        assert "EchoTool" in repr(tool)
