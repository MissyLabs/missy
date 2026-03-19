"""Tests for MCP tool name validation and gateway HEAD method (session 14)."""

from __future__ import annotations

from collections.abc import Generator
from unittest.mock import MagicMock, patch

import httpx
import pytest

from missy.config.settings import (
    FilesystemPolicy,
    MissyConfig,
    NetworkPolicy,
    PluginPolicy,
    ShellPolicy,
)
from missy.core.exceptions import PolicyViolationError
from missy.gateway.client import create_client
from missy.policy import engine as engine_module
from missy.policy.engine import init_policy_engine

# ---------------------------------------------------------------------------
# MCP tool name validation tests
# ---------------------------------------------------------------------------


class TestMcpToolNameValidation:
    """Verify MCP manager rejects unsafe tool names."""

    def _make_manager(self):
        from missy.mcp.manager import McpManager

        mgr = McpManager.__new__(McpManager)
        mgr._clients = {}
        mgr._lock = __import__("threading").Lock()
        return mgr

    def test_valid_tool_name_accepted(self):
        mgr = self._make_manager()
        client = MagicMock()
        client.call_tool.return_value = "result"
        mgr._clients["server"] = client

        result = mgr.call_tool("server__valid_tool", {"arg": 1})
        assert result == "result"

    def test_tool_name_with_hyphen_accepted(self):
        mgr = self._make_manager()
        client = MagicMock()
        client.call_tool.return_value = "ok"
        mgr._clients["server"] = client

        result = mgr.call_tool("server__my-tool-name", {})
        assert result == "ok"

    def test_tool_name_with_spaces_rejected(self):
        mgr = self._make_manager()
        client = MagicMock()
        mgr._clients["server"] = client

        result = mgr.call_tool("server__tool with spaces", {})
        assert "[MCP error] unsafe tool name" in result
        client.call_tool.assert_not_called()

    def test_tool_name_with_semicolon_rejected(self):
        mgr = self._make_manager()
        client = MagicMock()
        mgr._clients["server"] = client

        result = mgr.call_tool("server__tool;rm -rf /", {})
        assert "[MCP error] unsafe tool name" in result

    def test_tool_name_with_pipe_rejected(self):
        mgr = self._make_manager()
        client = MagicMock()
        mgr._clients["server"] = client

        result = mgr.call_tool("server__tool|cat /etc/passwd", {})
        assert "[MCP error] unsafe tool name" in result

    def test_tool_name_with_newline_rejected(self):
        mgr = self._make_manager()
        client = MagicMock()
        mgr._clients["server"] = client

        result = mgr.call_tool("server__tool\ninjected", {})
        assert "[MCP error] unsafe tool name" in result

    def test_tool_name_with_dots_rejected(self):
        mgr = self._make_manager()
        client = MagicMock()
        mgr._clients["server"] = client

        result = mgr.call_tool("server__../../etc/passwd", {})
        assert "[MCP error] unsafe tool name" in result

    def test_empty_tool_name_rejected(self):
        mgr = self._make_manager()
        client = MagicMock()
        mgr._clients["server"] = client

        result = mgr.call_tool("server__", {})
        assert "[MCP error] unsafe tool name" in result

    def test_no_double_underscore_rejected(self):
        mgr = self._make_manager()
        result = mgr.call_tool("invalidname", {})
        assert "[MCP error] invalid tool name" in result


# ---------------------------------------------------------------------------
# Gateway HEAD method tests
# ---------------------------------------------------------------------------


def _make_config(
    *,
    allowed_hosts: list[str] | None = None,
) -> MissyConfig:
    return MissyConfig(
        network=NetworkPolicy(
            default_deny=True,
            allowed_cidrs=[],
            allowed_domains=[],
            allowed_hosts=allowed_hosts or [],
        ),
        filesystem=FilesystemPolicy(),
        shell=ShellPolicy(),
        plugins=PluginPolicy(),
        providers={},
        workspace_path="/tmp",
        audit_log_path="/tmp/audit.log",
    )


@pytest.fixture(autouse=True)
def _policy() -> Generator[None]:
    cfg = _make_config(allowed_hosts=["api.example.com:443"])
    init_policy_engine(cfg)
    yield
    engine_module._engine = None


class TestSyncHeadMethod:
    """Test synchronous HTTP HEAD through PolicyHTTPClient."""

    def test_head_allowed_host(self):
        client = create_client(session_id="s1", task_id="t1")
        resp = httpx.Response(200, headers={"content-length": "1234"})
        with patch.object(client, "_get_sync_client") as mock_client:
            mock_client.return_value.head.return_value = resp
            result = client.head("https://api.example.com:443/resource")
        assert result.status_code == 200

    def test_head_denied_host(self):
        client = create_client(session_id="s1", task_id="t1")
        with pytest.raises(PolicyViolationError):
            client.head("https://evil.example.com/resource")

    def test_head_sanitizes_kwargs(self):
        client = create_client(session_id="s1", task_id="t1")
        resp = httpx.Response(200)
        with patch.object(client, "_get_sync_client") as mock_client:
            mock_client.return_value.head.return_value = resp
            client.head(
                "https://api.example.com:443/r",
                follow_redirects=True,
            )
            _, call_kwargs = mock_client.return_value.head.call_args
            assert "follow_redirects" not in call_kwargs
