"""Tests for missy.mcp.manager — MCP server lifecycle manager."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from missy.mcp.manager import McpManager


@pytest.fixture
def tmp_config(tmp_path):
    """Return a path for a temporary MCP config file."""
    return str(tmp_path / "mcp.json")


@pytest.fixture
def manager(tmp_config):
    return McpManager(config_path=tmp_config)


class TestConnectAll:
    def test_no_config_file(self, manager):
        manager.connect_all()
        assert manager.list_servers() == []

    def test_malformed_config(self, tmp_config):
        Path(tmp_config).write_text("NOT JSON")
        mgr = McpManager(config_path=tmp_config)
        mgr.connect_all()
        assert mgr.list_servers() == []

    def test_connects_configured_servers(self, tmp_config):
        config = [{"name": "fs", "command": "npx fs-server"}]
        Path(tmp_config).write_text(json.dumps(config))
        mgr = McpManager(config_path=tmp_config)
        mock_client = MagicMock()
        mock_client.tools = [{"name": "read"}]
        mock_client.is_alive.return_value = True
        with patch.object(mgr, "add_server", return_value=mock_client):
            mgr.connect_all()

    def test_handles_connect_failure(self, tmp_config):
        config = [{"name": "bad", "command": "nonexistent"}]
        Path(tmp_config).write_text(json.dumps(config))
        mgr = McpManager(config_path=tmp_config)
        with patch.object(mgr, "add_server", side_effect=RuntimeError("fail")):
            mgr.connect_all()
        assert mgr.list_servers() == []


class TestAddServer:
    def test_add_server_connects_and_persists(self, manager, tmp_config):
        mock_client = MagicMock()
        mock_client.tools = [{"name": "tool1"}, {"name": "tool2"}]
        mock_client._command = "echo hello"
        mock_client._url = None
        with patch("missy.mcp.manager.McpClient", return_value=mock_client):
            client = manager.add_server("test-server", command="echo hello")
        assert client is mock_client
        mock_client.connect.assert_called_once()
        # Config file should be persisted
        saved = json.loads(Path(tmp_config).read_text())
        assert len(saved) == 1
        assert saved[0]["name"] == "test-server"

    def test_add_server_propagates_connect_error(self, manager):
        mock_client = MagicMock()
        mock_client.connect.side_effect = RuntimeError("connection failed")
        with patch("missy.mcp.manager.McpClient", return_value=mock_client):
            with pytest.raises(RuntimeError, match="connection failed"):
                manager.add_server("bad", command="fail")


class TestRemoveServer:
    def test_remove_existing_server(self, manager, tmp_config):
        mock_client = MagicMock()
        mock_client.tools = []
        mock_client._command = "echo"
        mock_client._url = None
        with patch("missy.mcp.manager.McpClient", return_value=mock_client):
            manager.add_server("srv", command="echo")
        manager.remove_server("srv")
        mock_client.disconnect.assert_called_once()
        assert manager.list_servers() == []

    def test_remove_nonexistent_is_safe(self, manager):
        manager.remove_server("nonexistent")  # no error


class TestRestartServer:
    def test_restart_replaces_client(self, manager):
        old_client = MagicMock()
        old_client._command = "echo old"
        old_client._url = None
        old_client.tools = []
        manager._clients["srv"] = old_client

        new_client = MagicMock()
        new_client.tools = [{"name": "new_tool"}]
        new_client.is_alive.return_value = True
        with patch("missy.mcp.manager.McpClient", return_value=new_client):
            manager.restart_server("srv")
        old_client.disconnect.assert_called_once()
        new_client.connect.assert_called_once()
        assert manager._clients["srv"] is new_client

    def test_restart_nonexistent_is_safe(self, manager):
        manager.restart_server("nonexistent")  # no error


class TestHealthCheck:
    def test_restarts_dead_servers(self, manager):
        dead = MagicMock()
        dead.is_alive.return_value = False
        dead._command = "echo dead"
        dead._url = None
        alive = MagicMock()
        alive.is_alive.return_value = True
        manager._clients = {"dead_srv": dead, "alive_srv": alive}

        new_client = MagicMock()
        new_client.tools = []
        with patch("missy.mcp.manager.McpClient", return_value=new_client):
            manager.health_check()
        dead.disconnect.assert_called_once()
        new_client.connect.assert_called_once()

    def test_health_check_handles_restart_failure(self, manager):
        dead = MagicMock()
        dead.is_alive.return_value = False
        dead._command = "echo dead"
        dead._url = None
        manager._clients = {"dead_srv": dead}
        with patch.object(manager, "restart_server", side_effect=RuntimeError("fail")):
            manager.health_check()  # should not raise


class TestAllTools:
    def test_namespaces_tools(self, manager):
        client = MagicMock()
        client.tools = [
            {"name": "read_file", "description": "Read"},
            {"name": "write_file", "description": "Write"},
        ]
        manager._clients = {"fs": client}
        tools = manager.all_tools()
        assert len(tools) == 2
        assert tools[0]["name"] == "fs__read_file"
        assert tools[0]["_mcp_server"] == "fs"
        assert tools[0]["_mcp_tool"] == "read_file"

    def test_empty_when_no_servers(self, manager):
        assert manager.all_tools() == []


class TestCallTool:
    def test_call_valid_namespaced_tool(self, manager):
        client = MagicMock()
        client.call_tool.return_value = "result text"
        manager._clients = {"srv": client}
        result = manager.call_tool("srv__read", {"path": "/tmp"})
        assert result == "result text"
        client.call_tool.assert_called_once_with("read", {"path": "/tmp"})

    def test_call_invalid_name_format(self, manager):
        result = manager.call_tool("no_separator", {})
        assert "[MCP error]" in result

    def test_call_unknown_server(self, manager):
        result = manager.call_tool("unknown__tool", {})
        assert "[MCP error]" in result
        assert "not connected" in result


class TestListServers:
    def test_list_with_servers(self, manager):
        c1 = MagicMock()
        c1.is_alive.return_value = True
        c1.tools = [{"name": "t1"}]
        c2 = MagicMock()
        c2.is_alive.return_value = False
        c2.tools = []
        manager._clients = {"a": c1, "b": c2}
        servers = manager.list_servers()
        assert len(servers) == 2
        a = next(s for s in servers if s["name"] == "a")
        assert a["alive"] is True
        assert a["tools"] == 1


class TestShutdown:
    def test_shutdown_disconnects_all(self, manager):
        c1 = MagicMock()
        c2 = MagicMock()
        manager._clients = {"a": c1, "b": c2}
        manager.shutdown()
        c1.disconnect.assert_called_once()
        c2.disconnect.assert_called_once()

    def test_shutdown_suppresses_errors(self, manager):
        c = MagicMock()
        c.disconnect.side_effect = RuntimeError("fail")
        manager._clients = {"a": c}
        manager.shutdown()  # no error


class TestSaveConfig:
    def test_config_round_trip(self, manager, tmp_config):
        mock_client = MagicMock()
        mock_client.tools = []
        mock_client._command = "echo hello"
        mock_client._url = None
        with patch("missy.mcp.manager.McpClient", return_value=mock_client):
            manager.add_server("test", command="echo hello")
        saved = json.loads(Path(tmp_config).read_text())
        assert saved[0]["name"] == "test"
        assert saved[0]["command"] == "echo hello"
