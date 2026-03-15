"""Tests for missy.mcp.client — MCP JSON-RPC client."""

from __future__ import annotations

import json
import subprocess
from unittest.mock import MagicMock, patch

import pytest

from missy.mcp.client import McpClient


class TestMcpClientInit:
    def test_stores_name_and_command(self):
        c = McpClient(name="test", command="echo hello")
        assert c.name == "test"
        assert c._command == "echo hello"
        assert c._url is None
        assert c._proc is None
        assert c.tools == []

    def test_stores_url(self):
        c = McpClient(name="http-srv", url="http://localhost:3000")
        assert c._url == "http://localhost:3000"
        assert c._command is None


class TestMcpClientConnect:
    def test_connect_with_command_starts_process(self):
        c = McpClient(name="test", command="echo hello")
        init_resp = {
            "jsonrpc": "2.0",
            "result": {"capabilities": {}},
        }
        tools_resp = {
            "jsonrpc": "2.0",
            "result": {"tools": [{"name": "read_file", "description": "Read a file"}]},
        }
        mock_proc = MagicMock()
        mock_proc.stdin = MagicMock()
        mock_proc.stdout = MagicMock()
        mock_proc.stdout.readline.side_effect = [
            json.dumps(init_resp).encode() + b"\n",
            json.dumps(tools_resp).encode() + b"\n",
        ]
        with patch("subprocess.Popen", return_value=mock_proc):
            c.connect()
        assert len(c.tools) == 1
        assert c.tools[0]["name"] == "read_file"

    def test_connect_sanitizes_environment(self):
        """MCP subprocess must not inherit API keys or other secrets."""
        c = McpClient(name="test", command="echo hello")
        init_resp = {"jsonrpc": "2.0", "result": {"capabilities": {}}}
        tools_resp = {"jsonrpc": "2.0", "result": {"tools": []}}
        mock_proc = MagicMock()
        mock_proc.stdin = MagicMock()
        mock_proc.stdout = MagicMock()
        mock_proc.stdout.readline.side_effect = [
            json.dumps(init_resp).encode() + b"\n",
            json.dumps(tools_resp).encode() + b"\n",
        ]
        import os

        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "sk-secret", "PATH": "/usr/bin"}), \
             patch("subprocess.Popen", return_value=mock_proc) as mock_popen:
            c.connect()
        # Check that env was passed and doesn't contain the API key
        call_kwargs = mock_popen.call_args[1]
        assert "env" in call_kwargs
        assert "ANTHROPIC_API_KEY" not in call_kwargs["env"]
        assert "PATH" in call_kwargs["env"]

    def test_connect_without_command_raises(self):
        c = McpClient(name="test", url="http://localhost:3000")
        with pytest.raises(NotImplementedError, match="HTTP MCP transport"):
            c.connect()


class TestMcpClientRpc:
    def _make_connected_client(self):
        c = McpClient(name="test", command="echo")
        mock_proc = MagicMock()
        mock_proc.stdin = MagicMock()
        mock_proc.stdout = MagicMock()
        mock_proc.poll.return_value = None
        c._proc = mock_proc
        return c

    def test_rpc_sends_and_receives(self):
        c = self._make_connected_client()
        expected = {"jsonrpc": "2.0", "result": {"ok": True}}
        c._proc.stdout.readline.return_value = json.dumps(expected).encode() + b"\n"
        result = c._rpc("test/method", {"key": "value"})
        assert result == expected
        # Verify it wrote JSON-RPC to stdin
        c._proc.stdin.write.assert_called_once()
        sent = json.loads(c._proc.stdin.write.call_args[0][0].decode())
        assert sent["method"] == "test/method"
        assert sent["params"] == {"key": "value"}

    def test_rpc_without_params(self):
        c = self._make_connected_client()
        resp = {"jsonrpc": "2.0", "result": {}}
        c._proc.stdout.readline.return_value = json.dumps(resp).encode() + b"\n"
        c._rpc("test/method")
        sent = json.loads(c._proc.stdin.write.call_args[0][0].decode())
        assert "params" not in sent

    def test_rpc_no_process_raises(self):
        c = McpClient(name="test", command="echo")
        with pytest.raises(RuntimeError, match="not connected"):
            c._rpc("test")

    def test_rpc_empty_response_raises(self):
        c = self._make_connected_client()
        c._proc.stdout.readline.return_value = b""
        with pytest.raises(RuntimeError, match="closed connection"):
            c._rpc("test")


class TestMcpClientNotify:
    def test_notify_writes_without_id(self):
        c = McpClient(name="test", command="echo")
        c._proc = MagicMock()
        c._proc.stdin = MagicMock()
        c._notify("notifications/test", {"data": 1})
        sent = json.loads(c._proc.stdin.write.call_args[0][0].decode())
        assert "id" not in sent
        assert sent["method"] == "notifications/test"
        assert sent["params"] == {"data": 1}

    def test_notify_no_proc_is_safe(self):
        c = McpClient(name="test", command="echo")
        c._notify("test")  # no-op, no error


class TestMcpClientCallTool:
    def _make_connected_client(self):
        c = McpClient(name="test", command="echo")
        c._proc = MagicMock()
        c._proc.stdin = MagicMock()
        c._proc.stdout = MagicMock()
        return c

    def test_call_tool_returns_text_content(self):
        c = self._make_connected_client()
        resp = {
            "jsonrpc": "2.0",
            "result": {
                "content": [
                    {"type": "text", "text": "line1"},
                    {"type": "text", "text": "line2"},
                ]
            },
        }
        c._proc.stdout.readline.return_value = json.dumps(resp).encode() + b"\n"
        result = c.call_tool("read", {"path": "/tmp"})
        assert result == "line1\nline2"

    def test_call_tool_error(self):
        c = self._make_connected_client()
        resp = {"jsonrpc": "2.0", "error": {"code": -1, "message": "fail"}}
        c._proc.stdout.readline.return_value = json.dumps(resp).encode() + b"\n"
        result = c.call_tool("bad", {})
        assert "[MCP error]" in result

    def test_call_tool_no_text_content(self):
        c = self._make_connected_client()
        resp = {
            "jsonrpc": "2.0",
            "result": {"content": [{"type": "image", "data": "base64..."}]},
        }
        c._proc.stdout.readline.return_value = json.dumps(resp).encode() + b"\n"
        result = c.call_tool("screenshot", {})
        # Falls back to str(result)
        assert "content" in result

    def test_call_tool_empty_content(self):
        c = self._make_connected_client()
        resp = {"jsonrpc": "2.0", "result": {"content": []}}
        c._proc.stdout.readline.return_value = json.dumps(resp).encode() + b"\n"
        result = c.call_tool("empty", {})
        assert result  # str(result dict)


class TestMcpClientIsAlive:
    def test_alive_with_running_process(self):
        c = McpClient(name="test", command="echo")
        c._proc = MagicMock()
        c._proc.poll.return_value = None
        assert c.is_alive() is True

    def test_dead_with_exited_process(self):
        c = McpClient(name="test", command="echo")
        c._proc = MagicMock()
        c._proc.poll.return_value = 1
        assert c.is_alive() is False

    def test_dead_with_no_process(self):
        c = McpClient(name="test", command="echo")
        assert c.is_alive() is False


class TestMcpClientDisconnect:
    def test_disconnect_terminates_process(self):
        c = McpClient(name="test", command="echo")
        mock_proc = MagicMock()
        c._proc = mock_proc
        c.disconnect()
        mock_proc.terminate.assert_called_once()
        mock_proc.wait.assert_called_once_with(timeout=5)
        assert c._proc is None

    def test_disconnect_kills_on_timeout(self):
        c = McpClient(name="test", command="echo")
        mock_proc = MagicMock()
        mock_proc.wait.side_effect = subprocess.TimeoutExpired("cmd", 5)
        c._proc = mock_proc
        c.disconnect()
        mock_proc.kill.assert_called_once()

    def test_disconnect_no_proc_is_safe(self):
        c = McpClient(name="test", command="echo")
        c.disconnect()  # no-op
