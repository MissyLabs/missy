"""Tests for missy.mcp.client — MCP JSON-RPC client."""

from __future__ import annotations

import json
import subprocess
import time
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
        mock_proc.poll.return_value = None  # Process still running
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
        mock_proc.poll.return_value = None  # Process still running
        mock_proc.stdin = MagicMock()
        mock_proc.stdout = MagicMock()
        mock_proc.stdout.readline.side_effect = [
            json.dumps(init_resp).encode() + b"\n",
            json.dumps(tools_resp).encode() + b"\n",
        ]
        import os

        with (
            patch.dict(os.environ, {"ANTHROPIC_API_KEY": "sk-secret", "PATH": "/usr/bin"}),
            patch("subprocess.Popen", return_value=mock_proc) as mock_popen,
        ):
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


class TestMcpClientTimeoutTeardown:
    """Availability hardening: an RPC timeout must tear the connection down
    immediately, not leave a stale unread response sitting on stdout to
    desynchronize (and silently corrupt) the next, otherwise-unrelated
    call. Live-reproduced against a real slow-but-not-dead subprocess
    before this fix: the second call received the first call's
    now-arrived response and raised a confusing ID-mismatch error instead
    of getting its own response."""

    def _make_connected_client(self):
        c = McpClient(name="test", command="echo")
        mock_proc = MagicMock()
        mock_proc.stdin = MagicMock()
        mock_proc.stdout = MagicMock()
        mock_proc.poll.return_value = None
        c._proc = mock_proc
        return c

    def test_timeout_tears_down_the_subprocess(self):
        c = self._make_connected_client()
        proc = c._proc
        with (
            patch("select.select", return_value=([], [], [])),
            pytest.raises(TimeoutError, match="did not respond within"),
        ):
            c._rpc("slow_method", timeout=0.01)
        proc.kill.assert_called_once()
        assert c._proc is None

    def test_is_alive_false_after_timeout(self):
        c = self._make_connected_client()
        with patch("select.select", return_value=([], [], [])), pytest.raises(TimeoutError):
            c._rpc("slow_method", timeout=0.01)
        assert c.is_alive() is False

    def test_subsequent_call_after_timeout_fails_cleanly_not_desynced(self):
        """The whole point of tearing down: the next call must fail with
        a clear "not connected" error, never read a stale response and
        raise (or worse, silently accept) a mismatched one."""
        c = self._make_connected_client()
        with patch("select.select", return_value=([], [], [])), pytest.raises(TimeoutError):
            c._rpc("slow_method", timeout=0.01)

        with pytest.raises(RuntimeError, match="not connected"):
            c._rpc("tools/list")

    def test_timeout_teardown_closes_pipes(self):
        c = self._make_connected_client()
        proc = c._proc
        with patch("select.select", return_value=([], [], [])), pytest.raises(TimeoutError):
            c._rpc("slow_method", timeout=0.01)
        proc.stdin.close.assert_called_once()
        proc.stdout.close.assert_called_once()
        proc.stderr.close.assert_called_once()

    def test_kill_exception_during_teardown_does_not_mask_timeout_error(self):
        """If kill() itself raises (e.g. process already reaped), the
        caller must still see the TimeoutError, not an unrelated crash."""
        c = self._make_connected_client()
        c._proc.kill.side_effect = ProcessLookupError("already dead")
        with patch("select.select", return_value=([], [], [])), pytest.raises(TimeoutError):
            c._rpc("slow_method", timeout=0.01)
        assert c._proc is None

    def test_real_slow_server_timeout_does_not_desync_next_real_call(self, tmp_path):
        """End-to-end with a real subprocess (not mocks): a server that
        answers *after* the client's timeout must not corrupt the next
        call's response."""
        import textwrap

        server_script = tmp_path / "fake_mcp_server.py"
        server_script.write_text(
            textwrap.dedent(
                """
                import sys, json, time

                def send(obj):
                    sys.stdout.write(json.dumps(obj) + "\\n")
                    sys.stdout.flush()

                while True:
                    line = sys.stdin.readline()
                    if not line:
                        break
                    req = json.loads(line)
                    method = req.get("method")
                    req_id = req.get("id")
                    if method == "slow_call":
                        time.sleep(0.3)
                        send({"jsonrpc": "2.0", "id": req_id, "result": {"ok": "slow"}})
                    else:
                        send({"jsonrpc": "2.0", "id": req_id, "result": {"ok": True}})
                """
            )
        )
        c = McpClient(name="real-fake", command=f"python3 {server_script}")
        c._proc = subprocess.Popen(
            ["python3", str(server_script)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        try:
            with pytest.raises(TimeoutError):
                c._rpc("slow_call", timeout=0.1)
            assert c.is_alive() is False

            # A brand-new connection for the "next call" -- the whole
            # point is that the OLD one is gone, not reusable in a
            # corrupted state.
            c._proc = subprocess.Popen(
                ["python3", str(server_script)],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            resp = c._rpc("tools/list", timeout=2.0)
            assert resp["result"] == {"ok": True}
        finally:
            if c._proc is not None:
                c._proc.kill()
                c._proc.wait(timeout=5)

    def test_real_server_partial_response_then_stall_times_out(self, tmp_path):
        """Regression: select()-readiness only proves *some* bytes are
        available, not a full line. A server that writes a partial
        response (no trailing newline) and then stalls previously caused
        the bare readline() call to block indefinitely, still holding
        self._lock, with no way back -- the process stays alive
        (is_alive()/poll() both true), so McpManager's health_check()
        auto-recovery never kicks in. _read_line_with_deadline() must
        bound this within the requested timeout, not hang.
        """
        import textwrap

        server_script = tmp_path / "partial_response_server.py"
        server_script.write_text(
            textwrap.dedent(
                """
                import sys, time

                line = sys.stdin.readline()  # consume the one request
                # Write a syntactically-valid JSON *prefix*, deliberately
                # never send the trailing newline, then stall.
                sys.stdout.write('{"jsonrpc": "2.0", "id": "x"')
                sys.stdout.flush()
                time.sleep(10)
                """
            )
        )
        c = McpClient(name="partial-fake", command=f"python3 {server_script}")
        c._proc = subprocess.Popen(
            ["python3", str(server_script)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        try:
            start = time.monotonic()
            with pytest.raises(TimeoutError):
                c._rpc("whatever", timeout=1.0)
            elapsed = time.monotonic() - start
            assert elapsed < 5.0, f"expected to time out near 1.0s, took {elapsed:.1f}s"
            assert c.is_alive() is False
        finally:
            if c._proc is not None:
                c._proc.kill()
                c._proc.wait(timeout=5)


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
