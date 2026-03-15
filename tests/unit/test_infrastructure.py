"""Infrastructure module tests.

Covers:
- missy/channels/webhook.py  — HTTP webhook channel
- missy/config/hotreload.py  — ConfigWatcher file-polling hot-reload
- missy/mcp/manager.py       — McpManager lifecycle
- missy/mcp/client.py        — McpClient JSON-RPC transport
- missy/memory/resilient.py  — ResilientMemoryStore with in-memory fallback
- missy/observability/otel.py — OtelExporter OTLP integration
"""

from __future__ import annotations

import hashlib
import hmac
import io
import json
import threading
import time
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# WebhookChannel
# ---------------------------------------------------------------------------
from missy.channels.base import ChannelMessage
from missy.channels.webhook import WebhookChannel
from missy.config.hotreload import ConfigWatcher, _apply_config
from missy.mcp.client import McpClient
from missy.mcp.manager import McpManager
from missy.memory.resilient import ResilientMemoryStore
from missy.observability.otel import OtelExporter, init_otel


class TestWebhookChannelInit:
    def test_default_attributes(self):
        ch = WebhookChannel()
        assert ch._host == "127.0.0.1"
        assert ch._port == 9090
        assert ch._secret == b""
        assert ch._queue == []
        assert ch._server is None
        assert ch._thread is None

    def test_custom_host_and_port(self):
        ch = WebhookChannel(host="0.0.0.0", port=8080)
        assert ch._host == "0.0.0.0"
        assert ch._port == 8080

    def test_secret_encoded_to_bytes(self):
        ch = WebhookChannel(secret="mysecret")
        assert ch._secret == b"mysecret"

    def test_empty_secret_stays_empty_bytes(self):
        ch = WebhookChannel(secret="")
        assert ch._secret == b""

    def test_name_is_webhook(self):
        assert WebhookChannel.name == "webhook"


class TestWebhookChannelReceive:
    def test_receive_empty_queue_returns_none(self):
        ch = WebhookChannel()
        assert ch.receive() is None

    def test_receive_returns_queued_message(self):
        ch = WebhookChannel()
        msg = ChannelMessage(content="hello", sender="user", channel="webhook")
        with ch._lock:
            ch._queue.append(msg)
        result = ch.receive()
        assert result is msg

    def test_receive_pops_first_item(self):
        ch = WebhookChannel()
        m1 = ChannelMessage(content="first", channel="webhook")
        m2 = ChannelMessage(content="second", channel="webhook")
        with ch._lock:
            ch._queue.extend([m1, m2])
        assert ch.receive() is m1
        assert ch.receive() is m2
        assert ch.receive() is None

    def test_receive_thread_safe(self):
        ch = WebhookChannel()
        for i in range(20):
            ch._queue.append(ChannelMessage(content=str(i), channel="webhook"))

        results = []

        def reader():
            for _ in range(10):
                msg = ch.receive()
                if msg is not None:
                    results.append(msg)

        threads = [threading.Thread(target=reader) for _ in range(2)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(results) == 20


class TestWebhookChannelSend:
    def test_send_does_not_raise(self):
        ch = WebhookChannel()
        ch.send("some response text")  # Should log and not raise


class TestWebhookChannelStop:
    def test_stop_when_no_server_is_noop(self):
        ch = WebhookChannel()
        ch.stop()  # _server is None; should not raise

    def test_stop_calls_server_shutdown(self):
        ch = WebhookChannel()
        mock_server = MagicMock()
        ch._server = mock_server
        ch.stop()
        mock_server.shutdown.assert_called_once()


class TestWebhookChannelStart:
    def test_start_creates_server_and_thread(self):
        ch = WebhookChannel(port=0)
        mock_server = MagicMock()
        mock_thread = MagicMock()

        with (
            patch("missy.channels.webhook.HTTPServer", return_value=mock_server) as mock_http,
            patch("missy.channels.webhook.threading.Thread", return_value=mock_thread),
        ):
            ch.start()

        mock_http.assert_called_once_with(("127.0.0.1", 0), mock_http.call_args[0][1])
        mock_thread.start.assert_called_once()
        assert ch._server is mock_server
        assert ch._thread is mock_thread


class TestWebhookHandlerDoPost:
    """Drive the inner Handler.do_POST method directly without a live TCP server."""

    def _make_handler(self, body: bytes, headers: dict, channel: WebhookChannel):
        """Return a Handler instance wired to write into a BytesIO sink."""

        # Build a minimal handler by invoking start() and extracting the Handler class
        # by patching HTTPServer to capture the handler class argument.
        captured = {}

        def fake_httpserver(addr, handler_class):
            captured["handler"] = handler_class
            server = MagicMock()
            return server

        with (
            patch("missy.channels.webhook.HTTPServer", side_effect=fake_httpserver),
            patch("missy.channels.webhook.threading.Thread"),
        ):
            channel.start()

        HandlerClass = captured["handler"]

        rfile = io.BytesIO(body)
        wfile = io.BytesIO()

        mock_headers = {k.lower(): v for k, v in headers.items()}
        mock_headers_obj = MagicMock()
        mock_headers_obj.get = lambda k, default=None: mock_headers.get(k.lower(), default)

        handler = HandlerClass.__new__(HandlerClass)
        handler.rfile = rfile
        handler.wfile = wfile
        handler.headers = mock_headers_obj
        handler.client_address = ("127.0.0.1", 12345)

        _responses = []

        def fake_send_response(code):
            _responses.append(code)

        handler.send_response = fake_send_response
        handler.send_header = MagicMock()
        handler.end_headers = MagicMock()

        return handler, wfile, _responses

    def test_valid_post_queues_message(self):
        ch = WebhookChannel()
        body = json.dumps({"prompt": "Do something"}).encode()
        headers = {"Content-Length": str(len(body))}
        handler, wfile, responses = self._make_handler(body, headers, ch)
        handler.do_POST()
        assert responses == [202]
        msg = ch.receive()
        assert msg is not None
        assert msg.content == "Do something"
        assert msg.sender == "webhook"
        assert msg.channel == "webhook"

    def test_custom_sender_is_preserved(self):
        ch = WebhookChannel()
        body = json.dumps({"prompt": "Hi", "sender": "alice"}).encode()
        headers = {"Content-Length": str(len(body))}
        handler, _, _ = self._make_handler(body, headers, ch)
        handler.do_POST()
        msg = ch.receive()
        assert msg.sender == "alice"

    def test_invalid_json_returns_400(self):
        ch = WebhookChannel()
        body = b"NOT JSON"
        headers = {"Content-Length": str(len(body))}
        handler, _, responses = self._make_handler(body, headers, ch)
        handler.do_POST()
        assert responses == [400]
        assert ch.receive() is None

    def test_missing_prompt_returns_400(self):
        ch = WebhookChannel()
        body = json.dumps({"other_key": "value"}).encode()
        headers = {"Content-Length": str(len(body))}
        handler, _, responses = self._make_handler(body, headers, ch)
        handler.do_POST()
        assert responses == [400]

    def test_whitespace_only_prompt_returns_400(self):
        ch = WebhookChannel()
        body = json.dumps({"prompt": "   "}).encode()
        headers = {"Content-Length": str(len(body))}
        handler, _, responses = self._make_handler(body, headers, ch)
        handler.do_POST()
        assert responses == [400]

    def test_hmac_valid_signature_accepted(self):
        secret = "topsecret"
        ch = WebhookChannel(secret=secret)
        body = json.dumps({"prompt": "authenticated"}).encode()
        sig = "sha256=" + hmac.new(secret.encode(), body, hashlib.sha256).hexdigest()
        headers = {"Content-Length": str(len(body)), "X-Missy-Signature": sig}
        handler, _, responses = self._make_handler(body, headers, ch)
        handler.do_POST()
        assert responses == [202]

    def test_hmac_invalid_signature_returns_401(self):
        ch = WebhookChannel(secret="topsecret")
        body = json.dumps({"prompt": "sneaky"}).encode()
        headers = {"Content-Length": str(len(body)), "X-Missy-Signature": "sha256=badhash"}
        handler, _, responses = self._make_handler(body, headers, ch)
        handler.do_POST()
        assert responses == [401]
        assert ch.receive() is None

    def test_hmac_missing_signature_returns_401(self):
        ch = WebhookChannel(secret="topsecret")
        body = json.dumps({"prompt": "sneaky"}).encode()
        headers = {"Content-Length": str(len(body))}
        handler, _, responses = self._make_handler(body, headers, ch)
        handler.do_POST()
        assert responses == [401]

    def test_response_body_is_queued_json(self):
        ch = WebhookChannel()
        body = json.dumps({"prompt": "test"}).encode()
        headers = {"Content-Length": str(len(body))}
        handler, wfile, _ = self._make_handler(body, headers, ch)
        handler.do_POST()
        assert b"queued" in wfile.getvalue()


# ---------------------------------------------------------------------------
# ConfigWatcher
# ---------------------------------------------------------------------------


class TestConfigWatcherInit:
    def test_attributes_set(self, tmp_path):
        cfg = tmp_path / "config.yaml"
        cfg.write_text("key: value")

        def fn(c):
            pass

        watcher = ConfigWatcher(str(cfg), fn, debounce_seconds=1.5, poll_interval=0.5)
        assert watcher._path == cfg
        assert watcher._reload_fn is fn
        assert watcher._debounce == 1.5
        assert watcher._poll == 0.5

    def test_tilde_expanded_in_path(self):
        def fn(c):
            pass

        watcher = ConfigWatcher("~/.missy/config.yaml", fn)
        assert not str(watcher._path).startswith("~")

    def test_initial_mtime_is_zero(self, tmp_path):
        cfg = tmp_path / "config.yaml"

        def fn(c):
            pass

        watcher = ConfigWatcher(str(cfg), fn)
        assert watcher._last_mtime == 0.0
        assert watcher._thread is None


class TestConfigWatcherStartStop:
    def test_start_initialises_mtime_from_file(self, tmp_path):
        cfg = tmp_path / "config.yaml"
        cfg.write_text("key: value")
        watcher = ConfigWatcher(str(cfg), lambda c: None, poll_interval=100)
        watcher.start()
        try:
            assert watcher._last_mtime == cfg.stat().st_mtime
        finally:
            watcher.stop()

    def test_start_missing_file_sets_mtime_zero(self, tmp_path):
        cfg = tmp_path / "nonexistent.yaml"
        watcher = ConfigWatcher(str(cfg), lambda c: None, poll_interval=100)
        watcher.start()
        try:
            assert watcher._last_mtime == 0.0
        finally:
            watcher.stop()

    def test_start_creates_daemon_thread(self, tmp_path):
        cfg = tmp_path / "config.yaml"
        cfg.write_text("")
        watcher = ConfigWatcher(str(cfg), lambda c: None, poll_interval=100)
        watcher.start()
        try:
            assert watcher._thread is not None
            assert watcher._thread.daemon is True
            assert watcher._thread.name == "missy-hotreload"
        finally:
            watcher.stop()

    def test_stop_signals_thread(self, tmp_path):
        cfg = tmp_path / "config.yaml"
        cfg.write_text("")
        watcher = ConfigWatcher(str(cfg), lambda c: None, poll_interval=100)
        watcher.start()
        watcher.stop()
        assert watcher._stop.is_set()

    def test_stop_without_start_is_safe(self, tmp_path):
        cfg = tmp_path / "config.yaml"
        watcher = ConfigWatcher(str(cfg), lambda c: None)
        watcher.stop()  # _thread is None; must not raise


class TestConfigWatcherDoReload:
    def test_reload_calls_reload_fn_with_new_config(self, tmp_path):
        cfg = tmp_path / "config.yaml"
        cfg.write_text("key: value")
        cfg.chmod(0o600)  # safety check requires owner-only permissions
        received = []
        watcher = ConfigWatcher(str(cfg), lambda c: received.append(c), poll_interval=100)

        fake_config = object()
        # load_config is imported inside _do_reload, so patch at the source module
        with patch("missy.config.settings.load_config", return_value=fake_config):
            watcher._do_reload()

        assert received == [fake_config]

    def test_reload_exception_does_not_propagate(self, tmp_path):
        cfg = tmp_path / "config.yaml"
        cfg.write_text("")
        cfg.chmod(0o600)  # safety check requires owner-only permissions
        watcher = ConfigWatcher(str(cfg), lambda c: None, poll_interval=100)

        with patch("missy.config.settings.load_config", side_effect=RuntimeError("parse error")):
            watcher._do_reload()  # must not raise


class TestConfigWatcherDetectsChange:
    def test_reload_triggered_after_debounce(self, tmp_path):
        cfg = tmp_path / "config.yaml"
        cfg.write_text("initial: true")
        cfg.chmod(0o600)  # safety check requires owner-only permissions
        received = []
        fake_config = object()

        with patch("missy.config.settings.load_config", return_value=fake_config):
            watcher = ConfigWatcher(
                str(cfg),
                lambda c: received.append(c),
                debounce_seconds=0.05,
                poll_interval=0.02,
            )
            watcher.start()
            time.sleep(0.05)
            cfg.write_text("updated: true")
            cfg.chmod(0o600)  # re-apply after write
            time.sleep(0.4)
            watcher.stop()

        assert len(received) >= 1


class TestApplyConfig:
    def test_apply_config_calls_init_functions(self):
        fake_config = object()
        # Both are imported inside _apply_config; patch at their source modules
        with (
            patch("missy.policy.engine.init_policy_engine") as mock_pe,
            patch("missy.providers.registry.init_registry") as mock_reg,
        ):
            _apply_config(fake_config)
        mock_pe.assert_called_once_with(fake_config)
        mock_reg.assert_called_once_with(fake_config)


# ---------------------------------------------------------------------------
# McpClient
# ---------------------------------------------------------------------------


def _make_rpc_response(result=None, error=None, id="test-id"):
    resp = {"jsonrpc": "2.0", "id": id}
    if error is not None:
        resp["error"] = error
    else:
        resp["result"] = result or {}
    return (json.dumps(resp) + "\n").encode()


class TestMcpClientInit:
    def test_attributes_set(self):
        client = McpClient(name="fs", command="npx fs", url=None)
        assert client.name == "fs"
        assert client._command == "npx fs"
        assert client._url is None
        assert client._proc is None
        assert client._tools == []

    def test_tools_property_returns_list(self):
        client = McpClient(name="x")
        assert client.tools == []


class TestMcpClientIsAlive:
    def test_alive_when_proc_running(self):
        client = McpClient(name="x")
        mock_proc = MagicMock()
        mock_proc.poll.return_value = None
        client._proc = mock_proc
        assert client.is_alive() is True

    def test_dead_when_proc_exited(self):
        client = McpClient(name="x")
        mock_proc = MagicMock()
        mock_proc.poll.return_value = 0
        client._proc = mock_proc
        assert client.is_alive() is False

    def test_dead_when_no_proc(self):
        client = McpClient(name="x")
        assert client.is_alive() is False


class TestMcpClientConnect:
    def test_connect_without_command_raises_not_implemented(self):
        client = McpClient(name="x", url="http://localhost:8080")
        with pytest.raises(NotImplementedError, match="HTTP MCP transport"):
            client.connect()

    def test_connect_with_command_starts_process(self):
        client = McpClient(name="test", command="echo hello")
        init_resp = _make_rpc_response(result={"capabilities": {}})
        tools_resp = _make_rpc_response(result={"tools": [{"name": "mytool"}]})
        mock_proc = MagicMock()
        mock_proc.stdout.readline.side_effect = [init_resp, tools_resp]
        mock_proc.stdin = MagicMock()

        with patch("missy.mcp.client.subprocess.Popen", return_value=mock_proc):
            client.connect()

        assert client._proc is mock_proc
        assert len(client._tools) == 1
        assert client._tools[0]["name"] == "mytool"

    def test_connect_init_error_raises_runtime_error(self):
        client = McpClient(name="test", command="echo hello")
        error_resp = _make_rpc_response(error="init failed")
        mock_proc = MagicMock()
        mock_proc.stdout.readline.return_value = error_resp
        mock_proc.stdin = MagicMock()

        with (
            patch("missy.mcp.client.subprocess.Popen", return_value=mock_proc),
            pytest.raises(RuntimeError, match="MCP init failed"),
        ):
            client.connect()


class TestMcpClientRpc:
    def _connected_client(self, responses: list[bytes]) -> McpClient:
        client = McpClient(name="test", command="fake")
        mock_proc = MagicMock()
        mock_proc.stdout.readline.side_effect = responses
        mock_proc.stdin = MagicMock()
        client._proc = mock_proc
        return client

    def test_rpc_raises_when_no_proc(self):
        client = McpClient(name="test")
        with pytest.raises(RuntimeError, match="not connected"):
            client._rpc("some/method")

    def test_rpc_raises_on_empty_response(self):
        client = McpClient(name="test", command="fake")
        mock_proc = MagicMock()
        mock_proc.stdout.readline.return_value = b""
        mock_proc.stdin = MagicMock()
        client._proc = mock_proc
        with pytest.raises(RuntimeError, match="closed connection"):
            client._rpc("some/method")

    def test_rpc_sends_json_line(self):
        resp = _make_rpc_response(result={"ok": True})
        client = self._connected_client([resp])
        client._rpc("test/method", {"key": "value"})
        written = client._proc.stdin.write.call_args[0][0].decode()
        payload = json.loads(written)
        assert payload["method"] == "test/method"
        assert payload["params"] == {"key": "value"}
        assert payload["jsonrpc"] == "2.0"
        assert "id" in payload

    def test_rpc_returns_parsed_response(self):
        resp = _make_rpc_response(result={"data": 42})
        client = self._connected_client([resp])
        result = client._rpc("test/method")
        assert result["result"]["data"] == 42


class TestMcpClientCallTool:
    def _connected_client_with_tool_resp(self, content_items: list) -> McpClient:
        client = McpClient(name="test", command="fake")
        result_payload = {"content": content_items}
        resp = _make_rpc_response(result=result_payload)
        mock_proc = MagicMock()
        mock_proc.stdout.readline.return_value = resp
        mock_proc.stdin = MagicMock()
        client._proc = mock_proc
        return client

    def test_call_tool_returns_text_content(self):
        client = self._connected_client_with_tool_resp(
            [{"type": "text", "text": "hello from tool"}]
        )
        result = client.call_tool("mytool", {"arg": "val"})
        assert result == "hello from tool"

    def test_call_tool_joins_multiple_text_parts(self):
        client = self._connected_client_with_tool_resp(
            [{"type": "text", "text": "part1"}, {"type": "text", "text": "part2"}]
        )
        result = client.call_tool("mytool", {})
        assert result == "part1\npart2"

    def test_call_tool_skips_non_text_items(self):
        client = self._connected_client_with_tool_resp(
            [{"type": "image", "url": "http://example.com/img.png"}, {"type": "text", "text": "ok"}]
        )
        result = client.call_tool("mytool", {})
        assert result == "ok"

    def test_call_tool_returns_str_result_when_no_text_parts(self):
        client = self._connected_client_with_tool_resp([{"type": "image"}])
        result = client.call_tool("mytool", {})
        assert "image" in result

    def test_call_tool_error_response(self):
        client = McpClient(name="test", command="fake")
        error_resp = _make_rpc_response(error="tool not found")
        mock_proc = MagicMock()
        mock_proc.stdout.readline.return_value = error_resp
        mock_proc.stdin = MagicMock()
        client._proc = mock_proc
        result = client.call_tool("badtool", {})
        assert "[MCP error]" in result


class TestMcpClientDisconnect:
    def test_disconnect_terminates_process(self):
        client = McpClient(name="test")
        mock_proc = MagicMock()
        client._proc = mock_proc
        client.disconnect()
        mock_proc.terminate.assert_called_once()
        assert client._proc is None

    def test_disconnect_kills_on_timeout(self):
        client = McpClient(name="test")
        mock_proc = MagicMock()
        mock_proc.wait.side_effect = Exception("timeout")
        client._proc = mock_proc
        client.disconnect()
        mock_proc.kill.assert_called_once()
        assert client._proc is None

    def test_disconnect_when_no_proc_is_noop(self):
        client = McpClient(name="test")
        client.disconnect()  # Must not raise


class TestMcpClientNotify:
    def test_notify_sends_notification_without_id(self):
        client = McpClient(name="test", command="fake")
        mock_proc = MagicMock()
        mock_proc.stdin = MagicMock()
        client._proc = mock_proc
        client._notify("notifications/initialized")
        written = client._proc.stdin.write.call_args[0][0].decode()
        payload = json.loads(written)
        assert payload["method"] == "notifications/initialized"
        assert "id" not in payload

    def test_notify_includes_params_when_given(self):
        client = McpClient(name="test", command="fake")
        mock_proc = MagicMock()
        mock_proc.stdin = MagicMock()
        client._proc = mock_proc
        client._notify("some/event", {"key": "val"})
        written = client._proc.stdin.write.call_args[0][0].decode()
        payload = json.loads(written)
        assert payload["params"] == {"key": "val"}


# ---------------------------------------------------------------------------
# McpManager
# ---------------------------------------------------------------------------


def _mock_client(name="srv", tools=None, alive=True, command="echo", url=None):
    client = MagicMock(
        spec=["name", "tools", "is_alive", "connect", "disconnect", "_command", "_url", "call_tool"]
    )
    client.name = name
    client.tools = tools or []
    client.is_alive.return_value = alive
    client._command = command
    client._url = url
    return client


class TestMcpManagerInit:
    def test_default_config_path_expanded(self):
        mgr = McpManager()
        assert not str(mgr._config_path).startswith("~")

    def test_custom_config_path(self, tmp_path):
        cfg = tmp_path / "mcp.json"
        mgr = McpManager(config_path=str(cfg))
        assert mgr._config_path == cfg

    def test_clients_starts_empty(self, tmp_path):
        mgr = McpManager(config_path=str(tmp_path / "mcp.json"))
        assert mgr._clients == {}


class TestMcpManagerConnectAll:
    def test_no_config_file_is_noop(self, tmp_path):
        mgr = McpManager(config_path=str(tmp_path / "missing.json"))
        mgr.connect_all()  # Must not raise; no clients connected
        assert mgr._clients == {}

    def test_malformed_config_is_skipped(self, tmp_path):
        cfg = tmp_path / "mcp.json"
        cfg.write_text("NOT JSON")
        mgr = McpManager(config_path=str(cfg))
        mgr.connect_all()
        assert mgr._clients == {}

    def test_connects_servers_from_config(self, tmp_path):
        cfg = tmp_path / "mcp.json"
        cfg.write_text(json.dumps([{"name": "fs", "command": "npx fs-server"}]))
        mgr = McpManager(config_path=str(cfg))

        mock_c = _mock_client(name="fs", tools=[{"name": "read"}])
        with patch("missy.mcp.manager.McpClient", return_value=mock_c):
            mgr.connect_all()

        assert "fs" in mgr._clients

    def test_failed_server_connection_is_skipped(self, tmp_path):
        cfg = tmp_path / "mcp.json"
        cfg.write_text(
            json.dumps(
                [
                    {"name": "good", "command": "echo good"},
                    {"name": "bad", "command": "bad-cmd"},
                ]
            )
        )
        mgr = McpManager(config_path=str(cfg))

        good_client = _mock_client(name="good", tools=[])
        bad_client = MagicMock()
        bad_client.connect.side_effect = RuntimeError("cannot connect")

        def client_factory(name, command, url):
            if name == "bad":
                return bad_client
            return good_client

        with patch("missy.mcp.manager.McpClient", side_effect=client_factory):
            mgr.connect_all()

        assert "good" in mgr._clients
        assert "bad" not in mgr._clients


class TestMcpManagerAddServer:
    def test_add_server_stores_client(self, tmp_path):
        cfg = tmp_path / "mcp.json"
        mgr = McpManager(config_path=str(cfg))
        mock_c = _mock_client(name="myserver", tools=[])

        with patch("missy.mcp.manager.McpClient", return_value=mock_c):
            result = mgr.add_server("myserver", command="echo test")

        assert result is mock_c
        assert "myserver" in mgr._clients
        mock_c.connect.assert_called_once()

    def test_add_server_persists_config(self, tmp_path):
        cfg = tmp_path / "mcp.json"
        mgr = McpManager(config_path=str(cfg))
        mock_c = _mock_client(name="srv", tools=[], command="echo")

        with patch("missy.mcp.manager.McpClient", return_value=mock_c):
            mgr.add_server("srv", command="echo")

        saved = json.loads(cfg.read_text())
        assert any(e["name"] == "srv" for e in saved)


class TestMcpManagerRemoveServer:
    def test_remove_existing_server(self, tmp_path):
        cfg = tmp_path / "mcp.json"
        mgr = McpManager(config_path=str(cfg))
        mock_c = _mock_client(name="fs", command="echo")
        mgr._clients["fs"] = mock_c

        mgr.remove_server("fs")

        assert "fs" not in mgr._clients
        mock_c.disconnect.assert_called_once()

    def test_remove_nonexistent_server_is_noop(self, tmp_path):
        cfg = tmp_path / "mcp.json"
        mgr = McpManager(config_path=str(cfg))
        mgr.remove_server("unknown")  # Must not raise


class TestMcpManagerRestartServer:
    def test_restart_replaces_client(self, tmp_path):
        cfg = tmp_path / "mcp.json"
        mgr = McpManager(config_path=str(cfg))
        old_client = _mock_client(name="srv", command="echo", tools=[])
        mgr._clients["srv"] = old_client

        new_client = _mock_client(name="srv", command="echo", tools=[])

        with patch("missy.mcp.manager.McpClient", return_value=new_client):
            mgr.restart_server("srv")

        old_client.disconnect.assert_called_once()
        new_client.connect.assert_called_once()
        assert mgr._clients["srv"] is new_client

    def test_restart_nonexistent_server_is_noop(self, tmp_path):
        cfg = tmp_path / "mcp.json"
        mgr = McpManager(config_path=str(cfg))
        mgr.restart_server("ghost")  # Must not raise


class TestMcpManagerHealthCheck:
    def test_health_check_restarts_dead_server(self, tmp_path):
        cfg = tmp_path / "mcp.json"
        mgr = McpManager(config_path=str(cfg))
        dead_client = _mock_client(name="srv", command="echo", tools=[], alive=False)
        mgr._clients["srv"] = dead_client

        new_client = _mock_client(name="srv", command="echo", tools=[])
        with patch("missy.mcp.manager.McpClient", return_value=new_client):
            mgr.health_check()

        assert mgr._clients["srv"] is new_client

    def test_health_check_leaves_alive_servers_alone(self, tmp_path):
        cfg = tmp_path / "mcp.json"
        mgr = McpManager(config_path=str(cfg))
        alive_client = _mock_client(name="srv", command="echo", tools=[], alive=True)
        mgr._clients["srv"] = alive_client

        mgr.health_check()

        assert mgr._clients["srv"] is alive_client

    def test_health_check_handles_restart_failure(self, tmp_path):
        cfg = tmp_path / "mcp.json"
        mgr = McpManager(config_path=str(cfg))
        dead_client = _mock_client(name="srv", command="echo", tools=[], alive=False)
        mgr._clients["srv"] = dead_client

        failing_client = MagicMock()
        failing_client.connect.side_effect = RuntimeError("cannot restart")

        with patch("missy.mcp.manager.McpClient", return_value=failing_client):
            mgr.health_check()  # Must not propagate the error


class TestMcpManagerAllTools:
    def test_all_tools_namespaced(self, tmp_path):
        cfg = tmp_path / "mcp.json"
        mgr = McpManager(config_path=str(cfg))
        client = _mock_client(name="fs", tools=[{"name": "read", "description": "Read a file"}])
        mgr._clients["fs"] = client

        tools = mgr.all_tools()

        assert len(tools) == 1
        assert tools[0]["name"] == "fs__read"
        assert tools[0]["_mcp_server"] == "fs"
        assert tools[0]["_mcp_tool"] == "read"

    def test_all_tools_multiple_servers(self, tmp_path):
        cfg = tmp_path / "mcp.json"
        mgr = McpManager(config_path=str(cfg))
        mgr._clients["a"] = _mock_client(name="a", tools=[{"name": "t1"}, {"name": "t2"}])
        mgr._clients["b"] = _mock_client(name="b", tools=[{"name": "t3"}])

        tools = mgr.all_tools()

        names = {t["name"] for t in tools}
        assert names == {"a__t1", "a__t2", "b__t3"}

    def test_all_tools_empty_when_no_clients(self, tmp_path):
        cfg = tmp_path / "mcp.json"
        mgr = McpManager(config_path=str(cfg))
        assert mgr.all_tools() == []


class TestMcpManagerCallTool:
    def test_call_tool_routes_to_correct_server(self, tmp_path):
        cfg = tmp_path / "mcp.json"
        mgr = McpManager(config_path=str(cfg))
        mock_c = _mock_client(name="fs")
        mock_c.call_tool.return_value = "file contents"
        mgr._clients["fs"] = mock_c

        result = mgr.call_tool("fs__read", {"path": "/tmp/x"})

        mock_c.call_tool.assert_called_once_with("read", {"path": "/tmp/x"})
        assert result == "file contents"

    def test_call_tool_invalid_name_returns_error(self, tmp_path):
        cfg = tmp_path / "mcp.json"
        mgr = McpManager(config_path=str(cfg))
        result = mgr.call_tool("no_double_underscore", {})
        assert "invalid tool name" in result

    def test_call_tool_unknown_server_returns_error(self, tmp_path):
        cfg = tmp_path / "mcp.json"
        mgr = McpManager(config_path=str(cfg))
        result = mgr.call_tool("ghost__tool", {})
        assert "not connected" in result


class TestMcpManagerListServers:
    def test_list_servers_returns_dicts(self, tmp_path):
        cfg = tmp_path / "mcp.json"
        mgr = McpManager(config_path=str(cfg))
        mgr._clients["fs"] = _mock_client(name="fs", tools=[{"name": "read"}], alive=True)

        servers = mgr.list_servers()

        assert len(servers) == 1
        assert servers[0]["name"] == "fs"
        assert servers[0]["alive"] is True
        assert servers[0]["tools"] == 1

    def test_list_servers_empty_when_no_clients(self, tmp_path):
        cfg = tmp_path / "mcp.json"
        mgr = McpManager(config_path=str(cfg))
        assert mgr.list_servers() == []


class TestMcpManagerShutdown:
    def test_shutdown_disconnects_all_clients(self, tmp_path):
        cfg = tmp_path / "mcp.json"
        mgr = McpManager(config_path=str(cfg))
        c1 = _mock_client(name="a")
        c2 = _mock_client(name="b")
        mgr._clients["a"] = c1
        mgr._clients["b"] = c2

        mgr.shutdown()

        c1.disconnect.assert_called_once()
        c2.disconnect.assert_called_once()

    def test_shutdown_swallows_disconnect_errors(self, tmp_path):
        cfg = tmp_path / "mcp.json"
        mgr = McpManager(config_path=str(cfg))
        bad_client = _mock_client(name="x")
        bad_client.disconnect.side_effect = RuntimeError("boom")
        mgr._clients["x"] = bad_client

        mgr.shutdown()  # Must not raise


# ---------------------------------------------------------------------------
# ResilientMemoryStore
# ---------------------------------------------------------------------------


def _make_turn(session_id: str = "s1", content: str = "hello", timestamp=None):
    turn = MagicMock()
    turn.session_id = session_id
    turn.content = content
    turn.id = "turn-id"
    if timestamp is None:
        turn.timestamp = 0
    else:
        turn.timestamp = timestamp
    return turn


class TestResilientMemoryStoreInit:
    def test_starts_healthy(self):
        primary = MagicMock()
        store = ResilientMemoryStore(primary)
        assert store.is_healthy is True
        assert store._failures == 0

    def test_custom_max_failures(self):
        primary = MagicMock()
        store = ResilientMemoryStore(primary, max_failures=5)
        assert store._max_failures == 5

    def test_cache_starts_empty(self):
        primary = MagicMock()
        store = ResilientMemoryStore(primary)
        assert store._cache == {}


class TestResilientMemoryStoreAddTurn:
    def test_add_turn_writes_to_primary(self):
        primary = MagicMock()
        store = ResilientMemoryStore(primary)
        turn = _make_turn()
        store.add_turn(turn)
        primary.add_turn.assert_called_once_with(turn)

    def test_add_turn_cached_locally(self):
        primary = MagicMock()
        store = ResilientMemoryStore(primary)
        turn = _make_turn(session_id="abc")
        store.add_turn(turn)
        assert turn in store._cache["abc"]

    def test_add_turn_primary_failure_still_caches(self):
        primary = MagicMock()
        primary.add_turn.side_effect = RuntimeError("db down")
        store = ResilientMemoryStore(primary)
        turn = _make_turn()
        store.add_turn(turn)
        assert turn in store._cache["s1"]
        assert store.is_healthy is False

    def test_add_turn_multiple_sessions(self):
        primary = MagicMock()
        store = ResilientMemoryStore(primary)
        t1 = _make_turn(session_id="s1")
        t2 = _make_turn(session_id="s2")
        store.add_turn(t1)
        store.add_turn(t2)
        assert len(store._cache) == 2


class TestResilientMemoryStoreHealthTracking:
    def test_failures_increment_on_primary_error(self):
        primary = MagicMock()
        primary.add_turn.side_effect = RuntimeError("error")
        store = ResilientMemoryStore(primary, max_failures=3)
        for _ in range(2):
            store.add_turn(_make_turn())
        assert store._failures == 2
        assert store.is_healthy is False

    def test_health_restored_after_success(self):
        primary = MagicMock()
        primary.add_turn.side_effect = [RuntimeError("error"), None]
        store = ResilientMemoryStore(primary, max_failures=3)
        store.add_turn(_make_turn())  # fails
        assert store.is_healthy is False
        store.add_turn(_make_turn())  # succeeds
        assert store.is_healthy is True
        assert store._failures == 0

    def test_cache_synced_to_primary_on_recovery(self):
        primary = MagicMock()
        # First call fails; subsequent calls succeed
        primary.add_turn.side_effect = [RuntimeError("error")] + [None] * 10
        store = ResilientMemoryStore(primary, max_failures=3)
        turn1 = _make_turn(session_id="s1", content="first")
        turn2 = _make_turn(session_id="s1", content="second")
        store.add_turn(turn1)  # fails; turn1 cached
        store.add_turn(turn2)  # succeeds; turn2 also in cache before write, then sync replays both
        # call_count:
        #   1  — turn1 primary write (fails)
        #   1  — turn2 primary write (succeeds, triggers _on_success -> sync)
        #   2  — sync replays full cache: turn1 + turn2
        # total = 4
        assert primary.add_turn.call_count == 4
        # Verify the store reports healthy after recovery
        assert store.is_healthy is True


class TestResilientMemoryStoreClearSession:
    def test_clear_removes_from_cache_and_primary(self):
        primary = MagicMock()
        store = ResilientMemoryStore(primary)
        turn = _make_turn(session_id="s1")
        store._cache["s1"] = [turn]
        store.clear_session("s1")
        assert "s1" not in store._cache
        primary.clear_session.assert_called_once_with("s1")

    def test_clear_primary_failure_still_clears_cache(self):
        primary = MagicMock()
        primary.clear_session.side_effect = RuntimeError("db error")
        store = ResilientMemoryStore(primary)
        store._cache["s1"] = [_make_turn()]
        store.clear_session("s1")
        assert "s1" not in store._cache
        assert store.is_healthy is False


class TestResilientMemoryStoreSaveLearning:
    def test_save_learning_delegates_to_primary(self):
        primary = MagicMock()
        store = ResilientMemoryStore(primary)
        learning = MagicMock()
        store.save_learning(learning)
        primary.save_learning.assert_called_once_with(learning)

    def test_save_learning_primary_failure_does_not_raise(self):
        primary = MagicMock()
        primary.save_learning.side_effect = RuntimeError("db error")
        store = ResilientMemoryStore(primary)
        store.save_learning(MagicMock())  # Must not raise
        assert store.is_healthy is False


class TestResilientMemoryStoreGetSessionTurns:
    def test_returns_primary_result_on_success(self):
        primary = MagicMock()
        expected = [_make_turn()]
        primary.get_session_turns.return_value = expected
        store = ResilientMemoryStore(primary)
        result = store.get_session_turns("s1")
        assert result is expected

    def test_falls_back_to_cache_on_primary_failure(self):
        primary = MagicMock()
        primary.get_session_turns.side_effect = RuntimeError("db error")
        store = ResilientMemoryStore(primary)
        turn = _make_turn(session_id="s1")
        store._cache["s1"] = [turn]
        result = store.get_session_turns("s1")
        assert turn in result

    def test_cache_fallback_respects_limit(self):
        primary = MagicMock()
        primary.get_session_turns.side_effect = RuntimeError("db error")
        store = ResilientMemoryStore(primary)
        turns = [_make_turn(session_id="s1", content=str(i)) for i in range(10)]
        store._cache["s1"] = turns
        result = store.get_session_turns("s1", limit=3)
        assert len(result) == 3

    def test_empty_cache_on_fallback_returns_empty(self):
        primary = MagicMock()
        primary.get_session_turns.side_effect = RuntimeError("db error")
        store = ResilientMemoryStore(primary)
        result = store.get_session_turns("nonexistent")
        assert result == []


class TestResilientMemoryStoreGetRecentTurns:
    def test_returns_primary_result_on_success(self):
        primary = MagicMock()
        expected = [_make_turn()]
        primary.get_recent_turns.return_value = expected
        store = ResilientMemoryStore(primary)
        assert store.get_recent_turns() is expected

    def test_falls_back_to_cache_sorted_by_timestamp(self):
        primary = MagicMock()
        primary.get_recent_turns.side_effect = RuntimeError("db error")
        store = ResilientMemoryStore(primary)
        t1 = _make_turn(session_id="s1", timestamp=1)
        t2 = _make_turn(session_id="s2", timestamp=2)
        store._cache["s1"] = [t1]
        store._cache["s2"] = [t2]
        result = store.get_recent_turns(limit=10)
        assert result == [t1, t2]

    def test_cache_fallback_limit_respected(self):
        primary = MagicMock()
        primary.get_recent_turns.side_effect = RuntimeError("db error")
        store = ResilientMemoryStore(primary)
        turns = [_make_turn(session_id="s1", timestamp=i) for i in range(10)]
        store._cache["s1"] = turns
        result = store.get_recent_turns(limit=4)
        assert len(result) == 4


class TestResilientMemoryStoreSearch:
    def test_delegates_to_primary(self):
        primary = MagicMock()
        expected = [_make_turn()]
        primary.search.return_value = expected
        store = ResilientMemoryStore(primary)
        result = store.search("hello", limit=5, session_id="s1")
        primary.search.assert_called_once_with("hello", limit=5, session_id="s1")
        assert result is expected

    def test_fallback_keyword_scan_on_primary_failure(self):
        primary = MagicMock()
        primary.search.side_effect = RuntimeError("db error")
        store = ResilientMemoryStore(primary)
        t1 = _make_turn(content="hello world", session_id="s1")
        t2 = _make_turn(content="goodbye", session_id="s1")
        store._cache["s1"] = [t1, t2]
        result = store.search("hello")
        assert t1 in result
        assert t2 not in result

    def test_fallback_case_insensitive_search(self):
        primary = MagicMock()
        primary.search.side_effect = RuntimeError("db error")
        store = ResilientMemoryStore(primary)
        t = _make_turn(content="HELLO WORLD")
        store._cache["s1"] = [t]
        result = store.search("hello")
        assert t in result

    def test_fallback_filters_by_session_id(self):
        primary = MagicMock()
        primary.search.side_effect = RuntimeError("db error")
        store = ResilientMemoryStore(primary)
        t1 = _make_turn(content="hello", session_id="s1")
        t2 = _make_turn(content="hello", session_id="s2")
        store._cache["s1"] = [t1]
        store._cache["s2"] = [t2]
        result = store.search("hello", session_id="s1")
        assert t1 in result
        assert t2 not in result

    def test_fallback_limit_respected(self):
        primary = MagicMock()
        primary.search.side_effect = RuntimeError("db error")
        store = ResilientMemoryStore(primary)
        turns = [_make_turn(content="match", session_id="s1") for _ in range(10)]
        store._cache["s1"] = turns
        result = store.search("match", limit=3)
        assert len(result) == 3


class TestResilientMemoryStoreGetLearnings:
    def test_delegates_to_primary(self):
        primary = MagicMock()
        primary.get_learnings.return_value = ["lesson1"]
        store = ResilientMemoryStore(primary)
        result = store.get_learnings(task_type="coding", limit=3)
        primary.get_learnings.assert_called_once_with(task_type="coding", limit=3)
        assert result == ["lesson1"]

    def test_returns_empty_list_on_primary_failure(self):
        primary = MagicMock()
        primary.get_learnings.side_effect = RuntimeError("db error")
        store = ResilientMemoryStore(primary)
        result = store.get_learnings()
        assert result == []


class TestResilientMemoryStoreCleanup:
    def test_delegates_to_primary(self):
        primary = MagicMock()
        primary.cleanup.return_value = 42
        store = ResilientMemoryStore(primary)
        result = store.cleanup(older_than_days=7)
        primary.cleanup.assert_called_once_with(older_than_days=7)
        assert result == 42

    def test_returns_zero_on_primary_failure(self):
        primary = MagicMock()
        primary.cleanup.side_effect = RuntimeError("db error")
        store = ResilientMemoryStore(primary)
        result = store.cleanup()
        assert result == 0


class TestResilientMemoryStoreIsHealthy:
    def test_is_healthy_initially_true(self):
        store = ResilientMemoryStore(MagicMock())
        assert store.is_healthy is True

    def test_is_healthy_false_after_failure(self):
        primary = MagicMock()
        primary.add_turn.side_effect = RuntimeError("error")
        store = ResilientMemoryStore(primary)
        store.add_turn(_make_turn())
        assert store.is_healthy is False


# ---------------------------------------------------------------------------
# OtelExporter
# ---------------------------------------------------------------------------


class TestOtelExporterNoPackages:
    """Tests when opentelemetry packages are not installed."""

    def test_init_without_otel_packages_is_disabled(self):
        with patch.dict("sys.modules", {"opentelemetry": None}):
            exporter = OtelExporter()
        assert exporter.is_enabled is False
        assert exporter._tracer is None

    def test_export_event_when_disabled_is_noop(self):
        exporter = OtelExporter.__new__(OtelExporter)
        exporter._enabled = False
        exporter._tracer = None
        exporter.export_event({"event_type": "test"})  # Must not raise


class TestOtelExporterInit:
    def test_attributes_set_correctly(self):
        mock_trace = MagicMock()
        mock_resource_cls = MagicMock()
        mock_resource_cls.create.return_value = MagicMock()
        mock_provider = MagicMock()
        mock_tracer_provider_cls = MagicMock(return_value=mock_provider)
        mock_exporter = MagicMock()
        mock_exporter_cls = MagicMock(return_value=mock_exporter)
        mock_processor = MagicMock()
        mock_processor_cls = MagicMock(return_value=mock_processor)

        modules = {
            "opentelemetry": MagicMock(trace=mock_trace),
            "opentelemetry.sdk.resources": MagicMock(
                SERVICE_NAME="service.name", Resource=mock_resource_cls
            ),
            "opentelemetry.sdk.trace": MagicMock(TracerProvider=mock_tracer_provider_cls),
            "opentelemetry.exporter.otlp.proto.grpc.trace_exporter": MagicMock(
                OTLPSpanExporter=mock_exporter_cls
            ),
            "opentelemetry.sdk.trace.export": MagicMock(BatchSpanProcessor=mock_processor_cls),
        }

        with patch.dict("sys.modules", modules):
            exporter = OtelExporter(
                endpoint="http://collector:4317",
                protocol="grpc",
                service_name="test-service",
            )

        assert exporter._endpoint == "http://collector:4317"
        assert exporter._protocol == "grpc"
        assert exporter._service_name == "test-service"

    def test_grpc_protocol_uses_grpc_exporter(self):
        mock_trace = MagicMock()
        mock_resource_cls = MagicMock()
        mock_resource_cls.create.return_value = MagicMock()
        mock_provider = MagicMock()
        mock_tracer_provider_cls = MagicMock(return_value=mock_provider)
        grpc_exporter_cls = MagicMock()
        http_exporter_cls = MagicMock()
        mock_processor_cls = MagicMock()

        modules = {
            "opentelemetry": MagicMock(trace=mock_trace),
            "opentelemetry.sdk.resources": MagicMock(
                SERVICE_NAME="service.name", Resource=mock_resource_cls
            ),
            "opentelemetry.sdk.trace": MagicMock(TracerProvider=mock_tracer_provider_cls),
            "opentelemetry.exporter.otlp.proto.grpc.trace_exporter": MagicMock(
                OTLPSpanExporter=grpc_exporter_cls
            ),
            "opentelemetry.exporter.otlp.proto.http.trace_exporter": MagicMock(
                OTLPSpanExporter=http_exporter_cls
            ),
            "opentelemetry.sdk.trace.export": MagicMock(BatchSpanProcessor=mock_processor_cls),
        }

        with patch.dict("sys.modules", modules):
            OtelExporter(protocol="grpc")

        grpc_exporter_cls.assert_called_once()
        http_exporter_cls.assert_not_called()

    def test_http_protocol_uses_http_exporter(self):
        mock_trace = MagicMock()
        mock_resource_cls = MagicMock()
        mock_resource_cls.create.return_value = MagicMock()
        mock_provider = MagicMock()
        mock_tracer_provider_cls = MagicMock(return_value=mock_provider)
        grpc_exporter_cls = MagicMock()
        http_exporter_cls = MagicMock()
        mock_processor_cls = MagicMock()

        modules = {
            "opentelemetry": MagicMock(trace=mock_trace),
            "opentelemetry.sdk.resources": MagicMock(
                SERVICE_NAME="service.name", Resource=mock_resource_cls
            ),
            "opentelemetry.sdk.trace": MagicMock(TracerProvider=mock_tracer_provider_cls),
            "opentelemetry.exporter.otlp.proto.grpc.trace_exporter": MagicMock(
                OTLPSpanExporter=grpc_exporter_cls
            ),
            "opentelemetry.exporter.otlp.proto.http.trace_exporter": MagicMock(
                OTLPSpanExporter=http_exporter_cls
            ),
            "opentelemetry.sdk.trace.export": MagicMock(BatchSpanProcessor=mock_processor_cls),
        }

        with patch.dict("sys.modules", modules):
            OtelExporter(protocol="http/protobuf")

        http_exporter_cls.assert_called_once()

    def test_setup_exception_leaves_disabled(self):
        mock_trace = MagicMock()
        mock_resource_cls = MagicMock()
        mock_resource_cls.create.side_effect = RuntimeError("setup boom")

        modules = {
            "opentelemetry": MagicMock(trace=mock_trace),
            "opentelemetry.sdk.resources": MagicMock(
                SERVICE_NAME="service.name", Resource=mock_resource_cls
            ),
            "opentelemetry.sdk.trace": MagicMock(),
            "opentelemetry.exporter.otlp.proto.grpc.trace_exporter": MagicMock(),
            "opentelemetry.sdk.trace.export": MagicMock(),
        }

        with patch.dict("sys.modules", modules):
            exporter = OtelExporter()

        assert exporter.is_enabled is False


class TestOtelExporterExportEvent:
    def _enabled_exporter(self):
        exporter = OtelExporter.__new__(OtelExporter)
        exporter._enabled = True
        mock_tracer = MagicMock()
        mock_span = MagicMock()
        mock_tracer.start_as_current_span.return_value.__enter__ = MagicMock(return_value=mock_span)
        mock_tracer.start_as_current_span.return_value.__exit__ = MagicMock(return_value=False)
        exporter._tracer = mock_tracer
        return exporter, mock_tracer, mock_span

    def test_export_event_uses_event_type_as_span_name(self):
        exporter, mock_tracer, _ = self._enabled_exporter()
        exporter.export_event({"event_type": "tool_call", "session_id": "s1"})
        mock_tracer.start_as_current_span.assert_called_once_with("tool_call")

    def test_export_event_uses_default_span_name_when_no_event_type(self):
        exporter, mock_tracer, _ = self._enabled_exporter()
        exporter.export_event({"session_id": "s1"})
        mock_tracer.start_as_current_span.assert_called_once_with("missy.event")

    def test_export_event_sets_scalar_attributes(self):
        exporter, _, mock_span = self._enabled_exporter()
        exporter.export_event({"event_type": "test", "count": 5, "flag": True})
        calls = {c[0][0]: c[0][1] for c in mock_span.set_attribute.call_args_list}
        assert calls.get("missy.count") == 5
        assert calls.get("missy.flag") is True

    def test_export_event_flattens_detail_dict(self):
        exporter, _, mock_span = self._enabled_exporter()
        exporter.export_event({"event_type": "test", "detail": {"tool": "bash", "exit_code": 0}})
        calls = {c[0][0]: c[0][1] for c in mock_span.set_attribute.call_args_list}
        assert "missy.tool" in calls
        assert "missy.exit_code" in calls

    def test_export_event_exception_does_not_propagate(self):
        exporter = OtelExporter.__new__(OtelExporter)
        exporter._enabled = True
        mock_tracer = MagicMock()
        mock_tracer.start_as_current_span.side_effect = RuntimeError("tracer error")
        exporter._tracer = mock_tracer
        exporter.export_event({"event_type": "test"})  # Must not raise

    def test_export_event_skips_non_scalar_values(self):
        exporter, _, mock_span = self._enabled_exporter()
        exporter.export_event({"event_type": "test", "complex": [1, 2, 3]})
        attr_names = [c[0][0] for c in mock_span.set_attribute.call_args_list]
        assert "missy.complex" not in attr_names


class TestOtelExporterSubscribe:
    def test_subscribe_registers_handler_on_event_bus(self):
        exporter = OtelExporter.__new__(OtelExporter)
        exporter._enabled = True
        exporter._tracer = MagicMock()

        mock_bus = MagicMock()
        # event_bus is imported inside subscribe(); patch at its source module
        with patch("missy.core.events.event_bus", mock_bus):
            exporter.subscribe()

        mock_bus.subscribe.assert_called_once()

    def test_subscribe_handler_calls_export_event(self):
        exporter = OtelExporter.__new__(OtelExporter)
        exporter._enabled = True
        exporter._tracer = MagicMock()
        exporter.export_event = MagicMock()

        captured_handler = []
        mock_bus = MagicMock()
        mock_bus.subscribe.side_effect = lambda h: captured_handler.append(h)

        with patch("missy.core.events.event_bus", mock_bus):
            exporter.subscribe()

        assert len(captured_handler) == 1
        fake_event = MagicMock()
        fake_event.__dict__ = {"event_type": "tool_call", "session_id": "s1"}
        captured_handler[0](fake_event)
        exporter.export_event.assert_called_once_with(
            {"event_type": "tool_call", "session_id": "s1"}
        )

    def test_subscribe_failure_does_not_propagate(self):
        exporter = OtelExporter.__new__(OtelExporter)
        exporter._enabled = True
        exporter._tracer = MagicMock()

        # Simulate the import inside subscribe() raising an exception
        with patch("builtins.__import__", side_effect=ImportError("no events module")):
            exporter.subscribe()  # Must not raise

    def test_subscribe_handler_uses_empty_dict_when_event_has_no_dict(self):
        exporter = OtelExporter.__new__(OtelExporter)
        exporter._enabled = True
        exporter._tracer = MagicMock()
        exporter.export_event = MagicMock()

        captured_handler = []
        mock_bus = MagicMock()
        mock_bus.subscribe.side_effect = lambda h: captured_handler.append(h)

        with patch("missy.core.events.event_bus", mock_bus):
            exporter.subscribe()

        # Event object with no __dict__ (e.g. a plain string)
        captured_handler[0]("plain_string_event")
        exporter.export_event.assert_called_once_with({})


class TestInitOtel:
    def test_returns_disabled_stub_when_otel_not_enabled(self):
        config = MagicMock()
        config.observability.otel_enabled = False
        exporter = init_otel(config)
        # init_otel returns a bare OtelExporter.__new__() stub (no __init__ called)
        # when otel is disabled; it is the right type but has no _enabled attribute set.
        assert isinstance(exporter, OtelExporter)
        assert not hasattr(exporter, "_enabled")

    def test_returns_disabled_stub_when_no_observability_attr(self):
        config = MagicMock(spec=[])  # no observability attribute
        exporter = init_otel(config)
        assert isinstance(exporter, OtelExporter)
        assert not hasattr(exporter, "_enabled")

    def test_creates_exporter_with_config_values_when_enabled(self):
        config = MagicMock()
        config.observability.otel_enabled = True
        config.observability.otel_endpoint = "http://collector:4317"
        config.observability.otel_protocol = "grpc"
        config.observability.otel_service_name = "my-service"

        mock_exporter = MagicMock(spec=OtelExporter)
        mock_exporter.is_enabled = True

        with patch("missy.observability.otel.OtelExporter", return_value=mock_exporter) as mock_cls:
            result = init_otel(config)

        mock_cls.assert_called_once_with(
            endpoint="http://collector:4317",
            protocol="grpc",
            service_name="my-service",
        )
        mock_exporter.subscribe.assert_called_once()
        assert result is mock_exporter

    def test_does_not_subscribe_when_exporter_not_enabled(self):
        config = MagicMock()
        config.observability.otel_enabled = True
        config.observability.otel_endpoint = "http://collector:4317"
        config.observability.otel_protocol = "grpc"
        config.observability.otel_service_name = "my-service"

        mock_exporter = MagicMock(spec=OtelExporter)
        mock_exporter.is_enabled = False

        with patch("missy.observability.otel.OtelExporter", return_value=mock_exporter):
            init_otel(config)

        mock_exporter.subscribe.assert_not_called()
