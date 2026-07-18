"""Tests for the MCP HTTP transport + auth (F17)."""

from __future__ import annotations

import contextlib
from unittest.mock import MagicMock, patch

import pytest

from missy.mcp.client import McpClient
from missy.mcp.manager import _resolve_mcp_auth_headers, _resolve_secret


def _resp(json_body=None, *, ct="application/json", text="", headers=None, status=200):
    r = MagicMock()
    r.status_code = status
    r.headers = {"Content-Type": ct, **(headers or {})}
    r.json.return_value = json_body or {}
    r.text = text
    r.raise_for_status = MagicMock()
    return r


def _fake_http(handler):
    client = MagicMock()
    client.post.side_effect = handler
    return client


def _standard_handler(*, session=None):
    def handler(url, json=None, headers=None, timeout=None):
        m = json.get("method")
        rid = json.get("id")
        if m == "initialize":
            return _resp(
                {"jsonrpc": "2.0", "id": rid, "result": {"protocolVersion": "2024-11-05"}},
                headers={"MCP-Session-Id": session} if session else None,
            )
        if m == "tools/list":
            return _resp(
                {
                    "jsonrpc": "2.0",
                    "id": rid,
                    "result": {"tools": [{"name": "echo", "description": "e", "inputSchema": {}}]},
                }
            )
        if m == "tools/call":
            return _resp(
                {
                    "jsonrpc": "2.0",
                    "id": rid,
                    "result": {"content": [{"type": "text", "text": "hi"}]},
                }
            )
        return _resp({"jsonrpc": "2.0", "id": rid, "result": {}})

    return handler


class TestHttpTransport:
    def test_connect_and_list_tools(self) -> None:
        with patch("httpx.Client", return_value=_fake_http(_standard_handler())):
            c = McpClient("remote", url="https://mcp.example.com/rpc")
            c.connect()
        assert [t["name"] for t in c._tools] == ["echo"]
        assert c.is_alive() is True

    def test_call_tool(self) -> None:
        with patch("httpx.Client", return_value=_fake_http(_standard_handler())):
            c = McpClient("remote", url="https://mcp.example.com/rpc")
            c.connect()
            assert c.call_tool("echo", {"msg": "x"}) == "hi"

    def test_session_id_captured_and_echoed(self) -> None:
        fake = _fake_http(_standard_handler(session="sess-42"))
        with patch("httpx.Client", return_value=fake):
            c = McpClient("remote", url="https://mcp.example.com/rpc")
            c.connect()
            c.call_tool("echo", {})
        assert c._session_id == "sess-42"
        # A later request carried the session header.
        later = [
            call
            for call in fake.post.call_args_list
            if call.kwargs["json"]["method"] == "tools/call"
        ][0]
        assert later.kwargs["headers"].get("MCP-Session-Id") == "sess-42"

    def test_auth_headers_installed_on_client(self) -> None:
        captured = {}

        def fake_client_ctor(**kwargs):
            captured.update(kwargs)
            return _fake_http(_standard_handler())

        with patch("httpx.Client", side_effect=fake_client_ctor):
            McpClient(
                "remote", url="https://x/rpc", headers={"Authorization": "Bearer tok"}
            ).connect()
        assert captured["headers"]["Authorization"] == "Bearer tok"

    def test_401_raises_clear_auth_error(self) -> None:
        def handler(url, json=None, headers=None, timeout=None):
            return _resp(status=401)

        with patch("httpx.Client", return_value=_fake_http(handler)):
            c = McpClient("remote", url="https://x/rpc")
            with pytest.raises(RuntimeError, match="auth failed"):
                c.connect()

    def test_sse_response_parsed(self) -> None:
        def handler(url, json=None, headers=None, timeout=None):
            rid = json.get("id")
            m = json.get("method")
            if m == "initialize":
                body = f'data: {{"jsonrpc":"2.0","id":"{rid}","result":{{}}}}'
                return _resp(ct="text/event-stream", text=body)
            if m == "tools/list":
                body = (
                    f"event: message\n"
                    f'data: {{"jsonrpc":"2.0","id":"{rid}","result":{{"tools":[]}}}}\n'
                )
                return _resp(ct="text/event-stream", text=body)
            return _resp({"jsonrpc": "2.0", "id": rid, "result": {}})

        with patch("httpx.Client", return_value=_fake_http(handler)):
            c = McpClient("remote", url="https://x/rpc")
            c.connect()  # must parse SSE bodies without error
        assert c._tools == []

    def test_disconnect_closes_http(self) -> None:
        fake = _fake_http(_standard_handler())
        with patch("httpx.Client", return_value=fake):
            c = McpClient("remote", url="https://x/rpc")
            c.connect()
            c.disconnect()
        fake.close.assert_called_once()
        assert c.is_alive() is False

    def test_missing_command_and_url_errors(self) -> None:
        with pytest.raises(RuntimeError, match="requires either a command or a url"):
            McpClient("bad").connect()


class TestAuthResolution:
    def test_bearer_token(self) -> None:
        h = _resolve_mcp_auth_headers({"url": "http://x", "bearer_token": "abc"})
        assert h == {"Authorization": "Bearer abc"}

    def test_custom_headers(self) -> None:
        h = _resolve_mcp_auth_headers({"url": "http://x", "headers": {"X-Api-Key": "k"}})
        assert h == {"X-Api-Key": "k"}

    def test_bearer_and_headers_combined(self) -> None:
        h = _resolve_mcp_auth_headers(
            {"url": "http://x", "bearer_token": "t", "headers": {"X-Extra": "v"}}
        )
        assert h == {"Authorization": "Bearer t", "X-Extra": "v"}

    def test_no_url_returns_none(self) -> None:
        assert _resolve_mcp_auth_headers({"command": "npx server"}) is None

    def test_no_auth_returns_none(self) -> None:
        assert _resolve_mcp_auth_headers({"url": "http://x"}) is None

    def test_vault_reference_resolved(self) -> None:
        with patch("missy.security.vault.Vault") as V:
            V.return_value.resolve.return_value = "secret-value"
            h = _resolve_mcp_auth_headers({"url": "http://x", "bearer_token": "vault://TOK"})
        assert h == {"Authorization": "Bearer secret-value"}

    def test_resolve_secret_plain_passthrough(self) -> None:
        with patch("missy.security.vault.Vault", side_effect=RuntimeError("no vault")):
            assert _resolve_secret("plain") == "plain"


class TestManagerWiring:
    def test_add_server_passes_headers(self) -> None:
        from missy.mcp.manager import McpManager

        mgr = McpManager.__new__(McpManager)
        mgr._clients = {}
        import threading

        mgr._lock = threading.Lock()
        with patch("missy.mcp.manager.McpClient") as MC:
            instance = MC.return_value
            instance.connect = MagicMock()
            instance.list_tools = MagicMock(return_value=[])
            instance.get_tools = MagicMock(return_value=[])
            # downstream digest/registry steps may need more setup; we only
            # care that McpClient received the auth headers.
            with contextlib.suppress(Exception):
                mgr.add_server("remote", url="http://x", headers={"Authorization": "Bearer t"})
        _, kwargs = MC.call_args
        assert kwargs["headers"] == {"Authorization": "Bearer t"}
