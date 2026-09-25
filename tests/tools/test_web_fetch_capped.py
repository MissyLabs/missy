"""PERF-02 / DGAP-02: web_fetch streams a bounded prefix, closes its client,
and attributes its audit event to the calling session."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import httpx

from missy.gateway.client import PolicyHTTPClient
from missy.tools.base import _TOOL_CALL_CONTEXT, current_tool_context
from missy.tools.builtin.web_fetch import _MAX_RESPONSE_BYTES, WebFetchTool


def _client_with_body(body: bytes, reads: list[int]) -> PolicyHTTPClient:
    class CountingStream(httpx.SyncByteStream):
        def __iter__(self):
            for i in range(0, len(body), 4096):
                chunk = body[i : i + 4096]
                reads.append(len(chunk))
                yield chunk

    def handler(request):
        return httpx.Response(200, stream=CountingStream(), headers={"content-type": "text/plain"})

    client = PolicyHTTPClient(session_id="s", task_id="t")
    client._sync_client = httpx.Client(transport=httpx.MockTransport(handler))
    return client


def test_get_capped_stops_reading_after_limit():
    reads: list[int] = []
    client = _client_with_body(b"x" * 10_000_000, reads)
    with (
        patch.object(PolicyHTTPClient, "_check_url"),
        patch.object(PolicyHTTPClient, "_emit_request_event"),
    ):
        resp = client.get_capped("https://example.com/big", 65_536)
    assert resp.truncated is True
    assert len(resp.content) == 65_536
    assert sum(reads) < 200_000  # did not download the 10 MB body


def test_get_capped_small_body_not_truncated():
    client = _client_with_body(b"hello", [])
    with (
        patch.object(PolicyHTTPClient, "_check_url"),
        patch.object(PolicyHTTPClient, "_emit_request_event"),
    ):
        resp = client.get_capped("https://example.com/", 100)
    assert resp.truncated is False and resp.text == "hello"


def test_get_capped_enforces_policy():
    client = PolicyHTTPClient()
    with patch.object(PolicyHTTPClient, "_check_url", side_effect=PermissionError("denied")):
        try:
            client.get_capped("https://blocked.example/", 10)
        except PermissionError:
            pass
        else:  # pragma: no cover
            raise AssertionError("policy check not applied")


def test_web_fetch_uses_session_context_and_closes_client():
    resp = MagicMock(text="ok", status_code=200, truncated=False)
    client = MagicMock()
    client.get_capped.return_value = resp
    token = _TOOL_CALL_CONTEXT.set(("session-42", "task-7"))
    try:
        with patch("missy.gateway.client.create_client", return_value=client) as factory:
            result = WebFetchTool().execute(url="https://example.com")
    finally:
        _TOOL_CALL_CONTEXT.reset(token)
    assert result.success is True
    assert factory.call_args.kwargs["session_id"] == "session-42"
    assert factory.call_args.kwargs["task_id"] == "task-7"
    assert client.get_capped.call_args.args[1] == _MAX_RESPONSE_BYTES
    client.close.assert_called_once()


def test_client_closed_even_on_error():
    client = MagicMock()
    client.get_capped.side_effect = ConnectionError("down")
    with patch("missy.gateway.client.create_client", return_value=client):
        result = WebFetchTool().execute(url="https://example.com")
    assert result.success is False
    client.close.assert_called_once()


def test_registry_sets_tool_context():
    from missy.tools.base import BaseTool, ToolPermissions, ToolResult
    from missy.tools.registry import ToolRegistry

    seen = {}

    class Probe(BaseTool):
        name = "probe"
        description = "d"
        permissions = ToolPermissions()
        parameters = {}

        def execute(self, **kwargs):
            seen["ctx"] = current_tool_context()
            return ToolResult(success=True, output="ok")

    reg = ToolRegistry()
    reg.register(Probe())
    with patch.object(ToolRegistry, "_check_permissions"):
        reg.execute("probe", session_id="sess", task_id="task")
    assert seen["ctx"] == ("sess", "task")
    assert current_tool_context() == ("", "")
