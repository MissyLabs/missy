"""Security hardening tests.


Tests for fixes applied this session:
  H2: File tools enforce filesystem policy on actual path from kwargs
  H3: Gateway kwargs allowlist (strips verify, base_url, transport, auth)
  H1: Shell heredoc (<<<) and brace group ({ }) rejection
  M5: Webhook requires Content-Type: application/json
  M6: Webhook validates Content-Length as non-negative integer
  H4: Webhook metadata header filtering
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from missy.gateway.client import PolicyHTTPClient
from missy.policy.shell import ShellPolicyEngine

# ---------------------------------------------------------------------------
# H2: File tools enforce filesystem policy on actual path
# ---------------------------------------------------------------------------


class TestFileToolPolicyEnforcement:
    """ToolRegistry._check_permissions must check the actual path argument
    from kwargs against the filesystem policy engine."""

    def test_file_read_tool_checks_path_kwarg(self):
        """When a file_read tool is executed, the path from kwargs is
        validated against check_read."""
        from missy.core.exceptions import PolicyViolationError
        from missy.tools.base import BaseTool, ToolPermissions, ToolResult
        from missy.tools.registry import ToolRegistry

        class FakeReadTool(BaseTool):
            name = "file_read"
            description = "read"
            permissions = ToolPermissions(filesystem_read=True)

            def execute(self, **kwargs):
                return ToolResult(success=True, output="data")

        reg = ToolRegistry()
        reg.register(FakeReadTool())

        engine = MagicMock()
        # check_read raises PolicyViolationError for denied paths
        engine.check_read.side_effect = PolicyViolationError(
            "Read denied: /etc/shadow", category="filesystem", detail="blocked"
        )

        with patch("missy.tools.registry.get_policy_engine", return_value=engine):
            result = reg.execute("file_read", path="/etc/shadow")

        assert result.success is False
        assert "denied" in (result.error or "").lower() or "shadow" in (result.error or "").lower()
        # Verify check_read was called with the actual path
        calls = [str(c) for c in engine.check_read.call_args_list]
        assert any("/etc/shadow" in c for c in calls)

    def test_file_write_tool_checks_path_kwarg(self):
        """When a file_write tool is executed, the path from kwargs is
        validated against check_write."""
        from missy.core.exceptions import PolicyViolationError
        from missy.tools.base import BaseTool, ToolPermissions, ToolResult
        from missy.tools.registry import ToolRegistry

        class FakeWriteTool(BaseTool):
            name = "file_write"
            description = "write"
            permissions = ToolPermissions(filesystem_write=True)

            def execute(self, **kwargs):
                return ToolResult(success=True, output="ok")

        reg = ToolRegistry()
        reg.register(FakeWriteTool())

        engine = MagicMock()
        engine.check_write.side_effect = PolicyViolationError(
            "Write denied: /etc/passwd", category="filesystem", detail="blocked"
        )

        with patch("missy.tools.registry.get_policy_engine", return_value=engine):
            result = reg.execute("file_write", path="/etc/passwd", content="evil")

        assert result.success is False
        assert "denied" in (result.error or "").lower() or "passwd" in (result.error or "").lower()
        calls = [str(c) for c in engine.check_write.call_args_list]
        assert any("/etc/passwd" in c for c in calls)

    def test_file_read_no_path_kwarg_skips_actual_path_check(self):
        """When no path kwarg is provided, only static allowed_paths are checked."""
        from missy.tools.base import BaseTool, ToolPermissions, ToolResult
        from missy.tools.registry import ToolRegistry

        class FakeReadTool(BaseTool):
            name = "file_read"
            description = "read"
            permissions = ToolPermissions(filesystem_read=True)

            def execute(self, **kwargs):
                return ToolResult(success=True, output="data")

        reg = ToolRegistry()
        reg.register(FakeReadTool())

        engine = MagicMock()
        engine.check_read.return_value = True

        with patch("missy.tools.registry.get_policy_engine", return_value=engine):
            result = reg.execute("file_read")

        # Should not crash; check_read not called because path is None/missing
        assert result.success is True


# ---------------------------------------------------------------------------
# H3: Gateway kwargs allowlist
# ---------------------------------------------------------------------------


class TestGatewayKwargsAllowlist:
    """_sanitize_kwargs must use an allowlist, not a blocklist."""

    def test_verify_false_stripped(self):
        """verify=False must not pass through (disables TLS verification)."""
        result = PolicyHTTPClient._sanitize_kwargs({"verify": False})
        assert "verify" not in result

    def test_base_url_stripped(self):
        """base_url could redirect all traffic to an attacker-controlled server."""
        result = PolicyHTTPClient._sanitize_kwargs({"base_url": "http://evil.com"})
        assert "base_url" not in result

    def test_transport_stripped(self):
        """transport could bypass the policy layer entirely."""
        result = PolicyHTTPClient._sanitize_kwargs({"transport": MagicMock()})
        assert "transport" not in result

    def test_auth_stripped(self):
        """auth could inject credentials into requests."""
        result = PolicyHTTPClient._sanitize_kwargs({"auth": ("user", "pass")})
        assert "auth" not in result

    def test_event_hooks_stripped(self):
        """event_hooks could intercept responses."""
        result = PolicyHTTPClient._sanitize_kwargs({"event_hooks": {"response": []}})
        assert "event_hooks" not in result

    def test_follow_redirects_stripped(self):
        """follow_redirects is always stripped (enforced as False on client)."""
        result = PolicyHTTPClient._sanitize_kwargs({"follow_redirects": True})
        assert "follow_redirects" not in result

    def test_allowed_keys_pass_through(self):
        """Keys in the allowlist pass through unchanged."""
        kwargs = {
            "headers": {"X-Custom": "1"},
            "params": {"q": "test"},
            "data": b"body",
            "json": {"key": "value"},
            "content": b"raw",
            "cookies": {"session": "abc"},
            "timeout": 30,
            "files": {"file": ("f.txt", b"data")},
        }
        result = PolicyHTTPClient._sanitize_kwargs(kwargs)
        assert result == kwargs

    def test_mixed_allowed_and_blocked(self):
        """Only allowed keys survive, blocked ones are dropped."""
        kwargs = {
            "headers": {"Accept": "json"},
            "verify": False,
            "base_url": "http://evil",
            "timeout": 5,
            "transport": "bypass",
        }
        result = PolicyHTTPClient._sanitize_kwargs(kwargs)
        assert set(result.keys()) == {"headers", "timeout"}


# ---------------------------------------------------------------------------
# H1: Shell heredoc and brace group rejection
# ---------------------------------------------------------------------------


class TestShellHeredocAndBraceRejection:
    """ShellPolicyEngine must reject <<< (here-strings) and brace groups."""

    @pytest.fixture
    def engine(self):
        from missy.config.settings import ShellPolicy

        policy = ShellPolicy(enabled=True, allowed_commands=["cat", "echo", "ls"])
        return ShellPolicyEngine(policy)

    def test_here_string_rejected(self, engine):
        """<<< is a here-string that can feed arbitrary input."""
        from missy.core.exceptions import PolicyViolationError

        with pytest.raises(PolicyViolationError):
            engine.check_command("cat <<< 'malicious'")

    def test_brace_group_rejected(self, engine):
        """{ cmd; } is a brace group that can execute compound statements."""
        from missy.core.exceptions import PolicyViolationError

        with pytest.raises(PolicyViolationError):
            engine.check_command("{ echo evil; rm -rf /; }")

    def test_brace_group_with_semicolon_rejected(self, engine):
        """{; is another form of brace group start."""
        from missy.core.exceptions import PolicyViolationError

        with pytest.raises(PolicyViolationError):
            engine.check_command("{;echo evil;}")

    def test_normal_command_still_allowed(self, engine):
        """Regular whitelisted commands still pass."""
        result = engine.check_command("ls -la")
        assert result is True

    def test_heredoc_redirect_rejected(self, engine):
        """<<( is already in subshell markers."""
        from missy.core.exceptions import PolicyViolationError

        with pytest.raises(PolicyViolationError):
            engine.check_command("cat <<(echo evil)")


# ---------------------------------------------------------------------------
# M5: Webhook Content-Type validation
# ---------------------------------------------------------------------------


class TestWebhookContentTypeValidation:
    """Webhook must reject requests without Content-Type: application/json."""

    def _make_handler(self, headers, body=None):
        """Create a minimal handler mock."""
        import io
        import json

        from missy.channels.webhook import WebhookChannel

        ch = WebhookChannel()

        captured = {}

        def fake_httpserver(addr, handler_class):
            captured["handler"] = handler_class
            return MagicMock()

        with (
            patch("missy.channels.webhook.HTTPServer", side_effect=fake_httpserver),
            patch("missy.channels.webhook.threading.Thread"),
        ):
            ch.start()

        HandlerClass = captured["handler"]

        if body is None:
            body = json.dumps({"prompt": "test"}).encode()

        rfile = io.BytesIO(body)
        wfile = io.BytesIO()

        mock_headers = {k.lower(): v for k, v in headers.items()}
        mock_headers_obj = MagicMock()
        mock_headers_obj.get = lambda k, default=None: mock_headers.get(k.lower(), default)
        mock_headers_obj.items = lambda: list(mock_headers.items())

        handler = HandlerClass.__new__(HandlerClass)
        handler.rfile = rfile
        handler.wfile = wfile
        handler.headers = mock_headers_obj
        handler.client_address = ("127.0.0.1", 12345)

        responses = []
        handler.send_response = lambda code: responses.append(code)
        handler.send_header = MagicMock()
        handler.end_headers = MagicMock()

        return handler, responses

    def test_missing_content_type_returns_415(self):
        body = b'{"prompt": "test"}'
        handler, responses = self._make_handler({"Content-Length": str(len(body))}, body)
        handler.do_POST()
        assert responses == [415]

    def test_wrong_content_type_returns_415(self):
        body = b'{"prompt": "test"}'
        handler, responses = self._make_handler(
            {"Content-Length": str(len(body)), "Content-Type": "text/plain"}, body
        )
        handler.do_POST()
        assert responses == [415]

    def test_correct_content_type_accepted(self):
        body = b'{"prompt": "test"}'
        handler, responses = self._make_handler(
            {"Content-Length": str(len(body)), "Content-Type": "application/json"}, body
        )
        handler.do_POST()
        assert responses == [202]

    def test_content_type_with_charset_accepted(self):
        body = b'{"prompt": "test"}'
        handler, responses = self._make_handler(
            {
                "Content-Length": str(len(body)),
                "Content-Type": "application/json; charset=utf-8",
            },
            body,
        )
        handler.do_POST()
        assert responses == [202]


# ---------------------------------------------------------------------------
# M6: Webhook Content-Length validation
# ---------------------------------------------------------------------------


class TestWebhookContentLengthValidation:
    def _make_handler(self, headers, body=b'{"prompt": "test"}'):
        import io

        from missy.channels.webhook import WebhookChannel

        ch = WebhookChannel()
        captured = {}

        def fake_httpserver(addr, handler_class):
            captured["handler"] = handler_class
            return MagicMock()

        with (
            patch("missy.channels.webhook.HTTPServer", side_effect=fake_httpserver),
            patch("missy.channels.webhook.threading.Thread"),
        ):
            ch.start()

        HandlerClass = captured["handler"]
        rfile = io.BytesIO(body)
        wfile = io.BytesIO()

        mock_headers = {k.lower(): v for k, v in headers.items()}
        mock_headers.setdefault("content-type", "application/json")
        mock_headers_obj = MagicMock()
        mock_headers_obj.get = lambda k, default=None: mock_headers.get(k.lower(), default)
        mock_headers_obj.items = lambda: list(mock_headers.items())

        handler = HandlerClass.__new__(HandlerClass)
        handler.rfile = rfile
        handler.wfile = wfile
        handler.headers = mock_headers_obj
        handler.client_address = ("127.0.0.1", 12345)

        responses = []
        handler.send_response = lambda code: responses.append(code)
        handler.send_header = MagicMock()
        handler.end_headers = MagicMock()

        return handler, responses

    def test_negative_content_length_returns_400(self):
        handler, responses = self._make_handler({"Content-Length": "-1"})
        handler.do_POST()
        assert responses == [400]

    def test_non_integer_content_length_returns_400(self):
        handler, responses = self._make_handler({"Content-Length": "abc"})
        handler.do_POST()
        assert responses == [400]

    def test_empty_content_length_returns_400(self):
        handler, responses = self._make_handler({"Content-Length": ""})
        handler.do_POST()
        assert responses == [400]


# ---------------------------------------------------------------------------
# H4: Webhook metadata header filtering
# ---------------------------------------------------------------------------


class TestWebhookHeaderFiltering:
    """Webhook must not store sensitive headers in message metadata."""

    def _make_handler_and_post(self, headers):
        import io
        import json

        from missy.channels.webhook import WebhookChannel

        ch = WebhookChannel()
        captured = {}

        def fake_httpserver(addr, handler_class):
            captured["handler"] = handler_class
            return MagicMock()

        with (
            patch("missy.channels.webhook.HTTPServer", side_effect=fake_httpserver),
            patch("missy.channels.webhook.threading.Thread"),
        ):
            ch.start()

        HandlerClass = captured["handler"]
        body = json.dumps({"prompt": "test"}).encode()
        rfile = io.BytesIO(body)
        wfile = io.BytesIO()

        mock_headers = {k.lower(): v for k, v in headers.items()}
        mock_headers.setdefault("content-type", "application/json")
        mock_headers.setdefault("content-length", str(len(body)))
        mock_headers_obj = MagicMock()
        mock_headers_obj.get = lambda k, default=None: mock_headers.get(k.lower(), default)
        mock_headers_obj.items = lambda: list(mock_headers.items())

        handler = HandlerClass.__new__(HandlerClass)
        handler.rfile = rfile
        handler.wfile = wfile
        handler.headers = mock_headers_obj
        handler.client_address = ("127.0.0.1", 12345)

        responses = []
        handler.send_response = lambda code: responses.append(code)
        handler.send_header = MagicMock()
        handler.end_headers = MagicMock()

        handler.do_POST()
        msg = ch.receive()
        return msg

    def test_authorization_header_stripped_from_metadata(self):
        msg = self._make_handler_and_post(
            {
                "Authorization": "Bearer secret-token",
                "Content-Type": "application/json",
            }
        )
        assert msg is not None
        stored_headers = msg.metadata.get("webhook_headers", {})
        assert "authorization" not in stored_headers
        assert "Authorization" not in stored_headers

    def test_cookie_header_stripped_from_metadata(self):
        msg = self._make_handler_and_post(
            {
                "Cookie": "session=abc123",
                "Content-Type": "application/json",
            }
        )
        assert msg is not None
        stored_headers = msg.metadata.get("webhook_headers", {})
        assert "cookie" not in stored_headers
        assert "Cookie" not in stored_headers

    def test_safe_headers_preserved_in_metadata(self):
        msg = self._make_handler_and_post(
            {
                "Content-Type": "application/json",
                "User-Agent": "TestBot/1.0",
                "X-Request-Id": "req-123",
            }
        )
        assert msg is not None
        stored_headers = msg.metadata.get("webhook_headers", {})
        assert stored_headers.get("content-type") == "application/json"
        assert stored_headers.get("user-agent") == "TestBot/1.0"
        assert stored_headers.get("x-request-id") == "req-123"

    def test_x_forwarded_for_stripped_from_metadata(self):
        msg = self._make_handler_and_post(
            {
                "X-Forwarded-For": "10.0.0.1",
                "Content-Type": "application/json",
            }
        )
        assert msg is not None
        stored_headers = msg.metadata.get("webhook_headers", {})
        assert "x-forwarded-for" not in stored_headers
