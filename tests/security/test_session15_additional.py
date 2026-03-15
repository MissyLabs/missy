"""Session 15 additional security tests.

Tests for:
  - MCP config file atomic write with 0o600 permissions
  - Web fetch header sanitization
  - Tool registry audit event redaction
  - Gateway kwargs edge cases
"""

from __future__ import annotations

import json
import stat
from unittest.mock import MagicMock, patch

# ---------------------------------------------------------------------------
# MCP config file permissions (L2 fix)
# ---------------------------------------------------------------------------


class TestMcpConfigPermissions:
    def test_save_config_creates_file_with_0600(self, tmp_path):
        """MCP config saved with restrictive 0o600 permissions."""
        from missy.mcp.manager import McpManager

        config_path = tmp_path / "mcp.json"
        mgr = McpManager(config_path=str(config_path))

        # Add a mock client to have something to save
        mock_client = MagicMock()
        mock_client._command = "echo test"
        mock_client._url = None
        mgr._clients = {"test": mock_client}

        mgr._save_config()

        assert config_path.exists()
        mode = config_path.stat().st_mode
        assert mode & stat.S_IRWXG == 0, "Group permissions should be 0"
        assert mode & stat.S_IRWXO == 0, "Other permissions should be 0"
        assert mode & stat.S_IRUSR != 0, "Owner should have read"
        assert mode & stat.S_IWUSR != 0, "Owner should have write"

    def test_save_config_content_is_valid_json(self, tmp_path):
        """Saved config file contains valid JSON with expected structure."""
        from missy.mcp.manager import McpManager

        config_path = tmp_path / "mcp.json"
        mgr = McpManager(config_path=str(config_path))

        mock_client = MagicMock()
        mock_client._command = "npx server"
        mock_client._url = "http://localhost:3000"
        mgr._clients = {"myserver": mock_client}

        mgr._save_config()

        data = json.loads(config_path.read_text())
        assert isinstance(data, list)
        assert len(data) == 1
        assert data[0]["name"] == "myserver"
        assert data[0]["command"] == "npx server"
        assert data[0]["url"] == "http://localhost:3000"

    def test_save_config_creates_parent_dirs(self, tmp_path):
        """_save_config creates parent directories if they don't exist."""
        from missy.mcp.manager import McpManager

        config_path = tmp_path / "subdir" / "deep" / "mcp.json"
        mgr = McpManager(config_path=str(config_path))
        mgr._clients = {}

        mgr._save_config()
        assert config_path.exists()


# ---------------------------------------------------------------------------
# Web fetch header sanitization (M3 fix)
# ---------------------------------------------------------------------------


class TestWebFetchHeaderSanitization:
    def test_all_blocked_headers_stripped(self):
        """All security-sensitive headers are stripped."""
        from missy.tools.builtin.web_fetch import WebFetchTool

        tool = WebFetchTool()
        mock_resp = MagicMock()
        mock_resp.text = "ok"
        mock_resp.status_code = 200

        mock_client = MagicMock()
        mock_client.get.return_value = mock_resp

        blocked_headers = {
            "Host": "evil.com",
            "Authorization": "Bearer token",
            "Cookie": "session=abc",
            "X-Forwarded-For": "1.2.3.4",
            "X-Forwarded-Host": "evil",
            "X-Forwarded-Proto": "https",
            "X-Real-IP": "10.0.0.1",
            "Proxy-Authorization": "Basic creds",
        }

        with patch("missy.gateway.client.create_client", return_value=mock_client):
            tool.execute(url="https://example.com", headers=blocked_headers)

        # Get should be called without any headers (all blocked)
        call_args = mock_client.get.call_args
        assert "headers" not in (call_args.kwargs or {})

    def test_mixed_blocked_and_safe_headers(self):
        """Only safe headers pass through when mixed with blocked ones."""
        from missy.tools.builtin.web_fetch import WebFetchTool

        tool = WebFetchTool()
        mock_resp = MagicMock()
        mock_resp.text = "ok"
        mock_resp.status_code = 200

        mock_client = MagicMock()
        mock_client.get.return_value = mock_resp

        headers = {
            "Accept": "application/json",
            "Authorization": "Bearer secret",
            "X-Custom": "value",
        }

        with patch("missy.gateway.client.create_client", return_value=mock_client):
            tool.execute(url="https://example.com", headers=headers)

        call_args = mock_client.get.call_args
        passed_headers = call_args.kwargs.get("headers", {})
        assert "Accept" in passed_headers
        assert "X-Custom" in passed_headers
        assert "Authorization" not in passed_headers

    def test_case_insensitive_header_blocking(self):
        """Header blocking is case-insensitive."""
        from missy.tools.builtin.web_fetch import WebFetchTool

        tool = WebFetchTool()
        mock_resp = MagicMock()
        mock_resp.text = "ok"
        mock_resp.status_code = 200

        mock_client = MagicMock()
        mock_client.get.return_value = mock_resp

        headers = {
            "AUTHORIZATION": "Bearer secret",
            "host": "evil.com",
            "COOKIE": "x=y",
        }

        with patch("missy.gateway.client.create_client", return_value=mock_client):
            tool.execute(url="https://example.com", headers=headers)

        call_args = mock_client.get.call_args
        assert "headers" not in (call_args.kwargs or {})


# ---------------------------------------------------------------------------
# Tool registry audit redaction (L3 fix)
# ---------------------------------------------------------------------------


class TestToolRegistryAuditRedaction:
    def test_audit_event_detail_is_redacted(self):
        """Tool audit events redact secrets from detail messages."""
        from missy.tools.base import BaseTool, ToolPermissions, ToolResult
        from missy.tools.registry import ToolRegistry

        class LeakyTool(BaseTool):
            name = "leaky"
            description = "tool that leaks secrets"
            permissions = ToolPermissions()

            def execute(self, **kwargs):
                # Simulate a tool that returns a secret in its error
                return ToolResult(
                    success=False,
                    output=None,
                    error="Connection failed with key sk-ant-api03-SECRETKEY1234567890abcdef",
                )

        reg = ToolRegistry()
        reg.register(LeakyTool())

        events_published = []

        def capture_event(event):
            events_published.append(event)

        from missy.core.events import event_bus

        event_bus.subscribe("tool_execute", capture_event)
        try:
            result = reg.execute("leaky")
            assert result.success is False

            # The audit event should have the secret redacted
            assert len(events_published) > 0
            detail = events_published[-1].detail
            msg = detail.get("message", "")
            # The raw key should NOT appear in the audit event
            assert "SECRETKEY1234567890abcdef" not in msg
        finally:
            event_bus.unsubscribe("tool_execute", capture_event)


# ---------------------------------------------------------------------------
# Gateway kwargs edge cases
# ---------------------------------------------------------------------------


class TestGatewayKwargsEdgeCases:
    def test_empty_kwargs_returns_empty(self):
        from missy.gateway.client import PolicyHTTPClient

        assert PolicyHTTPClient._sanitize_kwargs({}) == {}

    def test_only_blocked_kwargs_returns_empty(self):
        from missy.gateway.client import PolicyHTTPClient

        result = PolicyHTTPClient._sanitize_kwargs({
            "verify": False,
            "base_url": "http://evil",
            "transport": "bypass",
            "auth": ("u", "p"),
            "follow_redirects": True,
        })
        assert result == {}

    def test_extensions_passes_through(self):
        """The 'extensions' kwarg should pass through."""
        from missy.gateway.client import PolicyHTTPClient

        result = PolicyHTTPClient._sanitize_kwargs({
            "extensions": {"trace": lambda x: x},
        })
        assert "extensions" in result

    def test_files_passes_through(self):
        """The 'files' kwarg should pass through for multipart uploads."""
        from missy.gateway.client import PolicyHTTPClient

        result = PolicyHTTPClient._sanitize_kwargs({
            "files": {"upload": ("file.txt", b"data")},
        })
        assert "files" in result
