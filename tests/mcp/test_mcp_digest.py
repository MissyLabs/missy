"""Tests for MCP digest verification (Feature 3)."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from missy.mcp.digest import compute_tool_manifest_digest, verify_digest


class TestComputeDigest:
    """Tests for compute_tool_manifest_digest."""

    def test_compute_digest_deterministic(self):
        tools = [
            {"name": "read", "description": "Read a file"},
            {"name": "write", "description": "Write a file"},
        ]
        d1 = compute_tool_manifest_digest(tools)
        d2 = compute_tool_manifest_digest(tools)
        assert d1 == d2
        assert d1.startswith("sha256:")

    def test_compute_digest_order_independent(self):
        tools_a = [
            {"name": "read", "description": "Read a file"},
            {"name": "write", "description": "Write a file"},
        ]
        tools_b = [
            {"name": "write", "description": "Write a file"},
            {"name": "read", "description": "Read a file"},
        ]
        assert compute_tool_manifest_digest(tools_a) == compute_tool_manifest_digest(tools_b)

    def test_compute_digest_different_tools(self):
        tools_a = [{"name": "read", "description": "Read a file"}]
        tools_b = [{"name": "read", "description": "Read a DIFFERENT file"}]
        assert compute_tool_manifest_digest(tools_a) != compute_tool_manifest_digest(tools_b)

    def test_compute_digest_empty(self):
        d = compute_tool_manifest_digest([])
        assert d.startswith("sha256:")


class TestVerifyDigest:
    """Tests for verify_digest."""

    def test_matching(self):
        assert verify_digest("sha256:abc", "sha256:abc") is True

    def test_mismatch(self):
        assert verify_digest("sha256:abc", "sha256:def") is False


class TestDigestMismatchRejected:
    """Tests for digest verification in McpManager."""

    def test_digest_mismatch_rejected(self, tmp_path):
        """Server should be rejected when digest doesn't match."""
        from missy.mcp.manager import McpManager

        # Write config with a pinned digest
        config_file = tmp_path / "mcp.json"
        config_file.write_text(
            json.dumps([{"name": "test-server", "command": "echo", "digest": "sha256:wrong"}])
        )
        # Make config owned by current user and not group-writable
        import os

        os.chmod(str(config_file), 0o600)

        mgr = McpManager(config_path=str(config_file))

        # Mock the McpClient to return known tools
        mock_client = MagicMock()
        mock_client.tools = [{"name": "tool1", "description": "A tool"}]
        mock_client.is_alive.return_value = True

        with (
            patch("missy.mcp.manager.McpClient", return_value=mock_client),
            pytest.raises(ValueError, match="digest mismatch"),
        ):
            mgr.add_server("test-server", command="echo")

        # Client should have been disconnected
        mock_client.disconnect.assert_called_once()

    def test_no_digest_passes(self, tmp_path):
        """Server without pinned digest should connect normally."""
        from missy.mcp.manager import McpManager

        config_file = tmp_path / "mcp.json"
        config_file.write_text(json.dumps([{"name": "test-server", "command": "echo"}]))
        import os

        os.chmod(str(config_file), 0o600)

        mgr = McpManager(config_path=str(config_file))

        mock_client = MagicMock()
        mock_client.tools = [{"name": "tool1", "description": "A tool"}]
        mock_client.is_alive.return_value = True
        mock_client._command = "echo"
        mock_client._url = None

        with patch("missy.mcp.manager.McpClient", return_value=mock_client):
            client = mgr.add_server("test-server", command="echo")

        assert client is not None
        mock_client.disconnect.assert_not_called()
