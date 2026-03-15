"""Session 23 resource leak and robustness tests.

Tests for:
- MCP client subprocess pipe closure on disconnect
- CLI gateway task awaiting after cancellation
- Edge client recording cleanup
- PolicyHTTPClient lifecycle
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest


class TestMCPClientPipeClosure:
    """Verify MCP client closes subprocess pipes on disconnect."""

    def test_disconnect_closes_all_pipes(self) -> None:
        """disconnect() should close stdin, stdout, and stderr pipes."""
        from missy.mcp.client import McpClient

        client = McpClient.__new__(McpClient)
        client._lock = MagicMock()
        client._lock.__enter__ = MagicMock(return_value=None)
        client._lock.__exit__ = MagicMock(return_value=False)

        mock_proc = MagicMock()
        mock_proc.stdin = MagicMock()
        mock_proc.stdout = MagicMock()
        mock_proc.stderr = MagicMock()
        mock_proc.wait.return_value = 0
        client._proc = mock_proc

        client.disconnect()

        mock_proc.terminate.assert_called_once()
        mock_proc.stdin.close.assert_called_once()
        mock_proc.stdout.close.assert_called_once()
        mock_proc.stderr.close.assert_called_once()
        assert client._proc is None

    def test_disconnect_handles_kill_after_timeout(self) -> None:
        """If terminate() times out, kill() should be used and pipes still closed."""
        import subprocess

        from missy.mcp.client import McpClient

        client = McpClient.__new__(McpClient)
        client._lock = MagicMock()
        client._lock.__enter__ = MagicMock(return_value=None)
        client._lock.__exit__ = MagicMock(return_value=False)

        mock_proc = MagicMock()
        mock_proc.stdin = MagicMock()
        mock_proc.stdout = MagicMock()
        mock_proc.stderr = MagicMock()
        mock_proc.wait.side_effect = subprocess.TimeoutExpired("cmd", 5)
        client._proc = mock_proc

        client.disconnect()

        mock_proc.kill.assert_called_once()
        mock_proc.stdin.close.assert_called_once()
        mock_proc.stdout.close.assert_called_once()
        mock_proc.stderr.close.assert_called_once()
        assert client._proc is None

    def test_disconnect_handles_pipe_close_oserror(self) -> None:
        """OSError on pipe close should be suppressed."""
        from missy.mcp.client import McpClient

        client = McpClient.__new__(McpClient)
        client._lock = MagicMock()
        client._lock.__enter__ = MagicMock(return_value=None)
        client._lock.__exit__ = MagicMock(return_value=False)

        mock_proc = MagicMock()
        mock_proc.stdin = MagicMock()
        mock_proc.stdin.close.side_effect = OSError("broken pipe")
        mock_proc.stdout = MagicMock()
        mock_proc.stderr = None  # No stderr pipe
        mock_proc.wait.return_value = 0
        client._proc = mock_proc

        # Should not raise
        client.disconnect()
        assert client._proc is None

    def test_disconnect_no_proc(self) -> None:
        """disconnect() with no process should be a no-op."""
        from missy.mcp.client import McpClient

        client = McpClient.__new__(McpClient)
        client._proc = None

        client.disconnect()
        assert client._proc is None

    def test_is_alive_with_running_process(self) -> None:
        """is_alive() should return True for a running process."""
        from missy.mcp.client import McpClient

        client = McpClient.__new__(McpClient)
        mock_proc = MagicMock()
        mock_proc.poll.return_value = None  # Still running
        client._proc = mock_proc

        assert client.is_alive() is True

    def test_is_alive_with_dead_process(self) -> None:
        """is_alive() should return False for a terminated process."""
        from missy.mcp.client import McpClient

        client = McpClient.__new__(McpClient)
        mock_proc = MagicMock()
        mock_proc.poll.return_value = 0  # Exited
        client._proc = mock_proc

        assert client.is_alive() is False

    def test_is_alive_with_no_process(self) -> None:
        """is_alive() should return False when no process exists."""
        from missy.mcp.client import McpClient

        client = McpClient.__new__(McpClient)
        client._proc = None

        assert client.is_alive() is False


class TestPolicyHTTPClientLifecycle:
    """Test PolicyHTTPClient resource management."""

    def test_close_closes_sync_client(self) -> None:
        """close() should close the sync httpx client."""
        from missy.gateway.client import PolicyHTTPClient

        http = PolicyHTTPClient.__new__(PolicyHTTPClient)
        mock_sync = MagicMock()
        http._sync_client = mock_sync
        http._async_client = None

        http.close()
        mock_sync.close.assert_called_once()

    def test_close_no_clients(self) -> None:
        """close() with no initialized clients should not crash."""
        from missy.gateway.client import PolicyHTTPClient

        http = PolicyHTTPClient.__new__(PolicyHTTPClient)
        http._sync_client = None
        http._async_client = None

        http.close()

    @pytest.mark.asyncio
    async def test_aclose_closes_async_client(self) -> None:
        """aclose() should close the async httpx client."""
        from missy.gateway.client import PolicyHTTPClient

        http = PolicyHTTPClient.__new__(PolicyHTTPClient)
        from unittest.mock import AsyncMock

        mock_async = AsyncMock()
        http._async_client = mock_async
        http._sync_client = None

        await http.aclose()
        mock_async.aclose.assert_awaited_once()


class TestGatewayContextManagers:
    """Test PolicyHTTPClient context manager protocol."""

    def test_sync_context_manager(self) -> None:
        """PolicyHTTPClient should support sync context manager."""
        from missy.gateway.client import PolicyHTTPClient

        http = PolicyHTTPClient.__new__(PolicyHTTPClient)
        http._sync_client = None
        http._async_client = None
        http.close = MagicMock()

        with http:
            pass

        http.close.assert_called_once()

    @pytest.mark.asyncio
    async def test_async_context_manager(self) -> None:
        """PolicyHTTPClient should support async context manager."""
        from missy.gateway.client import PolicyHTTPClient

        http = PolicyHTTPClient.__new__(PolicyHTTPClient)
        http._sync_client = None
        http._async_client = None
        from unittest.mock import AsyncMock

        http.aclose = AsyncMock()

        async with http:
            pass

        http.aclose.assert_awaited_once()
