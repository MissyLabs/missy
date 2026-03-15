"""Security fix tests for session 28.

Covers:
- MCP manager: block_injection default changed to True
- Voice server: connection limit, auth timeout, connection counter decrement
- Shell exec: command length limit
- Shell policy: eval/exec in launcher commands
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# MCP manager — block_injection default
# ---------------------------------------------------------------------------


class TestMCPBlockInjectionDefault:
    """MCP injection blocking should be enabled by default."""

    def test_default_block_injection_is_true(self) -> None:
        """McpManager should default to block_injection=True."""
        from missy.mcp.manager import McpManager

        McpManager.__new__(McpManager)
        # Invoke __init__ with default args
        with patch.object(McpManager, "__init__", lambda self, **kw: None):
            McpManager.__new__(McpManager)
        # Check the actual default in __init__ signature
        import inspect

        sig = inspect.signature(McpManager.__init__)
        assert sig.parameters["block_injection"].default is True

    def test_explicit_false_still_works(self) -> None:
        """Can explicitly set block_injection=False."""
        from missy.mcp.manager import McpManager

        mgr = McpManager(block_injection=False)
        assert mgr._block_injection is False

    def test_explicit_true_works(self) -> None:
        """Can explicitly set block_injection=True."""
        from missy.mcp.manager import McpManager

        mgr = McpManager(block_injection=True)
        assert mgr._block_injection is True


# ---------------------------------------------------------------------------
# Voice server — connection limit
# ---------------------------------------------------------------------------


class TestVoiceServerConnectionLimit:
    """Voice server must reject connections when at capacity."""

    def test_max_concurrent_connections_constant_defined(self) -> None:
        """_MAX_CONCURRENT_CONNECTIONS should be defined."""
        from missy.channels.voice.server import _MAX_CONCURRENT_CONNECTIONS

        assert _MAX_CONCURRENT_CONNECTIONS > 0
        assert _MAX_CONCURRENT_CONNECTIONS == 50

    def test_auth_timeout_constant_defined(self) -> None:
        """_AUTH_TIMEOUT_SECONDS should be defined."""
        from missy.channels.voice.server import _AUTH_TIMEOUT_SECONDS

        assert _AUTH_TIMEOUT_SECONDS > 0
        assert _AUTH_TIMEOUT_SECONDS == 10.0


# ---------------------------------------------------------------------------
# Shell exec — command length limit
# ---------------------------------------------------------------------------


class TestShellExecCommandLength:
    """Shell exec tool must reject oversized commands."""

    def test_command_length_limit_constant(self) -> None:
        """_MAX_COMMAND_LENGTH should be 8192."""
        from missy.tools.builtin.shell_exec import _MAX_COMMAND_LENGTH

        assert _MAX_COMMAND_LENGTH == 8192

    def test_oversized_command_rejected(self) -> None:
        """Command exceeding _MAX_COMMAND_LENGTH should return failure."""
        from missy.tools.builtin.shell_exec import ShellExecTool

        tool = ShellExecTool()
        long_cmd = "echo " + "x" * 9000
        result = tool.execute(command=long_cmd)
        assert result.success is False
        assert "maximum length" in result.error

    def test_command_at_limit_accepted(self) -> None:
        """Command exactly at _MAX_COMMAND_LENGTH should be accepted (not rejected)."""
        from missy.tools.builtin.shell_exec import _MAX_COMMAND_LENGTH, ShellExecTool

        tool = ShellExecTool()
        # This command is exactly at the limit — should NOT be rejected by length check
        # (it will be rejected by shell policy since shell is disabled by default)
        cmd = "x" * _MAX_COMMAND_LENGTH
        result = tool.execute(command=cmd)
        # Should not fail with "maximum length" error
        assert "maximum length" not in (result.error or "")

    def test_command_one_over_limit_rejected(self) -> None:
        """Command one byte over _MAX_COMMAND_LENGTH should be rejected."""
        from missy.tools.builtin.shell_exec import _MAX_COMMAND_LENGTH, ShellExecTool

        tool = ShellExecTool()
        cmd = "x" * (_MAX_COMMAND_LENGTH + 1)
        result = tool.execute(command=cmd)
        assert result.success is False
        assert "maximum length" in result.error


# ---------------------------------------------------------------------------
# Shell policy — eval/exec launcher commands
# ---------------------------------------------------------------------------


class TestShellPolicyLauncherCommands:
    """Shell policy should warn about eval/exec as launcher commands."""

    def test_eval_in_launcher_commands(self) -> None:
        """'eval' should be in _LAUNCHER_COMMANDS."""
        from missy.policy.shell import ShellPolicyEngine

        assert "eval" in ShellPolicyEngine._LAUNCHER_COMMANDS

    def test_exec_in_launcher_commands(self) -> None:
        """'exec' should be in _LAUNCHER_COMMANDS."""
        from missy.policy.shell import ShellPolicyEngine

        assert "exec" in ShellPolicyEngine._LAUNCHER_COMMANDS

    def test_eval_whitelisted_emits_warning(self) -> None:
        """When 'eval' is in allowed_commands, a warning should be logged."""
        import logging

        from missy.config.settings import ShellPolicy
        from missy.policy.shell import ShellPolicyEngine

        policy = ShellPolicy(enabled=True, allowed_commands=["eval"])
        engine = ShellPolicyEngine(policy)
        with patch.object(logging.getLogger("missy.policy.shell"), "warning") as mock_warn:
            engine.check_command("eval something", session_id="s1", task_id="t1")
            found_launcher_warning = any(
                "launcher" in str(call).lower() or "eval" in str(call).lower()
                for call in mock_warn.call_args_list
            )
            assert found_launcher_warning

    def test_exec_whitelisted_emits_warning(self) -> None:
        """When 'exec' is in allowed_commands, a warning should be logged."""
        import logging

        from missy.config.settings import ShellPolicy
        from missy.policy.shell import ShellPolicyEngine

        policy = ShellPolicy(enabled=True, allowed_commands=["exec"])
        engine = ShellPolicyEngine(policy)
        with patch.object(logging.getLogger("missy.policy.shell"), "warning") as mock_warn:
            engine.check_command("exec /bin/ls", session_id="s1", task_id="t1")
            found_launcher_warning = any(
                "launcher" in str(call).lower() or "exec" in str(call).lower()
                for call in mock_warn.call_args_list
            )
            assert found_launcher_warning


# ---------------------------------------------------------------------------
# Voice server — active_connections counter
# ---------------------------------------------------------------------------


class TestVoiceServerActiveConnections:
    """Voice server should track active connection count."""

    def _make_server(self) -> Any:
        from missy.channels.voice.server import VoiceServer

        srv = VoiceServer.__new__(VoiceServer)
        srv._running = True
        srv._connected_nodes = set()
        srv._active_connections = 0
        srv._registry = MagicMock()
        srv._pairing_manager = MagicMock()
        srv._presence_store = MagicMock()
        srv._stt = MagicMock()
        srv._tts = MagicMock()
        srv._agent_callback = MagicMock()
        srv._host = "127.0.0.1"
        srv._port = 8765
        srv._audio_chunk_size = 4096
        srv._debug_transcripts = False
        return srv

    @pytest.mark.asyncio
    async def test_connection_rejected_at_capacity(self) -> None:
        """When active_connections >= limit, new connections get 1013."""
        from missy.channels.voice.server import _MAX_CONCURRENT_CONNECTIONS

        srv = self._make_server()
        srv._active_connections = _MAX_CONCURRENT_CONNECTIONS

        ws = AsyncMock()
        ws.remote_address = ("1.2.3.4", 5678)

        await srv._handle_connection(ws)

        ws.close.assert_called_once()
        args = ws.close.call_args
        assert args[0][0] == 1013  # Server at capacity

        # Counter should not have incremented
        assert srv._active_connections == _MAX_CONCURRENT_CONNECTIONS

    @pytest.mark.asyncio
    async def test_connection_counter_decrements_on_close(self) -> None:
        """Active connections counter should decrement when handler exits."""
        srv = self._make_server()

        ws = AsyncMock()
        ws.remote_address = ("1.2.3.4", 5678)
        # Simulate connection closed before first frame
        import websockets.exceptions

        ws.recv.side_effect = websockets.exceptions.ConnectionClosed(None, None)

        await srv._handle_connection(ws)

        assert srv._active_connections == 0  # Incremented then decremented

    @pytest.mark.asyncio
    async def test_auth_timeout_decrements_counter(self) -> None:
        """Auth timeout should also decrement the connection counter."""
        srv = self._make_server()

        ws = AsyncMock()
        ws.remote_address = ("1.2.3.4", 5678)
        ws.recv = AsyncMock(side_effect=TimeoutError())

        # Patch asyncio.wait_for to raise TimeoutError
        with patch("missy.channels.voice.server.asyncio.wait_for", side_effect=TimeoutError):
            await srv._handle_connection(ws)

        assert srv._active_connections == 0
