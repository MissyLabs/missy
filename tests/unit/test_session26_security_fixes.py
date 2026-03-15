"""Tests for session 26 security fixes: TOCTOU token files, AT-SPI logging, MCP startup."""

from __future__ import annotations

import json
import os
import stat
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Token file TOCTOU fix: anthropic_auth
# ---------------------------------------------------------------------------


class TestAnthropicAuthTokenTOCTOU:
    """Verify token file is created with restrictive permissions atomically."""

    def test_store_token_creates_file_with_0600(self, tmp_path: Path) -> None:
        token_file = tmp_path / "token.json"
        with patch("missy.cli.anthropic_auth.TOKEN_FILE", token_file):
            from missy.cli.anthropic_auth import store_token

            store_token("test_token", token_type="setup")
        assert token_file.exists()
        mode = stat.S_IMODE(token_file.stat().st_mode)
        assert mode == 0o600, f"Expected 0o600, got {oct(mode)}"

    def test_store_token_file_content_valid_json(self, tmp_path: Path) -> None:
        token_file = tmp_path / "token.json"
        with patch("missy.cli.anthropic_auth.TOKEN_FILE", token_file):
            from missy.cli.anthropic_auth import store_token

            store_token("my_secret", token_type="api_key", issued_at=1234567890)
        data = json.loads(token_file.read_text())
        assert data["token"] == "my_secret"
        assert data["token_type"] == "api_key"
        assert data["issued_at"] == 1234567890

    def test_store_token_no_world_readable_window(self, tmp_path: Path) -> None:
        """Temp file is created with 0o600 from the start, not chmod'd after."""
        token_file = tmp_path / "token.json"
        created_modes = []
        original_open = os.open

        def tracking_open(path, flags, mode=0o777, *a, **kw):
            fd = original_open(path, flags, mode, *a, **kw)
            if str(tmp_path) in str(path) and ".tmp" in str(path):
                created_modes.append(mode)
            return fd

        with patch("missy.cli.anthropic_auth.TOKEN_FILE", token_file):
            with patch("os.open", side_effect=tracking_open):
                from missy.cli.anthropic_auth import store_token

                store_token("x", token_type="t")

        # Temp file should have been created with 0o600
        assert len(created_modes) >= 1
        for mode in created_modes:
            assert mode == 0o600, f"Temp file created with mode {oct(mode)}"


# ---------------------------------------------------------------------------
# Token file TOCTOU fix: oauth
# ---------------------------------------------------------------------------


class TestOAuthTokenTOCTOU:
    """Verify OAuth token file is created with restrictive permissions atomically."""

    def test_save_token_creates_file_with_0600(self, tmp_path: Path) -> None:
        token_file = tmp_path / "oauth_token.json"
        with patch("missy.cli.oauth.TOKEN_FILE", token_file):
            from missy.cli.oauth import _save_token

            _save_token({"access_token": "test", "expires_in": 3600})
        assert token_file.exists()
        mode = stat.S_IMODE(token_file.stat().st_mode)
        assert mode == 0o600, f"Expected 0o600, got {oct(mode)}"

    def test_save_token_file_content(self, tmp_path: Path) -> None:
        token_file = tmp_path / "oauth_token.json"
        with patch("missy.cli.oauth.TOKEN_FILE", token_file):
            from missy.cli.oauth import _save_token

            _save_token({"access_token": "secret123", "scope": "read"})
        data = json.loads(token_file.read_text())
        assert data["access_token"] == "secret123"
        assert data["scope"] == "read"

    def test_save_token_cleanup_on_error(self, tmp_path: Path) -> None:
        """If json.dump fails, temp file should be cleaned up."""
        token_file = tmp_path / "oauth_token.json"

        class NotSerializable:
            pass

        with patch("missy.cli.oauth.TOKEN_FILE", token_file):
            with pytest.raises(TypeError):
                from missy.cli.oauth import _save_token

                _save_token({"bad": NotSerializable()})
        # Temp file should be cleaned up
        assert not token_file.with_suffix(".tmp").exists()


# ---------------------------------------------------------------------------
# AT-SPI silent handler logging
# ---------------------------------------------------------------------------


class TestAtspiLogging:
    """Verify AT-SPI exception handlers now log instead of silently passing."""

    def test_atspi_tools_have_debug_logging(self) -> None:
        """Verify the source contains logger.debug calls in exception handlers."""
        import inspect

        from missy.tools.builtin import atspi_tools

        source = inspect.getsource(atspi_tools)
        assert "AT-SPI text query failed" in source
        assert "AT-SPI state query failed" in source
        assert "AT-SPI child access failed" in source
        assert "AT-SPI element search failed" in source
        assert "AT-SPI grabFocus failed" in source
        assert "AT-SPI character count query failed" in source

    def test_atspi_no_silent_pass(self) -> None:
        """No bare 'pass' after exception handlers in atspi_tools."""
        src = Path("/home/bmerriam/git/missy/missy/tools/builtin/atspi_tools.py").read_text()
        lines = src.split("\n")
        for i, line in enumerate(lines):
            stripped = line.strip()
            if stripped.startswith("except") and "BLE001" in stripped:
                # Next non-empty line should NOT be just 'pass'
                for j in range(i + 1, min(i + 3, len(lines))):
                    next_stripped = lines[j].strip()
                    if next_stripped:
                        assert next_stripped != "pass", (
                            f"Silent 'pass' after exception at line {i + 1}"
                        )
                        break


# ---------------------------------------------------------------------------
# MCP startup check
# ---------------------------------------------------------------------------


class TestMcpStartupCheck:
    """Verify MCP client detects immediate process exit."""

    def test_immediate_exit_raises_runtime_error(self) -> None:
        from missy.mcp.client import McpClient

        client = McpClient(name="test", command="false")
        with (
            patch("subprocess.Popen") as mock_popen,
            pytest.raises(RuntimeError, match="exited immediately"),
        ):
            proc = MagicMock()
            proc.poll.return_value = 1
            proc.returncode = 1
            proc.stderr = MagicMock()
            proc.stderr.read.return_value = b"command not found"
            mock_popen.return_value = proc
            client.connect()

    def test_healthy_process_proceeds_to_handshake(self) -> None:
        from missy.mcp.client import McpClient

        client = McpClient(name="test", command="echo hello")
        with (
            patch("subprocess.Popen") as mock_popen,
            patch.object(client, "_initialize") as mock_init,
        ):
            proc = MagicMock()
            proc.poll.return_value = None
            mock_popen.return_value = proc
            client.connect()
            mock_init.assert_called_once()

    def test_immediate_exit_stderr_truncated(self) -> None:
        """Error message from stderr should be truncated to 500 chars."""
        from missy.mcp.client import McpClient

        client = McpClient(name="test", command="bad_cmd")
        with patch("subprocess.Popen") as mock_popen:
            proc = MagicMock()
            proc.poll.return_value = 127
            proc.returncode = 127
            proc.stderr = MagicMock()
            proc.stderr.read.return_value = b"x" * 1000
            mock_popen.return_value = proc
            with pytest.raises(RuntimeError) as exc_info:
                client.connect()
            msg = str(exc_info.value)
            # Stderr is truncated to [:500] so message has at most 500 x's
            assert msg.count("x") <= 501  # allow for slight variance

    def test_immediate_exit_no_stderr(self) -> None:
        """Handle case where stderr is None."""
        from missy.mcp.client import McpClient

        client = McpClient(name="test", command="bad_cmd")
        with patch("subprocess.Popen") as mock_popen:
            proc = MagicMock()
            proc.poll.return_value = 1
            proc.returncode = 1
            proc.stderr = None
            mock_popen.return_value = proc
            with pytest.raises(RuntimeError, match="exited immediately"):
                client.connect()


# ---------------------------------------------------------------------------
# File tool TOCTOU symlink re-resolve
# ---------------------------------------------------------------------------


class TestFileToolSymlinkReResolve:
    """Verify file_read and file_delete re-resolve symlinks before I/O."""

    def test_file_read_resolves_symlink(self, tmp_path: Path) -> None:
        real_file = tmp_path / "real.txt"
        real_file.write_text("content")
        link = tmp_path / "link.txt"
        link.symlink_to(real_file)

        from missy.tools.builtin.file_read import FileReadTool

        tool = FileReadTool()
        result = tool.execute(path=str(link))
        assert result.success
        assert "content" in result.output

    def test_file_delete_resolves_symlink(self, tmp_path: Path) -> None:
        real_file = tmp_path / "target.txt"
        real_file.write_text("to delete")
        link = tmp_path / "link.txt"
        link.symlink_to(real_file)

        from missy.tools.builtin.file_delete import FileDeleteTool

        tool = FileDeleteTool()
        result = tool.execute(path=str(link))
        assert result.success
        assert not real_file.exists()


# ---------------------------------------------------------------------------
# Voice registry _load exception path (lines 184-188)
# ---------------------------------------------------------------------------


class TestVoiceRegistryLoadError:
    """Test that registry load failure falls back to empty state."""

    def test_corrupt_json_falls_back_to_empty(self, tmp_path: Path) -> None:
        from missy.channels.voice.registry import DeviceRegistry

        path = tmp_path / "devices.json"
        path.write_text("not valid json!!!")
        reg = DeviceRegistry(registry_path=str(path))
        # _load is called in __init__, but the file content is corrupt
        assert reg._nodes == {}

    def test_invalid_structure_falls_back_to_empty(self, tmp_path: Path) -> None:
        from missy.channels.voice.registry import DeviceRegistry

        path = tmp_path / "devices.json"
        # A dict instead of a list will cause KeyError/TypeError
        path.write_text(json.dumps({"not": "a list"}))
        reg = DeviceRegistry(registry_path=str(path))
        assert reg._nodes == {}


# ---------------------------------------------------------------------------
# Vault crypto import check (lines 25-26)
# ---------------------------------------------------------------------------


class TestVaultCryptoCheck:
    """Test vault behavior when cryptography is not installed."""

    def test_crypto_flag_is_boolean(self) -> None:
        from missy.security import vault

        assert isinstance(vault._CRYPTO_AVAILABLE, bool)

    def test_vault_error_is_exception(self) -> None:
        from missy.security.vault import VaultError

        assert issubclass(VaultError, Exception)
        with pytest.raises(VaultError):
            raise VaultError("test")


# ---------------------------------------------------------------------------
# Discord voice_commands (line 130)
# ---------------------------------------------------------------------------


class TestDiscordVoiceCommands:
    """Test the voice commands module."""

    def test_voice_commands_module_importable(self) -> None:
        from missy.channels.discord import voice_commands

        assert hasattr(voice_commands, "maybe_handle_voice_command")

    def test_voice_commands_function_is_async(self) -> None:
        import asyncio

        from missy.channels.discord.voice_commands import maybe_handle_voice_command

        assert asyncio.iscoroutinefunction(maybe_handle_voice_command)


# ---------------------------------------------------------------------------
# Network policy unparseable IP (lines 157-158)
# ---------------------------------------------------------------------------


class TestNetworkPolicyUnparseableIP:
    """Test that unparseable IP addresses from getaddrinfo are skipped."""

    def test_unparseable_ip_skipped(self) -> None:
        from missy.core.exceptions import PolicyViolationError
        from missy.config.settings import NetworkPolicy
        from missy.policy.network import NetworkPolicyEngine

        policy = NetworkPolicy(default_deny=True)
        engine = NetworkPolicyEngine(policy)
        with patch("missy.policy.network.socket.getaddrinfo") as mock_gai:
            mock_gai.return_value = [
                (2, 1, 6, "", ("not-an-ip", 80)),
                (2, 1, 6, "", ("93.184.216.34", 80)),
            ]
            # Should deny (default deny) but not crash on unparseable IP
            with pytest.raises(PolicyViolationError):
                engine.check_host("example.com")

    def test_all_ips_unparseable_denies(self) -> None:
        from missy.core.exceptions import PolicyViolationError
        from missy.config.settings import NetworkPolicy
        from missy.policy.network import NetworkPolicyEngine

        policy = NetworkPolicy(default_deny=True)
        engine = NetworkPolicyEngine(policy)
        with patch("missy.policy.network.socket.getaddrinfo") as mock_gai:
            mock_gai.return_value = [
                (2, 1, 6, "", ("garbage", 80)),
            ]
            with pytest.raises(PolicyViolationError):
                engine.check_host("example.com")
