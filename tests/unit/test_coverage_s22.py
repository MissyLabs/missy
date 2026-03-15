"""Session 22 coverage gap tests.

Targets remaining uncovered lines across multiple modules.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# VoiceChannel event loop error path (channel.py:249-250)
# ---------------------------------------------------------------------------


class TestVoiceChannelLoopError:
    """Voice channel _run_loop exception handler (line 249-250)."""

    def test_voice_channel_start_failure_raises(self) -> None:
        """When the voice server fails to start, RuntimeError is raised."""
        from missy.channels.voice.channel import VoiceChannel

        ch = VoiceChannel(
            host="127.0.0.1",
            port=8765,
        )

        mock_runtime = MagicMock()

        with (
            patch("missy.channels.voice.channel.VoiceServer") as MockServer,
            patch("missy.channels.voice.channel.DeviceRegistry"),
            patch("missy.channels.voice.channel.PairingManager"),
            patch("missy.channels.voice.channel.PresenceStore"),
        ):
            server_inst = MagicMock()
            server_inst.start = AsyncMock(side_effect=OSError("address in use"))
            server_inst._running = False
            MockServer.return_value = server_inst

            with pytest.raises(RuntimeError, match="address in use"):
                ch.start(mock_runtime)


# ---------------------------------------------------------------------------
# Discord channel — agent_runtime attribute
# ---------------------------------------------------------------------------


class TestDiscordAgentRuntime:
    """Verify set_agent_runtime stores runtime reference."""

    def test_set_agent_runtime_stores_reference(self) -> None:
        from missy.channels.discord.channel import DiscordChannel

        mock_account = MagicMock()
        mock_account.token_env_var = "DISCORD_BOT_TOKEN"
        mock_account.gateway_intents = 513

        with patch.dict("os.environ", {"DISCORD_BOT_TOKEN": "fake-token"}):
            ch = DiscordChannel(account_config=mock_account)

            mock_runtime = MagicMock()
            mock_runtime.run.return_value = "agent response"
            ch.set_agent_runtime(mock_runtime)

            assert ch._agent_runtime is mock_runtime


# ---------------------------------------------------------------------------
# Wizard / OAuth / Anthropic auth imports
# ---------------------------------------------------------------------------


class TestModuleImports:
    def test_wizard_module_imports(self) -> None:
        from missy.cli import wizard  # noqa: F401

    def test_oauth_module_imports(self) -> None:
        from missy.cli import oauth  # noqa: F401

    def test_anthropic_auth_module_imports(self) -> None:
        from missy.cli import anthropic_auth  # noqa: F401


# ---------------------------------------------------------------------------
# Voice registry save error (registry.py:184-188)
# ---------------------------------------------------------------------------


class TestVoiceRegistryAtomicWriteFailure:
    """save() tempfile creation failure path."""

    def test_save_handles_permission_error(self, tmp_path: Path) -> None:
        from missy.channels.voice.registry import DeviceRegistry, EdgeNode

        reg = DeviceRegistry(str(tmp_path / "devices.json"))

        node = EdgeNode(
            node_id="test-node",
            friendly_name="Test",
            token_hash="abc",
            room="Room",
            ip_address="192.168.1.1",
        )
        reg._nodes["test-node"] = node

        with patch("tempfile.mkstemp", side_effect=PermissionError("denied")):
            with pytest.raises(PermissionError):
                reg.save()


# ---------------------------------------------------------------------------
# Voice server — import check
# ---------------------------------------------------------------------------


class TestVoiceServerEdgeCases:
    def test_voice_server_importable(self) -> None:
        from missy.channels.voice import server  # noqa: F401


# ---------------------------------------------------------------------------
# Network policy — ValueError from ipaddress (network.py:157-158)
# ---------------------------------------------------------------------------


class TestNetworkPolicyIPParseError:
    """Lines 157-158: ipaddress.ip_address raises ValueError."""

    def test_unparseable_resolved_ip_is_skipped(self) -> None:
        from missy.config.settings import NetworkPolicy
        from missy.policy.network import NetworkPolicyEngine

        policy = NetworkPolicy(
            default_deny=True,
            allowed_domains=["example.com"],
        )
        engine = NetworkPolicyEngine(policy)

        # Mock getaddrinfo to return an unparseable IP
        fake_infos = [
            (2, 1, 6, "", ("not-an-ip", 80)),
            (2, 1, 6, "", ("93.184.216.34", 80)),
        ]
        with patch("socket.getaddrinfo", return_value=fake_infos):
            result = engine.check_host("example.com")

        # example.com is in allowed_domains, so this should pass
        assert result is True


# ---------------------------------------------------------------------------
# Discord voice commands — async function
# ---------------------------------------------------------------------------


class TestDiscordVoiceCommands:
    """Test maybe_handle_voice_command for unrecognized commands."""

    def test_non_bang_command_not_handled(self) -> None:
        from missy.channels.discord.voice_commands import maybe_handle_voice_command

        result = asyncio.run(
            maybe_handle_voice_command(
                content="hello there",
                channel_id="123",
                guild_id="456",
                author_id="789",
                voice=None,
            )
        )
        assert result.handled is False

    def test_unrecognized_bang_command(self) -> None:
        from missy.channels.discord.voice_commands import maybe_handle_voice_command

        result = asyncio.run(
            maybe_handle_voice_command(
                content="!nonexistent_cmd",
                channel_id="123",
                guild_id="456",
                author_id="789",
                voice=None,
            )
        )
        # Unrecognized ! commands should not be handled
        assert result.handled is False


# ---------------------------------------------------------------------------
# Vault crypto unavailable (vault.py:25-26)
# ---------------------------------------------------------------------------


class TestVaultCryptoUnavailable:
    """Lines 25-26: ChaCha20Poly1305 import fails."""

    def test_vault_raises_without_crypto(self, tmp_path: Path) -> None:
        from missy.security.vault import Vault, VaultError

        with patch("missy.security.vault._CRYPTO_AVAILABLE", False):
            with pytest.raises(VaultError, match="cryptography"):
                Vault(str(tmp_path / "secrets"))


# ---------------------------------------------------------------------------
# Code evolution manager — import check
# ---------------------------------------------------------------------------


class TestCodeEvolutionManager:
    def test_code_evolution_importable(self) -> None:
        from missy.agent.code_evolution import CodeEvolutionManager  # noqa: F401

    def test_evolution_status_values(self) -> None:
        from missy.agent.code_evolution import EvolutionStatus

        assert "proposed" in [s.value for s in EvolutionStatus]
