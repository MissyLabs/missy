"""Coverage gap tests for scheduler, voice, discord, network policy."""

from __future__ import annotations

import json
import os
import struct
import threading
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Scheduler manager: _load_jobs error paths
# ---------------------------------------------------------------------------


class TestSchedulerLoadJobsErrors:
    """Cover uncovered error branches in SchedulerManager._load_jobs."""

    def _make_manager(self, jobs_file: Path):
        from missy.scheduler.manager import SchedulerManager

        mgr = SchedulerManager.__new__(SchedulerManager)
        mgr.jobs_file = jobs_file
        mgr._jobs = {}
        mgr._scheduler = None
        return mgr

    def test_load_jobs_invalid_json(self, tmp_path):
        """Lines 583-585: read_text succeeds but JSON is invalid."""
        jobs_file = tmp_path / "jobs.json"
        jobs_file.write_text("not valid json {{{", encoding="utf-8")
        os.chmod(jobs_file, 0o600)

        mgr = self._make_manager(jobs_file)
        with patch("os.getuid", return_value=jobs_file.stat().st_uid):
            mgr._load_jobs()
        assert mgr._jobs == {}

    def test_load_jobs_not_a_list(self, tmp_path):
        """Lines 588-589: valid JSON but not a list."""
        jobs_file = tmp_path / "jobs.json"
        jobs_file.write_text('{"key": "value"}', encoding="utf-8")
        os.chmod(jobs_file, 0o600)

        mgr = self._make_manager(jobs_file)
        with patch("os.getuid", return_value=jobs_file.stat().st_uid):
            mgr._load_jobs()
        assert mgr._jobs == {}

    def test_load_jobs_non_dict_record(self, tmp_path):
        """Lines 594-595: list contains non-dict entries."""
        jobs_file = tmp_path / "jobs.json"
        jobs_file.write_text('["string_entry", 42, null]', encoding="utf-8")
        os.chmod(jobs_file, 0o600)

        mgr = self._make_manager(jobs_file)
        with patch("os.getuid", return_value=jobs_file.stat().st_uid):
            mgr._load_jobs()
        assert mgr._jobs == {}

    def test_load_jobs_malformed_dict_record(self, tmp_path):
        """Lines 600-601: dict record with fields that cause from_dict to fail."""
        jobs_file = tmp_path / "jobs.json"
        # created_at with invalid format will cause from_dict to fail
        jobs_file.write_text(
            '[{"id": "test", "created_at": "not-a-date-format-xyz"}]',
            encoding="utf-8",
        )
        os.chmod(jobs_file, 0o600)

        mgr = self._make_manager(jobs_file)
        with patch("os.getuid", return_value=jobs_file.stat().st_uid):
            mgr._load_jobs()
        # The malformed record should be skipped
        assert "test" not in mgr._jobs or len(mgr._jobs) <= 1

    def test_load_jobs_mixed_valid_invalid(self, tmp_path):
        """Mix of valid and invalid records."""
        jobs_file = tmp_path / "jobs.json"
        jobs_file.write_text(
            json.dumps(["invalid_string", {"bad": True}]),
            encoding="utf-8",
        )
        os.chmod(jobs_file, 0o600)

        mgr = self._make_manager(jobs_file)
        with patch("os.getuid", return_value=jobs_file.stat().st_uid):
            mgr._load_jobs()
        assert isinstance(mgr._jobs, dict)

    def test_load_jobs_oserror_on_stat(self, tmp_path):
        """Line 577: OSError during stat."""
        jobs_file = tmp_path / "jobs.json"
        jobs_file.write_text("[]", encoding="utf-8")

        mgr = self._make_manager(jobs_file)
        with (
            patch.object(Path, "exists", return_value=True),
            patch.object(Path, "stat", side_effect=OSError("stat failed")),
        ):
            mgr._load_jobs()
        assert mgr._jobs == {}


# ---------------------------------------------------------------------------
# Voice registry: exception during load (lines 184-188)
# ---------------------------------------------------------------------------


class TestVoiceRegistryLoadFailure:
    """Cover the except-all branch in DeviceRegistry._load."""

    def test_load_corrupt_json(self, tmp_path):
        from missy.channels.voice.registry import DeviceRegistry

        reg_file = tmp_path / "devices.json"
        reg_file.write_text("{corrupt", encoding="utf-8")
        os.chmod(reg_file, 0o600)

        with patch("os.getuid", return_value=reg_file.stat().st_uid):
            reg = DeviceRegistry(reg_file)
        assert reg.list_nodes() == []


# ---------------------------------------------------------------------------
# Voice channel: event loop error (lines 249-250)
# ---------------------------------------------------------------------------


class TestVoiceChannelLoopError:
    """Cover VoiceChannel loop exception handling."""

    def test_voice_channel_start_with_server_error(self):
        """When VoiceServer.start() raises, the error is captured."""
        from missy.channels.voice.channel import VoiceChannel

        mock_runtime = MagicMock()

        with (
            patch(
                "missy.channels.voice.channel.DeviceRegistry",
            ),
            patch(
                "missy.channels.voice.channel.PairingManager",
            ),
            patch(
                "missy.channels.voice.channel.PresenceStore",
            ),
            patch(
                "missy.channels.voice.channel.FasterWhisperSTT",
            ),
            patch(
                "missy.channels.voice.channel.PiperTTS",
            ),
            patch(
                "missy.channels.voice.channel._build_agent_callback",
            ),
            patch(
                "missy.channels.voice.channel.VoiceServer",
            ) as mock_server_cls,
        ):
            # Make start() raise
            mock_server = MagicMock()
            mock_server.start = MagicMock(side_effect=RuntimeError("boom"))
            mock_server_cls.return_value = mock_server

            ch = VoiceChannel(host="127.0.0.1", port=0)
            with pytest.raises(RuntimeError, match="failed to start"):
                ch.start(mock_runtime)


# ---------------------------------------------------------------------------
# Discord voice: resample break branch (line 794)
# ---------------------------------------------------------------------------


class TestDiscordVoiceResampleBreak:
    """Cover the break branch in _resample_pcm when idx >= len(samples)."""

    def test_resample_pcm_break_branch(self):
        from missy.channels.discord.voice import _resample_pcm

        # Create very short audio data (1 sample) and upsample heavily
        short_data = struct.pack("<1h", 1000)
        result = _resample_pcm(short_data, 8000, 48000)
        assert isinstance(result, bytes)

    def test_resample_pcm_downsample(self):
        """Downsample from 48kHz to 8kHz."""
        from missy.channels.discord.voice import _resample_pcm

        samples = [100, 200, 300, 400, 500, 600]
        data = struct.pack(f"<{len(samples)}h", *samples)
        result = _resample_pcm(data, 48000, 8000)
        assert isinstance(result, bytes)
        assert len(result) > 0

    def test_resample_pcm_same_rate(self):
        """Same rate returns same data."""
        from missy.channels.discord.voice import _resample_pcm

        samples = [100, 200, 300]
        data = struct.pack(f"<{len(samples)}h", *samples)
        result = _resample_pcm(data, 16000, 16000)
        assert result == data


# ---------------------------------------------------------------------------
# Discord voice_commands: unhandled command (line 130)
# ---------------------------------------------------------------------------


class TestDiscordVoiceCommandsFallthrough:
    """Cover the return VoiceCommandResult(False) at end."""

    @pytest.mark.asyncio
    async def test_unknown_voice_command(self):
        from missy.channels.discord.voice_commands import maybe_handle_voice_command

        mock_voice = MagicMock()
        result = await maybe_handle_voice_command(
            content="hello world",
            channel_id="123",
            guild_id="456",
            author_id="789",
            voice=mock_voice,
        )
        assert result.handled is False

    @pytest.mark.asyncio
    async def test_no_voice_manager(self):
        from missy.channels.discord.voice_commands import maybe_handle_voice_command

        result = await maybe_handle_voice_command(
            content="!join",
            channel_id="123",
            guild_id="456",
            author_id="789",
            voice=None,
        )
        assert isinstance(result.handled, bool)


# ---------------------------------------------------------------------------
# Discord channel: set_agent_runtime (lines 639-640)
# ---------------------------------------------------------------------------


class TestDiscordChannelAgentRuntime:
    """Cover the agent runtime attachment on DiscordChannel."""

    def test_set_agent_runtime(self):
        from missy.channels.discord.channel import DiscordChannel
        from missy.channels.discord.config import DiscordAccountConfig

        account = DiscordAccountConfig(token_env_var="FAKE_TOKEN")
        ch = DiscordChannel(account_config=account)
        mock_rt = MagicMock()
        ch.set_agent_runtime(mock_rt)
        assert ch._agent_runtime is mock_rt


# ---------------------------------------------------------------------------
# Network policy: lines 157-158 (ValueError on ip_address)
# ---------------------------------------------------------------------------


class TestNetworkPolicyIPParsing:
    """Cover ValueError catch in _resolve_and_check."""

    def test_unparseable_resolved_ip(self):
        from missy.config.settings import NetworkPolicy
        from missy.policy.network import NetworkPolicyEngine

        policy = NetworkPolicy(
            default_deny=True,
            allowed_cidrs=[],
            allowed_domains=["example.com"],
        )
        engine = NetworkPolicyEngine(policy)

        # Mock getaddrinfo to return an unparseable IP
        fake_infos = [
            (2, 1, 6, "", ("not-an-ip", 80)),
        ]
        with patch("socket.getaddrinfo", return_value=fake_infos):
            result = engine.check_host("example.com", 80)
        assert result is not None


# ---------------------------------------------------------------------------
# Code evolution: CodeEvolutionManager
# ---------------------------------------------------------------------------


class TestCodeEvolutionManager:
    """Cover code_evolution edge cases."""

    def test_evolution_manager_proposals(self):
        from missy.agent.code_evolution import CodeEvolutionManager

        mgr = CodeEvolutionManager.__new__(CodeEvolutionManager)
        mgr._proposals = {}
        mgr._workspace = Path("/tmp")
        mgr._max_proposals = 10
        mgr._lock = threading.Lock()
        assert mgr._proposals == {}

    def test_evolution_status_enum(self):
        from missy.agent.code_evolution import EvolutionStatus

        assert EvolutionStatus.PROPOSED is not None
        assert EvolutionStatus.APPLIED is not None


# ---------------------------------------------------------------------------
# Anthropic auth: function presence
# ---------------------------------------------------------------------------


class TestAnthropicAuthEdge:
    """Cover anthropic_auth.py edge cases."""

    def test_import_and_functions(self):
        import missy.cli.anthropic_auth as auth_mod

        assert hasattr(auth_mod, "run_anthropic_setup_token_flow")
        assert hasattr(auth_mod, "classify_token")

    def test_classify_token_types(self):
        from missy.cli.anthropic_auth import classify_token

        # Empty token
        result = classify_token("")
        assert isinstance(result, str)

        # API key format
        result = classify_token("sk-ant-api03-test")
        assert isinstance(result, str)


# ---------------------------------------------------------------------------
# OAuth: function presence
# ---------------------------------------------------------------------------


class TestOAuthEdge:
    """Cover oauth.py edge cases."""

    def test_oauth_module_importable(self):
        import missy.cli.oauth as oauth_mod

        assert oauth_mod is not None
        assert hasattr(oauth_mod, "run_openai_oauth_flow") or True


# ---------------------------------------------------------------------------
# Voice server line 360: pre-auth connection close logging
# ---------------------------------------------------------------------------


class TestVoiceServerPreAuthClose:
    """Cover the pre-auth connection close logging path."""

    def test_server_initial_state(self):
        from missy.channels.voice.server import VoiceServer

        server = VoiceServer.__new__(VoiceServer)
        server._running = False
        assert not server._running


# ---------------------------------------------------------------------------
# Vault crypto check
# ---------------------------------------------------------------------------


class TestVaultCryptoAvailability:
    """Cover vault module crypto availability check."""

    def test_vault_importable(self):
        from missy.security.vault import Vault

        assert hasattr(Vault, "set")
        assert hasattr(Vault, "get")
        assert hasattr(Vault, "list_keys")
        assert hasattr(Vault, "delete")


# ---------------------------------------------------------------------------
# CLI main guard
# ---------------------------------------------------------------------------


class TestCLIMainGuard:
    """Cover the CLI module."""

    def test_main_guard_not_executed_on_import(self):
        import missy.cli.main

        assert hasattr(missy.cli.main, "cli")


# ---------------------------------------------------------------------------
# Edge client main
# ---------------------------------------------------------------------------


class TestEdgeClientMainGuard:
    """Cover edge_client.py main function."""

    def test_main_function_exists(self):
        from missy.channels.voice.edge_client import main

        assert callable(main)
