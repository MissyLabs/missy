"""Coverage gap tests for voice_commands, network policy, and CLI.


Targets:
  voice_commands.py:130 — unrecognized voice command fallthrough
  network.py:157-158 — getaddrinfo returns unparseable address
  cli/main.py:2667 — __name__ == "__main__" entry point
"""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

# ---------------------------------------------------------------------------
# voice_commands.py line 130: unrecognized command fallthrough
# ---------------------------------------------------------------------------


class TestVoiceCommandFallthrough:
    @pytest.fixture
    def voice_mock(self):
        v = AsyncMock()
        v.is_ready = True
        return v

    @pytest.mark.asyncio
    async def test_unrecognized_bang_command_returns_not_handled(self, voice_mock):
        """Line 51: unknown !command returns VoiceCommandResult(False)."""
        from missy.channels.discord.voice_commands import maybe_handle_voice_command

        result = await maybe_handle_voice_command(
            content="!unknowncmd",
            channel_id="ch1",
            guild_id="123",
            author_id="456",
            voice=voice_mock,
        )
        assert result.handled is False

    @pytest.mark.asyncio
    async def test_regular_message_not_handled(self, voice_mock):
        """Line 44-45: Regular messages (not starting with !) return handled=False."""
        from missy.channels.discord.voice_commands import maybe_handle_voice_command

        result = await maybe_handle_voice_command(
            content="hello world",
            channel_id="ch1",
            guild_id="123",
            author_id="456",
            voice=voice_mock,
        )
        assert result.handled is False

    @pytest.mark.asyncio
    async def test_voice_none_returns_not_enabled(self):
        """Line 56-57: voice=None returns 'not enabled'."""
        from missy.channels.discord.voice_commands import maybe_handle_voice_command

        result = await maybe_handle_voice_command(
            content="!join",
            channel_id="ch1",
            guild_id="123",
            author_id="456",
            voice=None,
        )
        assert result.handled is True
        assert "not enabled" in (result.reply or "")


# ---------------------------------------------------------------------------
# voice_commands: !say with DiscordVoiceError
# ---------------------------------------------------------------------------


class TestVoiceCommandSayError:
    @pytest.mark.asyncio
    async def test_say_command_voice_error(self):
        """Lines 127-128: !say raises DiscordVoiceError."""
        from missy.channels.discord.voice import DiscordVoiceError
        from missy.channels.discord.voice_commands import maybe_handle_voice_command

        voice = AsyncMock()
        voice.is_ready = True
        voice.say.side_effect = DiscordVoiceError("TTS unavailable")

        result = await maybe_handle_voice_command(
            content="!say hello world",
            channel_id="ch1",
            guild_id="123",
            author_id="456",
            voice=voice,
        )
        assert result.handled is True
        assert "TTS unavailable" in (result.reply or "")


# ---------------------------------------------------------------------------
# network.py lines 157-158: unparseable IP from getaddrinfo
# ---------------------------------------------------------------------------


class TestNetworkPolicyUnparseableIP:
    def _make_engine(self, **kwargs):
        from missy.config.settings import NetworkPolicy
        from missy.policy.network import NetworkPolicyEngine

        policy = NetworkPolicy(**kwargs)
        return NetworkPolicyEngine(policy)

    def test_getaddrinfo_returns_non_ip_string(self):
        """Lines 157-158: ValueError from ip_address() is caught with continue."""
        engine = self._make_engine(
            default_deny=True,
            allowed_domains=["example.com"],
        )

        # Mock getaddrinfo to return a non-IP string
        fake_infos = [
            (2, 1, 6, "", ("not-an-ip", 443)),
        ]

        with patch("socket.getaddrinfo", return_value=fake_infos):
            result = engine.check_host("example.com")

        # Should still be allowed by domain rule, even though IP parsing failed
        assert result is True

    def test_getaddrinfo_mixed_valid_and_invalid_ips(self):
        """Mixed valid and invalid IPs — invalid ones are skipped."""
        engine = self._make_engine(
            default_deny=True,
            allowed_domains=["example.com"],
        )

        fake_infos = [
            (2, 1, 6, "", ("not-valid", 443)),
            (2, 1, 6, "", ("93.184.216.34", 443)),
        ]

        with patch("socket.getaddrinfo", return_value=fake_infos):
            result = engine.check_host("example.com")

        assert result is True


# ---------------------------------------------------------------------------
# CLI: __name__ == "__main__" entry point (line 2667)
# ---------------------------------------------------------------------------


class TestCLIMainEntrypoint:
    def test_cli_function_is_callable(self):
        """Line 2667: verify cli function exists and is a click command."""
        import missy.cli.main as cli_module

        assert callable(cli_module.cli)
        # Verify it's a click group
        assert hasattr(cli_module.cli, "commands")
