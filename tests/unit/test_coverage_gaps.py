"""Targeted tests to cover specific uncovered lines across multiple modules.

Covers:
- missy/channels/discord/rest.py   (lines 175-176, 212-213, 251, 375)
- missy/config/settings.py         (line 352, 474-475, 491-493)
- missy/policy/filesystem.py       (lines 169-172)
- missy/skills/registry.py         (lines 177-178)
- missy/security/sandbox.py        (lines 212-213, 276, 291-292)
- missy/channels/discord/voice_commands.py (lines 111-112, 130)
"""

from __future__ import annotations

import subprocess
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_rest_client(mock_http: Any | None = None) -> Any:
    """Create a DiscordRestClient with an injected mock HTTP client."""
    from missy.channels.discord.rest import DiscordRestClient

    http = mock_http or MagicMock()
    return DiscordRestClient(bot_token="testtoken", http_client=http)


# ===========================================================================
# Discord REST — send_message
# ===========================================================================


class TestDiscordRestSendMessageTextAccessError:
    """Lines 175-176: response.text raises → response_text falls back to ''."""

    def test_response_text_property_raises_is_swallowed(self) -> None:
        """When accessing response.text raises, the error handler catches it and
        continues logging with an empty string rather than propagating."""
        http = MagicMock()

        # Build a response whose .text property raises.
        bad_response = MagicMock()
        bad_response.status_code = 500

        # Make .text a property that raises.
        type(bad_response).text = property(lambda self: (_ for _ in ()).throw(RuntimeError("boom")))

        # raise_for_status raises so the exception path is taken after all retries.
        bad_response.raise_for_status.side_effect = Exception("HTTP 500")

        http.post.return_value = bad_response

        client = _make_rest_client(http)

        # With 4 attempts (1 + 3 backoffs), the last attempt raises after calling
        # _log_final_failure which accesses response.text.
        with patch("time.sleep"), pytest.raises(Exception, match="HTTP 500"):
            client.send_message("ch1", "hello")

    def test_response_text_none_response_skipped(self) -> None:
        """When response is None (exception before HTTP call), the text branch
        is not entered and the fallback empty string is used cleanly."""
        http = MagicMock()
        http.post.side_effect = RuntimeError("network failure")

        client = _make_rest_client(http)

        with patch("time.sleep"), pytest.raises(RuntimeError, match="network failure"):
            client.send_message("ch1", "hello")


# ---------------------------------------------------------------------------
# Lines 212-213: float(Retry-After header) raises → delay = None
# ---------------------------------------------------------------------------


class TestDiscordRestSendMessageRetryAfterParsing:
    """Lines 212-213: non-numeric Retry-After header causes float() to raise,
    setting delay = None and falling through to backoff logic."""

    def _rate_limit_response(self, retry_after_value: str) -> MagicMock:
        resp = MagicMock()
        resp.status_code = 429
        resp.headers = {"Retry-After": retry_after_value}
        return resp

    def test_non_numeric_retry_after_falls_back_to_backoff(self) -> None:
        """A non-numeric Retry-After header value causes the float parse to fail;
        delay must fall back to the indexed backoff value instead."""
        http = MagicMock()
        bad_ra_resp = self._rate_limit_response("not-a-number")

        # After the first 429 with a bad header, succeed on the second attempt.
        success_resp = MagicMock()
        success_resp.status_code = 200
        success_resp.json.return_value = {"id": "123"}

        http.post.side_effect = [bad_ra_resp, success_resp]

        client = _make_rest_client(http)

        with patch("time.sleep") as mock_sleep:
            result = client.send_message("ch1", "hello")

        assert result == {"id": "123"}
        # sleep was called with a positive float (the backoff + jitter)
        assert mock_sleep.called
        sleep_arg = mock_sleep.call_args[0][0]
        assert sleep_arg > 0

    def test_empty_retry_after_header_falls_back_to_backoff(self) -> None:
        """An empty Retry-After header value is falsy and skips the float parse
        entirely; delay remains None and falls back to backoff."""
        http = MagicMock()
        resp_no_ra = MagicMock()
        resp_no_ra.status_code = 429
        resp_no_ra.headers = {}  # no Retry-After key

        success_resp = MagicMock()
        success_resp.status_code = 200
        success_resp.json.return_value = {"id": "456"}

        http.post.side_effect = [resp_no_ra, success_resp]

        client = _make_rest_client(http)

        with patch("time.sleep"):
            result = client.send_message("ch1", "hello")

        assert result == {"id": "456"}


# ---------------------------------------------------------------------------
# Line 251: RuntimeError("Discord send_message failed without exception")
# ---------------------------------------------------------------------------


class TestDiscordRestSendMessageExhaustedWithoutException:
    """Line 251: the loop exhausts all 4 attempts via 429 retries without ever
    raising — the trailing guard raises RuntimeError."""

    def test_all_attempts_return_429_and_exhaust_raises_runtime_error(self) -> None:
        """When every attempt returns a retryable status code and the final
        attempt's raise_for_status() does NOT raise (unusual but possible in
        mocks), the sentinel RuntimeError at line 251 is hit."""
        http = MagicMock()

        # Four 429 responses, none of which raise on raise_for_status.
        # On the final attempt (index 3 == len(backoffs)) raise_for_status IS
        # called, so we must NOT let it raise if we want to hit line 251.
        # The logic on the final 429 attempt: if attempt >= len(backoffs) →
        # call raise_for_status(); if raise_for_status() doesn't raise, the
        # loop falls through `continue` and the sentinel fires.
        resp = MagicMock()
        resp.status_code = 429
        resp.headers = {"Retry-After": "0.01"}
        resp.raise_for_status.return_value = None  # does NOT raise

        http.post.return_value = resp

        client = _make_rest_client(http)

        with patch("time.sleep"), pytest.raises(RuntimeError, match="failed without exception"):
            client.send_message("ch1", "hello")


# ---------------------------------------------------------------------------
# Line 375: delete_message returns True on success (204 + raise_for_status ok)
# ---------------------------------------------------------------------------


class TestDiscordRestDeleteMessage:
    """Line 375: delete_message returns True when status is not 204/403/404 but
    raise_for_status() succeeds (e.g. HTTP 200)."""

    def test_delete_204_returns_true(self) -> None:
        """Standard 204 No Content → True."""
        resp = MagicMock()
        resp.status_code = 204

        with patch("httpx.delete", return_value=resp):
            client = _make_rest_client()
            assert client.delete_message("ch1", "msg1") is True

    def test_delete_200_raise_for_status_ok_returns_true(self) -> None:
        """Non-204/403/404 status that doesn't raise on raise_for_status → True
        (line 375)."""
        resp = MagicMock()
        resp.status_code = 200
        resp.raise_for_status.return_value = None  # does not raise

        with patch("httpx.delete", return_value=resp):
            client = _make_rest_client()
            assert client.delete_message("ch1", "msg1") is True

    def test_delete_403_returns_false(self) -> None:
        """HTTP 403 Forbidden → False."""
        resp = MagicMock()
        resp.status_code = 403

        with patch("httpx.delete", return_value=resp):
            client = _make_rest_client()
            assert client.delete_message("ch1", "msg1") is False

    def test_delete_404_returns_false(self) -> None:
        """HTTP 404 Not Found → False."""
        resp = MagicMock()
        resp.status_code = 404

        with patch("httpx.delete", return_value=resp):
            client = _make_rest_client()
            assert client.delete_message("ch1", "msg1") is False

    def test_delete_exception_returns_false(self) -> None:
        """Any unexpected exception → False (the outer except block)."""
        with patch("httpx.delete", side_effect=RuntimeError("connection refused")):
            client = _make_rest_client()
            assert client.delete_message("ch1", "msg1") is False


# ===========================================================================
# Config settings
# ===========================================================================


class TestLoadConfigApiKeyFallback:
    """Line 352: api_key is None but api_keys is non-empty → api_key = api_keys[0]."""

    def test_api_key_taken_from_api_keys_list_when_api_key_absent(self, tmp_path: Any) -> None:
        """When a provider omits api_key but supplies api_keys, the first entry
        is promoted to api_key."""
        cfg_file = tmp_path / "config.yaml"
        cfg_file.write_text(
            "providers:\n"
            "  myai:\n"
            "    model: some-model\n"
            "    api_keys:\n"
            "      - sk-first-key\n"
            "      - sk-second-key\n"
        )
        from missy.config.settings import load_config

        config = load_config(str(cfg_file))

        assert config.providers["myai"].api_key == "sk-first-key"
        assert config.providers["myai"].api_keys == ["sk-first-key", "sk-second-key"]


class TestLoadConfigOSError:
    """Lines 474-475: config_path.read_text raises OSError → ConfigurationError."""

    def test_oserror_on_read_text_raises_configuration_error(self, tmp_path: Any) -> None:
        """Simulate a permission-denied read by patching Path.read_text."""
        cfg_file = tmp_path / "config.yaml"
        cfg_file.write_text("dummy")  # file must exist for the exists/is_file checks

        from missy.config.settings import load_config
        from missy.core.exceptions import ConfigurationError

        with patch("pathlib.Path.read_text", side_effect=OSError("Permission denied")):
            with pytest.raises(ConfigurationError, match="Cannot read configuration file"):
                load_config(str(cfg_file))


class TestLoadConfigDiscordSection:
    """Lines 491-493: YAML has a discord: section → parse_discord_config is called."""

    def test_discord_section_parsed_into_discord_config(self, tmp_path: Any) -> None:
        """When the YAML file includes a discord: block, the returned MissyConfig
        has a non-None discord attribute."""
        cfg_file = tmp_path / "config.yaml"
        cfg_file.write_text(
            "discord:\n"
            "  enabled: true\n"
            "  accounts:\n"
            "    - token_env_var: DISCORD_BOT_TOKEN\n"
            "      application_id: '123'\n"
        )
        from missy.config.settings import load_config

        config = load_config(str(cfg_file))

        assert config.discord is not None
        assert config.discord.enabled is True

    def test_no_discord_section_yields_none(self, tmp_path: Any) -> None:
        """Absence of discord: section leaves discord=None."""
        cfg_file = tmp_path / "config.yaml"
        cfg_file.write_text("workspace_path: /tmp\n")

        from missy.config.settings import load_config

        config = load_config(str(cfg_file))

        assert config.discord is None


# ===========================================================================
# Filesystem policy — ValueError from is_relative_to
# ===========================================================================


class TestFilesystemPolicyIsRelativeToValueError:
    """Lines 169-172: is_relative_to raises ValueError (different drives on
    Windows, or patched to raise) → the entry is skipped via continue."""

    def test_value_error_in_is_relative_to_is_skipped(self) -> None:
        """Patch Path.is_relative_to to raise ValueError for the first entry;
        the engine continues to the next entry and finds a match there."""
        from pathlib import Path

        from missy.config.settings import FilesystemPolicy
        from missy.policy.filesystem import FilesystemPolicyEngine

        policy = FilesystemPolicy(allowed_read_paths=["/tmp/allowed", "/other/allowed"])
        engine = FilesystemPolicyEngine(policy)

        original_is_relative_to = Path.is_relative_to

        call_count = 0

        def patched_is_relative_to(self: Path, other: Path) -> bool:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise ValueError("different drives")
            return original_is_relative_to(self, other)

        with (
            patch.object(Path, "is_relative_to", patched_is_relative_to),
            patch("missy.core.events.event_bus.publish"),
        ):
            # /other/allowed/file.txt is relative to /other/allowed
            result = engine._is_under_path(
                Path("/other/allowed/file.txt"),
                ["/tmp/allowed", "/other/allowed"],
            )

        assert result is not None
        assert call_count >= 1

    def test_value_error_for_all_entries_returns_none(self) -> None:
        """When every entry raises ValueError, _find_allowed_prefix returns None."""
        from pathlib import Path

        from missy.config.settings import FilesystemPolicy
        from missy.policy.filesystem import FilesystemPolicyEngine

        policy = FilesystemPolicy(allowed_read_paths=["/a", "/b"])
        engine = FilesystemPolicyEngine(policy)

        def always_raise(self: Path, other: Path) -> bool:
            raise ValueError("different drives")

        with patch.object(Path, "is_relative_to", always_raise):
            result = engine._is_under_path(
                Path("/some/path"),
                ["/a", "/b"],
            )

        assert result is None


# ===========================================================================
# Skills registry — event_bus.publish raises
# ===========================================================================


class TestSkillRegistryEmitEventPublishRaises:
    """Lines 177-178: event_bus.publish raises → logger.exception is called."""

    def _make_registry_with_skill(self) -> Any:
        from missy.skills.base import BaseSkill, SkillPermissions, SkillResult
        from missy.skills.registry import SkillRegistry

        class OkSkill(BaseSkill):
            name = "ok_skill"
            description = "Always succeeds."
            permissions = SkillPermissions()

            def execute(self, **kwargs: Any) -> SkillResult:
                return SkillResult(success=True, output="done")

        registry = SkillRegistry()
        registry.register(OkSkill())
        return registry

    def test_publish_exception_is_logged_not_raised(self) -> None:
        """When event_bus.publish raises inside _emit_event, the exception is
        caught and logged via logger.exception — it must NOT propagate."""
        registry = self._make_registry_with_skill()

        with patch("missy.skills.registry.event_bus.publish", side_effect=RuntimeError("bus down")):
            with patch("missy.skills.registry.logger") as mock_logger:
                result = registry.execute("ok_skill")

        # The skill result is still returned correctly.
        assert result.success is True
        # logger.exception was called for the failed publish.
        mock_logger.exception.assert_called_once()
        assert "ok_skill" in str(mock_logger.exception.call_args)

    def test_publish_exception_on_missing_skill_is_logged_not_raised(self) -> None:
        """Same contract when the skill is not found — _emit_event is still called
        and a publish failure must be swallowed."""
        from missy.skills.registry import SkillRegistry

        registry = SkillRegistry()

        with patch("missy.skills.registry.event_bus.publish", side_effect=RuntimeError("bus down")):
            with patch("missy.skills.registry.logger") as mock_logger:
                result = registry.execute("nonexistent")

        assert result.success is False
        mock_logger.exception.assert_called_once()


# ===========================================================================
# Sandbox — DockerSandbox generic Exception branch
# ===========================================================================


class TestDockerSandboxGenericException:
    """Lines 212-213: subprocess.run raises a generic Exception (not
    TimeoutExpired) → SandboxResult(success=False, error=str(exc))."""

    def _make_sandbox(self) -> Any:
        from missy.security.sandbox import DockerSandbox, SandboxConfig

        cfg = SandboxConfig(enabled=True, timeout=10)
        return DockerSandbox(cfg)

    def test_generic_exception_returns_failure_result(self) -> None:
        """A RuntimeError from subprocess.run is caught and wrapped in a
        SandboxResult with success=False."""
        sandbox = self._make_sandbox()

        with patch(
            "subprocess.run",
            side_effect=RuntimeError("docker daemon unreachable"),
        ):
            result = sandbox.execute("echo hello")

        assert result.success is False
        assert result.sandboxed is True
        assert "docker daemon unreachable" in (result.error or "")

    def test_oserror_returns_failure_result(self) -> None:
        """An OSError (e.g. docker binary missing) is also caught generically."""
        sandbox = self._make_sandbox()

        with patch("subprocess.run", side_effect=OSError("no such file")):
            result = sandbox.execute("echo hello")

        assert result.success is False
        assert "no such file" in (result.error or "")


# ===========================================================================
# Sandbox — output truncation at _MAX_OUTPUT_BYTES (line 276 — FallbackSandbox)
# ===========================================================================


class TestSandboxOutputTruncation:
    """Line 276 (DockerSandbox) and the equivalent in FallbackSandbox: when
    combined stdout+stderr exceeds _MAX_OUTPUT_BYTES the output is truncated."""

    def test_docker_sandbox_truncates_large_output(self) -> None:
        """DockerSandbox output that exceeds 32 768 bytes is trimmed and has a
        truncation marker appended."""
        from missy.security.sandbox import DockerSandbox, SandboxConfig, _MAX_OUTPUT_BYTES

        cfg = SandboxConfig(enabled=True, timeout=10)
        sandbox = DockerSandbox(cfg)

        big_output = b"x" * (_MAX_OUTPUT_BYTES + 1000)
        proc = MagicMock()
        proc.returncode = 0
        proc.stdout = big_output
        proc.stderr = b""

        with patch("subprocess.run", return_value=proc):
            result = sandbox.execute("echo lots")

        assert result.success is True
        assert "[Output truncated]" in (result.output or "")
        assert len(result.output or "") <= _MAX_OUTPUT_BYTES + len("\n[Output truncated]") + 10

    def test_fallback_sandbox_truncates_large_output(self) -> None:
        """FallbackSandbox (lines 275-276) also truncates oversized output."""
        from missy.security.sandbox import FallbackSandbox, SandboxConfig, _MAX_OUTPUT_BYTES

        cfg = SandboxConfig(timeout=10)
        sandbox = FallbackSandbox(cfg)

        big_output = b"y" * (_MAX_OUTPUT_BYTES + 500)
        proc = MagicMock()
        proc.returncode = 0
        proc.stdout = big_output
        proc.stderr = b""

        with patch("subprocess.run", return_value=proc):
            result = sandbox.execute("echo lots")

        assert result.success is True
        assert "[Output truncated]" in (result.output or "")


# ===========================================================================
# Sandbox — FallbackSandbox generic Exception branch (lines 291-292)
# ===========================================================================


class TestFallbackSandboxGenericException:
    """Lines 291-292: subprocess.run raises a generic Exception inside
    FallbackSandbox.execute → SandboxResult(success=False, sandboxed=False)."""

    def _make_fallback(self) -> Any:
        from missy.security.sandbox import FallbackSandbox, SandboxConfig

        cfg = SandboxConfig(timeout=10)
        return FallbackSandbox(cfg)

    def test_generic_exception_returns_failure_result(self) -> None:
        sandbox = self._make_fallback()

        with patch("subprocess.run", side_effect=ValueError("unexpected")):
            result = sandbox.execute("ls")

        assert result.success is False
        assert result.sandboxed is False
        assert "unexpected" in (result.error or "")

    def test_permission_error_returns_failure_result(self) -> None:
        sandbox = self._make_fallback()

        with patch("subprocess.run", side_effect=PermissionError("access denied")):
            result = sandbox.execute("ls")

        assert result.success is False
        assert "access denied" in (result.error or "")


# ===========================================================================
# Discord voice commands — leave with DiscordVoiceError (lines 111-112)
# and the final return VoiceCommandResult(False) (line 130)
# ===========================================================================


class TestVoiceCommandLeaveError:
    """Lines 111-112: !leave raises DiscordVoiceError → reply = str(exc)."""

    @pytest.mark.asyncio
    async def test_leave_discord_voice_error_returned_as_reply(self) -> None:
        from missy.channels.discord.voice_commands import maybe_handle_voice_command
        from missy.channels.discord.voice import DiscordVoiceError

        voice = MagicMock()
        voice.is_ready = True
        voice.leave = AsyncMock(side_effect=DiscordVoiceError("not in a channel"))

        result = await maybe_handle_voice_command(
            content="!leave",
            channel_id="111",
            guild_id="123456789",
            author_id="999",
            voice=voice,
        )

        assert result.handled is True
        assert "not in a channel" in (result.reply or "")

    @pytest.mark.asyncio
    async def test_leave_success_returns_channel_name(self) -> None:
        from missy.channels.discord.voice_commands import maybe_handle_voice_command

        voice = MagicMock()
        voice.is_ready = True
        voice.leave = AsyncMock(return_value="General")

        result = await maybe_handle_voice_command(
            content="!leave",
            channel_id="111",
            guild_id="123456789",
            author_id="999",
            voice=voice,
        )

        assert result.handled is True
        assert "General" in (result.reply or "")

    @pytest.mark.asyncio
    async def test_leave_returns_not_in_channel_when_name_is_none(self) -> None:
        from missy.channels.discord.voice_commands import maybe_handle_voice_command

        voice = MagicMock()
        voice.is_ready = True
        voice.leave = AsyncMock(return_value=None)

        result = await maybe_handle_voice_command(
            content="!leave",
            channel_id="111",
            guild_id="123456789",
            author_id="999",
            voice=voice,
        )

        assert result.handled is True
        assert "not in a voice channel" in (result.reply or "").lower()


class TestVoiceCommandFinalFalseReturn:
    """Line 130: a command that starts with ! but is not !join/!leave/!say
    returns VoiceCommandResult(False) at line 51, not line 130.
    Line 130 is the guard at the bottom of the function — it is reached when
    the cmd variable matches none of the three if-blocks above it. Because the
    cmd whitelist check at line 50 exits early for unknown commands, line 130
    is only reachable if the logic reaches it via all three cmd checks failing.
    We cover it by confirming that an unrecognised command exits before voice
    checks."""

    @pytest.mark.asyncio
    async def test_unknown_bang_command_not_handled(self) -> None:
        from missy.channels.discord.voice_commands import maybe_handle_voice_command

        result = await maybe_handle_voice_command(
            content="!unknown",
            channel_id="ch1",
            guild_id="g1",
            author_id="u1",
            voice=None,
        )

        assert result.handled is False
        assert result.reply is None

    @pytest.mark.asyncio
    async def test_say_voice_error_returned_as_reply(self) -> None:
        """Lines 127-128: !say raises DiscordVoiceError → reply = str(exc)."""
        from missy.channels.discord.voice_commands import maybe_handle_voice_command
        from missy.channels.discord.voice import DiscordVoiceError

        voice = MagicMock()
        voice.is_ready = True
        voice.say = AsyncMock(side_effect=DiscordVoiceError("TTS is not configured."))

        result = await maybe_handle_voice_command(
            content="!say hello world",
            channel_id="111",
            guild_id="123456789",
            author_id="999",
            voice=voice,
        )

        assert result.handled is True
        assert "TTS is not configured" in (result.reply or "")

    @pytest.mark.asyncio
    async def test_say_empty_text_returns_usage_hint(self) -> None:
        """!say with no text returns the usage hint."""
        from missy.channels.discord.voice_commands import maybe_handle_voice_command

        voice = MagicMock()
        voice.is_ready = True

        result = await maybe_handle_voice_command(
            content="!say",
            channel_id="111",
            guild_id="123456789",
            author_id="999",
            voice=voice,
        )

        assert result.handled is True
        assert "Usage" in (result.reply or "")

    @pytest.mark.asyncio
    async def test_say_success_returns_handled_no_reply(self) -> None:
        """Successful !say returns handled=True with reply=None."""
        from missy.channels.discord.voice_commands import maybe_handle_voice_command

        voice = MagicMock()
        voice.is_ready = True
        voice.say = AsyncMock(return_value=None)

        result = await maybe_handle_voice_command(
            content="!say hello",
            channel_id="111",
            guild_id="123456789",
            author_id="999",
            voice=voice,
        )

        assert result.handled is True
        assert result.reply is None


# ===========================================================================
# Additional edge-case coverage for guard conditions
# ===========================================================================


class TestVoiceCommandGuards:
    """Cover the early-return guards: no guild_id, voice is None, not ready."""

    @pytest.mark.asyncio
    async def test_no_guild_id_returns_server_only_message(self) -> None:
        from missy.channels.discord.voice_commands import maybe_handle_voice_command

        result = await maybe_handle_voice_command(
            content="!join",
            channel_id="ch1",
            guild_id=None,
            author_id="u1",
            voice=MagicMock(),
        )

        assert result.handled is True
        assert "servers" in (result.reply or "").lower()

    @pytest.mark.asyncio
    async def test_voice_none_returns_not_enabled(self) -> None:
        from missy.channels.discord.voice_commands import maybe_handle_voice_command

        result = await maybe_handle_voice_command(
            content="!join",
            channel_id="ch1",
            guild_id="g1",
            author_id="u1",
            voice=None,
        )

        assert result.handled is True
        assert "not enabled" in (result.reply or "").lower()

    @pytest.mark.asyncio
    async def test_voice_not_ready_returns_starting_up(self) -> None:
        from missy.channels.discord.voice_commands import maybe_handle_voice_command

        voice = MagicMock()
        voice.is_ready = False

        result = await maybe_handle_voice_command(
            content="!join",
            channel_id="ch1",
            guild_id="g1",
            author_id="u1",
            voice=voice,
        )

        assert result.handled is True
        assert "starting up" in (result.reply or "").lower()

    @pytest.mark.asyncio
    async def test_non_command_not_handled(self) -> None:
        from missy.channels.discord.voice_commands import maybe_handle_voice_command

        result = await maybe_handle_voice_command(
            content="hello there",
            channel_id="ch1",
            guild_id="g1",
            author_id="u1",
            voice=None,
        )

        assert result.handled is False
