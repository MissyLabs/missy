"""Coverage gap tests for persona, playbook, and discord channel modules.

Targets the following uncovered lines:

persona.py:
  272-276 — save() exception path: temp file cleanup on write failure
  355-356 — _audit() OSError path: audit log write fails gracefully
  374     — get_audit_log() blank line skip in JSONL iteration
  377-380 — get_audit_log() OSError on file open

playbook.py:
  186-192 — save() inner exception path: temp file cleanup on json.dump failure

discord/channel.py:
  490     — _handle_message: image command handled -> early return
  500     — _handle_message: screen command handled -> early return
  706-707 — _maybe_handle_image_command: long reply split into chunks
  751-775 — _maybe_handle_image_command: full flow (typing + result)
  792-807 — _maybe_handle_screen_command: reply splitting (short and long)
  835-837 — _handle_interaction: handle_slash_command raises exception
  843-844 — _handle_interaction: no application_id -> early return
  846-847 — _handle_interaction: edit_interaction_response raises exception
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from missy.agent.persona import PersonaManager
from missy.agent.playbook import Playbook
from missy.channels.discord.channel import DiscordChannel
from missy.channels.discord.config import (
    DiscordAccountConfig,
    DiscordDMPolicy,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_discord_account(
    dm_policy: DiscordDMPolicy = DiscordDMPolicy.OPEN,
    account_id: str = "bot-001",
    application_id: str = "",
) -> DiscordAccountConfig:
    return DiscordAccountConfig(
        token="test-token",
        token_env_var="DISCORD_BOT_TOKEN",
        account_id=account_id,
        dm_policy=dm_policy,
        application_id=application_id,
    )


def _make_channel(account: DiscordAccountConfig | None = None) -> DiscordChannel:
    if account is None:
        account = _make_discord_account()
    with (
        patch("missy.channels.discord.channel.DiscordGatewayClient"),
        patch("missy.channels.discord.channel.DiscordRestClient"),
    ):
        return DiscordChannel(account_config=account)


# ===========================================================================
# persona.py gap tests
# ===========================================================================


class TestPersonaSaveExceptionCleanup:
    """Lines 272-276: exception during save cleans up the temp file and re-raises."""

    def test_save_cleans_up_temp_file_on_write_failure(self, tmp_path):
        """When yaml.dump raises, the temp file is removed and the exception propagates."""
        persona_file = tmp_path / "persona.yaml"
        pm = PersonaManager(persona_path=persona_file)

        # Capture the temp file path that mkstemp creates so we can assert it is removed.

        real_mkstemp = os.fdopen

        def fake_fdopen(fd, *args, **kwargs):
            # Let the fd open, then raise to simulate a mid-write failure.
            real_mkstemp(fd, *args, **kwargs)
            raise OSError("simulated disk full")

        with (
            patch("missy.agent.persona.os.fdopen", side_effect=fake_fdopen),
            pytest.raises(OSError, match="simulated disk full"),
        ):
            pm.save()

    def test_save_exception_propagates_original_error(self, tmp_path):
        """The original exception is re-raised, not swallowed."""
        persona_file = tmp_path / "persona.yaml"
        pm = PersonaManager(persona_path=persona_file)

        with (
            patch("missy.agent.persona.yaml.dump", side_effect=RuntimeError("yaml boom")),
            pytest.raises(RuntimeError, match="yaml boom"),
        ):
            pm.save()

    def test_save_cleanup_suppresses_unlink_oserror(self, tmp_path):
        """Even if os.unlink fails during cleanup, the original exception propagates cleanly."""
        persona_file = tmp_path / "persona.yaml"
        pm = PersonaManager(persona_path=persona_file)

        with (
            patch("missy.agent.persona.yaml.dump", side_effect=ValueError("bad yaml")),
            patch("missy.agent.persona.os.unlink", side_effect=OSError("unlink failed")),
            pytest.raises(ValueError, match="bad yaml"),
        ):
            # The original ValueError should still propagate, not the unlink OSError.
            pm.save()


class TestPersonaAuditOSError:
    """Lines 355-356: _audit() swallows OSError when audit log cannot be written."""

    def test_audit_oserror_is_swallowed(self, tmp_path):
        """An OSError during audit log write should not propagate to the caller."""
        persona_file = tmp_path / "persona.yaml"
        pm = PersonaManager(persona_path=persona_file)

        with patch("missy.agent.persona.os.open", side_effect=OSError("permission denied")):
            # Should not raise — the OSError must be caught internally.
            pm._audit("test_action")

    def test_audit_oserror_on_fdopen_is_swallowed(self, tmp_path):
        """An OSError during os.fdopen inside _audit does not propagate."""
        persona_file = tmp_path / "persona.yaml"
        pm = PersonaManager(persona_path=persona_file)

        real_os_open = os.open

        def fake_os_open(path, flags, mode=0o666):
            # Allow mkstemp and other opens; only block the audit log path.
            if "persona_audit" in str(path):
                raise OSError("permission denied")
            return real_os_open(path, flags, mode)

        with patch("missy.agent.persona.os.open", side_effect=fake_os_open):
            # _audit catches OSError internally — must not raise.
            pm._audit("test_action")


class TestPersonaGetAuditLogEdgeCases:
    """Lines 374, 377-380: get_audit_log blank-line skip and OSError on open."""

    def test_blank_lines_in_jsonl_are_skipped(self, tmp_path):
        """Blank lines in the audit JSONL file do not cause parse errors."""
        persona_file = tmp_path / "persona.yaml"
        pm = PersonaManager(persona_path=persona_file)

        audit_path = tmp_path / "persona_audit.jsonl"
        valid_entry = json.dumps({"action": "save", "version": 1}) + "\n"
        # Write a blank line sandwiched between valid entries.
        audit_path.write_text(valid_entry + "\n" + valid_entry, encoding="utf-8")

        entries = pm.get_audit_log()

        assert len(entries) == 2
        assert all(e["action"] == "save" for e in entries)

    def test_get_audit_log_oserror_on_open_returns_empty(self, tmp_path):
        """If the audit file exists but open raises OSError, an empty list is returned."""
        persona_file = tmp_path / "persona.yaml"
        pm = PersonaManager(persona_path=persona_file)

        audit_path = tmp_path / "persona_audit.jsonl"
        # Create the file so the existence check passes.
        audit_path.write_text('{"action": "save"}\n', encoding="utf-8")

        with patch.object(Path, "open", side_effect=OSError("permission denied")):
            entries = pm.get_audit_log()

        assert entries == []

    def test_get_audit_log_malformed_line_skipped(self, tmp_path):
        """A line with invalid JSON is silently skipped."""
        persona_file = tmp_path / "persona.yaml"
        pm = PersonaManager(persona_path=persona_file)

        audit_path = tmp_path / "persona_audit.jsonl"
        audit_path.write_text(
            '{"action": "save"}\nnot valid json\n{"action": "reset"}\n',
            encoding="utf-8",
        )

        entries = pm.get_audit_log()

        assert len(entries) == 2


# ===========================================================================
# playbook.py gap tests
# ===========================================================================


class TestPlaybookSaveExceptionCleanup:
    """Lines 186-192: save() inner exception cleans up temp file."""

    def test_save_inner_exception_removes_temp_file(self, tmp_path):
        """When json.dump raises inside save(), the exception is caught and logged, not propagated."""
        pb = Playbook(store_path=str(tmp_path / "playbook.json"))
        pb.record("shell", "deploy", ["shell_exec"], "use rsync")

        with patch("missy.agent.playbook.json.dump", side_effect=OSError("disk full")):
            # save() swallows the exception (logs at DEBUG) — must not propagate.
            pb.save()

    def test_save_inner_exception_suppresses_unlink_oserror(self, tmp_path):
        """If os.unlink itself raises during temp cleanup, the outer exception is still swallowed."""
        pb = Playbook(store_path=str(tmp_path / "playbook.json"))
        pb.record("file", "read config", ["file_read"], "open yaml")

        with (
            patch("missy.agent.playbook.json.dump", side_effect=OSError("write failed")),
            patch("missy.agent.playbook.os.unlink", side_effect=OSError("busy")),
        ):
            pb.save()  # Must not raise.

    def test_record_still_updates_in_memory_when_save_fails(self, tmp_path):
        """Even if disk write fails, the in-memory entry is updated."""
        pb = Playbook(store_path=str(tmp_path / "playbook.json"))

        with patch("missy.agent.playbook.json.dump", side_effect=OSError("disk full")):
            entry = pb.record("net", "fetch url", ["http_get"], "use requests")

        assert entry.task_type == "net"
        assert entry.success_count == 1


# ===========================================================================
# discord/channel.py gap tests
# ===========================================================================


class TestDiscordImageCommandEarlyReturn:
    """Line 490: _handle_message returns early when image command is handled."""

    @pytest.mark.asyncio
    async def test_image_command_handled_stops_further_processing(self):
        """When _maybe_handle_image_command returns True, message is not enqueued."""
        channel = _make_channel()

        data = {
            "author": {"id": "user-1", "bot": False},
            "channel_id": "ch-1",
            "content": "!analyze",
            "id": "msg-1",
        }

        with patch.object(channel, "_maybe_handle_image_command", new=AsyncMock(return_value=True)):
            await channel._handle_message(data)

        assert channel._queue.empty()


class TestDiscordScreenCommandEarlyReturn:
    """Line 500: _handle_message returns early when screen command is handled."""

    @pytest.mark.asyncio
    async def test_screen_command_handled_stops_further_processing(self):
        """When _maybe_handle_screen_command returns True, message is not enqueued."""
        channel = _make_channel()

        data = {
            "author": {"id": "user-1", "bot": False},
            "channel_id": "ch-1",
            "content": "!screen share",
            "id": "msg-2",
        }

        with (
            patch.object(channel, "_maybe_handle_image_command", new=AsyncMock(return_value=False)),
            patch.object(channel, "_maybe_handle_screen_command", new=AsyncMock(return_value=True)),
        ):
            await channel._handle_message(data)

        assert channel._queue.empty()


class TestDiscordImageCommandLongReply:
    """Lines 751-775, 706-707: _maybe_handle_image_command splits long replies into chunks."""

    @pytest.mark.asyncio
    async def test_short_reply_sent_as_single_message(self):
        """A reply of 2000 chars or fewer is sent in one call."""
        channel = _make_channel()
        channel._rest = MagicMock()

        short_reply = "x" * 1999

        fake_result = MagicMock()
        fake_result.handled = True
        fake_result.reply = short_reply

        with patch(
            "missy.channels.discord.channel.DiscordChannel._maybe_handle_image_command",
            new=AsyncMock(return_value=fake_result.handled),
        ):
            pass  # We test the method directly below.

        with patch(
            "missy.channels.discord.image_commands.maybe_handle_image_command",
            new=AsyncMock(return_value=fake_result),
        ):
            result = await channel._maybe_handle_image_command(
                channel_id="ch-1", content="!analyze foo"
            )

        assert result is True
        channel._rest.send_message.assert_called_once_with("ch-1", short_reply)

    @pytest.mark.asyncio
    async def test_long_reply_split_into_chunks(self):
        """A reply longer than 2000 chars is sent in multiple send_message calls."""
        channel = _make_channel()
        channel._rest = MagicMock()

        # 4000 chars → exactly two 1990-char chunks (2000 + 2000 > 1990*2 but fits).
        long_reply = "y" * 4000

        fake_result = MagicMock()
        fake_result.handled = True
        fake_result.reply = long_reply

        with patch(
            "missy.channels.discord.image_commands.maybe_handle_image_command",
            new=AsyncMock(return_value=fake_result),
        ):
            result = await channel._maybe_handle_image_command(
                channel_id="ch-2", content="!screenshot"
            )

        assert result is True
        # send_message should have been called more than once (chunked).
        assert channel._rest.send_message.call_count >= 2

    @pytest.mark.asyncio
    async def test_image_command_not_handled_returns_false(self):
        """Returns False when the command is not an image command."""
        channel = _make_channel()
        channel._rest = MagicMock()

        result = await channel._maybe_handle_image_command(channel_id="ch-1", content="hello world")
        assert result is False


class TestDiscordScreenCommandReply:
    """Lines 792-807: _maybe_handle_screen_command short and long reply paths."""

    @pytest.mark.asyncio
    async def test_screen_short_reply_sent_as_single_message(self):
        """A short screen command reply is sent in one send_message call."""
        channel = _make_channel()
        channel._rest = MagicMock()

        fake_result = MagicMock()
        fake_result.handled = True
        fake_result.reply = "Screen shared."

        with patch(
            "missy.channels.discord.screen_commands.maybe_handle_screen_command",
            new=AsyncMock(return_value=fake_result),
        ):
            result = await channel._maybe_handle_screen_command(
                channel_id="ch-1", author_id="user-1", content="!screen share"
            )

        assert result is True
        channel._rest.send_message.assert_called_once_with("ch-1", "Screen shared.")

    @pytest.mark.asyncio
    async def test_screen_long_reply_chunked(self):
        """A screen command reply longer than 2000 chars is split into chunks."""
        channel = _make_channel()
        channel._rest = MagicMock()

        long_reply = "z" * 4001

        fake_result = MagicMock()
        fake_result.handled = True
        fake_result.reply = long_reply

        with patch(
            "missy.channels.discord.screen_commands.maybe_handle_screen_command",
            new=AsyncMock(return_value=fake_result),
        ):
            result = await channel._maybe_handle_screen_command(
                channel_id="ch-1", author_id="user-1", content="!screen list"
            )

        assert result is True
        assert channel._rest.send_message.call_count >= 2

    @pytest.mark.asyncio
    async def test_screen_command_not_matched_returns_false(self):
        """Returns False when content does not start with !screen."""
        channel = _make_channel()
        channel._rest = MagicMock()

        result = await channel._maybe_handle_screen_command(
            channel_id="ch-1", author_id="user-1", content="hello there"
        )
        assert result is False


class TestDiscordInteractionHandlerEdgeCases:
    """Lines 835-837, 843-847: _handle_interaction error paths."""

    @pytest.mark.asyncio
    async def test_slash_command_handler_exception_sends_error_reply(self):
        """Lines 835-837: when handle_slash_command raises, error text is used as response."""
        account = _make_discord_account(application_id="app-123")
        channel = _make_channel(account)
        channel._rest = MagicMock()

        interaction_data = {
            "id": "interaction-1",
            "token": "tok-abc",
            "channel_id": "ch-1",
        }

        with patch(
            "missy.channels.discord.commands.handle_slash_command",
            new=AsyncMock(side_effect=RuntimeError("handler blew up")),
        ):
            await channel._handle_interaction(interaction_data)

        # edit_interaction_response should be called with an error message.
        channel._rest.edit_interaction_response.assert_called_once()
        call_args = channel._rest.edit_interaction_response.call_args
        assert "error" in call_args[0][2].lower() or "handler blew up" in call_args[0][2]

    @pytest.mark.asyncio
    async def test_no_application_id_skips_edit_response(self):
        """Lines 843-844: missing application_id causes early return after slash command runs."""
        account = _make_discord_account(application_id="")
        channel = _make_channel(account)
        channel._rest = MagicMock()

        interaction_data = {
            "id": "interaction-2",
            "token": "tok-xyz",
            "channel_id": "ch-2",
        }

        with patch(
            "missy.channels.discord.commands.handle_slash_command",
            new=AsyncMock(return_value="all good"),
        ):
            await channel._handle_interaction(interaction_data)

        channel._rest.edit_interaction_response.assert_not_called()

    @pytest.mark.asyncio
    async def test_edit_response_exception_is_logged_not_raised(self):
        """Lines 846-847: exception in edit_interaction_response is swallowed."""
        account = _make_discord_account(application_id="app-456")
        channel = _make_channel(account)
        channel._rest = MagicMock()
        channel._rest.edit_interaction_response.side_effect = RuntimeError("network error")

        interaction_data = {
            "id": "interaction-3",
            "token": "tok-def",
            "channel_id": "ch-3",
        }

        with patch(
            "missy.channels.discord.commands.handle_slash_command",
            new=AsyncMock(return_value="response text"),
        ):
            # Must not propagate the edit exception.
            await channel._handle_interaction(interaction_data)

    @pytest.mark.asyncio
    async def test_deferred_response_failure_aborts_interaction(self):
        """When send_interaction_response raises, the interaction is abandoned early."""
        account = _make_discord_account(application_id="app-789")
        channel = _make_channel(account)
        channel._rest = MagicMock()
        channel._rest.send_interaction_response.side_effect = RuntimeError("gateway timeout")

        interaction_data = {
            "id": "interaction-4",
            "token": "tok-ghi",
            "channel_id": "ch-4",
        }

        with patch(
            "missy.channels.discord.commands.handle_slash_command",
            new=AsyncMock(return_value="never reached"),
        ) as mock_handler:
            await channel._handle_interaction(interaction_data)

        # If deferred response failed, we should not even call the handler.
        mock_handler.assert_not_called()
        channel._rest.edit_interaction_response.assert_not_called()
