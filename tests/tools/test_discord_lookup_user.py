"""Tests for the DiscordLookupUserTool (Discord user tagging fallback)."""

from __future__ import annotations

import os
from unittest.mock import MagicMock, patch

from missy.tools.builtin.discord_lookup_user import DiscordLookupUserTool


class TestDiscordLookupUserTool:
    def test_no_token_returns_error(self):
        with patch.dict(os.environ, {}, clear=True):
            tool = DiscordLookupUserTool()
            result = tool.execute(guild_id="123", query="bob")
        assert result.success is False
        assert "DISCORD_BOT_TOKEN" in result.error

    def test_empty_query_returns_error(self):
        tool = DiscordLookupUserTool()
        result = tool.execute(guild_id="123", query="   ")
        assert result.success is False
        assert "query" in result.error.lower()

    def test_successful_lookup_returns_matches(self):
        mock_rest = MagicMock()
        mock_rest.search_guild_members.return_value = [
            {
                "user": {"id": "111", "username": "bob", "global_name": "Bobby"},
                "nick": "Bobby the Builder",
            },
            {"user": {"id": "222", "username": "bob2"}},
        ]
        with (
            patch.dict(os.environ, {"DISCORD_BOT_TOKEN": "fake-token"}),
            patch(
                "missy.channels.discord.rest.DiscordRestClient",
                return_value=mock_rest,
            ),
        ):
            tool = DiscordLookupUserTool()
            result = tool.execute(guild_id="999", query="bob")

        assert result.success is True
        assert result.output["query"] == "bob"
        matches = result.output["matches"]
        assert matches[0] == {
            "id": "111",
            "username": "bob",
            "display_name": "Bobby the Builder",
        }
        assert matches[1] == {"id": "222", "username": "bob2", "display_name": "bob2"}
        mock_rest.search_guild_members.assert_called_once_with(
            guild_id="999", query="bob", limit=10
        )

    def test_member_missing_user_id_is_skipped(self):
        mock_rest = MagicMock()
        mock_rest.search_guild_members.return_value = [{"user": {}}]
        with (
            patch.dict(os.environ, {"DISCORD_BOT_TOKEN": "fake-token"}),
            patch(
                "missy.channels.discord.rest.DiscordRestClient",
                return_value=mock_rest,
            ),
        ):
            tool = DiscordLookupUserTool()
            result = tool.execute(guild_id="999", query="bob")

        assert result.success is True
        assert result.output["matches"] == []

    def test_no_matches_still_succeeds(self):
        mock_rest = MagicMock()
        mock_rest.search_guild_members.return_value = []
        with (
            patch.dict(os.environ, {"DISCORD_BOT_TOKEN": "fake-token"}),
            patch(
                "missy.channels.discord.rest.DiscordRestClient",
                return_value=mock_rest,
            ),
        ):
            tool = DiscordLookupUserTool()
            result = tool.execute(guild_id="999", query="nobody-named-this")

        assert result.success is True
        assert result.output["matches"] == []

    def test_invalid_guild_id_reports_error_not_exception(self):
        with (
            patch.dict(os.environ, {"DISCORD_BOT_TOKEN": "fake-token"}),
        ):
            tool = DiscordLookupUserTool()
            result = tool.execute(guild_id="not-a-snowflake", query="bob")

        assert result.success is False
        assert result.error

    def test_generic_exception_is_captured(self):
        mock_rest = MagicMock()
        mock_rest.search_guild_members.side_effect = RuntimeError("upstream exploded")
        with (
            patch.dict(os.environ, {"DISCORD_BOT_TOKEN": "fake-token"}),
            patch(
                "missy.channels.discord.rest.DiscordRestClient",
                return_value=mock_rest,
            ),
        ):
            tool = DiscordLookupUserTool()
            result = tool.execute(guild_id="999", query="bob")

        assert result.success is False
        assert "upstream exploded" in result.error

    def test_tool_metadata(self):
        tool = DiscordLookupUserTool()
        assert tool.name == "discord_lookup_user"
        assert tool.permissions.network is True
        assert tool.permissions.filesystem_read is False
        assert tool.permissions.shell is False

    def test_schema_requires_guild_id_and_query(self):
        schema = DiscordLookupUserTool().get_schema()
        assert set(schema["parameters"]["required"]) == {"guild_id", "query"}
        assert "guild_id" in schema["parameters"]["properties"]
        assert "query" in schema["parameters"]["properties"]
