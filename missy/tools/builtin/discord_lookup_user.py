"""Built-in tool: resolve a Discord display name to a real user ID.

Discord only turns text into an actual @-ping when it's the literal
``<@USER_ID>`` syntax with a real numeric snowflake -- plain ``@name`` text
renders as inert text and pings no one. The agent's Discord conversation
context already carries real IDs for the current message's author and
anyone recently active in the channel (see ``cli/main.py``'s Discord
message loop), which covers the overwhelming majority of "tag @bob"
requests for free, with no extra API call. This tool is the fallback for
the remaining case: tagging someone who hasn't spoken in the channel
recently, via Discord's guild-member-search endpoint.

Read-only (no approval gate, unlike ``discord_upload_file``/OBS
streaming): it looks a name up and returns candidate IDs, it does not post
or ping anything by itself.
"""

from __future__ import annotations

import os
from typing import Any

from missy.tools.base import BaseTool, ToolPermissions, ToolResult

#: Cap on how many candidate matches are returned to the model -- a lookup
#: tool result re-enters the conversation as tokens the model reads, not a
#: dataset it programmatically filters.
_MAX_RESULTS = 10


class DiscordLookupUserTool(BaseTool):
    """Search a Discord guild's members by username/nickname prefix."""

    name = "discord_lookup_user"
    description = (
        "Look up a Discord guild member's real numeric user ID by username or "
        "nickname (prefix match). Use this to find the ID needed to tag/mention/"
        "ping someone who hasn't spoken recently in the current channel -- for "
        "anyone who HAS spoken recently, their real ID is already given in your "
        "conversation context, so check there first. Returns candidate matches; "
        "pick the right one and use <@THEIR_ID> in your reply to actually ping them."
    )
    permissions = ToolPermissions(network=True)
    parameters = {
        "guild_id": {
            "type": "string",
            "description": "The Discord guild (server) snowflake ID to search within.",
            "required": True,
        },
        "query": {
            "type": "string",
            "description": "Username or nickname prefix to search for, e.g. 'bob'.",
            "required": True,
        },
    }

    def execute(self, *, guild_id: str, query: str, **_kwargs: Any) -> ToolResult:
        query = str(query or "").strip()
        if not query:
            return ToolResult(success=False, output=None, error="'query' is required.")

        bot_token = os.environ.get("DISCORD_BOT_TOKEN", "")
        if not bot_token:
            return ToolResult(success=False, output=None, error="DISCORD_BOT_TOKEN not set")

        try:
            from missy.channels.discord.rest import DiscordRestClient

            rest = DiscordRestClient(bot_token=bot_token)
            members = rest.search_guild_members(
                guild_id=str(guild_id), query=query, limit=_MAX_RESULTS
            )
        except ValueError as exc:
            return ToolResult(success=False, output=None, error=str(exc))
        except Exception as exc:  # noqa: BLE001
            return ToolResult(
                success=False, output=None, error=f"Guild member search failed: {exc}"
            )

        matches = []
        for member in members:
            user = member.get("user") or {}
            user_id = str(user.get("id", "") or "")
            if not user_id:
                continue
            matches.append(
                {
                    "id": user_id,
                    "username": str(user.get("username", "") or ""),
                    "display_name": str(
                        member.get("nick") or user.get("global_name") or user.get("username") or ""
                    ),
                }
            )

        return ToolResult(
            success=True,
            output={"query": query, "matches": matches},
        )

    def get_schema(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "parameters": {
                "type": "object",
                "properties": {
                    "guild_id": {
                        "type": "string",
                        "description": "The Discord guild (server) snowflake ID to search within.",
                    },
                    "query": {
                        "type": "string",
                        "description": "Username or nickname prefix to search for, e.g. 'bob'.",
                    },
                },
                "required": ["guild_id", "query"],
            },
        }
