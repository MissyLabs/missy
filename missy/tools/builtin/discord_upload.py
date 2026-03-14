"""Built-in tool: upload a file to the current Discord channel."""

from __future__ import annotations

import os

from missy.tools.base import BaseTool, ToolPermissions, ToolResult


class DiscordUploadTool(BaseTool):
    """Upload a local file to a Discord channel using the configured bot token."""

    name = "discord_upload_file"
    description = (
        "Upload a file from the local filesystem to a Discord channel. "
        "Use this to post images, screenshots, documents, or any file into Discord. "
        "Requires file_path (absolute path) and channel_id (the Discord channel snowflake ID)."
    )
    permissions = ToolPermissions(network=True, filesystem_read=True)
    parameters = {
        "file_path": {
            "type": "string",
            "description": "Absolute path to the file to upload, e.g. /tmp/screenshot.png",
            "required": True,
        },
        "channel_id": {
            "type": "string",
            "description": "Discord channel snowflake ID to post the file into.",
            "required": True,
        },
        "caption": {
            "type": "string",
            "description": "Optional text message to include with the file.",
        },
    }

    def execute(
        self,
        *,
        file_path: str,
        channel_id: str,
        caption: str = "",
        **_kwargs,
    ) -> ToolResult:
        bot_token = os.environ.get("DISCORD_BOT_TOKEN", "")
        if not bot_token:
            return ToolResult(success=False, output=None, error="DISCORD_BOT_TOKEN not set")

        try:
            from missy.channels.discord.rest import DiscordRestClient

            rest = DiscordRestClient(bot_token=bot_token)
            result = rest.upload_file(channel_id=channel_id, file_path=file_path, caption=caption)
            return ToolResult(
                success=True,
                output=f"File uploaded to Discord (message id: {result.get('id', '?')})",
                error=None,
            )
        except FileNotFoundError as exc:
            return ToolResult(success=False, output=None, error=str(exc))
        except Exception as exc:
            return ToolResult(success=False, output=None, error=f"Upload failed: {exc}")
