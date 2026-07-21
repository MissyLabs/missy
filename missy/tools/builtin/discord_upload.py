"""Built-in tool: upload a file to the current Discord channel.

Posting a screenshot/clip to Discord is publishing it, potentially to a
channel with a broad membership -- so, per the same fail-closed
confirmation posture used for OBS's ``obs_start_streaming_confirmed``,
every upload requires human approval before it happens by default. There
is no allowlist bypass based on file type/name/channel: unlike
``obs_switch_scene``'s low-stakes local action, "post this file where
other people can see it" is exactly the class of action that should
never become silently automatic just because it's been done before.

An operator may explicitly opt out of the per-call prompt via
``discord.auto_approve_uploads: true`` (mirrors
``desktop.auto_approve_software_install``'s posture) -- e.g. for an
unattended deployment where no one is reliably available to answer the
prompt within its timeout, and the operator has decided that risk is
acceptable for their own deployment.
"""

from __future__ import annotations

import os

from missy.tools.base import BaseTool, ToolPermissions, ToolResult
from missy.tools.builtin._desktop_shared import load_missy_config, require_approval


def _auto_approve_uploads() -> bool:
    cfg = load_missy_config()
    discord_cfg = getattr(cfg, "discord", None) if cfg is not None else None
    return bool(getattr(discord_cfg, "auto_approve_uploads", False))


class DiscordUploadTool(BaseTool):
    """Upload a local file to a Discord channel using the configured bot token."""

    name = "discord_upload_file"
    description = (
        "Upload a file from the local filesystem to a Discord channel. "
        "Use this to post images, screenshots, documents, or any file into Discord. "
        "Requires file_path (absolute path) and channel_id (the Discord channel snowflake ID). "
        "Requires human approval before posting unless discord.auto_approve_uploads is set."
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
        # Check basic preconditions before bothering a human with an
        # approval prompt for an upload that can't succeed anyway.
        bot_token = os.environ.get("DISCORD_BOT_TOKEN", "")
        if not bot_token:
            return ToolResult(success=False, output=None, error="DISCORD_BOT_TOKEN not set")

        if not _auto_approve_uploads():
            denial = require_approval(
                action=f"Post {file_path!r} to Discord channel {channel_id}",
                reason=caption or "No caption given.",
                risk="medium",
            )
            if denial:
                return ToolResult(success=False, output=None, error=denial)

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
