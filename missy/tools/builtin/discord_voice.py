"""Built-in tools that drive the Discord voice manager.

Allow the agent to join, leave, speak in, and inspect Discord voice
channels in response to natural-language requests (text chat or voice
transcripts). All tools delegate to the active
:class:`~missy.channels.discord.voice.DiscordVoiceManager` via the
binding published in
:mod:`missy.channels.discord.voice_binding`.
"""

from __future__ import annotations

import asyncio
from typing import Any

from missy.channels.discord.voice_binding import get_voice_binding
from missy.tools.base import BaseTool, ToolPermissions, ToolResult

_CALL_TIMEOUT_S = 30.0
_DISCORD_VOICE_PERMISSIONS = ToolPermissions(
    network=True,
    allowed_hosts=["discord.com", "gateway.discord.gg"],
)


def _require_binding() -> tuple[Any, asyncio.AbstractEventLoop] | ToolResult:
    binding = get_voice_binding()
    if binding is None:
        return ToolResult(
            success=False,
            output=None,
            error=(
                "Discord voice is not active. Ask the user to start the bot with "
                "voice enabled (`pip install -e '.[discord_voice]'`) or to run "
                "`!join` once so the voice manager initializes."
            ),
        )
    if not binding.manager.is_ready:
        return ToolResult(
            success=False,
            output=None,
            error="Discord voice manager is still starting up; try again shortly.",
        )
    return binding.manager, binding.loop


def _coerce_guild_id(value: Any) -> int | None:
    if value in (None, "", 0):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _run_coro(loop: asyncio.AbstractEventLoop, coro: Any) -> Any:
    fut = asyncio.run_coroutine_threadsafe(coro, loop)
    return fut.result(timeout=_CALL_TIMEOUT_S)


class DiscordVoiceJoinTool(BaseTool):
    """Join a Discord voice channel."""

    name = "discord_voice_join"
    description = (
        "Join a Discord voice channel in the current server. "
        "Use this when the user asks Missy to join a voice channel, hop on a call, "
        "come into voice, etc. If the user does not specify a channel, pass their "
        "user_id and the bot will join whichever channel they are currently in. "
        "Otherwise pass channel as a channel name (e.g. 'General') or numeric "
        "snowflake ID. guild_id is required and supplied in the message context."
    )
    permissions = _DISCORD_VOICE_PERMISSIONS
    parameters = {
        "guild_id": {
            "type": "string",
            "description": (
                "Discord guild (server) snowflake ID. Pulled from the "
                "[DISCORD CONTEXT] line of the user message."
            ),
            "required": True,
        },
        "channel": {
            "type": "string",
            "description": (
                "Voice channel name or numeric ID to join. Omit to follow the user "
                "into whatever voice channel they are currently in."
            ),
        },
        "user_id": {
            "type": "string",
            "description": (
                "Discord user snowflake ID. Used when 'channel' is omitted to "
                "locate the channel the user is currently sitting in."
            ),
        },
    }

    def execute(
        self,
        *,
        guild_id: str = "",
        channel: str = "",
        user_id: str = "",
        **_kwargs: Any,
    ) -> ToolResult:
        binding = _require_binding()
        if isinstance(binding, ToolResult):
            return binding
        manager, loop = binding

        gid = _coerce_guild_id(guild_id)
        if gid is None:
            return ToolResult(
                success=False,
                output=None,
                error="guild_id is required (look for [DISCORD CONTEXT] in the user message).",
            )

        ch = (channel or "").strip()
        uid = _coerce_guild_id(user_id)

        join_kwargs: dict[str, Any] = {}
        if ch:
            if ch.isdigit():
                join_kwargs["channel_id"] = int(ch)
            else:
                join_kwargs["channel_name"] = ch
        elif uid is not None:
            join_kwargs["user_id"] = uid
        else:
            return ToolResult(
                success=False,
                output=None,
                error=(
                    "Specify a 'channel' name/ID, or pass 'user_id' so Missy can "
                    "follow the user into their current voice channel."
                ),
            )

        try:
            channel_name = _run_coro(loop, manager.join(gid, **join_kwargs))
        except Exception as exc:
            return ToolResult(success=False, output=None, error=f"Join failed: {exc}")

        capabilities = []
        if manager.can_listen:
            capabilities.append("listening")
        if manager.can_speak:
            capabilities.append("speaking")
        status = f" ({', '.join(capabilities)})" if capabilities else ""
        return ToolResult(
            success=True,
            output=f"Joined voice channel '{channel_name}'{status}.",
        )


class DiscordVoiceLeaveTool(BaseTool):
    """Leave the current Discord voice channel."""

    name = "discord_voice_leave"
    description = (
        "Leave the Discord voice channel Missy is currently connected to in the "
        "given guild. Use when the user asks Missy to leave, drop off, disconnect, "
        "or stop the voice call."
    )
    permissions = _DISCORD_VOICE_PERMISSIONS
    parameters = {
        "guild_id": {
            "type": "string",
            "description": (
                "Discord guild (server) snowflake ID. Pulled from the "
                "[DISCORD CONTEXT] line of the user message."
            ),
            "required": True,
        },
    }

    def execute(self, *, guild_id: str = "", **_kwargs: Any) -> ToolResult:
        binding = _require_binding()
        if isinstance(binding, ToolResult):
            return binding
        manager, loop = binding

        gid = _coerce_guild_id(guild_id)
        if gid is None:
            return ToolResult(
                success=False,
                output=None,
                error="guild_id is required (look for [DISCORD CONTEXT] in the user message).",
            )

        try:
            left = _run_coro(loop, manager.leave(gid))
        except Exception as exc:
            return ToolResult(success=False, output=None, error=f"Leave failed: {exc}")

        if left:
            return ToolResult(success=True, output=f"Left voice channel '{left}'.")
        return ToolResult(success=True, output="Was not in a voice channel.")


class DiscordVoiceSayTool(BaseTool):
    """Speak text in the current Discord voice channel via TTS."""

    name = "discord_voice_say"
    description = (
        "Speak text out loud in the Discord voice channel Missy is connected to. "
        "Only use this for explicit user requests to 'say', 'announce', or "
        "'speak' something — do NOT use it to reply to ordinary voice messages "
        "(those are spoken automatically by the voice loop)."
    )
    permissions = _DISCORD_VOICE_PERMISSIONS
    parameters = {
        "guild_id": {
            "type": "string",
            "description": (
                "Discord guild (server) snowflake ID. Pulled from the "
                "[DISCORD CONTEXT] line of the user message."
            ),
            "required": True,
        },
        "text": {
            "type": "string",
            "description": "The text to speak aloud (plain prose, no markdown).",
            "required": True,
        },
    }

    def execute(
        self,
        *,
        guild_id: str = "",
        text: str = "",
        **_kwargs: Any,
    ) -> ToolResult:
        binding = _require_binding()
        if isinstance(binding, ToolResult):
            return binding
        manager, loop = binding

        gid = _coerce_guild_id(guild_id)
        if gid is None:
            return ToolResult(
                success=False,
                output=None,
                error="guild_id is required (look for [DISCORD CONTEXT] in the user message).",
            )

        spoken = (text or "").strip()
        if not spoken:
            return ToolResult(success=False, output=None, error="text is required.")

        try:
            _run_coro(loop, manager.say(gid, spoken))
        except Exception as exc:
            return ToolResult(success=False, output=None, error=f"Speak failed: {exc}")

        return ToolResult(success=True, output=f"Spoke {len(spoken)} characters in voice.")


class DiscordVoiceStatusTool(BaseTool):
    """Report current voice connection and available channels for a guild."""

    name = "discord_voice_status"
    description = (
        "Report Missy's current Discord voice connection state for a guild and "
        "list available voice channels. Use when the user asks where Missy is, "
        "whether she is in voice, or to choose between channels."
    )
    permissions = _DISCORD_VOICE_PERMISSIONS
    parameters = {
        "guild_id": {
            "type": "string",
            "description": (
                "Discord guild (server) snowflake ID. Pulled from the "
                "[DISCORD CONTEXT] line of the user message."
            ),
            "required": True,
        },
    }

    def execute(self, *, guild_id: str = "", **_kwargs: Any) -> ToolResult:
        binding = _require_binding()
        if isinstance(binding, ToolResult):
            return binding
        manager, _loop = binding

        gid = _coerce_guild_id(guild_id)
        if gid is None:
            return ToolResult(
                success=False,
                output=None,
                error="guild_id is required (look for [DISCORD CONTEXT] in the user message).",
            )

        try:
            connected = manager.is_connected(gid)
            current = manager.current_channel_name(gid) if connected else None
            channels = manager.list_voice_channels(gid)
        except Exception as exc:
            return ToolResult(success=False, output=None, error=f"Status check failed: {exc}")

        return ToolResult(
            success=True,
            output={
                "connected": connected,
                "current_channel": current,
                "listening": manager.is_listening(gid),
                "can_listen": manager.can_listen,
                "can_speak": manager.can_speak,
                "available_channels": channels,
            },
        )
