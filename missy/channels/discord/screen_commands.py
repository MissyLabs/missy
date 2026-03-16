"""Screencast command handlers for Discord.

Commands:
- !screen share [label]   — create a session and return the share URL
- !screen list             — show active sessions with stats
- !screen stop [id]        — revoke a session
- !screen analyze [id]     — show latest analysis result
- !screen status           — show server status

Parsed from MESSAGE_CREATE events and routed to
:class:`~missy.channels.screencast.channel.ScreencastChannel`.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ScreenCommandResult:
    handled: bool
    reply: str | None = None


async def maybe_handle_screen_command(
    *,
    content: str,
    channel_id: str,
    author_id: str,
    screencast: Any | None,
) -> ScreenCommandResult:
    """Parse and execute a screen command if applicable.

    Args:
        content: Message content (already stripped of bot mentions).
        channel_id: Discord channel ID for the message.
        author_id: Discord user ID of the sender.
        screencast: The :class:`ScreencastChannel` instance, or ``None``.

    Returns:
        A result indicating whether the message was handled and an
        optional reply to send back.
    """
    text = (content or "").strip()
    if not text.startswith("!screen"):
        return ScreenCommandResult(False)

    parts = text.split(None, 2)  # ["!screen", subcommand, ...rest]
    if len(parts) < 2:
        return ScreenCommandResult(
            True,
            "Usage: `!screen share [label]` | `!screen list` | "
            "`!screen stop [id]` | `!screen analyze [id]` | `!screen status`",
        )

    subcmd = parts[1].lower()
    rest = parts[2].strip() if len(parts) > 2 else ""

    if subcmd not in ("share", "list", "stop", "analyze", "status"):
        return ScreenCommandResult(False)

    if screencast is None:
        return ScreenCommandResult(True, "Screencast is not enabled on this bot.")

    # ------- !screen share [label] -------
    if subcmd == "share":
        return _handle_share(
            screencast=screencast,
            channel_id=channel_id,
            author_id=author_id,
            label=rest,
        )

    # ------- !screen list -------
    if subcmd == "list":
        return _handle_list(screencast=screencast)

    # ------- !screen stop [session_id] -------
    if subcmd == "stop":
        return _handle_stop(screencast=screencast, session_id=rest)

    # ------- !screen analyze [session_id] -------
    if subcmd == "analyze":
        return _handle_analyze(screencast=screencast, session_id=rest)

    # ------- !screen status -------
    if subcmd == "status":
        return _handle_status(screencast=screencast)

    return ScreenCommandResult(False)


def _handle_share(
    *,
    screencast: Any,
    channel_id: str,
    author_id: str,
    label: str,
) -> ScreenCommandResult:
    """Create a new screencast session."""
    try:
        session_id, _token, share_url = screencast.create_session(
            created_by=author_id,
            discord_channel_id=channel_id,
            label=label or "screen share",
        )
    except Exception as exc:
        logger.error("!screen share failed: %s", exc, exc_info=True)
        return ScreenCommandResult(True, f"Failed to create session: {exc}")

    reply = (
        f"**Screen share session created**\n"
        f"Session: `{session_id}`\n"
        f"Label: {label or 'screen share'}\n\n"
        f"Open this link in your browser to start sharing:\n{share_url}\n\n"
        f"*Analysis results will be posted here automatically.*"
    )
    return ScreenCommandResult(True, reply)


def _handle_list(*, screencast: Any) -> ScreenCommandResult:
    """List active screencast sessions."""
    sessions = screencast.get_active_sessions()
    if not sessions:
        return ScreenCommandResult(True, "No active screencast sessions.")

    lines = ["**Active screencast sessions:**\n"]
    for s in sessions:
        elapsed = time.time() - s["created_at"]
        elapsed_str = _format_duration(elapsed)
        last_frame = ""
        if s["last_frame_at"]:
            ago = time.time() - s["last_frame_at"]
            last_frame = f", last frame {_format_duration(ago)} ago"
        lines.append(
            f"- `{s['session_id']}` — {s['label'] or 'unlabeled'} "
            f"(by <@{s['created_by']}>, {elapsed_str}, "
            f"{s['frame_count']} frames, {s['analysis_count']} analyses{last_frame})"
        )

    return ScreenCommandResult(True, "\n".join(lines))


def _handle_stop(*, screencast: Any, session_id: str) -> ScreenCommandResult:
    """Revoke a screencast session."""
    if not session_id:
        # If no session_id given, stop the most recent one.
        sessions = screencast.get_active_sessions()
        if not sessions:
            return ScreenCommandResult(True, "No active sessions to stop.")
        session_id = sessions[-1]["session_id"]

    if screencast.revoke_session(session_id):
        return ScreenCommandResult(True, f"Session `{session_id}` stopped.")
    return ScreenCommandResult(True, f"Session `{session_id}` not found.")


def _handle_analyze(*, screencast: Any, session_id: str) -> ScreenCommandResult:
    """Show the latest analysis result for a session."""
    if not session_id:
        # Use the most recent active session.
        sessions = screencast.get_active_sessions()
        if not sessions:
            return ScreenCommandResult(True, "No active sessions.")
        session_id = sessions[-1]["session_id"]

    result = screencast.get_latest_analysis(session_id)
    if result is None:
        return ScreenCommandResult(
            True,
            f"No analysis results yet for session `{session_id}`.",
        )

    header = f"**Latest analysis for `{session_id}` (frame #{result.frame_number}):**\n"
    text = result.analysis_text
    max_body = 2000 - len(header) - 30
    if len(text) > max_body:
        text = text[:max_body].rstrip() + "..."

    footer = f"\n*({result.processing_ms}ms, model: {result.model})*"
    return ScreenCommandResult(True, header + text + footer)


def _handle_status(*, screencast: Any) -> ScreenCommandResult:
    """Show server status."""
    status = screencast.get_status()
    if not status.get("running"):
        return ScreenCommandResult(True, "Screencast server is not running.")

    sessions_info = status.get("sessions", {})
    connected = sessions_info.get("connected_sessions", 0)
    max_s = sessions_info.get("max_sessions", 0)
    queue_size = sessions_info.get("queue_size", 0)

    reply = (
        f"**Screencast server status**\n"
        f"Running: yes\n"
        f"Address: `{status.get('host', '?')}:{status.get('port', '?')}`\n"
        f"Connected: {connected}/{max_s}\n"
        f"Queue depth: {queue_size}\n"
        f"Active connections: {status.get('active_connections', 0)}"
    )
    return ScreenCommandResult(True, reply)


def _format_duration(seconds: float) -> str:
    """Format seconds into a human-readable duration."""
    s = int(seconds)
    if s < 60:
        return f"{s}s"
    if s < 3600:
        return f"{s // 60}m {s % 60}s"
    h = s // 3600
    m = (s % 3600) // 60
    return f"{h}h {m}m"
