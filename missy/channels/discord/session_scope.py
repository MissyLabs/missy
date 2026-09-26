"""Stable, privacy-preserving conversation scopes for Discord traffic."""

from __future__ import annotations


def discord_session_id(author_id: str, guild_id: str = "", channel_id: str = "") -> str:
    """Return a session key isolated by principal and Discord location.

    A Discord user must not carry private DM history into a guild response,
    nor share history between guild channels or threads.  Discord snowflakes
    are stable identifiers, so the resulting key remains durable across
    process restarts without relying on display names.
    """
    principal = str(author_id or "anonymous")
    channel = str(channel_id or "unknown")
    guild = str(guild_id or "")
    if guild:
        return f"discord:user:{principal}:guild:{guild}:channel:{channel}"
    return f"discord:user:{principal}:dm:{channel}"
