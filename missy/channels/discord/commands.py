"""Discord slash command definitions and routing.

Defines the ``/ask``, ``/status``, ``/model``, and ``/help`` commands and
provides :func:`handle_slash_command` to route an incoming interaction
dict to the correct handler.

Example::

    from missy.channels.discord.commands import handle_slash_command

    response_text = await handle_slash_command(interaction, channel)
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from missy.channels.discord.channel import DiscordChannel

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Command definitions
# ---------------------------------------------------------------------------

#: Application command objects to register with Discord.
SLASH_COMMANDS: list[dict[str, Any]] = [
    {
        "name": "ask",
        "description": "Ask Missy a question.",
        "options": [
            {
                "type": 3,  # STRING
                "name": "prompt",
                "description": "Your question or instruction.",
                "required": True,
            }
        ],
    },
    {
        "name": "status",
        "description": "Show Missy's current connection status.",
        "options": [],
    },
    {
        "name": "model",
        "description": "Show or set the active AI model.",
        "options": [
            {
                "type": 3,  # STRING
                "name": "name",
                "description": "Model identifier to switch to (omit to show current).",
                "required": False,
            }
        ],
    },
    {
        "name": "help",
        "description": "Show available Missy commands.",
        "options": [],
    },
]

# ---------------------------------------------------------------------------
# Handlers
# ---------------------------------------------------------------------------


def _get_option(interaction: dict[str, Any], name: str) -> str | None:
    """Extract a named option value from an interaction data dict."""
    data = interaction.get("data") or {}
    options = data.get("options") or []
    for opt in options:
        if opt.get("name") == name:
            return str(opt.get("value", ""))
    return None


async def _handle_ask(interaction: dict[str, Any], channel: DiscordChannel) -> str:
    """Handle ``/ask`` — forward prompt to the agent and return the reply."""
    import asyncio

    prompt = _get_option(interaction, "prompt")
    if not prompt:
        return "Please provide a prompt with `/ask <your question>`."

    try:
        from missy.agent.runtime import DISCORD_SYSTEM_PROMPT, AgentConfig, AgentRuntime

        agent_cfg = AgentConfig(
            provider="anthropic",
            system_prompt=DISCORD_SYSTEM_PROMPT,
            capability_mode="discord",
        )
        agent = AgentRuntime(agent_cfg)
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, agent.run, prompt, "discord")
    except Exception as exc:
        logger.exception("Slash /ask handler failed: %s", exc)
        return f"Sorry, I encountered an error: {exc}"


async def _handle_status(interaction: dict[str, Any], channel: DiscordChannel) -> str:
    """Handle ``/status`` — report connection and config state."""
    account_cfg = channel.account_config
    bot_id = channel.bot_user_id or "unknown"
    lines = [
        "**Missy Discord Status**",
        f"- Bot user ID: `{bot_id}`",
        f"- DM policy: `{account_cfg.dm_policy.value}`",
        f"- Ignore bots: `{account_cfg.ignore_bots}`",
        f"- Guild policies configured: `{len(account_cfg.guild_policies)}`",
    ]
    return "\n".join(lines)


async def _handle_model(interaction: dict[str, Any], channel: DiscordChannel) -> str:
    """Handle ``/model`` — show current model (setting not yet implemented)."""
    name = _get_option(interaction, "name")
    if name:
        return "Dynamic model switching is not yet supported. Current model unchanged."
    try:
        from missy.providers.registry import get_registry

        registry = get_registry()
        provider_names = list(registry.list_providers())
        if provider_names:
            return f"Active providers: {', '.join(provider_names)}"
        return "No providers are currently configured."
    except Exception:
        return "Provider information is unavailable."


async def _handle_help(interaction: dict[str, Any], channel: DiscordChannel) -> str:
    """Handle ``/help`` — list available slash commands."""
    lines = ["**Available Missy commands:**", ""]
    for cmd in SLASH_COMMANDS:
        name = cmd["name"]
        desc = cmd.get("description", "")
        lines.append(f"`/{name}` — {desc}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------

_HANDLERS = {
    "ask": _handle_ask,
    "status": _handle_status,
    "model": _handle_model,
    "help": _handle_help,
}


async def handle_slash_command(
    interaction: dict[str, Any],
    channel: DiscordChannel,
) -> str:
    """Route an INTERACTION_CREATE payload to the correct command handler.

    Args:
        interaction: The raw Discord interaction dict.
        channel: The :class:`DiscordChannel` instance handling the request.

    Returns:
        A string response to send back to the user.
    """
    data = interaction.get("data") or {}
    command_name = data.get("name", "")
    handler = _HANDLERS.get(command_name)
    if handler is None:
        return f"Unknown command: `/{command_name}`"
    return await handler(interaction, channel)
