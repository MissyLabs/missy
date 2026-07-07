"""Scoped bindings between running Discord voice managers and tools.

The voice manager lives on the Discord asyncio loop inside the channel
process. Built-in tools (``discord_voice_*``) run on the agent's executor
thread and need to invoke the manager's coroutines from outside that loop.

This module exposes a small thread-safe registry so each Discord channel can
publish the active manager + its loop for a specific account/guild scope, and
the tools can fetch the matching scope on demand. Ambiguous lookups fail
closed by returning ``None``.
"""

from __future__ import annotations

import asyncio
import threading
from dataclasses import dataclass
from typing import Any

_lock = threading.Lock()
_bindings: dict[tuple[str, str], _VoiceBinding] = {}


@dataclass
class _VoiceBinding:
    manager: Any
    loop: asyncio.AbstractEventLoop
    account_id: str
    guild_id: str


def _normalize(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _scope(account_id: Any = "", guild_id: Any = "") -> tuple[str, str]:
    return _normalize(account_id), _normalize(guild_id)


def set_voice_binding(
    manager: Any,
    loop: asyncio.AbstractEventLoop,
    *,
    account_id: str = "",
    guild_id: str = "",
) -> None:
    """Publish a voice manager + loop for an account/guild scope."""
    account_key, guild_key = _scope(account_id, guild_id)
    with _lock:
        _bindings[(account_key, guild_key)] = _VoiceBinding(
            manager=manager,
            loop=loop,
            account_id=account_key,
            guild_id=guild_key,
        )


def clear_voice_binding(
    *,
    account_id: str | None = None,
    guild_id: str | None = None,
    manager: Any | None = None,
) -> None:
    """Remove matching bindings.

    With no arguments this clears the entire process-local registry. Passing a
    manager clears every scope published for that manager, which is useful when
    a Discord channel shuts down.
    """
    with _lock:
        if account_id is None and guild_id is None and manager is None:
            _bindings.clear()
            return

        account_key = _normalize(account_id) if account_id is not None else None
        guild_key = _normalize(guild_id) if guild_id is not None else None
        stale = [
            key
            for key, binding in _bindings.items()
            if (account_key is None or key[0] == account_key)
            and (guild_key is None or key[1] == guild_key)
            and (manager is None or binding.manager is manager)
        ]
        for key in stale:
            _bindings.pop(key, None)


def get_voice_binding(
    *,
    account_id: str | None = None,
    guild_id: str | None = None,
) -> _VoiceBinding | None:
    """Return the matching binding, or ``None`` when absent/ambiguous."""
    with _lock:
        if not _bindings:
            return None

        account_key = _normalize(account_id) if account_id is not None else None
        guild_key = _normalize(guild_id) if guild_id is not None else None

        if account_key is not None and guild_key is not None:
            return _bindings.get((account_key, guild_key))

        matches = [
            binding
            for (candidate_account, candidate_guild), binding in _bindings.items()
            if (account_key is None or candidate_account == account_key)
            and (guild_key is None or candidate_guild == guild_key)
        ]
        if len(matches) == 1:
            return matches[0]
        return None


def list_voice_bindings() -> list[dict[str, str]]:
    """Return a diagnostics-friendly snapshot of registered scopes."""
    with _lock:
        return [
            {"account_id": binding.account_id, "guild_id": binding.guild_id}
            for binding in _bindings.values()
        ]
