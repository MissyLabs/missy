"""Process-level binding between the running DiscordVoiceManager and tools.

The voice manager lives on the Discord asyncio loop inside the channel
process. Built-in tools (``discord_voice_*``) run on the agent's executor
thread and need to invoke the manager's coroutines from outside that loop.

This module exposes a small thread-safe binding so the channel can publish
the active manager + its loop once, and the tools can fetch them on demand
and use :func:`asyncio.run_coroutine_threadsafe` to dispatch calls.
"""

from __future__ import annotations

import asyncio
import threading
from dataclasses import dataclass
from typing import Any

_lock = threading.Lock()
_binding: _VoiceBinding | None = None


@dataclass
class _VoiceBinding:
    manager: Any
    loop: asyncio.AbstractEventLoop


def set_voice_binding(manager: Any, loop: asyncio.AbstractEventLoop) -> None:
    """Publish the active voice manager + loop for tool dispatch."""
    global _binding
    with _lock:
        _binding = _VoiceBinding(manager=manager, loop=loop)


def clear_voice_binding() -> None:
    """Remove the current binding (call from channel.stop())."""
    global _binding
    with _lock:
        _binding = None


def get_voice_binding() -> _VoiceBinding | None:
    with _lock:
        return _binding
