"""Voice channel package for Missy.

Provides the WebSocket-based voice channel, device registry, pairing workflow,
presence tracking, and the underlying server implementation.

Typical usage::

    from missy.channels.voice import VoiceChannel

    channel = VoiceChannel(host="127.0.0.1", port=8765)
    channel.start(agent_runtime)
"""

from __future__ import annotations

from missy.channels.voice.channel import VoiceChannel
from missy.channels.voice.pairing import PairingManager
from missy.channels.voice.presence import PresenceStore
from missy.channels.voice.registry import DeviceRegistry, EdgeNode
from missy.channels.voice.server import VoiceServer

__all__ = [
    "VoiceChannel",
    "DeviceRegistry",
    "EdgeNode",
    "PairingManager",
    "PresenceStore",
    "VoiceServer",
]
