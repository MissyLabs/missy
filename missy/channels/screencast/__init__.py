"""Screencast channel — browser-based screen capture for Missy.

Provides a self-hosted web page where users share their screen via the
browser's ``getDisplayMedia()`` API.  Frames stream to Missy via WebSocket
and are analyzed by the Ollama vision model.
"""

from missy.channels.screencast.auth import ScreencastSession, ScreencastTokenRegistry
from missy.channels.screencast.channel import ScreencastChannel
from missy.channels.screencast.session_manager import (
    AnalysisResult,
    FrameMetadata,
    SessionManager,
)

__all__ = [
    "AnalysisResult",
    "FrameMetadata",
    "ScreencastChannel",
    "ScreencastSession",
    "ScreencastTokenRegistry",
    "SessionManager",
]
