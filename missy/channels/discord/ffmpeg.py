"""FFmpeg availability helper for Discord voice MVP."""

from __future__ import annotations

import shutil


def ensure_ffmpeg_available() -> str:
    """Return the ffmpeg binary path, raising RuntimeError if missing."""

    path = shutil.which("ffmpeg")
    if not path:
        raise RuntimeError(
            "ffmpeg not found in PATH. Install it (e.g. apt install ffmpeg) before using voice."
        )
    return path
