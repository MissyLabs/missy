"""Blank/uniform screen-capture detection.

A display that is off, locked, or not reachable by the capturing process
(no compositor access) commonly yields a valid PNG whose every pixel is the
same colour -- e.g. an all-black 640x480 frame from ``scrot``.  Sending
such a frame to a vision model invites hallucination (the model "sees"
windows and buttons that are not there), so every screenshot -> vision
path checks the capture first and refuses to analyse a uniform frame.

The check is dependency-light: Pillow is used for files and numpy for
in-memory arrays.  When the needed library is missing or the image can't
be decoded the helpers return ``None`` ("unknown") and callers proceed
as before -- the guard degrades gracefully rather than blocking.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

#: A frame whose per-channel pixel standard deviation is below this value
#: (on a 0-255 scale) in every channel is treated as uniform.  A real
#: desktop -- even a dark terminal with a single line of text -- sits far
#: above this; a single blinking cursor on a black 640x480 screen is ~5.
UNIFORM_STDDEV_THRESHOLD = 1.5

BLANK_CAPTURE_MESSAGE = (
    "Screen capture is blank: every pixel is the same colour ({detail}). "
    "The display is probably off, locked/screensaver-blanked, or this "
    "process has no access to the compositor, so there is nothing to "
    "read. The image was NOT sent to the vision model. Report this to "
    "the user instead of describing screen content."
)


def _stats_detail(means: list[float]) -> str:
    rgb = ", ".join(f"{m:.0f}" for m in means[:3])
    return f"mean channel value {rgb}"


def uniform_image_file(path: str) -> str | None:
    """Return a short description if the image at *path* is uniform.

    Returns ``None`` when the image has visible variation, when Pillow is
    not installed, or when the file can't be decoded (fail open).
    """
    try:
        from PIL import Image, ImageStat
    except ImportError:
        logger.debug("blank_frame: Pillow not installed; skipping uniform check")
        return None
    try:
        with Image.open(path) as img:
            img.load()
            if img.mode not in ("L", "RGB"):
                img = img.convert("RGB")
            stat = ImageStat.Stat(img)
            stddevs = list(stat.stddev)
            means = list(stat.mean)
    except Exception as exc:  # noqa: BLE001 - unreadable image: fail open
        logger.debug("blank_frame: could not analyse %s: %s", path, exc)
        return None
    if stddevs and max(stddevs) < UNIFORM_STDDEV_THRESHOLD:
        return _stats_detail(means)
    return None


def uniform_image_array(image: Any) -> str | None:
    """Like :func:`uniform_image_file` for a numpy (OpenCV) image array."""
    try:
        import numpy as np
    except ImportError:
        return None
    try:
        arr = np.asarray(image)
        if arr.size == 0:
            return "empty image"
        if arr.ndim == 2:
            arr = arr[:, :, None]
        flat = arr.reshape(-1, arr.shape[-1]).astype("float64")
        stddevs = flat.std(axis=0)
        means = flat.mean(axis=0)
    except Exception as exc:  # noqa: BLE001
        logger.debug("blank_frame: could not analyse array: %s", exc)
        return None
    if float(stddevs.max()) < UNIFORM_STDDEV_THRESHOLD:
        # OpenCV arrays are BGR; report in RGB order for readability.
        ordered = list(means[::-1]) if len(means) == 3 else list(means)
        return _stats_detail([float(m) for m in ordered])
    return None


def blank_capture_error(detail: str) -> str:
    """Return the user/model-facing error for a blank capture."""
    return BLANK_CAPTURE_MESSAGE.format(detail=detail)
