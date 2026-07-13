"""Discord image analysis — download attachments and run vision model.

Downloads image attachments from Discord CDN and sends them to the
configured vision model (Ollama by default) for analysis. Used by the
``!analyze`` command and inline image processing.
"""

from __future__ import annotations

import base64
import logging
import os
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import urlparse

logger = logging.getLogger(__name__)

# Raised from the original 8MB/8192px/40MP defaults -- live audit-log
# evidence (discord.channel.image_attachment_allowed events) showed real
# user photos landing at 3-5MB and 3400+px wide well within normal use,
# and modern phone cameras routinely exceed the old caps. These bounds
# exist to stop a decompression-bomb-style or resource-exhaustion
# attachment, not to police ordinary photo/screenshot sizes -- the
# downstream ImagePipeline resizes/normalizes for the vision model
# regardless of the original dimensions.
MAX_IMAGE_ATTACHMENT_BYTES = 25 * 1024 * 1024
MAX_IMAGE_DIMENSION = 16384
MAX_IMAGE_PIXELS = 150_000_000

_DISCORD_ATTACHMENT_HOSTS = frozenset(
    {
        "cdn.discordapp.com",
        "media.discordapp.net",
    }
)

# Image content types we accept for vision analysis.
_IMAGE_CONTENT_TYPES = frozenset(
    {
        "image/png",
        "image/jpeg",
        "image/jpg",
        "image/gif",
        "image/webp",
    }
)

# File extensions we treat as images when content_type is missing.
_IMAGE_EXTENSIONS = frozenset(
    {
        ".png",
        ".jpg",
        ".jpeg",
        ".gif",
        ".webp",
        ".bmp",
        ".tiff",
    }
)


@dataclass(frozen=True)
class AttachmentValidation:
    """Metadata validation result for a Discord attachment."""

    allowed: bool
    reasons: list[str] = field(default_factory=list)
    details: dict[str, Any] = field(default_factory=dict)


def sanitize_attachment_filename(filename: str | None, default: str = "attachment") -> str:
    """Return a basename-only attachment filename safe for local metadata/use."""
    safe_filename = os.path.basename(filename or "").replace("\x00", "")
    return safe_filename or default


def _normalise_content_type(attachment: dict[str, Any]) -> str:
    return str(attachment.get("content_type") or "").split(";", 1)[0].strip().lower()


def _filename_extension(filename: str) -> str:
    return os.path.splitext(filename.lower())[1]


def _int_or_none(value: Any) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _attachment_host(url: str) -> str:
    try:
        return (urlparse(url).hostname or "").lower()
    except Exception:
        return ""


def validate_image_attachment(attachment: dict[str, Any]) -> AttachmentValidation:
    """Validate Discord image attachment metadata before routing or download."""
    reasons: list[str] = []
    raw_filename = str(attachment.get("filename") or "")
    safe_filename = sanitize_attachment_filename(raw_filename)
    ext = _filename_extension(safe_filename)
    content_type = _normalise_content_type(attachment)
    url = str(attachment.get("url") or attachment.get("proxy_url") or "")
    parsed = urlparse(url) if url else None
    host = (parsed.hostname or "").lower() if parsed else ""

    if not url:
        reasons.append("missing_url")
    elif parsed.scheme != "https" or host not in _DISCORD_ATTACHMENT_HOSTS:
        reasons.append("invalid_discord_cdn_url")

    # Deliberately does NOT also require the filename extension to match
    # content_type's "canonical" extension: live audit-log evidence
    # (discord.channel.attachment_denied events) showed Discord routinely
    # serving a pasted screenshot as content_type "image/webp" while
    # keeping Discord's own auto-generated filename "image.png" -- a
    # normal, extremely common client-side transcoding behavior, not a
    # spoofing attempt. content_type being a recognised real image MIME
    # type is the thing that actually matters (it's what the downstream
    # vision pipeline decodes against); the filename is cosmetic/display
    # only, already basename-sanitized above, and never trusted for
    # anything security-relevant.
    if content_type:
        if content_type not in _IMAGE_CONTENT_TYPES:
            reasons.append("unsupported_content_type")
    elif ext not in _IMAGE_EXTENSIONS:
        reasons.append("unsupported_file_extension")

    size = _int_or_none(attachment.get("size"))
    if size is not None:
        if size < 0:
            reasons.append("invalid_size")
        elif size > MAX_IMAGE_ATTACHMENT_BYTES:
            reasons.append("image_too_large")

    width = _int_or_none(attachment.get("width"))
    height = _int_or_none(attachment.get("height"))
    if width is not None and (width <= 0 or width > MAX_IMAGE_DIMENSION):
        reasons.append("invalid_width")
    if height is not None and (height <= 0 or height > MAX_IMAGE_DIMENSION):
        reasons.append("invalid_height")
    if (
        width is not None
        and height is not None
        and width > 0
        and height > 0
        and width * height > MAX_IMAGE_PIXELS
    ):
        reasons.append("image_dimensions_too_large")

    details = {
        "filename": safe_filename,
        "content_type": content_type,
        "size": size,
        "width": width,
        "height": height,
        "url_host": host or _attachment_host(url),
        "max_size": MAX_IMAGE_ATTACHMENT_BYTES,
        "max_dimension": MAX_IMAGE_DIMENSION,
        "max_pixels": MAX_IMAGE_PIXELS,
    }
    return AttachmentValidation(allowed=not reasons, reasons=reasons, details=details)


def is_image_attachment(attachment: dict[str, Any]) -> bool:
    """Return True if the Discord attachment dict looks like an image."""
    ct = _normalise_content_type(attachment)
    if ct in _IMAGE_CONTENT_TYPES:
        return True
    filename = (attachment.get("filename") or "").lower()
    return any(filename.endswith(ext) for ext in _IMAGE_EXTENSIONS)


def find_latest_image(messages: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Find the most recent image attachment across a list of messages.

    Args:
        messages: List of Discord Message objects (newest first).

    Returns:
        The attachment dict, or None if no image found.
    """
    for msg in messages:
        for att in msg.get("attachments") or []:
            if is_image_attachment(att):
                return att
    return None


def _get_ollama_base_url() -> str:
    """Return the Ollama base URL from config or default."""
    try:
        from missy.config.settings import load_config

        cfg = load_config()
        provider_cfg = cfg.providers.get("ollama")
        if provider_cfg and provider_cfg.base_url:
            return provider_cfg.base_url.rstrip("/")
    except Exception:
        logger.debug("Ollama config load failed", exc_info=True)
    return "http://localhost:11434"


def _get_vision_model() -> str:
    """Return the vision model name from config or default."""
    try:
        from missy.config.settings import load_config

        cfg = load_config()
        # Allow configuring via providers.ollama.vision_model or fall back
        provider_cfg = cfg.providers.get("ollama")
        if provider_cfg:
            # Check for vision_model in extra config
            extra = getattr(provider_cfg, "extra", None) or {}
            if isinstance(extra, dict) and extra.get("vision_model"):
                return extra["vision_model"]
    except Exception:
        logger.debug("Vision model config load failed", exc_info=True)
    return "minicpm-v"


def analyze_image_bytes(
    image_data: bytes,
    question: str = "Describe what you see in this image in detail.",
    *,
    timeout: int = 120,
) -> str:
    """Send image bytes to the Ollama vision model and return the analysis.

    Args:
        image_data: Raw image file bytes (PNG, JPEG, etc.).
        question: The prompt/question to ask the vision model.
        timeout: HTTP timeout in seconds.

    Returns:
        The vision model's text response.

    Raises:
        RuntimeError: If the vision model is unreachable or returns an error.
    """
    from missy.gateway.client import PolicyHTTPClient

    base_url = _get_ollama_base_url()
    model = _get_vision_model()
    b64_image = base64.b64encode(image_data).decode("ascii")

    body: dict[str, Any] = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": question,
                "images": [b64_image],
            }
        ],
        "stream": False,
    }

    client = PolicyHTTPClient(category="provider", timeout=timeout)
    try:
        resp = client.post(f"{base_url}/api/chat", json=body)
        resp.raise_for_status()
    except Exception as exc:
        raise RuntimeError(f"Vision model request failed: {exc}") from exc

    data = resp.json()
    return data.get("message", {}).get("content", "")


def analyze_discord_attachment(
    rest_client: Any,
    attachment: dict[str, Any],
    question: str = "Describe what you see in this image in detail.",
    *,
    save_path: str | None = None,
) -> dict[str, str]:
    """Download a Discord attachment and analyze it with the vision model.

    Args:
        rest_client: A :class:`DiscordRestClient` instance.
        attachment: Discord attachment dict (must have ``url`` key).
        question: The prompt to ask the vision model.
        save_path: If provided, save the image to this path for documentation.

    Returns:
        Dict with keys ``analysis`` (vision model text), ``filename``,
        and optionally ``saved_to`` if save_path was provided.
    """
    url = attachment.get("url") or attachment.get("proxy_url")
    if not url:
        raise ValueError("Attachment has no download URL.")
    validation = validate_image_attachment(attachment)
    if not validation.allowed:
        raise ValueError(
            "Attachment failed image metadata validation: " + ", ".join(validation.reasons)
        )

    filename = validation.details["filename"]
    image_data = rest_client.download_attachment(url)
    if len(image_data) > MAX_IMAGE_ATTACHMENT_BYTES:
        raise ValueError(
            f"Downloaded attachment exceeds maximum image size "
            f"({len(image_data)} > {MAX_IMAGE_ATTACHMENT_BYTES} bytes)."
        )

    # Optionally save for documentation.
    saved_to = None
    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True, mode=0o700)
        fd = os.open(save_path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "wb") as f:
            f.write(image_data)
        saved_to = save_path
        logger.info("Saved Discord attachment to %s (%d bytes)", save_path, len(image_data))

    analysis = analyze_image_bytes(image_data, question)

    result: dict[str, str] = {
        "analysis": analysis,
        "filename": filename,
    }
    if saved_to:
        result["saved_to"] = saved_to
    return result


def save_discord_attachment(
    rest_client: Any,
    attachment: dict[str, Any],
    save_dir: str = "~/workspace/screenshots",
) -> str:
    """Download and save a Discord attachment to disk.

    Args:
        rest_client: A :class:`DiscordRestClient` instance.
        attachment: Discord attachment dict.
        save_dir: Directory to save the file into.

    Returns:
        The absolute path where the file was saved.
    """
    import time

    url = attachment.get("url") or attachment.get("proxy_url")
    if not url:
        raise ValueError("Attachment has no download URL.")

    validation = validate_image_attachment(attachment)
    if not validation.allowed:
        raise ValueError(
            "Attachment failed image metadata validation: " + ", ".join(validation.reasons)
        )

    raw_filename = attachment.get("filename", "screenshot.png")
    safe_filename = validation.details["filename"]
    save_dir = os.path.expanduser(save_dir)
    os.makedirs(save_dir, exist_ok=True, mode=0o700)

    # Prefix with timestamp to avoid collisions.
    ts = time.strftime("%Y%m%d_%H%M%S")
    dest = os.path.join(save_dir, f"{ts}_{safe_filename}")
    # Final guard: resolved path must stay inside save_dir.
    if not os.path.realpath(dest).startswith(os.path.realpath(save_dir)):
        raise ValueError(f"Attachment filename {raw_filename!r} resolves outside save directory.")

    image_data = rest_client.download_attachment(url)
    if len(image_data) > MAX_IMAGE_ATTACHMENT_BYTES:
        raise ValueError(
            f"Downloaded attachment exceeds maximum image size "
            f"({len(image_data)} > {MAX_IMAGE_ATTACHMENT_BYTES} bytes)."
        )
    fd = os.open(dest, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "wb") as f:
        f.write(image_data)

    logger.info("Saved attachment to %s (%d bytes)", dest, len(image_data))
    return dest
