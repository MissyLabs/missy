"""Discord image analysis — download attachments and run vision model.

Downloads image attachments from Discord CDN and sends them to the
configured vision model (Ollama by default) for analysis. Used by the
``!analyze`` command and inline image processing.
"""

from __future__ import annotations

import base64
import logging
import os
from typing import Any

logger = logging.getLogger(__name__)

# Image content types we accept for vision analysis.
_IMAGE_CONTENT_TYPES = frozenset({
    "image/png",
    "image/jpeg",
    "image/jpg",
    "image/gif",
    "image/webp",
})

# File extensions we treat as images when content_type is missing.
_IMAGE_EXTENSIONS = frozenset({
    ".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp", ".tiff",
})


def is_image_attachment(attachment: dict[str, Any]) -> bool:
    """Return True if the Discord attachment dict looks like an image."""
    ct = (attachment.get("content_type") or "").lower()
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

    filename = attachment.get("filename", "unknown")
    image_data = rest_client.download_attachment(url)

    # Optionally save for documentation.
    saved_to = None
    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True, mode=0o700)
        with open(save_path, "wb") as f:
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

    filename = attachment.get("filename", "screenshot.png")
    save_dir = os.path.expanduser(save_dir)
    os.makedirs(save_dir, exist_ok=True, mode=0o700)

    # Prefix with timestamp to avoid collisions.
    ts = time.strftime("%Y%m%d_%H%M%S")
    dest = os.path.join(save_dir, f"{ts}_{filename}")

    image_data = rest_client.download_attachment(url)
    with open(dest, "wb") as f:
        f.write(image_data)

    logger.info("Saved attachment to %s (%d bytes)", dest, len(image_data))
    return dest
