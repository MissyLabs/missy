"""Provider-specific image format helpers for vision API calls.

Different LLM providers expect image data in different message formats.
This module handles the conversion from a base64-encoded image to the
correct provider-specific message structure.

Anthropic: ``{"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": "..."}}``
OpenAI:    ``{"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,..."}}``
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


def format_image_for_anthropic(
    image_base64: str,
    media_type: str = "image/jpeg",
) -> dict[str, Any]:
    """Format a base64 image for Anthropic's Messages API.

    Returns a content block suitable for inclusion in a user message's
    ``content`` list.
    """
    return {
        "type": "image",
        "source": {
            "type": "base64",
            "media_type": media_type,
            "data": image_base64,
        },
    }


def format_image_for_openai(
    image_base64: str,
    media_type: str = "image/jpeg",
    detail: str = "auto",
) -> dict[str, Any]:
    """Format a base64 image for OpenAI's Chat Completions API.

    Returns a content block suitable for inclusion in a user message's
    ``content`` list.
    """
    data_uri = f"data:{media_type};base64,{image_base64}"
    return {
        "type": "image_url",
        "image_url": {
            "url": data_uri,
            "detail": detail,
        },
    }


def format_image_for_provider(
    provider_name: str,
    image_base64: str,
    media_type: str = "image/jpeg",
) -> dict[str, Any]:
    """Format a base64 image for the given provider.

    Parameters
    ----------
    provider_name:
        Provider identifier: ``"anthropic"``, ``"openai"``, or ``"ollama"``.
    image_base64:
        Base64-encoded image data.
    media_type:
        MIME type of the image.

    Returns
    -------
    dict
        A content block in the provider's expected format.
    """
    name = provider_name.lower()

    if name == "anthropic":
        return format_image_for_anthropic(image_base64, media_type)
    elif name in ("openai", "gpt"):
        return format_image_for_openai(image_base64, media_type)
    elif name == "ollama":
        # Ollama uses the same format as OpenAI
        return format_image_for_openai(image_base64, media_type)
    else:
        # Default to Anthropic format
        logger.warning(
            "Unknown provider %r for image formatting — using Anthropic format",
            provider_name,
        )
        return format_image_for_anthropic(image_base64, media_type)


def build_vision_message(
    provider_name: str,
    image_base64: str,
    prompt: str,
    media_type: str = "image/jpeg",
) -> dict[str, Any]:
    """Build a complete user message with image + text for a provider.

    Returns
    -------
    dict
        A message dict with ``role`` and ``content`` fields.
    """
    image_block = format_image_for_provider(provider_name, image_base64, media_type)
    text_block = {"type": "text", "text": prompt}

    return {
        "role": "user",
        "content": [image_block, text_block],
    }
