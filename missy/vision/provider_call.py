"""Real multimodal image analysis via whichever configured provider supports it.

Sends actual image bytes to a vision-capable LLM and returns its genuine
reading of the image (a real visual description, including any text/
labels the model can read directly) -- as opposed to the metadata-only
capture tools (``vision_capture``/``vision_burst``) or a text prompt with
no image attached (the previous, broken shape of ``vision_analyze``).

Provider support:

- Anthropic, OpenAI: accept a list of content blocks (text + image) in
  ``Message.content`` natively already -- no special-casing needed here.
- openai-codex, ollama: ``missy/providers/codex_provider.py`` and
  ``missy/providers/ollama_provider.py`` were extended to translate a
  list-content ``Message`` into their own real API shape.
- acpx: structurally cannot carry real image bytes -- it flattens every
  message to plain text for the underlying CLI (see
  ``AcpxProvider._build_prompt()``). Deliberately excluded from the
  candidate list below rather than attempted and left to fail (or worse,
  silently send a garbled Python repr of the content-block list as text).
"""

from __future__ import annotations

import logging

from missy.core.exceptions import ProviderError
from missy.providers.base import Message
from missy.vision.provider_format import build_vision_message

logger = logging.getLogger(__name__)

#: Providers with no structural way to carry real image bytes -- see the
#: acpx note in the module docstring. Excluded from candidate selection
#: entirely, not merely deprioritized, since attempting one produces a
#: confusing garbled-text failure rather than a clean "not supported" one.
_IMAGE_INCOMPATIBLE_PROVIDER_NAMES = frozenset({"acpx"})


def _candidate_providers() -> list:
    """Return available providers to try, default/primary provider first."""
    from missy.providers.registry import get_registry

    registry = get_registry()
    available = [
        p for p in registry.get_available() if p.name not in _IMAGE_INCOMPATIBLE_PROVIDER_NAMES
    ]
    default_name = registry.get_default_name()
    if default_name:
        available.sort(key=lambda p: p.name != default_name)
    return available


def analyze_image_with_provider_fallback(
    prompt: str,
    image_base64: str,
    media_type: str = "image/jpeg",
    *,
    session_id: str = "",
    task_id: str = "",
) -> tuple[str, str]:
    """Send an image + prompt to the first available provider that can handle it.

    Args:
        prompt: The text instruction to send alongside the image (e.g.
            from :class:`~missy.vision.analysis.AnalysisPromptBuilder`).
        image_base64: Base64-encoded image bytes.
        media_type: Image MIME type (e.g. ``"image/jpeg"``).
        session_id: For audit events.
        task_id: For audit events.

    Returns:
        A ``(analysis_text, provider_name)`` tuple -- the real model
        response and which provider actually produced it.

    Raises:
        ProviderError: When no available, image-capable provider could
            complete the request (includes every candidate's own error
            for diagnosis).
    """
    candidates = _candidate_providers()
    if not candidates:
        raise ProviderError(
            "No available provider can analyze images right now "
            "(every configured provider is either unavailable or, like acpx, "
            "structurally unable to carry real image data)."
        )

    errors: list[str] = []
    for provider in candidates:
        try:
            msg_dict = build_vision_message(provider.name, image_base64, prompt, media_type)
            messages = [Message(role=msg_dict["role"], content=msg_dict["content"])]
            response = provider.complete(messages, session_id=session_id, task_id=task_id)
        except ProviderError as exc:
            logger.warning("Vision analysis via %r failed: %s", provider.name, exc)
            errors.append(f"{provider.name}: {exc}")
            continue
        except Exception as exc:
            logger.warning("Vision analysis via %r failed unexpectedly: %s", provider.name, exc)
            errors.append(f"{provider.name}: {exc}")
            continue
        return response.content, provider.name

    raise ProviderError(
        "No available provider could analyze the image. Attempts: " + "; ".join(errors)
    )
