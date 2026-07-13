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

If the only *enabled* provider fails or can't handle images, a
configured-but-disabled Ollama instance is tried as a last resort (see
:func:`_ollama_fallback_candidate`) -- as long as it's genuinely running,
regardless of its ``enabled`` flag for ordinary chat use.
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


def _ollama_fallback_candidate(already_present_names: set[str]) -> object | None:
    """Return a last-resort Ollama vision candidate, or ``None``.

    Constructed directly from ``config.providers["ollama"]`` even when
    Ollama is *disabled* for general chat use (``enabled: false``) and
    therefore never registered in the process-level
    :class:`~missy.providers.registry.ProviderRegistry` at all. An
    operator may deliberately keep Ollama out of normal provider
    rotation (e.g. it's slower, or a smaller local model) while still
    wanting it available as a vision-specific safety net for when the
    provider(s) actually enabled for chat can't -- or currently don't --
    handle images. This is a narrow, vision-only exception to the
    ``enabled`` gate; Ollama still never participates in ordinary chat
    completions unless the operator enables it there too.

    Only returned when:

    - Ollama isn't already one of the (enabled) candidates, and
    - an ``ollama`` entry actually exists in config, and
    - Ollama is genuinely reachable right now (skips a guaranteed-
      pointless attempt against an Ollama that was never set up or
      isn't currently running).

    Note this does not verify the configured Ollama *model* is itself
    vision-capable (e.g. a text-only model like ``qwen3.5:9b`` would
    still be attempted) -- that's a real, separate operator
    responsibility this function has no reliable way to check up front;
    a bad model choice surfaces as an unhelpful analysis rather than an
    error, not as a crash.

    Args:
        already_present_names: Names of providers already selected as
            candidates, so this never duplicates an already-enabled
            Ollama.

    Returns:
        An :class:`~missy.providers.ollama_provider.OllamaProvider`
        instance, or ``None`` when no fallback is available/needed.
    """
    if "ollama" in already_present_names:
        return None
    try:
        from missy.config.settings import load_config
        from missy.providers.ollama_provider import OllamaProvider

        ollama_cfg = load_config().providers.get("ollama")
        if ollama_cfg is None:
            return None
        provider = OllamaProvider(ollama_cfg)
        if not provider.is_available():
            return None
        return provider
    except Exception:
        logger.debug("Ollama vision fallback unavailable", exc_info=True)
        return None


def _candidate_providers() -> list:
    """Return available providers to try, default/primary provider first.

    Appends a last-resort Ollama candidate (see
    :func:`_ollama_fallback_candidate`) after every normally-enabled
    candidate, so an operator whose only enabled provider can't -- or
    currently doesn't -- handle images still gets a real answer instead
    of an outright failure, as long as Ollama is configured and running.
    """
    from missy.providers.registry import get_registry

    registry = get_registry()
    available = [
        p for p in registry.get_available() if p.name not in _IMAGE_INCOMPATIBLE_PROVIDER_NAMES
    ]
    default_name = registry.get_default_name()
    if default_name:
        available.sort(key=lambda p: p.name != default_name)

    fallback = _ollama_fallback_candidate({p.name for p in available})
    if fallback is not None:
        available.append(fallback)

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
