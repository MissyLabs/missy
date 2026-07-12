"""Provider failure classification for automatic key rotation and fallback.

Every concrete provider wraps SDK/HTTP errors in a single
:class:`~missy.core.exceptions.ProviderError`, but the underlying cause
(expired credentials vs. rate limiting vs. a transient timeout) determines
what :class:`~missy.agent.runtime.AgentRuntime` should try next: an
authentication failure is worth retrying on a rotated API key before
falling over to a different provider entirely, while a rate limit or
timeout is not (rotating credentials cannot fix either).
"""

from __future__ import annotations

from enum import StrEnum


class ProviderFailureClass(StrEnum):
    """Coarse classification of a :class:`ProviderError`'s root cause."""

    AUTH = "auth"
    RATE_LIMIT = "rate_limit"
    TIMEOUT = "timeout"
    UNKNOWN = "unknown"


_AUTH_MARKERS = ("authentication failed", "unauthorized", "invalid api key", "invalid x-api-key")
_RATE_LIMIT_MARKERS = ("rate limit", "rate limited", "429", "too many requests")
_TIMEOUT_MARKERS = ("timed out", "timeout")


def classify_provider_error(exc: BaseException) -> ProviderFailureClass:
    """Classify *exc* (typically a :class:`ProviderError`) by root cause.

    Anthropic, OpenAI, Ollama, and Codex each catch their own SDK's
    *structured* exception types (e.g. ``anthropic.AuthenticationError``)
    and deliberately construct their ``ProviderError`` message to mention
    "authentication failed", "rate limit(ed)", or "timed out" for exactly
    those cases -- this function reuses that existing message vocabulary
    rather than requiring providers to expose a structured error code.

    ``acpx_provider.py`` is a structural exception to this: it wraps an
    external CLI subprocess, not an SDK with typed exceptions, so it has
    no equivalent structured signal to classify from -- its generic
    failure/nonzero-exit paths relay the wrapped CLI's own raw stderr
    text verbatim (e.g. ``f"acpx exited with code {code}: {stderr}"``)
    with no attempt to detect auth/rate-limit conditions first. Unless
    that raw external-CLI text happens to contain one of this module's
    marker words, an acpx auth or rate-limit failure classifies as
    ``UNKNOWN`` -- silently skipping the ``rotate_key()``/fallback
    response an equivalent Anthropic/OpenAI/Codex failure would trigger.
    Only its subprocess-timeout path is exempt, since that one is raised
    by Missy's own code (not relayed CLI text) and explicitly says
    "timed out".

    Args:
        exc: The exception raised by a provider call.

    Returns:
        The best-effort :class:`ProviderFailureClass` for *exc*.
    """
    message = str(exc).lower()
    if any(marker in message for marker in _AUTH_MARKERS):
        return ProviderFailureClass.AUTH
    if any(marker in message for marker in _RATE_LIMIT_MARKERS):
        return ProviderFailureClass.RATE_LIMIT
    if any(marker in message for marker in _TIMEOUT_MARKERS):
        return ProviderFailureClass.TIMEOUT
    return ProviderFailureClass.UNKNOWN
