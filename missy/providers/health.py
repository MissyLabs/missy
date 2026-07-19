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


# Markers for a provider *content-policy / safety* refusal (as opposed to an
# operational failure). These are message fragments the active backends emit
# when the model declines on content grounds -- e.g. openai-codex/gpt-5.5's
# "This content was flagged for possible cybersecurity risk ... Trusted Access
# for Cyber program".
_CONTENT_POLICY_MARKERS = (
    "flagged for possible",
    "cybersecurity risk",
    "content policy",
    "content_policy",
    "trusted access",
    "usage policies",
    "safety system",
    "content management policy",
)


def is_content_policy_error(exc: BaseException) -> bool:
    """True if *exc* is a provider content-policy/safety refusal."""
    message = str(exc).lower()
    return any(marker in message for marker in _CONTENT_POLICY_MARKERS)


def user_facing_provider_error(exc: BaseException) -> str:
    """Return a clean, channel-safe message for a provider error.

    The raw provider exception must never reach an end user on a chat channel:
    it can name the underlying provider, embed a third party's marketing URL
    (openai-codex's cyber-program link), or expose internal error detail. The
    full error is still logged/audited for operators; this only governs what a
    Discord/voice user sees.
    """
    if is_content_policy_error(exc):
        return "I'm not able to help with that particular request."
    return "Sorry — I'm having trouble reaching my model right now. Please try again in a moment."
