"""Outbound response censoring: redact known credentials before delivery.

Usage::

    from missy.security.censor import censor_response
    safe_text = censor_response(response_text)
"""
from __future__ import annotations

from missy.security.secrets import SecretsDetector

_detector = SecretsDetector()


def censor_response(text: str) -> str:
    """Redact any detected secrets from *text* before it is sent to a channel.

    This is a last-resort filter independent of agent instructions.

    Args:
        text: The response text to scan.

    Returns:
        The text with any detected secrets replaced by [REDACTED].
    """
    if not text:
        return text
    return _detector.redact(text)
