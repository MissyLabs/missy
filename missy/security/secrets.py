"""Secrets detection to prevent accidental credential leakage.

:class:`SecretsDetector` scans text for common credential patterns (API
keys, private keys, passwords, tokens) and can either report matches or
redact them in place.  It is used by the CLI and agent runtime to prevent
secrets from appearing in logs, audit trails, or provider payloads.

Example::

    from missy.security.secrets import secrets_detector

    findings = secrets_detector.scan(some_text)
    if findings:
        safe_text = secrets_detector.redact(some_text)
"""

from __future__ import annotations

import re


class SecretsDetector:
    """Detect potential secrets and credentials in text.

    Each entry in :attr:`SECRET_PATTERNS` maps a human-readable secret type
    to a regular expression.  Patterns are compiled once at construction time.

    Attributes:
        SECRET_PATTERNS: Mapping of secret-type label to regex string.
    """

    SECRET_PATTERNS: dict[str, str] = {
        "api_key": r'(?i)(api[_-]?key|apikey)["\s:=]+[A-Za-z0-9_\-]{20,}',
        "aws_key": r"AKIA[0-9A-Z]{16}",
        "private_key": r"-----BEGIN (RSA |EC )?PRIVATE KEY-----",
        "github_token": r"ghp_[A-Za-z0-9]{36}",
        "password": r'(?i)(password|passwd|pwd)["\s:=]+\S{8,}',
        "token": r'(?i)(token|secret)["\s:=]+[A-Za-z0-9_\-]{20,}',
        "stripe_key": r"sk_(live|test)_[A-Za-z0-9]{24,}",
        "slack_token": r"xox[baprs]-[A-Za-z0-9\-]{10,}",
        "jwt": r"eyJ[A-Za-z0-9_\-]+\.eyJ[A-Za-z0-9_\-]+\.[A-Za-z0-9_\-]+",
    }

    def __init__(self) -> None:
        self._patterns: dict[str, re.Pattern[str]] = {
            k: re.compile(v) for k, v in self.SECRET_PATTERNS.items()
        }

    def scan(self, text: str) -> list[dict]:
        """Scan *text* for secret patterns and return all findings.

        Args:
            text: The text to inspect.

        Returns:
            A list of dicts, one per match, each containing:

            * ``"type"`` — the secret-type label (e.g. ``"api_key"``).
            * ``"match_start"`` — start index of the match in *text*.
            * ``"match_end"`` — end index (exclusive) of the match.

            Returns an empty list when no secrets are detected.
        """
        findings: list[dict] = []
        for secret_type, pattern in self._patterns.items():
            for match in pattern.finditer(text):
                findings.append(
                    {
                        "type": secret_type,
                        "match_start": match.start(),
                        "match_end": match.end(),
                    }
                )
        # Sort by position so callers get a predictable ordering.
        findings.sort(key=lambda f: f["match_start"])
        return findings

    def redact(self, text: str) -> str:
        """Return a copy of *text* with all detected secrets replaced by ``[REDACTED]``.

        Matches are replaced from right to left so that earlier match indices
        remain valid after each substitution.

        Args:
            text: The text to redact.

        Returns:
            The redacted string.  If no secrets are found the original text
            is returned unchanged.
        """
        findings = self.scan(text)
        if not findings:
            return text

        # Deduplicate overlapping spans and sort by position descending so
        # that right-to-left replacement keeps earlier offsets valid.
        findings_sorted = sorted(findings, key=lambda f: f["match_start"], reverse=True)

        result = text
        for finding in findings_sorted:
            start = finding["match_start"]
            end = finding["match_end"]
            result = result[:start] + "[REDACTED]" + result[end:]

        return result

    def has_secrets(self, text: str) -> bool:
        """Return ``True`` when any secret pattern matches *text*.

        This is a short-circuit scan — it stops at the first match.

        Args:
            text: The text to check.

        Returns:
            ``True`` if at least one secret pattern is found.
        """
        for pattern in self._patterns.values():
            if pattern.search(text):
                return True
        return False


#: Process-level singleton — import and use directly in most cases.
secrets_detector: SecretsDetector = SecretsDetector()
