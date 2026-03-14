"""Input sanitization to prevent prompt injection.

:class:`InputSanitizer` is the first line of defence against adversarial
user input.  It truncates oversized payloads and detects common prompt-
injection patterns, logging a warning whenever a suspicious string is found.
The original content is still returned (with truncation applied) so that
callers can decide whether to abort, redact, or proceed with caution.

Example::

    from missy.security.sanitizer import sanitizer

    clean = sanitizer.sanitize(user_text)
    matches = sanitizer.check_for_injection(user_text)
    if matches:
        print(f"Warning: injection patterns detected: {matches}")
"""

from __future__ import annotations

import logging
import re

logger = logging.getLogger(__name__)

#: Maximum number of characters accepted from a single user input.
MAX_INPUT_LENGTH: int = 10_000


class InputSanitizer:
    """Sanitizes user input to prevent prompt injection attacks.

    Detection is heuristic: the :attr:`INJECTION_PATTERNS` list covers the
    most common prompt-injection formulations, but determined adversaries
    may craft inputs that evade pattern matching.  This layer should be
    combined with other defences (output validation, privilege separation,
    human-in-the-loop review for sensitive operations).

    Attributes:
        INJECTION_PATTERNS: A list of regular expression strings.  Each
            pattern is compiled case-insensitively.  A match against any
            pattern causes :meth:`check_for_injection` to include that
            pattern string in its return value.
    """

    INJECTION_PATTERNS: list[str] = [
        r"ignore\s+(all\s+)?previous\s+instructions?",
        r"disregard\s+(all\s+)?previous\s+instructions?",
        r"forget\s+(all\s+)?previous\s+instructions?",
        r"you\s+are\s+now\s+(a\s+)?different",
        r"pretend\s+you\s+are",
        r"act\s+as\s+(if\s+you\s+are\s+)?a\s+",
        r"system\s*:\s*",
        r"<\s*system\s*>",
        r"\[INST\]",
        r"###\s*(System|Instruction)",
        r"<\|im_start\|>",
        r"<\|system\|>",
        r"override\s+(your\s+)?(previous\s+)?instructions?",
    ]

    def __init__(self) -> None:
        self._patterns: list[re.Pattern[str]] = [
            re.compile(p, re.IGNORECASE) for p in self.INJECTION_PATTERNS
        ]

    def sanitize(self, text: str) -> str:
        """Sanitize *text* and return the cleaned version.

        Steps applied in order:

        1. Truncate to :data:`MAX_INPUT_LENGTH` characters.
        2. Run injection-pattern detection; log a warning if any match.

        The text is returned as-is (after truncation) rather than being
        modified, allowing callers to make their own policy decisions.

        Args:
            text: Raw user input to sanitize.

        Returns:
            The sanitized (possibly truncated) input string.
        """
        text = self.truncate(text)

        matched = self.check_for_injection(text)
        if matched:
            logger.warning(
                "Potential prompt injection detected. Matched patterns: %s",
                matched,
            )

        return text

    def check_for_injection(self, text: str) -> list[str]:
        """Return the list of injection patterns that match *text*.

        Args:
            text: The input text to scan.

        Returns:
            A list of pattern strings (from :attr:`INJECTION_PATTERNS`) that
            matched.  An empty list means the input appears clean.
        """
        matched: list[str] = []
        for pattern, original in zip(self._patterns, self.INJECTION_PATTERNS, strict=False):
            if pattern.search(text):
                matched.append(original)
        return matched

    def truncate(self, text: str, max_length: int = MAX_INPUT_LENGTH) -> str:
        """Truncate *text* to at most *max_length* characters.

        When truncation is applied a suffix is appended to make it visible
        that content was removed.

        Args:
            text: The string to truncate.
            max_length: Maximum allowed length.  Defaults to
                :data:`MAX_INPUT_LENGTH`.

        Returns:
            The (possibly truncated) string.
        """
        if len(text) <= max_length:
            return text

        logger.warning(
            "Input truncated from %d to %d characters.",
            len(text),
            max_length,
        )
        return text[:max_length] + " [truncated]"


#: Process-level singleton — import and use directly in most cases.
sanitizer: InputSanitizer = InputSanitizer()
