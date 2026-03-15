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

import base64
import logging
import re
import unicodedata

logger = logging.getLogger(__name__)

#: Maximum number of characters accepted from a single user input.
MAX_INPUT_LENGTH: int = 10_000

#: Zero-width and invisible Unicode characters used to obfuscate keywords.
#: Stripping these before pattern matching defeats intra-word insertion attacks.
_ZERO_WIDTH_RE: re.Pattern[str] = re.compile(
    "["
    "\u200b"  # zero-width space
    "\u200c"  # zero-width non-joiner
    "\u200d"  # zero-width joiner
    "\u200e"  # left-to-right mark
    "\u200f"  # right-to-left mark
    "\u2060"  # word joiner
    "\u2061"  # function application
    "\u2062"  # invisible times
    "\u2063"  # invisible separator
    "\u2064"  # invisible plus
    "\ufeff"  # byte-order mark / zero-width no-break space
    "\ufe0f"  # variation selector-16
    "\ufe0e"  # variation selector-15
    "]"
)

#: Regex to find plausible base64 segments (at least 20 chars, valid charset,
#: optional padding).  Kept short to avoid pathological backtracking.
_BASE64_SEGMENT_RE: re.Pattern[str] = re.compile(
    r"[A-Za-z0-9+/]{20,}={0,2}"
)


def _strip_zero_width(text: str) -> str:
    """Remove zero-width and invisible Unicode characters from *text*."""
    return _ZERO_WIDTH_RE.sub("", text)


def _normalize_unicode(text: str) -> str:
    """Apply NFKC normalization to fold confusable Unicode to ASCII equivalents.

    NFKC (Compatibility Decomposition followed by Canonical Composition)
    converts characters like fullwidth Latin letters, circled letters, and
    many visually similar glyphs to their standard ASCII counterparts.
    """
    return unicodedata.normalize("NFKC", text)


def _decode_base64_segments(text: str) -> str | None:
    """Find and decode base64 segments embedded in *text*.

    Returns the decoded text of any segment that decodes to valid UTF-8,
    concatenated with spaces.  Returns ``None`` if nothing decodable is found.
    """
    decoded_parts: list[str] = []
    for match in _BASE64_SEGMENT_RE.finditer(text):
        segment = match.group(0)
        # Pad to multiple of 4 for decoding
        padded = segment + "=" * (-len(segment) % 4)
        try:
            raw = base64.b64decode(padded, validate=True)
            decoded = raw.decode("utf-8", errors="strict")
            # Only accept printable decoded text (not binary gibberish)
            if decoded.isprintable() or any(c in decoded for c in " \n\t"):
                decoded_parts.append(decoded)
        except Exception:
            continue
    return " ".join(decoded_parts) if decoded_parts else None


class InputSanitizer:
    """Sanitizes user input to prevent prompt injection attacks.

    Detection is heuristic: the :attr:`INJECTION_PATTERNS` list covers the
    most common prompt-injection formulations, but determined adversaries
    may craft inputs that evade pattern matching.  This layer should be
    combined with other defences (output validation, privilege separation,
    human-in-the-loop review for sensitive operations).

    Before pattern matching, the input is pre-processed to defeat common
    obfuscation techniques:

    - Zero-width Unicode characters are stripped so that keywords split by
      invisible characters (e.g., ``ig\\u200Dnore``) are reunited.
    - NFKC Unicode normalization folds fullwidth letters, circled letters,
      and other confusables to their ASCII equivalents.
    - Base64-encoded segments are decoded and scanned separately.

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
        # Roleplay / jailbreak patterns
        r"do\s+anything\s+now",
        r"developer\s+mode\s+(enabled|activated|on)",
        r"jailbreak(ed)?",
        r"bypass\s+(your\s+)?(safety|security|filter|restriction)",
        # Delimiter injection (attempt to close/reopen system prompt)
        r"</?(system|user|assistant)\s*>",
        r"\[/?INST\]",
        r"<\|im_end\|>",
        # --- Additional delimiter / template injection patterns ---
        r"<<\s*SYS\s*>>",                       # Llama 2 system prompt delimiters
        r"<\|endoftext\|>",                      # GPT end-of-text token
        r"<\|eot_id\|>",                         # Llama 3 end-of-turn token
        # --- Session/context reset attacks ---
        r"(new|start)\s+(a\s+)?(conversation|session|chat)\b",
        r"reset\s+(your\s+)?(context|memory|instructions?)",
        r"from\s+now\s+on\s+you\s+(are|will|must|should)",
        # --- Additional jailbreak / override patterns ---
        r"sudo\s+mode",
        r"respond\s+as\s+if\s+you\s+(are|were|have)",
        r"ignore\s+(your\s+)?(safety|security|content)\s+(guidelines?|rules?|policies?|filters?)",
        r"you\s+have\s+no\s+(restrictions?|rules?|limitations?|guidelines?)",
        r"(enable|activate|enter)\s+(god|admin|root|unrestricted)\s+mode",
        # --- Hidden instruction vectors ---
        r"<!--[\s\S]*?-->",                      # HTML comments hiding instructions
        # --- Multi-language injection keywords (OWASP LLM01) ---
        r"ignorar\s+(todas?\s+)?(las?\s+)?instrucciones?\s+(anteriores?|previas?)",  # Spanish
        r"ignorer\s+(toutes?\s+)?(les?\s+)?instructions?\s+(pr[e\u00e9]c[e\u00e9]dentes?|ant[e\u00e9]rieures?)",  # French
        r"ignoriere\s+(alle\s+)?(vorherigen?\s+)?anweisungen",  # German
        r"(new|nuove?)\s+istruzioni\s*:",         # Italian "new instructions:"
        # --- Unclosed tag / comment injection ---
        r"<!--\s*(system|ignore|override|inject|instruction)",  # Unclosed HTML comment with keywords
        # --- Data URI / protocol injection ---
        r"data:\s*text/(html|javascript|plain)\s*[;,]",  # data: URI with text content
        # --- Invisible instruction hiding ---
        r"<\s*div\s+style\s*=\s*[\"'].*?display\s*:\s*none",  # Hidden div instructions
        r"\[comment\]:",                           # Markdown-style hidden comment
        # --- Additional model-specific delimiters ---
        r"<\|begin_of_text\|>",                    # Llama 3 begin token
        r"<\|start_header_id\|>",                  # Llama 3 header token
        r"<\|end_header_id\|>",                    # Llama 3 header end token
        r"<\|reserved_special_token",              # Any reserved special token
        # --- Chained instruction patterns ---
        r"new\s+instructions?\s*:",                # "new instructions:"
        r"updated?\s+instructions?\s*:",           # "updated instructions:"
        r"revised?\s+instructions?\s*:",           # "revised instructions:"
        r"real\s+instructions?\s*:",               # "real instructions:"
        # --- Portuguese / Russian injection keywords ---
        r"ignore\s+as\s+instru[çc][õo]es\s+anteriores",  # Portuguese
        r"игнорируй\s+(все\s+)?предыдущие\s+инструкции",  # Russian
        # --- Tool/function abuse patterns ---
        r"call\s+the\s+function\s+with\s+these\s+(exact\s+)?parameters?",
        r"execute\s+this\s+(tool|function|command)\s+exactly\s+as\s+(written|shown|given)",
        # --- Anthropic-specific delimiters ---
        r"<\|?claude\|?>",
        r"\[/?SYSTEM\]",
        r"Human:\s*$",       # Attempting to inject a new Human: turn
        r"Assistant:\s*$",   # Attempting to inject an Assistant: turn
        # --- Prompt leaking / exfiltration ---
        r"(show|reveal|print|output|display)\s+(your\s+)?(system\s+)?(prompt|instructions?)",
        r"what\s+(are|is)\s+your\s+(system\s+)?(prompt|instructions?)",
        # --- Japanese injection keywords ---
        r"以前の指示を(無視|忘れ)",  # Japanese: ignore/forget previous instructions
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

        The input is pre-processed before matching to defeat obfuscation:

        1. Zero-width Unicode characters are stripped.
        2. NFKC normalization is applied.
        3. Patterns are checked against the normalized text.
        4. Any base64-encoded segments are decoded and scanned separately.

        Args:
            text: The input text to scan.

        Returns:
            A list of pattern strings (from :attr:`INJECTION_PATTERNS`) that
            matched.  An empty list means the input appears clean.
        """
        # Pre-process: strip zero-width chars and normalize Unicode
        normalized = _normalize_unicode(_strip_zero_width(text))

        matched: list[str] = []
        for pattern, original in zip(self._patterns, self.INJECTION_PATTERNS, strict=False):
            if pattern.search(normalized):
                matched.append(original)

        # Scan decoded base64 segments for injection payloads
        decoded_b64 = _decode_base64_segments(text)
        if decoded_b64:
            b64_normalized = _normalize_unicode(_strip_zero_width(decoded_b64))
            for pattern, original in zip(
                self._patterns, self.INJECTION_PATTERNS, strict=False
            ):
                if original not in matched and pattern.search(b64_normalized):
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
