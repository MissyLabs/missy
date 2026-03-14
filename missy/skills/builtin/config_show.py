"""Built-in skill: display a sanitized Missy configuration summary.

Reads ``~/.missy/config.yaml`` and returns a human-readable summary with
any API key values redacted (first 8 characters visible, remainder replaced
with ``"..."``).  The raw YAML text is never surfaced verbatim so that
secrets are not accidentally echoed to the user.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from missy.skills.base import BaseSkill, SkillPermissions, SkillResult

_CONFIG_PATH = Path("~/.missy/config.yaml")

# Regex that matches a YAML string value on an api_key(s) line.
# Supports both single-quoted, double-quoted, and bare values.
_API_KEY_RE = re.compile(
    r"(?P<indent>\s*)(?P<key>api_key[s]?\s*:\s*)(?P<value>['\"]?)(?P<secret>\S+)(?P=value)",
    re.IGNORECASE,
)

_PLACEHOLDER_MIN_LEN = 9  # only redact values long enough to be real keys


def _redact_api_keys(text: str) -> str:
    """Replace API key values with a truncated, redacted form.

    Values shorter than :data:`_PLACEHOLDER_MIN_LEN` characters are left
    intact so that placeholder strings such as ``"null"`` or ``"sk-..."``
    are not mangled when they are clearly not real secrets.

    Args:
        text: Raw YAML text.

    Returns:
        YAML text with real-looking key values replaced by
        ``"<first-8-chars>..."``.
    """

    def _replace(match: re.Match) -> str:  # type: ignore[type-arg]
        secret: str = match.group("secret")
        if len(secret) < _PLACEHOLDER_MIN_LEN or secret.lower() in {
            "null",
            "none",
            "false",
            "true",
        }:
            return match.group(0)
        visible = secret[:8]
        return f"{match.group('indent')}{match.group('key')}{match.group('value')}{visible}...{match.group('value')}"

    return _API_KEY_RE.sub(_replace, text)


def _summarize_yaml(text: str) -> str:
    """Return a sanitized summary of the YAML config text.

    Top-level section headers are preserved; deeply nested lines are kept
    to give a useful structural overview.  API key values are redacted via
    :func:`_redact_api_keys`.

    Args:
        text: Raw YAML text.

    Returns:
        Sanitized text suitable for display.
    """
    redacted = _redact_api_keys(text)
    lines = redacted.splitlines()
    summary_lines: list[str] = []
    for line in lines:
        stripped = line.strip()
        # Skip blank lines and YAML comments to keep output concise.
        if not stripped or stripped.startswith("#"):
            continue
        summary_lines.append(line.rstrip())
    return "\n".join(summary_lines)


class ConfigShowSkill(BaseSkill):
    """Displays a sanitized summary of the current Missy configuration."""

    name = "config_show"
    description = "Display current Missy configuration summary."
    version = "1.0.0"
    permissions = SkillPermissions(filesystem_read=True)

    def execute(self, **kwargs: Any) -> SkillResult:
        """Read and return a redacted config summary.

        Returns:
            :class:`~missy.skills.base.SkillResult` whose ``output`` is the
            sanitized YAML text, or an error if the config file is missing or
            unreadable.
        """
        config_path = _CONFIG_PATH.expanduser()
        if not config_path.exists():
            return SkillResult(
                success=False,
                output="",
                error=f"Config file not found: {config_path}",
            )

        try:
            raw = config_path.read_text(encoding="utf-8")
        except OSError as exc:
            return SkillResult(
                success=False,
                output="",
                error=f"Failed to read config file: {exc}",
            )

        summary = _summarize_yaml(raw)
        header = f"# Config: {config_path}\n"
        return SkillResult(success=True, output=header + summary)
