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
        "aws_secret": r'(?i)(aws_secret_access_key)["\s:=]+[A-Za-z0-9/+=]{40}',
        "private_key": r"-----BEGIN (RSA |EC |DSA |OPENSSH )?PRIVATE KEY-----",
        "github_token": r"gh[ps]_[A-Za-z0-9]{36}",
        "github_oauth": r"gho_[A-Za-z0-9]{36}",
        "password": r'(?i)(password|passwd|pwd)["\s:=]+\S{8,}',
        "token": r'(?i)(token|secret)["\s:=]+[A-Za-z0-9_\-]{20,}',
        "stripe_key": r"[sr]k_(live|test)_[A-Za-z0-9]{24,}",
        "slack_token": r"xox[baprs]-[A-Za-z0-9\-]{10,}",
        "jwt": r"eyJ[A-Za-z0-9_\-]+\.eyJ[A-Za-z0-9_\-]+\.[A-Za-z0-9_\-]+",
        "anthropic_key": r"sk-ant-[A-Za-z0-9_\-]{20,}",
        "openai_key": r"sk-(?:proj-)?[A-Za-z0-9_\-]{20,}",
        "gcp_key": r"AIza[A-Za-z0-9_\-]{35}",
        "discord_token": r"[MN][A-Za-z0-9]{23,}\.[A-Za-z0-9_\-]{6}\.[A-Za-z0-9_\-]{27,}",
        "gitlab_token": r"glpat-[A-Za-z0-9_\-]{20,}",
        "npm_token": r"npm_[A-Za-z0-9]{36}",
        "pypi_token": r"pypi-[A-Za-z0-9_\-]{50,}",
        "sendgrid_key": r"SG\.[A-Za-z0-9_\-]{22}\.[A-Za-z0-9_\-]{43}",
        "db_connection_string": r"(?i)(?:postgres|mysql|mongodb|redis)://[^:]+:[^@]+@",
        "azure_key": r"(?i)(DefaultEndpointsProtocol|AccountKey)\s*=\s*[A-Za-z0-9+/=]{20,}",
        "twilio_key": r"SK[a-f0-9]{32}",
        "mailgun_key": r"key-[a-f0-9]{32}",
        "huggingface_token": r"hf_[A-Za-z0-9]{34,}",
        "databricks_token": r"dapi[a-f0-9]{32}",
        "digitalocean_token": r"dop_v1_[a-f0-9]{64}",
        "linear_key": r"lin_api_[A-Za-z0-9]{40,}",
        "supabase_key": r"sbp_[a-f0-9]{40}",
        "vercel_token": r"(?i)vercel[_\s]*(?:token|key)[\"'\s:=]+[A-Za-z0-9_\-]{24,}",
        "cloudflare_token": r"(?i)(?:cf|cloudflare)[_\s]*(?:api[_\s]*(?:token|key))[\"'\s:=]+[A-Za-z0-9_\-]{37,}",
        "shopify_token": r"shp(?:at|ca|pa|ss)_[a-fA-F0-9]{32,}",
        "google_oauth_secret": r'(?i)client[_\s]?secret["\s:=]+[A-Za-z0-9_\-]{24,}',
        "hashicorp_vault_token": r"(?:hvs|hvb|hvr)\.[A-Za-z0-9_\-]{24,}",
        "firebase_key": r"(?i)firebase[_\s]*(?:api[_\s]*key|secret)[\"'\s:=]+[A-Za-z0-9_\-]{20,}",
        # Session 21 additions
        "grafana_token": r"glc_[A-Za-z0-9_\-]{32,}",
        "confluent_key": r"(?i)confluent[_\s]*(?:api[_\s]*(?:key|secret))[\"'\s:=]+[A-Za-z0-9_\-]{16,}",
        "datadog_key": r"(?i)(?:dd|datadog)[_\s]*(?:api[_\s]*key|app[_\s]*key)[\"'\s:=]+[a-f0-9]{32,}",
        "newrelic_key": r"NRAK-[A-Z0-9]{27}",
        "pagerduty_key": r"(?i)pagerduty[_\s]*(?:api[_\s]*key|token)[\"'\s:=]+[A-Za-z0-9_\-]{20,}",
        "ssh_key_content": r"AAAA[BCD][A-Za-z0-9+/]{100,}",
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

        # Merge overlapping spans so that partial redaction cannot leak
        # fragments of a secret.  Sort by start position, then merge any
        # spans whose ranges overlap or abut.
        spans = sorted(
            ((f["match_start"], f["match_end"]) for f in findings),
            key=lambda s: s[0],
        )
        merged: list[tuple[int, int]] = [spans[0]]
        for start, end in spans[1:]:
            prev_start, prev_end = merged[-1]
            if start <= prev_end:
                # Overlapping or abutting — extend the previous span.
                merged[-1] = (prev_start, max(prev_end, end))
            else:
                merged.append((start, end))

        # Replace from right to left so earlier offsets stay valid.
        result = text
        for start, end in reversed(merged):
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
        return any(pattern.search(text) for pattern in self._patterns.values())


#: Process-level singleton — import and use directly in most cases.
secrets_detector: SecretsDetector = SecretsDetector()
