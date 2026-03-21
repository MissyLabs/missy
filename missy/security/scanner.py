"""Security scanner for Missy installations.

:class:`SecurityScanner` audits a Missy installation for security
misconfigurations, policy gaps, and known vulnerability patterns.  Each
check produces zero or more :class:`Finding` instances ranked by
:class:`Severity`.

Example::

    from missy.security.scanner import SecurityScanner

    scanner = SecurityScanner(config_path="~/.missy/config.yaml")
    result = scanner.scan_all()
    print(result.format_report(verbose=True))
    if result.has_critical:
        raise SystemExit(1)
"""

from __future__ import annotations

import json
import logging
import os
import stat
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from missy.config.settings import MissyConfig

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

# Dangerous commands that should never appear in the shell allowlist because
# they allow data destruction, privilege escalation, or policy bypass.
_DANGEROUS_COMMANDS: frozenset[str] = frozenset(
    {
        "rm",
        "dd",
        "mkfs",
        "chmod",
        "chown",
        "chattr",
        "shred",
        "kill",
        "killall",
        "pkill",
        "reboot",
        "shutdown",
        "poweroff",
        "halt",
        "fdisk",
        "parted",
        "wipefs",
        "cryptsetup",
        "insmod",
        "rmmod",
        "modprobe",
        "iptables",
        "nft",
        "ufw",
        "passwd",
        "useradd",
        "userdel",
        "usermod",
        "groupadd",
        "sudo",
        "su",
        "mount",
        "umount",
    }
)

# Shell interpreter commands — if allowed, they can execute arbitrary code and
# bypass every other policy control the operator has configured.
_INTERPRETER_COMMANDS: frozenset[str] = frozenset(
    {
        "bash",
        "sh",
        "dash",
        "zsh",
        "fish",
        "ksh",
        "csh",
        "tcsh",
        "python",
        "python3",
        "python2",
        "node",
        "nodejs",
        "ruby",
        "perl",
        "php",
        "lua",
        "tclsh",
        "expect",
        "awk",
        "sed",
        "xargs",
        "eval",
    }
)

# Sensitive directory prefixes that must not appear in filesystem write paths.
_SENSITIVE_WRITE_PREFIXES: tuple[str, ...] = (
    "/etc",
    "/usr",
    "/bin",
    "/sbin",
    "/lib",
    "/lib64",
    "/boot",
    "/sys",
    "/proc",
    "/dev",
    "/run",
    "/var/lib",
    "/var/log",
    "/root",
)

# Common environment variable names that carry API keys/secrets.
_SECRET_ENV_VARS: tuple[str, ...] = (
    "ANTHROPIC_API_KEY",
    "OPENAI_API_KEY",
    "OPENAI_KEY",
    "GROQ_API_KEY",
    "COHERE_API_KEY",
    "HUGGINGFACE_API_KEY",
    "REPLICATE_API_KEY",
    "GITHUB_TOKEN",
    "GITHUB_API_TOKEN",
    "AWS_ACCESS_KEY_ID",
    "AWS_SECRET_ACCESS_KEY",
    "GCP_API_KEY",
    "GOOGLE_API_KEY",
    "DISCORD_TOKEN",
    "DISCORD_BOT_TOKEN",
    "GITLAB_TOKEN",
    "STRIPE_API_KEY",
    "SLACK_TOKEN",
    "NPM_TOKEN",
    "PYPI_TOKEN",
    "SENDGRID_API_KEY",
    "TWILIO_AUTH_TOKEN",
    "DATABRICKS_TOKEN",
    "DIGITALOCEAN_TOKEN",
)


class Severity(StrEnum):
    """Severity level for a security finding."""

    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


# Ordered from most to least severe for display sorting.
_SEVERITY_ORDER: dict[Severity, int] = {
    Severity.CRITICAL: 0,
    Severity.HIGH: 1,
    Severity.MEDIUM: 2,
    Severity.LOW: 3,
    Severity.INFO: 4,
}

# Rich markup colours per severity.
_SEVERITY_COLOUR: dict[Severity, str] = {
    Severity.CRITICAL: "bold red",
    Severity.HIGH: "red",
    Severity.MEDIUM: "yellow",
    Severity.LOW: "cyan",
    Severity.INFO: "dim",
}


@dataclass
class Finding:
    """A single security finding produced by :class:`SecurityScanner`.

    Attributes:
        id: Unique identifier, e.g. ``"SEC-010"``.
        title: Short human-readable title.
        description: Full description of the issue and its impact.
        severity: Severity classification.
        category: Logical grouping (``"config"``, ``"mcp"``, ``"tools"``,
            ``"network"``, ``"filesystem"``, ``"secrets"``, ``"identity"``).
        recommendation: Concrete fix instruction for the operator.
        details: Optional structured data for programmatic consumption.
    """

    id: str
    title: str
    description: str
    severity: Severity
    category: str
    recommendation: str
    details: dict = field(default_factory=dict)


@dataclass
class ScanResult:
    """The complete output of a :meth:`SecurityScanner.scan_all` run.

    Attributes:
        findings: All findings, sorted by severity then ID.
        scanned_at: ISO-8601 timestamp (UTC) of when the scan ran.
        scan_duration_ms: Wall-clock time for the scan in milliseconds.
        summary: Dict mapping severity label to count of findings at that level.
    """

    findings: list[Finding]
    scanned_at: str
    scan_duration_ms: float
    summary: dict[str, int]

    @property
    def critical_count(self) -> int:
        """Number of CRITICAL findings."""
        return sum(1 for f in self.findings if f.severity == Severity.CRITICAL)

    @property
    def has_critical(self) -> bool:
        """True if at least one CRITICAL finding exists."""
        return self.critical_count > 0

    def format_report(self, verbose: bool = False) -> str:
        """Render findings as a plain-text report.

        Args:
            verbose: When *True*, include the full description for each finding.
                     When *False*, only show the title and recommendation.

        Returns:
            A multi-line string suitable for printing to a terminal.
        """
        lines: list[str] = []

        # Header
        lines.append("Security Scan Results")
        lines.append("=" * 21)
        lines.append(f"Scanned at: {self.scanned_at} ({self.scan_duration_ms:.0f}ms)")
        lines.append("")

        # Severity bar
        bar_parts = []
        for sev in Severity:
            count = self.summary.get(sev.value, 0)
            bar_parts.append(f"{sev.value.upper()} ({count})")
        lines.append(" | ".join(bar_parts))

        if not self.findings:
            lines.append("\nNo findings — installation looks secure.")
            return "\n".join(lines)

        lines.append("")

        # Findings
        for finding in self.findings:
            sev_label = f"[{finding.severity.value.upper()}]"
            lines.append(f"{sev_label} {finding.id}: {finding.title}")
            if verbose and finding.description:
                for desc_line in finding.description.splitlines():
                    lines.append(f"  {desc_line}")
            lines.append(f"  Recommendation: {finding.recommendation}")
            if verbose and finding.details:
                for key, value in finding.details.items():
                    lines.append(f"  {key}: {value}")
            lines.append("")

        return "\n".join(lines)

    def to_json(self) -> dict:
        """Return a JSON-serialisable dict of the scan result."""
        return {
            "scanned_at": self.scanned_at,
            "scan_duration_ms": self.scan_duration_ms,
            "summary": self.summary,
            "findings": [
                {
                    "id": f.id,
                    "title": f.title,
                    "description": f.description,
                    "severity": f.severity.value,
                    "category": f.category,
                    "recommendation": f.recommendation,
                    "details": f.details,
                }
                for f in self.findings
            ],
        }


# ---------------------------------------------------------------------------
# Scanner
# ---------------------------------------------------------------------------


class SecurityScanner:
    """Scans a Missy installation for security misconfigurations and risk.

    The scanner operates on a loaded :class:`~missy.config.settings.MissyConfig`
    object (or none if the config does not exist / cannot be loaded) plus the
    Missy data directory on disk.

    Args:
        config: Pre-loaded config object.  When *None* the scanner loads
            the file at *config_path* itself.
        config_path: Path to ``config.yaml`` (default ``~/.missy/config.yaml``).
        missy_dir: Path to the Missy data directory (default ``~/.missy``).
            Override in tests to point at a temporary directory.
    """

    def __init__(
        self,
        config: MissyConfig | None = None,
        config_path: str = "~/.missy/config.yaml",
        missy_dir: str = "~/.missy",
    ) -> None:
        self.config = config
        self.config_path = Path(config_path).expanduser()
        self.missy_dir = Path(missy_dir).expanduser()
        self._findings: list[Finding] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def scan_all(self) -> ScanResult:
        """Run all security checks and return the consolidated result.

        Returns:
            A :class:`ScanResult` with all findings sorted by severity.
        """
        start = time.monotonic()
        self._findings = []

        # Lazily load config if not provided
        if self.config is None:
            self._try_load_config()

        self._check_config_security()
        self._check_network_policy()
        self._check_filesystem_policy()
        self._check_shell_policy()
        self._check_mcp_security()
        self._check_tool_permissions()
        self._check_secrets_and_vault()
        self._check_identity()
        self._check_file_permissions()
        self._check_known_vulnerabilities()

        # Sort by severity order then finding ID so output is deterministic.
        self._findings.sort(key=lambda f: (_SEVERITY_ORDER[f.severity], f.id))

        elapsed_ms = (time.monotonic() - start) * 1000

        return ScanResult(
            findings=list(self._findings),
            scanned_at=datetime.now(UTC).isoformat(),
            scan_duration_ms=elapsed_ms,
            summary=self._summarize(),
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _add(self, finding: Finding) -> None:
        """Append a finding to the internal collection."""
        self._findings.append(finding)

    def _try_load_config(self) -> None:
        """Attempt to load the config file; leaves self.config as None on failure."""
        if not self.config_path.exists():
            self._add(
                Finding(
                    id="SEC-000",
                    title="Config file not found",
                    description=(
                        f"No configuration file was found at {self.config_path}. "
                        "Many security checks cannot be performed without a config."
                    ),
                    severity=Severity.HIGH,
                    category="config",
                    recommendation=f"Run `missy init` to create {self.config_path}.",
                    details={"path": str(self.config_path)},
                )
            )
            return
        try:
            from missy.config.settings import load_config

            self.config = load_config(str(self.config_path))
        except Exception as exc:
            self._add(
                Finding(
                    id="SEC-000",
                    title="Config file could not be loaded",
                    description=(
                        f"The configuration at {self.config_path} could not be "
                        f"parsed: {exc}. Security checks requiring the config are skipped."
                    ),
                    severity=Severity.HIGH,
                    category="config",
                    recommendation="Fix the YAML syntax error and re-run the scan.",
                    details={"path": str(self.config_path), "error": str(exc)},
                )
            )

    # ------------------------------------------------------------------
    # Check: config security (SEC-001 .. SEC-004)
    # ------------------------------------------------------------------

    def _check_config_security(self) -> None:
        """Check config file for basic security issues."""
        # SEC-001: File permissions — only when the file is on disk.
        if self.config_path.exists():
            try:
                st = self.config_path.stat()
                mode = stat.S_IMODE(st.st_mode)
                if mode & (stat.S_IWGRP | stat.S_IWOTH):
                    self._add(
                        Finding(
                            id="SEC-001",
                            title="Config file is group/world-writable",
                            description=(
                                f"{self.config_path} has mode {oct(mode)}, which allows "
                                "other users to modify your configuration.  An attacker "
                                "could enable the shell, widen network access, or inject "
                                "a malicious provider endpoint."
                            ),
                            severity=Severity.HIGH,
                            category="config",
                            recommendation=f"Run: chmod 600 {self.config_path}",
                            details={"mode": oct(mode), "path": str(self.config_path)},
                        )
                    )
                if mode & stat.S_IROTH:
                    self._add(
                        Finding(
                            id="SEC-001b",
                            title="Config file is world-readable",
                            description=(
                                f"{self.config_path} has mode {oct(mode)}.  Any local "
                                "user can read your provider endpoints and policy "
                                "configuration — including any API key references."
                            ),
                            severity=Severity.MEDIUM,
                            category="config",
                            recommendation=f"Run: chmod 600 {self.config_path}",
                            details={"mode": oct(mode), "path": str(self.config_path)},
                        )
                    )
            except OSError as exc:
                logger.debug("SEC-001: cannot stat config file: %s", exc)

        if self.config is None:
            return

        # SEC-002: Plaintext API keys in config
        for provider_name, provider_cfg in self.config.providers.items():
            raw_key = provider_cfg.api_key or ""
            # Only flag real keys — skip vault/env references and short tokens.
            if raw_key and not raw_key.startswith(("vault://", "$")) and len(raw_key.strip()) > 10:
                self._add(
                    Finding(
                        id="SEC-002",
                        title=f"Plaintext API key in config for provider '{provider_name}'",
                        description=(
                            f"Provider '{provider_name}' has an api_key written "
                            "directly in config.yaml.  If the config file is "
                            "committed to version control or world-readable, the "
                            "key will be exposed."
                        ),
                        severity=Severity.HIGH,
                        category="secrets",
                        recommendation=(
                            "Store the key with `missy vault set KEY value`, then "
                            f"set providers.{provider_name}.api_key: vault://KEY "
                            "in config.yaml."
                        ),
                        details={"provider": provider_name},
                    )
                )

        # SEC-003: Config version outdated
        if self.config.config_version < 2:
            self._add(
                Finding(
                    id="SEC-003",
                    title="Config schema version is outdated",
                    description=(
                        f"config_version is {self.config.config_version} (current is 2). "
                        "The config may be missing security controls introduced in newer "
                        "versions (e.g. preset-based network policies)."
                    ),
                    severity=Severity.LOW,
                    category="config",
                    recommendation=(
                        "Run `missy run` once — the config migrator auto-upgrades "
                        "your file and backs up the original."
                    ),
                    details={"config_version": self.config.config_version},
                )
            )

        # SEC-004: default_deny — surfaced here as a config finding too, the
        # primary check is in _check_network_policy (SEC-010)

    # ------------------------------------------------------------------
    # Check: network policy (SEC-010 .. SEC-013)
    # ------------------------------------------------------------------

    def _check_network_policy(self) -> None:
        """Check network policy configuration."""
        if self.config is None:
            return

        net = self.config.network

        # SEC-010: default_deny disabled
        if not net.default_deny:
            self._add(
                Finding(
                    id="SEC-010",
                    title="Network default_deny is disabled",
                    description=(
                        "network.default_deny is false, meaning the agent can reach "
                        "any internet host without restriction.  This defeats network "
                        "policy enforcement entirely and allows data exfiltration, "
                        "SSRF, and C2 callbacks."
                    ),
                    severity=Severity.CRITICAL,
                    category="network",
                    recommendation=(
                        "Set `network.default_deny: true` in config.yaml, then "
                        "add only the hosts you need (or use presets: [anthropic])."
                    ),
                    details={"default_deny": False},
                )
            )

        # SEC-011: Overly broad CIDR allowances
        _OPEN_CIDRS = {"0.0.0.0/0", "::/0", "0.0.0.0/1", "128.0.0.0/1"}
        broad_cidrs = [c for c in net.allowed_cidrs if c in _OPEN_CIDRS]
        if broad_cidrs:
            self._add(
                Finding(
                    id="SEC-011",
                    title="Network policy contains overly broad CIDR allowances",
                    description=(
                        "One or more allowed_cidrs entries permit access to the "
                        "entire internet, negating the purpose of network policy. "
                        f"Offending CIDRs: {broad_cidrs}."
                    ),
                    severity=Severity.CRITICAL,
                    category="network",
                    recommendation=(
                        "Remove broad CIDRs from network.allowed_cidrs. "
                        "Use named presets (e.g. `presets: [anthropic, github]`) "
                        "or specific host:port entries instead."
                    ),
                    details={"cidrs": broad_cidrs},
                )
            )

        # SEC-012: No REST policies defined (no L7 controls)
        if net.default_deny is False:
            pass  # Already flagged as CRITICAL above
        elif not net.rest_policies:
            self._add(
                Finding(
                    id="SEC-012",
                    title="No REST (L7) policies are configured",
                    description=(
                        "network.rest_policies is empty.  Without REST policies, "
                        "the agent can use any HTTP method (DELETE, PUT, POST) on "
                        "any path of an allowed host.  L7 controls limit blast "
                        "radius when a host is allowlisted."
                    ),
                    severity=Severity.LOW,
                    category="network",
                    recommendation=(
                        "Add REST policies, e.g. allow only GET requests: "
                        "`rest_policies: [{host: api.github.com, method: GET, "
                        "path: /**, action: allow}]`."
                    ),
                )
            )

        # SEC-013: Wildcard domains too broad
        _VERY_BROAD = {".com", ".net", ".org", ".io", ".co"}
        broad_domains = [
            d
            for d in net.allowed_domains
            if any(d == suffix or d.endswith(suffix) for suffix in _VERY_BROAD)
            and d.count(".") <= 1
        ]
        if broad_domains:
            self._add(
                Finding(
                    id="SEC-013",
                    title="Network policy contains very broad domain allowances",
                    description=(
                        "One or more allowed_domains entries match almost any "
                        f"public hostname: {broad_domains}. This allows the agent "
                        "to reach arbitrary services on the internet."
                    ),
                    severity=Severity.HIGH,
                    category="network",
                    recommendation=(
                        "Replace broad domain patterns with specific hostnames, "
                        "e.g. `api.anthropic.com` instead of `.com`."
                    ),
                    details={"domains": broad_domains},
                )
            )

    # ------------------------------------------------------------------
    # Check: filesystem policy (SEC-020 .. SEC-022)
    # ------------------------------------------------------------------

    def _check_filesystem_policy(self) -> None:
        """Check filesystem policy configuration."""
        if self.config is None:
            return

        fs = self.config.filesystem

        # SEC-020: No read path restrictions
        if not fs.allowed_read_paths:
            self._add(
                Finding(
                    id="SEC-020",
                    title="No filesystem read-path restrictions configured",
                    description=(
                        "filesystem.allowed_read_paths is empty.  The agent has no "
                        "read restrictions and can access any file on the filesystem, "
                        "including SSH keys, browser cookies, password managers, and "
                        "the Missy vault."
                    ),
                    severity=Severity.MEDIUM,
                    category="filesystem",
                    recommendation=(
                        "Set filesystem.allowed_read_paths to specific directories, "
                        "e.g. `- ~/workspace` and `- /tmp`."
                    ),
                )
            )

        # SEC-021: Write access to sensitive directories
        sensitive_writes = [
            p
            for p in fs.allowed_write_paths
            if any(
                Path(p).expanduser().as_posix().startswith(prefix)
                for prefix in _SENSITIVE_WRITE_PREFIXES
            )
            or str(p) in ("/", "~", str(Path.home()))
        ]
        if sensitive_writes:
            self._add(
                Finding(
                    id="SEC-021",
                    title="Write access granted to sensitive system directories",
                    description=(
                        "filesystem.allowed_write_paths includes paths that should "
                        "never be writable by an AI agent: "
                        f"{sensitive_writes}. This could allow system file "
                        "tampering, privilege escalation, or data destruction."
                    ),
                    severity=Severity.CRITICAL,
                    category="filesystem",
                    recommendation=(
                        "Remove sensitive paths from allowed_write_paths.  Limit "
                        "write access to `~/workspace` and specific output directories."
                    ),
                    details={"write_paths": sensitive_writes},
                )
            )

        # SEC-022: Home directory fully writable (too broad but not critical)
        home = str(Path.home())
        broad_writes = [p for p in fs.allowed_write_paths if str(Path(p).expanduser()) == home]
        if broad_writes:
            self._add(
                Finding(
                    id="SEC-022",
                    title="Entire home directory is allowed for write access",
                    description=(
                        "filesystem.allowed_write_paths includes the home directory "
                        "itself.  The agent can overwrite shell profiles, SSH "
                        "authorised_keys, config files, and other sensitive data."
                    ),
                    severity=Severity.HIGH,
                    category="filesystem",
                    recommendation=(
                        "Replace the home directory entry with a specific "
                        "subdirectory such as `~/workspace`."
                    ),
                    details={"write_paths": broad_writes},
                )
            )

    # ------------------------------------------------------------------
    # Check: shell policy (SEC-030 .. SEC-032)
    # ------------------------------------------------------------------

    def _check_shell_policy(self) -> None:
        """Check shell policy configuration."""
        if self.config is None:
            return

        shell = self.config.shell
        if not shell.enabled:
            return  # Shell disabled — no shell findings

        allowed = set(shell.allowed_commands)

        # SEC-030: Shell enabled with no allowlist (anything goes)
        if not allowed:
            self._add(
                Finding(
                    id="SEC-030",
                    title="Shell is enabled with no command allowlist",
                    description=(
                        "shell.enabled is true but shell.allowed_commands is empty. "
                        "The agent can execute any command as the current user, "
                        "bypassing all policy controls."
                    ),
                    severity=Severity.CRITICAL,
                    category="shell",
                    recommendation=(
                        "Add an explicit allowlist of safe commands, e.g. "
                        "`allowed_commands: [git, ls, cat, grep]`.  Or set "
                        "`shell.enabled: false` if shell access is not needed."
                    ),
                )
            )
            return  # further checks are moot

        # SEC-031: Dangerous commands in allowlist
        dangerous = allowed & _DANGEROUS_COMMANDS
        if dangerous:
            self._add(
                Finding(
                    id="SEC-031",
                    title="Shell allowlist contains dangerous commands",
                    description=(
                        "The following commands in shell.allowed_commands can cause "
                        f"data loss, system damage, or privilege escalation: "
                        f"{sorted(dangerous)}."
                    ),
                    severity=Severity.HIGH,
                    category="shell",
                    recommendation=(
                        "Remove these commands from allowed_commands: "
                        + ", ".join(sorted(dangerous))
                        + ".  Run `missy security scan` after editing."
                    ),
                    details={"dangerous_commands": sorted(dangerous)},
                )
            )

        # SEC-032: Interpreter commands allowed (policy bypass risk)
        interpreters = allowed & _INTERPRETER_COMMANDS
        if interpreters:
            self._add(
                Finding(
                    id="SEC-032",
                    title="Shell allowlist contains interpreter/scripting commands",
                    description=(
                        "Interpreter commands in allowed_commands "
                        f"({sorted(interpreters)}) allow the agent to execute "
                        "arbitrary code, bypassing filesystem, network, and shell "
                        "policy controls entirely."
                    ),
                    severity=Severity.HIGH,
                    category="shell",
                    recommendation=(
                        "Remove interpreter commands from allowed_commands: "
                        + ", ".join(sorted(interpreters))
                        + ".  Use dedicated tools instead of raw interpreters."
                    ),
                    details={"interpreter_commands": sorted(interpreters)},
                )
            )

    # ------------------------------------------------------------------
    # Check: MCP security (SEC-040 .. SEC-043)
    # ------------------------------------------------------------------

    def _check_mcp_security(self) -> None:
        """Check MCP server configuration."""
        mcp_path = self.missy_dir / "mcp.json"
        if not mcp_path.exists():
            return  # No MCP configured

        # SEC-040: MCP config file permissions
        try:
            st = mcp_path.stat()
            mode = stat.S_IMODE(st.st_mode)
            if mode & (stat.S_IRGRP | stat.S_IROTH | stat.S_IWGRP | stat.S_IWOTH):
                self._add(
                    Finding(
                        id="SEC-040",
                        title="MCP config file has permissive permissions",
                        description=(
                            f"{mcp_path} has mode {oct(mode)}.  The file may "
                            "contain server commands, URLs, or credentials that "
                            "other local users could read or modify."
                        ),
                        severity=Severity.HIGH,
                        category="mcp",
                        recommendation=f"Run: chmod 600 {mcp_path}",
                        details={"mode": oct(mode), "path": str(mcp_path)},
                    )
                )
        except OSError as exc:
            logger.debug("SEC-040: cannot stat mcp.json: %s", exc)

        # Parse the MCP config
        try:
            servers: list[dict] = json.loads(mcp_path.read_text(encoding="utf-8"))
        except Exception as exc:
            logger.debug("SEC-04x: cannot parse mcp.json: %s", exc)
            return

        if not isinstance(servers, list):
            return

        _SHELL_INTERPRETERS_RE_TOKENS = {
            "bash",
            "sh",
            "python",
            "python3",
            "node",
            "ruby",
            "perl",
            "npx",
            "uvx",
            "pipx",
        }

        for server in servers:
            if not isinstance(server, dict):
                continue
            name = server.get("name", "<unknown>")
            has_digest = bool(server.get("digest"))
            command = server.get("command", "") or ""

            # SEC-041: No digest pin
            if not has_digest:
                self._add(
                    Finding(
                        id="SEC-041",
                        title=f"MCP server '{name}' has no digest pin",
                        description=(
                            f"MCP server '{name}' has no pinned tool-manifest "
                            "digest.  If the server binary or package is updated "
                            "or replaced by an attacker, Missy will load the new "
                            "(potentially malicious) tool definitions without "
                            "warning."
                        ),
                        severity=Severity.HIGH,
                        category="mcp",
                        recommendation=f"Run: missy mcp pin {name}",
                        details={"server": name},
                    )
                )

            # SEC-042: Shell interpreter in command
            cmd_tokens = set(command.split())
            interpreter_tokens = cmd_tokens & _SHELL_INTERPRETERS_RE_TOKENS
            if interpreter_tokens:
                self._add(
                    Finding(
                        id="SEC-042",
                        title=f"MCP server '{name}' command uses a shell interpreter",
                        description=(
                            f"MCP server '{name}' is launched via "
                            f"{sorted(interpreter_tokens)}.  If the MCP server "
                            "package is compromised, arbitrary code runs with full "
                            "user permissions.  Consider using pinned binaries "
                            "instead of package runners."
                        ),
                        severity=Severity.MEDIUM,
                        category="mcp",
                        recommendation=(
                            "Pin the MCP server digest (`missy mcp pin "
                            + name
                            + "`) and prefer direct binary invocations over npx/uvx/pipx."
                        ),
                        details={
                            "server": name,
                            "command": command,
                            "interpreters": sorted(interpreter_tokens),
                        },
                    )
                )

    # ------------------------------------------------------------------
    # Check: tool permissions (SEC-050 .. SEC-051)
    # ------------------------------------------------------------------

    def _check_tool_permissions(self) -> None:
        """Check registered tool permissions for excessive or absent controls."""
        try:
            from missy.tools.registry import get_tool_registry

            registry = get_tool_registry()
        except RuntimeError:
            return  # Tool registry not initialised; skip
        except Exception:
            return

        for tool_name in registry.list_tools():
            tool = registry.get(tool_name)
            if tool is None:
                continue
            perms = getattr(tool, "permissions", None)
            if perms is None:
                continue

            # SEC-050: Excessive permissions (network + filesystem + shell combined)
            combined_count = sum(
                [
                    bool(perms.network),
                    bool(perms.filesystem_read),
                    bool(perms.filesystem_write),
                    bool(perms.shell),
                ]
            )
            if combined_count >= 3:
                self._add(
                    Finding(
                        id="SEC-050",
                        title=f"Tool '{tool_name}' requests excessive permissions",
                        description=(
                            f"Tool '{tool_name}' requests {combined_count} permission "
                            "categories simultaneously (network, filesystem, shell). "
                            "A compromised or misbehaving tool with this many "
                            "permissions has a very large blast radius."
                        ),
                        severity=Severity.MEDIUM,
                        category="tools",
                        recommendation=(
                            f"Review whether tool '{tool_name}' truly needs all "
                            "requested permissions. Split into narrower tools where possible."
                        ),
                        details={
                            "tool": tool_name,
                            "network": perms.network,
                            "filesystem_read": perms.filesystem_read,
                            "filesystem_write": perms.filesystem_write,
                            "shell": perms.shell,
                        },
                    )
                )

    # ------------------------------------------------------------------
    # Check: secrets & vault (SEC-060 .. SEC-063)
    # ------------------------------------------------------------------

    def _check_secrets_and_vault(self) -> None:
        """Check secrets management configuration."""
        if self.config is None:
            return

        has_providers = bool(self.config.providers)
        vault_enabled = self.config.vault.enabled

        # SEC-060: Vault not enabled but API keys are configured
        if has_providers and not vault_enabled:
            # Only flag this if at least one provider has an API key not from vault
            has_plain_key = any(
                (p.api_key and not (p.api_key.startswith(("vault://", "$"))))
                or any(not k.startswith(("vault://", "$")) for k in (p.api_keys or []))
                for p in self.config.providers.values()
                if p.api_key or p.api_keys
            )
            if has_plain_key:
                self._add(
                    Finding(
                        id="SEC-060",
                        title="Vault is disabled but API keys are stored in plaintext",
                        description=(
                            "vault.enabled is false and at least one provider has "
                            "an API key stored directly in config.yaml.  The vault "
                            "provides ChaCha20-Poly1305 encryption for secrets at rest."
                        ),
                        severity=Severity.MEDIUM,
                        category="secrets",
                        recommendation=(
                            "Enable vault (`vault.enabled: true`), then migrate keys "
                            "with `missy vault set KEY value` and reference them as "
                            "`api_key: vault://KEY`."
                        ),
                    )
                )

        # SEC-061: Vault key file permissions too open
        vault_dir = Path(self.config.vault.vault_dir).expanduser()
        vault_key_path = vault_dir / "vault.key"
        if vault_key_path.exists():
            try:
                st = vault_key_path.stat()
                mode = stat.S_IMODE(st.st_mode)
                if mode & 0o077:
                    self._add(
                        Finding(
                            id="SEC-061",
                            title="Vault key file has permissive permissions",
                            description=(
                                f"{vault_key_path} has mode {oct(mode)}.  The vault "
                                "encryption key is accessible to other users, which "
                                "allows them to decrypt all stored secrets."
                            ),
                            severity=Severity.CRITICAL,
                            category="secrets",
                            recommendation=f"Run: chmod 600 {vault_key_path}",
                            details={"mode": oct(mode), "path": str(vault_key_path)},
                        )
                    )
            except OSError as exc:
                logger.debug("SEC-061: cannot stat vault key: %s", exc)

        # SEC-062: Common secret env vars are set
        set_secret_vars = [v for v in _SECRET_ENV_VARS if os.environ.get(v)]
        if set_secret_vars:
            self._add(
                Finding(
                    id="SEC-062",
                    title="API keys are present in environment variables",
                    description=(
                        "The following environment variables contain credentials: "
                        f"{set_secret_vars}.  Environment variables can be read by "
                        "any process running as the same user, child processes, and "
                        "may appear in process listings."
                    ),
                    severity=Severity.LOW,
                    category="secrets",
                    recommendation=(
                        "Consider migrating to the Missy vault: "
                        "`missy vault set KEY value`, then reference as "
                        "`api_key: vault://KEY`.  Use a process supervisor or "
                        "secret manager to avoid exposing keys in the shell environment."
                    ),
                    details={"env_vars": set_secret_vars},
                )
            )

    # ------------------------------------------------------------------
    # Check: identity (SEC-070 .. SEC-071)
    # ------------------------------------------------------------------

    def _check_identity(self) -> None:
        """Check agent identity key configuration."""
        identity_path = self.missy_dir / "identity.pem"

        # SEC-070: No identity key
        if not identity_path.exists():
            self._add(
                Finding(
                    id="SEC-070",
                    title="No agent identity key found",
                    description=(
                        f"No identity key was found at {identity_path}.  Audit "
                        "events cannot be cryptographically signed, making it "
                        "impossible to verify that audit logs have not been "
                        "tampered with."
                    ),
                    severity=Severity.MEDIUM,
                    category="identity",
                    recommendation=(
                        "Run `missy run` or `missy setup` once to auto-generate the identity key."
                    ),
                    details={"path": str(identity_path)},
                )
            )
            return

        # SEC-071: Identity key permissions too open
        try:
            st = identity_path.stat()
            mode = stat.S_IMODE(st.st_mode)
            if mode & 0o077:
                self._add(
                    Finding(
                        id="SEC-071",
                        title="Agent identity key has permissive permissions",
                        description=(
                            f"{identity_path} has mode {oct(mode)}.  The private "
                            "key is accessible to other users, allowing them to "
                            "forge signed audit events."
                        ),
                        severity=Severity.HIGH,
                        category="identity",
                        recommendation=f"Run: chmod 600 {identity_path}",
                        details={"mode": oct(mode), "path": str(identity_path)},
                    )
                )
        except OSError as exc:
            logger.debug("SEC-071: cannot stat identity.pem: %s", exc)

    # ------------------------------------------------------------------
    # Check: file permissions (SEC-080 .. SEC-082)
    # ------------------------------------------------------------------

    def _check_file_permissions(self) -> None:
        """Check Missy data directory and key file permissions."""
        if not self.missy_dir.exists():
            return

        # SEC-080: ~/.missy directory world-readable
        try:
            st = self.missy_dir.stat()
            mode = stat.S_IMODE(st.st_mode)
            if mode & (stat.S_IROTH | stat.S_IXOTH):
                self._add(
                    Finding(
                        id="SEC-080",
                        title="Missy data directory is world-accessible",
                        description=(
                            f"{self.missy_dir} has mode {oct(mode)}.  Other users "
                            "can list and access files in the Missy data directory, "
                            "including the audit log, memory database, and identity key."
                        ),
                        severity=Severity.HIGH,
                        category="filesystem",
                        recommendation=f"Run: chmod 700 {self.missy_dir}",
                        details={"mode": oct(mode), "path": str(self.missy_dir)},
                    )
                )
        except OSError as exc:
            logger.debug("SEC-080: cannot stat missy_dir: %s", exc)

        # SEC-081: audit.jsonl world-readable
        audit_path = self.missy_dir / "audit.jsonl"
        if audit_path.exists():
            try:
                st = audit_path.stat()
                mode = stat.S_IMODE(st.st_mode)
                if mode & stat.S_IROTH:
                    self._add(
                        Finding(
                            id="SEC-081",
                            title="Audit log is world-readable",
                            description=(
                                f"{audit_path} has mode {oct(mode)}.  The audit log "
                                "contains sensitive information: tool executions, "
                                "security events, session IDs, and command arguments."
                            ),
                            severity=Severity.MEDIUM,
                            category="filesystem",
                            recommendation=f"Run: chmod 600 {audit_path}",
                            details={"mode": oct(mode), "path": str(audit_path)},
                        )
                    )
            except OSError as exc:
                logger.debug("SEC-081: cannot stat audit.jsonl: %s", exc)

        # SEC-082: memory.db world-readable
        memory_path = self.missy_dir / "memory.db"
        if memory_path.exists():
            try:
                st = memory_path.stat()
                mode = stat.S_IMODE(st.st_mode)
                if mode & stat.S_IROTH:
                    self._add(
                        Finding(
                            id="SEC-082",
                            title="Memory database is world-readable",
                            description=(
                                f"{memory_path} has mode {oct(mode)}.  The memory "
                                "database stores conversation history, learnings, and "
                                "extracted facts that may contain sensitive user data."
                            ),
                            severity=Severity.MEDIUM,
                            category="filesystem",
                            recommendation=f"Run: chmod 600 {memory_path}",
                            details={"mode": oct(mode), "path": str(memory_path)},
                        )
                    )
            except OSError as exc:
                logger.debug("SEC-082: cannot stat memory.db: %s", exc)

    # ------------------------------------------------------------------
    # Check: known vulnerability patterns (SEC-090 .. SEC-093)
    # ------------------------------------------------------------------

    def _check_known_vulnerabilities(self) -> None:
        """Check for known configuration vulnerability patterns."""
        if self.config is None:
            return

        # SEC-090: Container sandbox disabled
        container_enabled = False
        if self.config.container is not None:
            container_enabled = getattr(self.config.container, "enabled", False)
        if not container_enabled:
            self._add(
                Finding(
                    id="SEC-090",
                    title="Container sandbox is disabled",
                    description=(
                        "container.enabled is false, so tool execution runs in the "
                        "host process with full user permissions.  A container "
                        "sandbox isolates tool execution with no network, limited "
                        "memory, and a restricted filesystem."
                    ),
                    severity=Severity.LOW,
                    category="config",
                    recommendation=(
                        "Enable Docker sandboxing: set `container.enabled: true` "
                        "and install Docker.  Run `missy sandbox status` to verify."
                    ),
                )
            )

        # SEC-091: max_spend not configured
        if self.config.max_spend_usd == 0.0:
            self._add(
                Finding(
                    id="SEC-091",
                    title="No per-session API spend limit configured",
                    description=(
                        "max_spend_usd is 0 (unlimited).  A runaway agent, prompt "
                        "injection attack, or infinite-loop bug could incur "
                        "unbounded API costs."
                    ),
                    severity=Severity.MEDIUM,
                    category="config",
                    recommendation=(
                        "Set `max_spend_usd: 1.00` (or a suitable limit) in "
                        "config.yaml to cap per-session API spending."
                    ),
                    details={"max_spend_usd": self.config.max_spend_usd},
                )
            )

        # SEC-092: No observability configured
        obs = self.config.observability
        if not obs.otel_enabled and obs.log_level in ("warning", "error", "critical"):
            self._add(
                Finding(
                    id="SEC-092",
                    title="Observability is minimal — security events may be missed",
                    description=(
                        "OpenTelemetry is disabled and log_level is "
                        f"'{obs.log_level}'.  Security-relevant events (policy "
                        "violations, prompt injection, trust score changes) are "
                        "written to the audit log, but without OTEL export they "
                        "cannot feed into alerting systems."
                    ),
                    severity=Severity.INFO,
                    category="config",
                    recommendation=(
                        "Consider enabling `observability.otel_enabled: true` and "
                        "pointing `otel_endpoint` at a local collector (Grafana, "
                        "Jaeger, etc.) for real-time security alerting."
                    ),
                )
            )

        # SEC-093: Shell enabled without approval gate
        if self.config.shell.enabled:
            self._add(
                Finding(
                    id="SEC-093",
                    title="Shell is enabled — consider requiring interactive approval",
                    description=(
                        "Shell execution is enabled.  Without an approval gate, the "
                        "agent can execute shell commands autonomously.  Interactive "
                        "approval lets the operator review each command before it runs."
                    ),
                    severity=Severity.INFO,
                    category="shell",
                    recommendation=(
                        "Run Missy in a terminal (TTY) to enable the interactive "
                        "approval TUI, which prompts on policy-sensitive operations."
                    ),
                )
            )

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def _summarize(self) -> dict[str, int]:
        counts: dict[str, int] = {sev.value: 0 for sev in Severity}
        for finding in self._findings:
            counts[finding.severity.value] += 1
        return counts
