"""Tests for missy.security.scanner.SecurityScanner.

Each check method is tested in isolation using synthetic
MissyConfig objects and temporary filesystem fixtures so that no
live Missy installation is required.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from missy.security.scanner import Finding, ScanResult, SecurityScanner, Severity

# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------


def _make_config(
    *,
    network_default_deny: bool = True,
    allowed_cidrs: list[str] | None = None,
    allowed_domains: list[str] | None = None,
    rest_policies: list[dict] | None = None,
    read_paths: list[str] | None = None,
    write_paths: list[str] | None = None,
    shell_enabled: bool = False,
    allowed_commands: list[str] | None = None,
    vault_enabled: bool = False,
    vault_dir: str = "~/.missy/secrets",
    providers: dict | None = None,
    max_spend_usd: float = 1.0,
    config_version: int = 2,
    container_enabled: bool = False,
    otel_enabled: bool = False,
    log_level: str = "warning",
):
    """Build a minimal MissyConfig-like object for testing without loading YAML."""
    from missy.config.settings import (
        FilesystemPolicy,
        MissyConfig,
        NetworkPolicy,
        ObservabilityConfig,
        PluginPolicy,
        ProviderConfig,
        ShellPolicy,
        VaultConfig,
        VisionConfig,
    )
    from missy.security.container import ContainerConfig

    net = NetworkPolicy(
        default_deny=network_default_deny,
        allowed_cidrs=allowed_cidrs or [],
        allowed_domains=allowed_domains or [],
        allowed_hosts=[],
        presets=[],
        rest_policies=rest_policies or [],
    )
    fs = FilesystemPolicy(
        allowed_read_paths=read_paths or [],
        allowed_write_paths=write_paths or [],
    )
    shell = ShellPolicy(enabled=shell_enabled, allowed_commands=allowed_commands or [])
    vault = VaultConfig(enabled=vault_enabled, vault_dir=vault_dir)

    raw_providers: dict[str, ProviderConfig] = {}
    if providers:
        for name, kw in providers.items():
            raw_providers[name] = ProviderConfig(
                name=name,
                model=kw.get("model", "claude-3-5-sonnet"),
                api_key=kw.get("api_key"),
                api_keys=kw.get("api_keys", []),
            )

    obs = ObservabilityConfig(
        otel_enabled=otel_enabled,
        log_level=log_level,
    )

    container = ContainerConfig(enabled=container_enabled)

    return MissyConfig(
        network=net,
        filesystem=fs,
        shell=shell,
        plugins=PluginPolicy(),
        providers=raw_providers,
        workspace_path="~/workspace",
        audit_log_path="~/.missy/audit.jsonl",
        vault=vault,
        observability=obs,
        container=container,
        vision=VisionConfig(),
        max_spend_usd=max_spend_usd,
        config_version=config_version,
    )


@pytest.fixture()
def missy_dir(tmp_path: Path) -> Path:
    """Return an empty temporary Missy data directory."""
    d = tmp_path / ".missy"
    d.mkdir(mode=0o700)
    return d


@pytest.fixture()
def secure_config():
    """A config that represents a well-hardened installation."""
    return _make_config(
        network_default_deny=True,
        allowed_cidrs=[],
        allowed_domains=["api.anthropic.com"],
        rest_policies=[
            {"host": "api.anthropic.com", "method": "POST", "path": "/**", "action": "allow"}
        ],
        read_paths=["~/workspace", "/tmp"],
        write_paths=["~/workspace"],
        shell_enabled=False,
        vault_enabled=True,
        providers={"anthropic": {"api_key": "vault://ANTHROPIC_KEY"}},
        max_spend_usd=2.0,
        config_version=2,
        container_enabled=False,
        otel_enabled=True,
        log_level="info",
    )


@pytest.fixture()
def insecure_config():
    """A config that represents worst-case security posture."""
    return _make_config(
        network_default_deny=False,
        allowed_cidrs=["0.0.0.0/0"],
        allowed_domains=[".com"],
        rest_policies=[],
        read_paths=[],
        write_paths=["/etc", "/usr"],
        shell_enabled=True,
        allowed_commands=[],
        vault_enabled=False,
        providers={"anthropic": {"api_key": "sk-ant-reallylong1234567890key"}},
        max_spend_usd=0.0,
        config_version=1,
        container_enabled=False,
    )


def _scanner(config=None, tmp_path=None) -> SecurityScanner:
    missy_dir = str(tmp_path) if tmp_path else "/nonexistent/.missy"
    return SecurityScanner(
        config=config, config_path="/nonexistent/config.yaml", missy_dir=missy_dir
    )


# ---------------------------------------------------------------------------
# Finding and ScanResult unit tests
# ---------------------------------------------------------------------------


class TestFinding:
    def test_dataclass_creation(self):
        f = Finding(
            id="SEC-001",
            title="Test Finding",
            description="A test description.",
            severity=Severity.HIGH,
            category="config",
            recommendation="Fix it.",
        )
        assert f.id == "SEC-001"
        assert f.severity == Severity.HIGH
        assert f.details == {}

    def test_details_default_is_independent(self):
        f1 = Finding("A", "T", "D", Severity.INFO, "c", "r")
        f2 = Finding("A", "T", "D", Severity.INFO, "c", "r")
        f1.details["key"] = "val"
        assert "key" not in f2.details


class TestScanResult:
    def _make_result(self, severities: list[Severity]) -> ScanResult:
        findings = [
            Finding(f"SEC-{i:03d}", "t", "d", sev, "c", "r") for i, sev in enumerate(severities)
        ]
        summary = {s.value: severities.count(s) for s in Severity}
        return ScanResult(
            findings=findings,
            scanned_at="2026-01-01T00:00:00+00:00",
            scan_duration_ms=42.0,
            summary=summary,
        )

    def test_critical_count(self):
        r = self._make_result([Severity.CRITICAL, Severity.CRITICAL, Severity.HIGH])
        assert r.critical_count == 2

    def test_has_critical_true(self):
        r = self._make_result([Severity.CRITICAL])
        assert r.has_critical is True

    def test_has_critical_false(self):
        r = self._make_result([Severity.HIGH, Severity.MEDIUM])
        assert r.has_critical is False

    def test_format_report_contains_finding_id(self):
        r = self._make_result([Severity.HIGH])
        report = r.format_report()
        assert "SEC-000" in report

    def test_format_report_verbose_includes_description(self):
        findings = [
            Finding(
                "SEC-777", "My Title", "Full description here.", Severity.MEDIUM, "c", "Fix it."
            )
        ]
        r = ScanResult(
            findings=findings, scanned_at="x", scan_duration_ms=1.0, summary={"medium": 1}
        )
        report = r.format_report(verbose=True)
        assert "Full description here." in report

    def test_format_report_empty_findings(self):
        r = ScanResult(findings=[], scanned_at="x", scan_duration_ms=1.0, summary={})
        report = r.format_report()
        assert "No findings" in report

    def test_to_json_structure(self):
        r = self._make_result([Severity.INFO])
        data = r.to_json()
        assert "findings" in data
        assert "summary" in data
        assert "scanned_at" in data
        assert data["findings"][0]["severity"] == "info"


# ---------------------------------------------------------------------------
# Scanner: config security (SEC-001, SEC-002, SEC-003)
# ---------------------------------------------------------------------------


class TestCheckConfigSecurity:
    def test_sec_001_world_writable_config(self, tmp_path):
        cfg_path = tmp_path / "config.yaml"
        cfg_path.write_text("config_version: 2\n")
        cfg_path.chmod(0o666)
        scanner = SecurityScanner(
            config=_make_config(), config_path=str(cfg_path), missy_dir=str(tmp_path / ".missy")
        )
        result = scanner.scan_all()
        ids = [f.id for f in result.findings]
        assert "SEC-001" in ids

    def test_sec_001_not_raised_for_correct_perms(self, tmp_path):
        cfg_path = tmp_path / "config.yaml"
        cfg_path.write_text("config_version: 2\n")
        cfg_path.chmod(0o600)
        scanner = SecurityScanner(
            config=_make_config(), config_path=str(cfg_path), missy_dir=str(tmp_path / ".missy")
        )
        result = scanner.scan_all()
        ids = [f.id for f in result.findings]
        assert "SEC-001" not in ids

    def test_sec_002_plaintext_api_key(self, tmp_path):
        cfg = _make_config(providers={"anthropic": {"api_key": "sk-ant-reallylong1234567890key"}})
        scanner = _scanner(config=cfg, tmp_path=tmp_path / ".missy")
        result = scanner.scan_all()
        ids = [f.id for f in result.findings]
        assert "SEC-002" in ids

    def test_sec_002_vault_ref_not_flagged(self, tmp_path):
        cfg = _make_config(providers={"anthropic": {"api_key": "vault://ANTHROPIC_KEY"}})
        scanner = _scanner(config=cfg, tmp_path=tmp_path / ".missy")
        result = scanner.scan_all()
        ids = [f.id for f in result.findings]
        assert "SEC-002" not in ids

    def test_sec_002_env_ref_not_flagged(self, tmp_path):
        cfg = _make_config(providers={"anthropic": {"api_key": "$ANTHROPIC_API_KEY"}})
        scanner = _scanner(config=cfg, tmp_path=tmp_path / ".missy")
        result = scanner.scan_all()
        ids = [f.id for f in result.findings]
        assert "SEC-002" not in ids

    def test_sec_003_old_config_version(self, tmp_path):
        cfg = _make_config(config_version=0)
        scanner = _scanner(config=cfg, tmp_path=tmp_path / ".missy")
        result = scanner.scan_all()
        ids = [f.id for f in result.findings]
        assert "SEC-003" in ids
        assert (
            result.findings[[f.id for f in result.findings].index("SEC-003")].severity
            == Severity.LOW
        )

    def test_sec_003_not_raised_for_current_version(self, tmp_path):
        cfg = _make_config(config_version=2)
        scanner = _scanner(config=cfg, tmp_path=tmp_path / ".missy")
        result = scanner.scan_all()
        ids = [f.id for f in result.findings]
        assert "SEC-003" not in ids


# ---------------------------------------------------------------------------
# Scanner: network policy (SEC-010 .. SEC-013)
# ---------------------------------------------------------------------------


class TestCheckNetworkPolicy:
    def test_sec_010_default_deny_false_is_critical(self, tmp_path):
        cfg = _make_config(network_default_deny=False)
        scanner = _scanner(config=cfg, tmp_path=tmp_path / ".missy")
        result = scanner.scan_all()
        ids = [f.id for f in result.findings]
        assert "SEC-010" in ids
        idx = ids.index("SEC-010")
        assert result.findings[idx].severity == Severity.CRITICAL

    def test_sec_010_not_raised_when_default_deny_true(self, tmp_path):
        cfg = _make_config(network_default_deny=True)
        scanner = _scanner(config=cfg, tmp_path=tmp_path / ".missy")
        result = scanner.scan_all()
        ids = [f.id for f in result.findings]
        assert "SEC-010" not in ids

    def test_sec_011_broad_cidr_critical(self, tmp_path):
        cfg = _make_config(allowed_cidrs=["0.0.0.0/0"])
        scanner = _scanner(config=cfg, tmp_path=tmp_path / ".missy")
        result = scanner.scan_all()
        ids = [f.id for f in result.findings]
        assert "SEC-011" in ids
        idx = ids.index("SEC-011")
        assert result.findings[idx].severity == Severity.CRITICAL

    def test_sec_011_specific_cidr_not_flagged(self, tmp_path):
        cfg = _make_config(allowed_cidrs=["203.0.113.0/24"])
        scanner = _scanner(config=cfg, tmp_path=tmp_path / ".missy")
        result = scanner.scan_all()
        ids = [f.id for f in result.findings]
        assert "SEC-011" not in ids

    def test_sec_012_no_rest_policies_low(self, tmp_path):
        cfg = _make_config(rest_policies=[])
        scanner = _scanner(config=cfg, tmp_path=tmp_path / ".missy")
        result = scanner.scan_all()
        ids = [f.id for f in result.findings]
        assert "SEC-012" in ids
        idx = ids.index("SEC-012")
        assert result.findings[idx].severity == Severity.LOW

    def test_sec_012_rest_policies_present_not_flagged(self, tmp_path):
        cfg = _make_config(
            rest_policies=[
                {"host": "api.anthropic.com", "method": "POST", "path": "/**", "action": "allow"}
            ]
        )
        scanner = _scanner(config=cfg, tmp_path=tmp_path / ".missy")
        result = scanner.scan_all()
        ids = [f.id for f in result.findings]
        assert "SEC-012" not in ids

    def test_sec_013_broad_domain_high(self, tmp_path):
        cfg = _make_config(allowed_domains=[".com"])
        scanner = _scanner(config=cfg, tmp_path=tmp_path / ".missy")
        result = scanner.scan_all()
        ids = [f.id for f in result.findings]
        assert "SEC-013" in ids
        idx = ids.index("SEC-013")
        assert result.findings[idx].severity == Severity.HIGH

    def test_sec_013_specific_domain_not_flagged(self, tmp_path):
        cfg = _make_config(allowed_domains=["api.anthropic.com"])
        scanner = _scanner(config=cfg, tmp_path=tmp_path / ".missy")
        result = scanner.scan_all()
        ids = [f.id for f in result.findings]
        assert "SEC-013" not in ids


# ---------------------------------------------------------------------------
# Scanner: filesystem policy (SEC-020 .. SEC-022)
# ---------------------------------------------------------------------------


class TestCheckFilesystemPolicy:
    def test_sec_020_no_read_paths_medium(self, tmp_path):
        cfg = _make_config(read_paths=[])
        scanner = _scanner(config=cfg, tmp_path=tmp_path / ".missy")
        result = scanner.scan_all()
        ids = [f.id for f in result.findings]
        assert "SEC-020" in ids
        idx = ids.index("SEC-020")
        assert result.findings[idx].severity == Severity.MEDIUM

    def test_sec_020_read_paths_configured_not_flagged(self, tmp_path):
        cfg = _make_config(read_paths=["~/workspace"])
        scanner = _scanner(config=cfg, tmp_path=tmp_path / ".missy")
        result = scanner.scan_all()
        ids = [f.id for f in result.findings]
        assert "SEC-020" not in ids

    def test_sec_021_write_to_etc_critical(self, tmp_path):
        cfg = _make_config(write_paths=["/etc"])
        scanner = _scanner(config=cfg, tmp_path=tmp_path / ".missy")
        result = scanner.scan_all()
        ids = [f.id for f in result.findings]
        assert "SEC-021" in ids
        idx = ids.index("SEC-021")
        assert result.findings[idx].severity == Severity.CRITICAL

    def test_sec_021_write_to_usr_critical(self, tmp_path):
        cfg = _make_config(write_paths=["/usr/local"])
        scanner = _scanner(config=cfg, tmp_path=tmp_path / ".missy")
        result = scanner.scan_all()
        ids = [f.id for f in result.findings]
        assert "SEC-021" in ids

    def test_sec_021_workspace_write_not_flagged(self, tmp_path):
        cfg = _make_config(write_paths=["~/workspace"])
        scanner = _scanner(config=cfg, tmp_path=tmp_path / ".missy")
        result = scanner.scan_all()
        ids = [f.id for f in result.findings]
        assert "SEC-021" not in ids
        assert "SEC-022" not in ids

    def test_sec_022_home_dir_write_high(self, tmp_path):
        home = str(Path.home())
        cfg = _make_config(write_paths=[home])
        scanner = _scanner(config=cfg, tmp_path=tmp_path / ".missy")
        result = scanner.scan_all()
        ids = [f.id for f in result.findings]
        assert "SEC-022" in ids
        idx = ids.index("SEC-022")
        assert result.findings[idx].severity == Severity.HIGH


# ---------------------------------------------------------------------------
# Scanner: shell policy (SEC-030 .. SEC-032)
# ---------------------------------------------------------------------------


class TestCheckShellPolicy:
    def test_shell_disabled_no_shell_findings(self, tmp_path):
        cfg = _make_config(shell_enabled=False)
        scanner = _scanner(config=cfg, tmp_path=tmp_path / ".missy")
        result = scanner.scan_all()
        shell_ids = [f.id for f in result.findings if f.id.startswith("SEC-03")]
        assert shell_ids == []

    def test_sec_030_shell_enabled_empty_allowlist_critical(self, tmp_path):
        cfg = _make_config(shell_enabled=True, allowed_commands=[])
        scanner = _scanner(config=cfg, tmp_path=tmp_path / ".missy")
        result = scanner.scan_all()
        ids = [f.id for f in result.findings]
        assert "SEC-030" in ids
        idx = ids.index("SEC-030")
        assert result.findings[idx].severity == Severity.CRITICAL

    def test_sec_031_dangerous_command_rm_high(self, tmp_path):
        cfg = _make_config(shell_enabled=True, allowed_commands=["git", "rm", "ls"])
        scanner = _scanner(config=cfg, tmp_path=tmp_path / ".missy")
        result = scanner.scan_all()
        ids = [f.id for f in result.findings]
        assert "SEC-031" in ids
        f = result.findings[ids.index("SEC-031")]
        assert f.severity == Severity.HIGH
        assert "rm" in f.details.get("dangerous_commands", [])

    def test_sec_031_multiple_dangerous_commands(self, tmp_path):
        cfg = _make_config(shell_enabled=True, allowed_commands=["rm", "dd", "chmod", "ls"])
        scanner = _scanner(config=cfg, tmp_path=tmp_path / ".missy")
        result = scanner.scan_all()
        ids = [f.id for f in result.findings]
        assert "SEC-031" in ids
        f = result.findings[ids.index("SEC-031")]
        dangerous = set(f.details.get("dangerous_commands", []))
        assert {"rm", "dd", "chmod"}.issubset(dangerous)

    def test_sec_031_safe_commands_not_flagged(self, tmp_path):
        cfg = _make_config(shell_enabled=True, allowed_commands=["git", "ls", "cat", "grep"])
        scanner = _scanner(config=cfg, tmp_path=tmp_path / ".missy")
        result = scanner.scan_all()
        ids = [f.id for f in result.findings]
        assert "SEC-031" not in ids

    def test_sec_032_python_interpreter_high(self, tmp_path):
        cfg = _make_config(shell_enabled=True, allowed_commands=["python3", "git"])
        scanner = _scanner(config=cfg, tmp_path=tmp_path / ".missy")
        result = scanner.scan_all()
        ids = [f.id for f in result.findings]
        assert "SEC-032" in ids
        f = result.findings[ids.index("SEC-032")]
        assert f.severity == Severity.HIGH

    def test_sec_032_bash_interpreter_high(self, tmp_path):
        cfg = _make_config(shell_enabled=True, allowed_commands=["bash", "git"])
        scanner = _scanner(config=cfg, tmp_path=tmp_path / ".missy")
        result = scanner.scan_all()
        ids = [f.id for f in result.findings]
        assert "SEC-032" in ids


# ---------------------------------------------------------------------------
# Scanner: MCP security (SEC-040 .. SEC-042)
# ---------------------------------------------------------------------------


class TestCheckMcpSecurity:
    def test_no_mcp_config_no_mcp_findings(self, tmp_path):
        d = tmp_path / ".missy"
        d.mkdir()
        scanner = SecurityScanner(
            config=_make_config(), config_path="/nonexistent", missy_dir=str(d)
        )
        result = scanner.scan_all()
        mcp_ids = [f.id for f in result.findings if f.id.startswith("SEC-04")]
        assert mcp_ids == []

    def test_sec_040_mcp_config_world_readable(self, tmp_path):
        d = tmp_path / ".missy"
        d.mkdir()
        mcp_path = d / "mcp.json"
        mcp_path.write_text(json.dumps([{"name": "fs", "command": "npx mcp-fs /tmp"}]))
        mcp_path.chmod(0o644)  # group/world readable
        scanner = SecurityScanner(
            config=_make_config(), config_path="/nonexistent", missy_dir=str(d)
        )
        result = scanner.scan_all()
        ids = [f.id for f in result.findings]
        assert "SEC-040" in ids

    def test_sec_040_correct_permissions_not_flagged(self, tmp_path):
        d = tmp_path / ".missy"
        d.mkdir()
        mcp_path = d / "mcp.json"
        mcp_path.write_text(
            json.dumps([{"name": "fs", "command": "npx mcp-fs /tmp", "digest": "sha256:abc"}])
        )
        mcp_path.chmod(0o600)
        scanner = SecurityScanner(
            config=_make_config(), config_path="/nonexistent", missy_dir=str(d)
        )
        result = scanner.scan_all()
        ids = [f.id for f in result.findings]
        assert "SEC-040" not in ids

    def test_sec_041_no_digest_high(self, tmp_path):
        d = tmp_path / ".missy"
        d.mkdir()
        mcp_path = d / "mcp.json"
        mcp_path.write_text(json.dumps([{"name": "filesystem", "command": "npx @mcp/fs /tmp"}]))
        mcp_path.chmod(0o600)
        scanner = SecurityScanner(
            config=_make_config(), config_path="/nonexistent", missy_dir=str(d)
        )
        result = scanner.scan_all()
        ids = [f.id for f in result.findings]
        assert "SEC-041" in ids
        f = result.findings[ids.index("SEC-041")]
        assert f.severity == Severity.HIGH
        assert "filesystem" in f.details.get("server", "")

    def test_sec_041_pinned_digest_not_flagged(self, tmp_path):
        d = tmp_path / ".missy"
        d.mkdir()
        mcp_path = d / "mcp.json"
        mcp_path.write_text(
            json.dumps(
                [
                    {
                        "name": "filesystem",
                        "command": "npx @mcp/fs /tmp",
                        "digest": "sha256:abc123",
                    }
                ]
            )
        )
        mcp_path.chmod(0o600)
        scanner = SecurityScanner(
            config=_make_config(), config_path="/nonexistent", missy_dir=str(d)
        )
        result = scanner.scan_all()
        ids = [f.id for f in result.findings]
        assert "SEC-041" not in ids

    def test_sec_042_npx_command_medium(self, tmp_path):
        d = tmp_path / ".missy"
        d.mkdir()
        mcp_path = d / "mcp.json"
        mcp_path.write_text(
            json.dumps(
                [
                    {
                        "name": "db",
                        "command": "npx @mcp/postgres postgresql://localhost/mydb",
                        "digest": "sha256:pinned",
                    }
                ]
            )
        )
        mcp_path.chmod(0o600)
        scanner = SecurityScanner(
            config=_make_config(), config_path="/nonexistent", missy_dir=str(d)
        )
        result = scanner.scan_all()
        ids = [f.id for f in result.findings]
        assert "SEC-042" in ids
        f = result.findings[ids.index("SEC-042")]
        assert f.severity == Severity.MEDIUM

    def test_sec_042_direct_binary_not_flagged(self, tmp_path):
        d = tmp_path / ".missy"
        d.mkdir()
        mcp_path = d / "mcp.json"
        mcp_path.write_text(
            json.dumps(
                [
                    {
                        "name": "mymcp",
                        "command": "/usr/local/bin/mymcp --config /etc/mymcp.conf",
                        "digest": "sha256:pinned",
                    }
                ]
            )
        )
        mcp_path.chmod(0o600)
        scanner = SecurityScanner(
            config=_make_config(), config_path="/nonexistent", missy_dir=str(d)
        )
        result = scanner.scan_all()
        ids = [f.id for f in result.findings]
        assert "SEC-042" not in ids

    def test_multiple_mcp_servers_each_checked(self, tmp_path):
        d = tmp_path / ".missy"
        d.mkdir()
        mcp_path = d / "mcp.json"
        mcp_path.write_text(
            json.dumps(
                [
                    {"name": "server_a", "command": "npx @mcp/a"},
                    {"name": "server_b", "command": "npx @mcp/b"},
                ]
            )
        )
        mcp_path.chmod(0o600)
        scanner = SecurityScanner(
            config=_make_config(), config_path="/nonexistent", missy_dir=str(d)
        )
        result = scanner.scan_all()
        sec_041 = [f for f in result.findings if f.id == "SEC-041"]
        assert len(sec_041) == 2
        server_names = {f.details.get("server") for f in sec_041}
        assert server_names == {"server_a", "server_b"}


# ---------------------------------------------------------------------------
# Scanner: secrets and vault (SEC-060 .. SEC-062)
# ---------------------------------------------------------------------------


class TestCheckSecretsAndVault:
    def test_sec_060_vault_disabled_plaintext_key_medium(self, tmp_path):
        cfg = _make_config(
            vault_enabled=False,
            providers={"anthropic": {"api_key": "sk-ant-longkey1234567890abcdef"}},
        )
        scanner = _scanner(config=cfg, tmp_path=tmp_path / ".missy")
        result = scanner.scan_all()
        ids = [f.id for f in result.findings]
        assert "SEC-060" in ids
        f = result.findings[ids.index("SEC-060")]
        assert f.severity == Severity.MEDIUM

    def test_sec_060_vault_enabled_no_flag(self, tmp_path):
        cfg = _make_config(
            vault_enabled=True,
            providers={"anthropic": {"api_key": "vault://ANTHROPIC_KEY"}},
        )
        scanner = _scanner(config=cfg, tmp_path=tmp_path / ".missy")
        result = scanner.scan_all()
        ids = [f.id for f in result.findings]
        assert "SEC-060" not in ids

    def test_sec_061_vault_key_permissive_critical(self, tmp_path):
        secrets_dir = tmp_path / "secrets"
        secrets_dir.mkdir()
        vault_key = secrets_dir / "vault.key"
        vault_key.write_bytes(b"x" * 32)
        vault_key.chmod(0o644)  # Too open
        cfg = _make_config(vault_dir=str(secrets_dir))
        d = tmp_path / ".missy"
        d.mkdir()
        scanner = SecurityScanner(config=cfg, config_path="/nonexistent", missy_dir=str(d))
        result = scanner.scan_all()
        ids = [f.id for f in result.findings]
        assert "SEC-061" in ids
        f = result.findings[ids.index("SEC-061")]
        assert f.severity == Severity.CRITICAL

    def test_sec_061_vault_key_600_not_flagged(self, tmp_path):
        secrets_dir = tmp_path / "secrets"
        secrets_dir.mkdir()
        vault_key = secrets_dir / "vault.key"
        vault_key.write_bytes(b"x" * 32)
        vault_key.chmod(0o600)
        cfg = _make_config(vault_dir=str(secrets_dir))
        d = tmp_path / ".missy"
        d.mkdir()
        scanner = SecurityScanner(config=cfg, config_path="/nonexistent", missy_dir=str(d))
        result = scanner.scan_all()
        ids = [f.id for f in result.findings]
        assert "SEC-061" not in ids

    def test_sec_062_env_var_set_low(self, tmp_path, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test123")
        cfg = _make_config()
        scanner = _scanner(config=cfg, tmp_path=tmp_path / ".missy")
        result = scanner.scan_all()
        ids = [f.id for f in result.findings]
        assert "SEC-062" in ids
        f = result.findings[ids.index("SEC-062")]
        assert f.severity == Severity.LOW
        assert "ANTHROPIC_API_KEY" in f.details.get("env_vars", [])

    def test_sec_062_no_env_vars_not_flagged(self, tmp_path, monkeypatch):
        from missy.security.scanner import _SECRET_ENV_VARS

        for var in _SECRET_ENV_VARS:
            monkeypatch.delenv(var, raising=False)
        cfg = _make_config()
        scanner = _scanner(config=cfg, tmp_path=tmp_path / ".missy")
        result = scanner.scan_all()
        ids = [f.id for f in result.findings]
        assert "SEC-062" not in ids


# ---------------------------------------------------------------------------
# Scanner: identity (SEC-070 .. SEC-071)
# ---------------------------------------------------------------------------


class TestCheckIdentity:
    def test_sec_070_no_identity_key_medium(self, tmp_path):
        d = tmp_path / ".missy"
        d.mkdir()
        # identity.pem does not exist
        scanner = SecurityScanner(
            config=_make_config(), config_path="/nonexistent", missy_dir=str(d)
        )
        result = scanner.scan_all()
        ids = [f.id for f in result.findings]
        assert "SEC-070" in ids
        f = result.findings[ids.index("SEC-070")]
        assert f.severity == Severity.MEDIUM

    def test_sec_071_identity_key_permissive_high(self, tmp_path):
        d = tmp_path / ".missy"
        d.mkdir()
        key_path = d / "identity.pem"
        key_path.write_bytes(b"-----BEGIN PRIVATE KEY-----\ntest\n-----END PRIVATE KEY-----\n")
        key_path.chmod(0o644)
        scanner = SecurityScanner(
            config=_make_config(), config_path="/nonexistent", missy_dir=str(d)
        )
        result = scanner.scan_all()
        ids = [f.id for f in result.findings]
        assert "SEC-071" in ids
        f = result.findings[ids.index("SEC-071")]
        assert f.severity == Severity.HIGH

    def test_sec_071_identity_key_600_not_flagged(self, tmp_path):
        d = tmp_path / ".missy"
        d.mkdir()
        key_path = d / "identity.pem"
        key_path.write_bytes(b"-----BEGIN PRIVATE KEY-----\ntest\n-----END PRIVATE KEY-----\n")
        key_path.chmod(0o600)
        scanner = SecurityScanner(
            config=_make_config(), config_path="/nonexistent", missy_dir=str(d)
        )
        result = scanner.scan_all()
        ids = [f.id for f in result.findings]
        assert "SEC-071" not in ids
        assert "SEC-070" not in ids  # key exists, so no SEC-070


# ---------------------------------------------------------------------------
# Scanner: file permissions (SEC-080 .. SEC-082)
# ---------------------------------------------------------------------------


class TestCheckFilePermissions:
    def test_sec_080_missy_dir_world_readable_high(self, tmp_path):
        d = tmp_path / ".missy"
        d.mkdir(mode=0o755)  # world-executable/readable
        scanner = SecurityScanner(
            config=_make_config(), config_path="/nonexistent", missy_dir=str(d)
        )
        result = scanner.scan_all()
        ids = [f.id for f in result.findings]
        assert "SEC-080" in ids
        f = result.findings[ids.index("SEC-080")]
        assert f.severity == Severity.HIGH

    def test_sec_080_missy_dir_700_not_flagged(self, tmp_path):
        d = tmp_path / ".missy"
        d.mkdir(mode=0o700)
        scanner = SecurityScanner(
            config=_make_config(), config_path="/nonexistent", missy_dir=str(d)
        )
        result = scanner.scan_all()
        ids = [f.id for f in result.findings]
        assert "SEC-080" not in ids

    def test_sec_081_audit_world_readable_medium(self, tmp_path):
        d = tmp_path / ".missy"
        d.mkdir(mode=0o700)
        audit = d / "audit.jsonl"
        audit.write_text("")
        audit.chmod(0o644)
        scanner = SecurityScanner(
            config=_make_config(), config_path="/nonexistent", missy_dir=str(d)
        )
        result = scanner.scan_all()
        ids = [f.id for f in result.findings]
        assert "SEC-081" in ids
        f = result.findings[ids.index("SEC-081")]
        assert f.severity == Severity.MEDIUM

    def test_sec_081_audit_600_not_flagged(self, tmp_path):
        d = tmp_path / ".missy"
        d.mkdir(mode=0o700)
        audit = d / "audit.jsonl"
        audit.write_text("")
        audit.chmod(0o600)
        scanner = SecurityScanner(
            config=_make_config(), config_path="/nonexistent", missy_dir=str(d)
        )
        result = scanner.scan_all()
        ids = [f.id for f in result.findings]
        assert "SEC-081" not in ids

    def test_sec_082_memory_db_world_readable_medium(self, tmp_path):
        d = tmp_path / ".missy"
        d.mkdir(mode=0o700)
        mem = d / "memory.db"
        mem.write_bytes(b"")
        mem.chmod(0o644)
        scanner = SecurityScanner(
            config=_make_config(), config_path="/nonexistent", missy_dir=str(d)
        )
        result = scanner.scan_all()
        ids = [f.id for f in result.findings]
        assert "SEC-082" in ids
        f = result.findings[ids.index("SEC-082")]
        assert f.severity == Severity.MEDIUM

    def test_sec_082_memory_db_600_not_flagged(self, tmp_path):
        d = tmp_path / ".missy"
        d.mkdir(mode=0o700)
        mem = d / "memory.db"
        mem.write_bytes(b"")
        mem.chmod(0o600)
        scanner = SecurityScanner(
            config=_make_config(), config_path="/nonexistent", missy_dir=str(d)
        )
        result = scanner.scan_all()
        ids = [f.id for f in result.findings]
        assert "SEC-082" not in ids


# ---------------------------------------------------------------------------
# Scanner: known vulnerabilities (SEC-090 .. SEC-093)
# ---------------------------------------------------------------------------


class TestCheckKnownVulnerabilities:
    def test_sec_090_container_disabled_low(self, tmp_path):
        cfg = _make_config(container_enabled=False)
        scanner = _scanner(config=cfg, tmp_path=tmp_path / ".missy")
        result = scanner.scan_all()
        ids = [f.id for f in result.findings]
        assert "SEC-090" in ids
        f = result.findings[ids.index("SEC-090")]
        assert f.severity == Severity.LOW

    def test_sec_091_no_spend_limit_medium(self, tmp_path):
        cfg = _make_config(max_spend_usd=0.0)
        scanner = _scanner(config=cfg, tmp_path=tmp_path / ".missy")
        result = scanner.scan_all()
        ids = [f.id for f in result.findings]
        assert "SEC-091" in ids
        f = result.findings[ids.index("SEC-091")]
        assert f.severity == Severity.MEDIUM

    def test_sec_091_spend_limit_not_flagged(self, tmp_path):
        cfg = _make_config(max_spend_usd=5.0)
        scanner = _scanner(config=cfg, tmp_path=tmp_path / ".missy")
        result = scanner.scan_all()
        ids = [f.id for f in result.findings]
        assert "SEC-091" not in ids

    def test_sec_092_no_otel_warning_level_info(self, tmp_path):
        cfg = _make_config(otel_enabled=False, log_level="warning")
        scanner = _scanner(config=cfg, tmp_path=tmp_path / ".missy")
        result = scanner.scan_all()
        ids = [f.id for f in result.findings]
        assert "SEC-092" in ids
        f = result.findings[ids.index("SEC-092")]
        assert f.severity == Severity.INFO

    def test_sec_092_otel_enabled_not_flagged(self, tmp_path):
        cfg = _make_config(otel_enabled=True, log_level="info")
        scanner = _scanner(config=cfg, tmp_path=tmp_path / ".missy")
        result = scanner.scan_all()
        ids = [f.id for f in result.findings]
        assert "SEC-092" not in ids

    def test_sec_093_shell_enabled_info(self, tmp_path):
        cfg = _make_config(shell_enabled=True, allowed_commands=["git", "ls"])
        scanner = _scanner(config=cfg, tmp_path=tmp_path / ".missy")
        result = scanner.scan_all()
        ids = [f.id for f in result.findings]
        assert "SEC-093" in ids
        f = result.findings[ids.index("SEC-093")]
        assert f.severity == Severity.INFO


# ---------------------------------------------------------------------------
# End-to-end posture tests
# ---------------------------------------------------------------------------


class TestPosture:
    def test_well_hardened_config_only_info_and_low(self, tmp_path, monkeypatch, secure_config):
        """A well-configured installation produces only INFO/LOW findings."""
        # Clear all tracked secret env vars so SEC-062 is not raised
        from missy.security.scanner import _SECRET_ENV_VARS

        for var in _SECRET_ENV_VARS:
            monkeypatch.delenv(var, raising=False)

        # Isolate from global tool registry state left by other tests
        from missy.tools import registry as _tr_mod

        monkeypatch.setattr(_tr_mod, "_registry", None)

        d = tmp_path / ".missy"
        d.mkdir(mode=0o700)
        # Create identity key with correct permissions
        identity = d / "identity.pem"
        identity.write_bytes(b"-----BEGIN PRIVATE KEY-----\nok\n-----END PRIVATE KEY-----\n")
        identity.chmod(0o600)
        # Create audit log and memory db with correct permissions
        (d / "audit.jsonl").write_text("")
        (d / "audit.jsonl").chmod(0o600)
        (d / "memory.db").write_bytes(b"")
        (d / "memory.db").chmod(0o600)

        scanner = SecurityScanner(
            config=secure_config,
            config_path="/nonexistent/config.yaml",
            missy_dir=str(d),
        )
        result = scanner.scan_all()

        high_plus = [
            f
            for f in result.findings
            if f.severity in (Severity.CRITICAL, Severity.HIGH, Severity.MEDIUM)
        ]
        assert high_plus == [], (
            "Expected no CRITICAL/HIGH/MEDIUM findings for hardened config, "
            f"got: {[(f.id, f.severity, f.title) for f in high_plus]}"
        )

    def test_worst_case_config_multiple_critical(self, tmp_path, monkeypatch, insecure_config):
        """A worst-case configuration produces multiple CRITICAL findings."""
        for var in ("ANTHROPIC_API_KEY",):
            monkeypatch.delenv(var, raising=False)

        d = tmp_path / ".missy"
        d.mkdir(mode=0o755)  # world-readable dir

        scanner = SecurityScanner(
            config=insecure_config,
            config_path="/nonexistent/config.yaml",
            missy_dir=str(d),
        )
        result = scanner.scan_all()
        assert result.has_critical, "Expected at least one CRITICAL finding for worst-case config"
        assert result.critical_count >= 2

    def test_findings_sorted_by_severity(self, tmp_path, insecure_config):
        d = tmp_path / ".missy"
        d.mkdir()
        scanner = SecurityScanner(
            config=insecure_config, config_path="/nonexistent", missy_dir=str(d)
        )
        result = scanner.scan_all()
        severities = [f.severity for f in result.findings]
        from missy.security.scanner import _SEVERITY_ORDER

        orders = [_SEVERITY_ORDER[s] for s in severities]
        assert orders == sorted(orders), "Findings should be sorted most-severe first"


# ---------------------------------------------------------------------------
# scan_all without config
# ---------------------------------------------------------------------------


class TestScanAllNoConfig:
    def test_missing_config_file_produces_sec_000(self, tmp_path):
        d = tmp_path / ".missy"
        d.mkdir()
        scanner = SecurityScanner(
            config=None,
            config_path=str(tmp_path / "does_not_exist.yaml"),
            missy_dir=str(d),
        )
        result = scanner.scan_all()
        ids = [f.id for f in result.findings]
        assert "SEC-000" in ids
        f = result.findings[ids.index("SEC-000")]
        assert f.severity == Severity.HIGH

    def test_summary_always_has_all_severity_keys(self, tmp_path):
        d = tmp_path / ".missy"
        d.mkdir()
        scanner = _scanner(config=_make_config(), tmp_path=d)
        result = scanner.scan_all()
        for sev in Severity:
            assert sev.value in result.summary

    def test_scan_duration_positive(self, tmp_path):
        cfg = _make_config()
        scanner = _scanner(config=cfg, tmp_path=tmp_path / ".missy")
        result = scanner.scan_all()
        assert result.scan_duration_ms >= 0.0

    def test_scanned_at_is_utc_iso(self, tmp_path):
        cfg = _make_config()
        scanner = _scanner(config=cfg, tmp_path=tmp_path / ".missy")
        result = scanner.scan_all()
        # Should parse without error and contain timezone info
        from datetime import datetime

        dt = datetime.fromisoformat(result.scanned_at)
        assert dt.tzinfo is not None
