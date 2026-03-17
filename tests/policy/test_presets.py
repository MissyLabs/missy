"""Tests for network policy presets (Feature 1)."""

from __future__ import annotations

import textwrap


class TestResolvePresets:
    """Tests for missy.policy.presets.resolve_presets."""

    def test_resolve_known_preset(self):
        from missy.policy.presets import resolve_presets

        hosts, domains, cidrs, unknown = resolve_presets(["anthropic"])
        assert "api.anthropic.com" in hosts
        assert "anthropic.com" in domains
        assert unknown == []

    def test_resolve_multiple_deduplication(self):
        from missy.policy.presets import resolve_presets

        # openai and discord both have unique entries; merging should not dup
        hosts, domains, cidrs, unknown = resolve_presets(["openai", "discord"])
        assert len(hosts) == len(set(hosts)), "hosts should be deduplicated"
        assert len(domains) == len(set(domains)), "domains should be deduplicated"
        assert unknown == []

    def test_unknown_preset_warning(self):
        from missy.policy.presets import resolve_presets

        hosts, domains, cidrs, unknown = resolve_presets(["anthropic", "nonexistent"])
        assert "nonexistent" in unknown
        # Known preset still resolved
        assert "api.anthropic.com" in hosts

    def test_empty_presets(self):
        from missy.policy.presets import resolve_presets

        hosts, domains, cidrs, unknown = resolve_presets([])
        assert hosts == []
        assert domains == []
        assert cidrs == []
        assert unknown == []

    def test_ollama_preset_has_cidrs(self):
        from missy.policy.presets import resolve_presets

        hosts, domains, cidrs, unknown = resolve_presets(["ollama"])
        assert "127.0.0.0/8" in cidrs
        assert "localhost:11434" in hosts


class TestParseNetworkWithPresets:
    """Tests for preset integration in _parse_network."""

    def test_parse_network_with_presets(self):
        from missy.config.settings import _parse_network

        data = {"presets": ["anthropic"], "allowed_hosts": []}
        policy = _parse_network(data)
        assert "api.anthropic.com" in policy.allowed_hosts
        assert "anthropic.com" in policy.allowed_domains
        assert policy.presets == ["anthropic"]

    def test_presets_plus_explicit_hosts(self):
        from missy.config.settings import _parse_network

        data = {
            "presets": ["anthropic"],
            "allowed_hosts": ["custom.example.com"],
            "allowed_domains": ["example.com"],
        }
        policy = _parse_network(data)
        # Both preset and explicit entries present
        assert "api.anthropic.com" in policy.allowed_hosts
        assert "custom.example.com" in policy.allowed_hosts
        assert "anthropic.com" in policy.allowed_domains
        assert "example.com" in policy.allowed_domains
        # No duplicates
        assert len(policy.allowed_hosts) == len(set(policy.allowed_hosts))

    def test_presets_with_load_config(self, tmp_path):
        """Full round-trip: YAML with presets → NetworkPolicy has merged values."""
        from missy.config.settings import load_config

        config_yaml = textwrap.dedent("""\
            network:
              default_deny: true
              presets:
                - anthropic
                - github
              allowed_hosts:
                - "custom.host.com"

            providers:
              anthropic:
                name: anthropic
                model: "claude-sonnet-4-6"

            workspace_path: "~/workspace"
            audit_log_path: "~/.missy/audit.jsonl"
        """)
        config_file = tmp_path / "config.yaml"
        config_file.write_text(config_yaml)

        cfg = load_config(str(config_file))
        assert "api.anthropic.com" in cfg.network.allowed_hosts
        assert "api.github.com" in cfg.network.allowed_hosts
        assert "custom.host.com" in cfg.network.allowed_hosts
        assert "github.com" in cfg.network.allowed_domains
