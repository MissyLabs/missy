"""SEC-02: Discord role allowlists must match unforgeable role IDs."""

from __future__ import annotations

import logging
from unittest.mock import MagicMock

from missy.channels.discord.config import DiscordGuildPolicy, _parse_guild_policy
from tests.channels.test_discord_channel_coverage import _make_account, _make_channel

REAL_ROLE = "111111111111111111"
FAKE_ROLE = "222222222222222222"


def _msg(role_ids: list[str]) -> dict:
    return {"channel": {"name": "general"}, "member": {"roles": role_ids}}


def _channel(policy: DiscordGuildPolicy):
    ch = _make_channel(_make_account(guild_policies={"g": policy}))
    ch._rest = MagicMock()
    # Two roles in the guild share the name "missy-admin" (one spoofed).
    ch._rest.get_guild_roles.return_value = [
        {"id": REAL_ROLE, "name": "missy-admin"},
        {"id": FAKE_ROLE, "name": "missy-admin"},
    ]
    return ch


class TestRoleIdAllowlist:
    def test_same_name_different_id_is_denied(self):
        ch = _channel(DiscordGuildPolicy(allowed_role_ids=[REAL_ROLE]))
        assert ch._check_guild_policy("g", "c", "u", "hi", _msg([FAKE_ROLE])) is False
        assert ch._check_guild_policy("g", "c", "u", "hi", _msg([REAL_ROLE])) is True
        ch._rest.get_guild_roles.assert_not_called()  # IDs need no lookup

    def test_ids_take_precedence_over_names(self):
        ch = _channel(
            DiscordGuildPolicy(allowed_roles=["missy-admin"], allowed_role_ids=[REAL_ROLE])
        )
        assert ch._check_guild_policy("g", "c", "u", "hi", _msg([FAKE_ROLE])) is False

    def test_legacy_names_still_work(self):
        ch = _channel(DiscordGuildPolicy(allowed_roles=["missy-admin"]))
        assert ch._check_guild_policy("g", "c", "u", "hi", _msg([FAKE_ROLE])) is True


class TestParsing:
    def test_snowflakes_under_allowed_roles_become_ids(self):
        policy = _parse_guild_policy({"allowed_roles": [REAL_ROLE, "mods"]})
        assert policy.allowed_role_ids == [REAL_ROLE]
        assert policy.allowed_roles == ["mods"]

    def test_name_entries_warn(self, caplog):
        with caplog.at_level(logging.WARNING, logger="missy.channels.discord.config"):
            _parse_guild_policy({"allowed_roles": ["mods"]})
        assert "allowed_role_ids" in caplog.text

    def test_allowed_role_ids_parsed_and_deduped(self):
        policy = _parse_guild_policy({"allowed_role_ids": [REAL_ROLE, int(REAL_ROLE)]})
        assert policy.allowed_role_ids == [REAL_ROLE]


class TestScanner:
    def _scan(self, tmp_path, policy):
        from missy.security.scanner import SecurityScanner

        cfg = MagicMock()
        cfg.discord.accounts = [_make_account(guild_policies={"g": policy})]
        scanner = SecurityScanner(config=cfg, missy_dir=str(tmp_path))
        scanner._findings = []
        scanner._check_known_vulnerabilities()
        return {f.id for f in scanner._findings}

    def test_name_based_roles_flagged(self, tmp_path):
        assert "SEC-095" in self._scan(tmp_path, DiscordGuildPolicy(allowed_roles=["mods"]))

    def test_id_based_roles_not_flagged(self, tmp_path):
        ids = self._scan(tmp_path, DiscordGuildPolicy(allowed_role_ids=[REAL_ROLE]))
        assert "SEC-095" not in ids


class TestMcpCleartextScan:
    def _scan(self, tmp_path, entry):
        import json

        from missy.security.scanner import SecurityScanner

        (tmp_path / "mcp.json").write_text(json.dumps([entry]))
        (tmp_path / "mcp.json").chmod(0o600)
        scanner = SecurityScanner(config=MagicMock(), missy_dir=str(tmp_path))
        scanner._findings = []
        scanner._check_mcp_security()
        return {f.id for f in scanner._findings}

    def test_bearer_over_http_flagged(self, tmp_path):
        ids = self._scan(tmp_path, {"name": "r", "url": "http://10.0.0.9/rpc", "bearer_token": "x"})
        assert "SEC-043" in ids

    def test_https_and_loopback_not_flagged(self, tmp_path):
        assert "SEC-043" not in self._scan(
            tmp_path, {"name": "r", "url": "https://x/rpc", "bearer_token": "x"}
        )
        assert "SEC-043" not in self._scan(
            tmp_path, {"name": "r", "url": "http://127.0.0.1:9/rpc", "bearer_token": "x"}
        )
