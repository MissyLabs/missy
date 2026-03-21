"""Tests for missy.cli.anthropic_auth — token classification, storage, and runtime helpers."""

from __future__ import annotations

import json
import time
from unittest.mock import MagicMock, patch

from missy.cli.anthropic_auth import (
    REFRESH_WARN_MARGIN,
    SETUP_TOKEN_TTL_SECONDS,
    classify_token,
    get_current_token,
    is_token_expiring,
    load_token,
    store_token,
)

# ---------------------------------------------------------------------------
# Token classification
# ---------------------------------------------------------------------------


class TestClassifyToken:
    """Tests for classify_token()."""

    def test_api_key_full_match(self):
        # 80+ chars after prefix
        key = "sk-ant-api03-" + "A" * 85
        assert classify_token(key) == "api_key"

    def test_setup_token_full_match(self):
        # 60+ chars after prefix
        token = "sk-ant-oat01-" + "B" * 65
        assert classify_token(token) == "setup_token"

    def test_api_key_prefix_heuristic(self):
        assert classify_token("sk-ant-api03-short") == "api_key"

    def test_setup_token_prefix_heuristic(self):
        assert classify_token("sk-ant-oat01-short") == "setup_token"

    def test_unknown_token(self):
        assert classify_token("some-random-value") == "unknown"

    def test_empty_string(self):
        assert classify_token("") == "unknown"

    def test_whitespace_stripped(self):
        key = "  sk-ant-api03-" + "A" * 85 + "  "
        assert classify_token(key) == "api_key"

    def test_partial_api_prefix(self):
        assert classify_token("sk-ant-api") == "api_key"

    def test_partial_oat_prefix(self):
        assert classify_token("sk-ant-oat") == "setup_token"

    def test_openai_key_unknown(self):
        assert classify_token("sk-proj-abc123def456") == "unknown"


# ---------------------------------------------------------------------------
# Token persistence
# ---------------------------------------------------------------------------


class TestStoreAndLoadToken:
    """Tests for store_token() and load_token()."""

    def test_store_and_load(self, tmp_path, monkeypatch):
        token_file = tmp_path / "secrets" / "anthropic-token.json"
        monkeypatch.setattr("missy.cli.anthropic_auth.TOKEN_FILE", token_file)
        store_token("sk-ant-api03-" + "A" * 85, "api_key")
        assert token_file.exists()
        data = load_token()
        assert data["token_type"] == "api_key"
        assert data["token"].startswith("sk-ant-api03-")
        assert "issued_at" in data

    def test_store_with_explicit_timestamp(self, tmp_path, monkeypatch):
        token_file = tmp_path / "secrets" / "anthropic-token.json"
        monkeypatch.setattr("missy.cli.anthropic_auth.TOKEN_FILE", token_file)
        store_token("tok", "setup_token", issued_at=1000)
        data = load_token()
        assert data["issued_at"] == 1000

    def test_store_file_permissions(self, tmp_path, monkeypatch):
        token_file = tmp_path / "secrets" / "anthropic-token.json"
        monkeypatch.setattr("missy.cli.anthropic_auth.TOKEN_FILE", token_file)
        store_token("tok", "api_key")
        mode = oct(token_file.stat().st_mode & 0o777)
        assert mode == "0o600"

    def test_store_creates_parent_dirs(self, tmp_path, monkeypatch):
        token_file = tmp_path / "deep" / "nested" / "anthropic-token.json"
        monkeypatch.setattr("missy.cli.anthropic_auth.TOKEN_FILE", token_file)
        store_token("tok", "api_key")
        assert token_file.exists()

    def test_load_nonexistent_returns_none(self, tmp_path, monkeypatch):
        token_file = tmp_path / "nope" / "anthropic-token.json"
        monkeypatch.setattr("missy.cli.anthropic_auth.TOKEN_FILE", token_file)
        assert load_token() is None

    def test_load_corrupt_json_returns_none(self, tmp_path, monkeypatch):
        token_file = tmp_path / "anthropic-token.json"
        token_file.write_text("not json at all {{{")
        monkeypatch.setattr("missy.cli.anthropic_auth.TOKEN_FILE", token_file)
        assert load_token() is None


# ---------------------------------------------------------------------------
# Token expiry
# ---------------------------------------------------------------------------


class TestTokenExpiry:
    """Tests for is_token_expiring()."""

    def test_api_key_never_expires(self):
        data = {"token_type": "api_key", "issued_at": 0}
        assert is_token_expiring(data) is False

    def test_setup_token_fresh_not_expiring(self):
        data = {"token_type": "setup_token", "issued_at": int(time.time())}
        assert is_token_expiring(data) is False

    def test_setup_token_near_expiry(self):
        # Issued long enough ago that remaining < REFRESH_WARN_MARGIN
        old_time = int(time.time()) - (SETUP_TOKEN_TTL_SECONDS - REFRESH_WARN_MARGIN + 60)
        data = {"token_type": "setup_token", "issued_at": old_time}
        assert is_token_expiring(data) is True

    def test_setup_token_expired(self):
        data = {"token_type": "setup_token", "issued_at": 0}
        assert is_token_expiring(data) is True

    def test_missing_issued_at_defaults_zero(self):
        data = {"token_type": "setup_token"}
        assert is_token_expiring(data) is True

    def test_missing_token_type(self):
        data = {"issued_at": 0}
        assert is_token_expiring(data) is False


# ---------------------------------------------------------------------------
# Runtime token resolution
# ---------------------------------------------------------------------------


class TestGetCurrentToken:
    """Tests for get_current_token()."""

    def test_env_var_takes_priority(self, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "env-key-123")
        assert get_current_token() == "env-key-123"

    def test_stored_token_used_when_no_env(self, tmp_path, monkeypatch):
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        token_file = tmp_path / "anthropic-token.json"
        token_file.write_text(
            json.dumps(
                {
                    "token_type": "api_key",
                    "token": "stored-key",
                    "issued_at": int(time.time()),
                }
            )
        )
        monkeypatch.setattr("missy.cli.anthropic_auth.TOKEN_FILE", token_file)
        assert get_current_token() == "stored-key"

    def test_vault_fallback(self, tmp_path, monkeypatch):
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        monkeypatch.setattr("missy.cli.anthropic_auth.TOKEN_FILE", tmp_path / "nope.json")
        mock_vault = MagicMock()
        mock_vault.return_value.get.return_value = "vault-key"
        with patch("missy.security.vault.Vault", mock_vault):
            result = get_current_token()
        assert result == "vault-key"

    def test_vault_failure_returns_none(self, tmp_path, monkeypatch):
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        monkeypatch.setattr("missy.cli.anthropic_auth.TOKEN_FILE", tmp_path / "nope.json")
        with patch("missy.security.vault.Vault", side_effect=Exception("no vault")):
            result = get_current_token()
        assert result is None

    def test_expiring_token_triggers_reminder(self, tmp_path, monkeypatch):
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        token_file = tmp_path / "anthropic-token.json"
        token_file.write_text(
            json.dumps(
                {
                    "token_type": "setup_token",
                    "token": "expiring-tok",
                    "issued_at": 0,  # Very old
                }
            )
        )
        monkeypatch.setattr("missy.cli.anthropic_auth.TOKEN_FILE", token_file)
        with patch("missy.cli.anthropic_auth.remind_refresh") as mock_remind:
            result = get_current_token()
        mock_remind.assert_called_once()
        assert result == "expiring-tok"

    def test_all_sources_fail_returns_none(self, tmp_path, monkeypatch):
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        monkeypatch.setattr("missy.cli.anthropic_auth.TOKEN_FILE", tmp_path / "nope.json")
        with patch("missy.security.vault.Vault", side_effect=ImportError):
            assert get_current_token() is None
