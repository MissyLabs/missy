"""Deep tests for missy.cli.anthropic_auth — token classification, persistence,
expiry detection, vault integration, and runtime resolution.
"""

from __future__ import annotations

import json
import os
import stat
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from missy.cli.anthropic_auth import (
    TOKEN_FILE,
    SETUP_TOKEN_TTL_SECONDS,
    REFRESH_WARN_MARGIN,
    classify_token,
    store_token,
    load_token,
    is_token_expiring,
    get_current_token,
    _try_store_in_vault,
)


# =========================================================================
# classify_token
# =========================================================================


class TestClassifyToken:
    """Token classification: api_key, setup_token, or unknown."""

    def test_full_api_key(self):
        key = "sk-ant-api03-" + "A" * 90
        assert classify_token(key) == "api_key"

    def test_full_setup_token(self):
        token = "sk-ant-oat01-" + "B" * 70
        assert classify_token(token) == "setup_token"

    def test_api_key_prefix_heuristic(self):
        """Short/mangled tokens still classified by prefix."""
        assert classify_token("sk-ant-api-short") == "api_key"

    def test_setup_token_prefix_heuristic(self):
        assert classify_token("sk-ant-oat-short") == "setup_token"

    def test_unknown_token(self):
        assert classify_token("some-random-token") == "unknown"

    def test_empty_string(self):
        assert classify_token("") == "unknown"

    def test_whitespace_stripped(self):
        key = "  sk-ant-api03-" + "A" * 90 + "  "
        assert classify_token(key) == "api_key"

    def test_openai_key_is_unknown(self):
        assert classify_token("sk-proj-abc123") == "unknown"


# =========================================================================
# store_token + load_token
# =========================================================================


class TestTokenPersistence:
    """Token storage and retrieval round-trip."""

    @pytest.fixture(autouse=True)
    def _patch_token_file(self, tmp_path, monkeypatch):
        self.token_file = tmp_path / "secrets" / "anthropic-token.json"
        monkeypatch.setattr("missy.cli.anthropic_auth.TOKEN_FILE", self.token_file)

    def test_store_and_load_round_trip(self):
        store_token("sk-ant-api03-test", "api_key", issued_at=1000000)
        data = load_token()
        assert data is not None
        assert data["token"] == "sk-ant-api03-test"
        assert data["token_type"] == "api_key"
        assert data["issued_at"] == 1000000
        assert data["provider"] == "anthropic"

    def test_store_creates_parent_directory(self, tmp_path):
        # Token file parent should be created automatically
        assert not self.token_file.parent.exists()
        store_token("test-token", "setup_token")
        assert self.token_file.parent.exists()

    def test_store_uses_secure_permissions(self):
        store_token("secure-token", "api_key")
        mode = self.token_file.stat().st_mode & 0o777
        assert mode == 0o600

    def test_store_default_issued_at(self):
        before = int(time.time())
        store_token("test-token", "setup_token")
        after = int(time.time())
        data = load_token()
        assert before <= data["issued_at"] <= after

    def test_load_missing_file(self):
        assert load_token() is None

    def test_load_corrupt_file(self):
        self.token_file.parent.mkdir(parents=True, exist_ok=True)
        self.token_file.write_text("not valid json{{{", encoding="utf-8")
        assert load_token() is None

    def test_store_overwrites_existing(self):
        store_token("token-1", "api_key")
        store_token("token-2", "setup_token")
        data = load_token()
        assert data["token"] == "token-2"
        assert data["token_type"] == "setup_token"


# =========================================================================
# is_token_expiring
# =========================================================================


class TestIsTokenExpiring:
    """Setup-token expiry detection."""

    def test_api_key_never_expires(self):
        data = {"token_type": "api_key", "issued_at": 0}
        assert is_token_expiring(data) is False

    def test_fresh_setup_token_not_expiring(self):
        data = {"token_type": "setup_token", "issued_at": int(time.time())}
        assert is_token_expiring(data) is False

    def test_old_setup_token_is_expiring(self):
        # Issued 3 hours ago (past TTL)
        data = {
            "token_type": "setup_token",
            "issued_at": int(time.time()) - SETUP_TOKEN_TTL_SECONDS - 100,
        }
        assert is_token_expiring(data) is True

    def test_near_expiry_setup_token(self):
        # Issued so that remaining < REFRESH_WARN_MARGIN
        remaining = REFRESH_WARN_MARGIN - 60  # 60 seconds less than margin
        issued_at = int(time.time()) - (SETUP_TOKEN_TTL_SECONDS - remaining)
        data = {"token_type": "setup_token", "issued_at": issued_at}
        assert is_token_expiring(data) is True

    def test_just_outside_margin_not_expiring(self):
        # Issued so that remaining > REFRESH_WARN_MARGIN
        remaining = REFRESH_WARN_MARGIN + 120
        issued_at = int(time.time()) - (SETUP_TOKEN_TTL_SECONDS - remaining)
        data = {"token_type": "setup_token", "issued_at": issued_at}
        assert is_token_expiring(data) is False

    def test_missing_issued_at(self):
        data = {"token_type": "setup_token"}
        # issued_at defaults to 0, so age is huge → expiring
        assert is_token_expiring(data) is True

    def test_unknown_type(self):
        data = {"token_type": "unknown", "issued_at": 0}
        assert is_token_expiring(data) is False


# =========================================================================
# get_current_token
# =========================================================================


class TestGetCurrentToken:
    """Runtime token resolution with env var, file, and vault fallback."""

    @pytest.fixture(autouse=True)
    def _patch_token_file(self, tmp_path, monkeypatch):
        self.token_file = tmp_path / "secrets" / "anthropic-token.json"
        monkeypatch.setattr("missy.cli.anthropic_auth.TOKEN_FILE", self.token_file)
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    def test_env_var_takes_priority(self, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "env-key")
        store_token("file-key", "api_key")
        assert get_current_token() == "env-key"

    def test_file_token_used_when_no_env(self):
        store_token("file-key", "api_key")
        assert get_current_token() == "file-key"

    def test_no_token_returns_none(self):
        with patch("missy.cli.anthropic_auth.load_token", return_value=None):
            result = get_current_token(vault_dir="/nonexistent")
            # Vault will fail since /nonexistent doesn't exist
            assert result is None

    def test_expiring_token_still_returned(self):
        # Even if expiring, the token should be returned
        data = {
            "provider": "anthropic",
            "token_type": "setup_token",
            "token": "expiring-token",
            "issued_at": int(time.time()) - SETUP_TOKEN_TTL_SECONDS,
        }
        self.token_file.parent.mkdir(parents=True, exist_ok=True)
        self.token_file.write_text(json.dumps(data), encoding="utf-8")
        result = get_current_token()
        assert result == "expiring-token"

    def test_vault_fallback(self):
        mock_vault = MagicMock()
        mock_vault.get.return_value = "vault-key"
        with patch("missy.security.vault.Vault", return_value=mock_vault):
            result = get_current_token()
            assert result == "vault-key"

    def test_vault_failure_returns_none(self):
        with patch("missy.cli.anthropic_auth.load_token", return_value=None):
            with patch(
                "missy.security.vault.Vault", side_effect=Exception("vault error")
            ):
                result = get_current_token()
                assert result is None


# =========================================================================
# _try_store_in_vault
# =========================================================================


class TestTryStoreInVault:
    """Vault integration helper."""

    def test_success(self):
        mock_vault = MagicMock()
        with patch("missy.security.vault.Vault", return_value=mock_vault):
            result = _try_store_in_vault("my-api-key")
            assert result == "vault://anthropic_api_key"
            mock_vault.set.assert_called_once_with("anthropic_api_key", "my-api-key")

    def test_vault_error_returns_none(self):
        with patch("missy.security.vault.Vault", side_effect=Exception("no vault")):
            result = _try_store_in_vault("my-api-key")
            assert result is None


# =========================================================================
# Constants
# =========================================================================


class TestConstants:
    """Verify configuration constants."""

    def test_ttl_is_3_hours(self):
        assert SETUP_TOKEN_TTL_SECONDS == 3 * 3600

    def test_warn_margin_is_20_minutes(self):
        assert REFRESH_WARN_MARGIN == 20 * 60
