"""Coverage tests for missy/channels/discord/config.py.

Targets uncovered lines 122-129:
  resolve_token() — vault:// token resolution path, including the except branch
  when Vault raises an exception.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from missy.channels.discord.config import DiscordAccountConfig


class TestResolveToken:
    def test_direct_token_returned(self):
        """When token is set without vault:// prefix, return it directly."""
        cfg = DiscordAccountConfig(token="direct-token-abc")
        assert cfg.resolve_token() == "direct-token-abc"

    def test_no_token_falls_back_to_env_var(self, monkeypatch):
        """When token is None, fall back to environment variable."""
        monkeypatch.setenv("DISCORD_BOT_TOKEN", "env-token-xyz")
        cfg = DiscordAccountConfig(token=None, token_env_var="DISCORD_BOT_TOKEN")
        assert cfg.resolve_token() == "env-token-xyz"

    def test_no_token_env_var_missing_returns_none(self, monkeypatch):
        """When token is None and env var is unset, return None."""
        monkeypatch.delenv("DISCORD_TEST_MISSING_TOKEN", raising=False)
        cfg = DiscordAccountConfig(token=None, token_env_var="DISCORD_TEST_MISSING_TOKEN")
        assert cfg.resolve_token() is None

    def test_vault_token_resolved_successfully(self):
        """Lines 122-126: vault:// prefix triggers Vault lookup and returns value."""
        cfg = DiscordAccountConfig(token="vault://MY_DISCORD_TOKEN")

        mock_vault = MagicMock()
        mock_vault.return_value.get.return_value = "vault-secret-token"

        with patch.dict("sys.modules", {"missy.security.vault": MagicMock(Vault=mock_vault)}):
            result = cfg.resolve_token()

        mock_vault.return_value.get.assert_called_once_with("MY_DISCORD_TOKEN")
        assert result == "vault-secret-token"

    def test_vault_token_exception_returns_raw_vault_string(self):
        """Lines 127-128: when Vault raises, the except block passes and the
        outer 'return self.token' executes, returning the raw 'vault://' string."""
        cfg = DiscordAccountConfig(token="vault://BROKEN_KEY", token_env_var="DISCORD_MISSING_VAR")

        mock_vault_cls = MagicMock(side_effect=RuntimeError("vault not initialised"))

        with patch.dict("sys.modules", {"missy.security.vault": MagicMock(Vault=mock_vault_cls)}):
            result = cfg.resolve_token()

        # After the vault exception, falls through to `return self.token` at line 129.
        assert result == "vault://BROKEN_KEY"

    def test_vault_key_extracted_correctly(self):
        """Confirm the key name after 'vault://' is passed correctly to Vault."""
        cfg = DiscordAccountConfig(token="vault://MY_SECRET_KEY_NAME")

        mock_vault_inst = MagicMock()
        mock_vault_inst.get.return_value = "resolved-value"
        mock_vault_cls = MagicMock(return_value=mock_vault_inst)

        with patch.dict("sys.modules", {"missy.security.vault": MagicMock(Vault=mock_vault_cls)}):
            result = cfg.resolve_token()

        mock_vault_inst.get.assert_called_once_with("MY_SECRET_KEY_NAME")
        assert result == "resolved-value"
