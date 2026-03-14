"""Tests for missy.security.vault.Vault and VaultError."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from missy.security.vault import Vault, VaultError

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_vault(tmp_path) -> Vault:
    return Vault(vault_dir=str(tmp_path / "vault"))


# ---------------------------------------------------------------------------
# Initialisation
# ---------------------------------------------------------------------------


class TestVaultInit:
    def test_creates_vault_directory(self, tmp_path):
        vault_dir = tmp_path / "vault"
        assert not vault_dir.exists()
        Vault(vault_dir=str(vault_dir))
        assert vault_dir.is_dir()

    def test_creates_key_file(self, tmp_path):
        vault = make_vault(tmp_path)
        assert vault._key_path.exists()

    def test_key_file_is_32_bytes(self, tmp_path):
        make_vault(tmp_path)
        key_path = tmp_path / "vault" / "vault.key"
        assert len(key_path.read_bytes()) == 32

    def test_second_init_reuses_existing_key(self, tmp_path):
        v1 = make_vault(tmp_path)
        key_first = v1._key
        v2 = Vault(vault_dir=str(tmp_path / "vault"))
        assert v2._key == key_first

    def test_nested_vault_dir_created(self, tmp_path):
        deep_dir = tmp_path / "a" / "b" / "c"
        Vault(vault_dir=str(deep_dir))
        assert deep_dir.is_dir()

    def test_crypto_unavailable_raises_vault_error(self, tmp_path):
        with (
            patch("missy.security.vault._CRYPTO_AVAILABLE", False),
            pytest.raises(VaultError, match="cryptography package"),
        ):
            Vault(vault_dir=str(tmp_path / "vault"))


# ---------------------------------------------------------------------------
# Invalid key file
# ---------------------------------------------------------------------------


class TestInvalidKeyFile:
    def test_wrong_key_length_raises_vault_error(self, tmp_path):
        vault_dir = tmp_path / "vault"
        vault_dir.mkdir()
        key_path = vault_dir / "vault.key"
        key_path.write_bytes(b"tooshort")
        with pytest.raises(VaultError, match="Invalid vault key length"):
            Vault(vault_dir=str(vault_dir))

    def test_zero_byte_key_raises_vault_error(self, tmp_path):
        vault_dir = tmp_path / "vault"
        vault_dir.mkdir()
        (vault_dir / "vault.key").write_bytes(b"")
        with pytest.raises(VaultError, match="Invalid vault key length"):
            Vault(vault_dir=str(vault_dir))


# ---------------------------------------------------------------------------
# set / get round-trip
# ---------------------------------------------------------------------------


class TestSetGet:
    def test_set_and_get_round_trip(self, tmp_path):
        vault = make_vault(tmp_path)
        vault.set("MY_KEY", "super-secret-value")
        assert vault.get("MY_KEY") == "super-secret-value"

    def test_get_missing_key_returns_none(self, tmp_path):
        vault = make_vault(tmp_path)
        assert vault.get("DOES_NOT_EXIST") is None

    def test_set_overwrites_existing_key(self, tmp_path):
        vault = make_vault(tmp_path)
        vault.set("K", "first")
        vault.set("K", "second")
        assert vault.get("K") == "second"

    def test_multiple_keys_stored_independently(self, tmp_path):
        vault = make_vault(tmp_path)
        vault.set("A", "alpha")
        vault.set("B", "beta")
        assert vault.get("A") == "alpha"
        assert vault.get("B") == "beta"

    def test_value_survives_reinitialisation(self, tmp_path):
        vault_dir = str(tmp_path / "vault")
        v1 = Vault(vault_dir=vault_dir)
        v1.set("PERSIST", "yes")
        v2 = Vault(vault_dir=vault_dir)
        assert v2.get("PERSIST") == "yes"

    def test_empty_string_value_stored(self, tmp_path):
        vault = make_vault(tmp_path)
        vault.set("EMPTY", "")
        assert vault.get("EMPTY") == ""

    def test_value_with_special_characters(self, tmp_path):
        vault = make_vault(tmp_path)
        special = "sk-abc!@#$%^&*()_+-={}[]|;':\",./<>?"
        vault.set("SPECIAL", special)
        assert vault.get("SPECIAL") == special


# ---------------------------------------------------------------------------
# delete
# ---------------------------------------------------------------------------


class TestDelete:
    def test_delete_existing_key_returns_true(self, tmp_path):
        vault = make_vault(tmp_path)
        vault.set("DEL_ME", "value")
        assert vault.delete("DEL_ME") is True

    def test_deleted_key_no_longer_retrievable(self, tmp_path):
        vault = make_vault(tmp_path)
        vault.set("DEL_ME", "value")
        vault.delete("DEL_ME")
        assert vault.get("DEL_ME") is None

    def test_delete_nonexistent_key_returns_false(self, tmp_path):
        vault = make_vault(tmp_path)
        assert vault.delete("NO_SUCH_KEY") is False

    def test_delete_leaves_other_keys_intact(self, tmp_path):
        vault = make_vault(tmp_path)
        vault.set("KEEP", "safe")
        vault.set("REMOVE", "gone")
        vault.delete("REMOVE")
        assert vault.get("KEEP") == "safe"

    def test_delete_then_reset_key(self, tmp_path):
        vault = make_vault(tmp_path)
        vault.set("K", "v1")
        vault.delete("K")
        vault.set("K", "v2")
        assert vault.get("K") == "v2"


# ---------------------------------------------------------------------------
# list_keys
# ---------------------------------------------------------------------------


class TestListKeys:
    def test_empty_vault_returns_empty_list(self, tmp_path):
        vault = make_vault(tmp_path)
        assert vault.list_keys() == []

    def test_lists_stored_key_names(self, tmp_path):
        vault = make_vault(tmp_path)
        vault.set("FOO", "1")
        vault.set("BAR", "2")
        keys = vault.list_keys()
        assert set(keys) == {"FOO", "BAR"}

    def test_deleted_key_not_in_list(self, tmp_path):
        vault = make_vault(tmp_path)
        vault.set("GONE", "x")
        vault.set("KEEP", "y")
        vault.delete("GONE")
        assert "GONE" not in vault.list_keys()
        assert "KEEP" in vault.list_keys()

    def test_list_keys_does_not_return_values(self, tmp_path):
        vault = make_vault(tmp_path)
        vault.set("MY_KEY", "secret-value")
        assert "secret-value" not in vault.list_keys()


# ---------------------------------------------------------------------------
# resolve
# ---------------------------------------------------------------------------


class TestResolveVaultPrefix:
    def test_vault_prefix_returns_stored_value(self, tmp_path):
        vault = make_vault(tmp_path)
        vault.set("DB_PASS", "hunter2")
        assert vault.resolve("vault://DB_PASS") == "hunter2"

    def test_vault_prefix_missing_key_raises(self, tmp_path):
        vault = make_vault(tmp_path)
        with pytest.raises(VaultError, match="not found in vault"):
            vault.resolve("vault://MISSING_KEY")

    def test_vault_prefix_error_includes_key_name(self, tmp_path):
        vault = make_vault(tmp_path)
        with pytest.raises(VaultError, match="SPECIFIC_KEY"):
            vault.resolve("vault://SPECIFIC_KEY")


class TestResolveEnvPrefix:
    def test_env_prefix_returns_env_var(self, tmp_path, monkeypatch):
        monkeypatch.setenv("MY_SECRET", "from-env")
        vault = make_vault(tmp_path)
        assert vault.resolve("$MY_SECRET") == "from-env"

    def test_env_prefix_missing_var_raises(self, tmp_path, monkeypatch):
        monkeypatch.delenv("NONEXISTENT_VAR", raising=False)
        vault = make_vault(tmp_path)
        with pytest.raises(VaultError, match="Environment variable"):
            vault.resolve("$NONEXISTENT_VAR")

    def test_env_prefix_error_includes_var_name(self, tmp_path, monkeypatch):
        monkeypatch.delenv("SPECIFIC_ENV_VAR", raising=False)
        vault = make_vault(tmp_path)
        with pytest.raises(VaultError, match="SPECIFIC_ENV_VAR"):
            vault.resolve("$SPECIFIC_ENV_VAR")


class TestResolvePlainValue:
    def test_plain_string_returned_as_is(self, tmp_path):
        vault = make_vault(tmp_path)
        assert vault.resolve("plain-api-key-value") == "plain-api-key-value"

    def test_empty_string_returned_as_is(self, tmp_path):
        vault = make_vault(tmp_path)
        assert vault.resolve("") == ""

    def test_url_like_string_without_vault_prefix_returned_as_is(self, tmp_path):
        vault = make_vault(tmp_path)
        assert vault.resolve("https://example.com") == "https://example.com"


# ---------------------------------------------------------------------------
# Decryption failure (corrupt vault file)
# ---------------------------------------------------------------------------


class TestDecryptionFailure:
    def test_corrupt_vault_file_raises_vault_error(self, tmp_path):
        vault = make_vault(tmp_path)
        vault.set("K", "v")
        # Overwrite the encrypted file with garbage
        vault._vault_path.write_bytes(b"\x00" * 64)
        with pytest.raises(VaultError, match="Cannot decrypt vault"):
            vault.get("K")

    def test_truncated_vault_file_raises_vault_error(self, tmp_path):
        vault = make_vault(tmp_path)
        vault.set("K", "v")
        vault._vault_path.write_bytes(b"\xde\xad\xbe\xef")
        with pytest.raises(VaultError, match="Cannot decrypt vault"):
            vault.list_keys()

    def test_wrong_key_cannot_decrypt(self, tmp_path):
        vault_dir = str(tmp_path / "vault")
        v1 = Vault(vault_dir=vault_dir)
        v1.set("SECRET", "value")
        # Replace the key file with a fresh random key
        import secrets as _secrets

        v1._key_path.write_bytes(_secrets.token_bytes(32))
        v2 = Vault(vault_dir=vault_dir)
        with pytest.raises(VaultError, match="Cannot decrypt vault"):
            v2.get("SECRET")
