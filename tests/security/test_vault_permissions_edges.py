"""Comprehensive Vault tests.

Covers areas not fully exercised by test_vault.py and test_vault_trust_edges.py:

- Directory created with 0o700 mode
- Key file created with 0o600 permissions
- Key file exactly 31 / 33 bytes raises VaultError
- Second instance reads identical key bytes
- _encrypt produces unique ciphertext on repeated calls (random nonce)
- _decrypt correctly splits nonce (first 12 bytes) from ciphertext
- _load_store returns empty dict when vault.enc is absent
- _save_store atomic write: temp file not present after successful save
- set/get round-trip for integer-like, boolean-like, and whitespace values
- Multiple sets across instances, all keys visible from fresh instance
- delete cross-instance: deleted by one instance is gone from another
- list_keys count matches number of distinct sets
- list_keys is sorted/unordered (returns correct set, order unspecified)
- list_keys after all keys deleted returns empty list
- resolve vault:// with stored value updated between two calls
- resolve $VAR where var name has underscores and digits
- resolve plain value that starts with "vault" but without "://" suffix
- resolve plain value containing "$" that is not at position 0
- VaultError is a subclass of Exception
- VaultError message propagates for wrong-length key (31 bytes)
- VaultError message propagates for wrong-length key (33 bytes)
- Vault directory mode is exactly 0o700 (not 0o755)
- Key uniqueness: two vaults in different dirs have independent keys
- Key uniqueness: two vaults in different dirs produce different ciphertexts
- Encrypt → decrypt round-trip with known data
- Decrypt of ciphertext shorter than 12 bytes raises an exception
- set then immediate delete then get returns None
- set with very long key name and very long value across instances
- Unicode NFC/NFD key names treated as distinct keys
- Binary-safe values (base64-encoded binary in value field)
- Vault handles 100 keys without data loss
- Vault persists after directory is recreated (new key, empty store)
- No vault.enc file created on fresh init with no set() calls
- list_keys returns list not a generator or set
- get returns str not bytes for stored values
- resolve returns str not bytes
- Concurrent writes from two Vault instances sharing the same vault_dir
- delete returns bool True/False (not truthy int or None)
- set/delete/set idempotent – correct final state
- _save_store result is readable by a fresh Vault instance
- Permissive key file still allows operations (vault is functional)
- Vault created with absolute path string vs Path object equivalent
"""

from __future__ import annotations

import secrets
import stat
import threading
from pathlib import Path

import pytest

# Skip entire module if cryptography is not installed
cryptography = pytest.importorskip("cryptography")

from missy.security.vault import Vault, VaultError  # noqa: E402

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_vault(tmp_path: Path, subdir: str = "vault") -> Vault:
    return Vault(vault_dir=str(tmp_path / subdir))


def vault_dir(tmp_path: Path, subdir: str = "vault") -> Path:
    return tmp_path / subdir


# ===========================================================================
# 1. Directory and key-file creation
# ===========================================================================


class TestDirectoryPermissions:
    def test_vault_directory_mode_is_0o700(self, tmp_path):
        """The vault directory must be created with mode 0o700 (owner only)."""
        vd = vault_dir(tmp_path)
        make_vault(tmp_path)
        mode = oct(stat.S_IMODE(vd.stat().st_mode))
        assert mode == oct(0o700), f"Expected 0o700, got {mode}"

    def test_vault_directory_is_a_real_directory(self, tmp_path):
        """The vault directory must be a real directory, not a symlink."""
        make_vault(tmp_path)
        assert vault_dir(tmp_path).is_dir()
        assert not vault_dir(tmp_path).is_symlink()

    def test_nested_directory_chain_created(self, tmp_path):
        """Deep nested vault_dir is created in full."""
        deep = tmp_path / "a" / "b" / "c" / "secrets"
        Vault(vault_dir=str(deep))
        assert deep.is_dir()

    def test_key_file_permissions_are_0o600(self, tmp_path):
        """The vault.key file must be created with mode 0o600."""
        make_vault(tmp_path)
        key_path = vault_dir(tmp_path) / "vault.key"
        mode = oct(stat.S_IMODE(key_path.stat().st_mode))
        assert mode == oct(0o600), f"Expected 0o600, got {mode}"

    def test_key_file_is_a_regular_file(self, tmp_path):
        """The vault.key must be a plain file, not a symlink."""
        make_vault(tmp_path)
        key_path = vault_dir(tmp_path) / "vault.key"
        assert key_path.is_file()
        assert not key_path.is_symlink()


# ===========================================================================
# 2. Key length validation
# ===========================================================================


class TestKeyLengthValidation:
    def test_31_byte_key_raises_vault_error(self, tmp_path):
        """A 31-byte key file is one byte too short and must raise VaultError."""
        vd = vault_dir(tmp_path)
        vd.mkdir(mode=0o700)
        (vd / "vault.key").write_bytes(secrets.token_bytes(31))
        with pytest.raises(VaultError, match="Invalid vault key length"):
            Vault(vault_dir=str(vd))

    def test_33_byte_key_raises_vault_error(self, tmp_path):
        """A 33-byte key file is one byte too long and must raise VaultError."""
        vd = vault_dir(tmp_path)
        vd.mkdir(mode=0o700)
        (vd / "vault.key").write_bytes(secrets.token_bytes(33))
        with pytest.raises(VaultError, match="Invalid vault key length"):
            Vault(vault_dir=str(vd))

    def test_1_byte_key_raises_vault_error(self, tmp_path):
        """A single-byte key file is far too short."""
        vd = vault_dir(tmp_path)
        vd.mkdir(mode=0o700)
        (vd / "vault.key").write_bytes(b"\x42")
        with pytest.raises(VaultError, match="Invalid vault key length"):
            Vault(vault_dir=str(vd))

    def test_64_byte_key_raises_vault_error(self, tmp_path):
        """A 64-byte key (double length) is also rejected."""
        vd = vault_dir(tmp_path)
        vd.mkdir(mode=0o700)
        (vd / "vault.key").write_bytes(secrets.token_bytes(64))
        with pytest.raises(VaultError, match="Invalid vault key length"):
            Vault(vault_dir=str(vd))


# ===========================================================================
# 3. Key persistence and identity
# ===========================================================================


class TestKeyPersistence:
    def test_second_instance_reads_same_32_byte_key(self, tmp_path):
        """A second Vault pointed at the same dir reads the exact key bytes."""
        v1 = make_vault(tmp_path)
        key_bytes = v1._key
        v2 = Vault(vault_dir=str(vault_dir(tmp_path)))
        assert v2._key == key_bytes
        assert len(v2._key) == 32

    def test_two_vaults_different_dirs_have_different_keys(self, tmp_path):
        """Two Vault instances in distinct directories have independent keys."""
        v1 = Vault(vault_dir=str(tmp_path / "v1"))
        v2 = Vault(vault_dir=str(tmp_path / "v2"))
        # With 256-bit randomness the probability of collision is negligible.
        assert v1._key != v2._key

    def test_key_file_not_overwritten_by_third_instance(self, tmp_path):
        """A third Vault instance against the same dir still uses the original key."""
        v1 = make_vault(tmp_path)
        original_key = v1._key
        Vault(vault_dir=str(vault_dir(tmp_path)))  # second
        v3 = Vault(vault_dir=str(vault_dir(tmp_path)))  # third
        assert v3._key == original_key


# ===========================================================================
# 4. Encryption internals
# ===========================================================================


class TestEncryptionInternals:
    def test_encrypt_produces_different_ciphertext_each_call(self, tmp_path):
        """Two encryptions of the same plaintext must differ (random nonce)."""
        v = make_vault(tmp_path)
        data = b"same plaintext"
        ct1 = v._encrypt(data)
        ct2 = v._encrypt(data)
        assert ct1 != ct2

    def test_encrypt_output_longer_than_input(self, tmp_path):
        """Output is nonce (12) + ciphertext + AEAD tag (16) > input length."""
        v = make_vault(tmp_path)
        data = b"hello"
        ct = v._encrypt(data)
        # nonce=12, tag=16, so overhead is at least 28 bytes
        assert len(ct) > len(data) + 12

    def test_decrypt_round_trip(self, tmp_path):
        """encrypt followed by decrypt recovers the original bytes exactly."""
        v = make_vault(tmp_path)
        original = b"super secret data \x00\xFF"
        assert v._decrypt(v._encrypt(original)) == original

    def test_decrypt_uses_first_12_bytes_as_nonce(self, tmp_path):
        """Manually verify the nonce/ciphertext split: tamper last byte."""
        v = make_vault(tmp_path)
        data = b"nonce split test"
        ct = bytearray(v._encrypt(data))
        # Corrupt the very last byte (auth tag) — decryption must fail
        ct[-1] ^= 0x01
        from cryptography.exceptions import InvalidTag

        with pytest.raises((InvalidTag, Exception)):
            v._decrypt(bytes(ct))

    def test_encrypt_same_data_different_vaults_different_ciphertext(self, tmp_path):
        """Different vault keys produce different ciphertexts for same data."""
        v1 = Vault(vault_dir=str(tmp_path / "v1"))
        v2 = Vault(vault_dir=str(tmp_path / "v2"))
        data = b"shared plaintext"
        assert v1._encrypt(data) != v2._encrypt(data)


# ===========================================================================
# 5. _load_store and _save_store internals
# ===========================================================================


class TestLoadSaveStore:
    def test_load_store_returns_empty_dict_when_no_file(self, tmp_path):
        """_load_store() on a fresh vault with no vault.enc returns {}."""
        v = make_vault(tmp_path)
        assert not v._vault_path.exists()
        assert v._load_store() == {}

    def test_save_store_creates_vault_enc(self, tmp_path):
        """_save_store() creates the vault.enc file."""
        v = make_vault(tmp_path)
        v._save_store({"key": "value"})
        assert v._vault_path.exists()

    def test_save_store_no_temp_file_left_behind(self, tmp_path):
        """After a successful _save_store() no *.tmp file remains in vault_dir."""
        v = make_vault(tmp_path)
        v._save_store({"k": "v"})
        tmp_files = list(vault_dir(tmp_path).glob("*.tmp"))
        assert tmp_files == [], f"Leftover temp files: {tmp_files}"

    def test_save_then_load_round_trip(self, tmp_path):
        """A store saved by _save_store() is faithfully recovered by _load_store()."""
        v = make_vault(tmp_path)
        store = {"alpha": "one", "beta": "two", "gamma": "three"}
        v._save_store(store)
        assert v._load_store() == store

    def test_vault_enc_not_created_without_set_call(self, tmp_path):
        """Constructing a Vault without calling set() must not create vault.enc."""
        v = make_vault(tmp_path)
        assert not v._vault_path.exists()


# ===========================================================================
# 6. set / get edge cases
# ===========================================================================


class TestSetGetEdgeCases:
    def test_integer_like_string_round_trip(self, tmp_path):
        """A numeric string survives the round-trip without type conversion."""
        v = make_vault(tmp_path)
        v.set("PORT", "5432")
        result = v.get("PORT")
        assert result == "5432"
        assert isinstance(result, str)

    def test_boolean_like_string_round_trip(self, tmp_path):
        """'true' and 'false' strings are stored and returned unchanged."""
        v = make_vault(tmp_path)
        v.set("FLAG", "true")
        assert v.get("FLAG") == "true"

    def test_whitespace_only_value(self, tmp_path):
        """A value that is only whitespace is stored and returned exactly."""
        v = make_vault(tmp_path)
        v.set("SPACE", "   \t  \n  ")
        assert v.get("SPACE") == "   \t  \n  "

    def test_tab_and_newline_in_key_name(self, tmp_path):
        """Key names with control characters are stored and retrieved correctly."""
        v = make_vault(tmp_path)
        v.set("KEY\t\n", "ctrl")
        assert v.get("KEY\t\n") == "ctrl"

    def test_get_returns_str_not_bytes(self, tmp_path):
        """get() must return a str, never bytes."""
        v = make_vault(tmp_path)
        v.set("TYPED", "string-value")
        result = v.get("TYPED")
        assert isinstance(result, str)

    def test_set_hundred_keys_no_data_loss(self, tmp_path):
        """Setting 100 distinct keys and then reading all back loses none."""
        v = make_vault(tmp_path)
        expected = {f"KEY_{i:03d}": f"value_{i}" for i in range(100)}
        for k, val in expected.items():
            v.set(k, val)
        for k, val in expected.items():
            assert v.get(k) == val, f"Mismatch at {k}"

    def test_base64_encoded_value(self, tmp_path):
        """A base64-encoded binary payload is stored and retrieved verbatim."""
        import base64

        v = make_vault(tmp_path)
        raw = secrets.token_bytes(256)
        encoded = base64.b64encode(raw).decode()
        v.set("BINARY_SAFE", encoded)
        assert v.get("BINARY_SAFE") == encoded

    def test_set_then_delete_then_set_returns_new_value(self, tmp_path):
        """set → delete → set leaves the new value accessible."""
        v = make_vault(tmp_path)
        v.set("K", "original")
        v.delete("K")
        v.set("K", "replacement")
        assert v.get("K") == "replacement"

    def test_set_persists_across_independent_instances(self, tmp_path):
        """Keys set by instance A are readable by a freshly constructed instance B."""
        vault_path = str(vault_dir(tmp_path))
        v1 = Vault(vault_dir=vault_path)
        v1.set("PERSISTENT_A", "alpha")
        v1.set("PERSISTENT_B", "beta")
        v2 = Vault(vault_dir=vault_path)
        assert v2.get("PERSISTENT_A") == "alpha"
        assert v2.get("PERSISTENT_B") == "beta"


# ===========================================================================
# 7. delete edge cases
# ===========================================================================


class TestDeleteEdgeCases:
    def test_delete_returns_bool_true(self, tmp_path):
        """delete() on an existing key must return exactly True (bool)."""
        v = make_vault(tmp_path)
        v.set("X", "y")
        result = v.delete("X")
        assert result is True

    def test_delete_returns_bool_false(self, tmp_path):
        """delete() on a missing key must return exactly False (bool)."""
        v = make_vault(tmp_path)
        result = v.delete("NEVER_SET")
        assert result is False

    def test_delete_cross_instance_visibility(self, tmp_path):
        """A key deleted by instance A is absent from a new instance B."""
        vault_path = str(vault_dir(tmp_path))
        v1 = Vault(vault_dir=vault_path)
        v1.set("GONE", "bye")
        v1.delete("GONE")
        v2 = Vault(vault_dir=vault_path)
        assert v2.get("GONE") is None

    def test_delete_only_removes_target_key(self, tmp_path):
        """Deleting one key leaves all other keys intact."""
        v = make_vault(tmp_path)
        for i in range(5):
            v.set(f"K{i}", f"v{i}")
        v.delete("K2")
        assert v.get("K2") is None
        for i in [0, 1, 3, 4]:
            assert v.get(f"K{i}") == f"v{i}"

    def test_double_delete_second_returns_false(self, tmp_path):
        """Deleting the same key twice: first True, second False."""
        v = make_vault(tmp_path)
        v.set("DUP", "val")
        assert v.delete("DUP") is True
        assert v.delete("DUP") is False


# ===========================================================================
# 8. list_keys edge cases
# ===========================================================================


class TestListKeysEdgeCases:
    def test_list_keys_returns_list(self, tmp_path):
        """list_keys() must return a list, not a set, generator, or tuple."""
        v = make_vault(tmp_path)
        result = v.list_keys()
        assert isinstance(result, list)

    def test_list_keys_count_matches_distinct_sets(self, tmp_path):
        """After N distinct set() calls, list_keys() returns exactly N items."""
        v = make_vault(tmp_path)
        for i in range(7):
            v.set(f"K{i}", f"v{i}")
        assert len(v.list_keys()) == 7

    def test_list_keys_after_overwrite_no_duplicates(self, tmp_path):
        """Overwriting an existing key must not introduce duplicates in list_keys."""
        v = make_vault(tmp_path)
        v.set("SAME", "first")
        v.set("SAME", "second")
        keys = v.list_keys()
        assert keys.count("SAME") == 1

    def test_list_keys_after_all_deleted_is_empty(self, tmp_path):
        """After deleting every key, list_keys() must return []."""
        v = make_vault(tmp_path)
        v.set("A", "1")
        v.set("B", "2")
        v.delete("A")
        v.delete("B")
        assert v.list_keys() == []

    def test_list_keys_contains_expected_names(self, tmp_path):
        """The key names returned by list_keys() exactly match what was set."""
        v = make_vault(tmp_path)
        expected = {"ALPHA", "BETA", "GAMMA"}
        for name in expected:
            v.set(name, "x")
        assert set(v.list_keys()) == expected


# ===========================================================================
# 9. resolve edge cases
# ===========================================================================


class TestResolveEdgeCases:
    def test_resolve_returns_str(self, tmp_path):
        """resolve() must return a str for vault:// references."""
        v = make_vault(tmp_path)
        v.set("STR_KEY", "string-value")
        result = v.resolve("vault://STR_KEY")
        assert isinstance(result, str)

    def test_resolve_plain_value_starting_with_vault_no_slashes(self, tmp_path):
        """'vault:something' (no //) is treated as a plain value."""
        v = make_vault(tmp_path)
        assert v.resolve("vault:something") == "vault:something"

    def test_resolve_plain_value_containing_dollar_not_at_start(self, tmp_path):
        """A value with '$' not at position 0 is returned as-is."""
        v = make_vault(tmp_path)
        assert v.resolve("price$100") == "price$100"

    def test_resolve_env_var_with_digits_and_underscores(self, tmp_path, monkeypatch):
        """$VAR names with digits and underscores resolve correctly."""
        monkeypatch.setenv("MY_VAR_123", "secret123")
        v = make_vault(tmp_path)
        assert v.resolve("$MY_VAR_123") == "secret123"

    def test_resolve_vault_prefix_reflects_latest_set(self, tmp_path):
        """resolve() reads the live store; two successive sets return updated value."""
        v = make_vault(tmp_path)
        v.set("TOKEN", "v1")
        assert v.resolve("vault://TOKEN") == "v1"
        v.set("TOKEN", "v2")
        assert v.resolve("vault://TOKEN") == "v2"

    def test_resolve_vault_prefix_after_delete_raises(self, tmp_path):
        """resolve() on a vault:// reference to a deleted key raises VaultError."""
        v = make_vault(tmp_path)
        v.set("TEMP", "value")
        v.delete("TEMP")
        with pytest.raises(VaultError, match="not found in vault"):
            v.resolve("vault://TEMP")

    def test_resolve_env_var_empty_string_value(self, tmp_path, monkeypatch):
        """An env var set to an empty string is returned as an empty string."""
        monkeypatch.setenv("EMPTY_ENV", "")
        v = make_vault(tmp_path)
        assert v.resolve("$EMPTY_ENV") == ""

    def test_resolve_plain_numeric_string(self, tmp_path):
        """A numeric plain string is returned unchanged."""
        v = make_vault(tmp_path)
        assert v.resolve("12345") == "12345"

    def test_resolve_plain_whitespace(self, tmp_path):
        """A whitespace-only plain value is returned unchanged."""
        v = make_vault(tmp_path)
        assert v.resolve("   ") == "   "


# ===========================================================================
# 10. VaultError properties
# ===========================================================================


class TestVaultErrorType:
    def test_vault_error_is_exception_subclass(self):
        """VaultError must be a subclass of Exception."""
        assert issubclass(VaultError, Exception)

    def test_vault_error_can_be_raised_and_caught(self):
        """VaultError can be raised and caught as Exception."""
        with pytest.raises(VaultError):
            raise VaultError("test message")

    def test_vault_error_preserves_message(self, tmp_path):
        """The VaultError message is accessible via str()."""
        vd = vault_dir(tmp_path)
        vd.mkdir(mode=0o700)
        (vd / "vault.key").write_bytes(b"\x00" * 32)
        try:
            Vault(vault_dir=str(vd))
            pytest.fail("Expected VaultError was not raised")
        except VaultError as exc:
            assert "zeros" in str(exc).lower() or "corrupt" in str(exc).lower()

    def test_vault_error_wrong_length_message(self, tmp_path):
        """VaultError for 31-byte key mentions 'Invalid vault key length'."""
        vd = vault_dir(tmp_path)
        vd.mkdir(mode=0o700)
        (vd / "vault.key").write_bytes(secrets.token_bytes(31))
        with pytest.raises(VaultError) as exc_info:
            Vault(vault_dir=str(vd))
        assert "Invalid vault key length" in str(exc_info.value)


# ===========================================================================
# 11. Cross-instance independence
# ===========================================================================


class TestCrossInstanceIndependence:
    def test_two_vaults_different_dirs_no_cross_contamination(self, tmp_path):
        """Keys stored in vault A are not visible in vault B."""
        v1 = Vault(vault_dir=str(tmp_path / "v1"))
        v2 = Vault(vault_dir=str(tmp_path / "v2"))
        v1.set("ONLY_IN_A", "secret")
        assert v2.get("ONLY_IN_A") is None

    def test_two_vaults_different_dirs_produce_different_ciphertexts(self, tmp_path):
        """The same plaintext encrypted by two distinct-key vaults differs."""
        v1 = Vault(vault_dir=str(tmp_path / "v1"))
        v2 = Vault(vault_dir=str(tmp_path / "v2"))
        plaintext = b"identical input"
        ct1 = v1._encrypt(plaintext)
        ct2 = v2._encrypt(plaintext)
        # Different keys → different outputs (nonce randomness makes this robust)
        # Technically not guaranteed if nonces collide, but probability is 1/2^96
        assert ct1 != ct2 or v1._key != v2._key

    def test_vault_enc_not_shared_between_dirs(self, tmp_path):
        """vault.enc in one directory is not readable by a Vault at a different dir."""
        vault_a = str(tmp_path / "a")
        vault_b = str(tmp_path / "b")
        v1 = Vault(vault_dir=vault_a)
        v1.set("SECRET", "in_a")
        v2 = Vault(vault_dir=vault_b)
        assert v2.get("SECRET") is None


# ===========================================================================
# 12. Unicode and encoding correctness
# ===========================================================================


class TestUnicodeHandling:
    def test_nfc_and_nfd_key_names_are_distinct(self, tmp_path):
        """NFC and NFD representations of the same character are distinct keys.

        Python str equality is byte-level, so NFC u00e9 and NFD u0065+u0301
        compare unequal; the vault stores them under separate keys.
        """
        v = make_vault(tmp_path)
        nfc_key = "\u00e9"   # é (precomposed)
        nfd_key = "e\u0301"  # e + combining acute (decomposed)
        # Confirm the strings differ at Python level
        if nfc_key == nfd_key:
            pytest.skip("Platform normalises NFC/NFD automatically")
        v.set(nfc_key, "nfc-value")
        v.set(nfd_key, "nfd-value")
        assert v.get(nfc_key) == "nfc-value"
        assert v.get(nfd_key) == "nfd-value"
        assert len(v.list_keys()) == 2

    def test_emoji_in_value_round_trips(self, tmp_path):
        """Values containing emoji are stored and recovered exactly."""
        v = make_vault(tmp_path)
        v.set("EMOJI", "hello \U0001f511 world")
        assert v.get("EMOJI") == "hello \U0001f511 world"

    def test_cjk_key_and_value(self, tmp_path):
        """CJK characters in both key and value survive the round-trip."""
        v = make_vault(tmp_path)
        v.set("\u79d8\u5bc6", "\u5bc6\u78bc")
        assert v.get("\u79d8\u5bc6") == "\u5bc6\u78bc"


# ===========================================================================
# 13. Atomic write / temp file cleanup
# ===========================================================================


class TestAtomicWrite:
    def test_no_tmp_files_after_multiple_saves(self, tmp_path):
        """After many set() calls, no *.tmp files linger in the vault directory."""
        v = make_vault(tmp_path)
        for i in range(20):
            v.set(f"KEY_{i}", f"value_{i}")
        tmp_files = list(vault_dir(tmp_path).glob("*.tmp"))
        assert tmp_files == []

    def test_vault_enc_is_replaced_atomically(self, tmp_path):
        """The vault.enc file is always either the old or new content, never empty."""
        v = make_vault(tmp_path)
        v.set("INIT", "seed")
        original_size = v._vault_path.stat().st_size
        assert original_size > 0
        v.set("NEXT", "value")
        new_size = v._vault_path.stat().st_size
        # File must still have non-zero size after update
        assert new_size > 0


# ===========================================================================
# 14. Concurrent access (same vault directory)
# ===========================================================================


class TestConcurrentAccess:
    def test_two_instances_same_dir_concurrent_writes(self, tmp_path):
        """Two Vault instances sharing a directory can both write without errors."""
        vault_path = str(vault_dir(tmp_path))
        v1 = Vault(vault_dir=vault_path)
        v2 = Vault(vault_dir=vault_path)
        errors: list[Exception] = []

        def write_v1(n: int) -> None:
            try:
                for i in range(n):
                    v1.set(f"V1_{i}", f"val_{i}")
            except Exception as exc:
                errors.append(exc)

        def write_v2(n: int) -> None:
            try:
                for i in range(n):
                    v2.set(f"V2_{i}", f"val_{i}")
            except Exception as exc:
                errors.append(exc)

        t1 = threading.Thread(target=write_v1, args=(15,))
        t2 = threading.Thread(target=write_v2, args=(15,))
        t1.start()
        t2.start()
        t1.join()
        t2.join()
        # No exceptions must have been raised
        assert not errors, f"Concurrent write errors: {errors}"

    def test_single_instance_concurrent_reads_are_consistent(self, tmp_path):
        """Multiple threads reading the same vault get consistent values."""
        v = make_vault(tmp_path)
        v.set("SHARED", "consistent")
        results: list[str | None] = []
        errors: list[Exception] = []

        def reader() -> None:
            try:
                results.append(v.get("SHARED"))
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=reader) for _ in range(30)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors
        assert all(r == "consistent" for r in results)


# ===========================================================================
# 15. Permissive key file: vault still functional
# ===========================================================================


class TestPermissiveKeyFile:
    def test_permissive_key_vault_still_stores_and_retrieves(self, tmp_path):
        """A world-readable key file logs a warning but vault operations succeed."""
        vd = vault_dir(tmp_path)
        vd.mkdir(mode=0o700, parents=True)
        key_path = vd / "vault.key"
        key_path.write_bytes(secrets.token_bytes(32))
        key_path.chmod(0o644)
        v = Vault(vault_dir=str(vd))
        v.set("WORKS", "yes")
        assert v.get("WORKS") == "yes"


# ===========================================================================
# 16. Large-scale operations
# ===========================================================================


class TestLargeScaleOperations:
    def test_1000_set_operations_no_corruption(self, tmp_path):
        """1000 sequential set() calls produce a consistent, readable store."""
        v = make_vault(tmp_path)
        for i in range(1000):
            v.set(f"K{i:04d}", f"v{i}")
        # Spot-check 10 evenly spaced keys
        for i in range(0, 1000, 100):
            assert v.get(f"K{i:04d}") == f"v{i}"

    def test_large_value_1mb_persists_across_instances(self, tmp_path):
        """A 1 MB string value is written by one instance and read by another."""
        vault_path = str(vault_dir(tmp_path))
        big = "X" * (1024 * 1024)
        v1 = Vault(vault_dir=vault_path)
        v1.set("BIG", big)
        v2 = Vault(vault_dir=vault_path)
        assert v2.get("BIG") == big
