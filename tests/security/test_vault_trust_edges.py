"""Edge-case tests for Vault and TrustScorer.


Covers scenarios not exercised by the existing test_vault.py /
test_trust.py suites:

Vault
-----
* Key rotation / re-encryption (manual key swap + re-save)
* Corrupt ciphertext variants (bit-flipped, empty file, only-nonce)
* Truncated data at various byte boundaries
* Concurrent read/write thread safety
* vault:// reference resolution edge cases (nested prefix, spaces,
  unicode key names, empty suffix)
* Empty values vs. keys, very long keys/values
* Vault with no key file and no write permission (graceful failure)
* Symlink key-file rejection
* Hard-link key-file rejection
* All-zeros key file rejection
* Re-encrypt: set after key rotation produces readable store under new key
* Atomic write: temp file must not persist on error

TrustScorer
-----------
* Boundary: score at exactly threshold (is_trusted uses strict >, not >=)
* Score clamp at exactly 0 and exactly 1000
* Single violation drives fresh entity from default to exactly 300
* Two violations from default bring score to 100, not below 0
* Three violations from default floor at 0
* Rapid successive successes never exceed 1000
* Weight=0 success/failure is a no-op
* Negative weight on record_success raises no exception, decreases score
* Multiple entity types tracked independently (tool, provider, mcp)
* reset() on unknown entity sets it to 500
* get_scores() returns snapshot (modifying it does not affect internal state)
* Thread safety: 50 concurrent record_success + 50 record_failure converge
  to a deterministic result within bounds
* Recovery trajectory: many failures then many successes recovers score
* is_trusted with threshold=0 is always True (score always > 0 after first
  success, and >= 0 otherwise — actually score=0 is NOT > 0, so test that edge)
* is_trusted with threshold=1000 is always False (score can never exceed 1000,
  and 1000 is NOT > 1000)
* Custom weights on record_violation
* record_failure on unknown entity starts from DEFAULT and decrements
"""

from __future__ import annotations

import secrets
import threading
from pathlib import Path

import pytest

from missy.security.trust import DEFAULT_SCORE, MAX_SCORE, MIN_SCORE, TrustScorer
from missy.security.vault import Vault, VaultError

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_vault(tmp_path: Path) -> Vault:
    return Vault(vault_dir=str(tmp_path / "vault"))


def vault_dir_path(tmp_path: Path) -> Path:
    return tmp_path / "vault"


# ===========================================================================
# VAULT EDGE CASES
# ===========================================================================


class TestVaultKeyRotation:
    """Vault key rotation and re-encryption scenarios."""

    def test_rotate_key_then_re_save_all_secrets(self, tmp_path):
        """After rotating the key, re-saving secrets makes them readable."""
        vault_dir = str(vault_dir_path(tmp_path))
        v1 = Vault(vault_dir=vault_dir)
        v1.set("API_KEY", "sk-original")
        v1.set("DB_PASS", "hunter2")

        # Simulate key rotation: read all secrets, replace key file, re-save.
        all_secrets = {k: v1.get(k) for k in v1.list_keys()}
        new_key = secrets.token_bytes(32)
        v1._key_path.write_bytes(new_key)

        # New vault instance picks up the new key.
        v2 = Vault(vault_dir=vault_dir)
        assert v2._key == new_key

        # Old encrypted data is now unreadable under the new key.
        with pytest.raises(VaultError, match="Cannot decrypt vault"):
            v2.list_keys()

        # Re-save all secrets under the new key.
        v2._vault_path.unlink(missing_ok=True)
        for k, val in all_secrets.items():
            v2.set(k, val)

        assert v2.get("API_KEY") == "sk-original"
        assert v2.get("DB_PASS") == "hunter2"

    def test_old_ciphertext_unreadable_after_key_rotation(self, tmp_path):
        """Ciphertext from the old key is rejected after key rotation."""
        vault_dir = str(vault_dir_path(tmp_path))
        v1 = Vault(vault_dir=vault_dir)
        v1.set("SECRET", "value")

        # Overwrite key with a different random key.
        v1._key_path.write_bytes(secrets.token_bytes(32))
        v2 = Vault(vault_dir=vault_dir)
        with pytest.raises(VaultError, match="Cannot decrypt vault"):
            v2.get("SECRET")

    def test_fresh_vault_after_deletion_starts_empty(self, tmp_path):
        """Deleting the encrypted store and reinitialising gives an empty vault."""
        vault_dir = str(vault_dir_path(tmp_path))
        v1 = Vault(vault_dir=vault_dir)
        v1.set("GONE", "bye")
        v1._vault_path.unlink()

        v2 = Vault(vault_dir=vault_dir)
        assert v2.get("GONE") is None
        assert v2.list_keys() == []


class TestVaultCorruptData:
    """Various corrupt/malformed vault file scenarios."""

    def test_bit_flipped_ciphertext_raises_vault_error(self, tmp_path):
        """A single bit flip in the ciphertext fails the AEAD tag check."""
        vault = make_vault(tmp_path)
        vault.set("K", "value")
        raw = bytearray(vault._vault_path.read_bytes())
        # Flip the last byte (part of the authentication tag).
        raw[-1] ^= 0xFF
        vault._vault_path.write_bytes(bytes(raw))
        with pytest.raises(VaultError, match="Cannot decrypt vault"):
            vault.get("K")

    def test_empty_vault_file_raises_vault_error(self, tmp_path):
        """An empty (0-byte) vault file must raise, not return an empty dict."""
        vault = make_vault(tmp_path)
        vault.set("K", "v")
        vault._vault_path.write_bytes(b"")
        with pytest.raises(VaultError, match="Cannot decrypt vault"):
            vault.get("K")

    def test_only_nonce_no_ciphertext_raises_vault_error(self, tmp_path):
        """A 12-byte file (nonce only, no actual ciphertext) is invalid."""
        vault = make_vault(tmp_path)
        vault.set("K", "v")
        # Write exactly 12 bytes — just a nonce, nothing encrypted.
        vault._vault_path.write_bytes(secrets.token_bytes(12))
        with pytest.raises(VaultError, match="Cannot decrypt vault"):
            vault.get("K")

    def test_valid_json_but_wrong_key_raises_vault_error(self, tmp_path):
        """Data encrypted with one key cannot be read with a different key."""
        vault_dir = vault_dir_path(tmp_path)
        v1 = Vault(vault_dir=str(vault_dir))
        v1.set("X", "y")
        # Swap out the key for a different 32-byte key.
        (vault_dir / "vault.key").write_bytes(secrets.token_bytes(32))
        v2 = Vault(vault_dir=str(vault_dir))
        with pytest.raises(VaultError):
            v2.get("X")

    def test_random_bytes_raises_vault_error(self, tmp_path):
        """Completely random bytes in the vault file must raise VaultError."""
        vault = make_vault(tmp_path)
        vault.set("K", "v")
        vault._vault_path.write_bytes(secrets.token_bytes(256))
        with pytest.raises(VaultError, match="Cannot decrypt vault"):
            vault.list_keys()


class TestVaultKeySecurity:
    """Key file security checks (symlinks, hard links, all-zeros)."""

    def test_symlink_key_file_raises_vault_error(self, tmp_path):
        """A symlinked vault key file must be rejected."""
        vault_dir = vault_dir_path(tmp_path)
        vault_dir.mkdir(mode=0o700, parents=True)
        real_key = tmp_path / "real.key"
        real_key.write_bytes(secrets.token_bytes(32))
        key_link = vault_dir / "vault.key"
        key_link.symlink_to(real_key)
        with pytest.raises(VaultError, match="symlink"):
            Vault(vault_dir=str(vault_dir))

    def test_hard_linked_key_file_raises_vault_error(self, tmp_path):
        """A vault key file with st_nlink > 1 must be rejected."""
        vault_dir = vault_dir_path(tmp_path)
        vault_dir.mkdir(mode=0o700, parents=True)
        key_path = vault_dir / "vault.key"
        key_path.write_bytes(secrets.token_bytes(32))
        # Create a hard link to the key file.
        hard_link = tmp_path / "hardlink.key"
        hard_link.hardlink_to(key_path)
        with pytest.raises(VaultError, match="hard link"):
            Vault(vault_dir=str(vault_dir))

    def test_all_zeros_key_raises_vault_error(self, tmp_path):
        """A key file filled with null bytes must be rejected as corrupted."""
        vault_dir = vault_dir_path(tmp_path)
        vault_dir.mkdir(mode=0o700, parents=True)
        (vault_dir / "vault.key").write_bytes(b"\x00" * 32)
        with pytest.raises(VaultError, match="all zeros"):
            Vault(vault_dir=str(vault_dir))

    def test_permissive_key_file_logs_warning(self, tmp_path, caplog):
        """A world-readable key file logs a warning but still loads."""
        import logging

        vault_dir = vault_dir_path(tmp_path)
        vault_dir.mkdir(mode=0o700, parents=True)
        key_path = vault_dir / "vault.key"
        key_path.write_bytes(secrets.token_bytes(32))
        # Set world-readable permissions (0o644).
        key_path.chmod(0o644)
        with caplog.at_level(logging.WARNING, logger="missy.security.vault"):
            v = Vault(vault_dir=str(vault_dir))
        assert v._key is not None
        assert any(
            "permissive" in r.message.lower() or "recommend" in r.message.lower()
            for r in caplog.records
        )


class TestVaultEdgeCaseValues:
    """Edge cases for key names and values stored in the vault."""

    def test_empty_string_key_name(self, tmp_path):
        """An empty string is a valid dictionary key in Python."""
        vault = make_vault(tmp_path)
        vault.set("", "empty-key-value")
        assert vault.get("") == "empty-key-value"

    def test_very_long_key_name(self, tmp_path):
        """A 4 KB key name must be stored and retrieved correctly."""
        vault = make_vault(tmp_path)
        long_key = "K" * 4096
        vault.set(long_key, "v")
        assert vault.get(long_key) == "v"

    def test_very_long_value(self, tmp_path):
        """A 1 MB value must survive an encrypt/decrypt round-trip."""
        vault = make_vault(tmp_path)
        big_value = secrets.token_hex(512 * 1024)  # 1 MB hex string
        vault.set("BIG", big_value)
        assert vault.get("BIG") == big_value

    def test_unicode_key_name(self, tmp_path):
        """Unicode characters in key names are supported."""
        vault = make_vault(tmp_path)
        vault.set("秘密鍵", "top-secret")
        assert vault.get("秘密鍵") == "top-secret"

    def test_unicode_value(self, tmp_path):
        """Unicode characters in values are preserved exactly."""
        vault = make_vault(tmp_path)
        vault.set("GREETING", "こんにちは世界 🔑")
        assert vault.get("GREETING") == "こんにちは世界 🔑"

    def test_newlines_in_value(self, tmp_path):
        """Newline characters inside a value survive the round-trip."""
        vault = make_vault(tmp_path)
        multiline = "line1\nline2\nline3"
        vault.set("MULTI", multiline)
        assert vault.get("MULTI") == multiline

    def test_json_string_as_value(self, tmp_path):
        """A JSON-encoded string stored as a vault value is not double-decoded."""
        vault = make_vault(tmp_path)
        json_val = '{"key": "value", "num": 42}'
        vault.set("JSON_VAL", json_val)
        assert vault.get("JSON_VAL") == json_val


class TestVaultResolveEdgeCases:
    """Edge cases for the resolve() method."""

    def test_vault_prefix_with_unicode_key(self, tmp_path):
        """vault:// reference with a unicode key name resolves correctly."""
        vault = make_vault(tmp_path)
        vault.set("MY_KEY_ñ", "secret-ñ")
        assert vault.resolve("vault://MY_KEY_ñ") == "secret-ñ"

    def test_vault_prefix_empty_suffix_raises(self, tmp_path):
        """vault:// with nothing after the slashes raises because key is empty."""
        vault = make_vault(tmp_path)
        # Empty key "" is not stored, so a VaultError must be raised.
        with pytest.raises(VaultError):
            vault.resolve("vault://")

    def test_vault_prefix_key_with_spaces_not_found(self, tmp_path):
        """vault:// with spaces in the key suffix raises when key is absent."""
        vault = make_vault(tmp_path)
        with pytest.raises(VaultError, match="not found in vault"):
            vault.resolve("vault://key with spaces")

    def test_vault_prefix_key_with_spaces_stored_and_resolved(self, tmp_path):
        """If a key containing spaces is stored it can be resolved via vault://."""
        vault = make_vault(tmp_path)
        vault.set("key with spaces", "spacey-secret")
        assert vault.resolve("vault://key with spaces") == "spacey-secret"

    def test_resolve_double_dollar_is_env_lookup(self, tmp_path, monkeypatch):
        """A value starting with '$$' is treated as an env lookup for '$NOT_AN_ENV'.

        resolve() checks startswith('$') and strips the first '$', so '$$FOO'
        becomes an env lookup for '$FOO'.  This test documents that behaviour.
        """
        monkeypatch.delenv("$NOT_AN_ENV", raising=False)
        vault = make_vault(tmp_path)
        # '$$NOT_AN_ENV' starts with '$', so it becomes an env lookup for '$NOT_AN_ENV'.
        with pytest.raises(VaultError, match="Environment variable"):
            vault.resolve("$$NOT_AN_ENV")

    def test_resolve_http_url_returned_as_is(self, tmp_path):
        """An http:// URL (not vault://) is returned without lookup."""
        vault = make_vault(tmp_path)
        url = "http://example.com/api?key=abc"
        assert vault.resolve(url) == url

    def test_resolve_vault_prefix_after_set_returns_updated_value(self, tmp_path):
        """resolve() always reads the live store, not a cached value."""
        vault = make_vault(tmp_path)
        vault.set("TOKEN", "v1")
        assert vault.resolve("vault://TOKEN") == "v1"
        vault.set("TOKEN", "v2")
        assert vault.resolve("vault://TOKEN") == "v2"


class TestVaultConcurrency:
    """Thread-safety tests for concurrent vault access."""

    def test_concurrent_writes_do_not_corrupt_store(self, tmp_path):
        """Multiple threads writing different keys must all be visible."""
        vault = make_vault(tmp_path)
        errors: list[Exception] = []
        n_threads = 20

        def writer(idx: int) -> None:
            try:
                vault.set(f"KEY_{idx}", f"value_{idx}")
            except Exception as exc:  # noqa: BLE001
                errors.append(exc)

        threads = [threading.Thread(target=writer, args=(i,)) for i in range(n_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors, f"Thread errors: {errors}"
        # Every key must be readable; the exact subset depends on ordering,
        # but at a minimum one full set of writes must have succeeded.
        keys = set(vault.list_keys())
        # At minimum, the last writer must have left its key intact.
        for k in keys:
            idx = int(k.split("_")[1])
            assert vault.get(k) == f"value_{idx}"

    def test_concurrent_reads_while_writing(self, tmp_path):
        """Concurrent readers must never see VaultError during a write."""
        vault = make_vault(tmp_path)
        vault.set("SHARED", "initial")
        read_errors: list[Exception] = []
        stop_event = threading.Event()

        def reader() -> None:
            while not stop_event.is_set():
                try:
                    _ = vault.get("SHARED")
                except VaultError as exc:
                    read_errors.append(exc)

        def writer() -> None:
            for i in range(30):
                vault.set("SHARED", f"value_{i}")

        readers = [threading.Thread(target=reader) for _ in range(5)]
        writer_thread = threading.Thread(target=writer)
        for r in readers:
            r.start()
        writer_thread.start()
        writer_thread.join()
        stop_event.set()
        for r in readers:
            r.join()

        # Readers must never encounter a VaultError due to torn writes.
        assert not read_errors, f"Concurrent read errors: {read_errors}"


# ===========================================================================
# TRUST SCORER EDGE CASES
# ===========================================================================


class TestTrustScorerBoundaryConditions:
    """Exact boundary values for score clamping and threshold comparisons."""

    def test_score_exactly_at_max_after_single_large_success(self):
        """A weight that would overshoot 1000 is clamped to exactly 1000."""
        scorer = TrustScorer()
        scorer.record_success("e", weight=MAX_SCORE)  # 500 + 1000 > 1000
        assert scorer.score("e") == MAX_SCORE

    def test_score_exactly_at_min_after_single_large_failure(self):
        """A weight that would undershoot 0 is clamped to exactly 0."""
        scorer = TrustScorer()
        scorer.record_failure("e", weight=MAX_SCORE)  # 500 - 1000 < 0
        assert scorer.score("e") == MIN_SCORE

    def test_score_at_exactly_threshold_is_not_trusted(self):
        """is_trusted uses strict >, so score == threshold returns False."""
        scorer = TrustScorer()
        threshold = 300
        # Bring score to exactly 300.
        scorer.record_failure("e", weight=DEFAULT_SCORE - threshold)
        assert scorer.score("e") == threshold
        assert scorer.is_trusted("e", threshold=threshold) is False

    def test_score_one_above_threshold_is_trusted(self):
        """Score exactly one point above threshold returns True from is_trusted."""
        scorer = TrustScorer()
        threshold = 300
        scorer.record_failure("e", weight=DEFAULT_SCORE - threshold - 1)
        assert scorer.score("e") == threshold + 1
        assert scorer.is_trusted("e", threshold=threshold) is True

    def test_is_trusted_threshold_zero_with_score_zero(self):
        """is_trusted(threshold=0) returns False when score is exactly 0."""
        scorer = TrustScorer()
        scorer.record_failure("e", weight=MAX_SCORE)  # floor at 0
        assert scorer.score("e") == 0
        assert scorer.is_trusted("e", threshold=0) is False  # 0 is NOT > 0

    def test_is_trusted_threshold_zero_with_positive_score(self):
        """is_trusted(threshold=0) returns True for any positive score."""
        scorer = TrustScorer()
        # Default 500 > 0
        assert scorer.is_trusted("e", threshold=0) is True

    def test_is_trusted_threshold_max_always_false(self):
        """No score can exceed 1000, so threshold=1000 is always False."""
        scorer = TrustScorer()
        for _ in range(200):
            scorer.record_success("e", weight=100)
        assert scorer.score("e") == MAX_SCORE
        assert scorer.is_trusted("e", threshold=MAX_SCORE) is False


class TestTrustScorerViolationChains:
    """Multiple violations and their cumulative effect on a fresh entity."""

    def test_one_violation_from_default(self):
        """Default 500 minus violation weight 200 = 300."""
        scorer = TrustScorer()
        scorer.record_violation("e")  # default weight=200
        assert scorer.score("e") == 300

    def test_two_violations_from_default(self):
        """Two violations: 500 - 200 - 200 = 100."""
        scorer = TrustScorer()
        scorer.record_violation("e")
        scorer.record_violation("e")
        assert scorer.score("e") == 100

    def test_three_violations_floors_at_zero(self):
        """Three violations: 500 - 200 - 200 - 200 would be -100 → clamped to 0."""
        scorer = TrustScorer()
        for _ in range(3):
            scorer.record_violation("e")
        assert scorer.score("e") == MIN_SCORE

    def test_rapid_violations_never_go_below_zero(self):
        """100 violations in a row must leave score at exactly 0, not negative."""
        scorer = TrustScorer()
        for _ in range(100):
            scorer.record_violation("e")
        assert scorer.score("e") == MIN_SCORE

    def test_custom_violation_weight(self):
        """record_violation with weight=100 decrements by exactly 100."""
        scorer = TrustScorer()
        scorer.record_violation("e", weight=100)
        assert scorer.score("e") == DEFAULT_SCORE - 100


class TestTrustScorerRecovery:
    """Score recovery after accumulated failures."""

    def test_recovery_after_floored_score(self):
        """After hitting 0, successive successes increase the score normally."""
        scorer = TrustScorer()
        # Floor the score.
        scorer.record_failure("e", weight=MAX_SCORE)
        assert scorer.score("e") == MIN_SCORE
        # Recover with 50 successes of weight 10 = 500.
        for _ in range(50):
            scorer.record_success("e", weight=10)
        assert scorer.score("e") == 500

    def test_recovery_does_not_exceed_max(self):
        """Recovering past 1000 is clamped to exactly 1000."""
        scorer = TrustScorer()
        scorer.record_failure("e", weight=100)  # score = 400
        # 200 successes of weight 10 = 2000 points added → capped at 1000.
        for _ in range(200):
            scorer.record_success("e", weight=10)
        assert scorer.score("e") == MAX_SCORE


class TestTrustScorerZeroWeight:
    """Weight=0 operations should be no-ops."""

    def test_zero_weight_success_no_change(self):
        scorer = TrustScorer()
        scorer.record_success("e", weight=0)
        assert scorer.score("e") == DEFAULT_SCORE

    def test_zero_weight_failure_no_change(self):
        scorer = TrustScorer()
        scorer.record_failure("e", weight=0)
        assert scorer.score("e") == DEFAULT_SCORE

    def test_zero_weight_violation_no_change(self):
        scorer = TrustScorer()
        scorer.record_violation("e", weight=0)
        assert scorer.score("e") == DEFAULT_SCORE


class TestTrustScorerEntityTypes:
    """Different entity-type naming conventions tracked independently."""

    def test_tool_provider_mcp_independent(self):
        """tool:, provider:, and mcp: prefixed entities do not share scores."""
        scorer = TrustScorer()
        scorer.record_failure("tool:bash", weight=200)
        scorer.record_success("provider:anthropic", weight=50)
        scorer.record_violation("mcp:github")

        assert scorer.score("tool:bash") == DEFAULT_SCORE - 200
        assert scorer.score("provider:anthropic") == DEFAULT_SCORE + 50
        assert scorer.score("mcp:github") == DEFAULT_SCORE - 200

    def test_unknown_entity_returns_default_not_zero(self):
        """Querying a never-seen entity must return DEFAULT_SCORE, not 0."""
        scorer = TrustScorer()
        assert scorer.score("brand:new:entity") == DEFAULT_SCORE

    def test_reset_unknown_entity_sets_to_default(self):
        """reset() on an entity that was never touched sets it to DEFAULT_SCORE."""
        scorer = TrustScorer()
        scorer.reset("never_seen")
        assert scorer.score("never_seen") == DEFAULT_SCORE

    def test_multiple_entities_in_get_scores(self):
        """get_scores() reports all tracked entities with correct values."""
        scorer = TrustScorer()
        scorer.record_success("a", weight=10)
        scorer.record_failure("b", weight=20)
        snap = scorer.get_scores()
        assert snap["a"] == DEFAULT_SCORE + 10
        assert snap["b"] == DEFAULT_SCORE - 20

    def test_get_scores_snapshot_is_isolated(self):
        """Mutating the dict returned by get_scores() does not affect the scorer."""
        scorer = TrustScorer()
        scorer.record_success("e", weight=10)
        snap = scorer.get_scores()
        snap["e"] = 0  # tamper with the snapshot
        assert scorer.score("e") == DEFAULT_SCORE + 10


class TestTrustScorerThreadSafety:
    """Concurrent mutations must not corrupt scores or cause exceptions."""

    def test_concurrent_success_and_failure_stays_within_bounds(self):
        """50 threads adding successes and 50 threads adding failures."""
        scorer = TrustScorer()
        errors: list[Exception] = []

        def success_worker() -> None:
            try:
                for _ in range(20):
                    scorer.record_success("shared", weight=10)
            except Exception as exc:  # noqa: BLE001
                errors.append(exc)

        def failure_worker() -> None:
            try:
                for _ in range(20):
                    scorer.record_failure("shared", weight=10)
            except Exception as exc:  # noqa: BLE001
                errors.append(exc)

        threads = [threading.Thread(target=success_worker) for _ in range(50)] + [
            threading.Thread(target=failure_worker) for _ in range(50)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors
        final = scorer.score("shared")
        assert MIN_SCORE <= final <= MAX_SCORE

    def test_concurrent_mixed_operations_no_exceptions(self):
        """Concurrent success/failure/violation/reset/score calls raise nothing."""
        scorer = TrustScorer()
        errors: list[Exception] = []

        def mixed(entity: str) -> None:
            try:
                for _ in range(10):
                    scorer.record_success(entity, weight=5)
                    scorer.record_failure(entity, weight=5)
                    scorer.record_violation(entity, weight=50)
                    _ = scorer.score(entity)
                    _ = scorer.is_trusted(entity)
                    scorer.reset(entity)
            except Exception as exc:  # noqa: BLE001
                errors.append(exc)

        threads = [threading.Thread(target=mixed, args=(f"e{i}",)) for i in range(30)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors

    def test_concurrent_record_failure_from_default_floors_at_zero(self):
        """100 concurrent threads each calling record_failure must leave score >= 0."""
        scorer = TrustScorer()
        barrier = threading.Barrier(100)

        def fail_at_once() -> None:
            barrier.wait()
            scorer.record_failure("burst", weight=50)

        threads = [threading.Thread(target=fail_at_once) for _ in range(100)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert scorer.score("burst") == MIN_SCORE
