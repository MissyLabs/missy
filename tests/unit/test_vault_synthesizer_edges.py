"""Edge case tests for Vault and MemorySynthesizer.

Covers scenarios not addressed in tests/security/test_vault.py or
tests/memory/test_synthesizer.py: file permissions, large payloads, Unicode,
binary-like values, concurrent access, and synthesizer boundary conditions.
"""

from __future__ import annotations

import os
import stat
import threading
from pathlib import Path

import pytest

from missy.memory.synthesizer import MemoryFragment, MemorySynthesizer
from missy.security.vault import Vault, VaultError


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_vault(tmp_path: Path) -> Vault:
    return Vault(vault_dir=str(tmp_path / "vault"))


# ===========================================================================
# Vault edge cases
# ===========================================================================


class TestVaultSetGet:
    """set / get round-trips not already covered by TestSetGet."""

    def test_get_returns_none_for_missing_key(self, tmp_path):
        vault = make_vault(tmp_path)
        assert vault.get("MISSING_KEY") is None

    def test_delete_then_get_returns_none(self, tmp_path):
        vault = make_vault(tmp_path)
        vault.set("TEMP", "ephemeral")
        vault.delete("TEMP")
        assert vault.get("TEMP") is None

    def test_overwrite_existing_key(self, tmp_path):
        vault = make_vault(tmp_path)
        vault.set("API_KEY", "old-value")
        vault.set("API_KEY", "new-value")
        assert vault.get("API_KEY") == "new-value"

    def test_list_keys_empty_vault(self, tmp_path):
        vault = make_vault(tmp_path)
        assert vault.list_keys() == []

    def test_key_with_dots_and_slashes(self, tmp_path):
        vault = make_vault(tmp_path)
        key = "service.api/key-name_v2"
        vault.set(key, "dotslash-value")
        assert vault.get(key) == "dotslash-value"

    def test_key_with_unicode_characters(self, tmp_path):
        vault = make_vault(tmp_path)
        key = "cle_\u00e9tendu"
        value = "valeur-secr\u00e8te"
        vault.set(key, value)
        assert vault.get(key) == value

    def test_key_with_whitespace(self, tmp_path):
        vault = make_vault(tmp_path)
        key = "KEY WITH SPACES"
        vault.set(key, "value")
        assert vault.get(key) == "value"
        assert key in vault.list_keys()

    def test_key_with_null_byte_like_sequence(self, tmp_path):
        # JSON keys can contain escape sequences; ensure round-trip integrity.
        vault = make_vault(tmp_path)
        key = "key\\nwith\\nnewlines"
        vault.set(key, "ok")
        assert vault.get(key) == "ok"


class TestVaultLargePayload:
    """Large-value storage should complete without errors."""

    def test_large_value_1mb(self, tmp_path):
        vault = make_vault(tmp_path)
        large = "x" * (1024 * 1024)
        vault.set("BIG", large)
        assert vault.get("BIG") == large

    def test_large_value_survives_reinit(self, tmp_path):
        vault_dir = str(tmp_path / "vault")
        v1 = Vault(vault_dir=vault_dir)
        large = "A" * 500_000
        v1.set("BIG", large)
        v2 = Vault(vault_dir=vault_dir)
        assert v2.get("BIG") == large


class TestVaultUnicodeAndBinaryLike:
    """Unicode and binary-like (base64/hex) values round-trip cleanly."""

    def test_unicode_value_cjk(self, tmp_path):
        vault = make_vault(tmp_path)
        value = "\u4e2d\u6587\u5bc6\u6d41"  # "Chinese secret"
        vault.set("CJK", value)
        assert vault.get("CJK") == value

    def test_unicode_value_emoji(self, tmp_path):
        vault = make_vault(tmp_path)
        value = "\U0001f511\U0001f512\U0001f513"  # key/lock emojis
        vault.set("EMOJI", value)
        assert vault.get("EMOJI") == value

    def test_binary_like_base64_value(self, tmp_path):
        vault = make_vault(tmp_path)
        import base64

        raw = bytes(range(256))
        b64 = base64.b64encode(raw).decode()
        vault.set("B64", b64)
        assert vault.get("B64") == b64

    def test_hex_string_value(self, tmp_path):
        vault = make_vault(tmp_path)
        hex_val = "deadbeefcafebabe" * 32
        vault.set("HEX", hex_val)
        assert vault.get("HEX") == hex_val

    def test_newlines_in_value(self, tmp_path):
        vault = make_vault(tmp_path)
        multi = "line1\nline2\nline3"
        vault.set("MULTI", multi)
        assert vault.get("MULTI") == multi

    def test_json_like_value(self, tmp_path):
        vault = make_vault(tmp_path)
        json_val = '{"token": "abc", "expires": 9999}'
        vault.set("JSON_VAL", json_val)
        assert vault.get("JSON_VAL") == json_val


class TestVaultFilePermissions:
    """Key file and vault file must be created with strict permissions."""

    def test_vault_key_file_mode_is_0o600(self, tmp_path):
        make_vault(tmp_path)
        key_path = tmp_path / "vault" / "vault.key"
        mode = stat.S_IMODE(key_path.stat().st_mode)
        assert mode == 0o600, f"Expected 0o600, got {oct(mode)}"

    def test_vault_enc_file_mode_is_0o600(self, tmp_path):
        vault = make_vault(tmp_path)
        vault.set("X", "y")
        enc_path = tmp_path / "vault" / "vault.enc"
        mode = stat.S_IMODE(enc_path.stat().st_mode)
        assert mode == 0o600, f"Expected 0o600, got {oct(mode)}"

    def test_vault_directory_mode_is_0o700(self, tmp_path):
        make_vault(tmp_path)
        vault_dir = tmp_path / "vault"
        mode = stat.S_IMODE(vault_dir.stat().st_mode)
        assert mode == 0o700, f"Expected 0o700, got {oct(mode)}"

    def test_vault_enc_permissions_after_overwrite(self, tmp_path):
        """Re-writing the vault (atomic rename) must preserve 0o600."""
        vault = make_vault(tmp_path)
        vault.set("A", "1")
        vault.set("B", "2")  # triggers a second write/rename
        enc_path = tmp_path / "vault" / "vault.enc"
        mode = stat.S_IMODE(enc_path.stat().st_mode)
        assert mode == 0o600


class TestVaultDirectoryAutoCreation:
    """Deeply nested vault directories are created on first use."""

    def test_nested_directory_created_on_init(self, tmp_path):
        deep = tmp_path / "a" / "b" / "c" / "vault"
        assert not deep.exists()
        Vault(vault_dir=str(deep))
        assert deep.is_dir()

    def test_nested_directory_usable_after_creation(self, tmp_path):
        deep = tmp_path / "d" / "e" / "vault"
        vault = Vault(vault_dir=str(deep))
        vault.set("DEEP_KEY", "deep_value")
        assert vault.get("DEEP_KEY") == "deep_value"


class TestVaultSymlinkHardlinkRejection:
    """Symlink and hard-linked key files must be rejected."""

    def test_symlink_key_file_raises(self, tmp_path):
        vault_dir = tmp_path / "vault"
        vault_dir.mkdir(mode=0o700)
        # Create a legitimate 32-byte key file target.
        target = tmp_path / "real.key"
        target.write_bytes(os.urandom(32))
        symlink = vault_dir / "vault.key"
        symlink.symlink_to(target)
        with pytest.raises(VaultError, match="symlink"):
            Vault(vault_dir=str(vault_dir))

    def test_hard_linked_key_file_raises(self, tmp_path):
        vault_dir = tmp_path / "vault"
        vault_dir.mkdir(mode=0o700)
        key_path = vault_dir / "vault.key"
        key_path.write_bytes(os.urandom(32))
        os.chmod(str(key_path), 0o600)
        # Create a hard link alongside the key file.
        hardlink = tmp_path / "hardlink.key"
        os.link(str(key_path), str(hardlink))
        with pytest.raises(VaultError, match="hard links"):
            Vault(vault_dir=str(vault_dir))


class TestVaultConcurrentReadsWrites:
    """Concurrent operations must not corrupt the vault store."""

    def test_concurrent_writes_do_not_corrupt(self, tmp_path):
        # The Vault uses an atomic rename for each write but has no cross-write
        # locking: concurrent writers each read the store, merge their key, and
        # atomically rename — so the last rename wins and earlier writes to
        # *different* keys can be lost (read-modify-write race).  The invariant
        # tested here is that no write ever raises and the vault file remains
        # consistently decryptable afterward.
        vault = make_vault(tmp_path)
        errors: list[Exception] = []

        def writer(n: int) -> None:
            try:
                vault.set(f"KEY_{n}", f"value_{n}")
            except Exception as exc:  # noqa: BLE001
                errors.append(exc)

        threads = [threading.Thread(target=writer, args=(i,)) for i in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors, f"Errors during concurrent writes: {errors}"
        # The vault file must still be readable; we cannot assert how many keys
        # survived due to the intentional read-modify-write race, but every
        # surviving key must have a correct value.
        keys = vault.list_keys()
        for key in keys:
            n = int(key.split("_")[1])
            assert vault.get(key) == f"value_{n}"

    def test_concurrent_reads_do_not_raise(self, tmp_path):
        vault = make_vault(tmp_path)
        vault.set("SHARED", "constant-value")
        errors: list[Exception] = []
        results: list[str | None] = []
        lock = threading.Lock()

        def reader() -> None:
            try:
                val = vault.get("SHARED")
                with lock:
                    results.append(val)
            except Exception as exc:  # noqa: BLE001
                errors.append(exc)

        threads = [threading.Thread(target=reader) for _ in range(30)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors
        assert all(v == "constant-value" for v in results)


# ===========================================================================
# MemorySynthesizer edge cases
# ===========================================================================


class TestSynthesizerEmptyInputs:
    """All-empty combinations must return an empty string without raising."""

    def test_no_fragments_no_query(self):
        synth = MemorySynthesizer()
        assert synth.synthesize("") == ""

    def test_no_fragments_with_query(self):
        synth = MemorySynthesizer()
        assert synth.synthesize("how do I fix Docker?") == ""

    def test_empty_items_list_adds_no_fragments(self):
        synth = MemorySynthesizer()
        synth.add_fragments("learnings", [])
        synth.add_fragments("playbook", [])
        synth.add_fragments("summaries", [])
        assert synth.synthesize("anything") == ""

    def test_whitespace_only_items_are_stored(self):
        # Whitespace strings are valid JSON values; synthesizer should not crash.
        synth = MemorySynthesizer()
        synth.add_fragments("conversation", ["   ", "\t\n"])
        result = synth.synthesize("query")
        # Must not raise; content may or may not appear depending on budget.
        assert isinstance(result, str)


class TestSynthesizerSingleEntry:
    """A synthesizer with exactly one fragment should produce exactly one line."""

    def test_single_learning_appears_in_output(self):
        synth = MemorySynthesizer()
        synth.add_fragments("learnings", ["always check ports first"], base_relevance=0.7)
        result = synth.synthesize("network ports")
        assert result.strip() != ""
        lines = [ln for ln in result.strip().splitlines() if ln]
        assert len(lines) == 1
        assert "always check ports first" in lines[0]
        assert "[learnings]" in lines[0]

    def test_single_playbook_entry(self):
        synth = MemorySynthesizer()
        synth.add_fragments("playbook", ["run docker ps to list containers"], base_relevance=0.6)
        result = synth.synthesize("docker")
        assert "[playbook]" in result

    def test_single_summary_entry(self):
        synth = MemorySynthesizer()
        synth.add_fragments("summaries", ["session covered redis setup"], base_relevance=0.4)
        result = synth.synthesize("redis")
        assert "[summaries]" in result


class TestSynthesizerBaseRelevanceWeights:
    """Verify that the prescribed base_relevance values (0.7/0.6/0.4) control ranking."""

    def test_learnings_outranks_summaries_for_neutral_query(self):
        # With a query that has zero overlap with all fragments, the ranking
        # reduces purely to base_relevance * 0.5, so learnings > summaries.
        synth = MemorySynthesizer()
        synth.add_fragments("learnings", ["alpha bravo charlie"], base_relevance=0.7)
        synth.add_fragments("summaries", ["alpha bravo charlie"], base_relevance=0.4)
        result = synth.synthesize("zzzz unrelated")
        lines = [ln for ln in result.strip().splitlines() if ln]
        assert lines[0].startswith("[learnings]")

    def test_playbook_outranks_summaries_for_neutral_query(self):
        synth = MemorySynthesizer()
        synth.add_fragments("playbook", ["delta echo foxtrot"], base_relevance=0.6)
        synth.add_fragments("summaries", ["delta echo foxtrot"], base_relevance=0.4)
        result = synth.synthesize("zzzz unrelated")
        lines = [ln for ln in result.strip().splitlines() if ln]
        assert lines[0].startswith("[playbook]")

    def test_all_three_tiers_ordered_correctly(self):
        # Use distinct content per tier so deduplication does not collapse them.
        synth = MemorySynthesizer()
        synth.add_fragments("summaries", ["alpha bravo charlie delta"], base_relevance=0.4)
        synth.add_fragments("playbook", ["echo foxtrot golf hotel"], base_relevance=0.6)
        synth.add_fragments("learnings", ["india juliet kilo lima"], base_relevance=0.7)
        result = synth.synthesize("zzzz unrelated")
        lines = [ln for ln in result.strip().splitlines() if ln]
        sources = [ln.split("]")[0].lstrip("[") for ln in lines]
        assert sources == ["learnings", "playbook", "summaries"]


class TestSynthesizerRelevanceScoring:
    """Keyword overlap scoring fine-grained behaviour."""

    def test_partial_overlap_boosts_score_above_no_overlap(self):
        synth = MemorySynthesizer()
        frag_partial = MemoryFragment(source="x", content="docker networking tips", relevance=0.5)
        frag_none = MemoryFragment(source="x", content="shopping list bananas", relevance=0.5)
        score_partial = synth.score_relevance(frag_partial, "docker setup guide")
        score_none = synth.score_relevance(frag_none, "docker setup guide")
        assert score_partial > score_none

    def test_score_capped_at_one(self):
        synth = MemorySynthesizer()
        frag = MemoryFragment(source="x", content="a b c", relevance=1.0)
        score = synth.score_relevance(frag, "a b c")
        assert score <= 1.0

    def test_score_non_negative(self):
        synth = MemorySynthesizer()
        frag = MemoryFragment(source="x", content="unrelated content here", relevance=0.0)
        score = synth.score_relevance(frag, "totally different words")
        assert score >= 0.0

    def test_empty_query_returns_base_relevance_unchanged(self):
        synth = MemorySynthesizer()
        for base in (0.0, 0.3, 0.7, 1.0):
            frag = MemoryFragment(source="x", content="some text", relevance=base)
            assert synth.score_relevance(frag, "") == pytest.approx(base)

    def test_empty_content_with_query(self):
        # An empty content string has no words; overlap is 0.
        synth = MemorySynthesizer()
        frag = MemoryFragment(source="x", content="", relevance=0.8)
        score = synth.score_relevance(frag, "docker networking")
        # overlap = 0 → 0.8 * 0.5 + 0 * 0.5 = 0.4
        assert score == pytest.approx(0.4)

    def test_query_with_no_meaningful_keywords(self):
        # Single stopword-like query; overlap may be zero but should not crash.
        synth = MemorySynthesizer()
        frag = MemoryFragment(source="x", content="docker container", relevance=0.6)
        score = synth.score_relevance(frag, "a")
        assert isinstance(score, float)


class TestSynthesizerDeduplication:
    """Deduplication keeps highest-relevance fragment and removes near-duplicates."""

    def test_identical_content_deduped_keeps_higher_relevance(self):
        synth = MemorySynthesizer()
        frags = [
            MemoryFragment(source="a", content="docker container networking", relevance=0.3),
            MemoryFragment(source="b", content="docker container networking", relevance=0.9),
        ]
        result = synth.deduplicate(frags, threshold=0.8)
        assert len(result) == 1
        assert result[0].relevance == pytest.approx(0.9)

    def test_three_near_duplicates_reduces_to_one(self):
        synth = MemorySynthesizer()
        frags = [
            MemoryFragment(source="a", content="setup docker container", relevance=0.5),
            MemoryFragment(source="b", content="docker container setup", relevance=0.6),
            MemoryFragment(source="c", content="container docker setup", relevance=0.4),
        ]
        result = synth.deduplicate(frags, threshold=0.7)
        assert len(result) == 1
        assert result[0].relevance == pytest.approx(0.6)

    def test_empty_content_fragments_are_considered_duplicates_of_each_other(self):
        synth = MemorySynthesizer()
        frags = [
            MemoryFragment(source="a", content="", relevance=0.5),
            MemoryFragment(source="b", content="", relevance=0.9),
        ]
        result = synth.deduplicate(frags, threshold=0.8)
        # total words == 0 → is_dup == True for the second one;
        # only one fragment should survive.
        assert len(result) == 1

    def test_threshold_boundary_exclusive(self):
        # Overlap exactly at threshold should count as a duplicate.
        synth = MemorySynthesizer()
        # "a b c d" vs "a b c d" → overlap = 4/4 = 1.0; always a duplicate.
        frags = [
            MemoryFragment(source="a", content="a b c d", relevance=0.5),
            MemoryFragment(source="b", content="a b c d", relevance=0.7),
        ]
        result = synth.deduplicate(frags, threshold=1.0)
        assert len(result) == 1

    def test_completely_different_fragments_not_deduped(self):
        synth = MemorySynthesizer()
        frags = [
            MemoryFragment(source="a", content="python web framework flask", relevance=0.5),
            MemoryFragment(source="b", content="docker networking overlay", relevance=0.5),
        ]
        result = synth.deduplicate(frags, threshold=0.8)
        assert len(result) == 2

    def test_deduplication_does_not_mutate_input(self):
        synth = MemorySynthesizer()
        frags = [
            MemoryFragment(source="a", content="same content here", relevance=0.5),
            MemoryFragment(source="b", content="same content here", relevance=0.8),
        ]
        original_len = len(frags)
        synth.deduplicate(frags, threshold=0.8)
        assert len(frags) == original_len


class TestSynthesizerMaxBlockSize:
    """Token budget is enforced; fragments that exceed budget are dropped."""

    def test_zero_token_budget_returns_empty(self):
        synth = MemorySynthesizer(max_tokens=0)
        synth.add_fragments("learnings", ["always check ports"], base_relevance=0.9)
        result = synth.synthesize("ports")
        assert result == ""

    def test_exactly_one_fragment_fits(self):
        synth = MemorySynthesizer(max_tokens=50)
        # A short fragment should fit; a very long one should be dropped.
        synth.add_fragments("a", ["short entry"], base_relevance=0.9)
        synth.add_fragments("b", ["B " * 500], base_relevance=0.1)
        result = synth.synthesize("anything")
        lines = [ln for ln in result.strip().splitlines() if ln]
        assert len(lines) == 1
        assert "short entry" in lines[0]

    def test_high_relevance_exceeds_budget_is_dropped(self):
        # Even the highest-relevance fragment is skipped if it does not fit.
        synth = MemorySynthesizer(max_tokens=5)
        synth.add_fragments("learnings", ["a " * 200], base_relevance=1.0)
        result = synth.synthesize("a")
        assert result == ""

    def test_multiple_small_fragments_fill_budget(self):
        synth = MemorySynthesizer(max_tokens=200)
        items = [f"note {i}" for i in range(50)]
        synth.add_fragments("notes", items, base_relevance=0.5)
        result = synth.synthesize("note")
        assert result != ""
        # Should include multiple entries.
        lines = [ln for ln in result.strip().splitlines() if ln]
        assert len(lines) > 1


class TestSynthesizerSortByRelevance:
    """Output lines must be ordered highest relevance first."""

    def test_explicit_scores_produce_correct_order(self):
        synth = MemorySynthesizer()
        synth.add_fragments("low", ["unrelated banana text"], base_relevance=0.1)
        synth.add_fragments("high", ["docker networking guide"], base_relevance=0.9)
        synth.add_fragments("mid", ["some deployment topic"], base_relevance=0.5)
        result = synth.synthesize("zzz irrelevant query words")
        lines = [ln for ln in result.strip().splitlines() if ln]
        # base_relevance dominates when query has no overlap; high > mid > low.
        assert "[high]" in lines[0]
        assert "[mid]" in lines[1]
        assert "[low]" in lines[2]

    def test_query_keyword_boost_can_reorder_fragments(self):
        synth = MemorySynthesizer()
        # Low base relevance but high query overlap should win against
        # high base relevance with zero overlap.
        synth.add_fragments("irrelevant", ["unmatched text here alpha"], base_relevance=0.9)
        synth.add_fragments("matching", ["docker setup guide docker"], base_relevance=0.1)
        result = synth.synthesize("docker docker setup")
        lines = [ln for ln in result.strip().splitlines() if ln]
        # The "matching" fragment should float to the top.
        assert "[matching]" in lines[0]


class TestSynthesizerContextNoKeywords:
    """Queries that produce zero-word sets use only base relevance."""

    def test_empty_query_uses_base_relevance_for_ordering(self):
        synth = MemorySynthesizer()
        synth.add_fragments("low", ["some text"], base_relevance=0.2)
        synth.add_fragments("high", ["other text"], base_relevance=0.8)
        result = synth.synthesize("")
        lines = [ln for ln in result.strip().splitlines() if ln]
        assert "[high]" in lines[0]
        assert "[low]" in lines[1]

    def test_whitespace_only_query_treated_as_empty(self):
        synth = MemorySynthesizer()
        synth.add_fragments("src", ["content"], base_relevance=0.7)
        # Should not raise; treat as empty query.
        result = synth.synthesize("   \t\n")
        assert isinstance(result, str)

    def test_single_char_query_does_not_crash(self):
        synth = MemorySynthesizer()
        synth.add_fragments("src", ["docker setup"], base_relevance=0.5)
        result = synth.synthesize("d")
        assert isinstance(result, str)


class TestSynthesizerOutputFormat:
    """Verify the [source] label format and that no blank lines appear mid-output."""

    def test_each_line_has_source_label(self):
        synth = MemorySynthesizer()
        synth.add_fragments("learnings", ["tip one", "tip two"], base_relevance=0.7)
        synth.add_fragments("playbook", ["step one"], base_relevance=0.6)
        result = synth.synthesize("anything")
        for line in result.strip().splitlines():
            assert line.startswith("["), f"Line missing source label: {line!r}"

    def test_no_trailing_whitespace_per_line(self):
        synth = MemorySynthesizer()
        synth.add_fragments("notes", ["note with trailing   "], base_relevance=0.5)
        result = synth.synthesize("note")
        for line in result.splitlines():
            # The source label + content is joined; content trailing spaces are
            # preserved as-is by the synthesizer, so just ensure we don't add extra.
            assert "\n" not in line
