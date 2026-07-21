"""Tests for missy.channels.discord.zip_extract — safe extraction of
untrusted zip archives: zip-slip / path traversal, zip bombs, symlinks,
encrypted entries, and per-entry size limits.
"""

from __future__ import annotations

import io
import zipfile

import pytest

from missy.channels.discord.zip_extract import (
    MAX_ZIP_ENTRIES,
    MAX_ZIP_ENTRY_UNCOMPRESSED_BYTES,
    safe_extract_zip,
)


def _zip_bytes(entries: dict[str, bytes], *, compression=zipfile.ZIP_DEFLATED) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression) as zf:
        for name, content in entries.items():
            zf.writestr(name, content)
    return buf.getvalue()


class TestSafeExtractHappyPath:
    def test_extracts_flat_files(self, tmp_path):
        data = _zip_bytes({"readme.txt": b"hello world", "notes.md": b"# Notes"})
        dest = tmp_path / "out"
        result = safe_extract_zip(data, dest)

        assert result.ok is True
        assert result.rejection_reason is None
        assert {f.relative_path for f in result.extracted} == {"readme.txt", "notes.md"}
        assert (dest / "readme.txt").read_bytes() == b"hello world"
        assert result.total_bytes_written == len(b"hello world") + len(b"# Notes")

    def test_extracts_nested_directories(self, tmp_path):
        data = _zip_bytes({"sub/dir/nested.txt": b"nested content"})
        dest = tmp_path / "out"
        result = safe_extract_zip(data, dest)

        assert result.ok is True
        assert (dest / "sub" / "dir" / "nested.txt").read_text() == "nested content"

    def test_dest_dir_created_with_restrictive_permissions(self, tmp_path):
        data = _zip_bytes({"a.txt": b"x"})
        dest = tmp_path / "out"
        safe_extract_zip(data, dest)
        assert (dest.stat().st_mode & 0o777) == 0o700

    def test_extracted_file_has_restrictive_permissions(self, tmp_path):
        data = _zip_bytes({"a.txt": b"x"})
        dest = tmp_path / "out"
        safe_extract_zip(data, dest)
        assert ((dest / "a.txt").stat().st_mode & 0o777) == 0o600

    def test_empty_archive_ok_with_nothing_extracted(self, tmp_path):
        data = _zip_bytes({})
        dest = tmp_path / "out"
        result = safe_extract_zip(data, dest)
        assert result.ok is True
        assert result.extracted == []


class TestCorruptArchive:
    def test_not_a_zip_file_rejected(self, tmp_path):
        result = safe_extract_zip(b"this is not a zip file at all", tmp_path / "out")
        assert result.ok is False
        assert "not a valid zip" in result.rejection_reason.lower()


class TestPathTraversal:
    def test_dotdot_relative_traversal_skipped_not_extracted(self, tmp_path):
        data = _zip_bytes({"../../etc/evil.txt": b"pwned", "good.txt": b"fine"})
        dest = tmp_path / "out"
        result = safe_extract_zip(data, dest)

        assert result.ok is True
        assert [f.relative_path for f in result.extracted] == ["good.txt"]
        assert result.skipped[0].reason == "unsafe_path"
        # Nothing was written outside dest.
        assert not (tmp_path / "etc").exists()

    def test_absolute_path_entry_skipped(self, tmp_path):
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w") as zf:
            zi = zipfile.ZipInfo("/etc/evil.txt")
            zf.writestr(zi, "pwned")
        dest = tmp_path / "out"
        result = safe_extract_zip(buf.getvalue(), dest)

        assert result.ok is True
        assert result.extracted == []
        assert result.skipped[0].reason == "unsafe_path"

    def test_backslash_traversal_skipped(self, tmp_path):
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w") as zf:
            zi = zipfile.ZipInfo("..\\..\\evil.txt")
            zf.writestr(zi, "pwned")
        dest = tmp_path / "out"
        result = safe_extract_zip(buf.getvalue(), dest)

        assert result.ok is True
        assert result.extracted == []
        assert result.skipped[0].reason == "unsafe_path"

    def test_null_byte_in_name_rejected_by_helper(self):
        """zipfile itself truncates a name at an embedded NUL on write
        (verified: ZipInfo("evil\\x00.txt") round-trips as just "evil"),
        so this can't be exercised through a real archive round-trip --
        test the name-safety helper directly instead."""
        from missy.channels.discord.zip_extract import _is_safe_member_name

        assert _is_safe_member_name("evil\x00.txt") is False


class TestSymlinkEntries:
    def test_symlink_entry_skipped_not_followed(self, tmp_path):
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w") as zf:
            zi = zipfile.ZipInfo("evil_link")
            zi.external_attr = 0o120777 << 16
            zf.writestr(zi, "/etc/passwd")
        dest = tmp_path / "out"
        result = safe_extract_zip(buf.getvalue(), dest)

        assert result.ok is True
        assert result.extracted == []
        assert result.skipped[0].reason == "symlink_not_extracted"
        assert not (dest / "evil_link").is_symlink()


class TestEncryptedEntries:
    def test_encrypted_flag_bit_detected(self):
        """zipfile's writestr doesn't actually encrypt without a dedicated
        AES/ZipCrypto call, so the flag bit itself is tested directly
        rather than round-tripping a real encrypted archive through the
        stdlib -- this is the exact bit safe_extract_zip checks per entry."""
        from missy.channels.discord.zip_extract import _is_encrypted_entry

        data = _zip_bytes({"secret.txt": b"top secret"})
        info = zipfile.ZipFile(io.BytesIO(data)).infolist()[0]
        assert _is_encrypted_entry(info) is False
        info.flag_bits |= 0x1
        assert _is_encrypted_entry(info) is True

    def test_encrypted_entry_skipped_during_extraction(self, tmp_path, monkeypatch):
        """Force is_encrypted true for every entry to verify the
        extraction loop actually consults it (not just the helper in
        isolation)."""
        import missy.channels.discord.zip_extract as zip_extract_module

        monkeypatch.setattr(zip_extract_module, "_is_encrypted_entry", lambda info: True)
        data = _zip_bytes({"secret.txt": b"top secret"})
        result = zip_extract_module.safe_extract_zip(data, tmp_path / "out")

        assert result.ok is True
        assert result.extracted == []
        assert result.skipped[0].reason == "password_protected"


class TestEntryCountAndSizeLimits:
    def test_too_many_entries_rejects_whole_archive(self, tmp_path):
        data = _zip_bytes({f"f{i}.txt": b"x" for i in range(MAX_ZIP_ENTRIES + 5)})
        dest = tmp_path / "out"
        result = safe_extract_zip(data, dest)

        assert result.ok is False
        assert "entries" in result.rejection_reason.lower()
        assert result.extracted == []
        assert not dest.exists()

    def test_at_entry_limit_is_allowed(self, tmp_path):
        data = _zip_bytes({f"f{i}.txt": b"x" for i in range(MAX_ZIP_ENTRIES)})
        dest = tmp_path / "out"
        result = safe_extract_zip(data, dest)
        assert result.ok is True
        assert len(result.extracted) == MAX_ZIP_ENTRIES

    def test_oversized_single_entry_skipped(self, tmp_path):
        # Low-compressibility content (os.urandom) so this trips only the
        # per-entry size cap, not the compression-ratio zip-bomb check
        # that repeated/all-zero bytes would trigger first.
        import os

        oversized = os.urandom(MAX_ZIP_ENTRY_UNCOMPRESSED_BYTES + 1)
        data = _zip_bytes(
            {"big.bin": oversized, "small.txt": b"ok"}, compression=zipfile.ZIP_STORED
        )
        dest = tmp_path / "out"
        result = safe_extract_zip(data, dest)

        assert result.ok is True
        assert [f.relative_path for f in result.extracted] == ["small.txt"]
        assert any(s.name == "big.bin" and s.reason == "entry_too_large" for s in result.skipped)


class TestZipBombDetection:
    def test_high_compression_ratio_rejects_whole_archive(self, tmp_path):
        # Highly compressible content (all zero bytes) triggers the ratio
        # check well before the absolute total-size cap would.
        data = _zip_bytes({"bomb.bin": b"\x00" * (20 * 1024 * 1024)})
        dest = tmp_path / "out"
        result = safe_extract_zip(data, dest)

        assert result.ok is False
        assert "compression ratio" in result.rejection_reason.lower()
        assert not dest.exists()

    def test_total_uncompressed_size_over_cap_rejected(self, tmp_path, monkeypatch):
        import missy.channels.discord.zip_extract as zip_extract_module

        # Lower the total cap so the test doesn't need to build a
        # multi-hundred-MB fixture; use incompressible-ish random-like
        # content per file to avoid tripping the ratio check first.
        monkeypatch.setattr(zip_extract_module, "MAX_ZIP_TOTAL_UNCOMPRESSED_BYTES", 100)
        data = _zip_bytes(
            {"a.bin": bytes(range(256)), "b.bin": bytes(range(256))},
            compression=zipfile.ZIP_STORED,
        )
        result = zip_extract_module.safe_extract_zip(data, tmp_path / "out")

        assert result.ok is False
        assert "exceeding the limit" in result.rejection_reason.lower()


class TestNoWriteOnRejection:
    @pytest.mark.parametrize(
        "build_bad_archive",
        [
            lambda: _zip_bytes({f"f{i}.txt": b"x" for i in range(MAX_ZIP_ENTRIES + 1)}),
            lambda: _zip_bytes({"bomb.bin": b"\x00" * (10 * 1024 * 1024)}),
        ],
    )
    def test_rejected_archive_writes_nothing(self, tmp_path, build_bad_archive):
        dest = tmp_path / "out"
        result = safe_extract_zip(build_bad_archive(), dest)
        assert result.ok is False
        assert not dest.exists()
