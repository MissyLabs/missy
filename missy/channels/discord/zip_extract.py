"""Safe extraction of untrusted zip archives.

Extracts an in-memory zip file to a destination directory while guarding
against the standard hazards of extracting an attacker-controlled
archive:

- **Zip-slip / path traversal**: an entry name like ``../../etc/passwd``
  or an absolute path escaping the destination directory. Every entry's
  resolved destination is verified to stay under ``dest_dir`` before
  anything is written.
- **Zip bombs**: a small download that expands to an enormous amount of
  data (or an enormous number of files), exhausting disk/memory. Guarded
  by a hard cap on total uncompressed size, entry count, per-entry size,
  and the overall compression ratio (a classic zip-bomb signature) —
  checked from the archive's central directory *before* extracting
  anything, and re-checked while streaming each entry's bytes in case the
  central directory under-reports (a crafted/corrupt entry).
- **Symlinks**: an entry can declare itself a symlink via its Unix
  external-file-attributes; such entries are skipped rather than
  followed, since a symlink could point outside ``dest_dir`` entirely.
- **Encrypted entries**: entries we have no password for are skipped
  rather than silently failing partway through.

A single dangerous *entry* is skipped (extraction continues with the
rest of the archive); a property of the *whole archive* (too many
entries, too large in total, corrupt, or a bomb-like compression ratio)
rejects the entire archive before writing anything.
"""

from __future__ import annotations

import contextlib
import os
import zipfile
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath

#: Hard cap on how many entries a single archive may contain.
MAX_ZIP_ENTRIES = 500

#: Hard cap on the sum of every entry's *uncompressed* size -- this is
#: the actual zip-bomb backstop (the compressed download itself is
#: already bounded by MAX_ZIP_ATTACHMENT_BYTES in zip_attachment.py, but
#: a small download can still expand to gigabytes on disk without this).
MAX_ZIP_TOTAL_UNCOMPRESSED_BYTES = 200 * 1024 * 1024

#: Hard cap on any single entry's uncompressed size.
MAX_ZIP_ENTRY_UNCOMPRESSED_BYTES = 50 * 1024 * 1024

#: uncompressed / compressed ratio above which an archive is treated as
#: a likely zip bomb (classic examples like "42.zip" exceed 1000:1; a
#: legitimate archive of already-compressed content rarely exceeds
#: single digits). Only applied when the archive's total compressed size
#: is non-trivial, to avoid flagging e.g. a handful of tiny text files
#: that legitimately compress from a few bytes to fewer bytes.
MAX_ZIP_COMPRESSION_RATIO = 100
_RATIO_CHECK_MIN_COMPRESSED_BYTES = 4096

#: Read/verify entry bytes in chunks rather than materializing an entire
#: (declared) file size at once.
_CHUNK_SIZE = 1024 * 1024


@dataclass(frozen=True)
class ExtractedFile:
    """One file successfully written to disk during extraction."""

    relative_path: str
    absolute_path: str
    size: int


@dataclass(frozen=True)
class SkippedEntry:
    """One archive entry that was not extracted, and why."""

    name: str
    reason: str


@dataclass
class ZipExtractionResult:
    """Outcome of a :func:`safe_extract_zip` call."""

    ok: bool
    dest_dir: str
    rejection_reason: str | None = None
    extracted: list[ExtractedFile] = field(default_factory=list)
    skipped: list[SkippedEntry] = field(default_factory=list)
    total_bytes_written: int = 0


def _is_safe_member_name(name: str) -> bool:
    """Reject anything that could escape ``dest_dir`` by name alone.

    Zip entry names are conventionally forward-slash-separated, but a
    maliciously crafted archive (or one built on Windows) could carry
    backslashes intending them as path separators on extraction, or an
    absolute path / drive letter. All of that is rejected outright here,
    before any path-resolution math is attempted.
    """
    if not name or "\x00" in name:
        return False
    normalized = name.replace("\\", "/")
    if normalized.startswith("/"):
        return False
    if len(normalized) >= 2 and normalized[1] == ":":  # e.g. "C:/..."
        return False
    parts = PurePosixPath(normalized).parts
    return ".." not in parts


def _is_symlink_entry(info: zipfile.ZipInfo) -> bool:
    mode = (info.external_attr >> 16) & 0o170000
    return mode == 0o120000


def _is_encrypted_entry(info: zipfile.ZipInfo) -> bool:
    return bool(info.flag_bits & 0x1)


def safe_extract_zip(data: bytes, dest_dir: Path) -> ZipExtractionResult:
    """Safely extract an in-memory zip archive to *dest_dir*.

    Args:
        data: Raw zip file bytes.
        dest_dir: Directory to extract into. Created (mode ``0o700``) if
            it doesn't already exist. Must not already exist with other
            content the caller cares about protecting -- callers should
            pass a fresh, dedicated directory per archive.

    Returns:
        A :class:`ZipExtractionResult`. ``ok=False`` means the archive as
        a whole was rejected (corrupt, too many entries, too large, or a
        suspected zip bomb) and *nothing* was written. ``ok=True`` means
        extraction proceeded, though individual dangerous entries may
        still have been skipped (see ``skipped``).
    """
    import io

    dest_dir = Path(dest_dir)

    try:
        zf = zipfile.ZipFile(io.BytesIO(data))
    except zipfile.BadZipFile as exc:
        return ZipExtractionResult(
            ok=False, dest_dir=str(dest_dir), rejection_reason=f"Not a valid zip file: {exc}"
        )

    with zf:
        try:
            infolist = zf.infolist()
        except Exception as exc:
            return ZipExtractionResult(
                ok=False,
                dest_dir=str(dest_dir),
                rejection_reason=f"Could not read zip central directory: {exc}",
            )

        if len(infolist) > MAX_ZIP_ENTRIES:
            return ZipExtractionResult(
                ok=False,
                dest_dir=str(dest_dir),
                rejection_reason=(
                    f"Archive has {len(infolist)} entries, exceeding the limit of "
                    f"{MAX_ZIP_ENTRIES}."
                ),
            )

        total_uncompressed = sum(i.file_size for i in infolist)
        total_compressed = sum(i.compress_size for i in infolist)

        if total_uncompressed > MAX_ZIP_TOTAL_UNCOMPRESSED_BYTES:
            return ZipExtractionResult(
                ok=False,
                dest_dir=str(dest_dir),
                rejection_reason=(
                    f"Archive expands to {total_uncompressed:,} bytes, exceeding the "
                    f"limit of {MAX_ZIP_TOTAL_UNCOMPRESSED_BYTES:,} bytes."
                ),
            )

        if (
            total_compressed > _RATIO_CHECK_MIN_COMPRESSED_BYTES
            and total_uncompressed / total_compressed > MAX_ZIP_COMPRESSION_RATIO
        ):
            ratio = total_uncompressed / total_compressed
            return ZipExtractionResult(
                ok=False,
                dest_dir=str(dest_dir),
                rejection_reason=(
                    f"Archive's compression ratio ({ratio:.0f}:1) exceeds the zip-bomb "
                    f"threshold of {MAX_ZIP_COMPRESSION_RATIO}:1."
                ),
            )

        os.makedirs(dest_dir, mode=0o700, exist_ok=True)
        os.chmod(dest_dir, 0o700)
        dest_root = dest_dir.resolve()

        result = ZipExtractionResult(ok=True, dest_dir=str(dest_dir))

        for info in infolist:
            name = info.filename

            if not _is_safe_member_name(name):
                result.skipped.append(SkippedEntry(name=name, reason="unsafe_path"))
                continue

            if _is_symlink_entry(info):
                result.skipped.append(SkippedEntry(name=name, reason="symlink_not_extracted"))
                continue

            if _is_encrypted_entry(info):
                result.skipped.append(SkippedEntry(name=name, reason="password_protected"))
                continue

            is_dir = name.endswith(("/", "\\"))
            target = (dest_dir / name.replace("\\", "/")).resolve()
            if target != dest_root and dest_root not in target.parents:
                result.skipped.append(SkippedEntry(name=name, reason="path_traversal"))
                continue

            if is_dir:
                os.makedirs(target, mode=0o700, exist_ok=True)
                os.chmod(target, 0o700)
                continue

            if info.file_size > MAX_ZIP_ENTRY_UNCOMPRESSED_BYTES:
                result.skipped.append(SkippedEntry(name=name, reason="entry_too_large"))
                continue

            os.makedirs(target.parent, mode=0o700, exist_ok=True)

            try:
                written = _extract_one(zf, info, target)
            except _EntryTooLarge:
                with contextlib.suppress(Exception):
                    target.unlink(missing_ok=True)
                result.skipped.append(
                    SkippedEntry(name=name, reason="entry_exceeded_declared_size")
                )
                continue
            except Exception as exc:
                with contextlib.suppress(Exception):
                    target.unlink(missing_ok=True)
                result.skipped.append(SkippedEntry(name=name, reason=f"extract_error: {exc}"))
                continue

            result.extracted.append(
                ExtractedFile(
                    relative_path=name,
                    absolute_path=str(target),
                    size=written,
                )
            )
            result.total_bytes_written += written

    return result


class _EntryTooLarge(Exception):
    """Raised when an entry streams more bytes than its declared size allows."""


def _extract_one(zf: zipfile.ZipFile, info: zipfile.ZipInfo, target: Path) -> int:
    """Stream one entry's bytes to *target*, enforcing the per-entry cap
    against actual bytes read (not just the archive's declared size, which
    a crafted entry could under-report)."""
    written = 0
    fd = os.open(str(target), os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    with os.fdopen(fd, "wb") as out, zf.open(info) as src:
        while True:
            chunk = src.read(_CHUNK_SIZE)
            if not chunk:
                break
            written += len(chunk)
            if written > MAX_ZIP_ENTRY_UNCOMPRESSED_BYTES:
                raise _EntryTooLarge(info.filename)
            out.write(chunk)
    return written
