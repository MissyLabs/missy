"""Discord .zip attachment validation.

Companion to :mod:`missy.channels.discord.text_attachment` — validates
Discord attachment *metadata* (content type, extension, size) for zip
archives before download, mirroring how
:func:`~missy.channels.discord.image_analyze.validate_image_attachment`
and :func:`~missy.channels.discord.text_attachment.validate_text_attachment`
are scoped. The actual download and safe extraction of the archive's
*contents* happens in :mod:`missy.channels.discord.zip_extract`
(:func:`~missy.channels.discord.zip_extract.safe_extract_zip`), driven
from :mod:`missy.channels.discord.attachment_context` — this module only
validates metadata.
"""

from __future__ import annotations

from typing import Any
from urllib.parse import urlparse

from missy.channels.discord.image_analyze import (
    _DISCORD_ATTACHMENT_HOSTS,
    AttachmentValidation,
    _attachment_host,
    _filename_extension,
    _int_or_none,
    _normalise_content_type,
    sanitize_attachment_filename,
)

# This is a cap on the *compressed* download size, not on what the archive
# expands to -- that's separately bounded by zip_extract.py's
# MAX_ZIP_TOTAL_UNCOMPRESSED_BYTES/MAX_ZIP_COMPRESSION_RATIO, which is
# where an actual zip-bomb (a small download that expands to gigabytes)
# gets caught. This cap just keeps the download itself bounded, matching
# MAX_IMAGE_ATTACHMENT_BYTES's role for images.
MAX_ZIP_ATTACHMENT_BYTES = 25 * 1024 * 1024

_ZIP_CONTENT_TYPES = frozenset(
    {
        "application/zip",
        "application/x-zip-compressed",
        "application/x-zip",
        "multipart/x-zip",
    }
)

_ZIP_EXTENSIONS = frozenset({".zip"})


def is_zip_attachment(attachment: dict[str, Any]) -> bool:
    """Return True if the Discord attachment dict looks like a zip archive.

    Args:
        attachment: Discord attachment dict.

    Returns:
        ``True`` when the content type or filename extension matches a
        recognised zip format.
    """
    ct = _normalise_content_type(attachment)
    if ct in _ZIP_CONTENT_TYPES:
        return True
    filename = (attachment.get("filename") or "").lower()
    return any(filename.endswith(ext) for ext in _ZIP_EXTENSIONS)


def validate_zip_attachment(attachment: dict[str, Any]) -> AttachmentValidation:
    """Validate Discord zip-attachment metadata before routing or download.

    Mirrors :func:`~missy.channels.discord.text_attachment.validate_text_attachment`:
    checks the download URL is a genuine Discord CDN URL, the content
    type/extension is recognised as a zip archive, and the declared size
    is within :data:`MAX_ZIP_ATTACHMENT_BYTES`. This only validates the
    outer attachment metadata -- the archive's *contents* are validated
    separately and much more strictly at extraction time (entry count,
    total uncompressed size, compression ratio, path traversal, symlinks;
    see :mod:`missy.channels.discord.zip_extract`), since none of that is
    knowable from Discord's attachment metadata alone.

    Args:
        attachment: Discord attachment dict.

    Returns:
        An :class:`AttachmentValidation` with ``allowed`` set and, when
        not allowed, machine-readable ``reasons``.
    """
    reasons: list[str] = []
    raw_filename = str(attachment.get("filename") or "")
    safe_filename = sanitize_attachment_filename(raw_filename)
    ext = _filename_extension(safe_filename)
    content_type = _normalise_content_type(attachment)
    url = str(attachment.get("url") or attachment.get("proxy_url") or "")
    parsed = urlparse(url) if url else None
    host = (parsed.hostname or "").lower() if parsed else ""

    if not url:
        reasons.append("missing_url")
    elif parsed.scheme != "https" or host not in _DISCORD_ATTACHMENT_HOSTS:
        reasons.append("invalid_discord_cdn_url")

    if content_type and content_type not in _ZIP_CONTENT_TYPES and ext not in _ZIP_EXTENSIONS:
        reasons.append("unsupported_content_type")
    elif not content_type and ext not in _ZIP_EXTENSIONS:
        reasons.append("unsupported_file_extension")

    size = _int_or_none(attachment.get("size"))
    if size is not None:
        if size < 0:
            reasons.append("invalid_size")
        elif size > MAX_ZIP_ATTACHMENT_BYTES:
            reasons.append("zip_attachment_too_large")

    details = {
        "filename": safe_filename,
        "content_type": content_type,
        "size": size,
        "url_host": host or _attachment_host(url),
        "max_size": MAX_ZIP_ATTACHMENT_BYTES,
    }
    return AttachmentValidation(allowed=not reasons, reasons=reasons, details=details)
