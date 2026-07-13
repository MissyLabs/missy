"""Discord text-file attachment validation.

Companion to :mod:`missy.channels.discord.image_analyze` — validates
Discord attachment *metadata* (content type, extension, size) for
plain-text-like files (``.md``, ``.txt``, ``.json``, ``.yaml``, ``.csv``,
``.log``) before download, so a request like "here's the spec, read it"
with an attached ``.md`` file can actually reach the model as text
instead of being denied outright.

Downloading, decoding, and prompt-injection scanning of the actual bytes
happens in :mod:`missy.cli.main`'s Discord message-processing loop — this
module only validates metadata, matching how
:func:`~missy.channels.discord.image_analyze.validate_image_attachment`
is scoped.
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

# Text attachments are spliced directly into the prompt as plain text, so
# the cap is far smaller than the image cap (MAX_IMAGE_ATTACHMENT_BYTES) --
# a large file would otherwise blow the context budget on a single
# attachment.
MAX_TEXT_ATTACHMENT_BYTES = 256 * 1024

_TEXT_CONTENT_TYPES = frozenset(
    {
        "text/plain",
        "text/markdown",
        "text/x-markdown",
        "application/json",
        "text/yaml",
        "application/yaml",
        "application/x-yaml",
        "text/csv",
        "text/x-log",
    }
)

_TEXT_EXTENSIONS = frozenset(
    {
        ".md",
        ".markdown",
        ".txt",
        ".json",
        ".yaml",
        ".yml",
        ".csv",
        ".log",
    }
)


def is_text_attachment(attachment: dict[str, Any]) -> bool:
    """Return True if the Discord attachment dict looks like a text file.

    Args:
        attachment: Discord attachment dict.

    Returns:
        ``True`` when the content type or filename extension matches a
        recognised text-like format.
    """
    ct = _normalise_content_type(attachment)
    if ct in _TEXT_CONTENT_TYPES:
        return True
    filename = (attachment.get("filename") or "").lower()
    return any(filename.endswith(ext) for ext in _TEXT_EXTENSIONS)


def validate_text_attachment(attachment: dict[str, Any]) -> AttachmentValidation:
    """Validate Discord text-attachment metadata before routing or download.

    Mirrors :func:`~missy.channels.discord.image_analyze.validate_image_attachment`:
    checks the download URL is a genuine Discord CDN URL, the content
    type/extension is a recognised text format, and the declared size is
    within :data:`MAX_TEXT_ATTACHMENT_BYTES`. Deliberately does not
    require an exact content-type/extension match the way the image
    validator does — Discord (and many clients) send arbitrary or absent
    content types for text files far more inconsistently than for
    well-known image MIME types, so extension-based recognition alone is
    treated as sufficient for text.

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

    if content_type and content_type not in _TEXT_CONTENT_TYPES and ext not in _TEXT_EXTENSIONS:
        reasons.append("unsupported_content_type")
    elif not content_type and ext not in _TEXT_EXTENSIONS:
        reasons.append("unsupported_file_extension")

    size = _int_or_none(attachment.get("size"))
    if size is not None:
        if size < 0:
            reasons.append("invalid_size")
        elif size > MAX_TEXT_ATTACHMENT_BYTES:
            reasons.append("text_attachment_too_large")

    details = {
        "filename": safe_filename,
        "content_type": content_type,
        "size": size,
        "url_host": host or _attachment_host(url),
        "max_size": MAX_TEXT_ATTACHMENT_BYTES,
    }
    return AttachmentValidation(allowed=not reasons, reasons=reasons, details=details)
