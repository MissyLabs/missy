"""Span-preserving text chunking for the on-device retrieval engine (F03).

Chunks keep the exact character offsets (``start``/``end``) they occupy in
their source document so that retrieval results can cite a precise
``source_span`` back into the original text rather than an opaque snippet.

The splitter prefers natural boundaries — paragraph breaks, then sentence
ends — and only falls back to a hard character cut when a single unbroken
run exceeds ``max_chars``. Consecutive chunks overlap by ``overlap``
characters so a passage straddling a boundary is still recallable from at
least one chunk.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

# Sentence/paragraph boundary detectors. Kept deliberately simple (no NLP
# dependency): a paragraph is a blank line; a sentence ends at ``.?!`` followed
# by whitespace. Good enough to bias cuts toward readable boundaries.
_PARA_RE = re.compile(r"\n\s*\n")
_SENT_END_RE = re.compile(r"(?<=[.!?])\s+")


@dataclass
class Chunk:
    """A contiguous slice of a source document with its character span.

    Attributes:
        text: The chunk's text (exactly ``source[start:end]``).
        start: Inclusive character offset of the chunk in the source doc.
        end: Exclusive character offset of the chunk in the source doc.
        doc_id: Identifier of the source document.
        index: 0-based position of this chunk within the document.
        metadata: Arbitrary per-document metadata carried onto the chunk.
    """

    text: str
    start: int
    end: int
    doc_id: str = ""
    index: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)


def _segment_bounds(text: str) -> list[tuple[int, int]]:
    """Return (start, end) spans of natural segments (paragraphs → sentences).

    Every returned span indexes back into ``text`` exactly; the spans cover
    all non-whitespace content in order.
    """
    segments: list[tuple[int, int]] = []
    pos = 0
    for para in _PARA_RE.split(text):
        if not para:
            pos += 2  # account for the split separator length lower-bound
        # Find the paragraph's real offset from ``pos`` to stay span-accurate
        # even when separators vary in length.
        start = text.find(para, pos) if para else pos
        if start < 0:
            start = pos
        end = start + len(para)
        pos = end
        # Sub-split the paragraph into sentences, preserving offsets.
        sent_start = start
        for piece in _SENT_END_RE.split(para):
            if not piece:
                continue
            s = text.find(piece, sent_start)
            if s < 0:
                s = sent_start
            e = s + len(piece)
            segments.append((s, e))
            sent_start = e
    return [(s, e) for (s, e) in segments if e > s]


def chunk_text(
    text: str,
    *,
    doc_id: str = "",
    max_chars: int = 800,
    overlap: int = 150,
    metadata: dict[str, Any] | None = None,
) -> list[Chunk]:
    """Split ``text`` into span-preserving chunks.

    Args:
        text: Source document text.
        doc_id: Identifier stamped onto every produced chunk.
        max_chars: Soft maximum chunk size. A chunk is emitted once adding the
            next segment would exceed this (segments larger than ``max_chars``
            on their own are hard-split).
        overlap: Number of trailing characters of each chunk to prepend as
            context to the next (bounded to ``max_chars - 1``).
        metadata: Optional metadata copied onto every chunk.

    Returns:
        Chunks in document order, each with accurate ``start``/``end`` offsets.
    """
    if max_chars <= 0:
        raise ValueError("max_chars must be positive")
    overlap = max(0, min(overlap, max_chars - 1))
    meta = dict(metadata or {})
    if not text.strip():
        return []

    segments = _segment_bounds(text) or [(0, len(text))]

    # Expand any segment longer than max_chars into hard sub-slices.
    expanded: list[tuple[int, int]] = []
    for s, e in segments:
        if e - s <= max_chars:
            expanded.append((s, e))
            continue
        cur = s
        while cur < e:
            expanded.append((cur, min(cur + max_chars, e)))
            cur += max_chars

    chunks: list[Chunk] = []
    cur_start: int | None = None
    cur_end: int | None = None
    for s, e in expanded:
        if cur_start is None:
            cur_start, cur_end = s, e
            continue
        assert cur_end is not None
        # Would appending this segment overflow the current chunk?
        if e - cur_start > max_chars:
            chunks.append(_mk(text, cur_start, cur_end, doc_id, len(chunks), meta))
            # Start the next chunk with overlap back into the just-emitted one.
            back = max(cur_start, cur_end - overlap) if overlap else s
            cur_start = min(back, s)
            cur_end = e
        else:
            cur_end = e
    if cur_start is not None and cur_end is not None:
        chunks.append(_mk(text, cur_start, cur_end, doc_id, len(chunks), meta))
    return chunks


def _mk(text: str, start: int, end: int, doc_id: str, index: int, meta: dict[str, Any]) -> Chunk:
    return Chunk(
        text=text[start:end],
        start=start,
        end=end,
        doc_id=doc_id,
        index=index,
        metadata=dict(meta),
    )
