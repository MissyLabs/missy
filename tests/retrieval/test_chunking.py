"""Tests for span-preserving chunking (F03)."""

from __future__ import annotations

from missy.retrieval.chunking import Chunk, chunk_text


class TestSpanAccuracy:
    def test_every_chunk_span_matches_source(self) -> None:
        text = (
            "Alpha paragraph about vaults and secrets.\n\n"
            "Beta paragraph about network policy. Gamma sentence here. "
            "Delta sentence follows.\n\n"
            "Epsilon closing paragraph with more words to force multiple chunks."
        ) * 4
        chunks = chunk_text(text, doc_id="doc", max_chars=150, overlap=30)
        assert chunks
        for c in chunks:
            assert text[c.start : c.end] == c.text

    def test_chunks_are_ordered_and_indexed(self) -> None:
        text = "one two three. " * 200
        chunks = chunk_text(text, max_chars=100, overlap=10)
        assert [c.index for c in chunks] == list(range(len(chunks)))
        # start offsets are non-decreasing
        starts = [c.start for c in chunks]
        assert starts == sorted(starts)


class TestBoundaries:
    def test_empty_and_whitespace_yield_no_chunks(self) -> None:
        assert chunk_text("") == []
        assert chunk_text("   \n\n  \t ") == []

    def test_short_text_is_single_chunk(self) -> None:
        chunks = chunk_text("just a short note", max_chars=800)
        assert len(chunks) == 1
        assert chunks[0].text == "just a short note"
        assert chunks[0].start == 0

    def test_oversized_segment_is_hard_split(self) -> None:
        # A single unbroken token longer than max_chars must still be split.
        text = "x" * 500
        chunks = chunk_text(text, max_chars=100, overlap=0)
        assert len(chunks) >= 5
        # Reassembling the non-overlapping cuts covers the whole source.
        assert chunks[0].start == 0
        assert chunks[-1].end == 500

    def test_metadata_and_doc_id_propagate(self) -> None:
        chunks = chunk_text(
            "hello world. " * 50,
            doc_id="notes",
            max_chars=80,
            metadata={"source": "unit"},
        )
        assert all(c.doc_id == "notes" for c in chunks)
        assert all(c.metadata == {"source": "unit"} for c in chunks)
        # metadata is copied, not shared
        chunks[0].metadata["source"] = "mutated"
        assert chunks[1].metadata["source"] == "unit"

    def test_overlap_clamped_below_max_chars(self) -> None:
        # overlap >= max_chars must not deadlock or raise.
        chunks = chunk_text("word " * 300, max_chars=50, overlap=999)
        assert chunks
        assert all(isinstance(c, Chunk) for c in chunks)

    def test_invalid_max_chars_raises(self) -> None:
        import pytest

        with pytest.raises(ValueError):
            chunk_text("hi", max_chars=0)
