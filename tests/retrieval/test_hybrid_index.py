"""Tests for the hybrid dense+sparse index and RRF fusion (F03)."""

from __future__ import annotations

import numpy as np

from missy.retrieval.embedder import HashingEmbedder
from missy.retrieval.hybrid_index import (
    BM25SparseIndex,
    DenseIndex,
    reciprocal_rank_fusion,
)


class TestDenseIndex:
    def test_empty_search_returns_empty(self) -> None:
        idx = DenseIndex(dimension=64)
        assert idx.search(np.zeros(64, dtype=np.float32), 5) == []
        assert idx.size == 0

    def test_add_and_nearest_neighbour(self) -> None:
        emb = HashingEmbedder(dimension=256)
        vecs = emb.embed(
            [
                "vault secret rotation guide",
                "network cidr allowlist policy",
                "vision camera capture pipeline",
            ]
        )
        idx = DenseIndex(dimension=256)
        idx.add([10, 20, 30], vecs)
        assert idx.size == 3
        q = emb.embed(["how to rotate a vault secret"])[0]
        hits = idx.search(q, 3)
        assert hits[0][0] == 10  # the vault doc ranks first

    def test_k_larger_than_size_is_clamped(self) -> None:
        emb = HashingEmbedder(dimension=64)
        idx = DenseIndex(dimension=64)
        idx.add([1, 2], emb.embed(["a", "b"]))
        assert len(idx.search(emb.embed(["a"])[0], 99)) == 2

    def test_dimension_mismatch_raises(self) -> None:
        import pytest

        idx = DenseIndex(dimension=32)
        with pytest.raises(ValueError):
            idx.add([1], np.zeros((1, 16), dtype=np.float32))


class TestBM25:
    def test_empty_search(self) -> None:
        assert BM25SparseIndex().search("anything", 5) == []

    def test_exact_term_match_ranks_first(self) -> None:
        idx = BM25SparseIndex()
        idx.add(1, "the vault stores encrypted secrets")
        idx.add(2, "network policy blocks external hosts")
        idx.add(3, "camera capture and vision analysis")
        hits = idx.search("encrypted secrets vault", 3)
        assert hits[0][0] == 1

    def test_rare_term_outweighs_common_term(self) -> None:
        idx = BM25SparseIndex()
        # "the" is common (low idf); "chacha20" is rare (high idf).
        idx.add(1, "the the the the encryption")
        idx.add(2, "the chacha20 poly1305 cipher")
        hits = dict(idx.search("the chacha20", 2))
        assert hits[2] > hits[1]

    def test_no_match_returns_empty(self) -> None:
        idx = BM25SparseIndex()
        idx.add(1, "completely unrelated content")
        assert idx.search("zzzzz qqqqq", 5) == []


class TestRRF:
    def test_fuses_and_ranks_by_combined_rank(self) -> None:
        dense = [(1, 0.9), (2, 0.8), (3, 0.1)]
        sparse = [(3, 5.0), (1, 4.0), (2, 1.0)]
        fused = reciprocal_rank_fusion([dense, sparse], k=60)
        ids = [i for i, _ in fused]
        # id 1 is rank1+rank2, should top the fused list.
        assert ids[0] == 1
        assert set(ids) == {1, 2, 3}

    def test_top_k_limits_output(self) -> None:
        dense = [(i, 1.0 / (i + 1)) for i in range(10)]
        sparse = [(i, float(10 - i)) for i in range(10)]
        assert len(reciprocal_rank_fusion([dense, sparse], top_k=3)) == 3

    def test_id_in_one_list_still_scored(self) -> None:
        fused = dict(reciprocal_rank_fusion([[(7, 1.0)], []]))
        assert 7 in fused and fused[7] > 0

    def test_empty_lists_return_empty(self) -> None:
        assert reciprocal_rank_fusion([[], []]) == []
