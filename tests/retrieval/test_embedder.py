"""Tests for on-device embedders (F03)."""

from __future__ import annotations

import numpy as np
import pytest

from missy.retrieval.embedder import (
    Embedder,
    HashingEmbedder,
    get_default_embedder,
)


class TestHashingEmbedder:
    def test_shape_and_dtype(self) -> None:
        emb = HashingEmbedder(dimension=128)
        out = emb.embed(["hello world", "another doc"])
        assert out.shape == (2, 128)
        assert out.dtype == np.float32

    def test_rows_are_l2_normalized(self) -> None:
        emb = HashingEmbedder(dimension=256)
        out = emb.embed(["the quick brown fox", "lazy dog sleeps"])
        norms = np.linalg.norm(out, axis=1)
        assert np.allclose(norms, 1.0, atol=1e-5)

    def test_deterministic_across_instances(self) -> None:
        a = HashingEmbedder(dimension=200).embed(["reproducible text"])
        b = HashingEmbedder(dimension=200).embed(["reproducible text"])
        assert np.array_equal(a, b)

    def test_similar_text_scores_higher_than_unrelated(self) -> None:
        emb = HashingEmbedder(dimension=512)
        q = emb.embed(["rotate the api keys in the vault"])[0]
        related = emb.embed(["how to rotate api keys using the vault"])[0]
        unrelated = emb.embed(["the weather is sunny with a chance of rain"])[0]
        assert float(q @ related) > float(q @ unrelated)

    def test_char_ngrams_capture_typo_overlap(self) -> None:
        # Char n-grams should give a nonzero similarity even with a typo.
        emb = HashingEmbedder(dimension=512)
        a = emb.embed(["configuration"])[0]
        b = emb.embed(["configuraton"])[0]  # typo
        assert float(a @ b) > 0.2

    def test_empty_text_is_zero_vector(self) -> None:
        emb = HashingEmbedder(dimension=64)
        out = emb.embed([""])
        assert out.shape == (1, 64)
        # zero (unnormalizable) vector stays all-zero
        assert np.allclose(out[0], 0.0)

    def test_invalid_dimension_raises(self) -> None:
        with pytest.raises(ValueError):
            HashingEmbedder(dimension=0)

    def test_name_and_dimension_properties(self) -> None:
        emb = HashingEmbedder(dimension=384)
        assert emb.dimension == 384
        assert "384" in emb.name

    def test_satisfies_embedder_protocol(self) -> None:
        assert isinstance(HashingEmbedder(), Embedder)


class TestDefaultEmbedder:
    def test_returns_working_embedder_without_optional_deps(self) -> None:
        emb = get_default_embedder(dimension=128)
        out = emb.embed(["works offline"])
        assert out.shape[1] == emb.dimension
        assert isinstance(emb, Embedder)

    def test_falls_back_to_hashing_when_sbert_import_fails(self, monkeypatch) -> None:
        import builtins

        real_import = builtins.__import__

        def fake_import(name, *args, **kwargs):
            if name.startswith("sentence_transformers"):
                raise ImportError("not installed")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", fake_import)
        emb = get_default_embedder(dimension=96)
        assert isinstance(emb, HashingEmbedder)
        assert emb.dimension == 96


class TestSentenceTransformerForwardCompat:
    """The sbert dimension probe must handle the >=5.x method rename."""

    def _fake_st_module(self, model):
        import types

        mod = types.ModuleType("sentence_transformers")
        mod.SentenceTransformer = lambda name: model
        return mod

    def test_uses_new_get_embedding_dimension(self, monkeypatch) -> None:
        import sys
        from unittest.mock import MagicMock

        model = MagicMock(spec=["get_embedding_dimension", "encode"])
        model.get_embedding_dimension.return_value = 384
        monkeypatch.setitem(sys.modules, "sentence_transformers", self._fake_st_module(model))
        from missy.retrieval.embedder import SentenceTransformerEmbedder

        emb = SentenceTransformerEmbedder("fake-model")
        assert emb.dimension == 384
        model.get_embedding_dimension.assert_called_once()

    def test_falls_back_to_old_method_name(self, monkeypatch) -> None:
        import sys
        from unittest.mock import MagicMock

        # Only the deprecated name exists (older sentence-transformers).
        model = MagicMock(spec=["get_sentence_embedding_dimension", "encode"])
        model.get_sentence_embedding_dimension.return_value = 768
        monkeypatch.setitem(sys.modules, "sentence_transformers", self._fake_st_module(model))
        from missy.retrieval.embedder import SentenceTransformerEmbedder

        emb = SentenceTransformerEmbedder("fake-model")
        assert emb.dimension == 768
