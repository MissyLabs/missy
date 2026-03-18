"""Tests for VectorMemoryStore — targeting uncovered lines (62% → ~95%).

Covers:
- No-FAISS graceful degradation (all methods)
- add/search/save/load with mocked FAISS
- Search with empty index, out-of-range indices
- Save/load file I/O paths
- SimpleVectorizer edge cases
"""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock

import pytest

# ---------------------------------------------------------------------------
# SimpleVectorizer tests (always available, no FAISS needed)
# ---------------------------------------------------------------------------
from missy.memory.vector_store import SimpleVectorizer, _tokenize


class TestTokenize:
    def test_basic(self):
        assert _tokenize("Hello World") == ["hello", "world"]

    def test_numbers(self):
        assert _tokenize("test123 abc") == ["test123", "abc"]

    def test_empty(self):
        assert _tokenize("") == []

    def test_special_chars_stripped(self):
        assert _tokenize("foo-bar_baz!") == ["foo", "bar", "baz"]

    def test_unicode_stripped(self):
        # Only alphanumeric kept
        assert _tokenize("café résumé") == ["caf", "r", "sum"]


class TestSimpleVectorizer:
    def test_default_dimension(self):
        v = SimpleVectorizer()
        assert v.dimension == 384

    def test_custom_dimension(self):
        v = SimpleVectorizer(dimension=64)
        assert v.dimension == 64

    def test_encode_basic(self):
        v = SimpleVectorizer(dimension=64)
        vec = v.encode("hello world")
        assert len(vec) == 64
        # Should be L2-normalized (norm ≈ 1.0)
        norm = sum(x * x for x in vec) ** 0.5
        assert abs(norm - 1.0) < 1e-6

    def test_encode_empty_text(self):
        v = SimpleVectorizer(dimension=64)
        vec = v.encode("")
        assert len(vec) == 64
        assert all(x == 0.0 for x in vec)

    def test_encode_special_chars_only(self):
        v = SimpleVectorizer(dimension=64)
        vec = v.encode("!@#$%^&*()")
        # No tokens → zero vector
        assert all(x == 0.0 for x in vec)

    def test_encode_deterministic(self):
        v = SimpleVectorizer(dimension=64)
        v1 = v.encode("test input")
        v2 = v.encode("test input")
        assert v1 == v2

    def test_encode_different_texts_differ(self):
        v = SimpleVectorizer(dimension=64)
        v1 = v.encode("deployment failed")
        v2 = v.encode("unicorn rainbow")
        assert v1 != v2

    def test_encode_repeated_tokens(self):
        v = SimpleVectorizer(dimension=64)
        vec = v.encode("test test test test")
        norm = sum(x * x for x in vec) ** 0.5
        assert abs(norm - 1.0) < 1e-6


# ---------------------------------------------------------------------------
# VectorMemoryStore tests with mocked FAISS
# ---------------------------------------------------------------------------


class FakeNPArray:
    """Minimal numpy array stand-in for tests."""

    def __init__(self, data, dtype=None):
        self.data = data
        self.dtype = dtype

    def __getitem__(self, idx):
        return self.data[idx]

    def __iter__(self):
        return iter(self.data)

    def __len__(self):
        return len(self.data)


class FakeNP:
    """Minimal numpy module stand-in."""

    float32 = "float32"
    int64 = "int64"

    @staticmethod
    def array(data, dtype=None):
        return FakeNPArray(data, dtype=dtype)


@pytest.fixture
def mock_faiss():
    """Create a fake faiss module with IndexFlatL2."""

    class FakeIndex:
        def __init__(self, dim):
            self.dim = dim
            self._vectors = []
            self.ntotal = 0

        def add(self, arr):
            for row in arr:
                self._vectors.append(list(row))
                self.ntotal += 1

        def search(self, arr, k):
            np = FakeNP()
            if self.ntotal == 0:
                return np.array([[]]), np.array([[-1]])

            query = arr[0]
            dists = []
            for i, v in enumerate(self._vectors):
                d = sum((a - b) ** 2 for a, b in zip(query, v, strict=False))
                dists.append((d, i))
            dists.sort()
            k = min(k, len(dists))
            d_arr = np.array([[d for d, _ in dists[:k]]])
            i_arr = np.array([[i for _, i in dists[:k]]])
            return d_arr, i_arr

    fake_faiss = types.ModuleType("faiss")
    fake_faiss.IndexFlatL2 = FakeIndex
    fake_faiss.write_index = MagicMock()
    fake_faiss.read_index = MagicMock(return_value=FakeIndex(384))
    return fake_faiss


@pytest.fixture
def mock_np():
    """Create a fake numpy module."""
    fake_np = types.ModuleType("numpy")
    fake_np.float32 = "float32"
    fake_np.int64 = "int64"
    fake_np.array = FakeNP.array
    return fake_np


@pytest.fixture
def vector_store_with_faiss(mock_faiss, mock_np, tmp_path):
    """Create a VectorMemoryStore with mocked FAISS and numpy."""
    import missy.memory.vector_store as vs_mod

    original_available = vs_mod._FAISS_AVAILABLE
    original_faiss = sys.modules.get("faiss")
    original_np = sys.modules.get("numpy")

    vs_mod._FAISS_AVAILABLE = True
    sys.modules["faiss"] = mock_faiss
    sys.modules["numpy"] = mock_np

    original_mod_faiss = getattr(vs_mod, "faiss", None)
    vs_mod.faiss = mock_faiss

    store = vs_mod.VectorMemoryStore(
        dimension=64,
        index_path=str(tmp_path / "test.faiss"),
    )

    yield store

    vs_mod._FAISS_AVAILABLE = original_available
    if original_faiss is not None:
        sys.modules["faiss"] = original_faiss
    elif "faiss" in sys.modules:
        del sys.modules["faiss"]
    if original_np is not None:
        sys.modules["numpy"] = original_np
    elif "numpy" in sys.modules:
        del sys.modules["numpy"]
    if original_mod_faiss is not None:
        vs_mod.faiss = original_mod_faiss


class TestVectorMemoryStoreNoFaiss:
    """Test graceful degradation when FAISS is not installed."""

    def test_init_no_faiss(self, tmp_path):
        import missy.memory.vector_store as vs_mod

        original = vs_mod._FAISS_AVAILABLE
        vs_mod._FAISS_AVAILABLE = False
        try:
            store = vs_mod.VectorMemoryStore(
                dimension=64,
                index_path=str(tmp_path / "nope.faiss"),
            )
            assert store._index is None
        finally:
            vs_mod._FAISS_AVAILABLE = original

    def test_add_no_faiss(self, tmp_path):
        import missy.memory.vector_store as vs_mod

        original = vs_mod._FAISS_AVAILABLE
        vs_mod._FAISS_AVAILABLE = False
        try:
            store = vs_mod.VectorMemoryStore(
                dimension=64,
                index_path=str(tmp_path / "nope.faiss"),
            )
            # Should not raise
            store.add("some text", {"key": "value"})
        finally:
            vs_mod._FAISS_AVAILABLE = original

    def test_search_no_faiss(self, tmp_path):
        import missy.memory.vector_store as vs_mod

        original = vs_mod._FAISS_AVAILABLE
        vs_mod._FAISS_AVAILABLE = False
        try:
            store = vs_mod.VectorMemoryStore(
                dimension=64,
                index_path=str(tmp_path / "nope.faiss"),
            )
            results = store.search("query")
            assert results == []
        finally:
            vs_mod._FAISS_AVAILABLE = original

    def test_save_no_faiss(self, tmp_path):
        import missy.memory.vector_store as vs_mod

        original = vs_mod._FAISS_AVAILABLE
        vs_mod._FAISS_AVAILABLE = False
        try:
            store = vs_mod.VectorMemoryStore(
                dimension=64,
                index_path=str(tmp_path / "nope.faiss"),
            )
            # Should not raise, should not create files
            store.save()
            assert not (tmp_path / "nope.faiss").exists()
        finally:
            vs_mod._FAISS_AVAILABLE = original

    def test_load_no_faiss(self, tmp_path):
        import missy.memory.vector_store as vs_mod

        original = vs_mod._FAISS_AVAILABLE
        vs_mod._FAISS_AVAILABLE = False
        try:
            store = vs_mod.VectorMemoryStore(
                dimension=64,
                index_path=str(tmp_path / "nope.faiss"),
            )
            # Should not raise
            store.load()
        finally:
            vs_mod._FAISS_AVAILABLE = original

    def test_count_no_faiss(self, tmp_path):
        import missy.memory.vector_store as vs_mod

        original = vs_mod._FAISS_AVAILABLE
        vs_mod._FAISS_AVAILABLE = False
        try:
            store = vs_mod.VectorMemoryStore(
                dimension=64,
                index_path=str(tmp_path / "nope.faiss"),
            )
            assert store.count() == 0
        finally:
            vs_mod._FAISS_AVAILABLE = original


class TestVectorMemoryStoreWithFaiss:
    """Test with mocked FAISS available."""

    def test_add_and_search(self, vector_store_with_faiss):
        store = vector_store_with_faiss
        store.add("deployment failed due to missing env var", {"category": "solution"})
        store.add("network timeout connecting to database", {"category": "error"})
        store.add("user authentication token expired", {"category": "auth"})

        results = store.search("environment variable error", top_k=3)
        assert len(results) > 0
        assert "text" in results[0]
        assert "metadata" in results[0]
        assert "score" in results[0]

    def test_search_empty_index(self, vector_store_with_faiss):
        store = vector_store_with_faiss
        results = store.search("anything")
        assert results == []

    def test_add_no_metadata(self, vector_store_with_faiss):
        store = vector_store_with_faiss
        store.add("plain text entry")
        assert store.count() == 1
        results = store.search("plain text")
        assert len(results) == 1
        assert results[0]["metadata"] == {}

    def test_count(self, vector_store_with_faiss):
        store = vector_store_with_faiss
        assert store.count() == 0
        store.add("entry 1")
        assert store.count() == 1
        store.add("entry 2")
        assert store.count() == 2

    def test_search_top_k_limits(self, vector_store_with_faiss):
        store = vector_store_with_faiss
        for i in range(10):
            store.add(f"entry number {i}")
        results = store.search("entry", top_k=3)
        assert len(results) == 3

    def test_search_top_k_exceeds_total(self, vector_store_with_faiss):
        store = vector_store_with_faiss
        store.add("only one entry")
        results = store.search("entry", top_k=100)
        assert len(results) == 1

    def test_save(self, vector_store_with_faiss, mock_faiss, tmp_path):
        store = vector_store_with_faiss
        store.add("test entry", {"k": "v"})
        store.save()

        # faiss.write_index should have been called
        mock_faiss.write_index.assert_called_once()

        # Metadata file should exist
        meta_path = Path(store._metadata_path)
        assert meta_path.exists()
        data = json.loads(meta_path.read_text())
        assert len(data) == 1
        assert data[0]["text"] == "test entry"

    def test_load_no_files(self, vector_store_with_faiss, tmp_path):
        store = vector_store_with_faiss
        # No files exist yet, load should not crash
        store.load()

    def test_load_with_files(self, vector_store_with_faiss, mock_faiss, tmp_path):
        store = vector_store_with_faiss
        # Create metadata file
        meta_path = Path(store._metadata_path)
        meta_path.parent.mkdir(parents=True, exist_ok=True)
        meta_path.write_text(json.dumps([{"text": "loaded", "metadata": {}}]))
        # Create index file
        Path(store.index_path).write_text("fake")

        store.load()
        assert len(store._entries) == 1
        assert store._entries[0]["text"] == "loaded"

    def test_search_skips_invalid_indices(self, vector_store_with_faiss):
        """Search results with negative or out-of-range indices are skipped."""
        store = vector_store_with_faiss
        store.add("entry one")

        original_search = store._index.search

        def patched_search(arr, k):
            d, i = original_search(arr, k)
            d = FakeNPArray([[0.1, 0.2]])
            i = FakeNPArray([[0, -1]])
            return d, i

        store._index.search = patched_search
        results = store.search("entry", top_k=5)
        # Only index 0 should be returned, -1 should be skipped
        assert len(results) == 1

    def test_search_skips_out_of_range_index(self, vector_store_with_faiss):
        store = vector_store_with_faiss
        store.add("single entry")

        def patched_search(arr, k):
            d = FakeNPArray([[0.1, 0.2]])
            i = FakeNPArray([[0, 999]])
            return d, i

        store._index.search = patched_search
        results = store.search("entry", top_k=5)
        assert len(results) == 1

    def test_save_creates_parent_directory(self, mock_faiss, mock_np, tmp_path):
        """Save should create parent directories if they don't exist."""
        import missy.memory.vector_store as vs_mod

        original = vs_mod._FAISS_AVAILABLE
        original_faiss = sys.modules.get("faiss")
        original_np_mod = sys.modules.get("numpy")
        vs_mod._FAISS_AVAILABLE = True
        sys.modules["faiss"] = mock_faiss
        sys.modules["numpy"] = mock_np
        original_mod_faiss = getattr(vs_mod, "faiss", None)
        vs_mod.faiss = mock_faiss

        try:
            deep_path = tmp_path / "a" / "b" / "c" / "test.faiss"
            store = vs_mod.VectorMemoryStore(
                dimension=64,
                index_path=str(deep_path),
            )
            store.add("test")
            store.save()
            assert deep_path.parent.exists()
        finally:
            vs_mod._FAISS_AVAILABLE = original
            if original_faiss is not None:
                sys.modules["faiss"] = original_faiss
            elif "faiss" in sys.modules:
                del sys.modules["faiss"]
            if original_np_mod is not None:
                sys.modules["numpy"] = original_np_mod
            elif "numpy" in sys.modules:
                del sys.modules["numpy"]
            if original_mod_faiss is not None:
                vs_mod.faiss = original_mod_faiss

    def test_metadata_file_permissions(self, vector_store_with_faiss, tmp_path):
        """Metadata file should be created with 0o600 permissions."""
        store = vector_store_with_faiss
        store.add("test entry")
        store.save()
        meta_path = Path(store._metadata_path)
        assert meta_path.exists()
        mode = meta_path.stat().st_mode & 0o777
        assert mode == 0o600
