"""
Tests for embeddings/cache.py - Embedding cache implementations.
"""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from pullcite.embeddings.base import (
    Embedder,
    EmbeddingResult,
    BatchEmbeddingResult,
    EmbeddingError,
)
from pullcite.embeddings.cache import (
    CachedEmbedder,
    MemoryCache,
    DiskCache,
    _hash_text,
)


class TestHashText:
    """Test the text hashing function."""

    def test_consistent_hashing(self):
        hash1 = _hash_text("hello world")
        hash2 = _hash_text("hello world")
        assert hash1 == hash2

    def test_different_texts_different_hashes(self):
        hash1 = _hash_text("hello")
        hash2 = _hash_text("world")
        assert hash1 != hash2

    def test_hash_is_hex_string(self):
        result = _hash_text("test")
        assert all(c in "0123456789abcdef" for c in result)


class TestMemoryCache:
    """Test MemoryCache class."""

    def test_get_nonexistent_key(self):
        cache = MemoryCache()
        assert cache.get("nonexistent") is None

    def test_set_and_get(self):
        cache = MemoryCache()
        vector = (0.1, 0.2, 0.3)
        cache.set("key1", vector)
        assert cache.get("key1") == vector

    def test_lru_eviction(self):
        cache = MemoryCache(max_size=2)
        cache.set("key1", (0.1,))
        cache.set("key2", (0.2,))
        cache.set("key3", (0.3,))  # Should evict key1

        assert cache.get("key1") is None
        assert cache.get("key2") == (0.2,)
        assert cache.get("key3") == (0.3,)

    def test_access_updates_lru_order(self):
        cache = MemoryCache(max_size=2)
        cache.set("key1", (0.1,))
        cache.set("key2", (0.2,))

        # Access key1 to make it more recent
        cache.get("key1")

        # Adding key3 should evict key2 (least recently used)
        cache.set("key3", (0.3,))

        assert cache.get("key1") == (0.1,)
        assert cache.get("key2") is None
        assert cache.get("key3") == (0.3,)

    def test_clear(self):
        cache = MemoryCache()
        cache.set("key1", (0.1,))
        cache.set("key2", (0.2,))
        cache.clear()

        assert cache.size == 0
        assert cache.get("key1") is None
        assert cache.get("key2") is None

    def test_size_property(self):
        cache = MemoryCache()
        assert cache.size == 0

        cache.set("key1", (0.1,))
        assert cache.size == 1

        cache.set("key2", (0.2,))
        assert cache.size == 2


class TestDiskCache:
    """Test DiskCache class."""

    def test_creates_directory(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_path = Path(tmpdir) / "subdir" / "cache.db"
            cache = DiskCache(cache_path=cache_path)
            cache._get_connection()

            assert cache_path.parent.exists()
            cache.close()

    def test_set_and_get(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_path = Path(tmpdir) / "cache.db"
            cache = DiskCache(cache_path=cache_path)

            vector = (0.1, 0.2, 0.3)
            cache.set("key1", "model1", 3, vector)

            result = cache.get("key1", "model1")
            assert result == vector

            cache.close()

    def test_get_nonexistent(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_path = Path(tmpdir) / "cache.db"
            cache = DiskCache(cache_path=cache_path)

            assert cache.get("nonexistent", "model1") is None
            cache.close()

    def test_model_scoping(self):
        """Same key, different models should be separate."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_path = Path(tmpdir) / "cache.db"
            cache = DiskCache(cache_path=cache_path)

            cache.set("key1", "model1", 2, (0.1, 0.2))
            cache.set("key1", "model2", 2, (0.3, 0.4))

            assert cache.get("key1", "model1") == (0.1, 0.2)
            assert cache.get("key1", "model2") == (0.3, 0.4)

            cache.close()

    def test_clear_all(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_path = Path(tmpdir) / "cache.db"
            cache = DiskCache(cache_path=cache_path)

            cache.set("key1", "model1", 2, (0.1, 0.2))
            cache.set("key2", "model2", 2, (0.3, 0.4))
            cache.clear()

            assert cache.size == 0
            cache.close()

    def test_clear_by_model(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_path = Path(tmpdir) / "cache.db"
            cache = DiskCache(cache_path=cache_path)

            cache.set("key1", "model1", 2, (0.1, 0.2))
            cache.set("key2", "model2", 2, (0.3, 0.4))
            cache.clear(model="model1")

            assert cache.get("key1", "model1") is None
            assert cache.get("key2", "model2") == (0.3, 0.4)

            cache.close()

    def test_size_property(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_path = Path(tmpdir) / "cache.db"
            cache = DiskCache(cache_path=cache_path)

            assert cache.size == 0

            cache.set("key1", "model1", 2, (0.1, 0.2))
            assert cache.size == 1

            cache.close()


class MockEmbedder(Embedder):
    """Mock embedder for testing."""

    def __init__(self):
        self.call_count = 0

    @property
    def model_name(self) -> str:
        return "mock-model"

    @property
    def dimensions(self) -> int:
        return 3

    def embed(self, text: str) -> EmbeddingResult:
        self.call_count += 1
        return EmbeddingResult(
            vector=(0.1, 0.2, 0.3),
            model=self.model_name,
            dimensions=self.dimensions,
            token_count=len(text.split()),
        )

    def embed_batch(self, texts: list[str]) -> BatchEmbeddingResult:
        self.call_count += 1
        vectors = tuple((0.1, 0.2, 0.3) for _ in texts)
        return BatchEmbeddingResult(
            vectors=vectors,
            model=self.model_name,
            dimensions=self.dimensions,
            total_tokens=sum(len(t.split()) for t in texts),
        )


class TestCachedEmbedderInit:
    """Test CachedEmbedder initialization."""

    def test_invalid_cache_type(self):
        embedder = MockEmbedder()
        with pytest.raises(EmbeddingError) as exc:
            CachedEmbedder(embedder, cache_type="invalid")
        assert "Invalid cache_type" in str(exc.value)

    def test_valid_cache_types(self):
        embedder = MockEmbedder()
        CachedEmbedder(embedder, cache_type="memory")
        CachedEmbedder(embedder, cache_type="disk")
        CachedEmbedder(embedder, cache_type="both")

    def test_properties_from_wrapped_embedder(self):
        embedder = MockEmbedder()
        cached = CachedEmbedder(embedder, cache_type="memory")

        assert cached.model_name == "mock-model"
        assert cached.dimensions == 3


class TestCachedEmbedderEmbed:
    """Test CachedEmbedder embed method."""

    def test_cache_hit(self):
        embedder = MockEmbedder()
        cached = CachedEmbedder(embedder, cache_type="memory")

        # First call - cache miss
        result1 = cached.embed("hello world")
        assert embedder.call_count == 1

        # Second call - cache hit
        result2 = cached.embed("hello world")
        assert embedder.call_count == 1  # No additional call

        assert result1.vector == result2.vector

    def test_cache_miss(self):
        embedder = MockEmbedder()
        cached = CachedEmbedder(embedder, cache_type="memory")

        cached.embed("hello")
        cached.embed("world")

        assert embedder.call_count == 2

    def test_cache_stats(self):
        embedder = MockEmbedder()
        cached = CachedEmbedder(embedder, cache_type="memory")

        cached.embed("hello")  # miss
        cached.embed("hello")  # hit
        cached.embed("world")  # miss
        cached.embed("hello")  # hit

        stats = cached.cache_stats
        assert stats["hits"] == 2
        assert stats["misses"] == 2
        assert stats["hit_rate"] == 0.5
        assert stats["total_requests"] == 4


class TestCachedEmbedderEmbedBatch:
    """Test CachedEmbedder embed_batch method."""

    def test_batch_empty(self):
        embedder = MockEmbedder()
        cached = CachedEmbedder(embedder, cache_type="memory")

        result = cached.embed_batch([])
        assert result.vectors == ()
        assert embedder.call_count == 0

    def test_batch_all_cached(self):
        embedder = MockEmbedder()
        cached = CachedEmbedder(embedder, cache_type="memory")

        # Cache the texts individually
        cached.embed("hello")
        cached.embed("world")
        assert embedder.call_count == 2

        # Batch request should be fully cached
        result = cached.embed_batch(["hello", "world"])
        assert embedder.call_count == 2  # No additional calls
        assert len(result.vectors) == 2

    def test_batch_partial_cache(self):
        embedder = MockEmbedder()
        cached = CachedEmbedder(embedder, cache_type="memory")

        # Cache only one text
        cached.embed("hello")
        assert embedder.call_count == 1

        # Batch with one cached, one new
        result = cached.embed_batch(["hello", "world"])
        assert embedder.call_count == 2  # One additional call for batch
        assert len(result.vectors) == 2


class TestCachedEmbedderDiskCache:
    """Test CachedEmbedder with disk cache."""

    def test_disk_cache_persistence(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_path = Path(tmpdir) / "cache.db"

            # Create first cached embedder
            embedder1 = MockEmbedder()
            cached1 = CachedEmbedder(
                embedder1, cache_type="disk", cache_path=str(cache_path)
            )
            cached1.embed("hello")
            cached1.close()

            # Create second cached embedder with same cache
            embedder2 = MockEmbedder()
            cached2 = CachedEmbedder(
                embedder2, cache_type="disk", cache_path=str(cache_path)
            )

            # Should hit cache
            cached2.embed("hello")
            assert embedder2.call_count == 0

            cached2.close()


class TestCachedEmbedderClearCache:
    """Test cache clearing."""

    def test_clear_cache(self):
        embedder = MockEmbedder()
        cached = CachedEmbedder(embedder, cache_type="memory")

        cached.embed("hello")
        cached.embed("hello")  # hit
        assert cached.cache_stats["hits"] == 1

        cached.clear_cache()

        assert cached.cache_stats["hits"] == 0
        assert cached.cache_stats["misses"] == 0

        # Should miss now
        cached.embed("hello")
        assert cached.cache_stats["misses"] == 1
