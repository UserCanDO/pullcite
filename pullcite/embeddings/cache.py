"""
Embedding cache implementation.

Provides caching wrappers for embedders to avoid redundant API calls.
"""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .base import (
    Embedder,
    EmbeddingResult,
    BatchEmbeddingResult,
    EmbeddingError,
)


def _hash_text(text: str) -> str:
    """Generate a hash key for text."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


@dataclass
class MemoryCache:
    """
    In-memory LRU cache for embeddings.

    Thread-safe cache with configurable size limit.

    Attributes:
        max_size: Maximum number of entries to cache.
    """

    max_size: int = 10000
    _cache: dict[str, tuple[float, ...]] = field(default_factory=dict, repr=False)
    _order: list[str] = field(default_factory=list, repr=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def get(self, key: str) -> tuple[float, ...] | None:
        """Get cached vector."""
        with self._lock:
            if key in self._cache:
                # Move to end (most recently used)
                self._order.remove(key)
                self._order.append(key)
                return self._cache[key]
            return None

    def set(self, key: str, vector: tuple[float, ...]) -> None:
        """Cache a vector."""
        with self._lock:
            if key in self._cache:
                self._order.remove(key)
            elif len(self._cache) >= self.max_size:
                # Remove oldest entry
                oldest = self._order.pop(0)
                del self._cache[oldest]

            self._cache[key] = vector
            self._order.append(key)

    def clear(self) -> None:
        """Clear the cache."""
        with self._lock:
            self._cache.clear()
            self._order.clear()

    @property
    def size(self) -> int:
        """Return number of cached entries."""
        return len(self._cache)


@dataclass
class DiskCache:
    """
    SQLite-based persistent cache for embeddings.

    Stores embeddings on disk for persistence across sessions.

    Attributes:
        cache_path: Path to the SQLite database file.
    """

    cache_path: str | Path = ".pullcite_cache/embeddings.db"
    _connection: sqlite3.Connection | None = field(default=None, repr=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def __post_init__(self) -> None:
        """Initialize the cache directory and database."""
        self.cache_path = Path(self.cache_path)
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)

    def _get_connection(self) -> sqlite3.Connection:
        """Get or create database connection."""
        if self._connection is None:
            self._connection = sqlite3.connect(
                str(self.cache_path),
                check_same_thread=False,  # We use our own lock
            )
            self._connection.execute(
                """
                CREATE TABLE IF NOT EXISTS embeddings (
                    key TEXT NOT NULL,
                    model TEXT NOT NULL,
                    dimensions INTEGER NOT NULL,
                    vector TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY (key, model)
                )
                """
            )
            self._connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_embeddings_key_model ON embeddings(key, model)"
            )
            self._connection.commit()
        return self._connection

    def get(self, key: str, model: str) -> tuple[float, ...] | None:
        """Get cached vector."""
        with self._lock:
            conn = self._get_connection()
            cursor = conn.execute(
                "SELECT vector FROM embeddings WHERE key = ? AND model = ?",
                (key, model),
            )
            row = cursor.fetchone()
            if row:
                return tuple(json.loads(row[0]))
            return None

    def set(
        self, key: str, model: str, dimensions: int, vector: tuple[float, ...]
    ) -> None:
        """Cache a vector."""
        with self._lock:
            conn = self._get_connection()
            conn.execute(
                """
                INSERT OR REPLACE INTO embeddings (key, model, dimensions, vector)
                VALUES (?, ?, ?, ?)
                """,
                (key, model, dimensions, json.dumps(list(vector))),
            )
            conn.commit()

    def clear(self, model: str | None = None) -> None:
        """Clear the cache (optionally for a specific model)."""
        with self._lock:
            conn = self._get_connection()
            if model:
                conn.execute("DELETE FROM embeddings WHERE model = ?", (model,))
            else:
                conn.execute("DELETE FROM embeddings")
            conn.commit()

    @property
    def size(self) -> int:
        """Return number of cached entries."""
        with self._lock:
            conn = self._get_connection()
            cursor = conn.execute("SELECT COUNT(*) FROM embeddings")
            return cursor.fetchone()[0]

    def close(self) -> None:
        """Close the database connection."""
        if self._connection:
            self._connection.close()
            self._connection = None


@dataclass
class CachedEmbedder(Embedder):
    """
    Caching wrapper for any embedder.

    Wraps an existing embedder and caches results to avoid redundant API calls.
    Supports both in-memory and disk-based caching.

    Attributes:
        _embedder: The underlying embedder to wrap.
        cache_type: Type of cache ('memory', 'disk', or 'both').
        cache_path: Path for disk cache (if using disk caching).
        memory_size: Max entries for memory cache.

    Example:
        >>> from pullcite.embeddings.openai import OpenAIEmbedder
        >>> base_embedder = OpenAIEmbedder()
        >>> cached = CachedEmbedder(base_embedder, cache_type="both")
        >>> result = cached.embed("Hello world")  # API call
        >>> result = cached.embed("Hello world")  # Cached (no API call)
    """

    _embedder: Embedder
    cache_type: str = "memory"
    cache_path: str | Path = ".pullcite_cache/embeddings.db"
    memory_size: int = 10000
    _memory_cache: MemoryCache | None = field(default=None, repr=False)
    _disk_cache: DiskCache | None = field(default=None, repr=False)
    _hits: int = field(default=0, repr=False)
    _misses: int = field(default=0, repr=False)

    def __post_init__(self) -> None:
        """Initialize caches based on configuration."""
        if self.cache_type not in ("memory", "disk", "both"):
            raise EmbeddingError(
                f"Invalid cache_type: {self.cache_type}. "
                "Must be 'memory', 'disk', or 'both'."
            )

        if self.cache_type in ("memory", "both"):
            self._memory_cache = MemoryCache(max_size=self.memory_size)

        if self.cache_type in ("disk", "both"):
            self._disk_cache = DiskCache(cache_path=self.cache_path)

    @property
    def model_name(self) -> str:
        """Return the model identifier."""
        return self._embedder.model_name

    @property
    def dimensions(self) -> int:
        """Return the embedding dimensions."""
        return self._embedder.dimensions

    @property
    def cache_stats(self) -> dict[str, Any]:
        """Return cache statistics."""
        total = self._hits + self._misses
        hit_rate = self._hits / total if total > 0 else 0.0
        return {
            "hits": self._hits,
            "misses": self._misses,
            "total_requests": total,
            "hit_rate": hit_rate,
            "memory_size": self._memory_cache.size if self._memory_cache else 0,
            "disk_size": self._disk_cache.size if self._disk_cache else 0,
        }

    def _get_cached(self, text: str) -> tuple[float, ...] | None:
        """Try to get a cached embedding."""
        key = _hash_text(text)

        # Try memory cache first
        if self._memory_cache:
            vector = self._memory_cache.get(key)
            if vector is not None:
                return vector

        # Try disk cache
        if self._disk_cache:
            vector = self._disk_cache.get(key, self.model_name)
            if vector is not None:
                # Populate memory cache for faster future access
                if self._memory_cache:
                    self._memory_cache.set(key, vector)
                return vector

        return None

    def _set_cached(self, text: str, vector: tuple[float, ...]) -> None:
        """Cache an embedding."""
        key = _hash_text(text)

        if self._memory_cache:
            self._memory_cache.set(key, vector)

        if self._disk_cache:
            self._disk_cache.set(key, self.model_name, self.dimensions, vector)

    def embed(self, text: str) -> EmbeddingResult:
        """
        Embed a single text with caching.

        Args:
            text: Text to embed.

        Returns:
            EmbeddingResult with vector and metadata.

        Raises:
            EmbeddingError: If embedding fails.
        """
        # Check cache
        cached_vector = self._get_cached(text)
        if cached_vector is not None:
            self._hits += 1
            return EmbeddingResult(
                vector=cached_vector,
                model=self.model_name,
                dimensions=self.dimensions,
                token_count=0,  # Cached, so no tokens used
            )

        # Cache miss - call underlying embedder
        self._misses += 1
        result = self._embedder.embed(text)

        # Cache the result
        self._set_cached(text, result.vector)

        return result

    def embed_batch(self, texts: list[str]) -> BatchEmbeddingResult:
        """
        Embed multiple texts with caching.

        Args:
            texts: Texts to embed.

        Returns:
            BatchEmbeddingResult with vectors in same order as input.

        Raises:
            EmbeddingError: If embedding fails.
        """
        if not texts:
            return BatchEmbeddingResult(
                vectors=(),
                model=self.model_name,
                dimensions=self.dimensions,
                total_tokens=0,
            )

        vectors: list[tuple[float, ...] | None] = []
        uncached_texts: list[str] = []
        uncached_indices: list[int] = []

        # Check cache for each text
        for i, text in enumerate(texts):
            cached_vector = self._get_cached(text)
            if cached_vector is not None:
                self._hits += 1
                vectors.append(cached_vector)
            else:
                self._misses += 1
                vectors.append(None)
                uncached_texts.append(text)
                uncached_indices.append(i)

        # Embed uncached texts
        total_tokens = 0
        if uncached_texts:
            batch_result = self._embedder.embed_batch(uncached_texts)
            total_tokens = batch_result.total_tokens

            # Cache and fill in results
            for idx, (text, vector) in enumerate(
                zip(uncached_texts, batch_result.vectors)
            ):
                original_idx = uncached_indices[idx]
                vectors[original_idx] = vector
                self._set_cached(text, vector)

        return BatchEmbeddingResult(
            vectors=tuple(v for v in vectors if v is not None),
            model=self.model_name,
            dimensions=self.dimensions,
            total_tokens=total_tokens,
        )

    def clear_cache(self) -> None:
        """Clear all caches."""
        if self._memory_cache:
            self._memory_cache.clear()
        if self._disk_cache:
            self._disk_cache.clear(model=self.model_name)
        self._hits = 0
        self._misses = 0

    def close(self) -> None:
        """Close the disk cache connection."""
        if self._disk_cache:
            self._disk_cache.close()


__all__ = [
    "CachedEmbedder",
    "MemoryCache",
    "DiskCache",
]
