"""
Base embedder interface.

This module defines the abstract interface for text embedding providers.
Embedders convert text into dense vector representations for semantic search.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass(frozen=True)
class EmbeddingResult:
    """
    Result of embedding text.

    Attributes:
        vector: Dense embedding vector.
        model: Model name that produced this embedding.
        dimensions: Vector dimensions.
        token_count: Tokens consumed (for cost tracking).
    """

    vector: tuple[float, ...]
    model: str
    dimensions: int
    token_count: int

    def __post_init__(self) -> None:
        """Validate embedding result."""
        if len(self.vector) != self.dimensions:
            raise ValueError(
                f"Vector length {len(self.vector)} doesn't match dimensions {self.dimensions}"
            )


@dataclass(frozen=True)
class BatchEmbeddingResult:
    """
    Result of embedding multiple texts.

    Attributes:
        vectors: Tuple of embedding vectors, same order as input.
        model: Model name.
        dimensions: Vector dimensions.
        total_tokens: Total tokens consumed.
    """

    vectors: tuple[tuple[float, ...], ...]
    model: str
    dimensions: int
    total_tokens: int


class Embedder(ABC):
    """
    Abstract base class for embedding providers.

    Embedders convert text to dense vectors for semantic similarity search.
    Implementations should handle batching and rate limiting internally.

    Example:
        >>> embedder = OpenAIEmbedder(api_key="...")
        >>> result = embedder.embed("What is the deductible?")
        >>> print(len(result.vector))  # 1536 for text-embedding-3-small
    """

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Return the model identifier."""
        ...

    @property
    @abstractmethod
    def dimensions(self) -> int:
        """Return the embedding dimensions."""
        ...

    @abstractmethod
    def embed(self, text: str) -> EmbeddingResult:
        """
        Embed a single text.

        Args:
            text: Text to embed.

        Returns:
            EmbeddingResult with vector and metadata.

        Raises:
            EmbeddingError: If embedding fails.
        """
        ...

    @abstractmethod
    def embed_batch(self, texts: list[str]) -> BatchEmbeddingResult:
        """
        Embed multiple texts efficiently.

        Args:
            texts: Texts to embed.

        Returns:
            BatchEmbeddingResult with vectors in same order as input.

        Raises:
            EmbeddingError: If embedding fails.
        """
        ...

    def embed_many(
        self, texts: list[str], batch_size: int = 100
    ) -> list[tuple[float, ...]]:
        """
        Embed many texts with automatic batching.

        Convenience method that handles batching and returns just vectors.

        Args:
            texts: Texts to embed.
            batch_size: Max texts per API call.

        Returns:
            List of vectors in same order as input.
        """
        if not texts:
            return []

        all_vectors: list[tuple[float, ...]] = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            result = self.embed_batch(batch)
            all_vectors.extend(result.vectors)

        return all_vectors


class EmbeddingError(Exception):
    """Raised when embedding fails."""

    def __init__(self, message: str, cause: Exception | None = None):
        super().__init__(message)
        self.cause = cause


__all__ = [
    "Embedder",
    "EmbeddingResult",
    "BatchEmbeddingResult",
    "EmbeddingError",
]
