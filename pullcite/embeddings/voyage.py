"""
Voyage AI embeddings implementation.

Uses Voyage AI's embedding models for high-quality text embeddings.
Voyage models are optimized for retrieval and semantic search tasks.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

from .base import (
    Embedder,
    EmbeddingResult,
    BatchEmbeddingResult,
    EmbeddingError,
)


# Model configurations
VOYAGE_MODELS = {
    "voyage-3": {"dimensions": 1024, "max_tokens": 32000},
    "voyage-3-lite": {"dimensions": 512, "max_tokens": 32000},
    "voyage-code-3": {"dimensions": 1024, "max_tokens": 32000},
    "voyage-finance-2": {"dimensions": 1024, "max_tokens": 32000},
    "voyage-law-2": {"dimensions": 1024, "max_tokens": 16000},
    "voyage-large-2": {"dimensions": 1536, "max_tokens": 16000},
    "voyage-2": {"dimensions": 1024, "max_tokens": 4000},
}


@dataclass
class VoyageEmbedder(Embedder):
    """
    Voyage AI embeddings provider.

    Uses the Voyage AI API for high-quality embeddings optimized for retrieval.
    Requires voyageai package.

    Attributes:
        api_key: Voyage API key. Defaults to VOYAGE_API_KEY env var.
        model: Model name. Defaults to voyage-3.
        input_type: Type of input ('query', 'document', or None for auto).

    Example:
        >>> embedder = VoyageEmbedder()
        >>> result = embedder.embed("What is machine learning?")
        >>> len(result.vector)
        1024
    """

    api_key: str | None = None
    model: str = "voyage-3"
    input_type: str | None = None

    def __post_init__(self) -> None:
        """Initialize and validate."""
        if self.api_key is None:
            self.api_key = os.environ.get("VOYAGE_API_KEY")

        if not self.api_key:
            raise EmbeddingError(
                "Voyage API key required. Set VOYAGE_API_KEY env var or pass api_key."
            )

        if self.model not in VOYAGE_MODELS:
            raise EmbeddingError(
                f"Unknown model: {self.model}. "
                f"Supported: {list(VOYAGE_MODELS.keys())}"
            )

        if self.input_type and self.input_type not in ("query", "document"):
            raise EmbeddingError(
                f"Invalid input_type: {self.input_type}. "
                "Must be 'query', 'document', or None."
            )

    @property
    def model_name(self) -> str:
        """Return the model identifier."""
        return self.model

    @property
    def dimensions(self) -> int:
        """Return the embedding dimensions."""
        return VOYAGE_MODELS[self.model]["dimensions"]

    def _get_client(self):
        """Get Voyage client (lazy import)."""
        try:
            import voyageai
        except ImportError:
            raise EmbeddingError(
                "voyageai package required. Install with: pip install voyageai"
            )

        return voyageai.Client(api_key=self.api_key)

    def embed(self, text: str) -> EmbeddingResult:
        """
        Embed a single text.

        Args:
            text: Text to embed.

        Returns:
            EmbeddingResult with vector and metadata.

        Raises:
            EmbeddingError: If API call fails.
        """
        if not text or not text.strip():
            raise EmbeddingError("Cannot embed empty text")

        try:
            client = self._get_client()

            kwargs: dict[str, Any] = {
                "texts": [text],
                "model": self.model,
            }

            if self.input_type:
                kwargs["input_type"] = self.input_type

            result = client.embed(**kwargs)

            embedding = result.embeddings[0]
            token_count = result.total_tokens

            return EmbeddingResult(
                vector=tuple(embedding),
                model=self.model,
                dimensions=self.dimensions,
                token_count=token_count,
            )

        except EmbeddingError:
            raise
        except Exception as e:
            raise EmbeddingError(f"Voyage embedding failed: {e}", cause=e)

    def embed_batch(self, texts: list[str]) -> BatchEmbeddingResult:
        """
        Embed multiple texts efficiently.

        Args:
            texts: Texts to embed.

        Returns:
            BatchEmbeddingResult with vectors in same order as input.

        Raises:
            EmbeddingError: If API call fails.
        """
        if not texts:
            return BatchEmbeddingResult(
                vectors=(),
                model=self.model,
                dimensions=self.dimensions,
                total_tokens=0,
            )

        for i, text in enumerate(texts):
            if not text or not text.strip():
                raise EmbeddingError(f"Cannot embed empty text at index {i}")

        try:
            client = self._get_client()

            kwargs: dict[str, Any] = {
                "texts": texts,
                "model": self.model,
            }

            if self.input_type:
                kwargs["input_type"] = self.input_type

            result = client.embed(**kwargs)

            vectors = tuple(tuple(emb) for emb in result.embeddings)
            total_tokens = result.total_tokens

            return BatchEmbeddingResult(
                vectors=vectors,
                model=self.model,
                dimensions=self.dimensions,
                total_tokens=total_tokens,
            )

        except EmbeddingError:
            raise
        except Exception as e:
            raise EmbeddingError(f"Voyage batch embedding failed: {e}", cause=e)

    def embed_query(self, text: str) -> EmbeddingResult:
        """
        Embed a query text (optimized for retrieval queries).

        This is a convenience method that sets input_type='query'.

        Args:
            text: Query text to embed.

        Returns:
            EmbeddingResult optimized for query embedding.
        """
        original_type = self.input_type
        try:
            object.__setattr__(self, "input_type", "query")
            return self.embed(text)
        finally:
            object.__setattr__(self, "input_type", original_type)

    def embed_documents(self, texts: list[str]) -> BatchEmbeddingResult:
        """
        Embed document texts (optimized for document storage).

        This is a convenience method that sets input_type='document'.

        Args:
            texts: Document texts to embed.

        Returns:
            BatchEmbeddingResult optimized for document embeddings.
        """
        original_type = self.input_type
        try:
            object.__setattr__(self, "input_type", "document")
            return self.embed_batch(texts)
        finally:
            object.__setattr__(self, "input_type", original_type)


__all__ = ["VoyageEmbedder", "VOYAGE_MODELS"]
