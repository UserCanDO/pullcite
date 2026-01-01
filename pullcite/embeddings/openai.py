"""
OpenAI embeddings implementation.

Uses OpenAI's text-embedding-3-small or text-embedding-3-large models.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

from .base import (
    Embedder,
    EmbeddingResult,
    BatchEmbeddingResult,
    EmbeddingError,
)


# Model configurations
OPENAI_MODELS = {
    "text-embedding-3-small": {"dimensions": 1536, "max_tokens": 8191},
    "text-embedding-3-large": {"dimensions": 3072, "max_tokens": 8191},
    "text-embedding-ada-002": {"dimensions": 1536, "max_tokens": 8191},
}


@dataclass
class OpenAIEmbedder(Embedder):
    """
    OpenAI embeddings provider.

    Uses the OpenAI API to generate embeddings. Requires openai package.

    Attributes:
        api_key: OpenAI API key. Defaults to OPENAI_API_KEY env var.
        model: Model name. Defaults to text-embedding-3-small.
        dimensions: Output dimensions (only for text-embedding-3-* models).

    Example:
        >>> embedder = OpenAIEmbedder()
        >>> result = embedder.embed("Hello world")
        >>> len(result.vector)
        1536
    """

    api_key: str | None = None
    model: str = "text-embedding-3-small"
    _dimensions: int | None = None

    def __post_init__(self) -> None:
        """Initialize and validate."""
        # Get API key from env if not provided
        if self.api_key is None:
            self.api_key = os.environ.get("OPENAI_API_KEY")

        if not self.api_key:
            raise EmbeddingError(
                "OpenAI API key required. Set OPENAI_API_KEY env var or pass api_key."
            )

        # Validate model
        if self.model not in OPENAI_MODELS:
            raise EmbeddingError(
                f"Unknown model: {self.model}. "
                f"Supported: {list(OPENAI_MODELS.keys())}"
            )

        # Set dimensions
        if self._dimensions is None:
            self._dimensions = OPENAI_MODELS[self.model]["dimensions"]

    @property
    def model_name(self) -> str:
        """Return the model identifier."""
        return self.model

    @property
    def dimensions(self) -> int:
        """Return the embedding dimensions."""
        return self._dimensions or OPENAI_MODELS[self.model]["dimensions"]

    def _get_client(self):
        """Get OpenAI client (lazy import)."""
        try:
            from openai import OpenAI
        except ImportError:
            raise EmbeddingError(
                "openai package required. Install with: pip install openai"
            )

        return OpenAI(api_key=self.api_key)

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

            # Build request kwargs
            kwargs = {
                "model": self.model,
                "input": text,
            }

            # text-embedding-3-* models support custom dimensions
            if self.model.startswith("text-embedding-3"):
                kwargs["dimensions"] = self.dimensions

            response = client.embeddings.create(**kwargs)

            embedding = response.data[0].embedding
            token_count = response.usage.total_tokens

            return EmbeddingResult(
                vector=tuple(embedding),
                model=self.model,
                dimensions=self.dimensions,
                token_count=token_count,
            )

        except EmbeddingError:
            raise
        except Exception as e:
            raise EmbeddingError(f"OpenAI embedding failed: {e}", cause=e)

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

        # Validate all texts
        for i, text in enumerate(texts):
            if not text or not text.strip():
                raise EmbeddingError(f"Cannot embed empty text at index {i}")

        try:
            client = self._get_client()

            kwargs = {
                "model": self.model,
                "input": texts,
            }

            if self.model.startswith("text-embedding-3"):
                kwargs["dimensions"] = self.dimensions

            response = client.embeddings.create(**kwargs)

            # OpenAI returns embeddings in same order as input
            vectors = tuple(tuple(item.embedding) for item in response.data)
            total_tokens = response.usage.total_tokens

            return BatchEmbeddingResult(
                vectors=vectors,
                model=self.model,
                dimensions=self.dimensions,
                total_tokens=total_tokens,
            )

        except EmbeddingError:
            raise
        except Exception as e:
            raise EmbeddingError(f"OpenAI batch embedding failed: {e}", cause=e)


__all__ = ["OpenAIEmbedder", "OPENAI_MODELS"]
