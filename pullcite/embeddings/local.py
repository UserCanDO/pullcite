"""
Local embeddings using Sentence Transformers.

Runs embedding models locally without API calls.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .base import (
    Embedder,
    EmbeddingResult,
    BatchEmbeddingResult,
    EmbeddingError,
)


# Common model configurations
LOCAL_MODELS = {
    "all-MiniLM-L6-v2": {"dimensions": 384},
    "all-mpnet-base-v2": {"dimensions": 768},
    "multi-qa-MiniLM-L6-cos-v1": {"dimensions": 384},
    "multi-qa-mpnet-base-dot-v1": {"dimensions": 768},
    "paraphrase-MiniLM-L6-v2": {"dimensions": 384},
    "paraphrase-mpnet-base-v2": {"dimensions": 768},
    "all-distilroberta-v1": {"dimensions": 768},
    "sentence-t5-base": {"dimensions": 768},
    "sentence-t5-large": {"dimensions": 768},
    "gtr-t5-base": {"dimensions": 768},
    "gtr-t5-large": {"dimensions": 768},
}


@dataclass
class LocalEmbedder(Embedder):
    """
    Local embeddings using Sentence Transformers.

    Runs embedding models locally on CPU or GPU. No API key required.
    Requires sentence-transformers package.

    Attributes:
        model: Model name or path. Defaults to all-MiniLM-L6-v2.
        device: Device to run on ('cpu', 'cuda', 'mps', or None for auto).
        normalize: Whether to normalize embeddings (default True).

    Example:
        >>> embedder = LocalEmbedder()
        >>> result = embedder.embed("Hello world")
        >>> len(result.vector)
        384
    """

    model: str = "all-MiniLM-L6-v2"
    device: str | None = None
    normalize: bool = True
    _model_instance: Any = field(default=None, repr=False, compare=False)
    _dimensions: int | None = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        """Initialize model lazily on first use."""
        pass

    def _load_model(self) -> Any:
        """Load the sentence transformer model."""
        if self._model_instance is not None:
            return self._model_instance

        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise EmbeddingError(
                "sentence-transformers package required. "
                "Install with: pip install sentence-transformers"
            )

        try:
            model = SentenceTransformer(self.model, device=self.device)
            object.__setattr__(self, "_model_instance", model)
            object.__setattr__(
                self, "_dimensions", model.get_sentence_embedding_dimension()
            )
            return model
        except Exception as e:
            raise EmbeddingError(f"Failed to load model '{self.model}': {e}", cause=e)

    @property
    def model_name(self) -> str:
        """Return the model identifier."""
        return self.model

    @property
    def dimensions(self) -> int:
        """Return the embedding dimensions."""
        if self._dimensions is not None:
            return self._dimensions

        # Check known models first
        if self.model in LOCAL_MODELS:
            return LOCAL_MODELS[self.model]["dimensions"]

        # Otherwise need to load model to get dimensions
        self._load_model()
        return self._dimensions or 0

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
        if not text or not text.strip():
            raise EmbeddingError("Cannot embed empty text")

        try:
            model = self._load_model()

            embedding = model.encode(
                text,
                normalize_embeddings=self.normalize,
                convert_to_numpy=True,
            )

            # Estimate token count (rough approximation)
            token_count = len(text.split())

            return EmbeddingResult(
                vector=tuple(float(x) for x in embedding),
                model=self.model,
                dimensions=self.dimensions,
                token_count=token_count,
            )

        except EmbeddingError:
            raise
        except Exception as e:
            raise EmbeddingError(f"Local embedding failed: {e}", cause=e)

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
            model = self._load_model()

            embeddings = model.encode(
                texts,
                normalize_embeddings=self.normalize,
                convert_to_numpy=True,
                show_progress_bar=False,
            )

            vectors = tuple(tuple(float(x) for x in emb) for emb in embeddings)
            total_tokens = sum(len(text.split()) for text in texts)

            return BatchEmbeddingResult(
                vectors=vectors,
                model=self.model,
                dimensions=self.dimensions,
                total_tokens=total_tokens,
            )

        except EmbeddingError:
            raise
        except Exception as e:
            raise EmbeddingError(f"Local batch embedding failed: {e}", cause=e)


__all__ = ["LocalEmbedder", "LOCAL_MODELS"]
