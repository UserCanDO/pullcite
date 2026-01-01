"""
Embeddings providers for semantic text representation.

This module provides abstractions and implementations for various embedding providers.
"""

from .base import (
    Embedder,
    EmbeddingResult,
    BatchEmbeddingResult,
    EmbeddingError,
)
from .openai import OpenAIEmbedder, OPENAI_MODELS
from .voyage import VoyageEmbedder, VOYAGE_MODELS
from .local import LocalEmbedder, LOCAL_MODELS
from .cache import CachedEmbedder, MemoryCache, DiskCache

__all__ = [
    # Base classes and types
    "Embedder",
    "EmbeddingResult",
    "BatchEmbeddingResult",
    "EmbeddingError",
    # Providers
    "OpenAIEmbedder",
    "OPENAI_MODELS",
    "VoyageEmbedder",
    "VOYAGE_MODELS",
    "LocalEmbedder",
    "LOCAL_MODELS",
    # Caching
    "CachedEmbedder",
    "MemoryCache",
    "DiskCache",
]
