"""
Retrieval providers for document chunk search.

This module provides abstractions and implementations for various retrieval backends.
"""

from .base import (
    Retriever,
    SearchResult,
    SearchResults,
    RetrieverError,
)
from .memory import MemoryRetriever
from .chroma import ChromaRetriever
from .pgvector import PgVectorRetriever

__all__ = [
    # Base classes and types
    "Retriever",
    "SearchResult",
    "SearchResults",
    "RetrieverError",
    # Providers
    "MemoryRetriever",
    "ChromaRetriever",
    "PgVectorRetriever",
]
