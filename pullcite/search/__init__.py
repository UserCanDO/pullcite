"""
Search module for text-based retrieval.

This module provides keyword-based search capabilities that complement
the semantic retrieval in pullcite.retrieval. Searchers don't require
embeddings and work well for exact term matching.

Key components:
- BM25Searcher: High-performance BM25 search (uses tantivy if available)
- HybridSearcher: Combines BM25 + semantic with rank fusion
- SearchResult: Result from search operations

Example:
    >>> from pullcite.search import BM25Searcher
    >>>
    >>> # Create searcher
    >>> searcher = BM25Searcher()
    >>>
    >>> # Index document chunks
    >>> chunks = [
    ...     "The annual deductible is $500.",
    ...     "Copay for primary care visits is $20.",
    ...     "Emergency room copay is $150.",
    ... ]
    >>> searcher.index(chunks)
    >>>
    >>> # Search for relevant chunks
    >>> results = searcher.search("deductible amount", top_k=3)
    >>> for r in results:
    ...     print(f"Score {r.score:.2f}: {r.text[:50]}...")
"""

from .base import (
    SearchResult,
    Searcher,
)

from .bm25 import (
    BM25Searcher,
    BM25SearcherError,
)

from .hybrid import (
    HybridSearcher,
)

__all__ = [
    # Base types
    "SearchResult",
    "Searcher",
    # Implementations
    "BM25Searcher",
    "BM25SearcherError",
    "HybridSearcher",
]
