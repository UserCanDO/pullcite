"""
Base types for the search module.

Searchers provide lightweight text search (BM25, keyword) without
requiring embeddings. They complement Retrievers for hybrid search.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Sequence


@dataclass(frozen=True)
class SearchResult:
    """
    Result from a text search.

    Attributes:
        text: The matching text/chunk content.
        score: Relevance score (higher = more relevant).
        chunk_index: Index of the chunk in the document.
        page: Page number if available.
        metadata: Additional metadata about the result.
    """

    text: str
    score: float
    chunk_index: int
    page: int | None = None
    metadata: dict[str, Any] | None = None

    def __lt__(self, other: SearchResult) -> bool:
        """Compare by score for sorting (descending)."""
        return self.score > other.score


class Searcher(ABC):
    """
    Abstract base class for text searchers.

    Searchers provide keyword-based search without requiring embeddings.
    They're faster to set up and work well for exact term matching.

    Unlike Retrievers (which use semantic/vector search), Searchers:
    - Don't require embedding generation
    - Work better for exact terms, numbers, codes
    - Are faster to index and search
    - Can be combined with Retrievers for hybrid search
    """

    @abstractmethod
    def index(self, chunks: Sequence[str], metadata: Sequence[dict] | None = None) -> None:
        """
        Index chunks for searching.

        Args:
            chunks: Text chunks to index.
            metadata: Optional metadata for each chunk (e.g., page number).
        """
        pass

    @abstractmethod
    def search(self, query: str, top_k: int = 5) -> list[SearchResult]:
        """
        Search for chunks matching the query.

        Args:
            query: Search query (keywords/terms).
            top_k: Maximum number of results to return.

        Returns:
            List of SearchResults ordered by relevance (highest first).
        """
        pass

    @abstractmethod
    def clear(self) -> None:
        """Clear the index."""
        pass

    @property
    @abstractmethod
    def is_indexed(self) -> bool:
        """Check if the searcher has indexed content."""
        pass

    @property
    @abstractmethod
    def document_count(self) -> int:
        """Number of documents/chunks in the index."""
        pass


__all__ = [
    "SearchResult",
    "Searcher",
]
