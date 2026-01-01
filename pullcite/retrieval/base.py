"""
Base retriever interface.

This module defines the abstract interface for document retrieval.
Retrievers find relevant chunks from a document given a query.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..core.chunk import Chunk
    from ..core.document import Document
    from ..embeddings.base import Embedder


@dataclass(frozen=True)
class SearchResult:
    """
    A single search result.

    Attributes:
        chunk: The matching chunk.
        score: Similarity score (higher = more relevant).
        rank: Position in results (0 = best match).
    """

    chunk: (
        "Chunk"  # Quotes for forward reference - Chunk imported only for TYPE_CHECKING
    )
    score: float
    rank: int

    @property
    def text(self) -> str:
        """Convenience accessor for chunk text."""
        return self.chunk.text

    @property
    def page(self) -> int | None:
        """Convenience accessor for chunk page."""
        return self.chunk.page

    @property
    def index(self) -> int:
        """Convenience accessor for chunk index."""
        return self.chunk.index


@dataclass(frozen=True)
class SearchResults:
    """
    Collection of search results.

    Attributes:
        results: Tuple of SearchResult, ordered by relevance.
        query: The search query used.
        total_chunks: Total chunks in the index.
    """

    results: tuple[SearchResult, ...]
    query: str
    total_chunks: int

    def __len__(self) -> int:
        return len(self.results)

    def __iter__(self):
        return iter(self.results)

    def __getitem__(self, idx: int) -> SearchResult:
        return self.results[idx]

    @property
    def top(self) -> SearchResult | None:
        """Get top result, or None if empty."""
        return self.results[0] if self.results else None

    @property
    def chunks(self) -> list["Chunk"]:
        """Get all chunks from results."""
        return [r.chunk for r in self.results]

    @property
    def texts(self) -> list[str]:
        """Get all texts from results."""
        return [r.chunk.text for r in self.results]

    def above_threshold(self, threshold: float) -> "SearchResults":
        """
        Filter to results above a score threshold.

        Args:
            threshold: Minimum score.

        Returns:
            New SearchResults with filtered results.
        """
        filtered = tuple(r for r in self.results if r.score >= threshold)
        # Re-rank after filtering
        reranked = tuple(
            SearchResult(chunk=r.chunk, score=r.score, rank=i)
            for i, r in enumerate(filtered)
        )
        return SearchResults(
            results=reranked,
            query=self.query,
            total_chunks=self.total_chunks,
        )


class Retriever(ABC):
    """
    Abstract base class for document retrievers.

    Retrievers index document chunks and find relevant ones given a query.
    Typical flow:

    1. Create retriever with embedder
    2. Call index(document) to build index
    3. Call search(query) to find relevant chunks

    Example:
        >>> retriever = MemoryRetriever(embedder)
        >>> retriever.index(document)
        >>> results = retriever.search("individual deductible", k=5)
        >>> for r in results:
        ...     print(f"{r.score:.2f}: {r.text[:50]}...")
    """

    @property
    @abstractmethod
    def embedder(self) -> "Embedder":
        """Return the embedder used by this retriever."""
        ...

    @property
    @abstractmethod
    def is_indexed(self) -> bool:
        """Check if a document has been indexed."""
        ...

    @property
    @abstractmethod
    def chunk_count(self) -> int:
        """Return number of indexed chunks."""
        ...

    @abstractmethod
    def index(self, document: "Document") -> None:
        """
        Index a document for retrieval.

        Replaces any existing index.

        Args:
            document: Document to index.

        Raises:
            RetrieverError: If indexing fails.
        """
        ...

    @abstractmethod
    def search(self, query: str, k: int = 5) -> SearchResults:
        """
        Search for relevant chunks.

        Args:
            query: Search query.
            k: Number of results to return.

        Returns:
            SearchResults with top k matches.

        Raises:
            RetrieverError: If search fails or no document indexed.
        """
        ...

    def search_many(self, queries: list[str], k: int = 5) -> list[SearchResults]:
        """
        Search with multiple queries.

        Default implementation calls search() for each query.
        Subclasses may override for efficiency.

        Args:
            queries: Search queries.
            k: Number of results per query.

        Returns:
            List of SearchResults, one per query.
        """
        return [self.search(q, k=k) for q in queries]

    @abstractmethod
    def clear(self) -> None:
        """Clear the index."""
        ...


class RetrieverError(Exception):
    """Raised when retrieval fails."""

    def __init__(self, message: str, cause: Exception | None = None):
        super().__init__(message)
        self.cause = cause


__all__ = [
    "Retriever",
    "SearchResult",
    "SearchResults",
    "RetrieverError",
]
