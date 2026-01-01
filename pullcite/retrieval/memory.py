"""
In-memory retriever using NumPy.

Simple retriever that stores embeddings in memory and uses
cosine similarity for search. Good for small to medium documents.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from .base import Retriever, SearchResult, SearchResults, RetrieverError

if TYPE_CHECKING:
    from ..core.chunk import Chunk
    from ..core.document import Document
    from ..embeddings.base import Embedder


@dataclass
class MemoryRetriever(Retriever):
    """
    In-memory retriever using NumPy for similarity search.

    Stores chunk embeddings in a NumPy array and uses cosine similarity.
    Fast for documents up to ~10k chunks. No persistence - index is lost
    when the object is garbage collected.

    Attributes:
        _embedder: Embedder for converting text to vectors.
        _chunks: Indexed chunks.
        _embeddings: Embedding matrix (n_chunks x dimensions).
        _document_id: ID of indexed document.

    Example:
        >>> from pullcite.embeddings.openai import OpenAIEmbedder
        >>> embedder = OpenAIEmbedder()
        >>> retriever = MemoryRetriever(embedder)
        >>> retriever.index(document)
        >>> results = retriever.search("deductible amount", k=3)
    """

    _embedder: "Embedder"
    _chunks: list["Chunk"] = field(default_factory=list)
    _embeddings: NDArray[np.float32] | None = field(default=None)
    _document_id: str | None = field(default=None)

    @property
    def embedder(self) -> "Embedder":
        """Return the embedder."""
        return self._embedder

    @property
    def is_indexed(self) -> bool:
        """Check if a document has been indexed."""
        return self._embeddings is not None and len(self._chunks) > 0

    @property
    def chunk_count(self) -> int:
        """Return number of indexed chunks."""
        return len(self._chunks)

    @property
    def document_id(self) -> str | None:
        """Return ID of indexed document."""
        return self._document_id

    def index(self, document: "Document") -> None:
        """
        Index a document for retrieval.

        Embeds all chunks and stores in memory. Replaces any existing index.

        Args:
            document: Document to index.

        Raises:
            RetrieverError: If embedding fails.
        """
        chunks = document.chunks

        if not chunks:
            # Empty document - clear index
            self.clear()
            self._document_id = document.id
            return

        try:
            # Get texts and embed
            texts = [chunk.text for chunk in chunks]
            vectors = self._embedder.embed_many(texts)

            # Store
            self._chunks = list(chunks)
            self._embeddings = np.array(vectors, dtype=np.float32)
            self._document_id = document.id

            # Normalize for cosine similarity
            norms = np.linalg.norm(self._embeddings, axis=1, keepdims=True)
            # Avoid division by zero
            norms = np.where(norms == 0, 1, norms)
            self._embeddings = self._embeddings / norms

        except Exception as e:
            raise RetrieverError(f"Failed to index document: {e}", cause=e)

    def search(self, query: str, k: int = 5) -> SearchResults:
        """
        Search for relevant chunks using cosine similarity.

        Args:
            query: Search query.
            k: Number of results to return.

        Returns:
            SearchResults with top k matches.

        Raises:
            RetrieverError: If no document indexed or search fails.
        """
        if not self.is_indexed:
            raise RetrieverError("No document indexed. Call index() first.")

        if k < 1:
            raise RetrieverError("k must be >= 1")

        try:
            # Embed query
            query_result = self._embedder.embed(query)
            query_vec = np.array(query_result.vector, dtype=np.float32)

            # Normalize
            query_norm = np.linalg.norm(query_vec)
            if query_norm > 0:
                query_vec = query_vec / query_norm

            # Cosine similarity (dot product of normalized vectors)
            similarities = self._embeddings @ query_vec

            # Get top k indices
            k = min(k, len(self._chunks))
            top_indices = np.argsort(similarities)[-k:][::-1]

            # Build results
            results = tuple(
                SearchResult(
                    chunk=self._chunks[idx],
                    score=float(similarities[idx]),
                    rank=rank,
                )
                for rank, idx in enumerate(top_indices)
            )

            return SearchResults(
                results=results,
                query=query,
                total_chunks=len(self._chunks),
            )

        except RetrieverError:
            raise
        except Exception as e:
            raise RetrieverError(f"Search failed: {e}", cause=e)

    def search_many(self, queries: list[str], k: int = 5) -> list[SearchResults]:
        """
        Search with multiple queries efficiently.

        Embeds all queries in one batch, then searches.

        Args:
            queries: Search queries.
            k: Number of results per query.

        Returns:
            List of SearchResults, one per query.
        """
        if not queries:
            return []

        if not self.is_indexed:
            raise RetrieverError("No document indexed. Call index() first.")

        try:
            # Batch embed queries
            query_vectors = self._embedder.embed_many(queries)
            query_matrix = np.array(query_vectors, dtype=np.float32)

            # Normalize
            norms = np.linalg.norm(query_matrix, axis=1, keepdims=True)
            norms = np.where(norms == 0, 1, norms)
            query_matrix = query_matrix / norms

            # Batch similarity (queries x chunks)
            similarities = query_matrix @ self._embeddings.T

            # Build results for each query
            all_results: list[SearchResults] = []
            k_actual = min(k, len(self._chunks))

            for i, query in enumerate(queries):
                top_indices = np.argsort(similarities[i])[-k_actual:][::-1]

                results = tuple(
                    SearchResult(
                        chunk=self._chunks[idx],
                        score=float(similarities[i, idx]),
                        rank=rank,
                    )
                    for rank, idx in enumerate(top_indices)
                )

                all_results.append(
                    SearchResults(
                        results=results,
                        query=query,
                        total_chunks=len(self._chunks),
                    )
                )

            return all_results

        except RetrieverError:
            raise
        except Exception as e:
            raise RetrieverError(f"Batch search failed: {e}", cause=e)

    def clear(self) -> None:
        """Clear the index."""
        self._chunks = []
        self._embeddings = None
        self._document_id = None


__all__ = ["MemoryRetriever"]
