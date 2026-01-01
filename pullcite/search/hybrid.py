"""
Hybrid search combining BM25 and semantic search.

Uses Reciprocal Rank Fusion (RRF) to combine results from multiple search methods.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

from .base import SearchResult, Searcher


@dataclass
class HybridSearcher(Searcher):
    """
    Hybrid searcher combining BM25 and semantic search.

    Uses Reciprocal Rank Fusion (RRF) to merge results from different
    search methods, giving better results than either alone.

    Attributes:
        bm25_searcher: BM25/keyword searcher.
        retriever: Semantic retriever (from pullcite.retrieval).
        bm25_weight: Weight for BM25 results (0.0 to 1.0). Default 0.5.
        rrf_k: RRF constant (higher = smoother fusion). Default 60.

    Example:
        >>> from pullcite.search import BM25Searcher, HybridSearcher
        >>> from pullcite.retrieval import MemoryRetriever
        >>> from pullcite.embeddings import OpenAIEmbedder
        >>>
        >>> bm25 = BM25Searcher()
        >>> retriever = MemoryRetriever(_embedder=OpenAIEmbedder())
        >>> hybrid = HybridSearcher(bm25_searcher=bm25, retriever=retriever)
        >>>
        >>> chunks = ["The deductible is $500.", "Copay is $20."]
        >>> hybrid.index(chunks)
        >>> results = hybrid.search("deductible")
    """

    bm25_searcher: Searcher
    retriever: Any  # Retriever from pullcite.retrieval
    bm25_weight: float = 0.5
    rrf_k: int = 60

    # Internal state
    _chunks: list[str] = field(default_factory=list, init=False, repr=False)
    _metadata: list[dict] = field(default_factory=list, init=False, repr=False)

    def __post_init__(self) -> None:
        """Validate configuration."""
        if not 0.0 <= self.bm25_weight <= 1.0:
            raise ValueError("bm25_weight must be between 0.0 and 1.0")

    def index(
        self,
        chunks: Sequence[str],
        metadata: Sequence[dict] | None = None,
    ) -> None:
        """
        Index chunks in both BM25 and semantic indexes.

        Args:
            chunks: Text chunks to index.
            metadata: Optional metadata for each chunk.
        """
        self._chunks = list(chunks)
        self._metadata = list(metadata) if metadata else [{} for _ in chunks]

        # Index in BM25
        self.bm25_searcher.index(chunks, metadata)

        # Index in semantic retriever
        # The retriever interface may differ, adapt as needed
        if hasattr(self.retriever, "add"):
            # MemoryRetriever style
            for i, chunk in enumerate(chunks):
                meta = self._metadata[i] if i < len(self._metadata) else {}
                self.retriever.add(
                    text=chunk,
                    metadata={**meta, "chunk_index": i},
                )
        elif hasattr(self.retriever, "index"):
            self.retriever.index(chunks, metadata)

    def search(self, query: str, top_k: int = 5) -> list[SearchResult]:
        """
        Search using both BM25 and semantic, then fuse results.

        Args:
            query: Search query.
            top_k: Maximum number of results to return.

        Returns:
            Fused results ordered by combined relevance.
        """
        # Get more results from each to have good overlap
        fetch_k = top_k * 3

        # BM25 results
        bm25_results = self.bm25_searcher.search(query, fetch_k)

        # Semantic results
        semantic_results = self._search_semantic(query, fetch_k)

        # Fuse results using RRF
        fused = self._rrf_fusion(bm25_results, semantic_results)

        return fused[:top_k]

    def _search_semantic(self, query: str, top_k: int) -> list[SearchResult]:
        """Get semantic search results from retriever."""
        if not hasattr(self.retriever, "search"):
            return []

        try:
            # Retriever returns different format, convert to SearchResult
            retriever_results = self.retriever.search(query, top_k=top_k)

            results = []
            for r in retriever_results:
                # Handle different retriever result formats
                if hasattr(r, "text"):
                    text = r.text
                    score = getattr(r, "score", 0.0)
                    chunk_idx = getattr(r, "chunk_index", 0)
                    page = getattr(r, "page", None)
                    metadata = getattr(r, "metadata", {})
                elif isinstance(r, dict):
                    text = r.get("text", "")
                    score = r.get("score", 0.0)
                    chunk_idx = r.get("chunk_index", 0)
                    page = r.get("page")
                    metadata = r.get("metadata", {})
                else:
                    continue

                results.append(
                    SearchResult(
                        text=text,
                        score=score,
                        chunk_index=chunk_idx,
                        page=page,
                        metadata=metadata,
                    )
                )

            return results

        except Exception:
            return []

    def _rrf_fusion(
        self,
        bm25_results: list[SearchResult],
        semantic_results: list[SearchResult],
    ) -> list[SearchResult]:
        """
        Fuse results using Reciprocal Rank Fusion.

        RRF score = sum(1 / (k + rank)) for each result list.
        """
        # Build chunk_index -> result mapping
        result_by_chunk: dict[int, SearchResult] = {}
        rrf_scores: dict[int, float] = {}

        # Weight for each search type
        semantic_weight = 1.0 - self.bm25_weight

        # Process BM25 results
        for rank, result in enumerate(bm25_results):
            chunk_idx = result.chunk_index
            rrf_score = self.bm25_weight / (self.rrf_k + rank + 1)

            rrf_scores[chunk_idx] = rrf_scores.get(chunk_idx, 0) + rrf_score

            if chunk_idx not in result_by_chunk:
                result_by_chunk[chunk_idx] = result

        # Process semantic results
        for rank, result in enumerate(semantic_results):
            chunk_idx = result.chunk_index
            rrf_score = semantic_weight / (self.rrf_k + rank + 1)

            rrf_scores[chunk_idx] = rrf_scores.get(chunk_idx, 0) + rrf_score

            if chunk_idx not in result_by_chunk:
                result_by_chunk[chunk_idx] = result

        # Sort by RRF score
        sorted_chunks = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)

        # Build final results with updated scores
        results = []
        for chunk_idx, rrf_score in sorted_chunks:
            original = result_by_chunk[chunk_idx]
            results.append(
                SearchResult(
                    text=original.text,
                    score=rrf_score,
                    chunk_index=chunk_idx,
                    page=original.page,
                    metadata=original.metadata,
                )
            )

        return results

    def clear(self) -> None:
        """Clear both indexes."""
        self._chunks = []
        self._metadata = []
        self.bm25_searcher.clear()

        if hasattr(self.retriever, "clear"):
            self.retriever.clear()

    @property
    def is_indexed(self) -> bool:
        """Check if the searcher has indexed content."""
        return self.bm25_searcher.is_indexed

    @property
    def document_count(self) -> int:
        """Number of documents/chunks in the index."""
        return self.bm25_searcher.document_count


__all__ = [
    "HybridSearcher",
]
