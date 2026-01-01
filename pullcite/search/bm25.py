"""
BM25 text search implementation.

Uses tantivy (Rust-based search engine) for high-performance BM25 search.
Falls back to a pure-Python implementation if tantivy is not installed.
"""

from __future__ import annotations

import math
import re
import tempfile
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

from .base import SearchResult, Searcher


class BM25SearcherError(Exception):
    """Error in BM25 search operations."""

    pass


@dataclass
class BM25Searcher(Searcher):
    """
    BM25 text searcher using tantivy.

    Tantivy is a fast, full-featured search engine library written in Rust.
    This implementation uses it for BM25 scoring with good tokenization.

    If tantivy is not installed, falls back to a pure-Python BM25 implementation.

    Attributes:
        index_path: Path to store the index. If None, uses temp directory.
        k1: BM25 k1 parameter (term frequency saturation). Default 1.2.
        b: BM25 b parameter (length normalization). Default 0.75.

    Example:
        >>> searcher = BM25Searcher()
        >>> chunks = ["The deductible is $500.", "Copay is $20 per visit."]
        >>> searcher.index(chunks)
        >>> results = searcher.search("deductible amount")
        >>> print(results[0].text)
        'The deductible is $500.'
    """

    index_path: str | Path | None = None
    k1: float = 1.2
    b: float = 0.75

    # Internal state
    _index: Any = field(default=None, init=False, repr=False)
    _chunks: list[str] = field(default_factory=list, init=False, repr=False)
    _metadata: list[dict] = field(default_factory=list, init=False, repr=False)
    _use_tantivy: bool = field(default=True, init=False, repr=False)
    _temp_dir: Any = field(default=None, init=False, repr=False)

    # Pure Python BM25 state (fallback)
    _doc_freqs: dict[str, int] = field(default_factory=dict, init=False, repr=False)
    _doc_lengths: list[int] = field(default_factory=list, init=False, repr=False)
    _avg_doc_length: float = field(default=0.0, init=False, repr=False)
    _doc_term_freqs: list[Counter] = field(default_factory=list, init=False, repr=False)

    def __post_init__(self) -> None:
        """Initialize the searcher."""
        self._try_init_tantivy()

    def _try_init_tantivy(self) -> None:
        """Try to initialize tantivy, fall back to pure Python if not available."""
        try:
            import tantivy

            self._use_tantivy = True
        except ImportError:
            self._use_tantivy = False

    def _get_index_path(self) -> Path:
        """Get or create index path."""
        if self.index_path:
            path = Path(self.index_path)
            path.mkdir(parents=True, exist_ok=True)
            return path
        else:
            if self._temp_dir is None:
                self._temp_dir = tempfile.TemporaryDirectory()
            return Path(self._temp_dir.name)

    def _create_tantivy_index(self) -> Any:
        """Create a new tantivy index."""
        import tantivy

        # Define schema with text field and metadata
        schema_builder = tantivy.SchemaBuilder()
        schema_builder.add_text_field("content", stored=True)
        schema_builder.add_integer_field("chunk_index", stored=True)
        schema_builder.add_integer_field("page", stored=True)
        schema = schema_builder.build()

        # Create index
        index_path = self._get_index_path()
        index = tantivy.Index(schema, path=str(index_path))

        return index

    def _tokenize(self, text: str) -> list[str]:
        """Simple tokenization for pure Python BM25."""
        # Lowercase, split on non-alphanumeric, filter empty
        text = text.lower()
        tokens = re.split(r"[^a-z0-9]+", text)
        return [t for t in tokens if t and len(t) > 1]

    def index(
        self,
        chunks: Sequence[str],
        metadata: Sequence[dict] | None = None,
    ) -> None:
        """
        Index chunks for searching.

        Args:
            chunks: Text chunks to index.
            metadata: Optional metadata for each chunk.
                      Expected keys: 'page' (int), 'chunk_index' (int)
        """
        self.clear()

        self._chunks = list(chunks)
        self._metadata = list(metadata) if metadata else [{} for _ in chunks]

        if not chunks:
            return

        if self._use_tantivy:
            self._index_tantivy(chunks, self._metadata)
        else:
            self._index_python(chunks)

    def _index_tantivy(self, chunks: Sequence[str], metadata: Sequence[dict]) -> None:
        """Index using tantivy."""
        import tantivy

        self._index = self._create_tantivy_index()
        writer = self._index.writer()

        for i, (chunk, meta) in enumerate(zip(chunks, metadata)):
            doc = tantivy.Document()
            doc.add_text("content", chunk)
            doc.add_integer("chunk_index", meta.get("chunk_index", i))
            doc.add_integer("page", meta.get("page", 0))
            writer.add_document(doc)

        writer.commit()
        self._index.reload()

    def _index_python(self, chunks: Sequence[str]) -> None:
        """Index using pure Python BM25."""
        self._doc_term_freqs = []
        self._doc_lengths = []
        self._doc_freqs = Counter()

        for chunk in chunks:
            tokens = self._tokenize(chunk)
            term_freq = Counter(tokens)

            self._doc_term_freqs.append(term_freq)
            self._doc_lengths.append(len(tokens))

            # Update document frequencies
            for term in set(tokens):
                self._doc_freqs[term] += 1

        total_length = sum(self._doc_lengths)
        self._avg_doc_length = total_length / len(chunks) if chunks else 0

    def search(self, query: str, top_k: int = 5) -> list[SearchResult]:
        """
        Search for chunks matching the query.

        Args:
            query: Search query (keywords/terms).
            top_k: Maximum number of results to return.

        Returns:
            List of SearchResults ordered by relevance.
        """
        if not self._chunks:
            return []

        if self._use_tantivy:
            return self._search_tantivy(query, top_k)
        else:
            return self._search_python(query, top_k)

    def _search_tantivy(self, query: str, top_k: int) -> list[SearchResult]:
        """Search using tantivy."""
        import tantivy

        if self._index is None:
            return []

        searcher = self._index.searcher()
        query_parser = tantivy.QueryParser.for_index(self._index, ["content"])

        try:
            parsed_query = query_parser.parse_query(query)
        except Exception:
            # If query parsing fails, try escaping special characters
            escaped = re.sub(r'([+\-&|!(){}[\]^"~*?:\\/])', r"\\\1", query)
            try:
                parsed_query = query_parser.parse_query(escaped)
            except Exception:
                return []

        results = []
        search_results = searcher.search(parsed_query, top_k).hits

        for score, doc_address in search_results:
            doc = searcher.doc(doc_address)
            content = doc.get_first("content")
            chunk_index = doc.get_first("chunk_index") or 0
            page = doc.get_first("page")

            # Get metadata for this chunk
            meta = self._metadata[chunk_index] if chunk_index < len(self._metadata) else {}

            results.append(
                SearchResult(
                    text=content,
                    score=float(score),
                    chunk_index=chunk_index,
                    page=page if page else meta.get("page"),
                    metadata=meta,
                )
            )

        return results

    def _search_python(self, query: str, top_k: int) -> list[SearchResult]:
        """Search using pure Python BM25."""
        query_tokens = self._tokenize(query)

        if not query_tokens:
            return []

        n = len(self._chunks)
        scores = []

        for i in range(n):
            score = self._compute_bm25_score(query_tokens, i)
            if score > 0:
                scores.append((i, score))

        # Sort by score descending
        scores.sort(key=lambda x: x[1], reverse=True)

        results = []
        for chunk_idx, score in scores[:top_k]:
            meta = self._metadata[chunk_idx] if chunk_idx < len(self._metadata) else {}

            results.append(
                SearchResult(
                    text=self._chunks[chunk_idx],
                    score=score,
                    chunk_index=chunk_idx,
                    page=meta.get("page"),
                    metadata=meta,
                )
            )

        return results

    def _compute_bm25_score(self, query_tokens: list[str], doc_idx: int) -> float:
        """Compute BM25 score for a document."""
        n = len(self._chunks)
        doc_len = self._doc_lengths[doc_idx]
        term_freqs = self._doc_term_freqs[doc_idx]

        score = 0.0
        for term in query_tokens:
            if term not in term_freqs:
                continue

            tf = term_freqs[term]
            df = self._doc_freqs.get(term, 0)

            # IDF with smoothing
            idf = math.log((n - df + 0.5) / (df + 0.5) + 1)

            # BM25 TF component
            numerator = tf * (self.k1 + 1)
            denominator = tf + self.k1 * (
                1 - self.b + self.b * (doc_len / self._avg_doc_length)
            )

            score += idf * (numerator / denominator)

        return score

    def clear(self) -> None:
        """Clear the index."""
        self._chunks = []
        self._metadata = []
        self._index = None

        # Clear Python BM25 state
        self._doc_freqs = {}
        self._doc_lengths = []
        self._avg_doc_length = 0.0
        self._doc_term_freqs = []

        # Clean up temp directory if using tantivy
        if self._temp_dir is not None:
            try:
                self._temp_dir.cleanup()
            except Exception:
                pass
            self._temp_dir = None

    @property
    def is_indexed(self) -> bool:
        """Check if the searcher has indexed content."""
        return len(self._chunks) > 0

    @property
    def document_count(self) -> int:
        """Number of documents/chunks in the index."""
        return len(self._chunks)

    @property
    def using_tantivy(self) -> bool:
        """Check if using tantivy (vs pure Python fallback)."""
        return self._use_tantivy

    def __del__(self) -> None:
        """Cleanup on deletion."""
        if self._temp_dir is not None:
            try:
                self._temp_dir.cleanup()
            except Exception:
                pass


__all__ = [
    "BM25Searcher",
    "BM25SearcherError",
]
