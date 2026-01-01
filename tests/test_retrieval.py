"""
Tests for retrieval/base.py - Retriever ABC and SearchResult types.
"""

import pytest
from pullcite.retrieval.base import (
    Retriever,
    SearchResult,
    SearchResults,
    RetrieverError,
)
from pullcite.core.chunk import Chunk


class TestSearchResult:
    """Test SearchResult dataclass."""

    def test_basic_creation(self):
        chunk = Chunk(text="Test content", index=0)
        result = SearchResult(chunk=chunk, score=0.95, rank=0)

        assert result.chunk is chunk
        assert result.score == 0.95
        assert result.rank == 0

    def test_convenience_accessors(self):
        chunk = Chunk(text="Test content", index=5, page=3)
        result = SearchResult(chunk=chunk, score=0.8, rank=1)

        assert result.text == "Test content"
        assert result.page == 3
        assert result.index == 5

    def test_immutability(self):
        chunk = Chunk(text="Test", index=0)
        result = SearchResult(chunk=chunk, score=0.9, rank=0)

        with pytest.raises(AttributeError):
            result.score = 0.5


class TestSearchResults:
    """Test SearchResults dataclass."""

    def test_basic_creation(self):
        chunks = [
            Chunk(text="First", index=0),
            Chunk(text="Second", index=1),
        ]
        results = (
            SearchResult(chunk=chunks[0], score=0.9, rank=0),
            SearchResult(chunk=chunks[1], score=0.7, rank=1),
        )
        sr = SearchResults(results=results, query="test query", total_chunks=10)

        assert len(sr) == 2
        assert sr.query == "test query"
        assert sr.total_chunks == 10

    def test_iteration(self):
        chunks = [Chunk(text=f"Chunk {i}", index=i) for i in range(3)]
        results = tuple(
            SearchResult(chunk=c, score=0.9 - i * 0.1, rank=i)
            for i, c in enumerate(chunks)
        )
        sr = SearchResults(results=results, query="q", total_chunks=3)

        texts = [r.text for r in sr]
        assert texts == ["Chunk 0", "Chunk 1", "Chunk 2"]

    def test_indexing(self):
        chunk = Chunk(text="Only one", index=0)
        result = SearchResult(chunk=chunk, score=0.9, rank=0)
        sr = SearchResults(results=(result,), query="q", total_chunks=1)

        assert sr[0] is result

    def test_top_property(self):
        chunks = [Chunk(text=f"C{i}", index=i) for i in range(2)]
        results = (
            SearchResult(chunk=chunks[0], score=0.9, rank=0),
            SearchResult(chunk=chunks[1], score=0.5, rank=1),
        )
        sr = SearchResults(results=results, query="q", total_chunks=2)

        assert sr.top is not None
        assert sr.top.text == "C0"
        assert sr.top.score == 0.9

    def test_top_empty(self):
        sr = SearchResults(results=(), query="q", total_chunks=0)
        assert sr.top is None

    def test_chunks_property(self):
        chunks = [Chunk(text=f"C{i}", index=i) for i in range(3)]
        results = tuple(
            SearchResult(chunk=c, score=0.5, rank=i) for i, c in enumerate(chunks)
        )
        sr = SearchResults(results=results, query="q", total_chunks=3)

        assert sr.chunks == chunks

    def test_texts_property(self):
        chunks = [Chunk(text=f"Text {i}", index=i) for i in range(2)]
        results = tuple(
            SearchResult(chunk=c, score=0.5, rank=i) for i, c in enumerate(chunks)
        )
        sr = SearchResults(results=results, query="q", total_chunks=2)

        assert sr.texts == ["Text 0", "Text 1"]

    def test_above_threshold(self):
        chunks = [Chunk(text=f"C{i}", index=i) for i in range(4)]
        results = (
            SearchResult(chunk=chunks[0], score=0.9, rank=0),
            SearchResult(chunk=chunks[1], score=0.7, rank=1),
            SearchResult(chunk=chunks[2], score=0.5, rank=2),
            SearchResult(chunk=chunks[3], score=0.3, rank=3),
        )
        sr = SearchResults(results=results, query="q", total_chunks=4)

        # Filter to score >= 0.6
        filtered = sr.above_threshold(0.6)

        assert len(filtered) == 2
        assert filtered[0].text == "C0"
        assert filtered[0].rank == 0  # Re-ranked
        assert filtered[1].text == "C1"
        assert filtered[1].rank == 1  # Re-ranked

    def test_above_threshold_preserves_metadata(self):
        chunk = Chunk(text="Test", index=0)
        result = SearchResult(chunk=chunk, score=0.9, rank=0)
        sr = SearchResults(results=(result,), query="original query", total_chunks=100)

        filtered = sr.above_threshold(0.5)

        assert filtered.query == "original query"
        assert filtered.total_chunks == 100


class TestRetrieverError:
    """Test RetrieverError exception."""

    def test_basic_error(self):
        error = RetrieverError("Index not found")
        assert str(error) == "Index not found"
        assert error.cause is None

    def test_error_with_cause(self):
        cause = IOError("Disk full")
        error = RetrieverError("Failed to save", cause=cause)
        assert error.cause is cause


class TestRetrieverABC:
    """Test Retriever abstract base class."""

    def test_cannot_instantiate_directly(self):
        with pytest.raises(TypeError):
            Retriever()

    def test_search_many_default_implementation(self):
        """Test that search_many calls search() by default."""
        from pullcite.embeddings.base import (
            Embedder,
            EmbeddingResult,
            BatchEmbeddingResult,
        )

        # Create mock embedder
        class MockEmbedder(Embedder):
            @property
            def model_name(self) -> str:
                return "mock"

            @property
            def dimensions(self) -> int:
                return 2

            def embed(self, text: str) -> EmbeddingResult:
                return EmbeddingResult(
                    vector=(0.0, 0.0),
                    model="mock",
                    dimensions=2,
                    token_count=1,
                )

            def embed_batch(self, texts: list[str]) -> BatchEmbeddingResult:
                return BatchEmbeddingResult(
                    vectors=tuple((0.0, 0.0) for _ in texts),
                    model="mock",
                    dimensions=2,
                    total_tokens=len(texts),
                )

        # Create mock retriever
        class MockRetriever(Retriever):
            def __init__(self):
                self._embedder = MockEmbedder()
                self.search_calls = []

            @property
            def embedder(self):
                return self._embedder

            @property
            def is_indexed(self):
                return True

            @property
            def chunk_count(self):
                return 10

            def index(self, document):
                pass

            def search(self, query: str, k: int = 5) -> SearchResults:
                self.search_calls.append(query)
                chunk = Chunk(text=f"Result for: {query}", index=0)
                result = SearchResult(chunk=chunk, score=0.9, rank=0)
                return SearchResults(
                    results=(result,),
                    query=query,
                    total_chunks=10,
                )

            def clear(self):
                pass

        retriever = MockRetriever()
        results = retriever.search_many(["q1", "q2", "q3"], k=3)

        assert len(results) == 3
        assert retriever.search_calls == ["q1", "q2", "q3"]
        assert results[0].query == "q1"
        assert results[1].query == "q2"
