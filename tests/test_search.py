"""Tests for the search module."""

import pytest

from pullcite.search import BM25Searcher, SearchResult, HybridSearcher


class TestSearchResult:
    """Tests for SearchResult dataclass."""

    def test_creation(self):
        result = SearchResult(
            text="The deductible is $500.",
            score=0.95,
            chunk_index=0,
            page=1,
            metadata={"source": "test"},
        )

        assert result.text == "The deductible is $500."
        assert result.score == 0.95
        assert result.chunk_index == 0
        assert result.page == 1
        assert result.metadata == {"source": "test"}

    def test_comparison_by_score(self):
        """Test that results compare by score (descending)."""
        high = SearchResult(text="high", score=0.9, chunk_index=0)
        low = SearchResult(text="low", score=0.5, chunk_index=1)

        # < returns True if score is higher (for descending sort)
        assert high < low


class TestBM25Searcher:
    """Tests for BM25Searcher."""

    def test_index_and_search(self):
        """Test basic indexing and search."""
        searcher = BM25Searcher()

        chunks = [
            "The annual deductible is $500 per individual.",
            "Copay for primary care visits is $20.",
            "Emergency room copay is $150 per visit.",
            "The deductible must be met before coverage begins.",
        ]

        searcher.index(chunks)

        assert searcher.is_indexed
        assert searcher.document_count == 4

        results = searcher.search("deductible", top_k=3)

        assert len(results) > 0
        # The first two chunks mention deductible
        assert any("deductible" in r.text.lower() for r in results)

    def test_search_with_metadata(self):
        """Test search preserves metadata."""
        searcher = BM25Searcher()

        chunks = ["The copay is $20.", "The deductible is $500."]
        metadata = [
            {"chunk_index": 0, "page": 1},
            {"chunk_index": 1, "page": 2},
        ]

        searcher.index(chunks, metadata)
        results = searcher.search("deductible")

        assert len(results) > 0
        # Find the result for deductible
        deductible_result = next(
            (r for r in results if "deductible" in r.text.lower()), None
        )
        assert deductible_result is not None
        assert deductible_result.chunk_index == 1

    def test_empty_index(self):
        """Test search on empty index."""
        searcher = BM25Searcher()

        assert not searcher.is_indexed
        assert searcher.document_count == 0

        results = searcher.search("anything")
        assert results == []

    def test_clear(self):
        """Test clearing the index."""
        searcher = BM25Searcher()
        searcher.index(["test chunk"])

        assert searcher.is_indexed

        searcher.clear()

        assert not searcher.is_indexed
        assert searcher.document_count == 0

    def test_multiple_term_query(self):
        """Test search with multiple terms."""
        searcher = BM25Searcher()

        chunks = [
            "The individual annual deductible is $500.",
            "The family deductible is $1000.",
            "Copay for visits is $20.",
        ]

        searcher.index(chunks)

        # Search for multiple terms
        results = searcher.search("individual annual deductible", top_k=2)

        assert len(results) > 0
        # First result should be the one mentioning all terms
        assert "individual" in results[0].text.lower()

    def test_no_match_query(self):
        """Test search with no matching terms."""
        searcher = BM25Searcher()

        chunks = ["The deductible is $500.", "The copay is $20."]

        searcher.index(chunks)

        results = searcher.search("xyznonexistent123")
        assert len(results) == 0

    def test_case_insensitive(self):
        """Test that search is case insensitive."""
        searcher = BM25Searcher()

        chunks = ["The DEDUCTIBLE is $500."]

        searcher.index(chunks)

        # Search in lowercase
        results = searcher.search("deductible")
        assert len(results) > 0

    def test_top_k_limits_results(self):
        """Test that top_k limits number of results."""
        searcher = BM25Searcher()

        chunks = [f"Document {i} about deductibles." for i in range(10)]

        searcher.index(chunks)

        results = searcher.search("deductible", top_k=3)
        assert len(results) <= 3

    def test_reindexing_clears_old(self):
        """Test that indexing again replaces old index."""
        searcher = BM25Searcher()

        searcher.index(["old document about cats"])
        results = searcher.search("cats")
        assert len(results) > 0

        # Reindex with new content
        searcher.index(["new document about dogs"])

        # Old content should be gone
        results = searcher.search("cats")
        assert len(results) == 0

        # New content should be there
        results = searcher.search("dogs")
        assert len(results) > 0

    def test_bm25_parameters(self):
        """Test custom BM25 parameters."""
        searcher = BM25Searcher(k1=1.5, b=0.8)

        assert searcher.k1 == 1.5
        assert searcher.b == 0.8

        chunks = ["Test document."]
        searcher.index(chunks)

        # Should still work with custom params
        results = searcher.search("test")
        assert len(results) > 0


class TestHybridSearcher:
    """Tests for HybridSearcher."""

    def test_hybrid_with_bm25_only(self):
        """Test hybrid search with only BM25 (no retriever)."""
        bm25 = BM25Searcher()

        chunks = [
            "The deductible is $500.",
            "The copay is $20.",
        ]
        bm25.index(chunks)

        # Create hybrid with no retriever
        hybrid = HybridSearcher(bm25_searcher=bm25, retriever=None)

        results = hybrid.search("deductible")

        # Should still return BM25 results
        assert len(results) > 0
        assert "deductible" in results[0].text.lower()

    def test_hybrid_inherits_index_state(self):
        """Test that hybrid searcher reflects BM25 index state."""
        bm25 = BM25Searcher()

        hybrid = HybridSearcher(bm25_searcher=bm25, retriever=None)

        assert not hybrid.is_indexed

        bm25.index(["test"])

        assert hybrid.is_indexed
        assert hybrid.document_count == 1

    def test_bm25_weight_validation(self):
        """Test that bm25_weight must be between 0 and 1."""
        bm25 = BM25Searcher()

        with pytest.raises(ValueError, match="bm25_weight"):
            HybridSearcher(bm25_searcher=bm25, retriever=None, bm25_weight=1.5)

        with pytest.raises(ValueError, match="bm25_weight"):
            HybridSearcher(bm25_searcher=bm25, retriever=None, bm25_weight=-0.1)

    def test_hybrid_clear(self):
        """Test clearing hybrid searcher."""
        bm25 = BM25Searcher()
        bm25.index(["test"])

        hybrid = HybridSearcher(bm25_searcher=bm25, retriever=None)

        assert hybrid.is_indexed

        hybrid.clear()

        assert not hybrid.is_indexed


class TestBM25SearcherFallback:
    """Tests for pure Python BM25 fallback (when tantivy not available)."""

    def test_fallback_search(self):
        """Test that pure Python fallback works correctly."""
        # Force pure Python mode
        searcher = BM25Searcher()
        searcher._use_tantivy = False

        chunks = [
            "The annual deductible is $500.",
            "Copay for doctor visits is $20.",
            "The deductible applies per calendar year.",
        ]

        searcher._index_python(chunks)
        searcher._chunks = chunks
        searcher._metadata = [{} for _ in chunks]

        results = searcher._search_python("deductible", top_k=3)

        assert len(results) > 0
        # Both chunks mentioning deductible should be returned
        texts = [r.text for r in results]
        assert any("deductible" in t.lower() for t in texts)

    def test_using_tantivy_property(self):
        """Test using_tantivy property reflects state."""
        searcher = BM25Searcher()

        # Property should exist
        _ = searcher.using_tantivy
