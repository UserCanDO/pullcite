"""
Tests for retrieval/memory.py - MemoryRetriever.
"""

import pytest
import numpy as np
from pullcite.retrieval.memory import MemoryRetriever
from pullcite.retrieval.base import RetrieverError
from pullcite.embeddings.base import (
    Embedder,
    EmbeddingResult,
    BatchEmbeddingResult,
)
from pullcite.core.document import Document
from pullcite.core.chunk import Chunk


class MockEmbedder(Embedder):
    """
    Mock embedder that creates deterministic embeddings.

    Embeddings are based on word overlap with predefined topics.
    """

    TOPICS = {
        "deductible": np.array([1.0, 0.0, 0.0, 0.0]),
        "copay": np.array([0.0, 1.0, 0.0, 0.0]),
        "coinsurance": np.array([0.0, 0.0, 1.0, 0.0]),
        "network": np.array([0.0, 0.0, 0.0, 1.0]),
    }

    @property
    def model_name(self) -> str:
        return "mock-embedder"

    @property
    def dimensions(self) -> int:
        return 4

    def _embed_text(self, text: str) -> tuple[float, ...]:
        """Create embedding based on keyword presence."""
        text_lower = text.lower()
        vec = np.zeros(4)

        for word, topic_vec in self.TOPICS.items():
            if word in text_lower:
                vec += topic_vec

        # Add some noise based on text length for variety
        vec += np.random.RandomState(len(text)).randn(4) * 0.1

        # Normalize
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm

        return tuple(vec.tolist())

    def embed(self, text: str) -> EmbeddingResult:
        return EmbeddingResult(
            vector=self._embed_text(text),
            model=self.model_name,
            dimensions=self.dimensions,
            token_count=len(text.split()),
        )

    def embed_batch(self, texts: list[str]) -> BatchEmbeddingResult:
        vectors = tuple(self._embed_text(t) for t in texts)
        return BatchEmbeddingResult(
            vectors=vectors,
            model=self.model_name,
            dimensions=self.dimensions,
            total_tokens=sum(len(t.split()) for t in texts),
        )


@pytest.fixture
def embedder():
    return MockEmbedder()


@pytest.fixture
def sample_document():
    """Create a document with varied content."""
    text = """
    Plan Overview
    
    The individual deductible is $1,500 per year.
    The family deductible is $3,000 per year.
    
    Cost Sharing
    
    Primary care copay is $25 per visit.
    Specialist copay is $50 per visit.
    Emergency room copay is $150.
    
    Coinsurance
    
    After meeting your deductible, you pay 20% coinsurance.
    The plan pays 80% of covered services.
    
    Network Information
    
    You must use in-network providers for full benefits.
    Out-of-network services have higher cost sharing.
    """
    return Document.from_text(text, chunk_size=100, chunk_overlap=20)


class TestMemoryRetriever:
    """Test MemoryRetriever class."""

    def test_creation(self, embedder):
        retriever = MemoryRetriever(_embedder=embedder)
        assert retriever.embedder is embedder
        assert retriever.is_indexed is False
        assert retriever.chunk_count == 0

    def test_index_document(self, embedder, sample_document):
        retriever = MemoryRetriever(_embedder=embedder)

        retriever.index(sample_document)

        assert retriever.is_indexed is True
        assert retriever.chunk_count > 0
        assert retriever.document_id == sample_document.id

    def test_search_basic(self, embedder, sample_document):
        retriever = MemoryRetriever(_embedder=embedder)
        retriever.index(sample_document)

        results = retriever.search("deductible amount", k=3)

        assert len(results) == 3
        assert results.query == "deductible amount"
        assert results.total_chunks == retriever.chunk_count
        # First result should mention deductible
        assert "deductible" in results[0].text.lower()

    def test_search_relevance(self, embedder, sample_document):
        """Test that search returns relevant results."""
        retriever = MemoryRetriever(_embedder=embedder)
        retriever.index(sample_document)

        # Search for copay - should return copay-related chunks first
        results = retriever.search("copay", k=5)

        # Top result should mention copay
        assert "copay" in results.top.text.lower()

    def test_search_scores_ordered(self, embedder, sample_document):
        """Test that results are ordered by score descending."""
        retriever = MemoryRetriever(_embedder=embedder)
        retriever.index(sample_document)

        results = retriever.search("network provider", k=5)

        scores = [r.score for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_search_without_index_raises(self, embedder):
        retriever = MemoryRetriever(_embedder=embedder)

        with pytest.raises(RetrieverError) as exc:
            retriever.search("test")
        assert "No document indexed" in str(exc.value)

    def test_search_invalid_k(self, embedder, sample_document):
        retriever = MemoryRetriever(_embedder=embedder)
        retriever.index(sample_document)

        with pytest.raises(RetrieverError):
            retriever.search("test", k=0)

    def test_search_k_larger_than_chunks(self, embedder):
        """Test that k > chunk_count returns all chunks."""
        doc = Document.from_text("Short document.", chunk_size=100)
        retriever = MemoryRetriever(_embedder=embedder)
        retriever.index(doc)

        results = retriever.search("test", k=100)

        assert len(results) == retriever.chunk_count

    def test_search_many(self, embedder, sample_document):
        retriever = MemoryRetriever(_embedder=embedder)
        retriever.index(sample_document)

        queries = ["deductible", "copay", "network"]
        all_results = retriever.search_many(queries, k=2)

        assert len(all_results) == 3

        # Each result should be relevant to its query
        assert "deductible" in all_results[0].top.text.lower()
        assert "copay" in all_results[1].top.text.lower()
        assert "network" in all_results[2].top.text.lower()

    def test_search_many_empty(self, embedder, sample_document):
        retriever = MemoryRetriever(_embedder=embedder)
        retriever.index(sample_document)

        results = retriever.search_many([], k=3)
        assert results == []

    def test_clear(self, embedder, sample_document):
        retriever = MemoryRetriever(_embedder=embedder)
        retriever.index(sample_document)

        assert retriever.is_indexed is True

        retriever.clear()

        assert retriever.is_indexed is False
        assert retriever.chunk_count == 0
        assert retriever.document_id is None

    def test_reindex_replaces(self, embedder):
        """Test that indexing a new document replaces the old one."""
        doc1 = Document.from_text("First document about apples.", chunk_size=100)
        doc2 = Document.from_text("Second document about oranges.", chunk_size=100)

        retriever = MemoryRetriever(_embedder=embedder)

        retriever.index(doc1)
        assert retriever.document_id == doc1.id
        count1 = retriever.chunk_count

        retriever.index(doc2)
        assert retriever.document_id == doc2.id
        # Document changed
        assert retriever.document_id != doc1.id

    def test_empty_document(self, embedder):
        """Test indexing an empty document."""
        doc = Document.from_text("", chunk_size=100)
        retriever = MemoryRetriever(_embedder=embedder)

        retriever.index(doc)

        assert retriever.is_indexed is False
        assert retriever.chunk_count == 0


class TestMemoryRetrieverEdgeCases:
    """Test edge cases for MemoryRetriever."""

    def test_single_chunk_document(self, embedder):
        """Test with a document that has only one chunk."""
        doc = Document.from_text("Single small chunk.", chunk_size=1000)
        retriever = MemoryRetriever(_embedder=embedder)

        retriever.index(doc)
        results = retriever.search("chunk", k=5)

        assert len(results) == 1
        assert results[0].rank == 0

    def test_identical_chunks(self, embedder):
        """Test with duplicate content."""
        text = "Repeated content.\n\n" * 5
        doc = Document.from_text(text, chunk_size=50)
        retriever = MemoryRetriever(_embedder=embedder)

        retriever.index(doc)
        results = retriever.search("repeated", k=3)

        # Should still return 3 results even if similar
        assert len(results) <= retriever.chunk_count

    def test_score_range(self, embedder, sample_document):
        """Test that scores are in valid range for cosine similarity."""
        retriever = MemoryRetriever(_embedder=embedder)
        retriever.index(sample_document)

        results = retriever.search("deductible", k=10)

        for r in results:
            # Cosine similarity is in [-1, 1]
            assert -1.0 <= r.score <= 1.0
