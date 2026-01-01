"""
Tests for retrieval/chroma.py - ChromaDB retriever implementation.
"""

from unittest.mock import MagicMock, patch

import pytest
from pullcite.retrieval.base import RetrieverError
from pullcite.retrieval.chroma import ChromaRetriever


class MockEmbedder:
    """Mock embedder for testing."""

    @property
    def model_name(self) -> str:
        return "mock-model"

    @property
    def dimensions(self) -> int:
        return 3

    def embed(self, text: str):
        result = MagicMock()
        result.vector = (0.1, 0.2, 0.3)
        return result

    def embed_many(self, texts: list[str]) -> list[tuple[float, ...]]:
        return [(0.1, 0.2, 0.3) for _ in texts]


class MockChunk:
    """Mock chunk for testing."""

    def __init__(self, index: int, text: str, page: int | None = None):
        self.index = index
        self.text = text
        self.page = page


class MockDocument:
    """Mock document for testing."""

    def __init__(self, id: str, chunks: list[MockChunk]):
        self.id = id
        self.chunks = chunks


class TestChromaRetrieverInit:
    """Test ChromaRetriever initialization."""

    def test_default_values(self):
        embedder = MockEmbedder()
        retriever = ChromaRetriever(embedder)

        assert retriever.collection_name == "pullcite_documents"
        assert retriever.persist_directory is None
        assert retriever.distance_metric == "cosine"

    def test_invalid_distance_metric(self):
        embedder = MockEmbedder()
        with pytest.raises(RetrieverError) as exc:
            ChromaRetriever(embedder, distance_metric="invalid")
        assert "Invalid distance_metric" in str(exc.value)

    def test_valid_distance_metrics(self):
        embedder = MockEmbedder()
        ChromaRetriever(embedder, distance_metric="cosine")
        ChromaRetriever(embedder, distance_metric="l2")
        ChromaRetriever(embedder, distance_metric="ip")


class TestChromaRetrieverProperties:
    """Test ChromaRetriever properties."""

    def test_embedder_property(self):
        embedder = MockEmbedder()
        retriever = ChromaRetriever(embedder)
        assert retriever.embedder is embedder

    def test_is_indexed_false_initially(self):
        embedder = MockEmbedder()
        retriever = ChromaRetriever(embedder)
        assert retriever.is_indexed is False

    def test_chunk_count_zero_initially(self):
        embedder = MockEmbedder()
        retriever = ChromaRetriever(embedder)
        assert retriever.chunk_count == 0

    def test_document_id_none_initially(self):
        embedder = MockEmbedder()
        retriever = ChromaRetriever(embedder)
        assert retriever.document_id is None


class TestChromaRetrieverIndex:
    """Test ChromaRetriever index method."""

    def test_index_empty_document(self):
        embedder = MockEmbedder()
        retriever = ChromaRetriever(embedder)

        mock_collection = MagicMock()

        with patch.object(retriever, "_get_collection", return_value=mock_collection):
            doc = MockDocument("doc1", [])
            retriever.index(doc)

        assert retriever.document_id == "doc1"
        assert retriever.chunk_count == 0

    def test_index_with_chunks(self):
        embedder = MockEmbedder()
        retriever = ChromaRetriever(embedder)

        mock_collection = MagicMock()

        with patch.object(retriever, "_get_collection", return_value=mock_collection):
            chunks = [
                MockChunk(0, "Hello world", page=1),
                MockChunk(1, "Test chunk", page=1),
            ]
            doc = MockDocument("doc1", chunks)
            retriever.index(doc)

        assert retriever.is_indexed
        assert retriever.chunk_count == 2
        assert retriever.document_id == "doc1"

        # Verify collection.add was called
        mock_collection.add.assert_called_once()
        call_kwargs = mock_collection.add.call_args[1]
        assert len(call_kwargs["ids"]) == 2
        assert len(call_kwargs["embeddings"]) == 2
        assert len(call_kwargs["documents"]) == 2


class TestChromaRetrieverSearch:
    """Test ChromaRetriever search method."""

    def test_search_not_indexed_raises(self):
        embedder = MockEmbedder()
        retriever = ChromaRetriever(embedder)

        with pytest.raises(RetrieverError) as exc:
            retriever.search("test query")
        assert "No document indexed" in str(exc.value)

    def test_search_k_less_than_1_raises(self):
        embedder = MockEmbedder()
        retriever = ChromaRetriever(embedder)

        # Manually set indexed state
        retriever._chunks = [MockChunk(0, "test")]

        with pytest.raises(RetrieverError) as exc:
            retriever.search("test", k=0)
        assert "k must be >= 1" in str(exc.value)

    def test_search_basic(self):
        embedder = MockEmbedder()
        retriever = ChromaRetriever(embedder)

        mock_collection = MagicMock()
        mock_collection.query.return_value = {
            "ids": [["doc1_0", "doc1_1"]],
            "distances": [[0.1, 0.3]],
            "documents": [["Hello world", "Test chunk"]],
            "metadatas": [[{"chunk_index": 0}, {"chunk_index": 1}]],
        }

        chunks = [
            MockChunk(0, "Hello world"),
            MockChunk(1, "Test chunk"),
        ]
        retriever._chunks = chunks
        retriever._document_id = "doc1"

        with patch.object(retriever, "_get_collection", return_value=mock_collection):
            results = retriever.search("hello", k=2)

        assert len(results) == 2
        assert results.query == "hello"
        assert results.total_chunks == 2

        # Check score conversion (cosine: score = 1 - distance)
        assert results[0].score == pytest.approx(0.9)
        assert results[1].score == pytest.approx(0.7)


class TestChromaRetrieverClear:
    """Test ChromaRetriever clear method."""

    def test_clear_resets_state(self):
        embedder = MockEmbedder()
        retriever = ChromaRetriever(embedder)

        # Set some state
        retriever._chunks = [MockChunk(0, "test")]
        retriever._document_id = "doc1"
        retriever._collection = MagicMock()
        retriever._client = MagicMock()

        retriever.clear()

        assert retriever._chunks == []
        assert retriever._document_id is None


class TestChromaRetrieverDistanceMetrics:
    """Test different distance metrics."""

    @pytest.mark.parametrize(
        "metric,distance,expected_score",
        [
            ("cosine", 0.2, 0.8),  # 1 - distance
            ("l2", 1.0, 0.5),  # 1 / (1 + distance)
            ("ip", -0.5, 0.5),  # -distance
        ],
    )
    def test_score_conversion(self, metric, distance, expected_score):
        embedder = MockEmbedder()
        retriever = ChromaRetriever(embedder, distance_metric=metric)

        mock_collection = MagicMock()
        mock_collection.query.return_value = {
            "ids": [["doc1_0"]],
            "distances": [[distance]],
            "documents": [["test"]],
            "metadatas": [[{"chunk_index": 0}]],
        }

        retriever._chunks = [MockChunk(0, "test")]
        retriever._document_id = "doc1"

        with patch.object(retriever, "_get_collection", return_value=mock_collection):
            results = retriever.search("query", k=1)

        assert results[0].score == pytest.approx(expected_score)
