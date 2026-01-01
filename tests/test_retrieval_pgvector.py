"""
Tests for retrieval/pgvector.py - PostgreSQL pgvector retriever implementation.
"""

from unittest.mock import MagicMock, patch

import pytest
from pullcite.retrieval.base import RetrieverError
from pullcite.retrieval.pgvector import PgVectorRetriever


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


class TestPgVectorRetrieverInit:
    """Test PgVectorRetriever initialization."""

    def test_default_values(self):
        embedder = MockEmbedder()
        retriever = PgVectorRetriever(embedder)

        assert retriever.connection_string == "postgresql://localhost/pullcite"
        assert retriever.table_name == "pullcite_chunks"
        assert retriever.distance_metric == "cosine"

    def test_invalid_distance_metric(self):
        embedder = MockEmbedder()
        with pytest.raises(RetrieverError) as exc:
            PgVectorRetriever(embedder, distance_metric="invalid")
        assert "Invalid distance_metric" in str(exc.value)

    def test_valid_distance_metrics(self):
        embedder = MockEmbedder()
        PgVectorRetriever(embedder, distance_metric="cosine")
        PgVectorRetriever(embedder, distance_metric="l2")
        PgVectorRetriever(embedder, distance_metric="ip")

    def test_custom_table_name(self):
        embedder = MockEmbedder()
        retriever = PgVectorRetriever(embedder, table_name="custom_chunks")
        assert retriever.table_name == "custom_chunks"


class TestPgVectorRetrieverProperties:
    """Test PgVectorRetriever properties."""

    def test_embedder_property(self):
        embedder = MockEmbedder()
        retriever = PgVectorRetriever(embedder)
        assert retriever.embedder is embedder

    def test_is_indexed_false_initially(self):
        embedder = MockEmbedder()
        retriever = PgVectorRetriever(embedder)
        assert retriever.is_indexed is False

    def test_chunk_count_zero_initially(self):
        embedder = MockEmbedder()
        retriever = PgVectorRetriever(embedder)
        assert retriever.chunk_count == 0

    def test_document_id_none_initially(self):
        embedder = MockEmbedder()
        retriever = PgVectorRetriever(embedder)
        assert retriever.document_id is None


class TestPgVectorRetrieverConnection:
    """Test database connection handling."""

    def test_connection_error_without_psycopg2(self):
        embedder = MockEmbedder()
        retriever = PgVectorRetriever(embedder)

        with patch.dict("sys.modules", {"psycopg2": None, "psycopg": None}):
            with patch.object(
                retriever,
                "_get_connection",
                side_effect=RetrieverError(
                    "psycopg2 or psycopg package required. "
                    "Install with: pip install psycopg2-binary or pip install psycopg"
                ),
            ):
                with pytest.raises(RetrieverError) as exc:
                    retriever._get_connection()
                assert "psycopg2 or psycopg package required" in str(exc.value)


class TestPgVectorRetrieverIndex:
    """Test PgVectorRetriever index method."""

    def test_index_empty_document(self):
        embedder = MockEmbedder()
        retriever = PgVectorRetriever(embedder)

        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_conn.cursor.return_value.__enter__ = MagicMock(return_value=mock_cursor)
        mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)

        with patch.object(retriever, "_get_connection", return_value=mock_conn):
            with patch.object(retriever, "_ensure_table"):
                doc = MockDocument("doc1", [])
                retriever.index(doc)

        assert retriever.document_id == "doc1"
        assert retriever.chunk_count == 0

    def test_index_with_chunks(self):
        embedder = MockEmbedder()
        retriever = PgVectorRetriever(embedder)

        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_conn.cursor.return_value.__enter__ = MagicMock(return_value=mock_cursor)
        mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)

        with patch.object(retriever, "_get_connection", return_value=mock_conn):
            with patch.object(retriever, "_ensure_table"):
                chunks = [
                    MockChunk(0, "Hello world", page=1),
                    MockChunk(1, "Test chunk", page=2),
                ]
                doc = MockDocument("doc1", chunks)
                retriever.index(doc)

        assert retriever.is_indexed
        assert retriever.chunk_count == 2
        assert retriever.document_id == "doc1"

        # Verify execute was called for each chunk
        assert mock_cursor.execute.call_count == 2


class TestPgVectorRetrieverSearch:
    """Test PgVectorRetriever search method."""

    def test_search_not_indexed_raises(self):
        embedder = MockEmbedder()
        retriever = PgVectorRetriever(embedder)

        with pytest.raises(RetrieverError) as exc:
            retriever.search("test query")
        assert "No document indexed" in str(exc.value)

    def test_search_k_less_than_1_raises(self):
        embedder = MockEmbedder()
        retriever = PgVectorRetriever(embedder)

        # Manually set indexed state
        retriever._chunks = [MockChunk(0, "test")]

        with pytest.raises(RetrieverError) as exc:
            retriever.search("test", k=0)
        assert "k must be >= 1" in str(exc.value)

    def test_search_basic(self):
        embedder = MockEmbedder()
        retriever = PgVectorRetriever(embedder)

        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_cursor.fetchall.return_value = [
            (0, 0.9),  # chunk_index, score
            (1, 0.7),
        ]
        mock_conn.cursor.return_value.__enter__ = MagicMock(return_value=mock_cursor)
        mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)

        chunks = [
            MockChunk(0, "Hello world"),
            MockChunk(1, "Test chunk"),
        ]
        retriever._chunks = chunks
        retriever._document_id = "doc1"

        with patch.object(retriever, "_get_connection", return_value=mock_conn):
            with patch.object(retriever, "_ensure_table"):
                results = retriever.search("hello", k=2)

        assert len(results) == 2
        assert results.query == "hello"
        assert results.total_chunks == 2
        assert results[0].score == pytest.approx(0.9)
        assert results[1].score == pytest.approx(0.7)


class TestPgVectorRetrieverClear:
    """Test PgVectorRetriever clear method."""

    def test_clear_resets_state(self):
        embedder = MockEmbedder()
        retriever = PgVectorRetriever(embedder)

        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_conn.cursor.return_value.__enter__ = MagicMock(return_value=mock_cursor)
        mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)

        # Set some state
        retriever._chunks = [MockChunk(0, "test")]
        retriever._document_id = "doc1"
        retriever._connection = mock_conn

        retriever.clear()

        assert retriever._chunks == []
        assert retriever._document_id is None


class TestPgVectorRetrieverClose:
    """Test PgVectorRetriever close method."""

    def test_close_connection(self):
        embedder = MockEmbedder()
        retriever = PgVectorRetriever(embedder)

        mock_conn = MagicMock()
        retriever._connection = mock_conn

        retriever.close()

        mock_conn.close.assert_called_once()
        assert retriever._connection is None


class TestPgVectorRetrieverDistanceOperators:
    """Test distance operator selection."""

    def test_cosine_operator(self):
        embedder = MockEmbedder()
        retriever = PgVectorRetriever(embedder, distance_metric="cosine")

        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_cursor.fetchall.return_value = [(0, 0.9)]
        mock_conn.cursor.return_value.__enter__ = MagicMock(return_value=mock_cursor)
        mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)

        retriever._chunks = [MockChunk(0, "test")]
        retriever._document_id = "doc1"

        with patch.object(retriever, "_get_connection", return_value=mock_conn):
            with patch.object(retriever, "_ensure_table"):
                retriever.search("test", k=1)

        # Verify the cosine operator is used
        call_args = mock_cursor.execute.call_args[0][0]
        assert "<=>" in call_args

    def test_l2_operator(self):
        embedder = MockEmbedder()
        retriever = PgVectorRetriever(embedder, distance_metric="l2")

        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_cursor.fetchall.return_value = [(0, 0.5)]
        mock_conn.cursor.return_value.__enter__ = MagicMock(return_value=mock_cursor)
        mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)

        retriever._chunks = [MockChunk(0, "test")]
        retriever._document_id = "doc1"

        with patch.object(retriever, "_get_connection", return_value=mock_conn):
            with patch.object(retriever, "_ensure_table"):
                retriever.search("test", k=1)

        call_args = mock_cursor.execute.call_args[0][0]
        assert "<->" in call_args

    def test_ip_operator(self):
        embedder = MockEmbedder()
        retriever = PgVectorRetriever(embedder, distance_metric="ip")

        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_cursor.fetchall.return_value = [(0, -0.8)]
        mock_conn.cursor.return_value.__enter__ = MagicMock(return_value=mock_cursor)
        mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)

        retriever._chunks = [MockChunk(0, "test")]
        retriever._document_id = "doc1"

        with patch.object(retriever, "_get_connection", return_value=mock_conn):
            with patch.object(retriever, "_ensure_table"):
                retriever.search("test", k=1)

        call_args = mock_cursor.execute.call_args[0][0]
        assert "<#>" in call_args
