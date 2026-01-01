"""
Tests for embeddings/voyage.py - Voyage AI embeddings implementation.
"""

from unittest.mock import MagicMock, patch

import pytest
from pullcite.embeddings.base import EmbeddingError
from pullcite.embeddings.voyage import VoyageEmbedder, VOYAGE_MODELS


class TestVoyageModels:
    """Test model configurations."""

    def test_supported_models(self):
        assert "voyage-3" in VOYAGE_MODELS
        assert "voyage-3-lite" in VOYAGE_MODELS
        assert "voyage-code-3" in VOYAGE_MODELS
        assert "voyage-finance-2" in VOYAGE_MODELS
        assert "voyage-law-2" in VOYAGE_MODELS

    def test_model_dimensions(self):
        assert VOYAGE_MODELS["voyage-3"]["dimensions"] == 1024
        assert VOYAGE_MODELS["voyage-3-lite"]["dimensions"] == 512


class TestVoyageEmbedderInit:
    """Test VoyageEmbedder initialization."""

    def test_requires_api_key(self):
        with patch.dict("os.environ", {}, clear=True):
            with pytest.raises(EmbeddingError) as exc:
                VoyageEmbedder()
            assert "API key required" in str(exc.value)

    def test_api_key_from_env(self):
        with patch.dict("os.environ", {"VOYAGE_API_KEY": "test-key"}):
            embedder = VoyageEmbedder()
            assert embedder.api_key == "test-key"

    def test_api_key_from_parameter(self):
        embedder = VoyageEmbedder(api_key="explicit-key")
        assert embedder.api_key == "explicit-key"

    def test_unknown_model_raises(self):
        with pytest.raises(EmbeddingError) as exc:
            VoyageEmbedder(api_key="key", model="unknown-model")
        assert "Unknown model" in str(exc.value)

    def test_invalid_input_type_raises(self):
        with pytest.raises(EmbeddingError) as exc:
            VoyageEmbedder(api_key="key", input_type="invalid")
        assert "Invalid input_type" in str(exc.value)

    def test_valid_input_types(self):
        # Should not raise
        VoyageEmbedder(api_key="key", input_type="query")
        VoyageEmbedder(api_key="key", input_type="document")
        VoyageEmbedder(api_key="key", input_type=None)

    def test_default_model(self):
        embedder = VoyageEmbedder(api_key="key")
        assert embedder.model == "voyage-3"

    def test_model_name_property(self):
        embedder = VoyageEmbedder(api_key="key", model="voyage-3-lite")
        assert embedder.model_name == "voyage-3-lite"

    def test_dimensions_property(self):
        embedder = VoyageEmbedder(api_key="key", model="voyage-3")
        assert embedder.dimensions == 1024

        embedder2 = VoyageEmbedder(api_key="key", model="voyage-3-lite")
        assert embedder2.dimensions == 512


class TestVoyageEmbed:
    """Test the embed method."""

    def test_embed_empty_text_raises(self):
        embedder = VoyageEmbedder(api_key="key")
        with pytest.raises(EmbeddingError) as exc:
            embedder.embed("")
        assert "Cannot embed empty text" in str(exc.value)

    def test_embed_whitespace_text_raises(self):
        embedder = VoyageEmbedder(api_key="key")
        with pytest.raises(EmbeddingError) as exc:
            embedder.embed("   ")
        assert "Cannot embed empty text" in str(exc.value)

    def test_embed_basic(self):
        embedder = VoyageEmbedder(api_key="test-key")

        # Mock the client
        mock_client = MagicMock()
        mock_result = MagicMock()
        mock_result.embeddings = [[0.1, 0.2, 0.3] * 341 + [0.4]]  # 1024 dims
        mock_result.total_tokens = 5

        mock_client.embed.return_value = mock_result

        with patch.object(embedder, "_get_client", return_value=mock_client):
            result = embedder.embed("Hello world")

        assert result.model == "voyage-3"
        assert result.dimensions == 1024
        assert len(result.vector) == 1024
        assert result.token_count == 5

        mock_client.embed.assert_called_once()
        call_kwargs = mock_client.embed.call_args[1]
        assert call_kwargs["texts"] == ["Hello world"]
        assert call_kwargs["model"] == "voyage-3"

    def test_embed_with_input_type(self):
        embedder = VoyageEmbedder(api_key="test-key", input_type="query")

        mock_client = MagicMock()
        mock_result = MagicMock()
        mock_result.embeddings = [[0.1] * 1024]
        mock_result.total_tokens = 3

        mock_client.embed.return_value = mock_result

        with patch.object(embedder, "_get_client", return_value=mock_client):
            embedder.embed("test query")

        call_kwargs = mock_client.embed.call_args[1]
        assert call_kwargs["input_type"] == "query"


class TestVoyageEmbedBatch:
    """Test the embed_batch method."""

    def test_embed_batch_empty(self):
        embedder = VoyageEmbedder(api_key="key")
        result = embedder.embed_batch([])
        assert result.vectors == ()
        assert result.total_tokens == 0

    def test_embed_batch_with_empty_text_raises(self):
        embedder = VoyageEmbedder(api_key="key")
        with pytest.raises(EmbeddingError) as exc:
            embedder.embed_batch(["hello", "", "world"])
        assert "Cannot embed empty text at index 1" in str(exc.value)

    def test_embed_batch_basic(self):
        embedder = VoyageEmbedder(api_key="test-key")

        mock_client = MagicMock()
        mock_result = MagicMock()
        mock_result.embeddings = [[0.1] * 1024, [0.2] * 1024]
        mock_result.total_tokens = 10

        mock_client.embed.return_value = mock_result

        with patch.object(embedder, "_get_client", return_value=mock_client):
            result = embedder.embed_batch(["hello", "world"])

        assert len(result.vectors) == 2
        assert result.total_tokens == 10


class TestVoyageConvenienceMethods:
    """Test convenience methods for query/document embedding."""

    def test_embed_query(self):
        embedder = VoyageEmbedder(api_key="test-key")

        mock_client = MagicMock()
        mock_result = MagicMock()
        mock_result.embeddings = [[0.1] * 1024]
        mock_result.total_tokens = 3

        mock_client.embed.return_value = mock_result

        with patch.object(embedder, "_get_client", return_value=mock_client):
            result = embedder.embed_query("test query")

        call_kwargs = mock_client.embed.call_args[1]
        assert call_kwargs["input_type"] == "query"
        assert result.dimensions == 1024

    def test_embed_documents(self):
        embedder = VoyageEmbedder(api_key="test-key")

        mock_client = MagicMock()
        mock_result = MagicMock()
        mock_result.embeddings = [[0.1] * 1024, [0.2] * 1024]
        mock_result.total_tokens = 15

        mock_client.embed.return_value = mock_result

        with patch.object(embedder, "_get_client", return_value=mock_client):
            result = embedder.embed_documents(["doc1", "doc2"])

        call_kwargs = mock_client.embed.call_args[1]
        assert call_kwargs["input_type"] == "document"
        assert len(result.vectors) == 2
