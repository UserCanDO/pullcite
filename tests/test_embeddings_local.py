"""
Tests for embeddings/local.py - Local embeddings using Sentence Transformers.
"""

from unittest.mock import MagicMock, patch
import numpy as np

import pytest
from pullcite.embeddings.base import EmbeddingError
from pullcite.embeddings.local import LocalEmbedder, LOCAL_MODELS


class TestLocalModels:
    """Test model configurations."""

    def test_supported_models(self):
        assert "all-MiniLM-L6-v2" in LOCAL_MODELS
        assert "all-mpnet-base-v2" in LOCAL_MODELS
        assert "multi-qa-MiniLM-L6-cos-v1" in LOCAL_MODELS

    def test_model_dimensions(self):
        assert LOCAL_MODELS["all-MiniLM-L6-v2"]["dimensions"] == 384
        assert LOCAL_MODELS["all-mpnet-base-v2"]["dimensions"] == 768


class TestLocalEmbedderInit:
    """Test LocalEmbedder initialization."""

    def test_default_model(self):
        embedder = LocalEmbedder()
        assert embedder.model == "all-MiniLM-L6-v2"

    def test_model_name_property(self):
        embedder = LocalEmbedder(model="all-mpnet-base-v2")
        assert embedder.model_name == "all-mpnet-base-v2"

    def test_dimensions_for_known_model(self):
        embedder = LocalEmbedder(model="all-MiniLM-L6-v2")
        assert embedder.dimensions == 384

    def test_custom_device(self):
        embedder = LocalEmbedder(device="cpu")
        assert embedder.device == "cpu"

    def test_normalize_default(self):
        embedder = LocalEmbedder()
        assert embedder.normalize is True


class TestLocalEmbedderEmbed:
    """Test the embed method."""

    def test_embed_empty_text_raises(self):
        embedder = LocalEmbedder()

        # Mock the model loading
        with patch.object(embedder, "_load_model"):
            with pytest.raises(EmbeddingError) as exc:
                embedder.embed("")
            assert "Cannot embed empty text" in str(exc.value)

    def test_embed_whitespace_text_raises(self):
        embedder = LocalEmbedder()

        with patch.object(embedder, "_load_model"):
            with pytest.raises(EmbeddingError) as exc:
                embedder.embed("   ")
            assert "Cannot embed empty text" in str(exc.value)

    def test_embed_basic(self):
        embedder = LocalEmbedder(model="all-MiniLM-L6-v2")

        # Mock the model
        mock_model = MagicMock()
        mock_model.encode.return_value = np.array([0.1] * 384)
        mock_model.get_sentence_embedding_dimension.return_value = 384

        with patch.object(embedder, "_load_model", return_value=mock_model):
            result = embedder.embed("Hello world")

        assert result.model == "all-MiniLM-L6-v2"
        assert result.dimensions == 384
        assert len(result.vector) == 384
        assert result.token_count == 2  # Rough word count

        mock_model.encode.assert_called_once_with(
            "Hello world",
            normalize_embeddings=True,
            convert_to_numpy=True,
        )


class TestLocalEmbedderEmbedBatch:
    """Test the embed_batch method."""

    def test_embed_batch_empty(self):
        embedder = LocalEmbedder()
        result = embedder.embed_batch([])
        assert result.vectors == ()
        assert result.total_tokens == 0

    def test_embed_batch_with_empty_text_raises(self):
        embedder = LocalEmbedder()

        with patch.object(embedder, "_load_model"):
            with pytest.raises(EmbeddingError) as exc:
                embedder.embed_batch(["hello", "", "world"])
            assert "Cannot embed empty text at index 1" in str(exc.value)

    def test_embed_batch_basic(self):
        embedder = LocalEmbedder(model="all-MiniLM-L6-v2")

        mock_model = MagicMock()
        mock_model.encode.return_value = np.array(
            [
                [0.1] * 384,
                [0.2] * 384,
            ]
        )
        mock_model.get_sentence_embedding_dimension.return_value = 384

        with patch.object(embedder, "_load_model", return_value=mock_model):
            result = embedder.embed_batch(["hello", "world"])

        assert len(result.vectors) == 2
        assert len(result.vectors[0]) == 384
        assert result.total_tokens == 2  # 1 word each

        mock_model.encode.assert_called_once_with(
            ["hello", "world"],
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )


class TestLocalEmbedderModelLoading:
    """Test model loading behavior."""

    def test_lazy_model_loading(self):
        """Model should not be loaded until needed."""
        embedder = LocalEmbedder()
        assert embedder._model_instance is None

    def test_model_loading_error(self):
        """Should raise EmbeddingError on model load failure."""
        embedder = LocalEmbedder(model="nonexistent-model-xyz")

        # Mock the load method to simulate failure
        with patch.object(
            embedder,
            "_load_model",
            side_effect=EmbeddingError(
                "Failed to load model 'nonexistent-model-xyz': Model not found"
            ),
        ):
            with pytest.raises(EmbeddingError):
                embedder.embed("test")

    def test_import_error(self):
        """Should raise helpful error if sentence-transformers not installed."""
        embedder = LocalEmbedder()

        with patch.dict("sys.modules", {"sentence_transformers": None}):
            with patch.object(
                embedder,
                "_load_model",
                side_effect=EmbeddingError(
                    "sentence-transformers package required. "
                    "Install with: pip install sentence-transformers"
                ),
            ):
                with pytest.raises(EmbeddingError) as exc:
                    embedder.embed("test")
                assert "sentence-transformers package required" in str(exc.value)
