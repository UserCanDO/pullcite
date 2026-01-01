"""
Tests for embeddings/base.py - Embedder ABC and result types.
"""

import pytest
from pullcite.embeddings.base import (
    Embedder,
    EmbeddingResult,
    BatchEmbeddingResult,
    EmbeddingError,
)


class TestEmbeddingResult:
    """Test EmbeddingResult dataclass."""

    def test_basic_creation(self):
        result = EmbeddingResult(
            vector=(0.1, 0.2, 0.3),
            model="test-model",
            dimensions=3,
            token_count=5,
        )
        assert result.vector == (0.1, 0.2, 0.3)
        assert result.model == "test-model"
        assert result.dimensions == 3
        assert result.token_count == 5

    def test_dimension_validation(self):
        with pytest.raises(ValueError) as exc:
            EmbeddingResult(
                vector=(0.1, 0.2),
                model="test",
                dimensions=3,  # Mismatch!
                token_count=1,
            )
        assert "doesn't match dimensions" in str(exc.value)

    def test_immutability(self):
        result = EmbeddingResult(
            vector=(0.1, 0.2),
            model="test",
            dimensions=2,
            token_count=1,
        )
        with pytest.raises(AttributeError):
            result.model = "other"


class TestBatchEmbeddingResult:
    """Test BatchEmbeddingResult dataclass."""

    def test_basic_creation(self):
        result = BatchEmbeddingResult(
            vectors=((0.1, 0.2), (0.3, 0.4)),
            model="test-model",
            dimensions=2,
            total_tokens=10,
        )
        assert len(result.vectors) == 2
        assert result.vectors[0] == (0.1, 0.2)
        assert result.total_tokens == 10

    def test_empty_batch(self):
        result = BatchEmbeddingResult(
            vectors=(),
            model="test",
            dimensions=2,
            total_tokens=0,
        )
        assert len(result.vectors) == 0


class TestEmbeddingError:
    """Test EmbeddingError exception."""

    def test_basic_error(self):
        error = EmbeddingError("Something went wrong")
        assert str(error) == "Something went wrong"
        assert error.cause is None

    def test_error_with_cause(self):
        cause = ValueError("Original error")
        error = EmbeddingError("Wrapper message", cause=cause)
        assert error.cause is cause


class TestEmbedderABC:
    """Test Embedder abstract base class."""

    def test_cannot_instantiate_directly(self):
        with pytest.raises(TypeError):
            Embedder()

    def test_concrete_implementation(self):
        """Test that a concrete implementation works."""

        class MockEmbedder(Embedder):
            @property
            def model_name(self) -> str:
                return "mock-model"

            @property
            def dimensions(self) -> int:
                return 4

            def embed(self, text: str) -> EmbeddingResult:
                # Simple mock: use text length to generate vector
                vec = tuple(float(i) / 10 for i in range(self.dimensions))
                return EmbeddingResult(
                    vector=vec,
                    model=self.model_name,
                    dimensions=self.dimensions,
                    token_count=len(text.split()),
                )

            def embed_batch(self, texts: list[str]) -> BatchEmbeddingResult:
                vectors = tuple(
                    tuple(float(i) / 10 for i in range(self.dimensions)) for _ in texts
                )
                return BatchEmbeddingResult(
                    vectors=vectors,
                    model=self.model_name,
                    dimensions=self.dimensions,
                    total_tokens=sum(len(t.split()) for t in texts),
                )

        embedder = MockEmbedder()
        assert embedder.model_name == "mock-model"
        assert embedder.dimensions == 4

        result = embedder.embed("hello world")
        assert len(result.vector) == 4
        assert result.token_count == 2

    def test_embed_many_batching(self):
        """Test embed_many handles batching."""

        class CountingEmbedder(Embedder):
            def __init__(self):
                self.batch_calls = 0

            @property
            def model_name(self) -> str:
                return "counter"

            @property
            def dimensions(self) -> int:
                return 2

            def embed(self, text: str) -> EmbeddingResult:
                return EmbeddingResult(
                    vector=(0.0, 0.0),
                    model=self.model_name,
                    dimensions=2,
                    token_count=1,
                )

            def embed_batch(self, texts: list[str]) -> BatchEmbeddingResult:
                self.batch_calls += 1
                vectors = tuple((0.0, 0.0) for _ in texts)
                return BatchEmbeddingResult(
                    vectors=vectors,
                    model=self.model_name,
                    dimensions=2,
                    total_tokens=len(texts),
                )

        embedder = CountingEmbedder()

        # 5 texts with batch_size=2 should make 3 calls
        texts = ["a", "b", "c", "d", "e"]
        vectors = embedder.embed_many(texts, batch_size=2)

        assert len(vectors) == 5
        assert embedder.batch_calls == 3  # ceil(5/2) = 3

    def test_embed_many_empty(self):
        """Test embed_many with empty list."""

        class SimpleEmbedder(Embedder):
            @property
            def model_name(self) -> str:
                return "simple"

            @property
            def dimensions(self) -> int:
                return 2

            def embed(self, text: str) -> EmbeddingResult:
                return EmbeddingResult(
                    vector=(0.0, 0.0),
                    model="simple",
                    dimensions=2,
                    token_count=1,
                )

            def embed_batch(self, texts: list[str]) -> BatchEmbeddingResult:
                raise AssertionError("Should not be called for empty list")

        embedder = SimpleEmbedder()
        result = embedder.embed_many([])
        assert result == []
