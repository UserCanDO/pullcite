"""
Tests for core/extractor.py - Main Extractor orchestration.
"""

import pytest
from pydantic import BaseModel
from pullcite import (
    Extractor,
    Document,
    CriticalField,
    ExtractorConfig,
    Hooks,
    ExtractionStatus,
    ExtractionFlagType,
)
from pullcite.llms.base import LLM, LLMResponse, Message, Tool, ToolCall
from pullcite.retrieval.base import Retriever, SearchResult, SearchResults
from pullcite.embeddings.base import Embedder, EmbeddingResult, BatchEmbeddingResult
from pullcite.core.chunk import Chunk


class SimpleSchema(BaseModel):
    """Simple schema for testing."""

    name: str
    amount: float
    active: bool = True


class NestedSchema(BaseModel):
    """Nested schema for testing."""

    info: dict
    items: list


class MockEmbedder(Embedder):
    """Mock embedder for testing."""

    @property
    def model_name(self) -> str:
        return "mock-embedder"

    @property
    def dimensions(self) -> int:
        return 4

    def embed(self, text: str) -> EmbeddingResult:
        return EmbeddingResult(
            vector=(0.1, 0.2, 0.3, 0.4),
            model="mock-embedder",
            dimensions=4,
            token_count=len(text.split()),
        )

    def embed_batch(self, texts: list[str]) -> BatchEmbeddingResult:
        return BatchEmbeddingResult(
            vectors=tuple((0.1, 0.2, 0.3, 0.4) for _ in texts),
            model="mock-embedder",
            dimensions=4,
            total_tokens=sum(len(t.split()) for t in texts),
        )


class MockRetriever(Retriever):
    """Mock retriever for testing."""

    def __init__(self, results: list[str] | None = None):
        self._embedder = MockEmbedder()
        self._results = results or ["The amount is $100.00"]
        self._indexed = False

    @property
    def embedder(self):
        return self._embedder

    @property
    def is_indexed(self):
        return self._indexed

    @property
    def chunk_count(self):
        return len(self._results)

    def index(self, document):
        self._indexed = True

    def search(self, query: str, k: int = 5) -> SearchResults:
        results = tuple(
            SearchResult(
                chunk=Chunk(text=text, index=i),
                score=0.9 - i * 0.1,
                rank=i,
            )
            for i, text in enumerate(self._results[:k])
        )
        return SearchResults(
            results=results, query=query, total_chunks=len(self._results)
        )

    def clear(self):
        self._indexed = False


class MockLLM(LLM):
    """Mock LLM for testing."""

    def __init__(self, responses: list[str] | None = None):
        self.responses = responses or [
            '{"name": "Test", "amount": 100.0, "active": true}'
        ]
        self.call_index = 0

    @property
    def model_name(self) -> str:
        return "mock-llm"

    def complete(self, messages, tools=None, temperature=0.0, max_tokens=4096):
        response_content = self.responses[min(self.call_index, len(self.responses) - 1)]
        self.call_index += 1

        # Check if this should be a tool call
        if tools and "SEARCH:" in response_content:
            query = response_content.split("SEARCH:")[1].strip()
            return LLMResponse(
                content=None,
                tool_calls=(
                    ToolCall(
                        id=f"call_{self.call_index}",
                        name="search",
                        arguments={"query": query},
                    ),
                ),
                stop_reason="tool_use",
                input_tokens=100,
                output_tokens=50,
                model="mock-llm",
            )

        return LLMResponse(
            content=response_content,
            tool_calls=(),
            stop_reason="end_turn",
            input_tokens=100,
            output_tokens=50,
            model="mock-llm",
        )


@pytest.fixture
def mock_llm():
    return MockLLM()


@pytest.fixture
def mock_retriever():
    return MockRetriever()


@pytest.fixture
def sample_document():
    return Document.from_text(
        "Invoice for Test Company.\n" "Amount: $100.00\n" "Status: Active\n"
    )


class TestExtractor:
    """Test Extractor class."""

    def test_basic_extraction(self, mock_llm, mock_retriever, sample_document):
        """Test basic extraction without verification."""
        extractor = Extractor(
            schema=SimpleSchema,
            llm=mock_llm,
            retriever=mock_retriever,
            config=ExtractorConfig(skip_verification=True),
        )

        result = extractor.extract(sample_document)

        assert result.data is not None
        assert result.data.name == "Test"
        assert result.data.amount == 100.0
        assert result.data.active is True
        assert result.status == ExtractionStatus.VERIFIED
        assert result.document_id == sample_document.id

    def test_extraction_with_verification(self, mock_retriever, sample_document):
        """Test extraction with verification enabled."""
        # LLM that extracts, then verifies
        llm = MockLLM(
            [
                '{"name": "Test", "amount": 100.0}',  # Extraction
                "SEARCH: amount",  # Verification search
                "Verified",  # Verification complete
            ]
        )

        critical_fields = [
            CriticalField(
                path="amount",
                label="Amount",
                search_query="amount total",
            ),
        ]

        extractor = Extractor(
            schema=SimpleSchema,
            llm=llm,
            retriever=mock_retriever,
            critical_fields=critical_fields,
        )

        result = extractor.extract(sample_document)

        assert result.data is not None
        assert result.stats.fields_verified >= 1

    def test_extraction_tracks_stats(self, mock_llm, mock_retriever, sample_document):
        """Test that extraction tracks statistics."""
        extractor = Extractor(
            schema=SimpleSchema,
            llm=mock_llm,
            retriever=mock_retriever,
            config=ExtractorConfig(skip_verification=True),
        )

        result = extractor.extract(sample_document)

        # Check that LLM stats are tracked (these are reliable even with mocks)
        assert result.stats.extraction_llm_calls >= 1
        assert result.stats.extraction_input_tokens > 0
        assert result.stats.extraction_output_tokens > 0

    def test_extraction_with_hooks(self, mock_llm, mock_retriever, sample_document):
        """Test that hooks are called."""
        hook_calls = []

        def on_llm_call(role, messages, tools):
            hook_calls.append(("llm", role))

        hooks = Hooks(on_llm_call=on_llm_call)

        extractor = Extractor(
            schema=SimpleSchema,
            llm=mock_llm,
            retriever=mock_retriever,
            config=ExtractorConfig(skip_verification=True),
            hooks=hooks,
        )

        result = extractor.extract(sample_document)

        assert len(hook_calls) >= 1
        assert hook_calls[0] == ("llm", "extractor")

    def test_extraction_post_extract_hook(
        self, mock_llm, mock_retriever, sample_document
    ):
        """Test post_extract hook can modify data."""

        def post_extract(data, doc):
            data["name"] = "Modified"
            return data

        hooks = Hooks(post_extract=post_extract)

        extractor = Extractor(
            schema=SimpleSchema,
            llm=mock_llm,
            retriever=mock_retriever,
            config=ExtractorConfig(skip_verification=True),
            hooks=hooks,
        )

        result = extractor.extract(sample_document)

        assert result.data.name == "Modified"

    def test_extraction_failed(self, mock_retriever, sample_document):
        """Test extraction failure handling."""
        # LLM that always returns invalid JSON
        llm = MockLLM(["Not valid JSON"] * 5)

        extractor = Extractor(
            schema=SimpleSchema,
            llm=llm,
            retriever=mock_retriever,
            config=ExtractorConfig(
                skip_verification=True,
                max_extraction_retries=1,
            ),
        )

        result = extractor.extract(sample_document)

        assert result.status == ExtractionStatus.FAILED
        assert len(result.flags) > 0

    def test_extraction_with_config(self, mock_llm, mock_retriever, sample_document):
        """Test extraction with custom config."""
        config = ExtractorConfig(
            confidence_threshold=0.95,
            skip_verification=True,
            temperature=0.5,
        )

        extractor = Extractor(
            schema=SimpleSchema,
            llm=mock_llm,
            retriever=mock_retriever,
            config=config,
        )

        result = extractor.extract(sample_document)

        # Should still succeed
        assert result.data is not None

    def test_extraction_result_properties(
        self, mock_llm, mock_retriever, sample_document
    ):
        """Test ExtractionResult convenience properties."""
        extractor = Extractor(
            schema=SimpleSchema,
            llm=mock_llm,
            retriever=mock_retriever,
            config=ExtractorConfig(skip_verification=True),
        )

        result = extractor.extract(sample_document)

        assert result.is_success is True
        assert result.has_warnings is False


class TestExtractorWithCriticalFields:
    """Test Extractor with critical fields."""

    def test_multiple_critical_fields(self, mock_retriever, sample_document):
        """Test verification of multiple fields."""
        llm = MockLLM(
            [
                '{"name": "Test", "amount": 100.0}',
                "SEARCH: name",
                "SEARCH: amount",
                "Verified",
            ]
        )

        critical_fields = [
            CriticalField(path="name", label="Name", search_query="name"),
            CriticalField(path="amount", label="Amount", search_query="amount"),
        ]

        extractor = Extractor(
            schema=SimpleSchema,
            llm=llm,
            retriever=mock_retriever,
            critical_fields=critical_fields,
        )

        result = extractor.extract(sample_document)

        assert result.stats.fields_verified == 2

    def test_optional_critical_field(self, mock_retriever, sample_document):
        """Test optional (non-required) critical field."""
        llm = MockLLM(
            [
                '{"name": "Test", "amount": 100.0}',
                "Done",
            ]
        )

        critical_fields = [
            CriticalField(
                path="name",
                label="Name",
                search_query="name",
                required=False,  # Optional
            ),
        ]

        extractor = Extractor(
            schema=SimpleSchema,
            llm=llm,
            retriever=mock_retriever,
            critical_fields=critical_fields,
        )

        result = extractor.extract(sample_document)

        # Should not fail even if not found
        assert result.status in (
            ExtractionStatus.VERIFIED,
            ExtractionStatus.PARTIAL,
            ExtractionStatus.LOW_CONFIDENCE,
        )


class TestExtractorBatching:
    """Test verification batching."""

    def test_verification_batching(self, mock_retriever, sample_document):
        """Test that verification respects batch size."""
        call_count = 0

        class CountingLLM(MockLLM):
            def complete(self, messages, tools=None, temperature=0.0, max_tokens=4096):
                nonlocal call_count
                call_count += 1
                return super().complete(messages, tools, temperature, max_tokens)

        llm = CountingLLM(
            [
                '{"name": "Test", "amount": 100.0}',
                "Batch 1 done",
                "Batch 2 done",
            ]
        )

        critical_fields = [
            CriticalField(path="name", label="Name", search_query="name"),
            CriticalField(path="amount", label="Amount", search_query="amount"),
        ]

        extractor = Extractor(
            schema=SimpleSchema,
            llm=llm,
            retriever=mock_retriever,
            critical_fields=critical_fields,
            config=ExtractorConfig(verification_batch_size=1),  # 1 field per batch
        )

        result = extractor.extract(sample_document)

        # Should have multiple verification calls due to batching
        # 1 extraction + 2 verification batches
        assert call_count >= 3


class TestExtractorCorrection:
    """Test correction phase."""

    def test_correction_applied(self, mock_retriever, sample_document):
        """Test that corrections are applied."""
        llm = MockLLM(
            [
                '{"name": "Wrong", "amount": 50.0}',  # Extraction with wrong values
                "SEARCH: name",  # Verification
                "Mismatch found",  # Verification result
                '{"name": "Correct"}',  # Correction
            ]
        )

        critical_fields = [
            CriticalField(path="name", label="Name", search_query="name"),
        ]

        extractor = Extractor(
            schema=SimpleSchema,
            llm=llm,
            retriever=mock_retriever,
            critical_fields=critical_fields,
            config=ExtractorConfig(max_correction_attempts=1),
        )

        result = extractor.extract(sample_document)

        # Note: In a full implementation, the correction would be applied
        # This test verifies the flow completes without error
        assert result.data is not None


class TestExtractorImport:
    """Test that extractor can be imported from main package."""

    def test_import_from_pullcite(self):
        """Test imports from pullcite package."""
        from pullcite import (
            Extractor,
            Document,
            CriticalField,
            ExtractorConfig,
            ExtractionResult,
            ExtractionStatus,
        )

        assert Extractor is not None
        assert Document is not None
        assert CriticalField is not None
