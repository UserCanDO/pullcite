"""
Tests for pipeline/strategy.py - ExtractionStrategy and DefaultStrategy.
"""

import pytest
from pydantic import BaseModel
from pullcite.pipeline.strategy import (
    ExtractionStrategy,
    DefaultStrategy,
    StrategyContext,
)
from pullcite.core.document import Document
from pullcite.core.fields import CriticalField
from pullcite.core.evidence import VerificationResult, VerificationStatus


class SampleSchema(BaseModel):
    """Sample schema for testing."""

    name: str
    amount: float
    active: bool = True


@pytest.fixture
def sample_document():
    return Document.from_text("This is a test document with some content.")


@pytest.fixture
def sample_fields():
    return [
        CriticalField(path="name", label="Name", search_query="name"),
        CriticalField(path="amount", label="Amount", search_query="amount"),
    ]


@pytest.fixture
def sample_context(sample_document, sample_fields):
    return StrategyContext(
        document=sample_document,
        schema=SampleSchema,
        critical_fields=sample_fields,
    )


class TestStrategyContext:
    """Test StrategyContext dataclass."""

    def test_basic_creation(self, sample_document, sample_fields):
        ctx = StrategyContext(
            document=sample_document,
            schema=SampleSchema,
            critical_fields=sample_fields,
        )
        assert ctx.document is sample_document
        assert ctx.schema is SampleSchema
        assert len(ctx.critical_fields) == 2
        assert ctx.extracted_data is None
        assert ctx.verification_results is None

    def test_with_extracted_data(self, sample_document, sample_fields):
        ctx = StrategyContext(
            document=sample_document,
            schema=SampleSchema,
            critical_fields=sample_fields,
            extracted_data={"name": "Test", "amount": 100.0},
        )
        assert ctx.extracted_data == {"name": "Test", "amount": 100.0}

    def test_with_verification_results(self, sample_document, sample_fields):
        vr = VerificationResult(
            path="name",
            status=VerificationStatus.MATCH,
            extracted_value="Test",
        )
        ctx = StrategyContext(
            document=sample_document,
            schema=SampleSchema,
            critical_fields=sample_fields,
            verification_results=[vr],
        )
        assert len(ctx.verification_results) == 1


class TestExtractionStrategyABC:
    """Test ExtractionStrategy abstract base class."""

    def test_cannot_instantiate_directly(self):
        with pytest.raises(TypeError):
            ExtractionStrategy()

    def test_concrete_implementation(self, sample_context):
        class SimpleStrategy(ExtractionStrategy):
            def build_extractor_prompt(self, context):
                return "Extract data"

            def build_verifier_prompt(self, context):
                return "Verify data"

            def build_corrector_prompt(self, context):
                return "Correct data"

        strategy = SimpleStrategy()
        assert strategy.build_extractor_prompt(sample_context) == "Extract data"
        assert strategy.build_verifier_prompt(sample_context) == "Verify data"
        assert strategy.build_corrector_prompt(sample_context) == "Correct data"


class TestDefaultStrategy:
    """Test DefaultStrategy implementation."""

    def test_extractor_prompt(self, sample_context):
        strategy = DefaultStrategy()
        prompt = strategy.build_extractor_prompt(sample_context)

        assert "SampleSchema" in prompt
        assert "extract" in prompt.lower()
        assert "JSON" in prompt

    def test_verifier_prompt(self, sample_context):
        strategy = DefaultStrategy()
        prompt = strategy.build_verifier_prompt(sample_context)

        assert "verify" in prompt.lower()
        assert "search" in prompt.lower()
        assert "MATCH" in prompt or "MISMATCH" in prompt

    def test_corrector_prompt(self, sample_context):
        strategy = DefaultStrategy()
        prompt = strategy.build_corrector_prompt(sample_context)

        assert "correct" in prompt.lower()
        assert "JSON" in prompt

    def test_extraction_user_message(self, sample_context):
        strategy = DefaultStrategy()
        msg = strategy.build_extraction_user_message(sample_context)

        assert "extract" in msg.lower()
        assert sample_context.document.full_text in msg

    def test_verification_user_message(self, sample_context, sample_fields):
        strategy = DefaultStrategy()

        # Add extracted data to context
        ctx = StrategyContext(
            document=sample_context.document,
            schema=sample_context.schema,
            critical_fields=sample_context.critical_fields,
            extracted_data={"name": "Test Name", "amount": 99.99},
        )

        msg = strategy.build_verification_user_message(ctx, sample_fields)

        assert "verify" in msg.lower()
        assert "Name" in msg  # Field label
        assert "name" in msg  # Field path
        assert "Test Name" in msg  # Extracted value

    def test_correction_user_message(self, sample_context):
        strategy = DefaultStrategy()

        # Create context with failed verification
        vr = VerificationResult(
            path="amount",
            status=VerificationStatus.MISMATCH,
            extracted_value=100.0,
            found_value=200.0,
        )
        ctx = StrategyContext(
            document=sample_context.document,
            schema=sample_context.schema,
            critical_fields=sample_context.critical_fields,
            verification_results=[vr],
        )

        msg = strategy.build_correction_user_message(ctx)

        assert "amount" in msg
        assert "100" in msg  # Extracted
        assert "200" in msg  # Found

    def test_schema_description(self, sample_context):
        strategy = DefaultStrategy()
        prompt = strategy.build_extractor_prompt(sample_context)

        # Should include field names and types
        assert "name" in prompt
        assert "amount" in prompt
        assert "string" in prompt or "str" in prompt.lower()
        assert "float" in prompt or "number" in prompt


class TestStrategyCustomization:
    """Test that strategies can be customized."""

    def test_override_user_message(self, sample_context):
        class CustomStrategy(DefaultStrategy):
            def build_extraction_user_message(self, context):
                return "CUSTOM MESSAGE"

        strategy = CustomStrategy()
        msg = strategy.build_extraction_user_message(sample_context)
        assert msg == "CUSTOM MESSAGE"

    def test_fully_custom_strategy(self, sample_context):
        class DomainStrategy(ExtractionStrategy):
            def build_extractor_prompt(self, context):
                return f"Extract from {context.schema.__name__}"

            def build_verifier_prompt(self, context):
                return f"Verify {len(context.critical_fields)} fields"

            def build_corrector_prompt(self, context):
                return "Apply corrections"

        strategy = DomainStrategy()
        assert "SampleSchema" in strategy.build_extractor_prompt(sample_context)
        assert "2 fields" in strategy.build_verifier_prompt(sample_context)


class TestDefaultStrategyCustomPrompts:
    """Test DefaultStrategy custom prompt fields."""

    def test_custom_extractor_prompt_overrides_default(self, sample_context):
        custom_prompt = "You are an SBC expert. Extract health insurance data."
        strategy = DefaultStrategy(extractor_prompt=custom_prompt)

        prompt = strategy.build_extractor_prompt(sample_context)

        assert prompt == custom_prompt
        assert "SampleSchema" not in prompt  # Default content not present

    def test_custom_verifier_prompt_overrides_default(self, sample_context):
        custom_prompt = "You are a verification specialist."
        strategy = DefaultStrategy(verifier_prompt=custom_prompt)

        prompt = strategy.build_verifier_prompt(sample_context)

        assert prompt == custom_prompt
        assert "search tool" not in prompt.lower()  # Default content not present

    def test_custom_corrector_prompt_overrides_default(self, sample_context):
        custom_prompt = "Fix the errors you find."
        strategy = DefaultStrategy(corrector_prompt=custom_prompt)

        prompt = strategy.build_corrector_prompt(sample_context)

        assert prompt == custom_prompt
        assert "verification result" not in prompt.lower()  # Default content not present

    def test_extra_instructions_appends_to_extractor(self, sample_context):
        instructions = "Focus on deductibles. This is a PPO plan."
        strategy = DefaultStrategy(extra_instructions=instructions)

        prompt = strategy.build_extractor_prompt(sample_context)

        # Default content still present
        assert "SampleSchema" in prompt
        assert "extract" in prompt.lower()
        # Extra instructions appended
        assert "ADDITIONAL INSTRUCTIONS" in prompt
        assert instructions in prompt

    def test_extra_instructions_appends_to_verifier(self, sample_context):
        instructions = "Pay special attention to copay amounts."
        strategy = DefaultStrategy(extra_instructions=instructions)

        prompt = strategy.build_verifier_prompt(sample_context)

        # Default content still present
        assert "search tool" in prompt.lower()
        assert "verify" in prompt.lower()
        # Extra instructions appended
        assert "ADDITIONAL INSTRUCTIONS" in prompt
        assert instructions in prompt

    def test_extra_instructions_appends_to_corrector(self, sample_context):
        instructions = "Use the found_value when available."
        strategy = DefaultStrategy(extra_instructions=instructions)

        prompt = strategy.build_corrector_prompt(sample_context)

        # Default content still present
        assert "correction" in prompt.lower()
        assert "JSON" in prompt
        # Extra instructions appended
        assert "ADDITIONAL INSTRUCTIONS" in prompt
        assert instructions in prompt

    def test_none_values_use_default_behavior(self, sample_context):
        strategy = DefaultStrategy(
            extractor_prompt=None,
            verifier_prompt=None,
            corrector_prompt=None,
            extra_instructions=None,
        )

        extractor = strategy.build_extractor_prompt(sample_context)
        verifier = strategy.build_verifier_prompt(sample_context)
        corrector = strategy.build_corrector_prompt(sample_context)

        # All should use defaults
        assert "SampleSchema" in extractor
        assert "search tool" in verifier.lower()
        assert "correction" in corrector.lower()
        # No additional instructions section
        assert "ADDITIONAL INSTRUCTIONS" not in extractor
        assert "ADDITIONAL INSTRUCTIONS" not in verifier
        assert "ADDITIONAL INSTRUCTIONS" not in corrector

    def test_custom_prompt_ignores_extra_instructions(self, sample_context):
        custom_prompt = "My custom prompt."
        instructions = "These should be ignored."
        strategy = DefaultStrategy(
            extractor_prompt=custom_prompt,
            extra_instructions=instructions,
        )

        prompt = strategy.build_extractor_prompt(sample_context)

        # Custom prompt returned as-is
        assert prompt == custom_prompt
        # Extra instructions not appended (custom prompt takes precedence)
        assert instructions not in prompt

    def test_dataclass_defaults(self):
        strategy = DefaultStrategy()

        assert strategy.extractor_prompt is None
        assert strategy.verifier_prompt is None
        assert strategy.corrector_prompt is None
        assert strategy.extra_instructions is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
