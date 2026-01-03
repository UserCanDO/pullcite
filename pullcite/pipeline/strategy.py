"""
Extraction strategy interface.

Strategies define how to build prompts for each role in the pipeline.
This is the main customization point for domain-specific extraction.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Type

if TYPE_CHECKING:
    from pydantic import BaseModel
    from ..core.document import Document
    from ..core.fields import CriticalField
    from ..core.evidence import VerificationResult


@dataclass(frozen=True)
class StrategyContext:
    """
    Context passed to strategy methods.

    Contains all information needed to build prompts.

    Attributes:
        document: Source document.
        schema: Pydantic model class for output.
        critical_fields: Fields requiring verification.
        extracted_data: Data from extraction (for verify/correct).
        verification_results: Results from verification (for correct).
    """

    document: "Document"
    schema: Type["BaseModel"]
    critical_fields: list["CriticalField"]
    extracted_data: dict[str, Any] | None = None
    verification_results: list["VerificationResult"] | None = None


class ExtractionStrategy(ABC):
    """
    Abstract base class for extraction strategies.

    Strategies build prompts for each role in the pipeline:
    - Extractor: Reads document, produces structured data
    - Verifier: Searches document, confirms values
    - Corrector: Fixes mismatched values

    Implement this to customize extraction for your domain.

    Example:
        class SBCStrategy(ExtractionStrategy):
            def build_extractor_prompt(self, ctx):
                return '''You are extracting health insurance data...'''
    """

    @abstractmethod
    def build_extractor_prompt(self, context: StrategyContext) -> str:
        """
        Build the system prompt for the Extractor role.

        The extractor reads the document and produces structured data
        matching the schema.

        Args:
            context: Strategy context with document and schema.

        Returns:
            System prompt string.
        """
        ...

    @abstractmethod
    def build_verifier_prompt(self, context: StrategyContext) -> str:
        """
        Build the system prompt for the Verifier role.

        The verifier searches the document to confirm extracted values
        are correct and finds supporting evidence.

        Args:
            context: Strategy context with document, schema, and extracted data.

        Returns:
            System prompt string.
        """
        ...

    @abstractmethod
    def build_corrector_prompt(self, context: StrategyContext) -> str:
        """
        Build the system prompt for the Corrector role.

        The corrector fixes values that failed verification.

        Args:
            context: Strategy context with verification results.

        Returns:
            System prompt string.
        """
        ...

    def build_extraction_user_message(self, context: StrategyContext) -> str:
        """
        Build the user message for extraction.

        Default includes document text. Override for custom formatting.

        Args:
            context: Strategy context.

        Returns:
            User message string.
        """
        return f"Please extract the required information from this document:\n\n{context.document.full_text}"

    def build_verification_user_message(
        self,
        context: StrategyContext,
        fields_to_verify: list["CriticalField"],
    ) -> str:
        """
        Build the user message for verification.

        Default lists fields to verify. Override for custom formatting.

        Args:
            context: Strategy context with extracted data.
            fields_to_verify: Fields to verify in this batch.

        Returns:
            User message string.
        """
        lines = ["Please verify these extracted values:\n"]

        for field in fields_to_verify:
            # Get the extracted value for this field
            from ..core.paths import get

            value = get(context.extracted_data, field.path)
            lines.append(f"- {field.label} ({field.path}): {value}")
            lines.append(f"  Search hint: {field.search_query}")

        return "\n".join(lines)

    def build_correction_user_message(self, context: StrategyContext) -> str:
        """
        Build the user message for correction.

        Default lists failed verifications. Override for custom formatting.

        Args:
            context: Strategy context with verification results.

        Returns:
            User message string.
        """
        from ..core.evidence import VerificationStatus

        lines = ["These values failed verification and need correction:\n"]

        for result in context.verification_results or []:
            if result.status in (
                VerificationStatus.MISMATCH,
                VerificationStatus.AMBIGUOUS,
            ):
                lines.append(f"- {result.path}")
                lines.append(f"  Extracted: {result.extracted_value}")
                if result.found_value is not None:
                    lines.append(f"  Found in document: {result.found_value}")
                if result.candidates:
                    lines.append(f"  Candidates found: {len(result.candidates)}")

        lines.append("\nPlease provide corrected values.")
        return "\n".join(lines)


@dataclass
class DefaultStrategy(ExtractionStrategy):
    """
    Default extraction strategy.

    Provides generic prompts that work for most documents.
    For better results, create a domain-specific strategy or provide
    custom prompts.

    Attributes:
        extractor_prompt: Custom extractor prompt (overrides default).
        verifier_prompt: Custom verifier prompt (overrides default).
        corrector_prompt: Custom corrector prompt (overrides default).
        extra_instructions: Additional instructions appended to default prompts.

    Example:
        # Full override
        strategy = DefaultStrategy(
            extractor_prompt="You are an SBC expert..."
        )

        # Append to defaults
        strategy = DefaultStrategy(
            extra_instructions="Focus on deductibles. This is a PPO plan."
        )
    """

    extractor_prompt: str | None = None
    verifier_prompt: str | None = None
    corrector_prompt: str | None = None
    extra_instructions: str | None = None

    def build_extractor_prompt(self, context: StrategyContext) -> str:
        """Build extraction prompt, using custom if provided."""
        if self.extractor_prompt is not None:
            return self.extractor_prompt

        schema_name = context.schema.__name__
        schema_fields = self._describe_schema(context.schema)

        prompt = f"""You are a precise document extraction assistant.

Your task is to extract structured data from the provided document.

OUTPUT SCHEMA: {schema_name}
{schema_fields}

INSTRUCTIONS:
1. Read the document carefully
2. Extract values that match the schema fields
3. Use exact values from the document when possible
4. If a value is not found, use null
5. Output valid JSON matching the schema

Be precise and accurate. Extract only what is explicitly stated in the document."""

        if self.extra_instructions:
            prompt += f"\n\nADDITIONAL INSTRUCTIONS:\n{self.extra_instructions}"

        return prompt

    def build_verifier_prompt(self, context: StrategyContext) -> str:
        """Build verification prompt, using custom if provided."""
        if self.verifier_prompt is not None:
            return self.verifier_prompt

        prompt = """You are a verification assistant with access to a search tool.

Your task is to verify that extracted values are correct by finding
supporting evidence in the document.

For each field to verify:
1. Use the search tool to find relevant text
2. Compare the extracted value to what you find
3. Report whether values match, mismatch, or weren't found

SEARCH TOOL:
You have access to a 'search' tool that searches the document.
Use specific queries to find the relevant sections.

OUTPUT FORMAT:
For each field, report:
- path: The field path
- status: MATCH, MISMATCH, NOT_FOUND, or AMBIGUOUS
- found_value: What you found (if different from extracted)
- quote: The exact text supporting your finding
- confidence: 0.0 to 1.0

Be thorough and accurate. Search multiple times if needed."""

        if self.extra_instructions:
            prompt += f"\n\nADDITIONAL INSTRUCTIONS:\n{self.extra_instructions}"

        return prompt

    def build_corrector_prompt(self, context: StrategyContext) -> str:
        """Build correction prompt, using custom if provided."""
        if self.corrector_prompt is not None:
            return self.corrector_prompt

        prompt = """You are a correction assistant.

Some extracted values were found to be incorrect during verification.
Your task is to provide corrected values based on the verification results.

For each field that needs correction:
1. Review the verification result
2. Determine the correct value based on what was found
3. Provide the corrected value

OUTPUT FORMAT:
Provide corrections as a JSON object with paths as keys:
{
    "field.path": corrected_value,
    ...
}

Only include fields that need correction."""

        if self.extra_instructions:
            prompt += f"\n\nADDITIONAL INSTRUCTIONS:\n{self.extra_instructions}"

        return prompt

    def _describe_schema(self, schema: Type["BaseModel"]) -> str:
        """Generate a description of the schema fields."""
        try:
            # Get JSON schema from Pydantic model
            json_schema = schema.model_json_schema()
            properties = json_schema.get("properties", {})

            lines = ["Fields:"]
            for name, prop in properties.items():
                field_type = prop.get("type", "any")
                description = prop.get("description", "")
                required = name in json_schema.get("required", [])

                line = f"  - {name}: {field_type}"
                if description:
                    line += f" ({description})"
                if required:
                    line += " [required]"
                lines.append(line)

            return "\n".join(lines)

        except Exception:
            return "  (schema description unavailable)"


__all__ = [
    "ExtractionStrategy",
    "DefaultStrategy",
    "StrategyContext",
]
