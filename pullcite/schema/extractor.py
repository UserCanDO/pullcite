"""
Schema-aware extractor for Django-style field definitions.

This extractor works with ExtractionSchema classes and uses field-level
search configurations for context gathering and verification.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any, Type, TypeVar

from .base import ExtractionSchema, Field, SearchType
from ..core.document import Document
from ..core.chunk import Chunk
from ..core.evidence import (
    Evidence,
    EvidenceCandidate,
    VerificationResult,
    VerificationStatus,
)
from ..core.result import (
    ExtractionResult,
    ExtractionStats,
    ExtractionFlag,
    ExtractionFlagType,
    ExtractionStatus,
)
from ..llms.base import LLM, Message, Role, Tool, ToolCall
from ..search.base import Searcher, SearchResult


T = TypeVar("T", bound=ExtractionSchema)


# Type for custom prompt builder callbacks
PromptBuilder = Any  # Callable[[Type[ExtractionSchema], dict[str, list[SearchResult]]], str]


@dataclass
class SchemaExtractor:
    """
    Extractor that uses Django-style schema definitions.

    Each field specifies its own search query and type, enabling field-aware
    context gathering for documents of any size.

    Key Parameters:
        schema: ExtractionSchema subclass defining fields to extract.
        llm: Language model for extraction.
        searcher: BM25 searcher for keyword search.
        top_k: Chunks retrieved per field (default 5). Critical for quality -
               higher = more context but more tokens. Start with 5, tune up/down.
        verify: Whether to verify values against source (default True).

    Example:
        >>> from pullcite import (
        ...     Extractor, ExtractionSchema, DecimalField, StringField,
        ...     SearchType, BM25Searcher,
        ... )
        >>> from pullcite.llms.anthropic import AnthropicLLM
        >>>
        >>> class Invoice(ExtractionSchema):
        ...     vendor = StringField(
        ...         query="vendor company name",
        ...         search_type=SearchType.BM25,
        ...         description="Company that issued the invoice",  # Helps LLM
        ...     )
        ...     total = DecimalField(
        ...         query="total amount due invoice",
        ...         search_type=SearchType.BM25,
        ...         description="Final total amount due",
        ...     )
        >>>
        >>> extractor = Extractor(
        ...     schema=Invoice,
        ...     llm=AnthropicLLM(),
        ...     searcher=BM25Searcher(),
        ...     top_k=5,  # 5 chunks per field
        ... )
        >>>
        >>> result = extractor.extract(doc)
        >>> # Or async:
        >>> result = await extractor.extract_async(doc)
    """

    # Required
    schema: Type[T]
    llm: LLM

    # Search configuration
    searcher: Searcher | None = None
    retriever: Any = None  # Semantic retriever for SEMANTIC/HYBRID fields
    top_k: int = 5  # Chunks per field. Critical param - tune for your docs.

    # Verification
    verify: bool = True

    # LLM settings
    temperature: float = 0.0
    max_tokens: int = 4096

    # Custom prompts
    system_prompt: str | None = None
    extra_instructions: str | None = None
    prompt_builder: PromptBuilder | None = None

    # Batching for large schemas
    max_fields_per_batch: int | None = None  # None = all fields in one call
    max_context_chars: int = 100000  # Max context chars per batch
    include_document_text: bool = True  # Include full doc in prompt
    max_document_chars: int = 50000  # Truncate doc text if too long

    # Internal state
    _indexed: bool = field(default=False, init=False, repr=False)

    def extract(self, document: Document) -> ExtractionResult[T]:
        """
        Extract structured data from a document.

        For each field in the schema:
        1. Search for relevant chunks using the field's query and search_type
        2. Build context from retrieved chunks
        3. Extract value using LLM (batched if schema is large)
        4. Optionally verify against source text

        Args:
            document: Document to extract from.

        Returns:
            ExtractionResult with data, evidence, and metrics.
        """
        start_time = time.time()
        flags: list[ExtractionFlag] = []

        # Index document if not already indexed
        if not self._indexed:
            self._index_document(document)

        # Gather context for each field
        field_contexts = self._gather_field_contexts(document)

        # Split fields into batches if needed
        batches = self._create_field_batches(field_contexts)

        # Extract using LLM (one call per batch)
        extracted_data: dict[str, Any] = {}
        total_input_tokens = 0
        total_output_tokens = 0
        llm_calls = 0

        try:
            for batch_fields, batch_contexts in batches:
                batch_prompt = self._build_extraction_prompt(
                    batch_contexts, batch_fields
                )
                batch_data, input_tokens, output_tokens = self._extract_with_llm(
                    batch_prompt, document, batch_fields
                )
                extracted_data.update(batch_data)
                total_input_tokens += input_tokens
                total_output_tokens += output_tokens
                llm_calls += 1
        except Exception as e:
            return self._failed_result(document, str(e), time.time() - start_time)

        # Parse values through field definitions
        parsed_data = {}
        for name, field_def in self.schema.get_fields().items():
            raw_value = extracted_data.get(name)
            if raw_value is not None:
                parsed_data[name] = field_def.parse(raw_value)
            else:
                parsed_data[name] = field_def.default

        # Verify if enabled
        verification_results: list[VerificationResult] = []
        evidence_map: dict[str, Evidence] = {}

        if self.verify:
            verification_results, evidence_map = self._verify_extraction(
                parsed_data, field_contexts, flags
            )

        # Create schema instance
        try:
            data_instance = self.schema(**parsed_data)
        except Exception as e:
            flags.append(
                ExtractionFlag(
                    type=ExtractionFlagType.SCHEMA_ERROR,
                    message=f"Failed to create schema instance: {e}",
                )
            )
            return self._failed_result(document, str(e), time.time() - start_time)

        # Compute status
        status = self._compute_status(verification_results, flags)
        confidence = self._compute_confidence(verification_results)

        # Build stats
        duration_ms = int((time.time() - start_time) * 1000)
        stats = ExtractionStats(
            total_duration_ms=duration_ms,
            extraction_duration_ms=duration_ms,
            extraction_input_tokens=total_input_tokens,
            extraction_output_tokens=total_output_tokens,
            extraction_llm_calls=llm_calls,
            fields_verified=len(verification_results),
            fields_passed=sum(
                1 for vr in verification_results
                if vr.status == VerificationStatus.MATCH
            ),
        )

        return ExtractionResult(
            data=data_instance,
            status=status,
            confidence=confidence,
            document_id=document.id,
            evidence_map=evidence_map,
            verification_results=tuple(verification_results),
            flags=tuple(flags),
            stats=stats,
        )

    async def extract_async(self, document: Document) -> ExtractionResult[T]:
        """
        Async version of extract for concurrent processing.

        Useful for processing multiple documents in parallel:

            >>> results = await asyncio.gather(*[
            ...     extractor.extract_async(doc) for doc in documents
            ... ])

        Args:
            document: Document to extract from.

        Returns:
            ExtractionResult with data, evidence, and metrics.
        """
        # Run sync extraction in thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.extract, document)

    def _index_document(self, document: Document) -> None:
        """Index document chunks in searcher(s)."""
        chunks = document.chunks

        if not chunks:
            return

        # Prepare chunk texts and metadata
        texts = [chunk.text for chunk in chunks]
        metadata = [
            {
                "chunk_index": chunk.index,
                "page": chunk.page,
            }
            for chunk in chunks
        ]

        # Index in BM25 searcher
        if self.searcher:
            self.searcher.index(texts, metadata)

        # Index in semantic retriever if available
        if self.retriever and hasattr(self.retriever, "index"):
            self.retriever.index(document)

        self._indexed = True

    def _gather_field_contexts(
        self, document: Document
    ) -> dict[str, list[SearchResult]]:
        """Gather relevant context for each field based on search type."""
        field_contexts: dict[str, list[SearchResult]] = {}

        for name, field_def in self.schema.get_fields().items():
            results: list[SearchResult] = []

            if field_def.search_type == SearchType.BM25:
                if self.searcher:
                    results = self.searcher.search(field_def.query, self.top_k)

            elif field_def.search_type == SearchType.SEMANTIC:
                if self.retriever and hasattr(self.retriever, "search"):
                    retriever_results = self.retriever.search(
                        field_def.query, top_k=self.top_k
                    )
                    results = self._convert_retriever_results(retriever_results)

            elif field_def.search_type == SearchType.HYBRID:
                # Combine BM25 and semantic results
                bm25_results = []
                semantic_results = []

                if self.searcher:
                    bm25_results = self.searcher.search(field_def.query, self.top_k)

                if self.retriever and hasattr(self.retriever, "search"):
                    retriever_results = self.retriever.search(
                        field_def.query, top_k=self.top_k
                    )
                    semantic_results = self._convert_retriever_results(retriever_results)

                results = self._merge_results(bm25_results, semantic_results)

            field_contexts[name] = results

        return field_contexts

    def _convert_retriever_results(self, results: list[Any]) -> list[SearchResult]:
        """Convert retriever results to SearchResult format."""
        search_results = []
        for r in results:
            if hasattr(r, "text"):
                search_results.append(
                    SearchResult(
                        text=r.text,
                        score=getattr(r, "score", 0.0),
                        chunk_index=getattr(r, "chunk_index", 0),
                        page=getattr(r, "page"),
                        metadata=getattr(r, "metadata", {}),
                    )
                )
            elif isinstance(r, dict):
                search_results.append(
                    SearchResult(
                        text=r.get("text", ""),
                        score=r.get("score", 0.0),
                        chunk_index=r.get("chunk_index", 0),
                        page=r.get("page"),
                        metadata=r.get("metadata", {}),
                    )
                )
        return search_results

    def _merge_results(
        self,
        bm25_results: list[SearchResult],
        semantic_results: list[SearchResult],
    ) -> list[SearchResult]:
        """Merge BM25 and semantic results using RRF."""
        # Simple merge for now - dedupe by chunk_index, prefer higher score
        seen: dict[int, SearchResult] = {}

        for r in bm25_results + semantic_results:
            if r.chunk_index not in seen or r.score > seen[r.chunk_index].score:
                seen[r.chunk_index] = r

        # Sort by score
        return sorted(seen.values(), key=lambda x: x.score, reverse=True)[: self.top_k]

    def _create_field_batches(
        self, field_contexts: dict[str, list[SearchResult]]
    ) -> list[tuple[list[str], dict[str, list[SearchResult]]]]:
        """
        Split fields into batches based on context size limits.

        Returns list of (field_names, field_contexts) tuples, one per batch.
        """
        all_fields = list(self.schema.get_fields().keys())

        # If no limits set, return all fields in one batch
        if self.max_fields_per_batch is None and self.max_context_chars >= 1000000:
            return [(all_fields, field_contexts)]

        batches: list[tuple[list[str], dict[str, list[SearchResult]]]] = []
        current_batch: list[str] = []
        current_contexts: dict[str, list[SearchResult]] = {}
        current_chars = 0

        for field_name in all_fields:
            contexts = field_contexts.get(field_name, [])

            # Calculate chars for this field's contexts
            field_chars = sum(len(ctx.text) for ctx in contexts)

            # Check if adding this field would exceed limits
            would_exceed_chars = (current_chars + field_chars) > self.max_context_chars
            would_exceed_fields = (
                self.max_fields_per_batch is not None
                and len(current_batch) >= self.max_fields_per_batch
            )

            # Start new batch if limits exceeded (and current batch not empty)
            if current_batch and (would_exceed_chars or would_exceed_fields):
                batches.append((current_batch, current_contexts))
                current_batch = []
                current_contexts = {}
                current_chars = 0

            # Add field to current batch
            current_batch.append(field_name)
            current_contexts[field_name] = contexts
            current_chars += field_chars

        # Don't forget the last batch
        if current_batch:
            batches.append((current_batch, current_contexts))

        return batches

    def _build_extraction_prompt(
        self,
        field_contexts: dict[str, list[SearchResult]],
        field_names: list[str] | None = None,
    ) -> str:
        """Build extraction prompt with field-specific context.

        Args:
            field_contexts: Retrieved contexts for each field.
            field_names: Subset of fields to include (None = all fields).

        Uses custom prompts if configured:
        1. If prompt_builder is set, calls it with (schema, field_contexts)
        2. If system_prompt is set, uses it directly
        3. Otherwise builds default prompt with extra_instructions appended
        """
        # Option 1: Custom prompt builder function
        if self.prompt_builder is not None:
            return self.prompt_builder(self.schema, field_contexts)

        # Option 2: Fully custom system prompt
        if self.system_prompt is not None:
            return self.system_prompt

        # Determine which fields to include
        all_fields = self.schema.get_fields()
        if field_names is not None:
            fields_to_include = {k: all_fields[k] for k in field_names if k in all_fields}
        else:
            fields_to_include = all_fields

        # Option 3: Default prompt with optional extra instructions
        lines = [
            "Extract the following fields from the document.",
            "For each field, relevant excerpts from the document are provided.",
            "",
        ]

        for name, field_def in fields_to_include.items():
            lines.append(f"## {field_def.label or name}")

            # Description is critical for LLM understanding
            if field_def.description:
                lines.append(f"Description: {field_def.description}")
            else:
                # Fallback: use the search query as a hint
                lines.append(f"Look for: {field_def.query}")

            lines.append(f"Required: {'Yes' if field_def.required else 'No'}")

            # Add context excerpts
            contexts = field_contexts.get(name, [])
            if contexts:
                lines.append("Relevant excerpts:")
                for i, ctx in enumerate(contexts, 1):
                    lines.append(f"  [{i}] {ctx.text[:500]}...")
            else:
                lines.append("(No specific excerpts found - extract from full document)")

            lines.append("")

        lines.append(
            "Return the extracted values as JSON matching the schema. "
            "Use null for values not found in the document."
        )

        # Append extra instructions if provided
        if self.extra_instructions:
            lines.append("")
            lines.append("ADDITIONAL INSTRUCTIONS:")
            lines.append(self.extra_instructions)

        return "\n".join(lines)

    def _extract_with_llm(
        self,
        prompt: str,
        document: Document,
        field_names: list[str] | None = None,
    ) -> tuple[dict[str, Any], int, int]:
        """Extract using LLM with structured output.

        Args:
            prompt: System prompt for extraction.
            document: Document to extract from.
            field_names: Subset of fields to extract (None = all fields).
        """
        # Build schema for specified fields only
        if field_names is not None:
            all_fields = self.schema.get_fields()
            properties = {}
            required = []
            for name in field_names:
                if name in all_fields:
                    field_def = all_fields[name]
                    prop_schema = field_def.to_json_schema()
                    if field_def.description:
                        prop_schema["description"] = field_def.description
                    properties[name] = prop_schema
                    if field_def.required:
                        required.append(name)
            schema_json = {
                "type": "object",
                "properties": properties,
                "required": required,
            }
        else:
            schema_json = self.schema.to_json_schema()

        # Build user message with document text
        doc_text = document.full_text
        if self.include_document_text:
            doc_text = doc_text[: self.max_document_chars]
            user_content = f"Document:\n\n{doc_text}"
        else:
            user_content = "Extract from the excerpts provided above."

        messages = [
            Message(role=Role.SYSTEM, content=prompt),
            Message(role=Role.USER, content=user_content),
        ]

        # Use tool calling for structured output
        tool = Tool(
            name="extract_data",
            description="Extract structured data from the document",
            parameters=schema_json,
        )

        response = self.llm.complete(
            messages=messages,
            tools=[tool],
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )

        # Parse tool call response
        if response.tool_calls:
            return (
                response.tool_calls[0].arguments,
                response.input_tokens,
                response.output_tokens,
            )

        # If no tool call, try to parse content as JSON
        import json

        try:
            data = json.loads(response.content or "{}")
            return data, response.input_tokens, response.output_tokens
        except json.JSONDecodeError:
            return {}, response.input_tokens, response.output_tokens

    def _verify_extraction(
        self,
        data: dict[str, Any],
        field_contexts: dict[str, list[SearchResult]],
        flags: list[ExtractionFlag],
    ) -> tuple[list[VerificationResult], dict[str, Evidence]]:
        """
        Verify extracted values against source text.

        Verification is field-type aware:
        1. For each extracted value, search the retrieved context chunks
        2. Use field.parse_from_text() to find values in chunk text
        3. Use field.compare() to check if extracted matches found value

        Field-specific comparison:
        - StringField: Case-insensitive, whitespace-normalized comparison
        - DecimalField: Tolerance-based comparison (default ±0.01)
        - PercentField: Tolerance-based comparison
        - BooleanField: Exact match after normalization
        - etc.

        Returns:
        - VerificationStatus.MATCH: Value verified in source
        - VerificationStatus.MISMATCH: Found different value in source
        - VerificationStatus.NOT_FOUND: Required field not found
        - VerificationStatus.SKIPPED: No context to verify against
        """
        results: list[VerificationResult] = []
        evidence_map: dict[str, Evidence] = {}

        for name, field_def in self.schema.get_fields().items():
            extracted_value = data.get(name)
            contexts = field_contexts.get(name, [])

            # Skip verification if no value or no context
            if extracted_value is None:
                if field_def.required:
                    results.append(
                        VerificationResult(
                            path=name,
                            status=VerificationStatus.NOT_FOUND,
                            extracted_value=None,
                        )
                    )
                    flags.append(
                        ExtractionFlag(
                            type=ExtractionFlagType.NOT_FOUND,
                            message=f"Required field not found: {name}",
                            path=name,
                        )
                    )
                continue

            if not contexts:
                # No context to verify against
                results.append(
                    VerificationResult(
                        path=name,
                        status=VerificationStatus.SKIPPED,
                        extracted_value=extracted_value,
                    )
                )
                continue

            # Try to find matching value in context
            candidates: list[EvidenceCandidate] = []

            for ctx in contexts:
                # Parse value from context text
                found_value = field_def.parse_from_text(ctx.text)

                if found_value is not None:
                    candidates.append(
                        EvidenceCandidate(
                            quote=ctx.text[:500],
                            chunk_index=ctx.chunk_index,
                            score=ctx.score,
                            page=ctx.page,
                            parsed_value=found_value,
                        )
                    )

            # Check if any candidate matches
            matching_candidate = None
            for candidate in candidates:
                if field_def.compare(extracted_value, candidate.parsed_value):
                    matching_candidate = candidate
                    break

            if matching_candidate:
                evidence = Evidence(
                    value=extracted_value,
                    quote=matching_candidate.quote,
                    page=matching_candidate.page,
                    bbox=matching_candidate.bbox,
                    chunk_index=matching_candidate.chunk_index,
                    confidence=min(1.0, matching_candidate.score),
                    verified=True,
                )
                evidence_map[name] = evidence

                results.append(
                    VerificationResult(
                        path=name,
                        status=VerificationStatus.MATCH,
                        extracted_value=extracted_value,
                        found_value=matching_candidate.parsed_value,
                        evidence=evidence,
                        candidates=tuple(candidates),
                    )
                )
            elif candidates:
                # Found candidates but no match - mismatch
                best = candidates[0]
                evidence = Evidence(
                    value=best.parsed_value,
                    quote=best.quote,
                    page=best.page,
                    bbox=best.bbox,
                    chunk_index=best.chunk_index,
                    confidence=min(1.0, best.score),
                    verified=False,
                )

                results.append(
                    VerificationResult(
                        path=name,
                        status=VerificationStatus.MISMATCH,
                        extracted_value=extracted_value,
                        found_value=best.parsed_value,
                        evidence=evidence,
                        candidates=tuple(candidates),
                    )
                )

                flags.append(
                    ExtractionFlag(
                        type=ExtractionFlagType.MISMATCH_UNCORRECTED,
                        message=f"Extracted {extracted_value}, found {best.parsed_value}",
                        path=name,
                    )
                )
            else:
                # No candidates found
                results.append(
                    VerificationResult(
                        path=name,
                        status=VerificationStatus.NOT_FOUND,
                        extracted_value=extracted_value,
                    )
                )

        return results, evidence_map

    def _compute_status(
        self,
        results: list[VerificationResult],
        flags: list[ExtractionFlag],
    ) -> ExtractionStatus:
        """Compute extraction status from verification results."""
        if not results:
            return ExtractionStatus.VERIFIED

        matches = sum(1 for r in results if r.status == VerificationStatus.MATCH)
        failures = sum(1 for r in results if r.is_failure)

        if failures == 0:
            return ExtractionStatus.VERIFIED
        elif matches > 0:
            return ExtractionStatus.PARTIAL
        else:
            return ExtractionStatus.FAILED

    def _compute_confidence(self, results: list[VerificationResult]) -> float:
        """Compute overall confidence from verification results."""
        if not results:
            return 1.0

        confidences = []
        for r in results:
            if r.evidence:
                confidences.append(r.evidence.confidence)
            elif r.status == VerificationStatus.MATCH:
                confidences.append(1.0)
            elif r.status == VerificationStatus.SKIPPED:
                confidences.append(0.5)
            else:
                confidences.append(0.0)

        return sum(confidences) / len(confidences) if confidences else 0.0

    def _failed_result(
        self, document: Document, error: str, elapsed: float
    ) -> ExtractionResult[T]:
        """Create a failed extraction result."""
        return ExtractionResult(
            data=None,  # type: ignore
            status=ExtractionStatus.FAILED,
            confidence=0.0,
            document_id=document.id,
            evidence_map={},
            verification_results=(),
            flags=(
                ExtractionFlag(
                    type=ExtractionFlagType.TOOL_ERROR,
                    message=f"Extraction failed: {error}",
                ),
            ),
            stats=ExtractionStats(total_duration_ms=int(elapsed * 1000)),
        )


__all__ = ["SchemaExtractor"]
