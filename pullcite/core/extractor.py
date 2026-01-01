"""
Main Extractor class.

This module provides the primary interface for extracting structured
data from documents with evidence-backed verification.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Generic, Type, TypeVar

if TYPE_CHECKING:
    from pydantic import BaseModel

from .config import ExtractorConfig, Hooks, DEFAULT_CONFIG
from .document import Document
from .evidence import VerificationResult, VerificationStatus
from .fields import CriticalField, expand_critical_fields
from .result import (
    ExtractionResult,
    ExtractionStats,
    ExtractionFlag,
    ExtractionFlagType,
    ExtractionStatus,
    compute_status,
    compute_confidence,
)
from ..llms.base import LLM
from ..retrieval.base import Retriever
from ..pipeline.strategy import ExtractionStrategy, DefaultStrategy, StrategyContext
from ..pipeline.roles import ExtractorRole, VerifierRole, CorrectorRole
from ..pipeline.patcher import Patcher, create_patches


T = TypeVar("T", bound="BaseModel")


@dataclass
class Extractor(Generic[T]):
    """
    Extracts structured data from documents with verification.

    The Extractor orchestrates the three-role pipeline:
    1. Extractor role: Reads document, produces structured data
    2. Verifier role: Searches document, confirms critical fields
    3. Corrector role: Fixes values that failed verification

    Attributes:
        schema: Pydantic model class for output.
        llm: Language model for all roles.
        retriever: Document retriever for verification.
        critical_fields: Fields that must be verified.
        strategy: Prompt building strategy.
        config: Extraction configuration.
        hooks: Lifecycle callbacks.

    Example:
        >>> from pydantic import BaseModel
        >>> from pullcite import Extractor, Document
        >>>
        >>> class Invoice(BaseModel):
        ...     total: float
        ...     vendor: str
        ...
        >>> extractor = Extractor(
        ...     schema=Invoice,
        ...     llm=my_llm,
        ...     retriever=my_retriever,
        ...     critical_fields=[
        ...         CriticalField(path="total", label="Total", search_query="total amount"),
        ...     ],
        ... )
        >>> result = extractor.extract(document)
        >>> print(result.data.total)
    """

    schema: Type[T]
    llm: LLM
    retriever: Retriever
    critical_fields: list[CriticalField] = field(default_factory=list)
    strategy: ExtractionStrategy = field(default_factory=DefaultStrategy)
    config: ExtractorConfig = field(default_factory=lambda: DEFAULT_CONFIG)
    hooks: Hooks | None = None

    def extract(self, document: Document) -> ExtractionResult[T]:
        """
        Extract structured data from a document.

        Runs the full pipeline: extract → verify → correct.

        Args:
            document: Document to extract from.

        Returns:
            ExtractionResult with data, evidence, and metrics.
        """
        start_time = time.time()

        # Initialize stats tracking
        stats = _StatsAccumulator()
        flags: list[ExtractionFlag] = []

        # Build context for strategy
        context = StrategyContext(
            document=document,
            schema=self.schema,
            critical_fields=self.critical_fields,
        )

        # Phase 1: Extraction
        extraction_start = time.time()
        extractor_role = ExtractorRole(
            llm=self.llm,
            schema=self.schema,
            max_retries=self.config.max_extraction_retries,
        )

        try:
            extracted_data, input_tokens, output_tokens = extractor_role.extract(
                system_prompt=self.strategy.build_extractor_prompt(context),
                user_message=self.strategy.build_extraction_user_message(context),
                temperature=self.config.temperature,
                hooks=self.hooks,
            )
        except Exception as e:
            # Extraction failed completely
            return self._failed_result(
                document=document,
                error=str(e),
                stats=stats.finalize(time.time() - start_time),
            )

        stats.extraction_input_tokens = input_tokens
        stats.extraction_output_tokens = output_tokens
        stats.extraction_llm_calls = 1
        stats.extraction_duration_ms = int((time.time() - extraction_start) * 1000)

        # Apply post_extract hook
        if self.hooks and self.hooks.post_extract:
            extracted_data = self.hooks.post_extract(extracted_data, document)

        # Update context with extracted data
        context = StrategyContext(
            document=document,
            schema=self.schema,
            critical_fields=self.critical_fields,
            extracted_data=extracted_data,
        )

        # Phase 2: Verification (if not skipped)
        verification_results: list[VerificationResult] = []
        evidence_map: dict[str, Any] = {}

        if not self.config.skip_verification and self.critical_fields:
            verification_start = time.time()

            # Index document for retrieval
            self.retriever.index(document)

            # Expand wildcard fields
            expanded_fields = expand_critical_fields(
                self.critical_fields, extracted_data
            )
            stats.fields_verified = len(expanded_fields)

            # Verify in batches
            verification_results = self._verify_fields(
                context=context,
                fields=expanded_fields,
                stats=stats,
            )

            stats.verification_duration_ms = int(
                (time.time() - verification_start) * 1000
            )

            # Build evidence map from results
            for vr in verification_results:
                if vr.evidence:
                    evidence_map[vr.path] = vr.evidence

            # Apply post_verify hook
            if self.hooks and self.hooks.post_verify:
                extracted_data = self.hooks.post_verify(
                    extracted_data, verification_results, document
                )

            # Update context with verification results
            context = StrategyContext(
                document=document,
                schema=self.schema,
                critical_fields=self.critical_fields,
                extracted_data=extracted_data,
                verification_results=verification_results,
            )

            # Phase 3: Correction (if needed)
            needs_correction = [
                vr
                for vr in verification_results
                if vr.status
                in (VerificationStatus.MISMATCH, VerificationStatus.AMBIGUOUS)
            ]

            if needs_correction and self.config.max_correction_attempts > 0:
                correction_start = time.time()

                extracted_data, correction_flags = self._correct_values(
                    context=context,
                    needs_correction=needs_correction,
                    stats=stats,
                )

                flags.extend(correction_flags)
                stats.correction_duration_ms = int(
                    (time.time() - correction_start) * 1000
                )

                # Apply post_correct hook
                if self.hooks and self.hooks.post_correct:
                    extracted_data = self.hooks.post_correct(
                        extracted_data, verification_results, document
                    )

        # Add flags for verification failures
        for vr in verification_results:
            if vr.status == VerificationStatus.NOT_FOUND:
                field = next(
                    (f for f in self.critical_fields if f.path == vr.path), None
                )
                if field and field.required:
                    flags.append(
                        ExtractionFlag(
                            type=ExtractionFlagType.NOT_FOUND,
                            message=f"Required field not found in document",
                            path=vr.path,
                        )
                    )
            elif vr.status == VerificationStatus.AMBIGUOUS:
                flags.append(
                    ExtractionFlag(
                        type=ExtractionFlagType.AMBIGUOUS,
                        message=f"Multiple conflicting values found",
                        path=vr.path,
                        details={"candidate_count": len(vr.candidates)},
                    )
                )

        # Check confidence levels
        for vr in verification_results:
            if (
                vr.evidence
                and vr.evidence.confidence < self.config.low_confidence_threshold
            ):
                flags.append(
                    ExtractionFlag(
                        type=ExtractionFlagType.LOW_CONFIDENCE,
                        message=f"Confidence {vr.evidence.confidence:.2f} below threshold",
                        path=vr.path,
                    )
                )

        # Compute final status and confidence
        required_paths = [f.path for f in self.critical_fields if f.required]
        # Expand required paths for comparison
        expanded_required = []
        for f in self.critical_fields:
            if f.required:
                if f.has_wildcard:
                    expanded = expand_critical_fields([f], extracted_data)
                    expanded_required.extend(ef.path for ef in expanded)
                else:
                    expanded_required.append(f.path)

        confidence = compute_confidence(verification_results)
        status = compute_status(
            confidence=confidence,
            confidence_threshold=self.config.confidence_threshold,
            verification_results=verification_results,
            critical_required=expanded_required,
        )

        # Build final result
        final_stats = stats.finalize(time.time() - start_time)

        # Count verification outcomes
        final_stats = ExtractionStats(
            total_duration_ms=final_stats.total_duration_ms,
            extraction_duration_ms=stats.extraction_duration_ms,
            verification_duration_ms=stats.verification_duration_ms,
            correction_duration_ms=stats.correction_duration_ms,
            extraction_input_tokens=stats.extraction_input_tokens,
            extraction_output_tokens=stats.extraction_output_tokens,
            extraction_llm_calls=stats.extraction_llm_calls,
            verification_input_tokens=stats.verification_input_tokens,
            verification_output_tokens=stats.verification_output_tokens,
            verification_llm_calls=stats.verification_llm_calls,
            verification_tool_calls=stats.verification_tool_calls,
            correction_input_tokens=stats.correction_input_tokens,
            correction_output_tokens=stats.correction_output_tokens,
            correction_llm_calls=stats.correction_llm_calls,
            llm_retries=stats.llm_retries,
            tool_retries=stats.tool_retries,
            fields_verified=len(verification_results),
            fields_passed=sum(
                1
                for vr in verification_results
                if vr.status == VerificationStatus.MATCH
            ),
            fields_corrected=stats.fields_corrected,
            fields_failed=sum(1 for vr in verification_results if vr.is_failure),
        )

        # Validate and create model instance
        try:
            model_instance = self.schema.model_validate(extracted_data)
        except Exception as e:
            flags.append(
                ExtractionFlag(
                    type=ExtractionFlagType.SCHEMA_ERROR,
                    message=f"Final data doesn't match schema: {e}",
                )
            )
            # Return with raw data
            return ExtractionResult(
                data=extracted_data,  # type: ignore
                status=ExtractionStatus.FAILED,
                confidence=confidence,
                document_id=document.id,
                evidence_map=evidence_map,
                verification_results=tuple(verification_results),
                flags=tuple(flags),
                stats=final_stats,
            )

        return ExtractionResult(
            data=model_instance,
            status=status,
            confidence=confidence,
            document_id=document.id,
            evidence_map=evidence_map,
            verification_results=tuple(verification_results),
            flags=tuple(flags),
            stats=final_stats,
        )

    def _verify_fields(
        self,
        context: StrategyContext,
        fields: list[CriticalField],
        stats: "_StatsAccumulator",
    ) -> list[VerificationResult]:
        """Verify fields in batches."""
        verifier_role = VerifierRole(
            llm=self.llm,
            retriever=self.retriever,
            policies=self.config.policies,
        )

        all_results: list[VerificationResult] = []

        # Determine batch size
        batch_size = self.config.verification_batch_size or len(fields)

        for i in range(0, len(fields), batch_size):
            batch = fields[i : i + batch_size]

            results, input_tokens, output_tokens, tool_calls = verifier_role.verify(
                system_prompt=self.strategy.build_verifier_prompt(context),
                user_message=self.strategy.build_verification_user_message(
                    context, batch
                ),
                fields=batch,
                extracted_data=context.extracted_data or {},
                max_tool_rounds=self.config.max_tool_rounds,
                temperature=self.config.temperature,
                hooks=self.hooks,
            )

            all_results.extend(results)
            stats.verification_input_tokens += input_tokens
            stats.verification_output_tokens += output_tokens
            stats.verification_llm_calls += 1
            stats.verification_tool_calls += tool_calls

        return all_results

    def _correct_values(
        self,
        context: StrategyContext,
        needs_correction: list[VerificationResult],
        stats: "_StatsAccumulator",
    ) -> tuple[dict[str, Any], list[ExtractionFlag]]:
        """Correct values that failed verification."""
        corrector_role = CorrectorRole(
            llm=self.llm,
            max_retries=self.config.max_correction_attempts,
        )

        flags: list[ExtractionFlag] = []
        data = dict(context.extracted_data or {})

        try:
            corrections, input_tokens, output_tokens = corrector_role.correct(
                system_prompt=self.strategy.build_corrector_prompt(context),
                user_message=self.strategy.build_correction_user_message(context),
                temperature=self.config.temperature,
                hooks=self.hooks,
            )

            stats.correction_input_tokens = input_tokens
            stats.correction_output_tokens = output_tokens
            stats.correction_llm_calls = 1

            if corrections:
                # Apply corrections using patcher
                patcher = Patcher(schema=self.schema, strict=False)
                patches = create_patches(
                    corrections, data, reason="verification_mismatch"
                )

                data = patcher.apply(data, patches)

                # Track successful corrections
                for result in patcher.applied_results:
                    if result.success:
                        stats.fields_corrected += 1
                        flags.append(
                            ExtractionFlag(
                                type=ExtractionFlagType.MISMATCH_CORRECTED,
                                message=f"Value corrected from {result.patch.old_value} to {result.patch.new_value}",
                                path=result.patch.path,
                            )
                        )
                    else:
                        flags.append(
                            ExtractionFlag(
                                type=ExtractionFlagType.MISMATCH_UNCORRECTED,
                                message=f"Correction failed: {result.error}",
                                path=result.patch.path,
                            )
                        )

        except Exception as e:
            # Correction failed - add flag but continue
            flags.append(
                ExtractionFlag(
                    type=ExtractionFlagType.TOOL_ERROR,
                    message=f"Correction phase failed: {e}",
                )
            )

        return data, flags

    def _failed_result(
        self,
        document: Document,
        error: str,
        stats: ExtractionStats,
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
            stats=stats,
        )


@dataclass
class _StatsAccumulator:
    """Mutable accumulator for stats during extraction."""

    extraction_duration_ms: int = 0
    verification_duration_ms: int = 0
    correction_duration_ms: int = 0

    extraction_input_tokens: int = 0
    extraction_output_tokens: int = 0
    extraction_llm_calls: int = 0

    verification_input_tokens: int = 0
    verification_output_tokens: int = 0
    verification_llm_calls: int = 0
    verification_tool_calls: int = 0

    correction_input_tokens: int = 0
    correction_output_tokens: int = 0
    correction_llm_calls: int = 0

    llm_retries: int = 0
    tool_retries: int = 0

    fields_verified: int = 0
    fields_corrected: int = 0

    def finalize(self, total_seconds: float) -> ExtractionStats:
        """Convert to immutable ExtractionStats."""
        return ExtractionStats(
            total_duration_ms=int(total_seconds * 1000),
            extraction_duration_ms=self.extraction_duration_ms,
            verification_duration_ms=self.verification_duration_ms,
            correction_duration_ms=self.correction_duration_ms,
            extraction_input_tokens=self.extraction_input_tokens,
            extraction_output_tokens=self.extraction_output_tokens,
            extraction_llm_calls=self.extraction_llm_calls,
            verification_input_tokens=self.verification_input_tokens,
            verification_output_tokens=self.verification_output_tokens,
            verification_llm_calls=self.verification_llm_calls,
            verification_tool_calls=self.verification_tool_calls,
            correction_input_tokens=self.correction_input_tokens,
            correction_output_tokens=self.correction_output_tokens,
            correction_llm_calls=self.correction_llm_calls,
            llm_retries=self.llm_retries,
            tool_retries=self.tool_retries,
            fields_verified=self.fields_verified,
            fields_corrected=self.fields_corrected,
        )


__all__ = ["Extractor"]
