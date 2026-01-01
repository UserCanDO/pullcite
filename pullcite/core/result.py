"""
Extraction result types.

This module defines the final output of extraction and metrics.

Key types:
- ExtractionResult: Complete result with data + evidence
- ExtractionStats: Performance and cost metrics
- ExtractionFlag: Warnings and issues from extraction
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Generic, TypeVar

from .evidence import Evidence, VerificationResult, VerificationStatus


T = TypeVar("T")
"""
Generic type variable for ExtractionResult.

Allows ExtractionResult to be typed with the specific Pydantic model
type used for the data field, providing better type safety.

Example:
    class MyModel(BaseModel):
        name: str

    result: ExtractionResult[MyModel] = ExtractionResult(...)
    # result.data is now typed as MyModel
"""


class ExtractionStatus(str, Enum):
    """Overall extraction outcome."""

    VERIFIED = "VERIFIED"
    """All critical fields verified, confidence meets threshold."""

    PARTIAL = "PARTIAL"
    """Some critical fields verified, some failed."""

    LOW_CONFIDENCE = "LOW_CONFIDENCE"
    """Extraction succeeded but confidence below threshold."""

    FAILED = "FAILED"
    """Extraction could not complete."""


class ExtractionFlagType(str, Enum):
    """Types of extraction flags."""

    LOW_CONFIDENCE = "LOW_CONFIDENCE"
    """Field extracted but confidence below threshold."""

    AMBIGUOUS = "AMBIGUOUS"
    """Multiple possible values found."""

    NOT_FOUND = "NOT_FOUND"
    """Required field not found in document."""

    MISMATCH_CORRECTED = "MISMATCH_CORRECTED"
    """Extracted value was wrong, corrector fixed it."""

    MISMATCH_UNCORRECTED = "MISMATCH_UNCORRECTED"
    """Extracted value was wrong, correction failed."""

    SCHEMA_ERROR = "SCHEMA_ERROR"
    """Output didn't match Pydantic schema."""

    TOOL_ERROR = "TOOL_ERROR"
    """Search tool failed during verification."""

    PARSE_ERROR = "PARSE_ERROR"
    """Couldn't parse value for comparison."""

    RETRY_EXHAUSTED = "RETRY_EXHAUSTED"
    """Max retries reached for LLM or tool call."""


@dataclass(frozen=True)
class ExtractionFlag:
    """
    A warning or issue from extraction.

    Flags provide visibility into problems that occurred during
    extraction without failing the entire process.

    Attributes:
        type: Category of the flag.
        path: Field path this flag relates to (if applicable).
        message: Human-readable description.
        details: Additional structured data.
    """

    type: ExtractionFlagType
    message: str
    path: str | None = None
    details: dict[str, Any] | None = None

    def __str__(self) -> str:
        if self.path:
            return f"[{self.type.value}] {self.path}: {self.message}"
        return f"[{self.type.value}] {self.message}"


@dataclass(frozen=True)
class ExtractionStats:
    """
    Metrics from extraction process.

    Tracks timing, token usage, and verification outcomes
    for cost analysis and debugging.

    All durations are in milliseconds.
    """

    # Timing
    total_duration_ms: int = 0
    """Total wall-clock time."""

    extraction_duration_ms: int = 0
    """Time spent in extraction phase."""

    verification_duration_ms: int = 0
    """Time spent in verification phase."""

    correction_duration_ms: int = 0
    """Time spent in correction phase."""

    # Token counts (extraction phase)
    extraction_input_tokens: int = 0
    extraction_output_tokens: int = 0
    extraction_llm_calls: int = 0

    # Token counts (verification phase)
    verification_input_tokens: int = 0
    verification_output_tokens: int = 0
    verification_llm_calls: int = 0
    verification_tool_calls: int = 0
    """Number of search tool invocations."""

    # Token counts (correction phase)
    correction_input_tokens: int = 0
    correction_output_tokens: int = 0
    correction_llm_calls: int = 0

    # Retries
    llm_retries: int = 0
    """LLM calls that were retried due to errors."""

    tool_retries: int = 0
    """Tool calls that were retried."""

    # Verification summary
    fields_verified: int = 0
    """Total fields that went through verification."""

    fields_passed: int = 0
    """Fields where extracted matched document."""

    fields_corrected: int = 0
    """Fields that were corrected after mismatch."""

    fields_failed: int = 0
    """Fields that couldn't be verified or corrected."""

    @property
    def total_input_tokens(self) -> int:
        """Total input tokens across all phases."""
        return (
            self.extraction_input_tokens
            + self.verification_input_tokens
            + self.correction_input_tokens
        )

    @property
    def total_output_tokens(self) -> int:
        """Total output tokens across all phases."""
        return (
            self.extraction_output_tokens
            + self.verification_output_tokens
            + self.correction_output_tokens
        )

    @property
    def total_tokens(self) -> int:
        """Total tokens (input + output)."""
        return self.total_input_tokens + self.total_output_tokens

    @property
    def total_llm_calls(self) -> int:
        """Total LLM API calls across all phases."""
        return (
            self.extraction_llm_calls
            + self.verification_llm_calls
            + self.correction_llm_calls
        )

    @property
    def verification_pass_rate(self) -> float:
        """Percentage of fields that passed verification."""
        if self.fields_verified == 0:
            return 0.0
        return self.fields_passed / self.fields_verified

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "total_duration_ms": self.total_duration_ms,
            "extraction_duration_ms": self.extraction_duration_ms,
            "verification_duration_ms": self.verification_duration_ms,
            "correction_duration_ms": self.correction_duration_ms,
            "extraction_input_tokens": self.extraction_input_tokens,
            "extraction_output_tokens": self.extraction_output_tokens,
            "extraction_llm_calls": self.extraction_llm_calls,
            "verification_input_tokens": self.verification_input_tokens,
            "verification_output_tokens": self.verification_output_tokens,
            "verification_llm_calls": self.verification_llm_calls,
            "verification_tool_calls": self.verification_tool_calls,
            "correction_input_tokens": self.correction_input_tokens,
            "correction_output_tokens": self.correction_output_tokens,
            "correction_llm_calls": self.correction_llm_calls,
            "llm_retries": self.llm_retries,
            "tool_retries": self.tool_retries,
            "fields_verified": self.fields_verified,
            "fields_passed": self.fields_passed,
            "fields_corrected": self.fields_corrected,
            "fields_failed": self.fields_failed,
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "total_tokens": self.total_tokens,
            "total_llm_calls": self.total_llm_calls,
            "verification_pass_rate": self.verification_pass_rate,
        }


@dataclass
class ExtractionResult(Generic[T]):
    """
    Complete result of extraction with evidence.

    This is the main output of the Extractor. Contains the extracted
    data, verification evidence, and metrics.

    Immutability Contract:
        This class is NOT frozen (frozen=True) because the evidence_map
        dict cannot be frozen at the dataclass level. However, callers
        MUST treat all fields as immutable after construction:

        - Do NOT modify evidence_map directly; it's populated during
          construction and should be read-only afterward.
        - verification_results and flags are tuples (immutable).
        - stats is a frozen dataclass.
        - data is a Pydantic model (treat as immutable).

        If you need to "modify" a result, create a new ExtractionResult
        with the updated values.

    Attributes:
        data: Extracted data as Pydantic model instance.
        status: Overall outcome.
        confidence: Overall confidence (0.0 - 1.0).
        evidence_map: Map of path -> Evidence. READ-ONLY after construction.
        verification_results: Detailed results for each verified field.
        flags: Warnings and issues.
        stats: Performance metrics.
        document_id: ID of source document.
    """

    data: T
    status: ExtractionStatus
    confidence: float
    document_id: str
    evidence_map: dict[str, Evidence] = field(default_factory=dict)
    verification_results: tuple[VerificationResult, ...] = ()
    flags: tuple[ExtractionFlag, ...] = ()
    stats: ExtractionStats = field(default_factory=ExtractionStats)

    def __post_init__(self) -> None:
        """Validate result."""
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(
                f"confidence must be between 0.0 and 1.0, got {self.confidence}"
            )

    def evidence(self, path: str) -> Evidence | None:
        """
        Get evidence for a specific path.

        Args:
            path: Field path to look up.

        Returns:
            Evidence if found, None otherwise.
        """
        return self.evidence_map.get(path)

    def is_verified(self, path: str) -> bool:
        """
        Check if a path was successfully verified.

        Args:
            path: Field path to check.

        Returns:
            True if path has verified evidence.
        """
        ev = self.evidence_map.get(path)
        return ev is not None and ev.verified

    @property
    def unverified_fields(self) -> list[str]:
        """Paths that failed verification."""
        return [
            vr.path
            for vr in self.verification_results
            if vr.status not in (VerificationStatus.MATCH, VerificationStatus.SKIPPED)
        ]

    @property
    def verified_fields(self) -> list[str]:
        """Paths that passed verification."""
        return [
            vr.path
            for vr in self.verification_results
            if vr.status == VerificationStatus.MATCH
        ]

    @property
    def corrected_fields(self) -> list[str]:
        """Paths that were corrected after mismatch."""
        return [
            flag.path
            for flag in self.flags
            if flag.type == ExtractionFlagType.MISMATCH_CORRECTED and flag.path
        ]

    @property
    def is_success(self) -> bool:
        """Check if extraction succeeded (VERIFIED or PARTIAL)."""
        return self.status in (ExtractionStatus.VERIFIED, ExtractionStatus.PARTIAL)

    @property
    def has_warnings(self) -> bool:
        """Check if there are any flags."""
        return len(self.flags) > 0

    def get_flags_for_path(self, path: str) -> list[ExtractionFlag]:
        """
        Get all flags for a specific path.

        Args:
            path: Field path to filter by.

        Returns:
            List of flags for that path.
        """
        return [f for f in self.flags if f.path == path]

    def get_flags_by_type(self, flag_type: ExtractionFlagType) -> list[ExtractionFlag]:
        """
        Get all flags of a specific type.

        Args:
            flag_type: Type to filter by.

        Returns:
            List of flags of that type.
        """
        return [f for f in self.flags if f.type == flag_type]

    def to_dict(self) -> dict[str, Any]:
        """
        Convert to dictionary for serialization.

        Note: data is not serialized (use data.model_dump() separately).

        Returns:
            Dict with result metadata.
        """
        return {
            "status": self.status.value,
            "confidence": self.confidence,
            "document_id": self.document_id,
            "verified_fields": self.verified_fields,
            "unverified_fields": self.unverified_fields,
            "flags": [str(f) for f in self.flags],
            "stats": self.stats.to_dict(),
        }


def compute_status(
    confidence: float,
    confidence_threshold: float,
    verification_results: list[VerificationResult],
    critical_required: list[str],
) -> ExtractionStatus:
    """
    Compute extraction status from results.

    Args:
        confidence: Overall confidence score.
        confidence_threshold: Minimum for VERIFIED status.
        verification_results: Results from verification.
        critical_required: Paths that must pass (required=True).

    Returns:
        Appropriate ExtractionStatus.
    """
    if not verification_results:
        if confidence >= confidence_threshold:
            return ExtractionStatus.VERIFIED
        return ExtractionStatus.LOW_CONFIDENCE

    # Check if all required fields passed
    passed_paths = {
        vr.path for vr in verification_results if vr.status == VerificationStatus.MATCH
    }

    failed_required = [p for p in critical_required if p not in passed_paths]

    if failed_required:
        # Some required fields failed
        if passed_paths:
            return ExtractionStatus.PARTIAL
        return ExtractionStatus.FAILED

    # All required passed
    if confidence >= confidence_threshold:
        return ExtractionStatus.VERIFIED

    return ExtractionStatus.LOW_CONFIDENCE


def compute_confidence(
    verification_results: list[VerificationResult],
    base_confidence: float = 1.0,
) -> float:
    """
    Compute overall confidence from verification results.

    Args:
        verification_results: Results from verification.
        base_confidence: Starting confidence (default 1.0).

    Returns:
        Confidence score between 0.0 and 1.0.
    """
    if not verification_results:
        return base_confidence

    # Average confidence of evidence, penalize failures
    total = 0.0
    count = 0

    for vr in verification_results:
        if vr.status == VerificationStatus.SKIPPED:
            continue

        count += 1

        if vr.status == VerificationStatus.MATCH and vr.evidence:
            total += vr.evidence.confidence
        elif vr.status == VerificationStatus.MISMATCH:
            total += 0.3  # Partial credit for found but wrong
        else:
            total += 0.0  # NOT_FOUND, AMBIGUOUS

    if count == 0:
        return base_confidence

    return total / count


__all__ = [
    "ExtractionResult",
    "ExtractionStats",
    "ExtractionFlag",
    "ExtractionFlagType",
    "ExtractionStatus",
    "compute_status",
    "compute_confidence",
]
