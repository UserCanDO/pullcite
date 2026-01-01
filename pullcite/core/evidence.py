"""
Evidence types for verification.

This module defines the types that represent proof a value came from
the source document. Used by the Verifier role to record what it found.

Key types:
- Evidence: Final proof for a verified field
- EvidenceCandidate: Potential match before selection
- VerificationResult: Outcome of verifying one field
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any


class VerificationStatus(str, Enum):
    """Outcome of verifying a single field."""

    MATCH = "MATCH"
    """Extracted value matches document."""

    MISMATCH = "MISMATCH"
    """Extracted value differs from document."""

    NOT_FOUND = "NOT_FOUND"
    """Value not found in document."""

    AMBIGUOUS = "AMBIGUOUS"
    """Multiple conflicting values found."""

    SKIPPED = "SKIPPED"
    """Verification skipped (e.g., non-critical field)."""


@dataclass(frozen=True)
class Evidence:
    """
    Proof that a value came from the document.

    This is the final, selected evidence for a verified field.
    Created after selecting the best candidate from search results.

    Attributes:
        value: The verified value (may differ from extracted if corrected).
        quote: Exact text from document supporting this value.
        page: Page number where quote was found (1-indexed).
        bbox: Bounding box of quote (x0, y0, x1, y1) in PDF points.
        chunk_index: Which chunk this evidence came from.
        confidence: Confidence score (0.0 - 1.0).
        verified: Did verification pass?
        verification_note: Explanation if verification failed or uncertain.
    """

    value: Any
    quote: str | None
    page: int | None
    bbox: tuple[float, float, float, float] | None
    chunk_index: int | None
    confidence: float
    verified: bool
    verification_note: str | None = None

    def __post_init__(self) -> None:
        """Validate evidence invariants."""
        # Confidence must be 0.0 - 1.0
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(
                f"confidence must be between 0.0 and 1.0, got {self.confidence}"
            )

        # Page must be >= 1 if set
        if self.page is not None and self.page < 1:
            raise ValueError(f"page must be >= 1, got {self.page}")

        # bbox must have 4 values if set
        if self.bbox is not None:
            if len(self.bbox) != 4:
                raise ValueError(f"bbox must have 4 values, got {len(self.bbox)}")

    def with_value(self, value: Any) -> Evidence:
        """Return new Evidence with updated value."""
        return Evidence(
            value=value,
            quote=self.quote,
            page=self.page,
            bbox=self.bbox,
            chunk_index=self.chunk_index,
            confidence=self.confidence,
            verified=self.verified,
            verification_note=self.verification_note,
        )

    def with_confidence(self, confidence: float) -> Evidence:
        """Return new Evidence with updated confidence."""
        return Evidence(
            value=self.value,
            quote=self.quote,
            page=self.page,
            bbox=self.bbox,
            chunk_index=self.chunk_index,
            confidence=confidence,
            verified=self.verified,
            verification_note=self.verification_note,
        )


@dataclass(frozen=True)
class EvidenceCandidate:
    """
    A potential evidence match before selection.

    Created during search phase when multiple chunks might contain
    the value. The best candidate is selected and converted to Evidence.

    Attributes:
        quote: Text that might support the value.
        page: Page number (1-indexed).
        bbox: Bounding box if available.
        chunk_index: Which chunk this came from.
        score: Similarity/relevance score from retriever.
        parsed_value: Value extracted from quote (if parsing succeeded).
    """

    quote: str
    chunk_index: int
    score: float
    page: int | None = None
    bbox: tuple[float, float, float, float] | None = None
    parsed_value: Any | None = None

    def __post_init__(self) -> None:
        """Validate candidate invariants."""
        if self.page is not None and self.page < 1:
            raise ValueError(f"page must be >= 1, got {self.page}")

        if self.bbox is not None and len(self.bbox) != 4:
            raise ValueError(f"bbox must have 4 values, got {len(self.bbox)}")

    def to_evidence(
        self,
        value: Any,
        verified: bool,
        confidence: float,
        note: str | None = None,
    ) -> Evidence:
        """
        Convert to final Evidence after selection.

        Args:
            value: The verified value.
            verified: Whether verification passed.
            confidence: Confidence score (0.0 - 1.0).
            note: Optional verification note.

        Returns:
            Evidence instance with this candidate's location info.
        """
        return Evidence(
            value=value,
            quote=self.quote,
            page=self.page,
            bbox=self.bbox,
            chunk_index=self.chunk_index,
            confidence=confidence,
            verified=verified,
            verification_note=note,
        )


@dataclass(frozen=True)
class VerificationResult:
    """
    Result of verifying a single field.

    Contains the outcome, values compared, evidence found,
    and all candidates considered (for debugging/audit).

    Attributes:
        path: Field path that was verified (e.g., 'services[PCP].copay').
        status: Outcome: MATCH, MISMATCH, NOT_FOUND, AMBIGUOUS, SKIPPED.
        extracted_value: What the extractor produced.
        found_value: What the verifier found in document (if different).
        evidence: Supporting evidence (if found).
        candidates: All candidates considered (for debugging/audit).
        search_queries: Queries used to find evidence (for debugging).
    """

    path: str
    status: VerificationStatus
    extracted_value: Any
    found_value: Any | None = None
    evidence: Evidence | None = None
    candidates: tuple[EvidenceCandidate, ...] = ()
    search_queries: tuple[str, ...] = ()

    @property
    def is_match(self) -> bool:
        """Check if verification passed."""
        return self.status == VerificationStatus.MATCH

    @property
    def is_mismatch(self) -> bool:
        """Check if extracted value was wrong."""
        return self.status == VerificationStatus.MISMATCH

    @property
    def needs_correction(self) -> bool:
        """Check if this field needs correction."""
        return self.status in (
            VerificationStatus.MISMATCH,
            VerificationStatus.AMBIGUOUS,
        )

    @property
    def is_failure(self) -> bool:
        """Check if verification failed (not found or ambiguous)."""
        return self.status in (
            VerificationStatus.NOT_FOUND,
            VerificationStatus.AMBIGUOUS,
            VerificationStatus.MISMATCH,
        )


def select_best_candidate(
    candidates: list[EvidenceCandidate],
) -> EvidenceCandidate | None:
    """
    Select best evidence candidate using deterministic tie-breaking.

    Priority (in order):
    1. Highest score
    2. Lowest page number (prefer earlier in document)
    3. Lowest chunk index (prefer earlier chunk)

    Args:
        candidates: List of potential matches.

    Returns:
        Best candidate, or None if list is empty.
    """
    if not candidates:
        return None

    def sort_key(c: EvidenceCandidate) -> tuple:
        # Negative score for descending sort
        # Use large number for None page/chunk to sort last
        return (
            -c.score,
            c.page if c.page is not None else float("inf"),
            c.chunk_index,
        )

    return min(candidates, key=sort_key)


def create_match_result(
    path: str,
    value: Any,
    evidence: Evidence,
    candidates: list[EvidenceCandidate] | None = None,
    search_queries: list[str] | None = None,
) -> VerificationResult:
    """
    Helper to create a MATCH verification result.

    Args:
        path: Field path.
        value: The matching value.
        evidence: Supporting evidence.
        candidates: All candidates considered.
        search_queries: Queries used.

    Returns:
        VerificationResult with MATCH status.
    """
    return VerificationResult(
        path=path,
        status=VerificationStatus.MATCH,
        extracted_value=value,
        found_value=value,
        evidence=evidence,
        candidates=tuple(candidates or []),
        search_queries=tuple(search_queries or []),
    )


def create_mismatch_result(
    path: str,
    extracted_value: Any,
    found_value: Any,
    evidence: Evidence,
    candidates: list[EvidenceCandidate] | None = None,
    search_queries: list[str] | None = None,
) -> VerificationResult:
    """
    Helper to create a MISMATCH verification result.

    Args:
        path: Field path.
        extracted_value: What extractor produced.
        found_value: What was actually in document.
        evidence: Supporting evidence for found value.
        candidates: All candidates considered.
        search_queries: Queries used.

    Returns:
        VerificationResult with MISMATCH status.
    """
    return VerificationResult(
        path=path,
        status=VerificationStatus.MISMATCH,
        extracted_value=extracted_value,
        found_value=found_value,
        evidence=evidence,
        candidates=tuple(candidates or []),
        search_queries=tuple(search_queries or []),
    )


def create_not_found_result(
    path: str,
    extracted_value: Any,
    candidates: list[EvidenceCandidate] | None = None,
    search_queries: list[str] | None = None,
) -> VerificationResult:
    """
    Helper to create a NOT_FOUND verification result.

    Args:
        path: Field path.
        extracted_value: What extractor produced.
        candidates: All candidates considered (may be empty).
        search_queries: Queries used.

    Returns:
        VerificationResult with NOT_FOUND status.
    """
    return VerificationResult(
        path=path,
        status=VerificationStatus.NOT_FOUND,
        extracted_value=extracted_value,
        found_value=None,
        evidence=None,
        candidates=tuple(candidates or []),
        search_queries=tuple(search_queries or []),
    )


def create_ambiguous_result(
    path: str,
    extracted_value: Any,
    candidates: list[EvidenceCandidate],
    search_queries: list[str] | None = None,
) -> VerificationResult:
    """
    Helper to create an AMBIGUOUS verification result.

    Args:
        path: Field path.
        extracted_value: What extractor produced.
        candidates: Conflicting candidates found.
        search_queries: Queries used.

    Returns:
        VerificationResult with AMBIGUOUS status.
    """
    return VerificationResult(
        path=path,
        status=VerificationStatus.AMBIGUOUS,
        extracted_value=extracted_value,
        found_value=None,
        evidence=None,
        candidates=tuple(candidates),
        search_queries=tuple(search_queries or []),
    )


__all__ = [
    "Evidence",
    "EvidenceCandidate",
    "VerificationResult",
    "VerificationStatus",
    "select_best_candidate",
    "create_match_result",
    "create_mismatch_result",
    "create_not_found_result",
    "create_ambiguous_result",
]
