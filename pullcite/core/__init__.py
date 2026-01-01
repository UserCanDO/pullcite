"""
Core types and utilities for Pullcite.

This module provides the fundamental building blocks for document processing,
extraction, and evidence tracking.
"""

from .chunk import Chunk
from .document import Document
from .extractor import Extractor
from .config import (
    ExtractorConfig,
    Hooks,
    DEFAULT_CONFIG,
    STRICT_CONFIG,
    FAST_CONFIG,
)
from .evidence import (
    Evidence,
    EvidenceCandidate,
    VerificationResult,
    VerificationStatus,
)
from .fields import (
    CriticalField,
    VerifierPolicy,
    Parsers,
    Comparators,
    INTEGER_POLICY,
    YES_NO_POLICY,
    STRING_POLICY,
    MONEY_POLICY,
    PERCENT_POLICY,
    LIMIT_POLICY,
    CONTAINS_POLICY,
)
from .result import (
    ExtractionResult,
    ExtractionStats,
    ExtractionFlag,
    ExtractionFlagType,
    ExtractionStatus,
)
from .paths import (
    get,
    set,
    delete,
    exists,
    expand,
    parse,
    validate,
    InvalidPathError,
    PathNotFoundError,
    AmbiguousPathError,
)

__all__ = [
    # Document processing
    "Chunk",
    "Document",
    "Extractor",
    # Configuration
    "ExtractorConfig",
    "Hooks",
    "DEFAULT_CONFIG",
    "STRICT_CONFIG",
    "FAST_CONFIG",
    # Evidence
    "Evidence",
    "EvidenceCandidate",
    "VerificationResult",
    "VerificationStatus",
    # Fields
    "CriticalField",
    "VerifierPolicy",
    "Parsers",
    "Comparators",
    "INTEGER_POLICY",
    "YES_NO_POLICY",
    "STRING_POLICY",
    "MONEY_POLICY",
    "PERCENT_POLICY",
    "LIMIT_POLICY",
    "CONTAINS_POLICY",
    # Results
    "ExtractionResult",
    "ExtractionStats",
    "ExtractionFlag",
    "ExtractionFlagType",
    "ExtractionStatus",
    # Paths
    "get",
    "set",
    "delete",
    "exists",
    "expand",
    "parse",
    "validate",
    "InvalidPathError",
    "PathNotFoundError",
    "AmbiguousPathError",
]
