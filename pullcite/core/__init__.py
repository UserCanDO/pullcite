"""
Core types and utilities for Pullcite.

This module provides the fundamental building blocks for document processing,
extraction, and evidence tracking.
"""

from .chunk import Chunk
from .chunker import (
    Chunker,
    ChunkConfig,
    SlidingWindowChunker,
    SentenceChunker,
    ParagraphChunker,
    DEFAULT_CHUNKER,
)
from .document import Document
from .evidence import (
    Evidence,
    EvidenceCandidate,
    VerificationResult,
    VerificationStatus,
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
    # Chunking
    "Chunker",
    "ChunkConfig",
    "SlidingWindowChunker",
    "SentenceChunker",
    "ParagraphChunker",
    "DEFAULT_CHUNKER",
    # Document processing
    "Chunk",
    "Document",
    # Evidence
    "Evidence",
    "EvidenceCandidate",
    "VerificationResult",
    "VerificationStatus",
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
