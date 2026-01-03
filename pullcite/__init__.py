"""
Pullcite - Evidence-backed structured extraction.

Pullcite extracts structured data from documents while providing proof
of where each value came from in the source.

Example:
    >>> from pydantic import BaseModel
    >>> from pullcite import Extractor, Document, CriticalField
    >>> from pullcite.llms.anthropic import AnthropicLLM
    >>> from pullcite.retrieval.memory import MemoryRetriever
    >>> from pullcite.embeddings.openai import OpenAIEmbedder
    >>>
    >>> class Invoice(BaseModel):
    ...     vendor: str
    ...     total: float
    ...     date: str
    ...
    >>> # Set up components
    >>> llm = AnthropicLLM()
    >>> embedder = OpenAIEmbedder()
    >>> retriever = MemoryRetriever(_embedder=embedder)
    >>>
    >>> # Define critical fields that need verification
    >>> critical_fields = [
    ...     CriticalField(
    ...         path="total",
    ...         label="Invoice Total",
    ...         search_query="total amount due",
    ...     ),
    ... ]
    >>>
    >>> # Create extractor
    >>> extractor = Extractor(
    ...     schema=Invoice,
    ...     llm=llm,
    ...     retriever=retriever,
    ...     critical_fields=critical_fields,
    ... )
    >>>
    >>> # Extract from document
    >>> doc = Document.from_file("invoice.pdf")
    >>> result = extractor.extract(doc)
    >>>
    >>> # Access extracted data
    >>> print(result.data.total)
    >>> print(result.status)  # VERIFIED, PARTIAL, or FAILED
    >>>
    >>> # Get evidence for a field
    >>> evidence = result.evidence("total")
    >>> print(evidence.quote)  # Exact text from document
    >>> print(evidence.page)   # Page number
"""

__version__ = "0.0.5"

# Core types
from .core.document import Document
from .core.chunk import Chunk
from .core.extractor import Extractor

# Evidence types
from .core.evidence import (
    Evidence,
    EvidenceCandidate,
    VerificationResult,
    VerificationStatus,
)

# Field definitions
from .core.fields import (
    CriticalField,
    VerifierPolicy,
    Parsers,
    Comparators,
)

# Results
from .core.result import (
    ExtractionResult,
    ExtractionStats,
    ExtractionFlag,
    ExtractionFlagType,
    ExtractionStatus,
)

# Configuration
from .core.config import (
    ExtractorConfig,
    Hooks,
    DEFAULT_CONFIG,
    STRICT_CONFIG,
    FAST_CONFIG,
)

# Path utilities
from .core.paths import (
    get,
    set,
    delete,
    exists,
    expand,
    parse,
    validate,
    InvalidPathError,
)

# Strategy
from .pipeline.strategy import (
    ExtractionStrategy,
    DefaultStrategy,
    StrategyContext,
)

__all__ = [
    # Version
    "__version__",
    # Core
    "Document",
    "Chunk",
    "Extractor",
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
    # Results
    "ExtractionResult",
    "ExtractionStats",
    "ExtractionFlag",
    "ExtractionFlagType",
    "ExtractionStatus",
    # Config
    "ExtractorConfig",
    "Hooks",
    "DEFAULT_CONFIG",
    "STRICT_CONFIG",
    "FAST_CONFIG",
    # Paths
    "get",
    "set",
    "delete",
    "exists",
    "expand",
    "parse",
    "validate",
    "InvalidPathError",
    # Strategy
    "ExtractionStrategy",
    "DefaultStrategy",
    "StrategyContext",
]
