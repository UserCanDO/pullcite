"""
Pullcite - Evidence-backed structured extraction.

Pullcite extracts structured data from documents while providing proof
of where each value came from in the source. Define schemas with
Django-style fields, each specifying how to search for evidence.

Example:
    >>> from pullcite import (
    ...     Document,
    ...     ExtractionSchema,
    ...     Extractor,
    ...     DecimalField,
    ...     StringField,
    ...     SearchType,
    ...     BM25Searcher,
    ... )
    >>> from pullcite.llms.anthropic import AnthropicLLM
    >>>
    >>> class Invoice(ExtractionSchema):
    ...     vendor = StringField(
    ...         query="vendor company name",
    ...         search_type=SearchType.BM25,
    ...     )
    ...     total = DecimalField(
    ...         query="total amount due invoice total",
    ...         search_type=SearchType.BM25,
    ...     )
    >>>
    >>> extractor = Extractor(
    ...     schema=Invoice,
    ...     llm=AnthropicLLM(),
    ...     searcher=BM25Searcher(),
    ... )
    >>>
    >>> doc = Document.from_file("invoice.pdf")
    >>> result = extractor.extract(doc)
    >>>
    >>> print(result.data.total)       # Decimal value
    >>> print(result.data.vendor)      # String value
    >>> print(result.status)           # VERIFIED, PARTIAL, or FAILED
    >>>
    >>> evidence = result.evidence_map["total"]
    >>> print(evidence.quote)          # Exact text from document
    >>> print(evidence.page)           # Page number
    >>> print(evidence.bbox)           # Bounding box coordinates
"""

__version__ = "0.2.1"

# Core types
from .core.document import Document
from .core.chunk import Chunk
from .core.chunker import (
    Chunker,
    SlidingWindowChunker,
    SentenceChunker,
    ParagraphChunker,
)

# Evidence types
from .core.evidence import (
    Evidence,
    EvidenceCandidate,
    VerificationResult,
    VerificationStatus,
)

# Results
from .core.result import (
    ExtractionResult,
    ExtractionStats,
    ExtractionFlag,
    ExtractionFlagType,
    ExtractionStatus,
)

# Schema and Fields
from .schema import (
    # Base
    ExtractionSchema,
    Field,
    SearchType,
    # Field types
    StringField,
    IntegerField,
    FloatField,
    DecimalField,
    CurrencyField,
    PercentField,
    BooleanField,
    DateField,
    ListField,
    EnumField,
    # Extractor
    SchemaExtractor as Extractor,
)

# Search
from .search import (
    BM25Searcher,
    HybridSearcher,
    SearchResult,
    Searcher,
)

__all__ = [
    # Version
    "__version__",
    # Core
    "Document",
    "Chunk",
    # Chunking
    "Chunker",
    "SlidingWindowChunker",
    "SentenceChunker",
    "ParagraphChunker",
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
    # Schema
    "ExtractionSchema",
    "Field",
    "SearchType",
    # Fields
    "StringField",
    "IntegerField",
    "FloatField",
    "DecimalField",
    "CurrencyField",
    "PercentField",
    "BooleanField",
    "DateField",
    "ListField",
    "EnumField",
    # Extractor
    "Extractor",
    # Search
    "BM25Searcher",
    "HybridSearcher",
    "SearchResult",
    "Searcher",
]
