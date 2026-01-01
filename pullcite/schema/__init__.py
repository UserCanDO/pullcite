"""
Schema module for Django-style extraction field definitions.

This module provides a declarative way to define extraction schemas
where each field specifies its own search strategy.

Example:
    >>> from pullcite.schema import (
    ...     ExtractionSchema,
    ...     StringField,
    ...     DecimalField,
    ...     PercentField,
    ...     SearchType,
    ... )
    >>>
    >>> class Invoice(ExtractionSchema):
    ...     vendor = StringField(
    ...         query="vendor company name",
    ...         search_type=SearchType.SEMANTIC,
    ...         description="The company that issued the invoice",
    ...     )
    ...     total = DecimalField(
    ...         query="total amount due invoice total",
    ...         search_type=SearchType.BM25,
    ...         description="Total amount due on the invoice",
    ...     )
    ...     tax_rate = PercentField(
    ...         query="tax rate percentage",
    ...         search_type=SearchType.BM25,
    ...         required=False,
    ...     )
    >>>
    >>> # Get JSON schema for LLM
    >>> schema = Invoice.to_json_schema()
    >>>
    >>> # Get fields by search type
    >>> bm25_fields = Invoice.get_bm25_fields()
    >>> semantic_fields = Invoice.get_semantic_fields()
"""

from .base import (
    Field,
    SearchType,
    ExtractionSchema,
    ExtractionSchemaMeta,
)

from .fields import (
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
)

from .extractor import SchemaExtractor

__all__ = [
    # Base types
    "Field",
    "SearchType",
    "ExtractionSchema",
    "ExtractionSchemaMeta",
    # Field types
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
    "SchemaExtractor",
]
