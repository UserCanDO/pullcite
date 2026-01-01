"""
Base types for the schema module.

This module provides Django-style field definitions for extraction schemas.
Each field can specify its own search query and search type (bm25 or semantic).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal
from enum import Enum
from typing import Any, Callable, ClassVar, Generic, TypeVar, get_type_hints

T = TypeVar("T")


class SearchType(str, Enum):
    """Search strategy for finding evidence."""

    BM25 = "bm25"
    """Keyword-based BM25 search (fast, no embeddings needed)."""

    SEMANTIC = "semantic"
    """Vector similarity search (requires embeddings)."""

    HYBRID = "hybrid"
    """Combined BM25 + semantic with rank fusion."""


@dataclass
class Field(Generic[T]):
    """
    Base field definition for extraction schemas.

    Fields define both the data type and how to search for evidence.
    Subclasses provide type-specific parsing and validation.

    Attributes:
        query: Search query to find this value in the document.
               Used by both extractor (for context) and verifier (for evidence).
        search_type: How to search: bm25, semantic, or hybrid.
        description: **Important** - Tells the LLM what this field means.
                    Without it, the LLM only sees the field name. Always provide
                    a clear description for better extraction accuracy.
        required: If True, extraction fails if value not found.
        label: Human-readable name for error messages.
        default: Default value if not found and not required.

    Example:
        >>> class Invoice(ExtractionSchema):
        ...     total = DecimalField(
        ...         query="total amount due invoice total",
        ...         search_type=SearchType.BM25,
        ...         description="The final total amount due on the invoice",
        ...     )
        ...     vendor = StringField(
        ...         query="vendor company supplier",
        ...         description="Name of the company that issued the invoice",
        ...     )
    """

    query: str
    search_type: SearchType = SearchType.BM25
    description: str | None = None  # Strongly recommended - helps LLM understand field
    required: bool = True
    label: str | None = None
    default: T | None = None

    # Set by __set_name__ when attached to a schema
    name: str = field(default="", init=False)
    _owner: type | None = field(default=None, init=False, repr=False)

    # Subclasses should set this
    python_type: ClassVar[type] = object

    def __set_name__(self, owner: type, name: str) -> None:
        """Called when field is assigned to a class attribute."""
        self.name = name
        self._owner = owner
        if self.label is None:
            # Auto-generate label from name: "invoice_total" -> "Invoice Total"
            self.label = name.replace("_", " ").title()

    def __post_init__(self) -> None:
        """Validate field configuration."""
        if not self.query or not self.query.strip():
            raise ValueError("query cannot be empty")

    def parse(self, value: Any) -> T | None:
        """
        Parse raw value to field's type.

        Override in subclasses for type-specific parsing.

        Args:
            value: Raw value from extraction.

        Returns:
            Parsed value or None if unparseable.
        """
        return value

    def parse_from_text(self, text: str) -> T | None:
        """
        Parse value from document text (for verification).

        Override in subclasses for type-specific text parsing.

        Args:
            text: Text snippet from document.

        Returns:
            Parsed value or None if unparseable.
        """
        return self.parse(text)

    def compare(self, extracted: T | None, found: T | None) -> bool:
        """
        Compare extracted value with found value.

        Override in subclasses for type-specific comparison.

        Args:
            extracted: Value from LLM extraction.
            found: Value found in document.

        Returns:
            True if values match.
        """
        return extracted == found

    def to_json_schema(self) -> dict[str, Any]:
        """
        Generate JSON schema for this field.

        Used to build the schema for LLM structured output.

        Returns:
            JSON Schema dict for this field's type.
        """
        return {"type": "string"}

    @property
    def json_schema_type(self) -> str:
        """JSON Schema type name."""
        return "string"


class ExtractionSchemaMeta(type):
    """Metaclass that collects Field definitions from class attributes."""

    def __new__(
        mcs,
        name: str,
        bases: tuple[type, ...],
        namespace: dict[str, Any],
    ) -> ExtractionSchemaMeta:
        # Collect fields from this class
        fields: dict[str, Field] = {}

        # Inherit fields from base classes
        for base in bases:
            if hasattr(base, "_fields"):
                fields.update(base._fields)

        # Collect new fields from namespace
        for attr_name, attr_value in namespace.items():
            if isinstance(attr_value, Field):
                fields[attr_name] = attr_value

        namespace["_fields"] = fields

        cls = super().__new__(mcs, name, bases, namespace)

        # Trigger __set_name__ for all fields
        for field_name, field_obj in fields.items():
            field_obj.__set_name__(cls, field_name)

        return cls


class ExtractionSchema(metaclass=ExtractionSchemaMeta):
    """
    Base class for defining extraction schemas.

    Subclass this and define fields as class attributes.
    The schema defines both what to extract and how to verify it.

    Example:
        >>> class Invoice(ExtractionSchema):
        ...     vendor = StringField(
        ...         query="vendor name company",
        ...         search_type=SearchType.SEMANTIC,
        ...     )
        ...     total = DecimalField(
        ...         query="total amount due",
        ...         search_type=SearchType.BM25,
        ...     )
        ...     date = StringField(
        ...         query="invoice date",
        ...         search_type=SearchType.BM25,
        ...     )
        ...
        >>> # Get all fields
        >>> Invoice.get_fields()
        {'vendor': StringField(...), 'total': DecimalField(...), ...}
        >>>
        >>> # Generate JSON schema for LLM
        >>> Invoice.to_json_schema()
        {'type': 'object', 'properties': {...}, 'required': [...]}
    """

    _fields: ClassVar[dict[str, Field]] = {}

    def __init__(self, **kwargs: Any) -> None:
        """
        Create schema instance with field values.

        Args:
            **kwargs: Field values keyed by field name.
        """
        for name, field_def in self._fields.items():
            value = kwargs.get(name, field_def.default)
            setattr(self, name, value)

    @classmethod
    def get_fields(cls) -> dict[str, Field]:
        """Get all field definitions."""
        return cls._fields.copy()

    @classmethod
    def get_field(cls, name: str) -> Field | None:
        """Get a specific field by name."""
        return cls._fields.get(name)

    @classmethod
    def get_required_fields(cls) -> dict[str, Field]:
        """Get only required fields."""
        return {k: v for k, v in cls._fields.items() if v.required}

    @classmethod
    def get_bm25_fields(cls) -> dict[str, Field]:
        """Get fields that use BM25 search."""
        return {
            k: v for k, v in cls._fields.items()
            if v.search_type == SearchType.BM25
        }

    @classmethod
    def get_semantic_fields(cls) -> dict[str, Field]:
        """Get fields that use semantic search."""
        return {
            k: v for k, v in cls._fields.items()
            if v.search_type == SearchType.SEMANTIC
        }

    @classmethod
    def get_hybrid_fields(cls) -> dict[str, Field]:
        """Get fields that use hybrid search."""
        return {
            k: v for k, v in cls._fields.items()
            if v.search_type == SearchType.HYBRID
        }

    @classmethod
    def to_json_schema(cls) -> dict[str, Any]:
        """
        Generate JSON Schema for LLM structured output.

        Returns:
            JSON Schema dict describing the extraction target.
        """
        properties = {}
        required = []

        for name, field_def in cls._fields.items():
            prop_schema = field_def.to_json_schema()
            if field_def.description:
                prop_schema["description"] = field_def.description
            properties[name] = prop_schema

            if field_def.required:
                required.append(name)

        return {
            "type": "object",
            "properties": properties,
            "required": required,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ExtractionSchema:
        """
        Create instance from dict, parsing values through fields.

        Args:
            data: Raw extracted data.

        Returns:
            Schema instance with parsed values.
        """
        parsed = {}
        for name, field_def in cls._fields.items():
            raw_value = data.get(name)
            if raw_value is not None:
                parsed[name] = field_def.parse(raw_value)
            else:
                parsed[name] = field_def.default
        return cls(**parsed)

    def to_dict(self) -> dict[str, Any]:
        """Convert instance to dict."""
        return {name: getattr(self, name) for name in self._fields}

    def __repr__(self) -> str:
        field_strs = [
            f"{name}={getattr(self, name)!r}"
            for name in self._fields
        ]
        return f"{self.__class__.__name__}({', '.join(field_strs)})"


__all__ = [
    "Field",
    "SearchType",
    "ExtractionSchema",
    "ExtractionSchemaMeta",
]
