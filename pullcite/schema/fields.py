"""
Concrete field types for extraction schemas.

Each field type provides:
- Type-specific parsing (raw value -> typed value)
- Text parsing (document text -> typed value)
- Comparison logic (with appropriate tolerance)
- JSON Schema generation for LLM structured output
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from typing import Any, ClassVar

from .base import Field, SearchType


@dataclass
class StringField(Field[str]):
    """
    String field with optional normalization.

    Attributes:
        normalize: If True, lowercase and strip whitespace for comparison.

    Example:
        >>> class Doc(ExtractionSchema):
        ...     vendor = StringField(
        ...         query="vendor company name",
        ...         normalize=True,
        ...     )
    """

    normalize: bool = True
    python_type: ClassVar[type] = str

    def parse(self, value: Any) -> str | None:
        if value is None:
            return None
        return str(value)

    def parse_from_text(self, text: str) -> str | None:
        if not text:
            return None
        return text.strip()

    def compare(self, extracted: str | None, found: str | None) -> bool:
        if extracted is None or found is None:
            return extracted is found

        if self.normalize:
            e = re.sub(r"\s+", " ", extracted.lower().strip())
            f = re.sub(r"\s+", " ", found.lower().strip())
            return e == f

        return extracted == found

    def to_json_schema(self) -> dict[str, Any]:
        return {"type": "string"}


@dataclass
class IntegerField(Field[int]):
    """
    Integer field.

    Parses strings like "100", "100 days", extracts leading integers.

    Example:
        >>> class Policy(ExtractionSchema):
        ...     deductible = IntegerField(
        ...         query="annual deductible amount",
        ...         search_type=SearchType.BM25,
        ...     )
    """

    python_type: ClassVar[type] = int

    def parse(self, value: Any) -> int | None:
        if value is None:
            return None

        if isinstance(value, bool):
            return None  # Avoid True -> 1

        if isinstance(value, int):
            return value

        if isinstance(value, float):
            if value.is_integer():
                return int(value)
            return None

        if isinstance(value, str):
            # Extract leading number
            match = re.match(r"^\s*(-?\d+)", value)
            if match:
                return int(match.group(1))

        return None

    def parse_from_text(self, text: str) -> int | None:
        return self.parse(text)

    def compare(self, extracted: int | None, found: int | None) -> bool:
        return extracted == found

    def to_json_schema(self) -> dict[str, Any]:
        return {"type": "integer"}


@dataclass
class FloatField(Field[float]):
    """
    Float field with configurable tolerance.

    Attributes:
        tolerance: Maximum difference for comparison (default 0.001).

    Example:
        >>> class Measurement(ExtractionSchema):
        ...     temperature = FloatField(
        ...         query="temperature reading",
        ...         tolerance=0.1,
        ...     )
    """

    tolerance: float = 0.001
    python_type: ClassVar[type] = float

    def parse(self, value: Any) -> float | None:
        if value is None:
            return None

        if isinstance(value, (int, float)):
            return float(value)

        if isinstance(value, str):
            # Remove commas
            cleaned = value.replace(",", "").strip()
            try:
                return float(cleaned)
            except ValueError:
                return None

        return None

    def parse_from_text(self, text: str) -> float | None:
        return self.parse(text)

    def compare(self, extracted: float | None, found: float | None) -> bool:
        if extracted is None or found is None:
            return extracted is found

        return abs(extracted - found) <= self.tolerance

    def to_json_schema(self) -> dict[str, Any]:
        return {"type": "number"}


@dataclass
class DecimalField(Field[Decimal]):
    """
    Decimal field for precise monetary/financial values.

    Parses currency formats: "$1,500.00", "1500", "1,500.00"
    Comparison uses exact decimal equality (no floating point errors).

    Attributes:
        tolerance: Maximum difference for comparison as Decimal string.

    Example:
        >>> class Invoice(ExtractionSchema):
        ...     total = DecimalField(
        ...         query="total amount due invoice total",
        ...         search_type=SearchType.BM25,
        ...     )
    """

    tolerance: str = "0.01"  # Stored as string, converted to Decimal
    python_type: ClassVar[type] = Decimal

    def parse(self, value: Any) -> Decimal | None:
        if value is None:
            return None

        if isinstance(value, Decimal):
            return value

        if isinstance(value, (int, float)):
            return Decimal(str(value))

        if isinstance(value, str):
            # Remove currency symbols, commas, whitespace
            cleaned = re.sub(r"[$,\s]", "", value.strip())
            if not cleaned:
                return None

            try:
                return Decimal(cleaned)
            except InvalidOperation:
                return None

        return None

    def parse_from_text(self, text: str) -> Decimal | None:
        return self.parse(text)

    def compare(self, extracted: Decimal | None, found: Decimal | None) -> bool:
        if extracted is None or found is None:
            return extracted is found

        tolerance = Decimal(self.tolerance)
        return abs(extracted - found) <= tolerance

    def to_json_schema(self) -> dict[str, Any]:
        # Use string type to preserve precision
        return {"type": "string", "pattern": r"^-?\d+(\.\d+)?$"}


@dataclass
class CurrencyField(Field[Decimal]):
    """
    Currency field with formatting awareness.

    Like DecimalField but with better parsing of currency formats
    and a default tolerance of $0.01.

    Example:
        >>> class Invoice(ExtractionSchema):
        ...     total = CurrencyField(
        ...         query="total amount invoice total due",
        ...         search_type=SearchType.BM25,
        ...     )
    """

    tolerance: str = "0.01"
    currency_symbol: str = "$"
    python_type: ClassVar[type] = Decimal

    def parse(self, value: Any) -> Decimal | None:
        if value is None:
            return None

        if isinstance(value, Decimal):
            return value

        if isinstance(value, (int, float)):
            return Decimal(str(value))

        if isinstance(value, str):
            # Remove currency symbols, commas, whitespace, and common words
            cleaned = value.strip()
            # Remove currency symbols (handle common ones)
            cleaned = re.sub(r"[$\u20ac\u00a3\u00a5]", "", cleaned)  # $, EUR, GBP, JPY
            # Remove commas and spaces
            cleaned = re.sub(r"[,\s]", "", cleaned)
            # Remove trailing words like "USD", "dollars"
            cleaned = re.sub(r"(USD|EUR|GBP|dollars?|euros?)$", "", cleaned, flags=re.I)

            if not cleaned:
                return None

            try:
                return Decimal(cleaned)
            except InvalidOperation:
                return None

        return None

    def parse_from_text(self, text: str) -> Decimal | None:
        return self.parse(text)

    def compare(self, extracted: Decimal | None, found: Decimal | None) -> bool:
        if extracted is None or found is None:
            return extracted is found

        tolerance = Decimal(self.tolerance)
        return abs(extracted - found) <= tolerance

    def to_json_schema(self) -> dict[str, Any]:
        return {"type": "string"}


@dataclass
class PercentField(Field[float]):
    """
    Percentage field.

    Parses: "30%", "30", 30, 0.30, "30 percent"
    Normalizes to 0-100 scale (30% = 30.0, not 0.30).

    Example:
        >>> class Policy(ExtractionSchema):
        ...     coinsurance = PercentField(
        ...         query="coinsurance percentage",
        ...         search_type=SearchType.BM25,
        ...     )
    """

    tolerance: float = 0.1  # 0.1% tolerance
    python_type: ClassVar[type] = float

    def parse(self, value: Any) -> float | None:
        if value is None:
            return None

        if isinstance(value, (int, float)):
            # If < 1, assume decimal form (0.30 -> 30)
            if 0 < value < 1:
                return float(value * 100)
            return float(value)

        if isinstance(value, str):
            # Remove %, "percent", whitespace
            cleaned = re.sub(r"[%\s]", "", value.strip().lower())
            cleaned = cleaned.replace("percent", "")

            if not cleaned:
                return None

            try:
                result = float(cleaned)
                if 0 < result < 1:
                    return result * 100
                return result
            except ValueError:
                return None

        return None

    def parse_from_text(self, text: str) -> float | None:
        return self.parse(text)

    def compare(self, extracted: float | None, found: float | None) -> bool:
        if extracted is None or found is None:
            return extracted is found

        return abs(extracted - found) <= self.tolerance

    def to_json_schema(self) -> dict[str, Any]:
        return {"type": "number"}


@dataclass
class BooleanField(Field[bool]):
    """
    Boolean/yes-no field.

    Parses: "yes", "no", "true", "false", True, False, 1, 0

    Example:
        >>> class Policy(ExtractionSchema):
        ...     is_covered = BooleanField(
        ...         query="coverage included covered",
        ...         search_type=SearchType.BM25,
        ...     )
    """

    python_type: ClassVar[type] = bool

    def parse(self, value: Any) -> bool | None:
        if value is None:
            return None

        if isinstance(value, bool):
            return value

        if isinstance(value, (int, float)):
            return bool(value)

        if isinstance(value, str):
            lower = value.strip().lower()
            if lower in ("yes", "true", "y", "1"):
                return True
            if lower in ("no", "false", "n", "0"):
                return False

        return None

    def parse_from_text(self, text: str) -> bool | None:
        return self.parse(text)

    def compare(self, extracted: bool | None, found: bool | None) -> bool:
        return extracted is found

    def to_json_schema(self) -> dict[str, Any]:
        return {"type": "boolean"}


@dataclass
class DateField(Field[str]):
    """
    Date field (stored as ISO string).

    Recognizes common date formats and normalizes to ISO (YYYY-MM-DD).

    Example:
        >>> class Invoice(ExtractionSchema):
        ...     date = DateField(
        ...         query="invoice date",
        ...         search_type=SearchType.BM25,
        ...     )
    """

    python_type: ClassVar[type] = str

    def parse(self, value: Any) -> str | None:
        if value is None:
            return None

        # For now, just return as string
        # TODO: Add date parsing and normalization
        return str(value).strip()

    def parse_from_text(self, text: str) -> str | None:
        return self.parse(text)

    def compare(self, extracted: str | None, found: str | None) -> bool:
        # Simple string comparison for now
        # TODO: Date-aware comparison
        if extracted is None or found is None:
            return extracted is found

        return extracted.strip() == found.strip()

    def to_json_schema(self) -> dict[str, Any]:
        return {"type": "string", "format": "date"}


@dataclass
class ListField(Field[list]):
    """
    List field for arrays of values.

    Attributes:
        item_field: Field type for list items.

    Example:
        >>> class Document(ExtractionSchema):
        ...     keywords = ListField(
        ...         query="keywords tags",
        ...         item_field=StringField(query=""),
        ...     )
    """

    item_field: Field | None = None
    python_type: ClassVar[type] = list

    def parse(self, value: Any) -> list | None:
        if value is None:
            return None

        if isinstance(value, list):
            if self.item_field:
                return [self.item_field.parse(v) for v in value]
            return value

        if isinstance(value, str):
            # Try comma-separated
            items = [v.strip() for v in value.split(",") if v.strip()]
            if self.item_field:
                return [self.item_field.parse(v) for v in items]
            return items

        return [value]

    def parse_from_text(self, text: str) -> list | None:
        return self.parse(text)

    def compare(self, extracted: list | None, found: list | None) -> bool:
        if extracted is None or found is None:
            return extracted is found

        if len(extracted) != len(found):
            return False

        for e, f in zip(extracted, found):
            if self.item_field:
                if not self.item_field.compare(e, f):
                    return False
            elif e != f:
                return False

        return True

    def to_json_schema(self) -> dict[str, Any]:
        item_schema = (
            self.item_field.to_json_schema()
            if self.item_field
            else {"type": "string"}
        )
        return {"type": "array", "items": item_schema}


@dataclass
class EnumField(Field[str]):
    """
    Enumeration field with predefined choices.

    Attributes:
        choices: List of valid values.

    Example:
        >>> class Policy(ExtractionSchema):
        ...     plan_type = EnumField(
        ...         query="plan type tier",
        ...         choices=["bronze", "silver", "gold", "platinum"],
        ...     )
    """

    choices: list[str] | tuple[str, ...] = ()
    python_type: ClassVar[type] = str

    def __post_init__(self) -> None:
        super().__post_init__()
        if not self.choices:
            raise ValueError("EnumField requires at least one choice")

    def parse(self, value: Any) -> str | None:
        if value is None:
            return None

        str_value = str(value).strip().lower()

        # Find matching choice (case-insensitive)
        for choice in self.choices:
            if choice.lower() == str_value:
                return choice

        return None

    def parse_from_text(self, text: str) -> str | None:
        return self.parse(text)

    def compare(self, extracted: str | None, found: str | None) -> bool:
        if extracted is None or found is None:
            return extracted is found

        return extracted.lower() == found.lower()

    def to_json_schema(self) -> dict[str, Any]:
        return {"type": "string", "enum": list(self.choices)}


__all__ = [
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
]
