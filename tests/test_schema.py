"""Tests for the schema module."""

from decimal import Decimal

import pytest

from pullcite.schema import (
    ExtractionSchema,
    Field,
    SearchType,
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


class TestSearchType:
    """Tests for SearchType enum."""

    def test_search_type_values(self):
        assert SearchType.BM25.value == "bm25"
        assert SearchType.SEMANTIC.value == "semantic"
        assert SearchType.HYBRID.value == "hybrid"


class TestExtractionSchema:
    """Tests for ExtractionSchema base class."""

    def test_schema_definition(self):
        """Test basic schema definition with fields."""

        class Invoice(ExtractionSchema):
            vendor = StringField(query="vendor name")
            total = DecimalField(query="total amount")

        fields = Invoice.get_fields()
        assert "vendor" in fields
        assert "total" in fields
        assert isinstance(fields["vendor"], StringField)
        assert isinstance(fields["total"], DecimalField)

    def test_field_name_assignment(self):
        """Test that field names are assigned correctly."""

        class Doc(ExtractionSchema):
            my_field = StringField(query="test")

        field = Doc.get_field("my_field")
        assert field is not None
        assert field.name == "my_field"
        assert field.label == "My Field"  # Auto-generated

    def test_custom_label(self):
        """Test custom field label."""

        class Doc(ExtractionSchema):
            total = DecimalField(query="total", label="Invoice Total")

        field = Doc.get_field("total")
        assert field.label == "Invoice Total"

    def test_required_fields(self):
        """Test filtering required fields."""

        class Doc(ExtractionSchema):
            required_field = StringField(query="req", required=True)
            optional_field = StringField(query="opt", required=False)

        required = Doc.get_required_fields()
        assert "required_field" in required
        assert "optional_field" not in required

    def test_search_type_filtering(self):
        """Test filtering fields by search type."""

        class Doc(ExtractionSchema):
            bm25_field = StringField(query="a", search_type=SearchType.BM25)
            semantic_field = StringField(query="b", search_type=SearchType.SEMANTIC)
            hybrid_field = StringField(query="c", search_type=SearchType.HYBRID)

        bm25 = Doc.get_bm25_fields()
        assert "bm25_field" in bm25
        assert "semantic_field" not in bm25

        semantic = Doc.get_semantic_fields()
        assert "semantic_field" in semantic
        assert "bm25_field" not in semantic

        hybrid = Doc.get_hybrid_fields()
        assert "hybrid_field" in hybrid

    def test_to_json_schema(self):
        """Test JSON schema generation."""

        class Doc(ExtractionSchema):
            name = StringField(query="name", description="The name")
            count = IntegerField(query="count", required=False)

        schema = Doc.to_json_schema()

        assert schema["type"] == "object"
        assert "name" in schema["properties"]
        assert "count" in schema["properties"]
        assert "name" in schema["required"]
        assert "count" not in schema["required"]
        assert schema["properties"]["name"]["type"] == "string"
        assert schema["properties"]["count"]["type"] == "integer"

    def test_from_dict(self):
        """Test creating instance from dict."""

        class Doc(ExtractionSchema):
            name = StringField(query="name")
            value = IntegerField(query="value")

        instance = Doc.from_dict({"name": "Test", "value": 42})

        assert instance.name == "Test"
        assert instance.value == 42

    def test_to_dict(self):
        """Test converting instance to dict."""

        class Doc(ExtractionSchema):
            name = StringField(query="name")
            value = IntegerField(query="value")

        instance = Doc(name="Test", value=42)
        result = instance.to_dict()

        assert result == {"name": "Test", "value": 42}

    def test_inheritance(self):
        """Test schema inheritance."""

        class BaseDoc(ExtractionSchema):
            id = StringField(query="id")

        class ExtendedDoc(BaseDoc):
            extra = StringField(query="extra")

        fields = ExtendedDoc.get_fields()
        assert "id" in fields
        assert "extra" in fields


class TestStringField:
    """Tests for StringField."""

    def test_parse(self):
        field = StringField(query="test")
        assert field.parse("hello") == "hello"
        assert field.parse(123) == "123"
        assert field.parse(None) is None

    def test_compare_normalized(self):
        field = StringField(query="test", normalize=True)
        assert field.compare("Hello World", "hello world")
        assert field.compare("  spaced  ", "spaced")
        assert not field.compare("different", "text")

    def test_compare_exact(self):
        field = StringField(query="test", normalize=False)
        assert field.compare("Hello", "Hello")
        assert not field.compare("Hello", "hello")


class TestIntegerField:
    """Tests for IntegerField."""

    def test_parse_int(self):
        field = IntegerField(query="test")
        assert field.parse(42) == 42
        assert field.parse("42") == 42
        assert field.parse(42.0) == 42
        assert field.parse(None) is None

    def test_parse_with_text(self):
        field = IntegerField(query="test")
        assert field.parse("100 days") == 100
        assert field.parse("-5 items") == -5

    def test_parse_bool_returns_none(self):
        field = IntegerField(query="test")
        assert field.parse(True) is None
        assert field.parse(False) is None

    def test_parse_non_integer_float(self):
        field = IntegerField(query="test")
        assert field.parse(42.5) is None


class TestFloatField:
    """Tests for FloatField."""

    def test_parse(self):
        field = FloatField(query="test")
        assert field.parse(3.14) == 3.14
        assert field.parse("3.14") == 3.14
        assert field.parse(42) == 42.0

    def test_compare_with_tolerance(self):
        field = FloatField(query="test", tolerance=0.01)
        assert field.compare(1.0, 1.005)
        assert not field.compare(1.0, 1.02)


class TestDecimalField:
    """Tests for DecimalField."""

    def test_parse_various_formats(self):
        field = DecimalField(query="test")
        assert field.parse("100.50") == Decimal("100.50")
        assert field.parse("1,000.50") == Decimal("1000.50")
        assert field.parse("$1,000.50") == Decimal("1000.50")
        assert field.parse(100) == Decimal("100")
        assert field.parse(100.5) == Decimal("100.5")

    def test_compare_with_tolerance(self):
        field = DecimalField(query="test", tolerance="0.01")
        assert field.compare(Decimal("100.00"), Decimal("100.005"))
        assert not field.compare(Decimal("100.00"), Decimal("100.02"))


class TestCurrencyField:
    """Tests for CurrencyField."""

    def test_parse_various_formats(self):
        field = CurrencyField(query="test")
        assert field.parse("$500") == Decimal("500")
        assert field.parse("$1,500.00") == Decimal("1500.00")
        assert field.parse("1500 USD") == Decimal("1500")
        assert field.parse("500 dollars") == Decimal("500")


class TestPercentField:
    """Tests for PercentField."""

    def test_parse_percentage_formats(self):
        field = PercentField(query="test")
        assert field.parse("30%") == 30.0
        assert field.parse("30 percent") == 30.0
        assert field.parse(30) == 30.0
        assert field.parse(0.30) == 30.0  # Decimal form

    def test_compare_with_tolerance(self):
        field = PercentField(query="test", tolerance=0.1)
        assert field.compare(30.0, 30.05)
        assert not field.compare(30.0, 30.5)


class TestBooleanField:
    """Tests for BooleanField."""

    def test_parse_various_formats(self):
        field = BooleanField(query="test")
        assert field.parse("yes") is True
        assert field.parse("no") is False
        assert field.parse("true") is True
        assert field.parse("false") is False
        assert field.parse(True) is True
        assert field.parse(1) is True
        assert field.parse(0) is False


class TestDateField:
    """Tests for DateField."""

    def test_parse(self):
        field = DateField(query="test")
        assert field.parse("2024-01-15") == "2024-01-15"
        assert field.parse(" 2024-01-15 ") == "2024-01-15"


class TestListField:
    """Tests for ListField."""

    def test_parse_list(self):
        field = ListField(query="test")
        assert field.parse(["a", "b", "c"]) == ["a", "b", "c"]

    def test_parse_comma_separated(self):
        field = ListField(query="test")
        assert field.parse("a, b, c") == ["a", "b", "c"]

    def test_parse_with_item_field(self):
        field = ListField(query="test", item_field=IntegerField(query="item"))
        assert field.parse(["1", "2", "3"]) == [1, 2, 3]

    def test_compare(self):
        field = ListField(query="test")
        assert field.compare(["a", "b"], ["a", "b"])
        assert not field.compare(["a", "b"], ["a", "c"])
        assert not field.compare(["a", "b"], ["a"])


class TestEnumField:
    """Tests for EnumField."""

    def test_parse_valid_choice(self):
        field = EnumField(query="test", choices=["bronze", "silver", "gold"])
        assert field.parse("gold") == "gold"
        assert field.parse("GOLD") == "gold"
        assert field.parse("  Gold  ") == "gold"

    def test_parse_invalid_choice(self):
        field = EnumField(query="test", choices=["bronze", "silver", "gold"])
        assert field.parse("platinum") is None

    def test_requires_choices(self):
        with pytest.raises(ValueError, match="requires at least one choice"):
            EnumField(query="test", choices=[])

    def test_json_schema(self):
        field = EnumField(query="test", choices=["a", "b"])
        schema = field.to_json_schema()
        assert schema["type"] == "string"
        assert schema["enum"] == ["a", "b"]


class TestFieldValidation:
    """Tests for field validation."""

    def test_empty_query_raises(self):
        with pytest.raises(ValueError, match="query cannot be empty"):
            StringField(query="")

    def test_whitespace_query_raises(self):
        with pytest.raises(ValueError, match="query cannot be empty"):
            StringField(query="   ")
