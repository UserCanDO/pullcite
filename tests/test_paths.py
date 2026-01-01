"""
Test cases for paths.py
"""

import pytest
from pullcite.core.paths import (
    parse,
    validate,
    get,
    get_strict,
    set,
    delete,
    exists,
    expand,
    expand_with_values,
    resolve_key_selector,
    get_item_key,
    join,
    parent,
    leaf,
    get_many,
    set_many,
    PathError,
    PathNotFoundError,
    AmbiguousPathError,
    InvalidPathError,
    ParsedPath,
    PathSegment,
)


class TestParse:
    """Test path parsing."""

    def test_simple_field(self):
        p = parse("vendor")
        assert len(p) == 1
        assert p.segments[0].field == "vendor"
        assert p.segments[0].selector is None
        assert p.segments[0].selector_type is None

    def test_nested_field(self):
        p = parse("vendor.address.city")
        assert len(p) == 3
        assert [s.field for s in p.segments] == ["vendor", "address", "city"]
        assert all(s.selector is None for s in p.segments)

    def test_index_selector(self):
        p = parse("items[0].price")
        assert p.segments[0].field == "items"
        assert p.segments[0].selector == 0
        assert p.segments[0].selector_type == "index"
        assert p.segments[1].field == "price"

    def test_large_index(self):
        p = parse("items[999].price")
        assert p.segments[0].selector == 999
        assert p.segments[0].selector_type == "index"

    def test_key_selector(self):
        p = parse("services[PCP_VISIT].copay")
        assert p.segments[0].field == "services"
        assert p.segments[0].selector == "PCP_VISIT"
        assert p.segments[0].selector_type == "key"

    def test_key_selector_with_leading_digit(self):
        """Keys starting with digits but containing letters are keys, not indices."""
        p = parse("items[123ABC].price")
        assert p.segments[0].selector == "123ABC"
        assert p.segments[0].selector_type == "key"

    def test_pure_digits_is_index(self):
        """Pure digit selectors are always index access."""
        p = parse("items[123].price")
        assert p.segments[0].selector == 123
        assert p.segments[0].selector_type == "index"

    def test_wildcard(self):
        p = parse("items[*].price")
        assert p.segments[0].selector == "*"
        assert p.segments[0].selector_type == "wildcard"

    def test_complex_path(self):
        p = parse("services[PCP_VISIT].coverage_by_tier[INN].copay")
        assert len(p) == 3
        assert p.segments[0].field == "services"
        assert p.segments[0].selector == "PCP_VISIT"
        assert p.segments[1].field == "coverage_by_tier"
        assert p.segments[1].selector == "INN"
        assert p.segments[2].field == "copay"

    def test_underscore_start(self):
        p = parse("_private.field")
        assert p.segments[0].field == "_private"

    def test_has_wildcard_property(self):
        assert parse("items[*].price").has_wildcard
        assert not parse("items[0].price").has_wildcard
        assert not parse("items.price").has_wildcard

    def test_parent_property(self):
        p = parse("vendor.address.city")
        parent = p.parent
        assert parent is not None
        assert str(parent) == "vendor.address"

        p2 = parse("vendor")
        assert p2.parent is None

    def test_leaf_property(self):
        p = parse("vendor.address.city")
        assert str(p.leaf) == "city"

        p2 = parse("items[0].price")
        assert str(p2.leaf) == "price"

    def test_str_roundtrip(self):
        paths = [
            "vendor",
            "vendor.name",
            "items[0].price",
            "services[PCP_VISIT].copay",
            "items[*].price",
        ]
        for path in paths:
            assert str(parse(path)) == path

    # Invalid path tests

    def test_invalid_empty(self):
        with pytest.raises(InvalidPathError) as exc:
            parse("")
        assert "empty" in exc.value.message.lower()

    def test_invalid_trailing_dot(self):
        with pytest.raises(InvalidPathError):
            parse("vendor.")

    def test_invalid_leading_dot(self):
        with pytest.raises(InvalidPathError):
            parse(".vendor")

    def test_invalid_double_dot(self):
        with pytest.raises(InvalidPathError):
            parse("vendor..name")

    def test_invalid_empty_selector(self):
        with pytest.raises(InvalidPathError) as exc:
            parse("items[].price")
        assert "empty" in exc.value.message.lower()

    def test_invalid_unclosed_bracket(self):
        with pytest.raises(InvalidPathError):
            parse("items[0.price")

    def test_invalid_start_with_digit(self):
        with pytest.raises(InvalidPathError):
            parse("123field")

    def test_invalid_selector_chars(self):
        with pytest.raises(InvalidPathError):
            parse("items[foo-bar].price")


class TestValidate:
    """Test validate() helper."""

    def test_valid_paths(self):
        assert validate("vendor.name") == (True, None)
        assert validate("items[0].price") == (True, None)
        assert validate("services[PCP].copay") == (True, None)

    def test_invalid_paths(self):
        valid, msg = validate("")
        assert valid is False
        assert msg is not None

        valid, msg = validate("items[].price")
        assert valid is False


class TestGet:
    """Test get() - soft getter that returns default on missing."""

    def test_simple(self):
        data = {"vendor": {"name": "Acme"}}
        assert get(data, "vendor.name") == "Acme"

    def test_nested(self):
        data = {"a": {"b": {"c": {"d": 42}}}}
        assert get(data, "a.b.c.d") == 42

    def test_missing_returns_default(self):
        data = {"vendor": {}}
        assert get(data, "vendor.name") is None
        assert get(data, "vendor.name", default="N/A") == "N/A"

    def test_missing_intermediate_returns_default(self):
        data = {"vendor": {}}
        assert get(data, "vendor.address.city") is None

    def test_none_value_is_not_missing(self):
        data = {"vendor": {"name": None}}
        assert get(data, "vendor.name") is None
        assert get(data, "vendor.name", default="MISSING") is None  # None is the value

    def test_index_access(self):
        data = {"items": [{"price": 10}, {"price": 20}]}
        assert get(data, "items[0].price") == 10
        assert get(data, "items[1].price") == 20

    def test_index_out_of_bounds_returns_default(self):
        data = {"items": [{"price": 10}]}
        assert get(data, "items[5].price") is None
        assert get(data, "items[5].price", default=-1) == -1

    def test_key_access(self):
        data = {
            "services": [
                {"service_code": "PCP", "copay": 25},
                {"service_code": "ER", "copay": 100},
            ]
        }
        assert get(data, "services[PCP].copay") == 25
        assert get(data, "services[ER].copay") == 100

    def test_key_not_found_returns_default(self):
        data = {"services": [{"service_code": "PCP"}]}
        assert get(data, "services[UNKNOWN].copay") is None

    def test_wrong_type_returns_default(self):
        data = {"items": "not a dict"}
        assert get(data, "items.field") is None

    def test_list_without_selector_returns_default(self):
        data = {"items": [1, 2, 3]}
        assert get(data, "items.field") is None


class TestGetStrict:
    """Test get_strict() - raises on missing."""

    def test_found(self):
        data = {"vendor": {"name": "Acme"}}
        assert get_strict(data, "vendor.name") == "Acme"

    def test_missing_raises(self):
        data = {"vendor": {}}
        with pytest.raises(PathNotFoundError) as exc:
            get_strict(data, "vendor.name")
        assert "name" in str(exc.value)
        assert exc.value.path == "vendor.name"

    def test_missing_intermediate_raises(self):
        data = {"vendor": {}}
        with pytest.raises(PathNotFoundError):
            get_strict(data, "vendor.address.city")

    def test_key_not_found_raises(self):
        data = {"services": [{"service_code": "PCP"}]}
        with pytest.raises(PathNotFoundError):
            get_strict(data, "services[UNKNOWN].copay")

    def test_index_out_of_bounds_raises(self):
        data = {"items": [{"price": 10}]}
        with pytest.raises(PathNotFoundError) as exc:
            get_strict(data, "items[5].price")
        assert "out of bounds" in exc.value.message.lower()


class TestAmbiguous:
    """Test ambiguous key handling."""

    def test_ambiguous_key_raises_in_get(self):
        data = {
            "tiers": [
                {"name": "INN", "type": "primary"},
                {"name": "INN", "type": "secondary"},
            ]
        }
        with pytest.raises(AmbiguousPathError) as exc:
            get(data, "tiers[INN].type")
        assert exc.value.matches == [0, 1]
        assert exc.value.selector == "INN"

    def test_ambiguous_key_raises_in_get_strict(self):
        data = {
            "tiers": [
                {"name": "INN", "type": "primary"},
                {"name": "INN", "type": "secondary"},
            ]
        }
        with pytest.raises(AmbiguousPathError):
            get_strict(data, "tiers[INN].type")

    def test_ambiguous_key_raises_in_set(self):
        data = {
            "tiers": [
                {"name": "INN", "value": 1},
                {"name": "INN", "value": 2},
            ]
        }
        with pytest.raises(AmbiguousPathError):
            set(data, "tiers[INN].value", 99)


class TestExists:
    """Test exists() function."""

    def test_exists_true(self):
        data = {"vendor": {"name": "Acme"}}
        assert exists(data, "vendor") is True
        assert exists(data, "vendor.name") is True

    def test_exists_false(self):
        data = {"vendor": {}}
        assert exists(data, "vendor.name") is False
        assert exists(data, "other") is False

    def test_exists_with_none_value(self):
        data = {"vendor": {"name": None}}
        assert exists(data, "vendor.name") is True  # None is a value


class TestSet:
    """Test set() function."""

    def test_simple_set(self):
        data = {"vendor": {}}
        set(data, "vendor.name", "Acme")
        assert data["vendor"]["name"] == "Acme"

    def test_overwrite(self):
        data = {"vendor": {"name": "Old"}}
        set(data, "vendor.name", "New")
        assert data["vendor"]["name"] == "New"

    def test_create_intermediate_dicts(self):
        data = {}
        set(data, "vendor.address.city", "Lagos")
        assert data == {"vendor": {"address": {"city": "Lagos"}}}

    def test_create_intermediates_false(self):
        data = {}
        with pytest.raises(PathNotFoundError):
            set(data, "vendor.name", "Acme", create_intermediates=False)

    def test_set_into_list_by_index(self):
        data = {"items": [{"price": 10}]}
        set(data, "items[0].price", 99)
        assert data["items"][0]["price"] == 99

    def test_set_list_item_directly(self):
        data = {"items": [1, 2, 3]}
        set(data, "items[1]", 99)
        assert data["items"][1] == 99

    def test_refuse_list_growth_by_default(self):
        data = {"items": [{"price": 10}]}
        with pytest.raises(PathNotFoundError):
            set(data, "items[1].price", 20)

    def test_allow_list_growth(self):
        data = {"items": [{"price": 10}]}
        set(data, "items[1].price", 20, allow_list_growth=True)
        assert len(data["items"]) == 2
        assert data["items"][1]["price"] == 20

    def test_list_growth_fills_with_dicts(self):
        data = {"items": []}
        set(data, "items[2].value", 42, allow_list_growth=True)
        assert len(data["items"]) == 3
        assert data["items"][0] == {}
        assert data["items"][1] == {}
        assert data["items"][2]["value"] == 42

    def test_set_by_key(self):
        data = {
            "services": [
                {"service_code": "PCP", "copay": 25},
            ]
        }
        set(data, "services[PCP].copay", 30)
        assert data["services"][0]["copay"] == 30

    def test_wildcard_in_set_raises(self):
        data = {"items": [{"price": 10}]}
        with pytest.raises(PathError):
            set(data, "items[*].price", 0)


class TestDelete:
    """Test delete() function."""

    def test_delete_dict_key(self):
        data = {"vendor": {"name": "Acme", "city": "Lagos"}}
        result = delete(data, "vendor.name")
        assert result is True
        assert "name" not in data["vendor"]
        assert "city" in data["vendor"]

    def test_delete_list_item_by_index(self):
        data = {"items": [{"id": 1}, {"id": 2}, {"id": 3}]}
        result = delete(data, "items[1]")
        assert result is True
        assert len(data["items"]) == 2
        assert data["items"][0]["id"] == 1
        assert data["items"][1]["id"] == 3  # Shifted

    def test_delete_list_item_by_key(self):
        data = {
            "services": [
                {"service_code": "PCP"},
                {"service_code": "ER"},
            ]
        }
        result = delete(data, "services[PCP]")
        assert result is True
        assert len(data["services"]) == 1
        assert data["services"][0]["service_code"] == "ER"

    def test_delete_missing_returns_false(self):
        data = {"vendor": {}}
        result = delete(data, "vendor.name")
        assert result is False

    def test_delete_missing_intermediate_returns_false(self):
        data = {}
        result = delete(data, "vendor.name")
        assert result is False

    def test_delete_wildcard_raises(self):
        data = {"items": [1, 2, 3]}
        with pytest.raises(PathError):
            delete(data, "items[*]")


class TestExpand:
    """Test wildcard expansion."""

    def test_expand_simple(self):
        data = {"items": [{"price": 10}, {"price": 20}]}
        paths = expand(data, "items[*].price")
        assert paths == ["items[0].price", "items[1].price"]

    def test_expand_uses_natural_key(self):
        data = {
            "services": [
                {"service_code": "PCP", "copay": 25},
                {"service_code": "ER", "copay": 100},
            ]
        }
        paths = expand(data, "services[*].copay")
        assert paths == ["services[PCP].copay", "services[ER].copay"]

    def test_expand_uses_index_when_no_key(self):
        data = {
            "items": [
                {"value": 1},  # No key field
                {"value": 2},
            ]
        }
        paths = expand(data, "items[*].value")
        assert paths == ["items[0].value", "items[1].value"]

    def test_expand_deep_wildcard(self):
        data = {
            "plans": [
                {"id": "A", "services": [{"code": "X"}, {"code": "Y"}]},
                {"id": "B", "services": [{"code": "Z"}]},
            ]
        }
        paths = expand(data, "plans[*].services[*].code")
        assert paths == [
            "plans[A].services[X].code",
            "plans[A].services[Y].code",
            "plans[B].services[Z].code",
        ]

    def test_expand_no_wildcard(self):
        data = {"items": [{"price": 10}]}
        paths = expand(data, "items[0].price")
        assert paths == ["items[0].price"]

    def test_expand_missing_path(self):
        data = {"items": []}
        paths = expand(data, "items[*].price")
        assert paths == []

    def test_expand_non_list(self):
        data = {"items": "not a list"}
        paths = expand(data, "items[*].price")
        assert paths == []

    def test_expand_with_values(self):
        data = {"items": [{"price": 10}, {"price": 20}]}
        result = expand_with_values(data, "items[*].price")
        assert result == [("items[0].price", 10), ("items[1].price", 20)]

    def test_expand_multiple_wildcards(self):
        data = {
            "matrix": [
                {"id": "row1", "cells": [{"id": "a", "v": 1}, {"id": "b", "v": 2}]},
                {"id": "row2", "cells": [{"id": "c", "v": 3}]},
            ]
        }
        paths = expand(data, "matrix[*].cells[*].v")
        assert paths == [
            "matrix[row1].cells[a].v",
            "matrix[row1].cells[b].v",
            "matrix[row2].cells[c].v",
        ]


class TestResolveKeySelector:
    """Test key selector resolution."""

    def test_finds_by_service_code(self):
        items = [{"service_code": "PCP"}, {"service_code": "ER"}]
        assert resolve_key_selector(items, "PCP", "") == 0
        assert resolve_key_selector(items, "ER", "") == 1

    def test_finds_by_code(self):
        items = [{"code": "A"}, {"code": "B"}]
        assert resolve_key_selector(items, "A", "") == 0

    def test_finds_by_id(self):
        items = [{"id": "x"}, {"id": "y"}]
        assert resolve_key_selector(items, "x", "") == 0

    def test_finds_by_key(self):
        items = [{"key": "foo"}, {"key": "bar"}]
        assert resolve_key_selector(items, "foo", "") == 0

    def test_finds_by_name(self):
        items = [{"name": "Alice"}, {"name": "Bob"}]
        assert resolve_key_selector(items, "Alice", "") == 0

    def test_priority_order(self):
        """service_code takes priority over name."""
        items = [{"service_code": "A", "name": "B"}]
        assert resolve_key_selector(items, "A", "") == 0
        # "B" won't match because service_code matches first

    def test_not_found_raises(self):
        items = [{"service_code": "PCP"}]
        with pytest.raises(PathNotFoundError):
            resolve_key_selector(items, "UNKNOWN", "test.path")

    def test_ambiguous_raises(self):
        items = [{"name": "X"}, {"name": "X"}]
        with pytest.raises(AmbiguousPathError) as exc:
            resolve_key_selector(items, "X", "test.path")
        assert exc.value.matches == [0, 1]

    def test_skips_non_dicts(self):
        items = ["string", {"service_code": "PCP"}, 42]
        assert resolve_key_selector(items, "PCP", "") == 1


class TestGetItemKey:
    """Test get_item_key() helper."""

    def test_returns_service_code(self):
        assert get_item_key({"service_code": "PCP", "name": "Primary"}) == "PCP"

    def test_returns_code(self):
        assert get_item_key({"code": "ABC"}) == "ABC"

    def test_returns_id(self):
        assert get_item_key({"id": "123"}) == "123"

    def test_returns_key(self):
        assert get_item_key({"key": "mykey"}) == "mykey"

    def test_returns_name(self):
        assert get_item_key({"name": "Foo"}) == "Foo"

    def test_priority_service_code_over_name(self):
        assert get_item_key({"service_code": "SC", "name": "Name"}) == "SC"

    def test_returns_none_if_no_key_field(self):
        assert get_item_key({"foo": "bar"}) is None

    def test_handles_non_dict(self):
        assert get_item_key("not a dict") is None
        assert get_item_key(123) is None
        assert get_item_key(None) is None

    def test_skips_none_values(self):
        assert get_item_key({"service_code": None, "name": "Foo"}) == "Foo"

    def test_converts_to_string(self):
        assert get_item_key({"id": 123}) == "123"


class TestPathUtilities:
    """Test path manipulation utilities."""

    def test_join_simple(self):
        assert join("vendor", "name") == "vendor.name"

    def test_join_with_selector(self):
        assert join("items[0]", "price") == "items[0].price"

    def test_join_empty_parts(self):
        assert join("", "name") == "name"
        assert join("vendor", "") == "vendor"
        assert join("", "", "name") == "name"

    def test_join_single(self):
        assert join("vendor") == "vendor"

    def test_parent_nested(self):
        assert parent("vendor.address.city") == "vendor.address"

    def test_parent_two_levels(self):
        assert parent("vendor.name") == "vendor"

    def test_parent_single(self):
        assert parent("vendor") is None

    def test_parent_with_selector(self):
        assert parent("items[0].price") == "items[0]"

    def test_leaf_nested(self):
        assert leaf("vendor.address.city") == "city"

    def test_leaf_with_selector(self):
        assert leaf("items[0].price") == "price"

    def test_leaf_selector_segment(self):
        assert leaf("items[0]") == "items[0]"

    def test_leaf_single(self):
        assert leaf("vendor") == "vendor"


class TestBatchOperations:
    """Test get_many and set_many."""

    def test_get_many(self):
        data = {
            "vendor": {"name": "Acme", "city": "Lagos"},
            "total": 100,
        }
        result = get_many(data, ["vendor.name", "vendor.city", "total"])
        assert result == {
            "vendor.name": "Acme",
            "vendor.city": "Lagos",
            "total": 100,
        }

    def test_get_many_with_missing(self):
        data = {"vendor": {"name": "Acme"}}
        result = get_many(data, ["vendor.name", "vendor.missing"], default="N/A")
        assert result == {
            "vendor.name": "Acme",
            "vendor.missing": "N/A",
        }

    def test_set_many(self):
        data = {"vendor": {}}
        result_paths = set_many(
            data,
            {
                "vendor.name": "Acme",
                "vendor.city": "Lagos",
            },
        )
        assert data["vendor"]["name"] == "Acme"
        assert data["vendor"]["city"] == "Lagos"
        assert sorted(result_paths) == sorted(["vendor.name", "vendor.city"])

    def test_set_many_creates_intermediates(self):
        data = {}
        set_many(
            data,
            {
                "a.b.c": 1,
                "a.b.d": 2,
            },
        )
        assert data == {"a": {"b": {"c": 1, "d": 2}}}


class TestEdgeCases:
    """Test edge cases and tricky scenarios."""

    def test_sbc_style_path(self):
        """Test a real SBC-style path."""
        data = {
            "services": [
                {
                    "service_code": "PCP_VISIT",
                    "coverage_by_tier": {
                        "INN": {"copay": {"amount": 25}},
                        "OON": {"copay": {"amount": 50}},
                    },
                },
            ]
        }

        assert get(data, "services[PCP_VISIT].coverage_by_tier.INN.copay.amount") == 25

        set(data, "services[PCP_VISIT].coverage_by_tier.INN.copay.amount", 30)
        assert data["services"][0]["coverage_by_tier"]["INN"]["copay"]["amount"] == 30

    def test_accumulator_path(self):
        """Test accumulator-style nested paths."""
        data = {
            "accumulators": {
                "deductible": {
                    "amounts_by_tier": {
                        "INN": {"individual": 500, "family": 1000},
                    }
                }
            }
        }

        assert (
            get(data, "accumulators.deductible.amounts_by_tier.INN.individual") == 500
        )

    def test_numeric_string_key(self):
        """Test that pure numeric keys are treated as indices."""
        data = {"items": [{"id": "123", "value": "a"}, {"id": "456", "value": "b"}]}

        # [0] is index, gets first item
        assert get(data, "items[0].value") == "a"

        # [123] is index 123, not key "123" - so returns None (out of bounds)
        assert get(data, "items[123].value") is None

    def test_deeply_nested_wildcard(self):
        """Test deeply nested wildcard expansion."""
        data = {
            "root": [
                {
                    "id": "a",
                    "children": [
                        {"id": "a1", "items": [{"id": "x", "v": 1}]},
                        {"id": "a2", "items": [{"id": "y", "v": 2}]},
                    ],
                },
            ]
        }

        paths = expand(data, "root[*].children[*].items[*].v")
        assert paths == [
            "root[a].children[a1].items[x].v",
            "root[a].children[a2].items[y].v",
        ]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
