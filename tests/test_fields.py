"""
Tests for fields.py - CriticalField, VerifierPolicy, Parsers, Comparators.
"""

import pytest
from pullcite.core.fields import (
    CriticalField,
    VerifierPolicy,
    Parsers,
    Comparators,
    find_policy,
    expand_critical_fields,
    MONEY_POLICY,
    PERCENT_POLICY,
    INTEGER_POLICY,
    YES_NO_POLICY,
    STRING_POLICY,
    LIMIT_POLICY,
    CONTAINS_POLICY,
)
from pullcite.core.paths import InvalidPathError


class TestCriticalField:
    """Test CriticalField dataclass."""

    def test_basic_creation(self):
        field = CriticalField(
            path="deductible.individual",
            label="Individual Deductible",
            search_query="individual deductible amount",
        )
        assert field.path == "deductible.individual"
        assert field.label == "Individual Deductible"
        assert field.search_query == "individual deductible amount"
        assert field.required is True
        assert field.policy_id is None

    def test_optional_field(self):
        field = CriticalField(
            path="optional.field",
            label="Optional",
            search_query="optional",
            required=False,
        )
        assert field.required is False

    def test_with_policy_id(self):
        field = CriticalField(
            path="amount",
            label="Amount",
            search_query="amount",
            policy_id="money",
        )
        assert field.policy_id == "money"

    def test_invalid_path_raises(self):
        with pytest.raises(InvalidPathError):
            CriticalField(
                path="invalid..path",
                label="Bad",
                search_query="test",
            )

    def test_empty_search_query_raises(self):
        with pytest.raises(ValueError) as exc:
            CriticalField(
                path="valid.path",
                label="Test",
                search_query="",
            )
        assert "search_query cannot be empty" in str(exc.value)

    def test_whitespace_search_query_raises(self):
        with pytest.raises(ValueError):
            CriticalField(
                path="valid.path",
                label="Test",
                search_query="   ",
            )

    def test_has_wildcard(self):
        field_with = CriticalField(
            path="services[*].copay",
            label="Copay",
            search_query="copay",
        )
        assert field_with.has_wildcard is True

        field_without = CriticalField(
            path="services[PCP].copay",
            label="Copay",
            search_query="copay",
        )
        assert field_without.has_wildcard is False

    def test_with_path(self):
        original = CriticalField(
            path="services[*].copay",
            label="Service Copay",
            search_query="copay amount",
            required=True,
            policy_id="money",
        )

        expanded = original.with_path("services[PCP].copay")

        assert expanded.path == "services[PCP].copay"
        assert expanded.label == "Service Copay"
        assert expanded.search_query == "copay amount"
        assert expanded.required is True
        assert expanded.policy_id == "money"

        # Original unchanged
        assert original.path == "services[*].copay"

    def test_immutability(self):
        field = CriticalField(
            path="test",
            label="Test",
            search_query="test",
        )
        with pytest.raises(AttributeError):
            field.path = "other"


class TestVerifierPolicy:
    """Test VerifierPolicy dataclass."""

    def test_basic_creation(self):
        policy = VerifierPolicy(
            id="test_policy",
            field_pattern="*.amount",
        )
        assert policy.id == "test_policy"
        assert policy.field_pattern == "*.amount"
        assert policy.parse_extracted is None
        assert policy.comparator is None
        assert policy.match_mode == "exact"

    def test_with_functions(self):
        policy = VerifierPolicy(
            id="money_policy",
            field_pattern="*.amount",
            parse_extracted=Parsers.money,
            parse_found=Parsers.money,
            comparator=Comparators.money_equivalent,
            tolerance=0.01,
            match_mode="numeric",
        )
        assert policy.parse_extracted is Parsers.money
        assert policy.match_mode == "numeric"

    def test_matches_exact(self):
        policy = VerifierPolicy(id="p", field_pattern="deductible.individual")

        assert policy.matches("deductible.individual") is True
        assert policy.matches("deductible.family") is False

    def test_matches_wildcard_star(self):
        policy = VerifierPolicy(id="p", field_pattern="*.amount")

        assert policy.matches("deductible.amount") is True
        assert policy.matches("copay.amount") is True
        assert policy.matches("deductible.value") is False

    def test_matches_wildcard_middle(self):
        policy = VerifierPolicy(id="p", field_pattern="services[*].copay")

        assert policy.matches("services[PCP].copay") is True
        assert policy.matches("services[ER].copay") is True
        assert policy.matches("services[PCP].coinsurance") is False

    def test_specificity(self):
        # Longer = more specific
        p1 = VerifierPolicy(id="1", field_pattern="a")
        p2 = VerifierPolicy(id="2", field_pattern="ab")
        assert p2.specificity > p1.specificity

        # Wildcards reduce specificity
        p3 = VerifierPolicy(id="3", field_pattern="services[*].copay")
        p4 = VerifierPolicy(id="4", field_pattern="services[PCP].copay")
        assert p4.specificity > p3.specificity

    def test_immutability(self):
        policy = VerifierPolicy(id="p", field_pattern="*")
        with pytest.raises(AttributeError):
            policy.id = "other"


class TestParsers:
    """Test Parsers class methods."""

    class TestMoney:
        def test_none(self):
            assert Parsers.money(None) is None

        def test_int(self):
            assert Parsers.money(1500) == 1500.0

        def test_float(self):
            assert Parsers.money(1500.50) == 1500.50

        def test_string_plain(self):
            assert Parsers.money("1500") == 1500.0

        def test_string_with_dollar(self):
            assert Parsers.money("$1500") == 1500.0

        def test_string_with_comma(self):
            assert Parsers.money("1,500") == 1500.0

        def test_string_with_dollar_and_comma(self):
            assert Parsers.money("$1,500") == 1500.0

        def test_string_with_decimals(self):
            assert Parsers.money("$1,500.00") == 1500.0

        def test_string_with_spaces(self):
            assert Parsers.money("  $1,500  ") == 1500.0

        def test_empty_string(self):
            assert Parsers.money("") is None

        def test_invalid_string(self):
            assert Parsers.money("not a number") is None

        def test_negative(self):
            assert Parsers.money("-500") == -500.0

    class TestPercent:
        def test_none(self):
            assert Parsers.percent(None) is None

        def test_int(self):
            assert Parsers.percent(30) == 30.0

        def test_float(self):
            assert Parsers.percent(30.5) == 30.5

        def test_decimal_converts(self):
            # 0.30 should become 30
            assert Parsers.percent(0.30) == 30.0

        def test_string_plain(self):
            assert Parsers.percent("30") == 30.0

        def test_string_with_percent(self):
            assert Parsers.percent("30%") == 30.0

        def test_string_with_percent_word(self):
            assert Parsers.percent("30 percent") == 30.0

        def test_string_decimal(self):
            assert Parsers.percent("30.5%") == 30.5

        def test_empty_string(self):
            assert Parsers.percent("") is None

        def test_invalid_string(self):
            assert Parsers.percent("not a percent") is None

        def test_zero(self):
            assert Parsers.percent(0) == 0.0

        def test_hundred(self):
            assert Parsers.percent(100) == 100.0

    class TestYesNo:
        def test_none(self):
            assert Parsers.yes_no(None) is None

        def test_bool_true(self):
            assert Parsers.yes_no(True) is True

        def test_bool_false(self):
            assert Parsers.yes_no(False) is False

        def test_int_one(self):
            assert Parsers.yes_no(1) is True

        def test_int_zero(self):
            assert Parsers.yes_no(0) is False

        def test_string_yes(self):
            assert Parsers.yes_no("yes") is True
            assert Parsers.yes_no("YES") is True
            assert Parsers.yes_no("Yes") is True

        def test_string_no(self):
            assert Parsers.yes_no("no") is False
            assert Parsers.yes_no("NO") is False

        def test_string_true_false(self):
            assert Parsers.yes_no("true") is True
            assert Parsers.yes_no("false") is False

        def test_string_y_n(self):
            assert Parsers.yes_no("y") is True
            assert Parsers.yes_no("n") is False

        def test_invalid_string(self):
            assert Parsers.yes_no("maybe") is None

    class TestInteger:
        def test_none(self):
            assert Parsers.integer(None) is None

        def test_int(self):
            assert Parsers.integer(100) == 100

        def test_float_whole(self):
            assert Parsers.integer(100.0) == 100

        def test_float_decimal(self):
            assert Parsers.integer(100.5) is None

        def test_string_plain(self):
            assert Parsers.integer("100") == 100

        def test_string_with_suffix(self):
            assert Parsers.integer("100 days") == 100

        def test_string_negative(self):
            assert Parsers.integer("-50") == -50

        def test_empty_string(self):
            assert Parsers.integer("") is None

        def test_bool_excluded(self):
            # Avoid True -> 1 confusion
            assert Parsers.integer(True) is None
            assert Parsers.integer(False) is None

    class TestStringNormalized:
        def test_basic(self):
            assert Parsers.string_normalized("Hello World") == "hello world"

        def test_whitespace(self):
            assert Parsers.string_normalized("  hello   world  ") == "hello world"

        def test_none(self):
            assert Parsers.string_normalized(None) == ""

        def test_number(self):
            assert Parsers.string_normalized(123) == "123"

    class TestNotApplicable:
        def test_none(self):
            assert Parsers.not_applicable(None) is None

        def test_empty_string(self):
            assert Parsers.not_applicable("") is None

        def test_na_variants(self):
            assert Parsers.not_applicable("N/A") == "N/A"
            assert Parsers.not_applicable("n/a") == "N/A"
            assert Parsers.not_applicable("NA") == "N/A"
            assert Parsers.not_applicable("na") == "N/A"
            assert Parsers.not_applicable("N.A.") == "N/A"

        def test_not_applicable_variants(self):
            assert Parsers.not_applicable("Not Applicable") == "N/A"
            assert Parsers.not_applicable("not applicable") == "N/A"
            assert Parsers.not_applicable("Not Available") == "N/A"

        def test_none_null_variants(self):
            assert Parsers.not_applicable("None") == "N/A"
            assert Parsers.not_applicable("none") == "N/A"
            assert Parsers.not_applicable("null") == "N/A"

        def test_dash_variants(self):
            assert Parsers.not_applicable("-") == "N/A"
            assert Parsers.not_applicable("—") == "N/A"  # em dash
            assert Parsers.not_applicable("–") == "N/A"  # en dash

        def test_coverage_variants(self):
            assert Parsers.not_applicable("No Coverage") == "N/A"
            assert Parsers.not_applicable("not covered") == "N/A"
            assert Parsers.not_applicable("no") == "N/A"

        def test_regular_string_normalized(self):
            # Regular strings get normalized, not converted to N/A
            assert Parsers.not_applicable("Hello World") == "hello world"
            assert Parsers.not_applicable("  Some Value  ") == "some value"

        def test_non_string(self):
            assert Parsers.not_applicable(123) == "123"

    class TestLimit:
        def test_none(self):
            assert Parsers.limit(None) is None

        def test_int(self):
            assert Parsers.limit(10000) == 10000.0

        def test_float(self):
            assert Parsers.limit(10000.50) == 10000.50

        def test_money_string(self):
            assert Parsers.limit("$10,000") == 10000.0
            assert Parsers.limit("$10,000.00") == 10000.0

        def test_plain_number_string(self):
            assert Parsers.limit("10000") == 10000.0

        def test_unlimited_variants(self):
            assert Parsers.limit("unlimited") == "unlimited"
            assert Parsers.limit("Unlimited") == "unlimited"
            assert Parsers.limit("UNLIMITED") == "unlimited"
            assert Parsers.limit("no limit") == "unlimited"
            assert Parsers.limit("No Maximum") == "unlimited"
            assert Parsers.limit("no cap") == "unlimited"

        def test_infinity_variants(self):
            assert Parsers.limit("∞") == "unlimited"
            assert Parsers.limit("infinity") == "unlimited"
            assert Parsers.limit("inf") == "unlimited"

        def test_na_variants(self):
            assert Parsers.limit("N/A") == "N/A"
            assert Parsers.limit("n/a") == "N/A"
            assert Parsers.limit("NA") == "N/A"
            assert Parsers.limit("-") == "N/A"

        def test_empty_string(self):
            assert Parsers.limit("") is None

        def test_none_means_unlimited(self):
            # "none" in limit context means no limit
            assert Parsers.limit("none") == "unlimited"


class TestComparators:
    """Test Comparators class methods."""

    def test_exact_match(self):
        assert Comparators.exact(100, 100) is True
        assert Comparators.exact("a", "a") is True
        assert Comparators.exact(100, 101) is False

    def test_exact_type_matters(self):
        assert Comparators.exact(100, "100") is False

    def test_normalized_string(self):
        assert Comparators.normalized_string("Hello", "hello") is True
        assert Comparators.normalized_string("  hello  ", "hello") is True
        assert Comparators.normalized_string("hello", "world") is False

    def test_numeric_tolerance_within(self):
        cmp = Comparators.numeric_tolerance(0.01)
        assert cmp(100.0, 100.005) is True
        assert cmp(100.0, 100.009) is True

    def test_numeric_tolerance_outside(self):
        cmp = Comparators.numeric_tolerance(0.01)
        assert cmp(100.0, 100.02) is False

    def test_numeric_tolerance_none(self):
        cmp = Comparators.numeric_tolerance(0.01)
        assert cmp(None, None) is True
        assert cmp(100, None) is False
        assert cmp(None, 100) is False

    def test_numeric_tolerance_strings(self):
        cmp = Comparators.numeric_tolerance(0.01)
        assert cmp("100.0", "100.005") is True

    def test_numeric_tolerance_invalid(self):
        cmp = Comparators.numeric_tolerance(0.01)
        assert cmp("not a number", 100) is False

    def test_money_equivalent(self):
        assert Comparators.money_equivalent(1500, "$1,500") is True
        assert Comparators.money_equivalent("$1,500.00", 1500) is True
        assert Comparators.money_equivalent(100, 100.005) is True
        assert Comparators.money_equivalent(100, 101) is False

    def test_money_equivalent_none(self):
        assert Comparators.money_equivalent(None, None) is True
        assert Comparators.money_equivalent(100, None) is False

    def test_percent_equivalent(self):
        assert Comparators.percent_equivalent(30, "30%") is True
        assert Comparators.percent_equivalent(0.30, 30) is True
        assert Comparators.percent_equivalent(30, 30.05) is True
        assert Comparators.percent_equivalent(30, 31) is False

    def test_integer_equivalent(self):
        assert Comparators.integer_equivalent(100, 100) is True
        assert Comparators.integer_equivalent(100, "100") is True
        assert Comparators.integer_equivalent("100 days", 100) is True
        assert Comparators.integer_equivalent(100.0, 100) is True
        assert Comparators.integer_equivalent(100, 101) is False

    def test_integer_equivalent_none(self):
        assert Comparators.integer_equivalent(None, None) is True
        assert Comparators.integer_equivalent(100, None) is False
        assert Comparators.integer_equivalent(None, 100) is False

    def test_yes_no_equivalent(self):
        assert Comparators.yes_no_equivalent(True, "yes") is True
        assert Comparators.yes_no_equivalent(False, "no") is True
        assert Comparators.yes_no_equivalent("YES", True) is True
        assert Comparators.yes_no_equivalent(1, "true") is True
        assert Comparators.yes_no_equivalent(0, "false") is True
        assert Comparators.yes_no_equivalent(True, False) is False

    def test_yes_no_equivalent_none(self):
        assert Comparators.yes_no_equivalent(None, None) is True
        assert Comparators.yes_no_equivalent(True, None) is False

    def test_limit_equivalent_numeric(self):
        assert Comparators.limit_equivalent(10000, "$10,000") is True
        assert Comparators.limit_equivalent("$10,000.00", 10000) is True
        assert Comparators.limit_equivalent(10000, 10000.005) is True
        assert Comparators.limit_equivalent(10000, 10001) is False

    def test_limit_equivalent_unlimited(self):
        assert Comparators.limit_equivalent("unlimited", "no limit") is True
        assert Comparators.limit_equivalent("Unlimited", "unlimited") is True
        assert Comparators.limit_equivalent("no maximum", "unlimited") is True

    def test_limit_equivalent_na(self):
        assert Comparators.limit_equivalent("N/A", "n/a") is True
        assert Comparators.limit_equivalent("-", "N/A") is True

    def test_limit_equivalent_mixed_types(self):
        # Numeric vs string don't match
        assert Comparators.limit_equivalent(10000, "unlimited") is False
        assert Comparators.limit_equivalent("unlimited", 10000) is False
        assert Comparators.limit_equivalent(10000, "N/A") is False

    def test_limit_equivalent_none(self):
        assert Comparators.limit_equivalent(None, None) is True
        assert Comparators.limit_equivalent(10000, None) is False

    def test_contains_basic(self):
        assert Comparators.contains("copay", "Your copay is $30") is True
        assert Comparators.contains("Your copay is $30", "copay") is True
        assert Comparators.contains("deductible", "copay") is False

    def test_contains_case_insensitive(self):
        assert Comparators.contains("COPAY", "your copay is $30") is True
        assert Comparators.contains("Copay", "COPAY") is True

    def test_contains_whitespace_normalized(self):
        assert Comparators.contains("  copay  ", "copay") is True

    def test_contains_empty(self):
        assert Comparators.contains("", "") is True
        assert Comparators.contains("", "something") is False
        assert Comparators.contains("something", "") is False

    def test_contains_none(self):
        assert Comparators.contains(None, None) is True
        assert Comparators.contains("value", None) is False


class TestFindPolicy:
    """Test find_policy function."""

    def test_no_match(self):
        policies = [
            VerifierPolicy(id="1", field_pattern="other.*"),
        ]
        result = find_policy("deductible.individual", policies)
        assert result is None

    def test_single_match(self):
        policies = [
            VerifierPolicy(id="1", field_pattern="deductible.*"),
        ]
        result = find_policy("deductible.individual", policies)
        assert result is not None
        assert result.id == "1"

    def test_most_specific_wins(self):
        policies = [
            VerifierPolicy(id="generic", field_pattern="*"),
            VerifierPolicy(id="specific", field_pattern="deductible.*"),
            VerifierPolicy(id="most_specific", field_pattern="deductible.individual"),
        ]
        result = find_policy("deductible.individual", policies)
        assert result.id == "most_specific"

    def test_wildcard_less_specific(self):
        policies = [
            VerifierPolicy(id="wildcard", field_pattern="services[*].copay"),
            VerifierPolicy(id="concrete", field_pattern="services[PCP].copay"),
        ]
        result = find_policy("services[PCP].copay", policies)
        assert result.id == "concrete"

    def test_empty_policies(self):
        result = find_policy("any.path", [])
        assert result is None

    def test_tuple_policies(self):
        policies = (VerifierPolicy(id="1", field_pattern="*"),)
        result = find_policy("test", policies)
        assert result is not None


class TestExpandCriticalFields:
    """Test expand_critical_fields function."""

    def test_no_wildcards(self):
        fields = [
            CriticalField(
                path="deductible.individual", label="Ded", search_query="ded"
            ),
        ]
        data = {"deductible": {"individual": 1500}}

        result = expand_critical_fields(fields, data)

        assert len(result) == 1
        assert result[0].path == "deductible.individual"

    def test_expand_wildcard(self):
        fields = [
            CriticalField(
                path="services[*].copay", label="Copay", search_query="copay"
            ),
        ]
        data = {
            "services": [
                {"service_code": "PCP", "copay": 25},
                {"service_code": "ER", "copay": 100},
            ]
        }

        result = expand_critical_fields(fields, data)

        assert len(result) == 2
        paths = [f.path for f in result]
        assert "services[PCP].copay" in paths
        assert "services[ER].copay" in paths

    def test_preserves_other_attributes(self):
        fields = [
            CriticalField(
                path="services[*].copay",
                label="Service Copay",
                search_query="copay amount",
                required=True,
                policy_id="money",
            ),
        ]
        data = {"services": [{"service_code": "PCP", "copay": 25}]}

        result = expand_critical_fields(fields, data)

        assert len(result) == 1
        assert result[0].label == "Service Copay"
        assert result[0].search_query == "copay amount"
        assert result[0].required is True
        assert result[0].policy_id == "money"

    def test_mixed_fields(self):
        fields = [
            CriticalField(path="plan_name", label="Name", search_query="name"),
            CriticalField(
                path="services[*].copay", label="Copay", search_query="copay"
            ),
        ]
        data = {
            "plan_name": "Test Plan",
            "services": [
                {"service_code": "PCP", "copay": 25},
            ],
        }

        result = expand_critical_fields(fields, data)

        assert len(result) == 2
        paths = [f.path for f in result]
        assert "plan_name" in paths
        assert "services[PCP].copay" in paths

    def test_empty_list_expansion(self):
        fields = [
            CriticalField(
                path="services[*].copay", label="Copay", search_query="copay"
            ),
        ]
        data = {"services": []}

        result = expand_critical_fields(fields, data)

        assert len(result) == 0


class TestBuiltInPolicies:
    """Test built-in policy constants."""

    def test_money_policy(self):
        assert MONEY_POLICY.id == "money"
        assert MONEY_POLICY.parse_extracted is Parsers.money
        assert MONEY_POLICY.comparator is Comparators.money_equivalent
        assert MONEY_POLICY.match_mode == "numeric"

    def test_percent_policy(self):
        assert PERCENT_POLICY.id == "percent"
        assert PERCENT_POLICY.parse_extracted is Parsers.percent
        assert PERCENT_POLICY.comparator is Comparators.percent_equivalent
        assert PERCENT_POLICY.match_mode == "numeric"

    def test_integer_policy(self):
        assert INTEGER_POLICY.id == "integer"
        assert INTEGER_POLICY.parse_extracted is Parsers.integer
        assert INTEGER_POLICY.parse_found is Parsers.integer
        assert INTEGER_POLICY.comparator is Comparators.integer_equivalent
        assert INTEGER_POLICY.match_mode == "exact"

    def test_yes_no_policy(self):
        assert YES_NO_POLICY.id == "yes_no"
        assert YES_NO_POLICY.parse_extracted is Parsers.yes_no
        assert YES_NO_POLICY.parse_found is Parsers.yes_no
        assert YES_NO_POLICY.comparator is Comparators.yes_no_equivalent
        assert YES_NO_POLICY.match_mode == "exact"

    def test_string_policy(self):
        assert STRING_POLICY.id == "string"
        assert STRING_POLICY.parse_extracted is Parsers.string_normalized
        assert STRING_POLICY.parse_found is Parsers.string_normalized
        assert STRING_POLICY.comparator is Comparators.normalized_string
        assert STRING_POLICY.match_mode == "normalized"

    def test_limit_policy(self):
        assert LIMIT_POLICY.id == "limit"
        assert LIMIT_POLICY.parse_extracted is Parsers.limit
        assert LIMIT_POLICY.parse_found is Parsers.limit
        assert LIMIT_POLICY.comparator is Comparators.limit_equivalent
        assert LIMIT_POLICY.match_mode == "numeric"

    def test_contains_policy(self):
        assert CONTAINS_POLICY.id == "contains"
        assert CONTAINS_POLICY.parse_extracted is Parsers.string_normalized
        assert CONTAINS_POLICY.parse_found is Parsers.string_normalized
        assert CONTAINS_POLICY.comparator is Comparators.contains
        assert CONTAINS_POLICY.match_mode == "fuzzy"

    def test_all_policies_match_any_path(self):
        """All built-in policies use '*' pattern to match any path."""
        policies = [
            MONEY_POLICY,
            PERCENT_POLICY,
            INTEGER_POLICY,
            YES_NO_POLICY,
            STRING_POLICY,
            LIMIT_POLICY,
            CONTAINS_POLICY,
        ]
        for policy in policies:
            assert policy.field_pattern == "*"
            assert policy.matches("any.path.here") is True
            assert policy.matches("services[PCP].copay") is True
