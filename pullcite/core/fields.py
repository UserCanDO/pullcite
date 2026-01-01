"""
Field definitions and verification policies.

This module defines what fields need verification and how to verify them.

Key types:
- CriticalField: A field that must be verified with evidence
- VerifierPolicy: Custom parsing and comparison logic for fields
- Parsers: Built-in value parsers (money, percent, etc.)
- Comparators: Built-in value comparators (exact, numeric tolerance, etc.)
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Callable, Literal

from .paths import validate, expand, InvalidPathError


@dataclass(frozen=True)
class CriticalField:
    """
    A field that requires verification with evidence.

    Critical fields are checked by the Verifier role, which searches
    the document for supporting evidence.

    Attributes:
        path: Path to the field (e.g., 'deductible.individual.in_network').
              May contain wildcards for expansion (e.g., 'services[*].copay').
        label: Human-readable name for error messages and UI.
        search_query: Hint for vector search (keywords to find this value).
        required: If True, NOT_FOUND is an error. If False, acceptable.
        policy_id: Optional ID linking to a VerifierPolicy.
    """

    path: str
    label: str
    search_query: str
    required: bool = True
    policy_id: str | None = None

    def __post_init__(self) -> None:
        """Validate field configuration."""
        # Validate path syntax (allows wildcards)
        is_valid, error = validate(self.path)
        if not is_valid:
            raise InvalidPathError(f"Invalid path: {error}", self.path)

        # search_query should not be empty
        if not self.search_query or not self.search_query.strip():
            raise ValueError("search_query cannot be empty")

    @property
    def has_wildcard(self) -> bool:
        """Check if path contains wildcards."""
        return "[*]" in self.path

    def with_path(self, new_path: str) -> CriticalField:
        """Return a copy with a different path (for expansion)."""
        return CriticalField(
            path=new_path,
            label=self.label,
            search_query=self.search_query,
            required=self.required,
            policy_id=self.policy_id,
        )


@dataclass(frozen=True)
class VerifierPolicy:
    """
    Custom verification logic for specific fields.

    Policies control how values are parsed and compared during verification.
    Use field_pattern to match multiple fields with glob syntax.

    Attributes:
        id: Unique identifier for this policy.
        field_pattern: Glob pattern matching field paths
                       (e.g., 'services[*].copay', '*.amount').
        parse_extracted: Normalize value from LLM extraction.
        parse_found: Parse value from document quote.
        comparator: Compare parsed values. Default is equality.
        tolerance: For numeric comparisons, acceptable difference.
        match_mode: Comparison strategy hint.
    """

    id: str
    field_pattern: str
    parse_extracted: Callable[[Any], Any] | None = None
    parse_found: Callable[[str], Any] | None = None
    comparator: Callable[[Any, Any], bool] | None = None
    tolerance: float | None = None
    match_mode: Literal["exact", "normalized", "fuzzy", "numeric"] = "exact"

    def matches(self, path: str) -> bool:
        """
        Check if this policy applies to a path.

        Converts the field_pattern (glob-like syntax) to a regex for matching.
        We can't use fnmatch directly because it treats [] as character classes,
        but we need [*] to mean "any bracket content" and literal brackets
        like [PCP] to match exactly.

        Pattern syntax:
            *       -> matches any sequence of characters (like regex .*)
            ?       -> matches any single character (like regex .)
            [*]     -> matches any bracketed content: [PCP], [0], [ER_VISIT], etc.
            [X]     -> matches literal [X] (exact bracket content)
            other   -> matches literally

        Examples:
            Pattern                 Path                        Match?
            -------                 ----                        ------
            "services[*].copay"     "services[PCP].copay"       Yes
            "services[*].copay"     "services[0].copay"         Yes
            "services[*].copay"     "services[ER_VISIT].copay"  Yes
            "services[PCP].copay"   "services[PCP].copay"       Yes
            "services[PCP].copay"   "services[ER].copay"        No
            "*.amount"              "deductible.amount"         Yes
            "*.amount"              "foo.bar.amount"            Yes
            "deductible.*"          "deductible.in_network"     Yes

        Args:
            path: Field path to check (e.g., "services[PCP].copay").

        Returns:
            True if this policy's pattern matches the path.
        """
        # Strategy: Convert our glob-like pattern to a regex.
        #
        # The tricky part is handling brackets:
        # - [*] means "any bracket content" -> regex: \[[^\]]+\]
        # - [PCP] means "literal [PCP]"     -> regex: \[PCP\]
        #
        # We use a placeholder technique to handle this:
        # 1. Replace [*] with a unique placeholder (null bytes)
        # 2. Escape all other special chars (including literal brackets)
        # 3. Replace placeholder with the "any bracket" regex
        pattern = self.field_pattern

        # Placeholder: Use null bytes because they can't appear in valid paths
        # and won't be confused with any regex metacharacters.
        placeholder = "\x00WILDCARD\x00"
        pattern = pattern.replace("[*]", placeholder)

        # Build regex by processing each character
        regex_pattern = ""
        i = 0
        while i < len(pattern):
            if pattern[i:].startswith(placeholder):
                # [*] placeholder -> match any non-empty bracket content
                # \[       = literal opening bracket
                # [^\]]+   = one or more chars that aren't closing bracket
                # \]       = literal closing bracket
                regex_pattern += r"\[[^\]]+\]"
                i += len(placeholder)

            elif pattern[i] == "*":
                # Glob * -> match any sequence (including empty)
                # .* in regex means "zero or more of any character"
                regex_pattern += ".*"
                i += 1

            elif pattern[i] == "?":
                # Glob ? -> match exactly one character
                # . in regex means "any single character"
                regex_pattern += "."
                i += 1

            elif pattern[i] in r"\.^$+{}()|":
                # Regex metacharacters that need escaping for literal match
                # (these have special meaning in regex but should be literal here)
                regex_pattern += "\\" + pattern[i]
                i += 1

            elif pattern[i] == "[":
                # Literal opening bracket (not [*], which was replaced)
                # Must escape because [ starts a character class in regex
                regex_pattern += r"\["
                i += 1

            elif pattern[i] == "]":
                # Literal closing bracket
                # Must escape because ] ends a character class in regex
                regex_pattern += r"\]"
                i += 1

            else:
                # Regular character, use as-is
                regex_pattern += pattern[i]
                i += 1

        # Anchor the pattern to match the entire string
        # ^ = start of string, $ = end of string
        regex_pattern = "^" + regex_pattern + "$"

        return bool(re.match(regex_pattern, path))

    @property
    def specificity(self) -> int:
        """
        Calculate pattern specificity for precedence.

        More specific patterns (longer, fewer wildcards) win.
        Used to sort policies when multiple match.

        Returns:
            Specificity score (higher = more specific).
        """
        # Penalize wildcards, reward length
        wildcard_penalty = self.field_pattern.count("*") * 10
        return len(self.field_pattern) - wildcard_penalty


class Parsers:
    """Built-in value parsers for common types."""

    @staticmethod
    def money(value: Any) -> float | None:
        """
        Parse monetary value.

        Handles: "$1,500", "1500.00", "$1,500.00", "1,500", 1500, 1500.0

        Args:
            value: Value to parse.

        Returns:
            Float amount or None if unparseable.
        """
        if value is None:
            return None

        if isinstance(value, (int, float)):
            return float(value)

        if isinstance(value, str):
            # Remove currency symbols, commas, whitespace
            cleaned = re.sub(r"[$,\s]", "", value.strip())

            # Handle empty string after cleaning
            if not cleaned:
                return None

            # Try to parse as float
            try:
                return float(cleaned)
            except ValueError:
                return None

        return None

    @staticmethod
    def percent(value: Any) -> float | None:
        """
        Parse percentage.

        Handles: "30%", "30", 30, 0.30, "30 percent", "30.5%"

        Returns whole number (30.0 not 0.30).

        Args:
            value: Value to parse.

        Returns:
            Float percentage (0-100 scale) or None if unparseable.
        """
        if value is None:
            return None

        if isinstance(value, (int, float)):
            # If it's less than 1, assume it's a decimal (0.30 -> 30)
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
                # If parsed value is less than 1, assume decimal
                if 0 < result < 1:
                    return result * 100
                return result
            except ValueError:
                return None

        return None

    @staticmethod
    def yes_no(value: Any) -> bool | None:
        """
        Parse yes/no/boolean.

        Handles: "yes", "no", "true", "false", True, False, 1, 0

        Args:
            value: Value to parse.

        Returns:
            Bool or None if unparseable.
        """
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

    @staticmethod
    def integer(value: Any) -> int | None:
        """
        Parse integer.

        Handles: "100", 100, 100.0, "100 days"

        Args:
            value: Value to parse.

        Returns:
            Int or None if unparseable.
        """
        if value is None:
            return None

        if isinstance(value, bool):
            # Avoid True -> 1, False -> 0 confusion
            return None

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

    @staticmethod
    def string_normalized(value: Any) -> str:
        """
        Normalize string for comparison.

        - Lowercase
        - Strip whitespace
        - Collapse multiple spaces

        Args:
            value: Value to normalize.

        Returns:
            Normalized string.
        """
        if value is None:
            return ""

        s = str(value).lower().strip()
        # Collapse multiple whitespace
        s = re.sub(r"\s+", " ", s)
        return s

    @staticmethod
    def not_applicable(value: Any) -> str | None:
        """
        Normalize "not applicable" style values.

        Handles: "N/A", "n/a", "NA", "None", "Not Applicable", "-", "—", ""

        Returns a canonical form or the original value if not a N/A variant.

        Args:
            value: Value to parse.

        Returns:
            "N/A" if value represents not-applicable, None if empty/None,
            otherwise the original string (normalized).
        """
        if value is None:
            return None

        if isinstance(value, str):
            normalized = value.strip().lower()

            # Empty string
            if not normalized:
                return None

            # Common N/A variants
            na_variants = {
                "n/a",
                "na",
                "n.a.",
                "n.a",
                "not applicable",
                "none",
                "null",
                "-",
                "—",
                "–",
                "not available",
                "no",
                "no coverage",
                "not covered",
            }
            if normalized in na_variants:
                return "N/A"

            # Return normalized string for non-N/A values
            return Parsers.string_normalized(value)

        # Non-string values: convert and normalize
        return Parsers.string_normalized(value)

    @staticmethod
    def limit(value: Any) -> float | str | None:
        """
        Parse limit/maximum values.

        Handles monetary limits, "unlimited", and N/A values.

        Examples: "$10,000", "10000", "unlimited", "no limit", "N/A"

        Args:
            value: Value to parse.

        Returns:
            Float for numeric limits, "unlimited" string, "N/A", or None.
        """
        if value is None:
            return None

        if isinstance(value, (int, float)):
            return float(value)

        if isinstance(value, str):
            normalized = value.strip().lower()

            if not normalized:
                return None

            # Check for unlimited variants
            unlimited_variants = {
                "unlimited",
                "no limit",
                "no maximum",
                "none",
                "no cap",
                "∞",
                "infinity",
                "inf",
            }
            if normalized in unlimited_variants:
                return "unlimited"

            # Check for N/A
            na_variants = {"n/a", "na", "not applicable", "-", "—"}
            if normalized in na_variants:
                return "N/A"

            # Try to parse as money
            return Parsers.money(value)

        return None


class Comparators:
    """Built-in value comparators."""

    @staticmethod
    def exact(a: Any, b: Any) -> bool:
        """
        Exact equality comparison.

        Args:
            a: First value.
            b: Second value.

        Returns:
            True if values are equal.
        """
        return a == b

    @staticmethod
    def normalized_string(a: Any, b: Any) -> bool:
        """
        Case-insensitive, whitespace-normalized string comparison.

        Args:
            a: First value.
            b: Second value.

        Returns:
            True if normalized strings match.
        """
        return Parsers.string_normalized(a) == Parsers.string_normalized(b)

    @staticmethod
    def numeric_tolerance(tolerance: float) -> Callable[[Any, Any], bool]:
        """
        Factory for numeric comparison with tolerance.

        Args:
            tolerance: Maximum acceptable difference.

        Returns:
            Comparator function.

        Example:
            >>> cmp = Comparators.numeric_tolerance(0.01)
            >>> cmp(100.0, 100.009)
            True
        """

        def compare(a: Any, b: Any) -> bool:
            if a is None or b is None:
                return a is b
            try:
                return abs(float(a) - float(b)) <= tolerance
            except (TypeError, ValueError):
                return False

        return compare

    @staticmethod
    def money_equivalent(a: Any, b: Any) -> bool:
        """
        Compare monetary values with $0.01 tolerance.

        Parses both values as money first.

        Args:
            a: First value.
            b: Second value.

        Returns:
            True if values represent same amount (within $0.01).
        """
        parsed_a = Parsers.money(a)
        parsed_b = Parsers.money(b)

        if parsed_a is None or parsed_b is None:
            return parsed_a is parsed_b

        return abs(parsed_a - parsed_b) <= 0.01

    @staticmethod
    def percent_equivalent(a: Any, b: Any) -> bool:
        """
        Compare percentage values with 0.1% tolerance.

        Parses both values as percent first.

        Args:
            a: First value.
            b: Second value.

        Returns:
            True if values represent same percentage (within 0.1%).
        """
        parsed_a = Parsers.percent(a)
        parsed_b = Parsers.percent(b)

        if parsed_a is None or parsed_b is None:
            return parsed_a is parsed_b

        return abs(parsed_a - parsed_b) <= 0.1

    @staticmethod
    def integer_equivalent(a: Any, b: Any) -> bool:
        """
        Compare integer values.

        Parses both values as integers first.

        Args:
            a: First value.
            b: Second value.

        Returns:
            True if values represent same integer.
        """
        parsed_a = Parsers.integer(a)
        parsed_b = Parsers.integer(b)

        if parsed_a is None or parsed_b is None:
            return parsed_a is parsed_b

        return parsed_a == parsed_b

    @staticmethod
    def yes_no_equivalent(a: Any, b: Any) -> bool:
        """
        Compare boolean/yes-no values.

        Parses both values as yes/no first.

        Args:
            a: First value.
            b: Second value.

        Returns:
            True if values represent same boolean.
        """
        parsed_a = Parsers.yes_no(a)
        parsed_b = Parsers.yes_no(b)

        if parsed_a is None or parsed_b is None:
            return parsed_a is parsed_b

        return parsed_a == parsed_b

    @staticmethod
    def limit_equivalent(a: Any, b: Any) -> bool:
        """
        Compare limit values (handles "unlimited", N/A, and numeric).

        Parses both values as limits first.

        Args:
            a: First value.
            b: Second value.

        Returns:
            True if values represent same limit.
        """
        parsed_a = Parsers.limit(a)
        parsed_b = Parsers.limit(b)

        if parsed_a is None or parsed_b is None:
            return parsed_a is parsed_b

        # Both are strings (unlimited, N/A)
        if isinstance(parsed_a, str) and isinstance(parsed_b, str):
            return parsed_a == parsed_b

        # Both are numeric - use money tolerance
        if isinstance(parsed_a, float) and isinstance(parsed_b, float):
            return abs(parsed_a - parsed_b) <= 0.01

        # Mixed types don't match
        return False

    @staticmethod
    def contains(a: Any, b: Any) -> bool:
        """
        Check if one string contains the other (case-insensitive).

        Useful for fuzzy matching where extracted value might be
        a substring of the document text or vice versa.

        Args:
            a: First value (extracted).
            b: Second value (found in document).

        Returns:
            True if either contains the other.
        """
        str_a = Parsers.string_normalized(a)
        str_b = Parsers.string_normalized(b)

        if not str_a or not str_b:
            return str_a == str_b

        return str_a in str_b or str_b in str_a


def find_policy(
    path: str,
    policies: list[VerifierPolicy] | tuple[VerifierPolicy, ...],
) -> VerifierPolicy | None:
    """
    Find the most specific policy matching a path.

    When multiple policies match, the most specific one wins
    (longer patterns, fewer wildcards).

    Args:
        path: Field path to find policy for.
        policies: Available policies.

    Returns:
        Most specific matching policy, or None if no match.
    """
    matching = [p for p in policies if p.matches(path)]

    if not matching:
        return None

    # Sort by specificity (highest first)
    matching.sort(key=lambda p: p.specificity, reverse=True)
    return matching[0]


def expand_critical_fields(
    fields: list[CriticalField],
    data: dict[str, Any],
) -> list[CriticalField]:
    """
    Expand wildcards in critical field paths.

    Transforms fields with [*] selectors into concrete fields
    using the actual data structure.

    Args:
        fields: Critical fields (may contain wildcards).
        data: Extracted data to expand against.

    Returns:
        List of fields with wildcards expanded to concrete paths.

    Example:
        Input:  [CriticalField(path="services[*].copay", ...)]
        Output: [
            CriticalField(path="services[PCP].copay", ...),
            CriticalField(path="services[ER].copay", ...),
        ]
    """
    result: list[CriticalField] = []

    for field in fields:
        if field.has_wildcard:
            # Expand wildcards
            concrete_paths = expand(data, field.path)
            for concrete_path in concrete_paths:
                result.append(field.with_path(concrete_path))
        else:
            result.append(field)

    return result


# =============================================================================
# Common policies for reuse
# =============================================================================
# These policies provide sensible defaults for common field types.
# Use them directly or as templates for custom policies.
#
# Usage:
#     config = ExtractorConfig().with_policies(MONEY_POLICY, PERCENT_POLICY)
#
# Or for specific field patterns:
#     custom = VerifierPolicy(
#         id="copay",
#         field_pattern="services[*].copay",
#         parse_extracted=Parsers.money,
#         parse_found=Parsers.money,
#         comparator=Comparators.money_equivalent,
#     )

MONEY_POLICY = VerifierPolicy(
    id="money",
    field_pattern="*",
    parse_extracted=Parsers.money,
    parse_found=Parsers.money,
    comparator=Comparators.money_equivalent,
    match_mode="numeric",
)
"""Policy for monetary values ($1,500, 1500.00, etc.)."""

PERCENT_POLICY = VerifierPolicy(
    id="percent",
    field_pattern="*",
    parse_extracted=Parsers.percent,
    parse_found=Parsers.percent,
    comparator=Comparators.percent_equivalent,
    match_mode="numeric",
)
"""Policy for percentage values (30%, 0.30, 30 percent, etc.)."""

INTEGER_POLICY = VerifierPolicy(
    id="integer",
    field_pattern="*",
    parse_extracted=Parsers.integer,
    parse_found=Parsers.integer,
    comparator=Comparators.integer_equivalent,
    match_mode="exact",
)
"""Policy for integer values (30, "30 days", 30.0, etc.)."""

YES_NO_POLICY = VerifierPolicy(
    id="yes_no",
    field_pattern="*",
    parse_extracted=Parsers.yes_no,
    parse_found=Parsers.yes_no,
    comparator=Comparators.yes_no_equivalent,
    match_mode="exact",
)
"""Policy for boolean/yes-no values (yes, no, true, false, etc.)."""

STRING_POLICY = VerifierPolicy(
    id="string",
    field_pattern="*",
    parse_extracted=Parsers.string_normalized,
    parse_found=Parsers.string_normalized,
    comparator=Comparators.normalized_string,
    match_mode="normalized",
)
"""Policy for case-insensitive, whitespace-normalized string comparison."""

LIMIT_POLICY = VerifierPolicy(
    id="limit",
    field_pattern="*",
    parse_extracted=Parsers.limit,
    parse_found=Parsers.limit,
    comparator=Comparators.limit_equivalent,
    match_mode="numeric",
)
"""Policy for limit values ($10,000, unlimited, no limit, N/A, etc.)."""

CONTAINS_POLICY = VerifierPolicy(
    id="contains",
    field_pattern="*",
    parse_extracted=Parsers.string_normalized,
    parse_found=Parsers.string_normalized,
    comparator=Comparators.contains,
    match_mode="fuzzy",
)
"""Policy for fuzzy string matching (checks if one contains the other)."""


__all__ = [
    # Core types
    "CriticalField",
    "VerifierPolicy",
    "Parsers",
    "Comparators",
    # Functions
    "find_policy",
    "expand_critical_fields",
    # Reusable policies
    "MONEY_POLICY",
    "PERCENT_POLICY",
    "INTEGER_POLICY",
    "YES_NO_POLICY",
    "STRING_POLICY",
    "LIMIT_POLICY",
    "CONTAINS_POLICY",
]
