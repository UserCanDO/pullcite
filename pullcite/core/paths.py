"""
Path resolution for nested data structures.

This module provides utilities for navigating, reading, and writing
nested dicts and lists using dot-notation paths with array selectors.

Path Grammar
============

    path          = segment ("." segment)*
    segment       = identifier selector?
    identifier    = [a-zA-Z_][a-zA-Z0-9_]*
    selector      = "[" selector_key "]"
    selector_key  = index | wildcard | key
    index         = [0-9]+
    wildcard      = "*"
    key           = [a-zA-Z0-9_]+      # Note: allows leading digits like "123ABC"

Selector Resolution:
    - All digits (^[0-9]+$) → index access (0-based)
    - Single asterisk (*) → wildcard (all items)
    - Anything else matching [a-zA-Z0-9_]+ → key lookup

Key Lookup Fields (checked in order):
    1. service_code
    2. code
    3. id
    4. key
    5. name

Behavior Summary:
    - get() → soft, returns default on missing, never raises PathNotFoundError
    - get_strict() → raises PathNotFoundError if path doesn't exist
    - Both raise AmbiguousPathError if [KEY] matches multiple items
    - Both raise InvalidPathError if path syntax is wrong
    - set() creates intermediate dicts, refuses to grow lists by default
    - delete() removes dict keys or list elements (with index shift)
    - Wildcards forbidden in set() and delete()
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Iterator


# =============================================================================
# Constants
# =============================================================================

# Fields checked (in order) when resolving [KEY] selectors
KEY_LOOKUP_FIELDS = ("service_code", "code", "id", "key", "name")

# Regex patterns
_IDENTIFIER_PATTERN = re.compile(r"^[a-zA-Z_][a-zA-Z0-9_]*$")
_SELECTOR_KEY_PATTERN = re.compile(r"^[a-zA-Z0-9_]+$")
_INDEX_PATTERN = re.compile(r"^[0-9]+$")

# Sentinel for "no value" (distinguishes None from missing)
_MISSING = object()


# =============================================================================
# Exceptions
# =============================================================================


class PathError(Exception):
    """Base exception for path operations."""

    def __init__(self, message: str, path: str):
        self.message = message
        self.path = path
        super().__init__(f"{message} [path: {path}]")


class PathNotFoundError(PathError):
    """Raised when a path cannot be resolved."""

    def __init__(
        self,
        message: str,
        path: str,
        resolved_to: str | None = None,
        remaining: str | None = None,
    ):
        self.resolved_to = resolved_to
        self.remaining = remaining
        detail = message
        if resolved_to is not None:
            detail += f" (resolved to: '{resolved_to}', remaining: '{remaining}')"
        super().__init__(detail, path)


class AmbiguousPathError(PathError):
    """Raised when a path selector matches multiple items."""

    def __init__(
        self,
        message: str,
        path: str,
        selector: str,
        matches: list[int],
    ):
        self.selector = selector
        self.matches = matches
        detail = f"{message} Selector [{selector}] matched indices: {matches}"
        super().__init__(detail, path)


class InvalidPathError(PathError):
    """Raised when a path string is syntactically invalid."""

    def __init__(self, message: str, path: str, position: int | None = None):
        self.position = position
        detail = message
        if position is not None:
            detail += f" at position {position}"
        super().__init__(detail, path)


# =============================================================================
# Path Parsing
# =============================================================================


@dataclass(frozen=True)
class PathSegment:
    """
    A single segment of a parsed path.

    Attributes:
        field: The field/key name.
        selector: Optional selector value (int for index, str for key/wildcard).
        selector_type: Type of selector: "index", "key", "wildcard", or None.
    """

    field: str
    selector: str | int | None = None
    selector_type: str | None = None  # "index" | "key" | "wildcard" | None

    def __str__(self) -> str:
        """Convert back to path string format."""
        if self.selector is None:
            return self.field
        elif self.selector_type == "wildcard":
            return f"{self.field}[*]"
        else:
            return f"{self.field}[{self.selector}]"


@dataclass(frozen=True)
class ParsedPath:
    """
    A fully parsed path.

    Attributes:
        original: The original path string.
        segments: Tuple of PathSegment objects.
    """

    original: str
    segments: tuple[PathSegment, ...]

    def __str__(self) -> str:
        """Convert back to path string format."""
        return ".".join(str(seg) for seg in self.segments)

    def __iter__(self) -> Iterator[PathSegment]:
        """Iterate over segments."""
        return iter(self.segments)

    def __len__(self) -> int:
        """Number of segments."""
        return len(self.segments)

    @property
    def has_wildcard(self) -> bool:
        """Check if any segment has a wildcard selector."""
        return any(seg.selector_type == "wildcard" for seg in self.segments)

    @property
    def parent(self) -> ParsedPath | None:
        """Get parent path (all segments except last)."""
        if len(self.segments) <= 1:
            return None
        parent_segs = self.segments[:-1]
        parent_str = ".".join(str(seg) for seg in parent_segs)
        return ParsedPath(original=parent_str, segments=parent_segs)

    @property
    def leaf(self) -> PathSegment:
        """Get the final segment."""
        return self.segments[-1]


def parse(path: str) -> ParsedPath:
    """
    Parse a path string into structured segments.

    Single-pass scanner that handles:
    - Dot-separated field names
    - Bracket selectors: [0], [KEY], [*]

    Examples:
        "user.name" → [Field("user"), Field("name")]
        "items[0].price" → [Field("items"), Index(0), Field("price")]
        "data[KEY]" → [Field("data"), Key("KEY")]
        "users[*].email" → [Field("users"), All(), Field("email")]
        "config.settings.debug" → [Field("config"), Field("settings"), Field("debug")]

    Args:
        path: Path string to parse.

    Returns:
        ParsedPath with all segments.

    Raises:
        InvalidPathError: If path syntax is invalid.
    """
    if not path:
        raise InvalidPathError("Path cannot be empty", path)

    segments: list[PathSegment] = []
    pos = 0
    length = len(path)

    while pos < length:
        # Skip leading dot (except at start)
        if pos > 0:
            if path[pos] != ".":
                raise InvalidPathError(
                    f"Expected '.' between segments, got '{path[pos]}'",
                    path,
                    position=pos,
                )
            pos += 1
            if pos >= length:
                raise InvalidPathError("Path cannot end with '.'", path, position=pos)

        # Parse identifier
        ident_start = pos
        if not (path[pos].isalpha() or path[pos] == "_"):
            raise InvalidPathError(
                f"Identifier must start with letter or underscore, got '{path[pos]}'",
                path,
                position=pos,
            )

        while pos < length and (path[pos].isalnum() or path[pos] == "_"):
            pos += 1

        field = path[ident_start:pos]

        # Check for selector
        selector = None
        selector_type = None

        if pos < length and path[pos] == "[":
            bracket_start = pos
            pos += 1  # skip '['

            # Find closing bracket
            selector_start = pos
            while pos < length and path[pos] != "]":
                pos += 1

            if pos >= length:
                raise InvalidPathError(
                    "Unclosed bracket selector",
                    path,
                    position=bracket_start,
                )

            selector_str = path[selector_start:pos]
            pos += 1  # skip ']'

            if not selector_str:
                raise InvalidPathError(
                    "Empty selector",
                    path,
                    position=bracket_start,
                )

            # Determine selector type
            if selector_str == "*":
                selector = "*"
                selector_type = "wildcard"
            elif _INDEX_PATTERN.match(selector_str):
                selector = int(selector_str)
                selector_type = "index"
            elif _SELECTOR_KEY_PATTERN.match(selector_str):
                selector = selector_str
                selector_type = "key"
            else:
                raise InvalidPathError(
                    f"Invalid selector: '{selector_str}'. Must be integer, identifier, or '*'",
                    path,
                    position=selector_start,
                )

        segments.append(
            PathSegment(
                field=field,
                selector=selector,
                selector_type=selector_type,
            )
        )

    return ParsedPath(original=path, segments=tuple(segments))


def validate(path: str) -> tuple[bool, str | None]:
    """
    Check if a path string is syntactically valid.

    Does not check if path exists in any data structure.

    Args:
        path: Path string to validate.

    Returns:
        Tuple of (is_valid, error_message or None).
    """
    try:
        parse(path)
        return (True, None)
    except InvalidPathError as e:
        return (False, e.message)


# =============================================================================
# Key Selector Resolution
# =============================================================================

"""
This section handles path selectors like [KEY] that match items by field values.

For example, given:
    items = [
        {"service_code": "ABC123", "name": "Service A"},
        {"code": "XYZ789", "name": "Service B"}, 
        {"id": "DEF456", "name": "Service C"}
    ]

The selector [ABC123] would find the first item (matches service_code)
The selector [XYZ789] would find the second item (matches code)
The selector [DEF456] would find the third item (matches id)

KEY_LOOKUP_FIELDS defines the priority order for field matching.
"""


def resolve_key_selector(
    items: list,
    key: str,
    path_context: str = "",
) -> int:
    """
    Find the index of item matching key.

    Checks KEY_LOOKUP_FIELDS in order for a match.

    Args:
        items: List of dicts to search.
        key: Key value to find.
        path_context: Full path string for error messages.

    Returns:
        Index of matching item.

    Raises:
        PathNotFoundError: If no item matches.
        AmbiguousPathError: If multiple items match.
    """
    matches: list[int] = []

    for idx, item in enumerate(items):
        if not isinstance(item, dict):
            continue

        for lookup_field in KEY_LOOKUP_FIELDS:
            if lookup_field in item and item[lookup_field] == key:
                matches.append(idx)
                break  # Don't check other fields for this item

    if len(matches) == 0:
        raise PathNotFoundError(
            f"No item matches selector [{key}]",
            path_context,
        )

    if len(matches) > 1:
        raise AmbiguousPathError(
            f"Multiple items match selector",
            path_context,
            selector=key,
            matches=matches,
        )

    return matches[0]


def get_item_key(item: dict) -> str | None:
    """
    Get the "natural key" of a dict for path representation.

    Checks KEY_LOOKUP_FIELDS in order and returns the first found.
    Used when expanding wildcards to produce readable paths.

    Args:
        item: Dict to get key from.

    Returns:
        Key value as string if found, None otherwise.
    """
    if not isinstance(item, dict):
        return None

    for lookup_field in KEY_LOOKUP_FIELDS:
        if lookup_field in item:
            value = item[lookup_field]
            if value is not None:
                return str(value)

    return None


# =============================================================================
# Internal Resolution Helpers
# =============================================================================


def _resolve_path(
    data: Any,
    parsed: ParsedPath,
    allow_missing: bool = False,
) -> tuple[Any, str]:
    """
    Resolve a path to its value.

    Args:
        data: Data structure to traverse.
        parsed: Parsed path.
        allow_missing: If True, return _MISSING instead of raising.

    Returns:
        Tuple of (value, resolved_path_string).
        Value is _MISSING if allow_missing=True and path not found.

    Raises:
        PathNotFoundError: If path doesn't exist and allow_missing=False.
        AmbiguousPathError: If selector matches multiple items.
    """
    current = data
    resolved_parts: list[str] = []

    for i, segment in enumerate(parsed.segments):
        remaining_segments = parsed.segments[i:]
        remaining_str = ".".join(str(s) for s in remaining_segments)
        resolved_str = ".".join(resolved_parts) if resolved_parts else "(root)"

        # Access field
        if isinstance(current, dict):
            if segment.field not in current:
                if allow_missing:
                    return (_MISSING, resolved_str)
                raise PathNotFoundError(
                    f"Key '{segment.field}' not found in dict",
                    parsed.original,
                    resolved_to=resolved_str,
                    remaining=remaining_str,
                )
            current = current[segment.field]
        else:
            if allow_missing:
                return (_MISSING, resolved_str)
            raise PathNotFoundError(
                f"Cannot access field '{segment.field}' on non-dict type {type(current).__name__}",
                parsed.original,
                resolved_to=resolved_str,
                remaining=remaining_str,
            )

        # Build resolved part (field only so far)
        resolved_part = segment.field

        # Apply selector if present
        if segment.selector is not None:
            if not isinstance(current, list):
                if allow_missing:
                    return (_MISSING, ".".join(resolved_parts + [resolved_part]))
                raise PathNotFoundError(
                    f"Cannot apply selector [{segment.selector}] to non-list type {type(current).__name__}",
                    parsed.original,
                    resolved_to=".".join(resolved_parts + [resolved_part]),
                    remaining=remaining_str,
                )

            if segment.selector_type == "wildcard":
                # Wildcards should be handled by expand(), not here
                raise PathError(
                    "Cannot resolve wildcard selector directly. Use expand() first.",
                    parsed.original,
                )

            elif segment.selector_type == "index":
                idx = segment.selector
                if idx < 0 or idx >= len(current):
                    if allow_missing:
                        return (_MISSING, ".".join(resolved_parts + [resolved_part]))
                    raise PathNotFoundError(
                        f"Index {idx} out of bounds (list length: {len(current)})",
                        parsed.original,
                        resolved_to=".".join(resolved_parts + [resolved_part]),
                        remaining=remaining_str,
                    )
                current = current[idx]
                resolved_part = f"{segment.field}[{idx}]"

            elif segment.selector_type == "key":
                try:
                    idx = resolve_key_selector(
                        current,
                        segment.selector,
                        path_context=parsed.original,
                    )
                except PathNotFoundError:
                    if allow_missing:
                        return (_MISSING, ".".join(resolved_parts + [resolved_part]))
                    raise
                current = current[idx]
                resolved_part = f"{segment.field}[{segment.selector}]"

        resolved_parts.append(resolved_part)

    return (current, ".".join(resolved_parts))


def _resolve_parent_for_set(
    data: Any,
    parsed: ParsedPath,
    create_intermediates: bool,
    allow_list_growth: bool,
) -> tuple[Any, PathSegment, str]:
    """
    Resolve to the parent container for a set operation.

    Returns:
        Tuple of (parent_container, leaf_segment, resolved_parent_path).

    Raises:
        PathNotFoundError: If parent path doesn't exist and can't be created.
        AmbiguousPathError: If selector matches multiple items.
        PathError: If path contains wildcard.
    """
    if parsed.has_wildcard:
        raise PathError(
            "Cannot set value at wildcard path. Use expand() first.",
            parsed.original,
        )

    if len(parsed.segments) == 1:
        # Root-level set
        return (data, parsed.segments[0], "(root)")

    # Navigate to parent
    current = data
    resolved_parts: list[str] = []

    for i, segment in enumerate(parsed.segments[:-1]):
        # Access or create field
        if isinstance(current, dict):
            if segment.field not in current:
                if not create_intermediates:
                    raise PathNotFoundError(
                        f"Key '{segment.field}' not found and create_intermediates=False",
                        parsed.original,
                        resolved_to=(
                            ".".join(resolved_parts) if resolved_parts else "(root)"
                        ),
                        remaining=".".join(str(s) for s in parsed.segments[i:]),
                    )
                # Create intermediate: dict or list based on next selector
                if segment.selector is not None:
                    # This segment has a selector, so field should be a list
                    current[segment.field] = []
                else:
                    current[segment.field] = {}
            current = current[segment.field]
        else:
            raise PathNotFoundError(
                f"Cannot access field '{segment.field}' on non-dict type {type(current).__name__}",
                parsed.original,
                resolved_to=".".join(resolved_parts) if resolved_parts else "(root)",
                remaining=".".join(str(s) for s in parsed.segments[i:]),
            )

        resolved_part = segment.field

        # Apply selector if present
        if segment.selector is not None:
            if not isinstance(current, list):
                raise PathNotFoundError(
                    f"Cannot apply selector [{segment.selector}] to non-list type {type(current).__name__}",
                    parsed.original,
                    resolved_to=".".join(resolved_parts + [resolved_part]),
                    remaining=".".join(str(s) for s in parsed.segments[i:]),
                )

            if segment.selector_type == "wildcard":
                raise PathError(
                    "Cannot set value at wildcard path. Use expand() first.",
                    parsed.original,
                )

            elif segment.selector_type == "index":
                idx = segment.selector

                # Handle list growth
                if idx >= len(current):
                    if not allow_list_growth:
                        raise PathNotFoundError(
                            f"Index {idx} out of bounds (list length: {len(current)}) "
                            f"and allow_list_growth=False",
                            parsed.original,
                            resolved_to=".".join(resolved_parts + [resolved_part]),
                        )
                    # Grow list with empty dict placeholders
                    while len(current) <= idx:
                        current.append({})

                current = current[idx]
                resolved_part = f"{segment.field}[{idx}]"

            elif segment.selector_type == "key":
                idx = resolve_key_selector(
                    current,
                    segment.selector,
                    path_context=parsed.original,
                )
                current = current[idx]
                resolved_part = f"{segment.field}[{segment.selector}]"

        resolved_parts.append(resolved_part)

    return (current, parsed.leaf, ".".join(resolved_parts))


# =============================================================================
# Public API: Get
# =============================================================================


def get(data: Any, path: str | ParsedPath, default: Any = None) -> Any:
    """
    Get value at path, or default if not found.

    Soft getter: never raises PathNotFoundError.
    Does raise on ambiguous paths or invalid syntax.

    Examples:
        data = {"user": {"name": "Alice"}}
        get(data, "user.name") → "Alice"
        get(data, "user.age", "unknown") → "unknown"
        get(data, "missing.path", None) → None

    Args:
        data: Data structure to traverse.
        path: Path string or ParsedPath.
        default: Value to return if path not found.

    Returns:
        Value at path, or default.

    Raises:
        InvalidPathError: If path syntax is invalid.
        AmbiguousPathError: If [KEY] selector matches multiple items.
    """
    parsed = path if isinstance(path, ParsedPath) else parse(path)

    if parsed.has_wildcard:
        raise PathError(
            "Cannot use get() with wildcard path. Use expand() first.",
            parsed.original,
        )

    try:
        value, _ = _resolve_path(data, parsed, allow_missing=True)
        return default if value is _MISSING else value
    except PathNotFoundError:
        return default


def get_strict(data: Any, path: str | ParsedPath) -> Any:
    """
    Get value at path, raising if not found.

    Examples:
        data = {"user": {"name": "Alice"}}
        get_strict(data, "user.name") → "Alice"
        get_strict(data, "user.age") → raises PathNotFoundError

    Args:
        data: Data structure to traverse.
        path: Path string or ParsedPath.

    Returns:
        Value at path.

    Raises:
        InvalidPathError: If path syntax is invalid.
        PathNotFoundError: If path does not exist.
        AmbiguousPathError: If [KEY] selector matches multiple items.
    """
    parsed = path if isinstance(path, ParsedPath) else parse(path)

    if parsed.has_wildcard:
        raise PathError(
            "Cannot use get_strict() with wildcard path. Use expand() first.",
            parsed.original,
        )

    value, _ = _resolve_path(data, parsed, allow_missing=False)
    return value


def exists(data: Any, path: str | ParsedPath) -> bool:
    """
    Check if path exists in data.

    Examples:
        data = {"user": {"name": "Alice", "age": None}}
        exists(data, "user.name") → True
        exists(data, "user.age") → True  # None counts as existing
        exists(data, "user.email") → False
        exists(data, "missing.path") → False

    Args:
        data: Data structure to check.
        path: Path string or ParsedPath.

    Returns:
        True if path resolves to a value (including None).

    Raises:
        InvalidPathError: If path syntax is invalid.
        AmbiguousPathError: If [KEY] selector matches multiple items.
    """
    parsed = path if isinstance(path, ParsedPath) else parse(path)

    if parsed.has_wildcard:
        raise PathError(
            "Cannot use exists() with wildcard path. Use expand() first.",
            parsed.original,
        )

    value, _ = _resolve_path(data, parsed, allow_missing=True)
    return value is not _MISSING


# =============================================================================
# Public API: Set
# =============================================================================


def set(
    data: Any,
    path: str | ParsedPath,
    value: Any,
    *,
    allow_list_growth: bool = False,
    create_intermediates: bool = True,
) -> None:
    """
    Set value at path, modifying data in place.

    Creates intermediate dicts as needed (if create_intermediates=True).
    Does NOT create intermediate list items by default.

    Examples:
        data = {}
        set(data, "user.name", "Alice")  # data becomes {"user": {"name": "Alice"}}
        set(data, "items[0].price", 10.99, allow_list_growth=True)  # Creates list with placeholder
        set(data, "config.debug", True)  # Creates intermediate dict

    Args:
        data: Data structure to modify (must be mutable dict at root).
        path: Path string or ParsedPath.
        value: Value to set.
        allow_list_growth: If True, extend lists with {} placeholders as needed.
        create_intermediates: If True, create missing dict keys.

    Raises:
        InvalidPathError: If path syntax is invalid.
        PathNotFoundError: If path cannot be created.
        AmbiguousPathError: If [KEY] selector matches multiple items.
        PathError: If path contains wildcard.
    """
    parsed = path if isinstance(path, ParsedPath) else parse(path)

    parent, leaf, _ = _resolve_parent_for_set(
        data, parsed, create_intermediates, allow_list_growth
    )

    # Now set the leaf value
    if isinstance(parent, dict):
        if leaf.selector is None:
            # Simple field set
            parent[leaf.field] = value
        else:
            # Field is a list, we need to set into it
            if leaf.field not in parent:
                if not create_intermediates:
                    raise PathNotFoundError(
                        f"Key '{leaf.field}' not found",
                        parsed.original,
                    )
                parent[leaf.field] = []

            target_list = parent[leaf.field]
            if not isinstance(target_list, list):
                raise PathNotFoundError(
                    f"Cannot apply selector to non-list",
                    parsed.original,
                )

            if leaf.selector_type == "index":
                idx = leaf.selector
                if idx >= len(target_list):
                    if not allow_list_growth:
                        raise PathNotFoundError(
                            f"Index {idx} out of bounds and allow_list_growth=False",
                            parsed.original,
                        )
                    while len(target_list) <= idx:
                        target_list.append({})
                target_list[idx] = value

            elif leaf.selector_type == "key":
                idx = resolve_key_selector(target_list, leaf.selector, parsed.original)
                target_list[idx] = value

    elif isinstance(parent, list):
        # Parent is a list (shouldn't happen with proper path structure)
        raise PathError(
            "Cannot set field on list. Path structure error.",
            parsed.original,
        )
    else:
        raise PathNotFoundError(
            f"Cannot set value on {type(parent).__name__}",
            parsed.original,
        )


# =============================================================================
# Public API: Delete
# =============================================================================


def delete(data: Any, path: str | ParsedPath) -> bool:
    """
    Delete value at path, modifying data in place.

    For lists, removes the element (shifts indices).
    For dicts, removes the key.

    Examples:
        data = {"user": {"name": "Alice", "age": 30}}
        delete(data, "user.age") → True  # data becomes {"user": {"name": "Alice"}}
        delete(data, "user.email") → False  # path didn't exist

        data = {"items": [{"name": "A"}, {"name": "B"}]}
        delete(data, "items[0]") → True  # removes first item, shifts indices

    Args:
        data: Data structure to modify.
        path: Path string or ParsedPath.

    Returns:
        True if value was deleted, False if path didn't exist.

    Raises:
        InvalidPathError: If path syntax is invalid.
        AmbiguousPathError: If [KEY] selector matches multiple items.
        PathError: If path contains wildcard.
    """
    parsed = path if isinstance(path, ParsedPath) else parse(path)

    if parsed.has_wildcard:
        raise PathError(
            "Cannot delete at wildcard path. Use expand() first.",
            parsed.original,
        )

    try:
        parent, leaf, _ = _resolve_parent_for_set(
            data, parsed, create_intermediates=False, allow_list_growth=False
        )
    except PathNotFoundError:
        return False

    if isinstance(parent, dict):
        if leaf.selector is None:
            if leaf.field in parent:
                del parent[leaf.field]
                return True
            return False
        else:
            if leaf.field not in parent:
                return False
            target_list = parent[leaf.field]
            if not isinstance(target_list, list):
                return False

            if leaf.selector_type == "index":
                idx = leaf.selector
                if idx < len(target_list):
                    target_list.pop(idx)
                    return True
                return False

            elif leaf.selector_type == "key":
                try:
                    idx = resolve_key_selector(
                        target_list, leaf.selector, parsed.original
                    )
                    target_list.pop(idx)
                    return True
                except PathNotFoundError:
                    return False

    return False


# =============================================================================
# Public API: Wildcard Expansion
# =============================================================================


def expand(data: Any, path: str | ParsedPath) -> list[str]:
    """
    Expand wildcards in path to concrete paths.

    Given a path with [*] selectors, returns all concrete paths
    that match in the data structure.

    Ordering: list order, depth-first.
    Key preference: uses natural key (get_item_key) if available, else index.

    Examples:
        data = {"users": [{"name": "Alice"}, {"name": "Bob"}], "admin": {"name": "Carol"}}
        expand(data, "users[*].name") → ["users[0].name", "users[1].name"]
        expand(data, "*.name") → ["users[0].name", "users[1].name", "admin.name"]
        expand(data, "missing[*].field") → []  # no matches

    Args:
        data: Data structure to expand against.
        path: Path string or ParsedPath (may contain wildcards).

    Returns:
        List of concrete path strings (no wildcards).
        Empty list if no matches.

    Raises:
        InvalidPathError: If path syntax is invalid.
        AmbiguousPathError: If non-wildcard [KEY] selector matches multiple items.
    """
    parsed = path if isinstance(path, ParsedPath) else parse(path)

    results: list[str] = []
    _expand_recursive(data, parsed.segments, 0, [], results)
    return results


def _expand_recursive(
    current: Any,
    segments: tuple[PathSegment, ...],
    seg_idx: int,
    path_parts: list[str],
    results: list[str],
) -> None:
    """Recursive helper for expand()."""
    if seg_idx >= len(segments):
        # Reached end of path
        results.append(".".join(path_parts))
        return

    segment = segments[seg_idx]

    # Access field
    if not isinstance(current, dict) or segment.field not in current:
        return  # Path doesn't exist, no expansion

    field_value = current[segment.field]

    if segment.selector is None:
        # No selector, continue with field value
        _expand_recursive(
            field_value,
            segments,
            seg_idx + 1,
            path_parts + [segment.field],
            results,
        )

    elif segment.selector_type == "wildcard":
        # Expand wildcard
        if not isinstance(field_value, list):
            return  # Can't expand non-list

        for idx, item in enumerate(field_value):
            # Prefer natural key, fall back to index
            natural_key = get_item_key(item) if isinstance(item, dict) else None
            if natural_key is not None:
                part = f"{segment.field}[{natural_key}]"
            else:
                part = f"{segment.field}[{idx}]"

            _expand_recursive(
                item,
                segments,
                seg_idx + 1,
                path_parts + [part],
                results,
            )

    elif segment.selector_type == "index":
        idx = segment.selector
        if not isinstance(field_value, list) or idx >= len(field_value):
            return

        _expand_recursive(
            field_value[idx],
            segments,
            seg_idx + 1,
            path_parts + [f"{segment.field}[{idx}]"],
            results,
        )

    elif segment.selector_type == "key":
        if not isinstance(field_value, list):
            return

        try:
            idx = resolve_key_selector(field_value, segment.selector, "")
            _expand_recursive(
                field_value[idx],
                segments,
                seg_idx + 1,
                path_parts + [f"{segment.field}[{segment.selector}]"],
                results,
            )
        except PathNotFoundError:
            return  # Key not found, no expansion


def expand_with_values(
    data: Any,
    path: str | ParsedPath,
) -> list[tuple[str, Any]]:
    """
    Expand wildcards and return paths with their values.

    Args:
        data: Data structure to expand against.
        path: Path string or ParsedPath (may contain wildcards).

    Returns:
        List of (path, value) tuples.
    """
    concrete_paths = expand(data, path)
    return [(p, get(data, p)) for p in concrete_paths]


# =============================================================================
# Path Manipulation Utilities
# =============================================================================


def join(*parts: str) -> str:
    """
    Join path parts with dots.

    Handles empty parts and parts that already have selectors.

    Args:
        *parts: Path parts to join.

    Returns:
        Joined path string.
    """
    non_empty = [p for p in parts if p]
    return ".".join(non_empty)


def parent(path: str) -> str | None:
    """
    Get parent path.

    Examples:
        parent("user.profile.name") → "user.profile"
        parent("user") → None  # root-level
        parent("items[0].price") → "items[0]"

    Args:
        path: Path string.

    Returns:
        Parent path, or None if root-level.
    """
    parsed = parse(path)
    parent_path = parsed.parent
    return str(parent_path) if parent_path else None


def leaf(path: str) -> str:
    """
    Get the leaf (final segment) of a path as string.

    Examples:
        leaf("user.profile.name") → "name"
        leaf("items[0].price") → "price"
        leaf("user") → "user"

    Args:
        path: Path string.

    Returns:
        Final segment as string.
    """
    parsed = parse(path)
    return str(parsed.leaf)


# =============================================================================
# Batch Operations
# =============================================================================


def get_many(
    data: Any,
    paths: list[str],
    default: Any = None,
) -> dict[str, Any]:
    """
    Get multiple values by path.

    Examples:
        data = {"user": {"name": "Alice", "age": 30}, "config": {"debug": True}}
        get_many(data, ["user.name", "user.email", "config.debug"])
        → {"user.name": "Alice", "user.email": None, "config.debug": True}
        get_many(data, ["missing.path"], default="N/A")
        → {"missing.path": "N/A"}

    Args:
        data: Data structure to traverse.
        paths: List of path strings.
        default: Default for missing paths.

    Returns:
        Dict mapping path → value.

    Raises:
        InvalidPathError: If any path is syntactically invalid.
        AmbiguousPathError: If any [KEY] selector matches multiple items.
    """
    return {p: get(data, p, default) for p in paths}


def set_many(
    data: Any,
    updates: dict[str, Any],
    *,
    allow_list_growth: bool = False,
    create_intermediates: bool = True,
) -> list[str]:
    """
    Set multiple values by path.

    Examples:
        data = {}
        set_many(data, {"user.name": "Alice", "user.age": 30, "config.debug": True})
        → ["user.name", "user.age", "config.debug"]
        # data becomes {"user": {"name": "Alice", "age": 30}, "config": {"debug": True}}

        data = {"items": []}
        set_many(data, {"items[0].name": "Item 1", "items[1].name": "Item 2"}, allow_list_growth=True)
        → ["items[0].name", "items[1].name"]

    Args:
        data: Data structure to modify.
        updates: Dict mapping path → new value.
        allow_list_growth: Allow creating new list items.
        create_intermediates: Create missing dict keys.

    Returns:
        List of paths that were successfully set.

    Raises:
        InvalidPathError: If any path is syntactically invalid.
        PathNotFoundError: If any path cannot be created.
        AmbiguousPathError: If any [KEY] selector matches multiple items.
        PathError: If any path contains wildcard.
    """
    successful: list[str] = []
    for path, value in updates.items():
        set(
            data,
            path,
            value,
            allow_list_growth=allow_list_growth,
            create_intermediates=create_intermediates,
        )
        successful.append(path)
    return successful
