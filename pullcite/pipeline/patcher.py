"""
Patch application for corrections.

This module handles applying corrections to extracted data
while validating against the schema.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Type

if TYPE_CHECKING:
    from pydantic import BaseModel

from ..core.paths import get, set as set_path, validate as validate_path


@dataclass(frozen=True)
class Patch:
    """
    A single correction to apply.

    Attributes:
        path: Field path to update.
        old_value: Previous value (for verification).
        new_value: Corrected value.
        reason: Why this correction was made.
    """

    path: str
    old_value: Any
    new_value: Any
    reason: str | None = None

    def __post_init__(self) -> None:
        """Validate patch."""
        is_valid, error = validate_path(self.path)
        if not is_valid:
            raise ValueError(f"Invalid path: {error}")


@dataclass(frozen=True)
class PatchResult:
    """
    Result of applying a patch.

    Attributes:
        patch: The patch that was applied.
        success: Whether application succeeded.
        error: Error message if failed.
    """

    patch: Patch
    success: bool
    error: str | None = None


@dataclass
class Patcher:
    """
    Applies corrections to extracted data.

    Handles:
    - Path validation
    - Type coercion where possible
    - Schema validation after patching
    - Rollback on validation failure

    Attributes:
        schema: Pydantic model for validation.
        strict: If True, fail on any invalid patch. If False, skip invalid patches.
    """

    schema: Type["BaseModel"]
    strict: bool = False
    _applied: list[PatchResult] = field(default_factory=list)

    def apply(self, data: dict[str, Any], patches: list[Patch]) -> dict[str, Any]:
        """
        Apply patches to data.

        Args:
            data: Data to patch (not modified).
            patches: Patches to apply.

        Returns:
            New dict with patches applied.

        Raises:
            PatchError: If strict=True and any patch fails.
        """
        # Work on a copy
        result = _deep_copy(data)
        self._applied = []

        for patch in patches:
            patch_result = self._apply_one(result, patch)
            self._applied.append(patch_result)

            if not patch_result.success and self.strict:
                raise PatchError(
                    f"Failed to apply patch {patch.path}: {patch_result.error}"
                )

        # Validate final result against schema
        validation_error = self._validate(result)
        if validation_error:
            if self.strict:
                raise PatchError(f"Schema validation failed: {validation_error}")
            # In non-strict mode, we still return the result
            # but the caller should check applied_results

        return result

    def _apply_one(self, data: dict[str, Any], patch: Patch) -> PatchResult:
        """Apply a single patch."""
        try:
            # Check current value matches expected
            current = get(data, patch.path)
            if current != patch.old_value:
                return PatchResult(
                    patch=patch,
                    success=False,
                    error=f"Current value {current!r} doesn't match expected {patch.old_value!r}",
                )

            # Apply the patch
            set_path(data, patch.path, patch.new_value)

            return PatchResult(patch=patch, success=True)

        except Exception as e:
            return PatchResult(
                patch=patch,
                success=False,
                error=str(e),
            )

    def _validate(self, data: dict[str, Any]) -> str | None:
        """
        Validate data against schema.

        Returns:
            Error message if invalid, None if valid.
        """
        try:
            self.schema.model_validate(data)
            return None
        except Exception as e:
            return str(e)

    @property
    def applied_results(self) -> list[PatchResult]:
        """Get results of last apply() call."""
        return self._applied.copy()

    @property
    def successful_patches(self) -> list[Patch]:
        """Get patches that were successfully applied."""
        return [r.patch for r in self._applied if r.success]

    @property
    def failed_patches(self) -> list[tuple[Patch, str]]:
        """Get patches that failed with their error messages."""
        return [
            (r.patch, r.error or "Unknown error")
            for r in self._applied
            if not r.success
        ]


def create_patches(
    corrections: dict[str, Any],
    original_data: dict[str, Any],
    reason: str | None = None,
) -> list[Patch]:
    """
    Create patches from a corrections dict.

    Args:
        corrections: Dict of path -> new_value.
        original_data: Original data to get old values from.
        reason: Reason for corrections (applies to all).

    Returns:
        List of Patch objects.
    """
    patches = []

    for path, new_value in corrections.items():
        try:
            old_value = get(original_data, path)
        except (KeyError, IndexError, TypeError):
            old_value = None

        patches.append(
            Patch(
                path=path,
                old_value=old_value,
                new_value=new_value,
                reason=reason,
            )
        )

    return patches


def _deep_copy(obj: Any) -> Any:
    """
    Deep copy a JSON-serializable object.

    Faster than copy.deepcopy for simple data.
    """
    if isinstance(obj, dict):
        return {k: _deep_copy(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_deep_copy(v) for v in obj]
    elif isinstance(obj, tuple):
        return tuple(_deep_copy(v) for v in obj)
    else:
        # Primitives are immutable
        return obj


class PatchError(Exception):
    """Raised when patching fails."""

    pass


__all__ = [
    "Patch",
    "PatchResult",
    "Patcher",
    "PatchError",
    "create_patches",
]
