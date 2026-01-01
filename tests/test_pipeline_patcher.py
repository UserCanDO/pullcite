"""
Tests for pipeline/patcher.py - Patch application and validation.
"""

import pytest
from pydantic import BaseModel, ConfigDict
from pullcite.pipeline.patcher import (
    Patch,
    PatchResult,
    Patcher,
    PatchError,
    create_patches,
)


class SimpleSchema(BaseModel):
    """Simple schema for testing."""

    name: str
    value: int


class NestedSchema(BaseModel):
    """Nested schema for testing."""

    model_config = ConfigDict(extra="allow")

    info: dict
    items: list


class TestPatch:
    """Test Patch dataclass."""

    def test_basic_creation(self):
        patch = Patch(
            path="name",
            old_value="old",
            new_value="new",
            reason="correction",
        )
        assert patch.path == "name"
        assert patch.old_value == "old"
        assert patch.new_value == "new"
        assert patch.reason == "correction"

    def test_without_reason(self):
        patch = Patch(path="value", old_value=1, new_value=2)
        assert patch.reason is None

    def test_invalid_path_raises(self):
        with pytest.raises(ValueError):
            Patch(path="invalid..path", old_value=1, new_value=2)

    def test_complex_path(self):
        patch = Patch(
            path="info.nested.value",
            old_value=None,
            new_value=100,
        )
        assert patch.path == "info.nested.value"


class TestPatchResult:
    """Test PatchResult dataclass."""

    def test_success_result(self):
        patch = Patch(path="name", old_value="a", new_value="b")
        result = PatchResult(patch=patch, success=True)

        assert result.success is True
        assert result.error is None

    def test_failure_result(self):
        patch = Patch(path="name", old_value="a", new_value="b")
        result = PatchResult(patch=patch, success=False, error="Value mismatch")

        assert result.success is False
        assert result.error == "Value mismatch"


class TestPatcher:
    """Test Patcher class."""

    def test_apply_single_patch(self):
        patcher = Patcher(schema=SimpleSchema)

        data = {"name": "old", "value": 10}
        patches = [Patch(path="name", old_value="old", new_value="new")]

        result = patcher.apply(data, patches)

        assert result["name"] == "new"
        assert result["value"] == 10  # Unchanged
        assert data["name"] == "old"  # Original unchanged

    def test_apply_multiple_patches(self):
        patcher = Patcher(schema=SimpleSchema)

        data = {"name": "a", "value": 1}
        patches = [
            Patch(path="name", old_value="a", new_value="b"),
            Patch(path="value", old_value=1, new_value=2),
        ]

        result = patcher.apply(data, patches)

        assert result["name"] == "b"
        assert result["value"] == 2

    def test_apply_nested_patch(self):
        patcher = Patcher(schema=NestedSchema)

        data = {
            "info": {"level": 1, "detail": "test"},
            "items": [1, 2, 3],
        }
        patches = [Patch(path="info.level", old_value=1, new_value=2)]

        result = patcher.apply(data, patches)

        assert result["info"]["level"] == 2
        assert result["info"]["detail"] == "test"  # Unchanged

    def test_apply_list_index_patch(self):
        patcher = Patcher(schema=NestedSchema)

        data = {
            "info": {},
            "items": ["a", "b", "c"],
        }
        patches = [Patch(path="items[1]", old_value="b", new_value="B")]

        result = patcher.apply(data, patches)

        assert result["items"] == ["a", "B", "c"]

    def test_old_value_mismatch_non_strict(self):
        patcher = Patcher(schema=SimpleSchema, strict=False)

        data = {"name": "actual", "value": 10}
        patches = [Patch(path="name", old_value="expected", new_value="new")]

        result = patcher.apply(data, patches)

        # Patch should fail but not raise
        assert result["name"] == "actual"  # Unchanged
        assert len(patcher.failed_patches) == 1

    def test_old_value_mismatch_strict(self):
        patcher = Patcher(schema=SimpleSchema, strict=True)

        data = {"name": "actual", "value": 10}
        patches = [Patch(path="name", old_value="expected", new_value="new")]

        with pytest.raises(PatchError):
            patcher.apply(data, patches)

    def test_applied_results(self):
        patcher = Patcher(schema=SimpleSchema)

        data = {"name": "old", "value": 10}
        patches = [Patch(path="name", old_value="old", new_value="new")]

        patcher.apply(data, patches)

        results = patcher.applied_results
        assert len(results) == 1
        assert results[0].success is True

    def test_successful_patches(self):
        patcher = Patcher(schema=SimpleSchema, strict=False)

        data = {"name": "a", "value": 1}
        patches = [
            Patch(path="name", old_value="a", new_value="b"),
            Patch(path="value", old_value=999, new_value=2),  # Will fail
        ]

        patcher.apply(data, patches)

        successful = patcher.successful_patches
        assert len(successful) == 1
        assert successful[0].path == "name"

    def test_failed_patches(self):
        patcher = Patcher(schema=SimpleSchema, strict=False)

        data = {"name": "a", "value": 1}
        patches = [
            Patch(path="name", old_value="wrong", new_value="b"),
            Patch(path="value", old_value=1, new_value=2),
        ]

        patcher.apply(data, patches)

        failed = patcher.failed_patches
        assert len(failed) == 1
        assert failed[0][0].path == "name"
        assert "doesn't match" in failed[0][1]


class TestCreatePatches:
    """Test create_patches helper function."""

    def test_basic_creation(self):
        corrections = {"name": "new_name", "value": 42}
        original = {"name": "old_name", "value": 10}

        patches = create_patches(corrections, original, reason="test")

        assert len(patches) == 2

        name_patch = next(p for p in patches if p.path == "name")
        assert name_patch.old_value == "old_name"
        assert name_patch.new_value == "new_name"
        assert name_patch.reason == "test"

        value_patch = next(p for p in patches if p.path == "value")
        assert value_patch.old_value == 10
        assert value_patch.new_value == 42

    def test_missing_original_value(self):
        corrections = {"new_field": "value"}
        original = {"other": "data"}

        patches = create_patches(corrections, original)

        assert len(patches) == 1
        assert patches[0].old_value is None
        assert patches[0].new_value == "value"

    def test_nested_path(self):
        corrections = {"info.level": 2}
        original = {"info": {"level": 1}}

        patches = create_patches(corrections, original)

        assert len(patches) == 1
        assert patches[0].old_value == 1
        assert patches[0].new_value == 2


class TestPatcherSchemaValidation:
    """Test that Patcher validates against schema."""

    def test_valid_after_patch(self):
        patcher = Patcher(schema=SimpleSchema)

        data = {"name": "test", "value": 10}
        patches = [Patch(path="value", old_value=10, new_value=20)]

        result = patcher.apply(data, patches)

        # Should succeed - result is valid
        assert result["value"] == 20

    def test_invalid_after_patch_non_strict(self):
        patcher = Patcher(schema=SimpleSchema, strict=False)

        data = {"name": "test", "value": 10}
        patches = [Patch(path="value", old_value=10, new_value="not an int")]

        # Non-strict mode: returns result even if invalid
        result = patcher.apply(data, patches)
        assert result["value"] == "not an int"

    def test_invalid_after_patch_strict(self):
        patcher = Patcher(schema=SimpleSchema, strict=True)

        data = {"name": "test", "value": 10}
        patches = [Patch(path="value", old_value=10, new_value="not an int")]

        with pytest.raises(PatchError) as exc:
            patcher.apply(data, patches)
        assert "validation failed" in str(exc.value).lower()


class TestDeepCopy:
    """Test that patches don't modify original data."""

    def test_nested_dict_isolation(self):
        patcher = Patcher(schema=NestedSchema)

        original = {
            "info": {"nested": {"deep": 1}},
            "items": [],
        }
        patches = [Patch(path="info.nested.deep", old_value=1, new_value=2)]

        result = patcher.apply(original, patches)

        # Original should be unchanged
        assert original["info"]["nested"]["deep"] == 1
        assert result["info"]["nested"]["deep"] == 2

    def test_list_isolation(self):
        patcher = Patcher(schema=NestedSchema)

        original = {
            "info": {},
            "items": [1, 2, 3],
        }
        patches = [Patch(path="items[0]", old_value=1, new_value=100)]

        result = patcher.apply(original, patches)

        # Original should be unchanged
        assert original["items"][0] == 1
        assert result["items"][0] == 100
