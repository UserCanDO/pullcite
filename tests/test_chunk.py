"""
Tests for Chunk - immutable document chunks.
"""

import pytest
import json
from pullcite.core.chunk import Chunk, _normalize_metadata


class TestChunkCreation:
    """Test basic chunk creation."""

    def test_minimal_chunk(self):
        chunk = Chunk(index=0, text="Hello world")
        assert chunk.index == 0
        assert chunk.text == "Hello world"
        assert chunk.page is None
        assert chunk.bbox is None
        assert chunk.metadata == ()

    def test_full_chunk(self):
        chunk = Chunk(
            index=5,
            text="Sample text",
            page=3,
            bbox=(10.0, 20.0, 100.0, 50.0),
            metadata=(("key", "value"),),
        )
        assert chunk.index == 5
        assert chunk.text == "Sample text"
        assert chunk.page == 3
        assert chunk.bbox == (10.0, 20.0, 100.0, 50.0)
        assert chunk.metadata == (("key", "value"),)

    def test_metadata_as_dict(self):
        """Metadata can be passed as dict and gets normalized."""
        chunk = Chunk(
            index=0,
            text="text",
            metadata={"b": 2, "a": 1},
        )
        # Should be sorted tuple of pairs
        assert chunk.metadata == (("a", 1), ("b", 2))

    def test_metadata_none_becomes_empty(self):
        chunk = Chunk(index=0, text="text", metadata=None)
        assert chunk.metadata == ()


class TestChunkValidation:
    """Test validation invariants."""

    def test_negative_index_raises(self):
        with pytest.raises(ValueError) as exc:
            Chunk(index=-1, text="text")
        assert "index must be >= 0" in str(exc.value)

    def test_zero_index_ok(self):
        chunk = Chunk(index=0, text="text")
        assert chunk.index == 0

    def test_page_zero_raises(self):
        with pytest.raises(ValueError) as exc:
            Chunk(index=0, text="text", page=0)
        assert "page must be >= 1" in str(exc.value)

    def test_page_one_ok(self):
        chunk = Chunk(index=0, text="text", page=1)
        assert chunk.page == 1

    def test_negative_page_raises(self):
        with pytest.raises(ValueError):
            Chunk(index=0, text="text", page=-1)

    def test_bbox_wrong_length_raises(self):
        with pytest.raises(ValueError) as exc:
            Chunk(index=0, text="text", bbox=(1.0, 2.0, 3.0))
        assert "bbox must have 4 values" in str(exc.value)

    def test_bbox_non_numeric_raises(self):
        with pytest.raises(TypeError) as exc:
            Chunk(index=0, text="text", bbox=(1.0, 2.0, "3", 4.0))
        assert "must be numeric" in str(exc.value)

    def test_bbox_list_raises(self):
        """bbox must be tuple, not list."""
        with pytest.raises(TypeError) as exc:
            Chunk(index=0, text="text", bbox=[1.0, 2.0, 3.0, 4.0])
        assert "must be a tuple" in str(exc.value)


class TestChunkImmutability:
    """Test that chunks are truly immutable."""

    def test_cannot_modify_index(self):
        chunk = Chunk(index=0, text="text")
        with pytest.raises(AttributeError):
            chunk.index = 1

    def test_cannot_modify_text(self):
        chunk = Chunk(index=0, text="text")
        with pytest.raises(AttributeError):
            chunk.text = "new text"

    def test_cannot_modify_page(self):
        chunk = Chunk(index=0, text="text", page=1)
        with pytest.raises(AttributeError):
            chunk.page = 2

    def test_cannot_modify_metadata(self):
        chunk = Chunk(index=0, text="text", metadata={"a": 1})
        with pytest.raises(AttributeError):
            chunk.metadata = (("b", 2),)


class TestGetMetadata:
    """Test metadata access."""

    def test_get_existing_key(self):
        chunk = Chunk(index=0, text="text", metadata={"foo": "bar", "num": 42})
        assert chunk.get_metadata("foo") == "bar"
        assert chunk.get_metadata("num") == 42

    def test_get_missing_key_returns_default(self):
        chunk = Chunk(index=0, text="text", metadata={"foo": "bar"})
        assert chunk.get_metadata("missing") is None
        assert chunk.get_metadata("missing", default="N/A") == "N/A"

    def test_get_from_empty_metadata(self):
        chunk = Chunk(index=0, text="text")
        assert chunk.get_metadata("anything") is None


class TestWithMetadata:
    """Test creating new chunks with modified metadata."""

    def test_add_metadata(self):
        chunk = Chunk(index=0, text="text")
        new_chunk = chunk.with_metadata(key="value")

        # Original unchanged
        assert chunk.metadata == ()

        # New chunk has metadata
        assert new_chunk.get_metadata("key") == "value"

    def test_merge_metadata(self):
        chunk = Chunk(index=0, text="text", metadata={"a": 1, "b": 2})
        new_chunk = chunk.with_metadata(b=99, c=3)

        # Original unchanged
        assert chunk.get_metadata("b") == 2

        # New chunk has merged metadata (b overwritten)
        assert new_chunk.get_metadata("a") == 1
        assert new_chunk.get_metadata("b") == 99
        assert new_chunk.get_metadata("c") == 3

    def test_metadata_stays_sorted(self):
        chunk = Chunk(index=0, text="text")
        new_chunk = chunk.with_metadata(z=1, a=2, m=3)

        keys = [k for k, v in new_chunk.metadata]
        assert keys == ["a", "m", "z"]

    def test_non_json_serializable_raises(self):
        chunk = Chunk(index=0, text="text")

        with pytest.raises(TypeError) as exc:
            chunk.with_metadata(bad=object())
        assert "not JSON serializable" in str(exc.value)

    def test_preserves_other_fields(self):
        chunk = Chunk(
            index=5,
            text="original",
            page=3,
            bbox=(1.0, 2.0, 3.0, 4.0),
        )
        new_chunk = chunk.with_metadata(added="value")

        assert new_chunk.index == 5
        assert new_chunk.text == "original"
        assert new_chunk.page == 3
        assert new_chunk.bbox == (1.0, 2.0, 3.0, 4.0)


class TestWithPage:
    """Test creating new chunks with modified page."""

    def test_set_page(self):
        chunk = Chunk(index=0, text="text")
        new_chunk = chunk.with_page(5)

        assert chunk.page is None  # Original unchanged
        assert new_chunk.page == 5

    def test_clear_page(self):
        chunk = Chunk(index=0, text="text", page=5)
        new_chunk = chunk.with_page(None)

        assert chunk.page == 5  # Original unchanged
        assert new_chunk.page is None


class TestWithBbox:
    """Test creating new chunks with modified bbox."""

    def test_set_bbox(self):
        chunk = Chunk(index=0, text="text")
        new_chunk = chunk.with_bbox((10.0, 20.0, 30.0, 40.0))

        assert chunk.bbox is None  # Original unchanged
        assert new_chunk.bbox == (10.0, 20.0, 30.0, 40.0)

    def test_clear_bbox(self):
        chunk = Chunk(index=0, text="text", bbox=(1.0, 2.0, 3.0, 4.0))
        new_chunk = chunk.with_bbox(None)

        assert chunk.bbox == (1.0, 2.0, 3.0, 4.0)  # Original unchanged
        assert new_chunk.bbox is None


class TestSerialization:
    """Test to_dict and from_dict."""

    def test_to_dict_minimal(self):
        chunk = Chunk(index=0, text="Hello")
        d = chunk.to_dict()

        assert d == {
            "index": 0,
            "text": "Hello",
            "page": None,
            "bbox": None,
            "metadata": {},
        }

    def test_to_dict_full(self):
        chunk = Chunk(
            index=5,
            text="Sample",
            page=3,
            bbox=(1.0, 2.0, 3.0, 4.0),
            metadata={"key": "value"},
        )
        d = chunk.to_dict()

        assert d == {
            "index": 5,
            "text": "Sample",
            "page": 3,
            "bbox": [1.0, 2.0, 3.0, 4.0],  # Converted to list for JSON
            "metadata": {"key": "value"},
        }

    def test_from_dict_minimal(self):
        d = {"index": 0, "text": "Hello"}
        chunk = Chunk.from_dict(d)

        assert chunk.index == 0
        assert chunk.text == "Hello"
        assert chunk.page is None
        assert chunk.bbox is None
        assert chunk.metadata == ()

    def test_from_dict_full(self):
        d = {
            "index": 5,
            "text": "Sample",
            "page": 3,
            "bbox": [1.0, 2.0, 3.0, 4.0],  # List from JSON
            "metadata": {"key": "value"},
        }
        chunk = Chunk.from_dict(d)

        assert chunk.index == 5
        assert chunk.text == "Sample"
        assert chunk.page == 3
        assert chunk.bbox == (1.0, 2.0, 3.0, 4.0)  # Converted to tuple
        assert chunk.get_metadata("key") == "value"

    def test_roundtrip(self):
        """to_dict -> from_dict preserves all data."""
        original = Chunk(
            index=10,
            text="Round trip test",
            page=5,
            bbox=(10.5, 20.5, 30.5, 40.5),
            metadata={"a": 1, "b": "two", "c": [1, 2, 3]},
        )

        d = original.to_dict()
        restored = Chunk.from_dict(d)

        assert restored.index == original.index
        assert restored.text == original.text
        assert restored.page == original.page
        assert restored.bbox == original.bbox
        assert dict(restored.metadata) == dict(original.metadata)

    def test_json_roundtrip(self):
        """Can serialize to JSON and back."""
        original = Chunk(
            index=1,
            text="JSON test",
            page=2,
            bbox=(0.0, 0.0, 100.0, 100.0),
            metadata={"nested": {"a": 1}},
        )

        json_str = json.dumps(original.to_dict())
        d = json.loads(json_str)
        restored = Chunk.from_dict(d)

        assert restored.index == original.index
        assert restored.text == original.text


class TestNormalizeMetadata:
    """Test the _normalize_metadata helper."""

    def test_none_becomes_empty_tuple(self):
        assert _normalize_metadata(None) == ()

    def test_dict_sorted_by_key(self):
        result = _normalize_metadata({"z": 1, "a": 2, "m": 3})
        assert result == (("a", 2), ("m", 3), ("z", 1))

    def test_tuple_deduped_last_wins(self):
        result = _normalize_metadata((("a", 1), ("a", 2)))
        assert result == (("a", 2),)

    def test_empty_dict(self):
        assert _normalize_metadata({}) == ()

    def test_invalid_type_raises(self):
        with pytest.raises(TypeError):
            _normalize_metadata("not valid")


class TestChunkLength:
    """Test __len__ method."""

    def test_len_returns_text_length(self):
        chunk = Chunk(index=0, text="Hello")
        assert len(chunk) == 5

    def test_len_empty_text(self):
        chunk = Chunk(index=0, text="")
        assert len(chunk) == 0


class TestChunkRepr:
    """Test __repr__ method."""

    def test_repr_short_text(self):
        chunk = Chunk(index=0, text="Hello")
        r = repr(chunk)
        assert "index=0" in r
        assert "Hello" in r

    def test_repr_long_text_truncated(self):
        chunk = Chunk(index=0, text="x" * 100)
        r = repr(chunk)
        assert "..." in r
        assert len(r) < 150  # Should be reasonable length

    def test_repr_with_page(self):
        chunk = Chunk(index=5, text="text", page=3)
        r = repr(chunk)
        assert "page=3" in r

    def test_repr_newlines_escaped(self):
        chunk = Chunk(index=0, text="line1\nline2")
        r = repr(chunk)
        assert "\\n" in r


class TestChunkEquality:
    """Test equality comparison (from frozen dataclass)."""

    def test_equal_chunks(self):
        c1 = Chunk(index=0, text="text", page=1)
        c2 = Chunk(index=0, text="text", page=1)
        assert c1 == c2

    def test_different_index(self):
        c1 = Chunk(index=0, text="text")
        c2 = Chunk(index=1, text="text")
        assert c1 != c2

    def test_different_text(self):
        c1 = Chunk(index=0, text="text1")
        c2 = Chunk(index=0, text="text2")
        assert c1 != c2

    def test_different_metadata(self):
        c1 = Chunk(index=0, text="text", metadata={"a": 1})
        c2 = Chunk(index=0, text="text", metadata={"a": 2})
        assert c1 != c2


class TestChunkHashability:
    """Test that chunks can be used in sets and as dict keys."""

    def test_chunk_is_hashable(self):
        chunk = Chunk(index=0, text="text")
        # Should not raise
        hash(chunk)

    def test_chunk_in_set(self):
        c1 = Chunk(index=0, text="text")
        c2 = Chunk(index=0, text="text")
        c3 = Chunk(index=1, text="other")

        s = {c1, c2, c3}
        assert len(s) == 2  # c1 and c2 are equal

    def test_chunk_as_dict_key(self):
        c1 = Chunk(index=0, text="text")
        d = {c1: "value"}

        c2 = Chunk(index=0, text="text")
        assert d[c2] == "value"  # Can look up by equal chunk


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
