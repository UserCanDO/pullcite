"""
Chunk - A piece of a document with location information.

Chunks are immutable value objects that know where text came from
in the original document. They do NOT contain embeddings - that's
the Retriever's job.

Key properties:
- Immutable (frozen dataclass)
- Stable metadata storage (sorted tuple of pairs)
- Validation on construction
- JSON serializable
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any


def _normalize_metadata(
    metadata: dict[str, Any] | tuple | None,
) -> tuple[tuple[str, Any], ...]:
    """
    Normalize metadata to a sorted tuple of pairs.

    This ensures:
    - Immutability (tuple, not dict)
    - Deterministic ordering (sorted by key)
    - Last-write-wins on duplicates
    - Stable serialization

    Args:
        metadata: Dict, tuple of pairs, or None.

    Returns:
        Tuple of (key, value) pairs, sorted by key.
    """
    if metadata is None:
        return ()

    if isinstance(metadata, tuple):
        # Already tuple of pairs - convert to dict to dedupe, then back
        d = dict(metadata)
    elif isinstance(metadata, dict):
        d = metadata
    else:
        raise TypeError(
            f"metadata must be dict or tuple of pairs, got {type(metadata).__name__}"
        )

    # Sort by key for deterministic ordering
    return tuple(sorted(d.items(), key=lambda x: x[0]))


def _validate_json_serializable(value: Any, path: str = "value") -> None:
    """
    Validate that a value is JSON serializable.

    Args:
        value: Value to check.
        path: Path for error messages.

    Raises:
        TypeError: If value is not JSON serializable.
    """
    try:
        json.dumps(value)
    except (TypeError, ValueError) as e:
        raise TypeError(f"Metadata {path} is not JSON serializable: {e}")


@dataclass(frozen=True)
class Chunk:
    """
    A piece of a document with location information.

    Immutable value object. Knows where text came from, nothing else.
    Does NOT contain embeddings - that's the Retriever's responsibility.

    Attributes:
        index: Position in document (0, 1, 2, ...). Must be >= 0.
        text: The actual text content of this chunk.
        page: Page number (1-indexed). None if unknown or not applicable.
        bbox: Bounding box as (x0, y0, x1, y1) in PDF points. None if unknown.
        metadata: Additional info as tuple of (key, value) pairs, sorted by key.

    Invariants:
        - index >= 0
        - page is None or page >= 1
        - bbox is None or has exactly 4 float values
        - metadata is sorted tuple of pairs (immutable, deterministic)
    """

    index: int
    text: str
    page: int | None = None
    bbox: tuple[float, float, float, float] | None = None
    metadata: tuple[tuple[str, Any], ...] = ()

    def __post_init__(self) -> None:
        """Validate chunk invariants."""
        # Validate index
        if self.index < 0:
            raise ValueError(f"index must be >= 0, got {self.index}")

        # Validate page
        if self.page is not None and self.page < 1:
            raise ValueError(f"page must be >= 1, got {self.page}")

        # Validate bbox
        if self.bbox is not None:
            if not isinstance(self.bbox, tuple):
                raise TypeError(f"bbox must be a tuple, got {type(self.bbox).__name__}")
            if len(self.bbox) != 4:
                raise ValueError(f"bbox must have 4 values, got {len(self.bbox)}")
            # Ensure all values are numeric
            for i, v in enumerate(self.bbox):
                if not isinstance(v, (int, float)):
                    raise TypeError(
                        f"bbox[{i}] must be numeric, got {type(v).__name__}"
                    )

        # Normalize metadata if it came in as dict (handles __init__ call)
        # Since frozen, we use object.__setattr__
        if isinstance(self.metadata, dict):
            object.__setattr__(self, "metadata", _normalize_metadata(self.metadata))
        elif self.metadata is None:
            object.__setattr__(self, "metadata", ())
        elif not isinstance(self.metadata, tuple):
            raise TypeError(
                f"metadata must be dict or tuple, got {type(self.metadata).__name__}"
            )

    def get_metadata(self, key: str, default: Any = None) -> Any:
        """
        Get a metadata value by key.

        Args:
            key: Metadata key to look up.
            default: Value to return if key not found.

        Returns:
            Metadata value or default.
        """
        for k, v in self.metadata:
            if k == key:
                return v
        return default

    def with_metadata(self, **kwargs: Any) -> Chunk:
        """
        Return a new Chunk with additional/updated metadata.

        Since Chunk is frozen, this creates a copy with merged metadata.
        Later values override earlier ones (last-write-wins).

        Args:
            **kwargs: Metadata key-value pairs to add/update.

        Returns:
            New Chunk with merged metadata.

        Raises:
            TypeError: If any value is not JSON serializable.
        """
        # Validate new values are JSON serializable
        for key, value in kwargs.items():
            _validate_json_serializable(value, f"'{key}'")

        # Merge existing metadata with new
        current = dict(self.metadata)
        current.update(kwargs)

        return Chunk(
            index=self.index,
            text=self.text,
            page=self.page,
            bbox=self.bbox,
            metadata=_normalize_metadata(current),
        )

    def with_page(self, page: int | None) -> Chunk:
        """
        Return a new Chunk with updated page number.

        Args:
            page: New page number (1-indexed) or None.

        Returns:
            New Chunk with updated page.
        """
        return Chunk(
            index=self.index,
            text=self.text,
            page=page,
            bbox=self.bbox,
            metadata=self.metadata,
        )

    def with_bbox(self, bbox: tuple[float, float, float, float] | None) -> Chunk:
        """
        Return a new Chunk with updated bounding box.

        Args:
            bbox: New bounding box (x0, y0, x1, y1) or None.

        Returns:
            New Chunk with updated bbox.
        """
        return Chunk(
            index=self.index,
            text=self.text,
            page=self.page,
            bbox=bbox,
            metadata=self.metadata,
        )

    def to_dict(self) -> dict[str, Any]:
        """
        Convert to dictionary for serialization.

        Returns:
            Dict with all chunk attributes.
            Metadata is converted to a plain dict.
        """
        return {
            "index": self.index,
            "text": self.text,
            "page": self.page,
            "bbox": list(self.bbox) if self.bbox else None,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Chunk:
        """
        Create a Chunk from a dictionary.

        Args:
            data: Dict with chunk attributes.

        Returns:
            New Chunk instance.
        """
        bbox = data.get("bbox")
        if bbox is not None and isinstance(bbox, list):
            bbox = tuple(bbox)

        metadata = data.get("metadata")
        if metadata is not None and isinstance(metadata, dict):
            metadata = _normalize_metadata(metadata)

        return cls(
            index=data["index"],
            text=data["text"],
            page=data.get("page"),
            bbox=bbox,
            metadata=metadata or (),
        )

    def __len__(self) -> int:
        """Return length of text."""
        return len(self.text)

    def __repr__(self) -> str:
        """Concise representation for debugging."""
        text_preview = self.text[:50] + "..." if len(self.text) > 50 else self.text
        text_preview = text_preview.replace("\n", "\\n")

        parts = [f"index={self.index}"]
        if self.page is not None:
            parts.append(f"page={self.page}")
        parts.append(f"text={text_preview!r}")

        return f"Chunk({', '.join(parts)})"
