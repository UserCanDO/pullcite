"""
Document - A loaded document ready for extraction.

Uses Docling for PDF/DOCX parsing, preserving tables and layout.
Chunking is explicit and configurable - critical for extraction quality.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from io import BytesIO
from pathlib import Path
from typing import Any, Iterator

from .chunk import Chunk
from .chunker import Chunker, ChunkConfig, SentenceChunker, DEFAULT_CHUNKER

# Optional docling imports - will be imported lazily to avoid hard dependency
try:
    from docling.datamodel.document import DocumentStream
    from docling.document_converter import DocumentConverter
except ImportError:
    DocumentStream = None
    DocumentConverter = None


def _compute_content_hash(data: bytes) -> str:
    """Compute deterministic hash from content bytes."""
    return hashlib.sha256(data).hexdigest()[:16]


def _chunker_from_params(
    chunker: Chunker | None,
    chunk_size: int | None,
    chunk_overlap: int | None,
) -> Chunker:
    """
    Resolve chunker from explicit chunker or legacy size/overlap params.

    Priority:
    1. Explicit chunker object
    2. chunk_size/chunk_overlap params (legacy, creates SentenceChunker)
    3. DEFAULT_CHUNKER
    """
    if chunker is not None:
        return chunker
    if chunk_size is not None:
        overlap = chunk_overlap if chunk_overlap is not None else chunk_size // 6
        return SentenceChunker(target_size=chunk_size, overlap=overlap)
    return DEFAULT_CHUNKER


def _apply_chunker(text: str, chunker: Chunker) -> list[ChunkConfig]:
    """Apply chunker to text and return chunk configs."""
    return chunker.chunk(text)


@dataclass
class Document:
    """
    A loaded document ready for extraction.

    Chunking Strategy:
        Chunking directly affects extraction quality. You can configure it:

        1. Using a Chunker object (recommended):
           >>> from pullcite import SlidingWindowChunker
           >>> doc = Document.from_file("doc.pdf", chunker=SlidingWindowChunker(size=500, stride=250))

        2. Using size/overlap params (legacy):
           >>> doc = Document.from_file("doc.pdf", chunk_size=500, chunk_overlap=100)

        3. Using defaults (SentenceChunker with 1200 char target, 200 char overlap):
           >>> doc = Document.from_file("doc.pdf")

    Attributes:
        id: Unique document identifier (content hash if not provided).
        filename: Original filename for reference.
        chunks: List of Chunk objects for search/extraction.
        page_count: Number of pages (if available from parser).
        metadata: Additional document metadata.
    """

    id: str
    filename: str
    chunks: list[Chunk] = field(default_factory=list)
    page_count: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_file(
        cls,
        path: str | Path,
        *,
        chunker: Chunker | None = None,
        chunk_size: int | None = None,
        chunk_overlap: int | None = None,
        document_id: str | None = None,
    ) -> Document:
        """
        Load a document from a file path.

        Auto-detects format from extension. Uses Docling for PDF/DOCX/images,
        plain text loading for .txt files.

        Args:
            path: Path to document file.
            chunker: Chunking strategy. See SlidingWindowChunker, SentenceChunker.
            chunk_size: Target chunk size (legacy, prefer chunker param).
            chunk_overlap: Chunk overlap (legacy, prefer chunker param).
            document_id: Optional explicit ID. If None, generated from content hash.

        Returns:
            Document with chunks ready for extraction.

        Example:
            >>> # Explicit chunking (recommended)
            >>> doc = Document.from_file(
            ...     "report.pdf",
            ...     chunker=SlidingWindowChunker(size=500, stride=250),
            ... )
            >>>
            >>> # Using defaults
            >>> doc = Document.from_file("report.pdf")
        """
        path = Path(path)
        suffix = path.suffix.lower()

        # Plain text files - no Docling needed
        if suffix in {".txt", ".text"}:
            text = path.read_text(encoding="utf-8")
            return cls.from_text(
                text,
                filename=path.name,
                chunker=chunker,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                document_id=document_id,
            )

        # For PDF, DOCX, images, etc. - use Docling
        resolved_chunker = _chunker_from_params(chunker, chunk_size, chunk_overlap)
        return cls._from_docling_path(
            path=str(path),
            chunker=resolved_chunker,
            document_id=document_id,
        )

    @classmethod
    def _from_docling_path(
        cls,
        path: str,
        chunker: Chunker,
        document_id: str | None = None,
    ) -> Document:
        """Internal: Load via Docling from file path."""
        if DocumentConverter is None:
            raise ImportError(
                "Docling support requires docling. "
                "Install with: pip install 'pullcite[docling]'"
            )

        # Read raw bytes for content hash
        with open(path, "rb") as f:
            raw_bytes = f.read()

        # Parse with Docling
        converter = DocumentConverter()
        result = converter.convert(path)

        return cls._from_docling_result(
            docling_doc=result.document,
            raw_bytes=raw_bytes,
            filename=path.split("/")[-1],
            chunker=chunker,
            document_id=document_id,
        )

    @classmethod
    def _from_docling_result(
        cls,
        docling_doc: Any,
        raw_bytes: bytes,
        filename: str,
        chunker: Chunker,
        document_id: str | None,
    ) -> Document:
        """
        Internal method to create Document from a Docling conversion result.

        Args:
            docling_doc: The Docling document object from conversion.
            raw_bytes: Raw bytes of the source file (for hashing).
            filename: Filename for reference.
            chunker: Chunker instance for splitting long text.
            document_id: Optional explicit ID.

        Returns:
            Document with chunks extracted.
        """
        if document_id is None:
            document_id = _compute_content_hash(raw_bytes)

        chunks: list[Chunk] = []
        chunk_index = 0

        # Extract tables as dedicated chunks (high value for SBC)
        for table in docling_doc.tables:
            table_md = table.export_to_markdown()
            if table_md.strip():
                # Get page number if available
                page = None
                bbox = None
                prov = table.prov
                if prov and len(prov) > 0:
                    page = prov[0].page_no
                    bbox_obj = prov[0].bbox
                    if bbox_obj:
                        bbox = (bbox_obj.l, bbox_obj.t, bbox_obj.r, bbox_obj.b)

                chunks.append(
                    Chunk(
                        index=chunk_index,
                        text=table_md,
                        page=page,
                        bbox=bbox,
                        metadata={"type": "table", "source": "docling"},
                    )
                )
                chunk_index += 1

        # Extract text content (non-table)
        # Use iterate_items for structured access
        for item, level in docling_doc.iterate_items():
            if hasattr(item, "text") and item.text:
                text = item.text.strip()
                if not text:
                    continue

                # Skip if this is part of a table (already extracted)
                if hasattr(item, "label") and "table" in str(item.label).lower():
                    continue

                # Get provenance
                page = None
                bbox = None
                if hasattr(item, "prov") and item.prov and len(item.prov) > 0:
                    prov = item.prov[0]
                    page = prov.page_no
                    if hasattr(prov, "bbox") and prov.bbox:
                        bbox = (prov.bbox.l, prov.bbox.t, prov.bbox.r, prov.bbox.b)

                # Apply chunker to long text
                chunk_configs = _apply_chunker(text, chunker)
                for cfg in chunk_configs:
                    chunks.append(
                        Chunk(
                            index=chunk_index,
                            text=cfg.text,
                            page=page,
                            bbox=bbox,  # Same bbox for all splits (approximate)
                            metadata={"type": "text", "source": "docling"},
                        )
                    )
                    chunk_index += 1

        return cls(
            id=document_id,
            filename=filename,
            chunks=chunks,
            page_count=(
                docling_doc.num_pages if hasattr(docling_doc, "num_pages") else None
            ),
            metadata={
                "source": "docling",
                "content_hash": _compute_content_hash(raw_bytes),
                "chunker": type(chunker).__name__,
            },
        )

    @classmethod
    def from_docling_file_path(
        cls,
        path: str,
        *,
        chunker: Chunker | None = None,
        chunk_size: int | None = None,
        chunk_overlap: int | None = None,
        document_id: str | None = None,
    ) -> Document:
        """
        Load a document from a file path using Docling.

        Supports PDF, DOCX, HTML, images, and more.
        Preserves tables, layout, and page information.

        Note: Prefer Document.from_file() which auto-detects format.

        Args:
            path: Path to document file.
            chunker: Chunking strategy (recommended).
            chunk_size: Target chunk size (legacy, prefer chunker).
            chunk_overlap: Overlap between chunks (legacy).
            document_id: Optional explicit ID.

        Returns:
            Document with chunks extracted.
        """
        resolved_chunker = _chunker_from_params(chunker, chunk_size, chunk_overlap)
        return cls._from_docling_path(
            path=path,
            chunker=resolved_chunker,
            document_id=document_id,
        )

    @classmethod
    def from_docling_bytes_io(
        cls,
        data: BytesIO | bytes,
        filename: str = "document.pdf",
        *,
        chunker: Chunker | None = None,
        chunk_size: int | None = None,
        chunk_overlap: int | None = None,
        document_id: str | None = None,
    ) -> Document:
        """
        Load a document from bytes or BytesIO using Docling.

        Supports PDF, DOCX, HTML, images, and more.
        Preserves tables, layout, and page information.

        Args:
            data: Document as bytes or BytesIO.
            filename: Filename for reference (must include extension to infer format).
            chunker: Chunking strategy (recommended).
            chunk_size: Target chunk size (legacy, prefer chunker).
            chunk_overlap: Overlap between chunks (legacy).
            document_id: Optional explicit ID.

        Returns:
            Document with chunks extracted.
        """
        if DocumentConverter is None or DocumentStream is None:
            raise ImportError(
                "Docling support requires docling. "
                "Install with: pip install 'pullcite[docling]'"
            )

        resolved_chunker = _chunker_from_params(chunker, chunk_size, chunk_overlap)

        # Normalize input
        if isinstance(data, bytes):
            raw_bytes = data
            stream = BytesIO(data)
        else:
            data.seek(0)
            raw_bytes = data.read()
            data.seek(0)
            stream = data

        # Create DocumentStream for converter
        doc_stream = DocumentStream(name=filename, stream=stream)

        # Parse with Docling
        converter = DocumentConverter()
        result = converter.convert(doc_stream)

        return cls._from_docling_result(
            docling_doc=result.document,
            raw_bytes=raw_bytes,
            filename=filename,
            chunker=resolved_chunker,
            document_id=document_id,
        )

    @classmethod
    def from_text(
        cls,
        text: str,
        filename: str = "document.txt",
        *,
        chunker: Chunker | None = None,
        chunk_size: int | None = None,
        chunk_overlap: int | None = None,
        document_id: str | None = None,
    ) -> Document:
        """
        Create a document from plain text.

        No page numbers or bounding boxes available.

        Args:
            text: Plain text content.
            filename: Filename for reference.
            chunker: Chunking strategy (recommended).
            chunk_size: Target chunk size (legacy, prefer chunker).
            chunk_overlap: Overlap between chunks (legacy).
            document_id: Optional explicit ID.

        Returns:
            Document with chunks extracted.
        """
        raw_bytes = text.encode("utf-8")
        if document_id is None:
            document_id = _compute_content_hash(raw_bytes)

        resolved_chunker = _chunker_from_params(chunker, chunk_size, chunk_overlap)
        chunk_configs = _apply_chunker(text, resolved_chunker)

        chunks = [
            Chunk(
                index=i,
                text=cfg.text,
                page=None,
                bbox=None,
                metadata={"source": "text", "filename": filename},
            )
            for i, cfg in enumerate(chunk_configs)
            if cfg.text.strip()
        ]

        return cls(
            id=document_id,
            filename=filename,
            chunks=chunks,
            page_count=None,
            metadata={
                "source": "text",
                "content_hash": _compute_content_hash(raw_bytes),
                "chunker": type(resolved_chunker).__name__,
            },
        )

    @classmethod
    def from_chunks(
        cls,
        chunks: list[Chunk],
        filename: str = "document",
        document_id: str | None = None,
        page_count: int | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Document:
        """
        Create a document from pre-built chunks.

        Useful for testing or when you have custom chunking logic.

        Args:
            chunks: List of Chunk objects.
            filename: Filename for reference.
            document_id: Optional explicit ID. If None, generated from chunk text.
            page_count: Optional page count.
            metadata: Optional document metadata.

        Returns:
            Document with provided chunks.
        """
        if document_id is None:
            # Hash all chunk text
            combined = "".join(c.text for c in chunks)
            document_id = _compute_content_hash(combined.encode("utf-8"))

        return cls(
            id=document_id,
            filename=filename,
            chunks=chunks,
            page_count=page_count,
            metadata=metadata or {},
        )

    @property
    def full_text(self) -> str:
        """
        Get all text concatenated.

        Useful for passing to extraction prompt when document
        is small enough to fit in context.

        Returns:
            All chunk text joined with newlines.
        """
        return "\n".join(chunk.text for chunk in self.chunks)

    def iter_chunks(self) -> Iterator[Chunk]:
        """
        Iterate over chunks.

        Yields:
            Chunk objects in order.
        """
        return iter(self.chunks)

    def get_chunk(self, index: int) -> Chunk | None:
        """
        Get a specific chunk by index.

        Args:
            index: Chunk index (0-based).

        Returns:
            Chunk if found, None otherwise.
        """
        for chunk in self.chunks:
            if chunk.index == index:
                return chunk
        return None

    def get_chunks_by_page(self, page: int) -> list[Chunk]:
        """
        Get all chunks from a specific page.

        Args:
            page: Page number (1-indexed).

        Returns:
            List of chunks on that page.
        """
        return [c for c in self.chunks if c.page == page]

    def __len__(self) -> int:
        """Return number of chunks."""
        return len(self.chunks)

    def __iter__(self) -> Iterator[Chunk]:
        """Iterate over chunks."""
        return iter(self.chunks)

    def to_dict(self) -> dict[str, Any]:
        """
        Convert to dictionary for serialization.

        Returns:
            Dict with all document attributes including chunks.
        """
        return {
            "id": self.id,
            "filename": self.filename,
            "chunks": [c.to_dict() for c in self.chunks],
            "page_count": self.page_count,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Document:
        """
        Create a Document from a dictionary.

        Args:
            data: Dict with document attributes.

        Returns:
            New Document instance.
        """
        chunks = [Chunk.from_dict(c) for c in data.get("chunks", [])]

        return cls(
            id=data["id"],
            filename=data["filename"],
            chunks=chunks,
            page_count=data.get("page_count"),
            metadata=data.get("metadata", {}),
        )

    def __repr__(self) -> str:
        """Concise representation for debugging."""
        return (
            f"Document(id={self.id!r}, filename={self.filename!r}, "
            f"chunks={len(self.chunks)}, pages={self.page_count})"
        )
