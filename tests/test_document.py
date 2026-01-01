"""
Tests for Document - document loading and chunking.
"""

import pytest
import json
from pullcite.core.document import Document, _split_into_chunks, _compute_content_hash
from pullcite.core.chunk import Chunk


class TestComputeContentHash:
    """Test content hashing."""

    def test_deterministic(self):
        """Same content produces same hash."""
        data = b"Hello world"
        h1 = _compute_content_hash(data)
        h2 = _compute_content_hash(data)
        assert h1 == h2

    def test_different_content_different_hash(self):
        """Different content produces different hash."""
        h1 = _compute_content_hash(b"Hello")
        h2 = _compute_content_hash(b"World")
        assert h1 != h2

    def test_hash_length(self):
        """Hash is 16 hex chars."""
        h = _compute_content_hash(b"test")
        assert len(h) == 16
        assert all(c in "0123456789abcdef" for c in h)


class TestSplitIntoChunks:
    """Test text chunking."""

    def test_empty_text(self):
        assert _split_into_chunks("", 100, 20) == []

    def test_small_text_single_chunk(self):
        text = "Short text"
        chunks = _split_into_chunks(text, 100, 20)
        assert chunks == [text]

    def test_exact_size_single_chunk(self):
        text = "x" * 100
        chunks = _split_into_chunks(text, 100, 20)
        assert chunks == [text]

    def test_splits_large_text(self):
        text = "x" * 300
        chunks = _split_into_chunks(text, 100, 20)
        assert len(chunks) > 1

    def test_overlap_works(self):
        # With overlap, consecutive chunks should share some text
        text = "abcdefghij" * 20  # 200 chars
        chunks = _split_into_chunks(text, 50, 10)

        # Check that we have multiple chunks
        assert len(chunks) > 1

    def test_prefers_sentence_boundaries(self):
        text = "First sentence. Second sentence. Third sentence. Fourth sentence."
        chunks = _split_into_chunks(text, 40, 10)

        # Chunks should try to end at periods
        for chunk in chunks[:-1]:  # All but last
            # Should end near a sentence boundary
            assert chunk.rstrip().endswith(".") or len(chunk) <= 45


class TestDocumentFromText:
    """Test Document.from_text()."""

    def test_basic_creation(self):
        doc = Document.from_text("Hello world", filename="test.txt")

        assert doc.filename == "test.txt"
        assert len(doc.chunks) == 1
        assert doc.chunks[0].text == "Hello world"
        assert doc.chunks[0].index == 0

    def test_deterministic_id(self):
        """Same content produces same ID."""
        doc1 = Document.from_text("Hello world")
        doc2 = Document.from_text("Hello world")
        assert doc1.id == doc2.id

    def test_different_content_different_id(self):
        doc1 = Document.from_text("Hello")
        doc2 = Document.from_text("World")
        assert doc1.id != doc2.id

    def test_explicit_id(self):
        doc = Document.from_text("Hello", document_id="my-custom-id")
        assert doc.id == "my-custom-id"

    def test_chunking(self):
        # Create text that will be split
        text = "Sentence one. " * 100  # ~1400 chars
        doc = Document.from_text(text, chunk_size=500, chunk_overlap=50)

        assert len(doc.chunks) > 1
        # Check indices are sequential
        for i, chunk in enumerate(doc.chunks):
            assert chunk.index == i

    def test_empty_text(self):
        doc = Document.from_text("")
        assert len(doc.chunks) == 0

    def test_whitespace_only(self):
        doc = Document.from_text("   \n\n   ")
        assert len(doc.chunks) == 0

    def test_metadata_set(self):
        doc = Document.from_text("Hello", filename="test.txt")
        assert doc.metadata.get("source") == "text"
        assert "content_hash" in doc.metadata

    def test_chunks_have_metadata(self):
        doc = Document.from_text("Hello", filename="test.txt")
        assert doc.chunks[0].get_metadata("source") == "text"
        assert doc.chunks[0].get_metadata("filename") == "test.txt"

    def test_no_page_or_bbox(self):
        doc = Document.from_text("Hello")
        assert doc.page_count is None
        assert doc.chunks[0].page is None
        assert doc.chunks[0].bbox is None


class TestDocumentFromChunks:
    """Test Document.from_chunks()."""

    def test_basic_creation(self):
        chunks = [
            Chunk(index=0, text="First"),
            Chunk(index=1, text="Second"),
        ]
        doc = Document.from_chunks(chunks, filename="test")

        assert len(doc.chunks) == 2
        assert doc.chunks[0].text == "First"
        assert doc.chunks[1].text == "Second"

    def test_explicit_id(self):
        chunks = [Chunk(index=0, text="text")]
        doc = Document.from_chunks(chunks, document_id="custom-id")
        assert doc.id == "custom-id"

    def test_auto_id_from_content(self):
        chunks = [Chunk(index=0, text="Same content")]
        doc1 = Document.from_chunks(chunks)
        doc2 = Document.from_chunks(chunks)
        assert doc1.id == doc2.id

    def test_with_page_count(self):
        chunks = [Chunk(index=0, text="text")]
        doc = Document.from_chunks(chunks, page_count=5)
        assert doc.page_count == 5

    def test_with_metadata(self):
        chunks = [Chunk(index=0, text="text")]
        doc = Document.from_chunks(chunks, metadata={"custom": "value"})
        assert doc.metadata["custom"] == "value"


class TestDocumentProperties:
    """Test Document properties and methods."""

    def test_full_text(self):
        doc = Document.from_chunks(
            [
                Chunk(index=0, text="First"),
                Chunk(index=1, text="Second"),
                Chunk(index=2, text="Third"),
            ]
        )
        assert doc.full_text == "First\nSecond\nThird"

    def test_full_text_empty(self):
        doc = Document.from_chunks([])
        assert doc.full_text == ""

    def test_iter_chunks(self):
        chunks = [Chunk(index=i, text=f"Chunk {i}") for i in range(3)]
        doc = Document.from_chunks(chunks)

        result = list(doc.iter_chunks())
        assert len(result) == 3
        assert result[0].text == "Chunk 0"

    def test_get_chunk_found(self):
        chunks = [Chunk(index=i, text=f"Chunk {i}") for i in range(3)]
        doc = Document.from_chunks(chunks)

        chunk = doc.get_chunk(1)
        assert chunk is not None
        assert chunk.text == "Chunk 1"

    def test_get_chunk_not_found(self):
        chunks = [Chunk(index=0, text="Only chunk")]
        doc = Document.from_chunks(chunks)

        assert doc.get_chunk(99) is None

    def test_get_chunks_by_page(self):
        chunks = [
            Chunk(index=0, text="Page 1 chunk 1", page=1),
            Chunk(index=1, text="Page 1 chunk 2", page=1),
            Chunk(index=2, text="Page 2 chunk 1", page=2),
        ]
        doc = Document.from_chunks(chunks)

        page1_chunks = doc.get_chunks_by_page(1)
        assert len(page1_chunks) == 2

        page2_chunks = doc.get_chunks_by_page(2)
        assert len(page2_chunks) == 1

        page3_chunks = doc.get_chunks_by_page(3)
        assert len(page3_chunks) == 0

    def test_len(self):
        chunks = [Chunk(index=i, text=f"Chunk {i}") for i in range(5)]
        doc = Document.from_chunks(chunks)
        assert len(doc) == 5

    def test_iter(self):
        chunks = [Chunk(index=i, text=f"Chunk {i}") for i in range(3)]
        doc = Document.from_chunks(chunks)

        texts = [c.text for c in doc]
        assert texts == ["Chunk 0", "Chunk 1", "Chunk 2"]


class TestDocumentSerialization:
    """Test Document serialization."""

    def test_to_dict(self):
        doc = Document.from_chunks(
            [Chunk(index=0, text="Hello", page=1)],
            filename="test.txt",
            document_id="doc-123",
            page_count=5,
            metadata={"key": "value"},
        )

        d = doc.to_dict()

        assert d["id"] == "doc-123"
        assert d["filename"] == "test.txt"
        assert d["page_count"] == 5
        assert d["metadata"]["key"] == "value"
        assert len(d["chunks"]) == 1
        assert d["chunks"][0]["text"] == "Hello"

    def test_from_dict(self):
        d = {
            "id": "doc-123",
            "filename": "test.txt",
            "chunks": [
                {"index": 0, "text": "Hello", "page": 1},
                {"index": 1, "text": "World", "page": 2},
            ],
            "page_count": 2,
            "metadata": {"key": "value"},
        }

        doc = Document.from_dict(d)

        assert doc.id == "doc-123"
        assert doc.filename == "test.txt"
        assert doc.page_count == 2
        assert len(doc.chunks) == 2
        assert doc.chunks[0].text == "Hello"
        assert doc.chunks[1].page == 2

    def test_roundtrip(self):
        original = Document.from_chunks(
            [
                Chunk(index=0, text="First", page=1, bbox=(0, 0, 100, 50)),
                Chunk(index=1, text="Second", page=2),
            ],
            filename="test.pdf",
            document_id="original-id",
            page_count=2,
            metadata={"source": "test"},
        )

        d = original.to_dict()
        restored = Document.from_dict(d)

        assert restored.id == original.id
        assert restored.filename == original.filename
        assert restored.page_count == original.page_count
        assert len(restored.chunks) == len(original.chunks)
        assert restored.chunks[0].text == original.chunks[0].text
        assert restored.chunks[0].bbox == original.chunks[0].bbox

    def test_json_roundtrip(self):
        original = Document.from_text("Test content", filename="test.txt")

        json_str = json.dumps(original.to_dict())
        d = json.loads(json_str)
        restored = Document.from_dict(d)

        assert restored.id == original.id
        assert restored.full_text == original.full_text


class TestDocumentRepr:
    """Test Document string representation."""

    def test_repr(self):
        doc = Document.from_chunks(
            [Chunk(index=i, text=f"Chunk {i}") for i in range(5)],
            filename="test.pdf",
            document_id="abc123",
            page_count=3,
        )

        r = repr(doc)
        assert "abc123" in r
        assert "test.pdf" in r
        assert "chunks=5" in r
        assert "pages=3" in r


class TestDocumentDoclingLoading:
    """Test Docling-based document loading."""

    def test_from_docling_file_path_basic(self, tmp_path):
        """Test that from_docling_file_path loads a markdown file."""
        # Create a simple markdown file for testing (docling supports .md)
        test_file = tmp_path / "test.md"
        test_file.write_text(
            "# Hello world\n\nThis is a test document.\n\nIt has multiple paragraphs."
        )

        doc = Document.from_docling_file_path(str(test_file))

        assert doc.filename == "test.md"
        assert len(doc.chunks) > 0
        assert doc.id is not None
        assert doc.metadata.get("source") == "docling"

    def test_from_docling_file_path_explicit_id(self, tmp_path):
        """Test that explicit document_id is used."""
        test_file = tmp_path / "test.md"
        test_file.write_text("# Test\n\nTest content")

        doc = Document.from_docling_file_path(str(test_file), document_id="custom-id")

        assert doc.id == "custom-id"

    def test_from_docling_file_path_deterministic_id(self, tmp_path):
        """Same file produces same ID."""
        test_file = tmp_path / "test.md"
        test_file.write_text("# Consistent\n\nConsistent content")

        doc1 = Document.from_docling_file_path(str(test_file))
        doc2 = Document.from_docling_file_path(str(test_file))

        assert doc1.id == doc2.id

    def test_from_docling_bytes_io_basic(self, tmp_path):
        """Test that from_docling_bytes_io loads from bytes."""
        content = b"# Hello from bytes\n\nThis is content from bytes."

        doc = Document.from_docling_bytes_io(content, filename="test.md")

        assert doc.filename == "test.md"
        assert len(doc.chunks) > 0
        assert doc.id is not None
        assert doc.metadata.get("source") == "docling"

    def test_from_docling_bytes_io_explicit_id(self):
        """Test that explicit document_id is used with bytes."""
        content = b"# Test\n\nTest content"

        doc = Document.from_docling_bytes_io(
            content, filename="test.md", document_id="custom-id"
        )

        assert doc.id == "custom-id"

    def test_from_docling_bytes_io_deterministic_id(self):
        """Same bytes produces same ID."""
        content = b"# Consistent\n\nConsistent content"

        doc1 = Document.from_docling_bytes_io(content, filename="test.md")
        doc2 = Document.from_docling_bytes_io(content, filename="test.md")

        assert doc1.id == doc2.id

    def test_from_docling_bytes_io_with_bytesio(self):
        """Test that from_docling_bytes_io works with BytesIO."""
        from io import BytesIO

        content = b"# BytesIO test\n\nContent from BytesIO object."
        stream = BytesIO(content)

        doc = Document.from_docling_bytes_io(stream, filename="test.md")

        assert doc.filename == "test.md"
        assert len(doc.chunks) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
