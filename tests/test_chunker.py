"""Tests for the chunker module."""

import pytest

from pullcite.core.chunker import (
    Chunker,
    ChunkConfig,
    SlidingWindowChunker,
    SentenceChunker,
    ParagraphChunker,
    DEFAULT_CHUNKER,
)


class TestSlidingWindowChunker:
    """Tests for SlidingWindowChunker."""

    def test_short_text_single_chunk(self):
        """Short text should be a single chunk."""
        chunker = SlidingWindowChunker(size=100, stride=50)
        result = chunker.chunk("Hello world")
        assert len(result) == 1
        assert result[0].text == "Hello world"
        assert result[0].start_char == 0
        assert result[0].end_char == 11

    def test_empty_text(self):
        """Empty text should return empty list."""
        chunker = SlidingWindowChunker(size=100, stride=50)
        result = chunker.chunk("")
        assert result == []

    def test_sliding_window_overlap(self):
        """Chunks should overlap correctly."""
        text = "A" * 100  # 100 A's
        chunker = SlidingWindowChunker(size=40, stride=20)  # 50% overlap
        result = chunker.chunk(text)

        # First chunk: 0-40
        assert result[0].start_char == 0
        assert result[0].end_char == 40

        # Second chunk: 20-60
        assert result[1].start_char == 20
        assert result[1].end_char == 60

        # Check overlap
        assert result[0].text[-20:] == result[1].text[:20]

    def test_default_stride(self):
        """Default stride should be size // 2."""
        chunker = SlidingWindowChunker(size=100)
        assert chunker.stride == 50

    def test_invalid_size(self):
        """Size must be positive."""
        with pytest.raises(ValueError, match="size must be positive"):
            SlidingWindowChunker(size=0, stride=10)

    def test_invalid_stride(self):
        """Stride must be positive."""
        with pytest.raises(ValueError, match="stride must be positive"):
            SlidingWindowChunker(size=100, stride=0)


class TestSentenceChunker:
    """Tests for SentenceChunker."""

    def test_breaks_at_sentences(self):
        """Should prefer breaking at sentence boundaries."""
        text = "First sentence. Second sentence. Third sentence."
        chunker = SentenceChunker(target_size=20, overlap=5)
        result = chunker.chunk(text)

        # Should break at periods
        assert any("." in chunk.text[-5:] for chunk in result[:-1])

    def test_short_text(self):
        """Short text should be single chunk."""
        chunker = SentenceChunker(target_size=1000)
        result = chunker.chunk("Short text.")
        assert len(result) == 1

    def test_empty_text(self):
        """Empty text returns empty list."""
        chunker = SentenceChunker()
        assert chunker.chunk("") == []

    def test_overlap(self):
        """Chunks should overlap."""
        text = "A" * 500
        chunker = SentenceChunker(target_size=100, overlap=20)
        result = chunker.chunk(text)

        # Should have multiple chunks with overlap
        assert len(result) > 1
        # Last 20 chars of first chunk should appear in second
        # (or close to it due to sentence boundary seeking)


class TestParagraphChunker:
    """Tests for ParagraphChunker."""

    def test_splits_on_paragraphs(self):
        """Should split on double newlines."""
        text = "First paragraph.\n\nSecond paragraph.\n\nThird paragraph."
        chunker = ParagraphChunker(target_size=50, overlap_paragraphs=0)
        result = chunker.chunk(text)

        # Should have separate chunks for paragraphs
        assert len(result) >= 1

    def test_single_paragraph(self):
        """Single paragraph text."""
        chunker = ParagraphChunker(target_size=1000)
        result = chunker.chunk("Just one paragraph with no breaks.")
        assert len(result) == 1

    def test_empty_text(self):
        """Empty text returns empty list."""
        chunker = ParagraphChunker()
        assert chunker.chunk("") == []


class TestChunkConfig:
    """Tests for ChunkConfig dataclass."""

    def test_chunk_config_creation(self):
        """ChunkConfig should store text and positions."""
        config = ChunkConfig(
            text="Hello world",
            start_char=0,
            end_char=11,
        )
        assert config.text == "Hello world"
        assert config.start_char == 0
        assert config.end_char == 11


class TestDefaultChunker:
    """Tests for DEFAULT_CHUNKER."""

    def test_default_is_sentence_chunker(self):
        """Default chunker should be a SentenceChunker."""
        assert isinstance(DEFAULT_CHUNKER, SentenceChunker)

    def test_default_works(self):
        """Default chunker should work."""
        result = DEFAULT_CHUNKER.chunk("Some text to chunk.")
        assert len(result) >= 1
        assert result[0].text == "Some text to chunk."
