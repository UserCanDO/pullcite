"""
Chunker - Configurable text chunking strategies.

Chunking is critical for extraction quality. This module provides explicit
control over how documents are split into searchable chunks.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class ChunkConfig:
    """Configuration for chunk output."""

    text: str
    start_char: int
    end_char: int


class Chunker(ABC):
    """
    Abstract base class for text chunking strategies.

    Chunking directly affects extraction quality:
    - Too small: Context fragments lose meaning
    - Too large: Search results become diluted
    - No overlap: Important text at boundaries gets split

    Choose your strategy based on document type:
    - SlidingWindowChunker: General purpose, predictable behavior
    - SentenceChunker: Better for prose, respects sentence boundaries
    """

    @abstractmethod
    def chunk(self, text: str) -> list[ChunkConfig]:
        """
        Split text into chunks.

        Args:
            text: Full text to chunk.

        Returns:
            List of ChunkConfig with text and character positions.
        """
        pass


@dataclass
class SlidingWindowChunker(Chunker):
    """
    Fixed-size sliding window chunker.

    Splits text into fixed-size chunks with configurable overlap.
    Simple and predictable - good default for most documents.

    Args:
        size: Target chunk size in characters. Default 1000.
        stride: Step size between chunk starts. Default size // 2.
               Overlap = size - stride.

    Example:
        >>> chunker = SlidingWindowChunker(size=500, stride=250)
        >>> # Creates 500-char chunks with 250-char overlap (50%)
        >>>
        >>> chunker = SlidingWindowChunker(size=1000, stride=800)
        >>> # Creates 1000-char chunks with 200-char overlap (20%)
    """

    size: int = 1000
    stride: int | None = None

    def __post_init__(self) -> None:
        if self.stride is None:
            self.stride = self.size // 2
        if self.stride <= 0:
            raise ValueError("stride must be positive")
        if self.size <= 0:
            raise ValueError("size must be positive")

    def chunk(self, text: str) -> list[ChunkConfig]:
        if not text:
            return []

        if len(text) <= self.size:
            return [ChunkConfig(text=text, start_char=0, end_char=len(text))]

        chunks: list[ChunkConfig] = []
        start = 0

        while start < len(text):
            end = min(start + self.size, len(text))
            chunks.append(
                ChunkConfig(
                    text=text[start:end],
                    start_char=start,
                    end_char=end,
                )
            )

            if end >= len(text):
                break

            start += self.stride  # type: ignore[operator]

        return chunks


@dataclass
class SentenceChunker(Chunker):
    """
    Sentence-boundary aware chunker.

    Tries to break at sentence boundaries (., !, ?, newlines) while
    staying within the target size. Better for prose documents.

    Args:
        target_size: Target chunk size in characters. Default 1000.
        overlap: Characters to overlap between chunks. Default 200.
        min_size: Minimum chunk size (avoids tiny final chunks). Default 100.

    Example:
        >>> chunker = SentenceChunker(target_size=1000, overlap=200)
        >>> # ~1000-char chunks breaking at sentences, 200-char overlap
    """

    target_size: int = 1000
    overlap: int = 200
    min_size: int = 100

    def chunk(self, text: str) -> list[ChunkConfig]:
        if not text:
            return []

        if len(text) <= self.target_size:
            return [ChunkConfig(text=text, start_char=0, end_char=len(text))]

        chunks: list[ChunkConfig] = []
        start = 0

        while start < len(text):
            end = min(start + self.target_size, len(text))

            if end < len(text):
                # Look for sentence boundary backwards from target
                search_start = max(start + self.target_size // 2, start + self.min_size)
                best_break = end

                for i in range(end, search_start, -1):
                    if text[i - 1] in ".!?\n":
                        best_break = i
                        break

                end = best_break

            chunks.append(
                ChunkConfig(
                    text=text[start:end],
                    start_char=start,
                    end_char=end,
                )
            )

            if end >= len(text):
                break

            # Move start, accounting for overlap
            new_start = end - self.overlap
            if new_start <= start:
                new_start = end  # Prevent infinite loop
            start = new_start

        return chunks


@dataclass
class ParagraphChunker(Chunker):
    """
    Paragraph-based chunker.

    Splits on double newlines (paragraph breaks), then combines paragraphs
    to reach target size. Best for well-structured documents.

    Args:
        target_size: Target chunk size in characters. Default 1000.
        overlap_paragraphs: Number of paragraphs to overlap. Default 1.

    Example:
        >>> chunker = ParagraphChunker(target_size=1500, overlap_paragraphs=1)
    """

    target_size: int = 1000
    overlap_paragraphs: int = 1

    def chunk(self, text: str) -> list[ChunkConfig]:
        if not text:
            return []

        # Split into paragraphs
        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]

        if not paragraphs:
            return [ChunkConfig(text=text, start_char=0, end_char=len(text))]

        chunks: list[ChunkConfig] = []
        current_paras: list[str] = []
        current_start = 0

        # Track character positions
        para_positions: list[tuple[int, int]] = []
        pos = 0
        for p in paragraphs:
            # Find actual position in original text
            idx = text.find(p, pos)
            if idx >= 0:
                para_positions.append((idx, idx + len(p)))
                pos = idx + len(p)
            else:
                para_positions.append((pos, pos + len(p)))

        for i, para in enumerate(paragraphs):
            current_paras.append(para)
            combined = "\n\n".join(current_paras)

            if len(combined) >= self.target_size or i == len(paragraphs) - 1:
                # Create chunk
                start_pos = para_positions[i - len(current_paras) + 1][0]
                end_pos = para_positions[i][1]

                chunks.append(
                    ChunkConfig(
                        text=combined,
                        start_char=start_pos,
                        end_char=end_pos,
                    )
                )

                # Keep overlap paragraphs
                if self.overlap_paragraphs > 0:
                    current_paras = current_paras[-self.overlap_paragraphs :]
                else:
                    current_paras = []

        return chunks


# Default chunker used when none specified
DEFAULT_CHUNKER = SentenceChunker(target_size=1200, overlap=200)


__all__ = [
    "Chunker",
    "ChunkConfig",
    "SlidingWindowChunker",
    "SentenceChunker",
    "ParagraphChunker",
    "DEFAULT_CHUNKER",
]
