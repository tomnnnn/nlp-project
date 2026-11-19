from abc import ABC, abstractmethod, abstractproperty
import re
from typing import Generator, Iterator, List, Iterable

class BoundaryStrategy(ABC):
    """
    Abstract base class for defining boundaries in text chunking.
    """

    @abstractmethod
    def split(self, text: str) -> list[str]:
        pass

class ParagraphBoundary(BoundaryStrategy):
    """Splits at paragraph boundaries (double newlines)."""
    def split(self, text: str) -> list[str]:
        return [line for para in text.split("\n\n") for line in para.splitlines() if line.strip()]

class SentenceBoundary(BoundaryStrategy):
    """Splits at sentence boundaries (periods), preserving trailing periods."""
    sep = ". "

    def split(self, text: str) -> list[str]:
        parts = text.split(self.sep)
        if len(parts) == 1:
            return [text.strip()] if text.strip() else []

        out = [f"{p}.".strip() for p in parts[:-1] if p.strip()]
        if parts[-1].strip():
            out.append(parts[-1].strip())
        return out

class WordBoundary(BoundaryStrategy):
    """Splits at word boundaries (spaces)."""
    sep = " "

    def split(self, text: str) -> list[str]:
        return text.split(self.sep)


class Chunker:
    """
    Recursively splits text by boundary strategies up to max_chunk_len,
    then applies overlap using a configurable unit boundary.
    """

    def __init__(
        self,
        boundaries: list[BoundaryStrategy],
        max_chunk_len: int,
        overlap_units: int = 0,
        overlap_boundary: BoundaryStrategy | None = None,
        overlap_sep: str = " ",
    ):
        if max_chunk_len <= 0:
            raise ValueError("max_chunk_len must be positive")
        if overlap_units < 0:
            raise ValueError("overlap_units must be non-negative")
        if overlap_units > 0 and overlap_boundary is None:
            raise ValueError("overlap_boundary must be provided when overlap_units > 0")

        self.boundaries = boundaries
        self.max_len = max_chunk_len
        self.overlap_units = overlap_units
        self.overlap_boundary = overlap_boundary
        self.overlap_sep = overlap_sep

    def _split_recursive(self, text: str, boundary_idx: int) -> Iterator[str]:
        if not text or not text.strip():
            return

        if len(text.strip()) <= self.max_len or boundary_idx >= len(self.boundaries):
            yield text.strip()
            return

        for part in self.boundaries[boundary_idx].split(text):
            yield from self._split_recursive(part, boundary_idx + 1)

    def _build_overlap_prefix(self, prev_chunk: str, next_piece: str) -> str:
        if self.overlap_units == 0 or self.overlap_boundary is None or not prev_chunk:
            return ""

        units = [u for u in self.overlap_boundary.split(prev_chunk) if u]
        if not units:
            return ""

        return self.overlap_sep.join(units[-self.overlap_units:]).strip()

    def chunk(self, documents: Iterable[str]) -> Iterator[str]:
        """Yield flat text chunks lazily across all documents."""
        prev_chunk = ""
        for doc in documents:
            for piece in self._split_recursive(doc, 0):
                prefix = self._build_overlap_prefix(prev_chunk, piece)
                out = f"{prefix}{self.overlap_sep}{piece}" if prefix else piece
                yield out
                prev_chunk = out


if __name__ == "__main__":
    """Example"""

    text= ["""Hello world. This is a short sentence. This is another short sentence.
This is a very long sentence with a lot of words that will for sure contain more than 50 characters."""]

    chunker = Chunker([ParagraphBoundary(), SentenceBoundary()], 50, 2, WordBoundary(), WordBoundary.sep)
    chunks = list(chunker.chunk(text))
    print(chunks)
