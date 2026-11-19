from abc import ABC, abstractmethod
from typing import Generator, Iterator, List, Iterable

class BoundaryStrategy(ABC):
    """
    Abstract base class for defining boundaries in text chunking.
    """

    @abstractmethod
    def split(self, text: str) -> list[str]:
        pass


class ParagraphBoundary(BoundaryStrategy):
    """
    Concrete implementation of BoundaryStrategy that finds paragraph boundaries.
    """

    def split(self, text: str) -> list[str]:
        return [line for para in text.split("\n\n") for line in para.splitlines() if line.strip()]

class SentenceBoundary(BoundaryStrategy):
    def split(self, text: str) -> list[str]:
        return [t.strip() for t in text.split(".")]

class WordBoundary(BoundaryStrategy):
    def split(self, text: str) -> list[str]:
        return text.split(" ")

class Chunker:
    def __init__(self, boundaries, max_chunk_len: int, overlap: int = 0):
        self.boundaries = boundaries
        self.max_len = max_chunk_len
        self.overlap = overlap

    def _split_recursive(self, text: str, max_len: int, boundary_idx: int) -> Iterator[str]:
        if not text:
            return

        if len(text) <= max_len or boundary_idx >= len(self.boundaries):
            yield text.strip()
            return

        for part in self.boundaries[boundary_idx].split(text):
            yield from self._split_recursive(part, max_len, boundary_idx + 1)

    def chunk(self, documents: Iterable[str]) -> Iterator[str]:
        """Yield flat text chunks lazily across all documents."""
        for doc in documents:
            yield from self._split_recursive(doc, self.max_len, 0)

if __name__ == "__main__":
    text= ["""Paragraph 1

Paragaph 2. Ahoj jak se mas. Sentence 2. Sentence 3. 
Paragraph 3. Sentence 4. Sentence 5. Ahoj jak se mas."""]

    chunker = Chunker([ParagraphBoundary(), SentenceBoundary(), WordBoundary()], 50, 0)
    chunks = list(chunker.chunk(text))
    print(chunks)
