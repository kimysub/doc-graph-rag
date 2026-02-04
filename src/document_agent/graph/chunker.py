"""Text chunking module for document processing."""

from dataclasses import dataclass
from typing import Optional

import tiktoken

from document_agent.config import settings


@dataclass
class Chunk:
    """A chunk of text from a document."""

    id: str
    content: str
    doc_id: str
    page_num: Optional[int] = None
    position: int = 0  # Position within the document
    token_count: int = 0
    metadata: dict = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class TextChunker:
    """Chunker that splits text into semantic chunks."""

    def __init__(
        self,
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None,
        encoding_name: str = "cl100k_base",
    ):
        """Initialize the text chunker.

        Args:
            chunk_size: Target chunk size in tokens.
            chunk_overlap: Overlap between chunks in tokens.
            encoding_name: Tiktoken encoding name.
        """
        self.chunk_size = chunk_size or settings.chunk_size
        self.chunk_overlap = chunk_overlap or settings.chunk_overlap
        self.encoding = tiktoken.get_encoding(encoding_name)

    def count_tokens(self, text: str) -> int:
        """Count the number of tokens in text.

        Args:
            text: Text to count tokens for.

        Returns:
            Number of tokens.
        """
        return len(self.encoding.encode(text))

    def chunk_text(
        self,
        text: str,
        doc_id: str,
        page_num: Optional[int] = None,
    ) -> list[Chunk]:
        """Split text into chunks.

        Args:
            text: Text to chunk.
            doc_id: Document ID.
            page_num: Optional page number.

        Returns:
            List of Chunk objects.
        """
        if not text.strip():
            return []

        chunks = []
        paragraphs = self._split_into_paragraphs(text)

        current_chunk = []
        current_tokens = 0
        position = 0

        for paragraph in paragraphs:
            para_tokens = self.count_tokens(paragraph)

            # If single paragraph exceeds chunk size, split it further
            if para_tokens > self.chunk_size:
                # First, save current chunk if exists
                if current_chunk:
                    chunk_text = "\n\n".join(current_chunk)
                    chunks.append(
                        Chunk(
                            id=f"{doc_id}_chunk_{len(chunks)}",
                            content=chunk_text,
                            doc_id=doc_id,
                            page_num=page_num,
                            position=position,
                            token_count=current_tokens,
                        )
                    )
                    position += 1
                    current_chunk = []
                    current_tokens = 0

                # Split large paragraph by sentences
                sub_chunks = self._split_large_paragraph(paragraph, doc_id, page_num, position)
                for sub_chunk in sub_chunks:
                    sub_chunk.id = f"{doc_id}_chunk_{len(chunks)}"
                    sub_chunk.position = position
                    chunks.append(sub_chunk)
                    position += 1

            # If adding paragraph would exceed chunk size
            elif current_tokens + para_tokens > self.chunk_size:
                # Save current chunk
                if current_chunk:
                    chunk_text = "\n\n".join(current_chunk)
                    chunks.append(
                        Chunk(
                            id=f"{doc_id}_chunk_{len(chunks)}",
                            content=chunk_text,
                            doc_id=doc_id,
                            page_num=page_num,
                            position=position,
                            token_count=current_tokens,
                        )
                    )
                    position += 1

                # Start new chunk with overlap
                if self.chunk_overlap > 0 and current_chunk:
                    # Keep last paragraph(s) for overlap
                    overlap_chunk = current_chunk[-1:]
                    current_chunk = overlap_chunk + [paragraph]
                    current_tokens = sum(self.count_tokens(p) for p in current_chunk)
                else:
                    current_chunk = [paragraph]
                    current_tokens = para_tokens
            else:
                current_chunk.append(paragraph)
                current_tokens += para_tokens

        # Don't forget the last chunk
        if current_chunk:
            chunk_text = "\n\n".join(current_chunk)
            chunks.append(
                Chunk(
                    id=f"{doc_id}_chunk_{len(chunks)}",
                    content=chunk_text,
                    doc_id=doc_id,
                    page_num=page_num,
                    position=position,
                    token_count=current_tokens,
                )
            )

        return chunks

    def chunk_document(
        self,
        content: str,
        doc_id: str,
        pages: Optional[list[str]] = None,
    ) -> list[Chunk]:
        """Chunk an entire document.

        Args:
            content: Full document content.
            doc_id: Document ID.
            pages: Optional list of page contents.

        Returns:
            List of Chunk objects.
        """
        all_chunks = []

        if pages:
            # Chunk by pages
            for page_num, page_content in enumerate(pages, 1):
                page_chunks = self.chunk_text(page_content, doc_id, page_num)
                all_chunks.extend(page_chunks)
        else:
            # Chunk entire content
            all_chunks = self.chunk_text(content, doc_id)

        # Re-number chunks
        for i, chunk in enumerate(all_chunks):
            chunk.id = f"{doc_id}_chunk_{i}"
            chunk.position = i

        return all_chunks

    def _split_into_paragraphs(self, text: str) -> list[str]:
        """Split text into paragraphs.

        Args:
            text: Text to split.

        Returns:
            List of paragraphs.
        """
        # Split by double newlines or markdown headers
        paragraphs = []
        current = []

        for line in text.split("\n"):
            stripped = line.strip()

            # Check for paragraph break
            if not stripped:
                if current:
                    paragraphs.append("\n".join(current))
                    current = []
            # Check for markdown header (treat as new paragraph)
            elif stripped.startswith("#"):
                if current:
                    paragraphs.append("\n".join(current))
                    current = []
                paragraphs.append(line)
            else:
                current.append(line)

        if current:
            paragraphs.append("\n".join(current))

        return [p for p in paragraphs if p.strip()]

    def _split_large_paragraph(
        self,
        paragraph: str,
        doc_id: str,
        page_num: Optional[int],
        start_position: int,
    ) -> list[Chunk]:
        """Split a large paragraph into smaller chunks.

        Args:
            paragraph: Large paragraph to split.
            doc_id: Document ID.
            page_num: Page number.
            start_position: Starting position counter.

        Returns:
            List of Chunk objects.
        """
        chunks = []
        sentences = self._split_into_sentences(paragraph)

        current_chunk = []
        current_tokens = 0

        for sentence in sentences:
            sent_tokens = self.count_tokens(sentence)

            if current_tokens + sent_tokens > self.chunk_size:
                if current_chunk:
                    chunk_text = " ".join(current_chunk)
                    chunks.append(
                        Chunk(
                            id="",  # Will be set later
                            content=chunk_text,
                            doc_id=doc_id,
                            page_num=page_num,
                            position=start_position + len(chunks),
                            token_count=current_tokens,
                        )
                    )
                current_chunk = [sentence]
                current_tokens = sent_tokens
            else:
                current_chunk.append(sentence)
                current_tokens += sent_tokens

        if current_chunk:
            chunk_text = " ".join(current_chunk)
            chunks.append(
                Chunk(
                    id="",
                    content=chunk_text,
                    doc_id=doc_id,
                    page_num=page_num,
                    position=start_position + len(chunks),
                    token_count=current_tokens,
                )
            )

        return chunks

    def _split_into_sentences(self, text: str) -> list[str]:
        """Split text into sentences.

        Args:
            text: Text to split.

        Returns:
            List of sentences.
        """
        import re

        # Simple sentence splitting
        sentences = re.split(r"(?<=[.!?])\s+", text)
        return [s.strip() for s in sentences if s.strip()]
