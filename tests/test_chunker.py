"""Tests for text chunker."""

import pytest

from document_agent.graph import TextChunker


class TestTextChunker:
    """Tests for TextChunker."""

    def test_count_tokens(self):
        """Test token counting."""
        chunker = TextChunker()
        count = chunker.count_tokens("Hello, world!")
        assert count > 0
        assert isinstance(count, int)

    def test_chunk_short_text(self):
        """Test chunking short text."""
        chunker = TextChunker(chunk_size=100, chunk_overlap=10)
        text = "This is a short text."
        chunks = chunker.chunk_text(text, "doc_1")

        assert len(chunks) == 1
        assert chunks[0].content == text
        assert chunks[0].doc_id == "doc_1"

    def test_chunk_long_text(self):
        """Test chunking long text into multiple chunks."""
        chunker = TextChunker(chunk_size=50, chunk_overlap=10)

        # Create a longer text
        paragraphs = [f"This is paragraph {i}. It contains some text." for i in range(10)]
        text = "\n\n".join(paragraphs)

        chunks = chunker.chunk_text(text, "doc_2")

        assert len(chunks) > 1
        for chunk in chunks:
            assert chunk.doc_id == "doc_2"
            assert chunk.content

    def test_chunk_with_headers(self):
        """Test chunking text with markdown headers."""
        chunker = TextChunker(chunk_size=100, chunk_overlap=10)

        text = """# Introduction

This is the intro.

# Main Content

This is the main content.

# Conclusion

This is the conclusion."""

        chunks = chunker.chunk_text(text, "doc_3")

        assert len(chunks) >= 1
        # Check that headers are preserved
        full_content = " ".join(c.content for c in chunks)
        assert "Introduction" in full_content
        assert "Conclusion" in full_content

    def test_empty_text(self):
        """Test chunking empty text."""
        chunker = TextChunker()
        chunks = chunker.chunk_text("", "doc_4")

        assert len(chunks) == 0

    def test_chunk_document(self):
        """Test chunking with pages."""
        chunker = TextChunker(chunk_size=50, chunk_overlap=10)

        pages = [
            "Page 1 content here.",
            "Page 2 content here.",
            "Page 3 content here.",
        ]

        chunks = chunker.chunk_document(
            "\n\n".join(pages),
            "doc_5",
            pages=pages,
        )

        assert len(chunks) >= 3
        # Check page numbers are assigned
        page_nums = {c.page_num for c in chunks}
        assert page_nums == {1, 2, 3}
