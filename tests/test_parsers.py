"""Tests for document parsers."""

import tempfile
from pathlib import Path

import pytest

from document_agent.parsers import TextParser


class TestTextParser:
    """Tests for TextParser."""

    def test_can_parse_txt(self):
        """Test that TextParser can parse .txt files."""
        assert TextParser.can_parse("test.txt")
        assert TextParser.can_parse("test.md")
        assert not TextParser.can_parse("test.pdf")

    def test_parse_txt_file(self):
        """Test parsing a text file."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("Hello, World!\n\nThis is a test document.")
            tmp_path = Path(f.name)

        try:
            parser = TextParser()
            result = parser.parse(tmp_path)

            assert result.doc_type == "text"
            assert "Hello, World!" in result.content
            assert result.name == tmp_path.name
        finally:
            tmp_path.unlink()

    def test_parse_markdown_file(self):
        """Test parsing a markdown file."""
        content = """# Title

## Section 1

Some content here.

## Section 2

More content.
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as f:
            f.write(content)
            tmp_path = Path(f.name)

        try:
            parser = TextParser()
            result = parser.parse(tmp_path)

            assert result.doc_type == "markdown"
            assert "# Title" in result.content
            assert len(result.pages) > 0  # Should split by headers
        finally:
            tmp_path.unlink()


class TestParserBase:
    """Tests for BaseParser."""

    def test_generate_doc_id(self):
        """Test document ID generation."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("Test content")
            tmp_path = Path(f.name)

        try:
            parser = TextParser()
            doc_id = parser.generate_doc_id(tmp_path)

            assert isinstance(doc_id, str)
            assert len(doc_id) == 16  # SHA256 truncated to 16 chars
        finally:
            tmp_path.unlink()
