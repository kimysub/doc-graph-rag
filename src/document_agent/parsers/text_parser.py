"""Text file parser for MD and TXT files."""

from pathlib import Path

from .base import BaseParser, ParsedDocument


class TextParser(BaseParser):
    """Parser for plain text files (MD, TXT)."""

    supported_extensions = [".md", ".txt", ".markdown", ".text"]

    def parse(self, file_path: str | Path) -> ParsedDocument:
        """Parse a text file.

        Args:
            file_path: Path to the text file.

        Returns:
            ParsedDocument containing the file content.
        """
        file_path = Path(file_path)
        doc_id = self.generate_doc_id(file_path)

        # Determine document type
        suffix = file_path.suffix.lower()
        doc_type = "markdown" if suffix in [".md", ".markdown"] else "text"

        # Read file content
        content = file_path.read_text(encoding="utf-8")

        # For markdown, try to split by headers
        pages = []
        if doc_type == "markdown":
            pages = self._split_by_headers(content)

        return ParsedDocument(
            doc_id=doc_id,
            name=file_path.name,
            path=file_path,
            doc_type=doc_type,
            content=content,
            pages=pages,
            images=[],  # Text files don't have embedded images
            metadata={
                "encoding": "utf-8",
                "size_bytes": file_path.stat().st_size,
            },
        )

    def _split_by_headers(self, content: str) -> list[str]:
        """Split markdown content by top-level headers.

        Args:
            content: Markdown content.

        Returns:
            List of sections split by headers.
        """
        sections = []
        current_section = []

        for line in content.split("\n"):
            # Check for top-level header (# or ##)
            if line.startswith("# ") or line.startswith("## "):
                if current_section:
                    sections.append("\n".join(current_section))
                current_section = [line]
            else:
                current_section.append(line)

        if current_section:
            sections.append("\n".join(current_section))

        return sections if len(sections) > 1 else []
