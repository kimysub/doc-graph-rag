"""Base parser interface and common data structures."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class ImageInfo:
    """Information about an extracted image."""

    id: str
    path: Path
    page_num: Optional[int] = None
    description: Optional[str] = None


@dataclass
class ParsedDocument:
    """Result of parsing a document."""

    doc_id: str
    name: str
    path: Path
    doc_type: str
    content: str
    pages: list[str] = field(default_factory=list)
    images: list[ImageInfo] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)


class BaseParser(ABC):
    """Abstract base class for document parsers."""

    supported_extensions: list[str] = []

    @classmethod
    def can_parse(cls, file_path: str | Path) -> bool:
        """Check if this parser can handle the given file.

        Args:
            file_path: Path to the file.

        Returns:
            True if this parser supports the file extension.
        """
        path = Path(file_path)
        return path.suffix.lower() in cls.supported_extensions

    @abstractmethod
    def parse(self, file_path: str | Path) -> ParsedDocument:
        """Parse a document file.

        Args:
            file_path: Path to the document file.

        Returns:
            ParsedDocument containing the extracted content.
        """
        pass

    @staticmethod
    def generate_doc_id(file_path: Path) -> str:
        """Generate a unique document ID based on file path and modification time.

        Args:
            file_path: Path to the file.

        Returns:
            A unique document ID string.
        """
        import hashlib

        path = Path(file_path)
        stat = path.stat()
        content = f"{path.absolute()}:{stat.st_mtime}:{stat.st_size}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]
