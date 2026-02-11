"""Base OCR provider interface."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class OCRResult:
    """Result from OCR processing."""

    text: str
    confidence: Optional[float] = None
    language: Optional[str] = None
    metadata: dict = field(default_factory=dict)


class BaseOCR(ABC):
    """Abstract base class for OCR providers."""

    name: str = "base"

    @abstractmethod
    def process_image(self, image_path: str | Path) -> OCRResult:
        """Process a single image and extract text.

        Args:
            image_path: Path to the image file.

        Returns:
            OCRResult containing extracted text and metadata.
        """
        pass

    def process_images(self, image_paths: list[str | Path]) -> list[OCRResult]:
        """Process multiple images.

        Args:
            image_paths: List of image paths.

        Returns:
            List of OCRResult objects.
        """
        return [self.process_image(path) for path in image_paths]

    @classmethod
    def is_available(cls) -> bool:
        """Check if this OCR provider is available.

        Returns:
            True if the provider can be used.
        """
        return True

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"
