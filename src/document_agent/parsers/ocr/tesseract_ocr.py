"""Tesseract OCR provider."""

from pathlib import Path
from typing import Optional

from .base import BaseOCR, OCRResult


class TesseractOCR(BaseOCR):
    """OCR provider using Tesseract OCR.

    Tesseract is a free, open-source OCR engine.
    Requires tesseract to be installed on the system.

    Install:
        - macOS: brew install tesseract
        - Ubuntu: sudo apt install tesseract-ocr
        - pip install pytesseract
    """

    name = "tesseract"

    def __init__(
        self,
        lang: str = "eng+kor",
        config: str = "",
    ):
        """Initialize Tesseract OCR provider.

        Args:
            lang: Language(s) for OCR (e.g., 'eng', 'kor', 'eng+kor')
            config: Additional Tesseract configuration
        """
        self.lang = lang
        self.config = config
        self._pytesseract = None

    @property
    def pytesseract(self):
        """Lazy load pytesseract."""
        if self._pytesseract is None:
            try:
                import pytesseract

                self._pytesseract = pytesseract
            except ImportError:
                raise ImportError(
                    "pytesseract is required for TesseractOCR. "
                    "Install with: pip install pytesseract"
                )
        return self._pytesseract

    def process_image(self, image_path: str | Path) -> OCRResult:
        """Process an image using Tesseract OCR.

        Args:
            image_path: Path to the image file.

        Returns:
            OCRResult with extracted text.
        """
        image_path = Path(image_path)

        try:
            from PIL import Image

            # Open image
            image = Image.open(image_path)

            # Run OCR
            text = self.pytesseract.image_to_string(
                image,
                lang=self.lang,
                config=self.config,
            )

            # Get confidence data
            data = self.pytesseract.image_to_data(
                image,
                lang=self.lang,
                output_type=self.pytesseract.Output.DICT,
            )

            # Calculate average confidence
            confidences = [
                int(c) for c in data["conf"] if c != "-1" and str(c).isdigit()
            ]
            avg_confidence = sum(confidences) / len(confidences) if confidences else None

            return OCRResult(
                text=text.strip(),
                confidence=avg_confidence / 100 if avg_confidence else None,
                language=self.lang,
                metadata={
                    "provider": self.name,
                    "lang": self.lang,
                },
            )

        except Exception as e:
            return OCRResult(
                text=f"[Tesseract Error: {str(e)}]",
                metadata={"error": str(e)},
            )

    @classmethod
    def is_available(cls) -> bool:
        """Check if Tesseract is available."""
        try:
            import pytesseract

            pytesseract.get_tesseract_version()
            return True
        except Exception:
            return False
