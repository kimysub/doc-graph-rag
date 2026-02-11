"""PDF parser with pluggable OCR providers."""

from pathlib import Path
from typing import Optional, Union

import fitz  # PyMuPDF

from document_agent.config import settings

from .base import BaseParser, ImageInfo, ParsedDocument
from .ocr import BaseOCR, get_ocr_provider


class PDFParser(BaseParser):
    """Parser for PDF files with pluggable OCR support.

    Supports multiple OCR providers:
    - glm: GLM-OCR (requires external API server)
    - tesseract: Tesseract OCR (local, open-source)
    - vision_llm: Vision LLM (GPT-4V, Claude Vision, etc.)
    - gpt4v: GPT-4 Vision
    - claude: Claude Vision
    """

    supported_extensions = [".pdf"]

    def __init__(
        self,
        ocr_provider: Optional[Union[str, BaseOCR]] = None,
        images_path: Optional[Path] = None,
        **ocr_kwargs,
    ):
        """Initialize the PDF parser.

        Args:
            ocr_provider: OCR provider name or instance.
                         Options: 'glm', 'tesseract', 'vision_llm', 'gpt4v', 'claude'
                         Default: settings.ocr_provider
            images_path: Directory to save extracted images.
            **ocr_kwargs: Additional arguments passed to OCR provider.
        """
        self.images_path = Path(images_path or settings.images_path)

        # Initialize OCR provider
        if ocr_provider is None:
            ocr_provider = settings.ocr_provider

        if isinstance(ocr_provider, str):
            self.ocr = get_ocr_provider(ocr_provider, **ocr_kwargs)
        elif isinstance(ocr_provider, BaseOCR):
            self.ocr = ocr_provider
        else:
            raise ValueError(f"Invalid ocr_provider type: {type(ocr_provider)}")

    def parse(self, file_path: str | Path) -> ParsedDocument:
        """Parse a PDF file using the configured OCR provider.

        Args:
            file_path: Path to the PDF file.

        Returns:
            ParsedDocument containing the extracted content.
        """
        file_path = Path(file_path)
        doc_id = self.generate_doc_id(file_path)

        # Create directory for this document's images
        doc_images_dir = self.images_path / doc_id
        doc_images_dir.mkdir(parents=True, exist_ok=True)

        # Open PDF with PyMuPDF
        pdf_doc = fitz.open(file_path)

        pages_content = []
        images = []

        for page_num, page in enumerate(pdf_doc):
            # Render page to image
            pix = page.get_pixmap(dpi=150)
            image_path = doc_images_dir / f"page_{page_num + 1}.png"
            pix.save(str(image_path))

            # Use OCR provider to extract text
            ocr_result = self.ocr.process_image(image_path)
            pages_content.append(ocr_result.text)

            # Extract embedded images from the page
            page_images = self._extract_page_images(page, page_num, doc_id, doc_images_dir)
            images.extend(page_images)

        pdf_doc.close()

        # Combine all pages
        full_content = "\n\n---\n\n".join(
            f"## Page {i + 1}\n\n{content}" for i, content in enumerate(pages_content)
        )

        return ParsedDocument(
            doc_id=doc_id,
            name=file_path.name,
            path=file_path,
            doc_type="pdf",
            content=full_content,
            pages=pages_content,
            images=images,
            metadata={
                "page_count": len(pages_content),
                "ocr_provider": self.ocr.name,
            },
        )

    def _extract_page_images(
        self,
        page: fitz.Page,
        page_num: int,
        doc_id: str,
        doc_images_dir: Path,
    ) -> list[ImageInfo]:
        """Extract embedded images from a PDF page.

        Args:
            page: PyMuPDF page object.
            page_num: Page number (0-indexed).
            doc_id: Document ID.
            doc_images_dir: Directory to save images.

        Returns:
            List of ImageInfo for extracted images.
        """
        images = []
        image_list = page.get_images(full=True)

        for img_idx, img_info in enumerate(image_list):
            xref = img_info[0]
            try:
                base_image = page.parent.extract_image(xref)
                image_bytes = base_image["image"]
                image_ext = base_image["ext"]

                image_id = f"{doc_id}_p{page_num + 1}_img{img_idx + 1}"
                image_path = doc_images_dir / f"img_p{page_num + 1}_{img_idx + 1}.{image_ext}"

                with open(image_path, "wb") as f:
                    f.write(image_bytes)

                images.append(
                    ImageInfo(
                        id=image_id,
                        path=image_path,
                        page_num=page_num + 1,
                    )
                )
            except Exception:
                # Skip images that can't be extracted
                continue

        return images
