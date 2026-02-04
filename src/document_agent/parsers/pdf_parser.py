"""PDF parser using GLM-OCR API."""

import base64
import tempfile
from pathlib import Path
from typing import Optional

import fitz  # PyMuPDF
from openai import OpenAI

from document_agent.config import settings

from .base import BaseParser, ImageInfo, ParsedDocument


class PDFParser(BaseParser):
    """Parser for PDF files using GLM-OCR API."""

    supported_extensions = [".pdf"]

    def __init__(
        self,
        glm_ocr_base_url: Optional[str] = None,
        images_path: Optional[Path] = None,
    ):
        """Initialize the PDF parser.

        Args:
            glm_ocr_base_url: Base URL for GLM-OCR API server.
            images_path: Directory to save extracted images.
        """
        self.glm_ocr_base_url = glm_ocr_base_url or settings.glm_ocr_base_url
        self.images_path = Path(images_path or settings.images_path)

        # Initialize GLM-OCR client
        self.glm_client = OpenAI(
            base_url=self.glm_ocr_base_url,
            api_key="dummy",  # GLM-OCR doesn't require a real API key
        )

    def parse(self, file_path: str | Path) -> ParsedDocument:
        """Parse a PDF file using GLM-OCR.

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

            # Use GLM-OCR to extract text
            page_text = self._ocr_page(image_path)
            pages_content.append(page_text)

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
            },
        )

    def _ocr_page(self, image_path: Path) -> str:
        """Use GLM-OCR to extract text from a page image.

        Args:
            image_path: Path to the page image.

        Returns:
            Extracted text from the page.
        """
        # Read and encode the image
        with open(image_path, "rb") as f:
            image_data = base64.b64encode(f.read()).decode("utf-8")

        try:
            response = self.glm_client.chat.completions.create(
                model="glm-ocr",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{image_data}",
                                },
                            },
                            {
                                "type": "text",
                                "text": "Text Recognition:",
                            },
                        ],
                    }
                ],
                max_tokens=8192,
            )
            return response.choices[0].message.content
        except Exception as e:
            # Fallback: try to extract text with PyMuPDF if GLM-OCR fails
            return f"[OCR Error: {str(e)}]"

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
