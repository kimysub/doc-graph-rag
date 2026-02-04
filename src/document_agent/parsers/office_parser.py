"""Office document parser using MarkItDown."""

from pathlib import Path
from typing import Optional

from markitdown import MarkItDown
from openai import OpenAI

from document_agent.config import settings

from .base import BaseParser, ImageInfo, ParsedDocument


class OfficeParser(BaseParser):
    """Parser for Office documents (DOCX, PPTX, XLSX) using MarkItDown."""

    supported_extensions = [".docx", ".pptx", ".xlsx", ".xls", ".doc"]

    def __init__(
        self,
        llm_client: Optional[OpenAI] = None,
        llm_model: Optional[str] = None,
        images_path: Optional[Path] = None,
    ):
        """Initialize the Office parser.

        Args:
            llm_client: OpenAI client for image descriptions. If provided,
                        MarkItDown will generate descriptions for embedded images.
            llm_model: Model to use for image descriptions.
            images_path: Directory to save extracted images.
        """
        self.images_path = Path(images_path or settings.images_path)
        self.llm_model = llm_model or settings.vision_llm_model

        # Initialize MarkItDown with optional LLM for image descriptions
        if llm_client:
            self.md = MarkItDown(
                llm_client=llm_client,
                llm_model=self.llm_model,
            )
        else:
            self.md = MarkItDown()

    def parse(self, file_path: str | Path) -> ParsedDocument:
        """Parse an Office document using MarkItDown.

        Args:
            file_path: Path to the Office document.

        Returns:
            ParsedDocument containing the extracted content.
        """
        file_path = Path(file_path)
        doc_id = self.generate_doc_id(file_path)

        # Determine document type
        suffix = file_path.suffix.lower()
        doc_type_map = {
            ".docx": "docx",
            ".doc": "doc",
            ".pptx": "pptx",
            ".xlsx": "xlsx",
            ".xls": "xls",
        }
        doc_type = doc_type_map.get(suffix, "office")

        # Convert using MarkItDown
        result = self.md.convert(str(file_path))
        content = result.text_content

        # Extract images if any (MarkItDown handles this internally)
        images = self._extract_images(file_path, doc_id)

        # For PPTX, try to split by slides
        pages = []
        if doc_type == "pptx":
            pages = self._split_pptx_slides(content)

        return ParsedDocument(
            doc_id=doc_id,
            name=file_path.name,
            path=file_path,
            doc_type=doc_type,
            content=content,
            pages=pages,
            images=images,
            metadata={
                "original_format": suffix,
            },
        )

    def _split_pptx_slides(self, content: str) -> list[str]:
        """Split PPTX content into slides.

        Args:
            content: Full markdown content from MarkItDown.

        Returns:
            List of slide contents.
        """
        # MarkItDown typically uses "---" or "# Slide" patterns
        slides = []
        current_slide = []

        for line in content.split("\n"):
            if line.strip().startswith("# Slide") or line.strip() == "---":
                if current_slide:
                    slides.append("\n".join(current_slide))
                current_slide = [line]
            else:
                current_slide.append(line)

        if current_slide:
            slides.append("\n".join(current_slide))

        return slides if len(slides) > 1 else []

    def _extract_images(self, file_path: Path, doc_id: str) -> list[ImageInfo]:
        """Extract images from Office documents.

        Args:
            file_path: Path to the Office document.
            doc_id: Document ID.

        Returns:
            List of ImageInfo for extracted images.
        """
        images = []
        doc_images_dir = self.images_path / doc_id
        suffix = file_path.suffix.lower()

        try:
            if suffix == ".pptx":
                images = self._extract_pptx_images(file_path, doc_id, doc_images_dir)
            elif suffix == ".docx":
                images = self._extract_docx_images(file_path, doc_id, doc_images_dir)
            # XLSX typically doesn't have many images worth extracting
        except Exception:
            # Silently handle extraction errors
            pass

        return images

    def _extract_pptx_images(
        self, file_path: Path, doc_id: str, doc_images_dir: Path
    ) -> list[ImageInfo]:
        """Extract images from PPTX files.

        Args:
            file_path: Path to the PPTX file.
            doc_id: Document ID.
            doc_images_dir: Directory to save images.

        Returns:
            List of ImageInfo for extracted images.
        """
        from zipfile import ZipFile

        images = []
        doc_images_dir.mkdir(parents=True, exist_ok=True)

        with ZipFile(file_path, "r") as zip_file:
            for file_info in zip_file.filelist:
                if file_info.filename.startswith("ppt/media/"):
                    # Extract image
                    image_name = Path(file_info.filename).name
                    image_data = zip_file.read(file_info.filename)

                    image_path = doc_images_dir / image_name
                    with open(image_path, "wb") as f:
                        f.write(image_data)

                    image_id = f"{doc_id}_{image_name}"
                    images.append(
                        ImageInfo(
                            id=image_id,
                            path=image_path,
                        )
                    )

        return images

    def _extract_docx_images(
        self, file_path: Path, doc_id: str, doc_images_dir: Path
    ) -> list[ImageInfo]:
        """Extract images from DOCX files.

        Args:
            file_path: Path to the DOCX file.
            doc_id: Document ID.
            doc_images_dir: Directory to save images.

        Returns:
            List of ImageInfo for extracted images.
        """
        from zipfile import ZipFile

        images = []
        doc_images_dir.mkdir(parents=True, exist_ok=True)

        with ZipFile(file_path, "r") as zip_file:
            for file_info in zip_file.filelist:
                if file_info.filename.startswith("word/media/"):
                    # Extract image
                    image_name = Path(file_info.filename).name
                    image_data = zip_file.read(file_info.filename)

                    image_path = doc_images_dir / image_name
                    with open(image_path, "wb") as f:
                        f.write(image_data)

                    image_id = f"{doc_id}_{image_name}"
                    images.append(
                        ImageInfo(
                            id=image_id,
                            path=image_path,
                        )
                    )

        return images
