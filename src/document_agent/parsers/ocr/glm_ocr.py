"""GLM-OCR provider for PDF text extraction."""

import base64
from pathlib import Path
from typing import Optional

from openai import OpenAI

from document_agent.config import settings

from .base import BaseOCR, OCRResult


class GLMOCR(BaseOCR):
    """OCR provider using GLM-OCR API server.

    GLM-OCR is a multimodal OCR model optimized for document understanding.
    Requires running GLM-OCR as an API server (vLLM, SGLang, or Ollama).
    """

    name = "glm-ocr"

    # Available prompts for different extraction modes
    PROMPTS = {
        "text": "Text Recognition:",
        "table": "Table Recognition:",
        "formula": "Formula Recognition:",
    }

    def __init__(
        self,
        base_url: Optional[str] = None,
        model: str = "glm-ocr",
        mode: str = "text",
    ):
        """Initialize GLM-OCR provider.

        Args:
            base_url: Base URL for GLM-OCR API server.
            model: Model name (default: glm-ocr)
            mode: Extraction mode ('text', 'table', 'formula')
        """
        self.base_url = base_url or settings.glm_ocr_base_url
        self.model = model
        self.mode = mode

        self.client = OpenAI(
            base_url=self.base_url,
            api_key="dummy",  # GLM-OCR doesn't require real API key
        )

    def process_image(self, image_path: str | Path) -> OCRResult:
        """Process an image using GLM-OCR.

        Args:
            image_path: Path to the image file.

        Returns:
            OCRResult with extracted text.
        """
        image_path = Path(image_path)

        # Read and encode the image
        with open(image_path, "rb") as f:
            image_data = base64.b64encode(f.read()).decode("utf-8")

        # Determine media type
        suffix = image_path.suffix.lower()
        media_types = {
            ".png": "image/png",
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".gif": "image/gif",
            ".webp": "image/webp",
        }
        media_type = media_types.get(suffix, "image/png")

        # Get prompt for mode
        prompt = self.PROMPTS.get(self.mode, self.PROMPTS["text"])

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:{media_type};base64,{image_data}",
                                },
                            },
                            {
                                "type": "text",
                                "text": prompt,
                            },
                        ],
                    }
                ],
                max_tokens=8192,
            )

            text = response.choices[0].message.content
            return OCRResult(
                text=text,
                metadata={
                    "provider": self.name,
                    "model": self.model,
                    "mode": self.mode,
                },
            )

        except Exception as e:
            return OCRResult(
                text=f"[GLM-OCR Error: {str(e)}]",
                metadata={"error": str(e)},
            )

    @classmethod
    def is_available(cls) -> bool:
        """Check if GLM-OCR server is available."""
        try:
            import httpx

            response = httpx.get(
                f"{settings.glm_ocr_base_url}/models",
                timeout=5.0,
            )
            return response.status_code == 200
        except Exception:
            return False
