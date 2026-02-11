"""Vision LLM based OCR provider."""

import base64
from pathlib import Path
from typing import Optional

from openai import OpenAI

from document_agent.config import settings

from .base import BaseOCR, OCRResult


class VisionLLMOCR(BaseOCR):
    """OCR provider using Vision LLM (GPT-4V, Claude Vision, etc.).

    Uses multimodal LLMs with vision capabilities for text extraction.
    Supports any OpenAI-compatible vision API.
    """

    name = "vision-llm"

    DEFAULT_PROMPT = """이 이미지에서 모든 텍스트를 추출해주세요.

지침:
1. 이미지에 보이는 모든 텍스트를 정확하게 추출하세요.
2. 문서의 구조(제목, 단락, 목록, 표)를 최대한 보존하세요.
3. 표가 있으면 Markdown 표 형식으로 변환하세요.
4. 수식이 있으면 LaTeX 형식으로 표현하세요.
5. 텍스트만 출력하고, 추가 설명은 하지 마세요."""

    def __init__(
        self,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        prompt: Optional[str] = None,
    ):
        """Initialize Vision LLM OCR provider.

        Args:
            base_url: Base URL for API (default: settings.llm_base_url)
            api_key: API key (default: settings.llm_api_key)
            model: Model name (default: settings.vision_llm_model)
            prompt: Custom extraction prompt
        """
        self.base_url = base_url or settings.llm_base_url
        self.api_key = api_key or settings.llm_api_key
        self.model = model or settings.vision_llm_model
        self.prompt = prompt or self.DEFAULT_PROMPT

        self.client = OpenAI(
            base_url=self.base_url,
            api_key=self.api_key,
        )

    def process_image(self, image_path: str | Path) -> OCRResult:
        """Process an image using Vision LLM.

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
                                "text": self.prompt,
                            },
                        ],
                    }
                ],
                max_tokens=4096,
            )

            text = response.choices[0].message.content
            return OCRResult(
                text=text,
                metadata={
                    "provider": self.name,
                    "model": self.model,
                },
            )

        except Exception as e:
            return OCRResult(
                text=f"[Vision LLM OCR Error: {str(e)}]",
                metadata={"error": str(e)},
            )

    @classmethod
    def is_available(cls) -> bool:
        """Check if Vision LLM is available."""
        return bool(settings.llm_api_key)
