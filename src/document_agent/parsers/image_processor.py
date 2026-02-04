"""Image processing module for extracting and describing images."""

import base64
from pathlib import Path
from typing import Optional

from openai import OpenAI

from document_agent.config import settings

from .base import ImageInfo


class ImageProcessor:
    """Processor for extracting and describing images from documents."""

    DEFAULT_PROMPT = """이 이미지를 상세히 설명해주세요. 
다음 내용을 포함해주세요:
- 이미지의 유형 (차트, 다이어그램, 사진, 스크린샷 등)
- 주요 시각적 요소와 텍스트
- 이미지가 전달하는 핵심 정보
- 차트/그래프인 경우 데이터 트렌드나 수치

간결하지만 정보가 풍부하게 설명해주세요."""

    def __init__(
        self,
        llm_client: Optional[OpenAI] = None,
        model: Optional[str] = None,
        images_path: Optional[Path] = None,
    ):
        """Initialize the image processor.

        Args:
            llm_client: OpenAI client for vision API.
            model: Vision model to use.
            images_path: Directory for storing images.
        """
        self.model = model or settings.vision_llm_model
        self.images_path = Path(images_path or settings.images_path)

        if llm_client:
            self.client = llm_client
        else:
            self.client = OpenAI(
                base_url=settings.llm_base_url,
                api_key=settings.llm_api_key,
            )

    def describe_image(
        self,
        image_path: str | Path,
        prompt: Optional[str] = None,
    ) -> str:
        """Generate a description for an image using Vision LLM.

        Args:
            image_path: Path to the image file.
            prompt: Custom prompt for the description.

        Returns:
            Text description of the image.
        """
        image_path = Path(image_path)

        if not image_path.exists():
            return "[Image not found]"

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
            ".bmp": "image/bmp",
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
                                "text": prompt or self.DEFAULT_PROMPT,
                            },
                        ],
                    }
                ],
                max_tokens=1024,
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"[Image description error: {str(e)}]"

    def process_images(
        self,
        images: list[ImageInfo],
        prompt: Optional[str] = None,
    ) -> list[ImageInfo]:
        """Process a list of images and add descriptions.

        Args:
            images: List of ImageInfo objects.
            prompt: Custom prompt for descriptions.

        Returns:
            Updated list of ImageInfo with descriptions.
        """
        processed = []
        for image in images:
            description = self.describe_image(image.path, prompt)
            processed.append(
                ImageInfo(
                    id=image.id,
                    path=image.path,
                    page_num=image.page_num,
                    description=description,
                )
            )
        return processed

    def save_image(
        self,
        image_data: bytes,
        doc_id: str,
        image_name: str,
    ) -> Path:
        """Save image data to disk.

        Args:
            image_data: Raw image bytes.
            doc_id: Document ID for organizing images.
            image_name: Name for the image file.

        Returns:
            Path where the image was saved.
        """
        doc_images_dir = self.images_path / doc_id
        doc_images_dir.mkdir(parents=True, exist_ok=True)

        image_path = doc_images_dir / image_name
        with open(image_path, "wb") as f:
            f.write(image_data)

        return image_path
