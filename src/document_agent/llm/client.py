"""LLM client for OpenAI-compatible APIs."""

import base64
from pathlib import Path
from typing import Any, Optional

from openai import OpenAI

from document_agent.config import settings


class LLMClient:
    """Client for interacting with OpenAI-compatible LLM APIs."""

    def __init__(
        self,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
    ):
        """Initialize the LLM client.

        Args:
            base_url: Base URL for the API. Defaults to settings.llm_base_url.
            api_key: API key. Defaults to settings.llm_api_key.
            model: Model name. Defaults to settings.llm_model.
        """
        self.base_url = base_url or settings.llm_base_url
        self.api_key = api_key or settings.llm_api_key
        self.model = model or settings.llm_model

        self.client = OpenAI(
            base_url=self.base_url,
            api_key=self.api_key,
        )

    def chat(
        self,
        messages: list[dict[str, Any]],
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        **kwargs,
    ) -> str:
        """Send a chat completion request.

        Args:
            messages: List of message dicts with 'role' and 'content'.
            model: Model to use. Defaults to self.model.
            temperature: Sampling temperature.
            max_tokens: Maximum tokens in response.
            **kwargs: Additional arguments passed to the API.

        Returns:
            The assistant's response text.
        """
        response = self.client.chat.completions.create(
            model=model or self.model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs,
        )
        return response.choices[0].message.content

    def chat_with_image(
        self,
        prompt: str,
        image_path: str | Path,
        model: Optional[str] = None,
        max_tokens: int = 1024,
    ) -> str:
        """Send a chat completion request with an image.

        Args:
            prompt: Text prompt for the image.
            image_path: Path to the image file.
            model: Model to use. Defaults to settings.vision_llm_model.
            max_tokens: Maximum tokens in response.

        Returns:
            The assistant's response text.
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

        messages = [
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
        ]

        response = self.client.chat.completions.create(
            model=model or settings.vision_llm_model,
            messages=messages,
            max_tokens=max_tokens,
        )
        return response.choices[0].message.content

    def embed(
        self,
        texts: list[str],
        model: Optional[str] = None,
    ) -> list[list[float]]:
        """Generate embeddings for texts.

        Args:
            texts: List of texts to embed.
            model: Embedding model to use. Defaults to settings.embedding_model.

        Returns:
            List of embedding vectors.
        """
        response = self.client.embeddings.create(
            model=model or settings.embedding_model,
            input=texts,
        )
        return [item.embedding for item in response.data]
