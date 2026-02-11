"""OCR providers for PDF text extraction."""

from .base import BaseOCR, OCRResult
from .glm_ocr import GLMOCR
from .tesseract_ocr import TesseractOCR
from .vision_llm_ocr import VisionLLMOCR

__all__ = [
    "BaseOCR",
    "OCRResult",
    "GLMOCR",
    "TesseractOCR",
    "VisionLLMOCR",
]


def get_ocr_provider(provider: str, **kwargs) -> BaseOCR:
    """Factory function to get OCR provider by name.

    Args:
        provider: Provider name ('glm', 'tesseract', 'vision_llm', 'gpt4v', 'claude')
        **kwargs: Provider-specific configuration

    Returns:
        OCR provider instance

    Raises:
        ValueError: If provider is not supported
    """
    providers = {
        "glm": GLMOCR,
        "glm-ocr": GLMOCR,
        "tesseract": TesseractOCR,
        "vision_llm": VisionLLMOCR,
        "gpt4v": lambda **kw: VisionLLMOCR(model="gpt-4o", **kw),
        "gpt-4o": lambda **kw: VisionLLMOCR(model="gpt-4o", **kw),
        "claude": lambda **kw: VisionLLMOCR(model="claude-3-sonnet-20240229", **kw),
    }

    provider_lower = provider.lower()
    if provider_lower not in providers:
        available = ", ".join(providers.keys())
        raise ValueError(f"Unknown OCR provider: {provider}. Available: {available}")

    provider_class = providers[provider_lower]
    if callable(provider_class) and not isinstance(provider_class, type):
        return provider_class(**kwargs)
    return provider_class(**kwargs)
