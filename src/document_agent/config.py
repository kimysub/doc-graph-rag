"""Configuration management for Document Agent."""

from pathlib import Path
from typing import Literal, Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # OCR Settings (PDF OCR)
    ocr_provider: str = Field(
        default="glm",
        description="OCR provider: 'glm', 'tesseract', 'vision_llm', 'gpt4v', 'claude'",
    )
    glm_ocr_base_url: str = Field(
        default="http://localhost:8080/v1",
        description="Base URL for GLM-OCR API server",
    )
    tesseract_lang: str = Field(
        default="eng+kor",
        description="Tesseract OCR language(s)",
    )

    # LLM Settings (Q&A + Entity Extraction)
    llm_base_url: str = Field(
        default="https://api.openai.com/v1",
        description="Base URL for LLM API",
    )
    llm_api_key: str = Field(
        default="",
        description="API key for LLM",
    )
    llm_model: str = Field(
        default="gpt-4o",
        description="Model name for LLM",
    )

    # Vision LLM Settings (Image Description)
    vision_llm_model: str = Field(
        default="gpt-4o",
        description="Model name for Vision LLM",
    )

    # Embedding Settings
    embedding_model: str = Field(
        default="text-embedding-3-small",
        description="Model name for text embeddings",
    )

    # Data Storage Paths
    data_dir: Path = Field(
        default=Path("./data"),
        description="Root directory for data storage",
    )
    kuzu_db_path: Optional[Path] = Field(
        default=None,
        description="Path for Kuzu graph database",
    )
    chroma_db_path: Optional[Path] = Field(
        default=None,
        description="Path for ChromaDB vector store",
    )
    images_path: Optional[Path] = Field(
        default=None,
        description="Path for extracted images",
    )

    # Chunking Settings
    chunk_size: int = Field(
        default=512,
        description="Target chunk size in tokens",
    )
    chunk_overlap: int = Field(
        default=50,
        description="Overlap between chunks in tokens",
    )

    def model_post_init(self, __context) -> None:
        """Set default paths based on data_dir if not explicitly set."""
        if self.kuzu_db_path is None:
            self.kuzu_db_path = self.data_dir / "kuzu_db"
        if self.chroma_db_path is None:
            self.chroma_db_path = self.data_dir / "chroma_db"
        if self.images_path is None:
            self.images_path = self.data_dir / "images"

    def ensure_directories(self) -> None:
        """Create necessary directories if they don't exist."""
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.kuzu_db_path.mkdir(parents=True, exist_ok=True)
        self.chroma_db_path.mkdir(parents=True, exist_ok=True)
        self.images_path.mkdir(parents=True, exist_ok=True)


# Global settings instance
settings = Settings()
