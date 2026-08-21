from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    database_url: str = "postgresql://product_memory:product_memory_local_only@db:5432/product_memory"
    knowledge_dir: Path = Path("/knowledge")
    scan_interval_seconds: float = 30.0
    supported_extensions: str = (
        ".txt,.md,.markdown,.rst,.log,.vtt,.srt,.pdf,.docx,.pptx,.xlsx,.msg,.eml,"
        ".png,.jpg,.jpeg,.tif,.tiff,.bmp,.webp,.gif"
    )

    chunk_size: int = Field(default=1800, ge=300)
    chunk_overlap: int = Field(default=250, ge=0)

    enable_ocr: bool = True
    ocr_languages: str = "eng+pol"
    ocr_max_images_per_document: int = Field(default=100, ge=1, le=2000)
    ocr_min_image_pixels: int = Field(default=10_000, ge=0)
    ocr_min_characters: int = Field(default=12, ge=1)
    ocr_min_words: int = Field(default=6, ge=1)
    ocr_min_word_confidence: int = Field(default=60, ge=0, le=100)
    ocr_timeout_seconds: float = Field(default=30.0, gt=0)

    embedding_provider: Literal["local_hf", "openai"] = "local_hf"
    embedding_model: str = "intfloat/multilingual-e5-base"
    embedding_revision: str | None = None
    embedding_dimensions: int | None = Field(default=None, ge=1)
    embedding_batch_size: int = Field(default=16, ge=1, le=256)
    local_embedding_threads: int = Field(default=4, ge=1)
    hf_home: Path = Path("/models/huggingface")

    openai_api_key: str | None = None
    openai_base_url: str | None = None

    semantic_weight: float = Field(default=0.72, ge=0)
    lexical_weight: float = Field(default=0.13, ge=0)
    recency_weight: float = Field(default=0.15, ge=0)
    recency_half_life_days: float = Field(default=180, gt=0)

    # Gates on meaning alone. Blending recency in here used to hide old documents that still answer
    # the question, which is the wrong call for signed contracts.
    min_semantic_score: float = Field(default=0.60, ge=0, le=1)
    max_returned_documents: int = Field(default=25, ge=1, le=25)
    default_top_k_chunks: int = Field(default=10, ge=1, le=50)
    default_top_k_documents: int = Field(default=7, ge=1, le=25)
    default_context_chars: int = Field(default=24000, ge=2000, le=250000)
    max_full_document_chars: int = Field(default=120000, ge=1000, le=2_000_000)

    mcp_allowed_hosts: str = "localhost,localhost:*,127.0.0.1,127.0.0.1:*,[::1],[::1]:*"
    log_level: str = "INFO"

    @field_validator(
        "embedding_revision",
        "embedding_dimensions",
        "openai_api_key",
        "openai_base_url",
        mode="before",
    )
    @classmethod
    def empty_to_none(cls, value: object) -> object:
        if isinstance(value, str) and not value.strip():
            return None
        return value

    @field_validator("chunk_overlap")
    @classmethod
    def overlap_smaller_than_chunk(cls, value: int, info):  # type: ignore[no-untyped-def]
        chunk_size = info.data.get("chunk_size", 1800)
        if value >= chunk_size:
            raise ValueError("CHUNK_OVERLAP must be smaller than CHUNK_SIZE")
        return value

    @property
    def extensions(self) -> set[str]:
        return {item.strip().lower() for item in self.supported_extensions.split(",") if item.strip()}

    @property
    def ocr_language_list(self) -> list[str]:
        return [item.strip() for item in self.ocr_languages.replace(",", "+").split("+") if item.strip()]

    @property
    def allowed_hosts(self) -> list[str]:
        return [item.strip() for item in self.mcp_allowed_hosts.split(",") if item.strip()]


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
