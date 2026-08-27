from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class ImageRef(BaseModel):
    """A picture that can be downloaded and attached, not the picture itself."""

    id: str
    source_path: str
    label: str
    media_type: str
    width: int
    height: int
    byte_size: int
    url: str
    text: str = ""
    score: float = 0.0


class ChunkResult(BaseModel):
    id: str
    document_id: str
    document_title: str
    source_path: str
    chunk_index: int
    content: str
    start_char: int
    end_char: int
    effective_at: datetime
    metadata: dict[str, Any] = Field(default_factory=dict)
    semantic_score: float
    lexical_score: float
    recency_score: float
    score: float
    rerank_score: float | None = None
    images: list[ImageRef] = Field(default_factory=list)


class DocumentResult(BaseModel):
    id: str
    title: str
    source_path: str
    content: str | None = None
    effective_at: datetime
    source_modified_at: datetime
    metadata: dict[str, Any] = Field(default_factory=dict)
    score: float = 0.0
    matched_chunk_ids: list[str] = Field(default_factory=list)
    truncated: bool = False


class RetrievalResponse(BaseModel):
    query: str
    chunks: list[ChunkResult]
    documents: list[DocumentResult]
    context_pack: str
    index_profile: dict[str, Any]
    images: list[ImageRef] = Field(default_factory=list)


class ImageSearchResponse(BaseModel):
    query: str
    images: list[ImageRef]


class SearchItem(BaseModel):
    id: str
    title: str
    url: str
    text: str
    score: float
    metadata: dict[str, Any] = Field(default_factory=dict)


class SearchResponse(BaseModel):
    results: list[SearchItem]


class FetchResponse(BaseModel):
    id: str
    title: str
    text: str
    url: str
    metadata: dict[str, Any] = Field(default_factory=dict)
