from __future__ import annotations

from typing import Any

import numpy as np
from openai import OpenAI

from product_memory.embeddings.base import EmbeddingProvider
from product_memory.settings import Settings


class OpenAIEmbeddingProvider(EmbeddingProvider):
    def __init__(self, settings: Settings):
        if not settings.openai_api_key:
            raise ValueError("OPENAI_API_KEY is required when EMBEDDING_PROVIDER=openai")
        self.settings = settings
        kwargs: dict[str, Any] = {"api_key": settings.openai_api_key}
        if settings.openai_base_url:
            kwargs["base_url"] = settings.openai_base_url
        self.client = OpenAI(**kwargs)
        self._dimension: int | None = settings.embedding_dimensions

    def _embed(self, texts: list[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for start in range(0, len(texts), self.settings.embedding_batch_size):
            batch = texts[start : start + self.settings.embedding_batch_size]
            kwargs: dict[str, Any] = {"model": self.settings.embedding_model, "input": batch}
            if self.settings.embedding_dimensions:
                kwargs["dimensions"] = self.settings.embedding_dimensions
            response = self.client.embeddings.create(**kwargs)
            ordered = sorted(response.data, key=lambda item: item.index)
            vectors.extend(np.asarray(item.embedding, dtype=np.float32).tolist() for item in ordered)
        if vectors and self._dimension is None:
            self._dimension = len(vectors[0])
        return vectors

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._embed(texts) if texts else []

    def embed_query(self, text: str) -> list[float]:
        return self._embed([text])[0]

    def profile(self) -> dict[str, Any]:
        if self._dimension is None:
            self.embed_query("embedding dimension probe")
        return {
            "provider": "openai",
            "model": self.settings.embedding_model,
            "revision": self.settings.embedding_revision or "provider-managed",
            "dimension": self._dimension,
            "base_url": self.settings.openai_base_url or "default",
            "normalized": False,
        }
