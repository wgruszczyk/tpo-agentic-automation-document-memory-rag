from __future__ import annotations

import hashlib
import json
from abc import ABC, abstractmethod
from typing import Any


def passage_text(title: str, content: str) -> str:
    """Exactly what gets embedded for a chunk.

    Anything comparing against stored vectors has to build its input the same way; embedding the
    bare content instead silently handicaps whatever is being compared.
    """
    return f"Title: {title}\n\n{content}"


class EmbeddingProvider(ABC):
    @abstractmethod
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        raise NotImplementedError

    @abstractmethod
    def embed_query(self, text: str) -> list[float]:
        raise NotImplementedError

    @abstractmethod
    def profile(self) -> dict[str, Any]:
        raise NotImplementedError

    def fingerprint(self, extra: dict[str, Any] | None = None) -> str:
        payload = {**self.profile(), **(extra or {})}
        canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()
