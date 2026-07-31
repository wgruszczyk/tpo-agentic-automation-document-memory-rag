from typing import Any

from product_memory.embeddings.base import EmbeddingProvider


class FakeProvider(EmbeddingProvider):
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [[1.0] for _ in texts]

    def embed_query(self, text: str) -> list[float]:
        return [1.0]

    def profile(self) -> dict[str, Any]:
        return {"provider": "fake", "model": "one", "dimension": 1}


def test_fingerprint_changes_with_chunking() -> None:
    provider = FakeProvider()
    one = provider.fingerprint({"chunk_size": 1000})
    two = provider.fingerprint({"chunk_size": 1200})
    assert one != two
    assert one == provider.fingerprint({"chunk_size": 1000})
