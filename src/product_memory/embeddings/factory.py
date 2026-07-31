from product_memory.embeddings.base import EmbeddingProvider
from product_memory.embeddings.local_hf import LocalHFEmbeddingProvider
from product_memory.embeddings.openai_provider import OpenAIEmbeddingProvider
from product_memory.settings import Settings


def create_embedding_provider(settings: Settings) -> EmbeddingProvider:
    if settings.embedding_provider == "local_hf":
        return LocalHFEmbeddingProvider(settings)
    if settings.embedding_provider == "openai":
        return OpenAIEmbeddingProvider(settings)
    raise ValueError(f"Unsupported embedding provider: {settings.embedding_provider}")
