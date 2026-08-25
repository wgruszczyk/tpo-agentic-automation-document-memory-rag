from product_memory.embeddings.local_hf import LocalHFEmbeddingProvider
from product_memory.settings import Settings


class CurrentModel:
    def get_embedding_dimension(self) -> int:
        return 384

    def get_sentence_embedding_dimension(self) -> int:
        raise AssertionError("deprecated dimension API was called")

    def _first_module(self) -> None:
        return None


class LegacyModel:
    def get_sentence_embedding_dimension(self) -> int:
        return 384

    def _first_module(self) -> None:
        return None


def provider_with_model(model: object) -> LocalHFEmbeddingProvider:
    provider = LocalHFEmbeddingProvider(Settings(_env_file=None))
    provider._model = model
    return provider


def test_profile_uses_current_dimension_api() -> None:
    profile = provider_with_model(CurrentModel()).profile()

    assert profile["dimension"] == 384


def test_profile_supports_legacy_dimension_api() -> None:
    profile = provider_with_model(LegacyModel()).profile()

    assert profile["dimension"] == 384
