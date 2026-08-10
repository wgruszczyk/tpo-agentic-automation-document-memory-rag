from product_memory.settings import Settings


def test_default_scan_interval_is_30_seconds() -> None:
    settings = Settings(_env_file=None)
    assert settings.scan_interval_seconds == 30.0


def test_default_supported_extensions_include_documents() -> None:
    settings = Settings(_env_file=None)
    assert {".txt", ".md", ".vtt", ".srt", ".pdf", ".docx", ".pptx", ".xlsx", ".msg"}.issubset(
        settings.extensions
    )


def test_default_supported_extensions_include_images() -> None:
    settings = Settings(_env_file=None)
    assert {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp", ".gif"}.issubset(
        settings.extensions
    )


def test_default_retrieval_limits_are_conservative() -> None:
    settings = Settings(_env_file=None)
    assert settings.min_semantic_score == 0.60
    assert settings.default_top_k_documents == 7
    assert settings.max_returned_documents == 25


def test_empty_optional_embedding_dimensions_is_none() -> None:
    settings = Settings(embedding_dimensions="")
    assert settings.embedding_dimensions is None
