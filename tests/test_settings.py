from product_memory.settings import Settings


def test_default_scan_interval_is_30_seconds() -> None:
    settings = Settings(_env_file=None)
    assert settings.scan_interval_seconds == 30.0


def test_empty_optional_embedding_dimensions_is_none() -> None:
    settings = Settings(embedding_dimensions="")
    assert settings.embedding_dimensions is None
