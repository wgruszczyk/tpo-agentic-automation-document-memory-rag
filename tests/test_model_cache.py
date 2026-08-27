import os
from pathlib import Path

import pytest

from product_memory.model_cache import ModelDownloadDisabled, is_model_cached, prepare_model_load

MODEL = "intfloat/multilingual-e5-small"


def _cache_snapshot(root: Path, filename: str) -> None:
    snapshot = root / "models--intfloat--multilingual-e5-small" / "snapshots" / "abc123"
    snapshot.mkdir(parents=True)
    (snapshot / filename).write_bytes(b"weights")


def test_a_model_is_not_cached_when_nothing_was_downloaded(tmp_path: Path) -> None:
    assert is_model_cached(tmp_path, MODEL) is False


def test_a_model_is_cached_once_its_weights_are_on_disk(tmp_path: Path) -> None:
    _cache_snapshot(tmp_path, "model.safetensors")

    assert is_model_cached(tmp_path, MODEL) is True


def test_an_interrupted_download_does_not_count_as_cached(tmp_path: Path) -> None:
    _cache_snapshot(tmp_path, "config.json")

    assert is_model_cached(tmp_path, MODEL) is False


def test_ctranslate2_weights_count_as_cached(tmp_path: Path) -> None:
    # faster-whisper ships a single model.bin rather than safetensors, and not recognising it
    # makes a downloaded model look permanently absent.
    _cache_snapshot(tmp_path, "model.bin")

    assert is_model_cached(tmp_path, MODEL) is True


def test_a_cached_model_loads_offline(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    monkeypatch.delenv("TRANSFORMERS_OFFLINE", raising=False)
    _cache_snapshot(tmp_path, "model.safetensors")

    assert prepare_model_load(tmp_path, MODEL, allow_download=False) is True
    assert os.environ["HF_HUB_OFFLINE"] == "1"
    assert os.environ["TRANSFORMERS_OFFLINE"] == "1"


def test_a_cold_cache_refuses_to_download_when_downloads_are_off(tmp_path: Path) -> None:
    with pytest.raises(ModelDownloadDisabled, match="ALLOW_MODEL_DOWNLOAD"):
        prepare_model_load(tmp_path, MODEL, allow_download=False)


def test_a_cold_cache_is_allowed_online_when_downloads_are_on(tmp_path: Path) -> None:
    assert prepare_model_load(tmp_path, MODEL, allow_download=True) is False
