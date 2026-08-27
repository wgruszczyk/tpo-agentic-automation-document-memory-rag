from __future__ import annotations

import os
from pathlib import Path

# Enough to tell a finished download from an interrupted one, across the formats in use:
# safetensors and pytorch for transformers, model.bin for CTranslate2.
_WEIGHT_PATTERNS = ("*.safetensors", "pytorch_model.bin", "model.bin")


class ModelDownloadDisabled(RuntimeError):
    """Raised when a model is missing from the cache and fetching it is not allowed."""


def is_model_cached(hf_home: Path, model: str) -> bool:
    # An interrupted download leaves config files behind without weights. Treating that as
    # cached would switch on offline mode and make the failure permanent, so require weights.
    snapshots_dir = hf_home / f"models--{model.replace('/', '--')}" / "snapshots"
    if not snapshots_dir.is_dir():
        return False
    return any(
        any(snapshot.glob(pattern))
        for snapshot in snapshots_dir.iterdir()
        if snapshot.is_dir()
        for pattern in _WEIGHT_PATTERNS
    )


def prepare_model_load(hf_home: Path, model: str, allow_download: bool) -> bool:
    """Return whether the model can be loaded offline, refusing a surprise download when asked to."""
    if is_model_cached(hf_home, model):
        # huggingface_hub snapshots its env-based constants at import time, so local_files_only
        # stays the authoritative switch; these only cover libraries that read them lazily.
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
        return True
    if not allow_download:
        raise ModelDownloadDisabled(
            f"{model} is not in the model cache at {hf_home}, and ALLOW_MODEL_DOWNLOAD is false. "
            "Run 'product-memory warmup' to fetch it deliberately."
        )
    return False
