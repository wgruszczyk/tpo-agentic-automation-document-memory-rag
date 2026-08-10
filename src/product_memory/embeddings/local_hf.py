from __future__ import annotations

import os
import threading
from typing import Any

import numpy as np

from product_memory.embeddings.base import EmbeddingProvider
from product_memory.settings import Settings


class LocalHFEmbeddingProvider(EmbeddingProvider):
    def __init__(self, settings: Settings):
        self.settings = settings
        self._model = None
        self._lock = threading.Lock()
        self._encode_lock = threading.Lock()
        os.environ.setdefault("HF_HOME", str(settings.hf_home))

    def _load_model(self):  # type: ignore[no-untyped-def]
        if self._model is not None:
            return self._model
        with self._lock:
            if self._model is None:
                cached = self._is_model_cached()
                if cached:
                    # Model already downloaded: skip Hub freshness checks so restarts stay fully
                    # offline. The env vars cover libraries that read them lazily; local_files_only
                    # is the authoritative switch because huggingface_hub snapshots its env-based
                    # constants at import time.
                    os.environ.setdefault("HF_HUB_OFFLINE", "1")
                    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

                import torch
                from sentence_transformers import SentenceTransformer

                torch.set_num_threads(self.settings.local_embedding_threads)
                kwargs: dict[str, Any] = {
                    "device": "cpu",
                    "cache_folder": str(self.settings.hf_home),
                    "local_files_only": cached,
                }
                if self.settings.embedding_revision:
                    kwargs["revision"] = self.settings.embedding_revision
                self._model = SentenceTransformer(self.settings.embedding_model, **kwargs)
        return self._model

    def _is_model_cached(self) -> bool:
        # An interrupted download leaves config files behind without weights. Treating that as
        # cached would switch on offline mode and make the failure permanent, so require weights.
        cache_dir_name = "models--" + self.settings.embedding_model.replace("/", "--")
        snapshots_dir = self.settings.hf_home / cache_dir_name / "snapshots"
        if not snapshots_dir.is_dir():
            return False
        return any(
            any(snapshot.glob(pattern))
            for snapshot in snapshots_dir.iterdir()
            if snapshot.is_dir()
            for pattern in ("*.safetensors", "pytorch_model.bin")
        )

    @property
    def uses_e5_prefixes(self) -> bool:
        return "e5" in self.settings.embedding_model.lower()

    def _document_text(self, text: str) -> str:
        return f"passage: {text}" if self.uses_e5_prefixes else text

    def _query_text(self, text: str) -> str:
        return f"query: {text}" if self.uses_e5_prefixes else text

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        model = self._load_model()
        with self._encode_lock:
            vectors = model.encode(
                [self._document_text(text) for text in texts],
                batch_size=self.settings.embedding_batch_size,
                normalize_embeddings=True,
                show_progress_bar=False,
                convert_to_numpy=True,
            )
        return np.asarray(vectors, dtype=np.float32).tolist()

    def embed_query(self, text: str) -> list[float]:
        model = self._load_model()
        with self._encode_lock:
            vector = model.encode(
                self._query_text(text),
                normalize_embeddings=True,
                show_progress_bar=False,
                convert_to_numpy=True,
            )
        return np.asarray(vector, dtype=np.float32).tolist()

    def profile(self) -> dict[str, Any]:
        model = self._load_model()
        get_dimension = getattr(model, "get_embedding_dimension", None)
        if get_dimension is None:
            get_dimension = model.get_sentence_embedding_dimension
        dimension = int(get_dimension())
        resolved_revision = self.settings.embedding_revision
        if not resolved_revision:
            try:
                first_module = model._first_module()  # noqa: SLF001 - stable SentenceTransformer helper
                resolved_revision = getattr(first_module.auto_model.config, "_commit_hash", None)
            except (AttributeError, IndexError):
                resolved_revision = None
        return {
            "provider": "local_hf",
            "model": self.settings.embedding_model,
            "revision": resolved_revision or "unresolved",
            "dimension": dimension,
            "normalized": True,
            "query_prefix": "query: " if self.uses_e5_prefixes else "",
            "document_prefix": "passage: " if self.uses_e5_prefixes else "",
        }
