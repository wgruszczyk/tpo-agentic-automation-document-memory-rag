from __future__ import annotations

import logging
import os
import threading

from product_memory.models import ChunkResult
from product_memory.settings import Settings

LOGGER = logging.getLogger(__name__)


def _positions(chunks: list[ChunkResult], key) -> dict[str, int]:  # type: ignore[no-untyped-def]
    ranked = sorted(chunks, key=key, reverse=True)
    return {chunk.id: position for position, chunk in enumerate(ranked)}


class Reranker:
    """Reads each candidate together with the question and scores how well it answers it.

    The retrieval score compares two summaries written independently of each other: the question
    is turned into a vector now, the passage was turned into one long before the question existed.
    A passage that merely discusses the right subject can therefore outrank the one that states
    the answer. Reading both at once is far too slow to search with, but affordable over a
    shortlist, which is why this runs last and only reorders what retrieval already found.
    """

    def __init__(self, settings: Settings):
        self.settings = settings
        self._model = None
        self._lock = threading.Lock()
        self._score_lock = threading.Lock()
        os.environ.setdefault("HF_HOME", str(settings.hf_home))

    def _is_model_cached(self) -> bool:
        cache_dir_name = "models--" + self.settings.reranker_model.replace("/", "--")
        snapshots_dir = self.settings.hf_home / cache_dir_name / "snapshots"
        if not snapshots_dir.is_dir():
            return False
        return any(
            any(snapshot.glob(pattern))
            for snapshot in snapshots_dir.iterdir()
            if snapshot.is_dir()
            for pattern in ("*.safetensors", "pytorch_model.bin")
        )

    def _load_model(self):  # type: ignore[no-untyped-def]
        if self._model is not None:
            return self._model
        with self._lock:
            if self._model is None:
                cached = self._is_model_cached()
                if cached:
                    os.environ.setdefault("HF_HUB_OFFLINE", "1")
                    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

                import torch
                from sentence_transformers import CrossEncoder

                torch.set_num_threads(self.settings.reranker_threads)
                self._model = CrossEncoder(
                    self.settings.reranker_model,
                    device="cpu",
                    max_length=self.settings.reranker_max_length,
                    cache_folder=str(self.settings.hf_home),
                    local_files_only=cached,
                )
        return self._model

    @staticmethod
    def _passage(chunk: ChunkResult) -> str:
        """What the reranker reads: the document it came from, then the text.

        Retrieval weights the title and the path on purpose, because many questions name a
        document rather than describe its contents. Scoring the bare text would discard that at
        the very last step, and demote the right chunk of the right file for saying in passing
        what its title says outright.
        """
        return f"{chunk.document_title}\n\n{chunk.content}"

    def rerank(self, query: str, chunks: list[ChunkResult], limit: int) -> list[ChunkResult]:
        """Return the best `limit` chunks for this query, most convincing first."""
        if not chunks or limit < 1:
            return []
        if len(chunks) == 1:
            return chunks[:limit]
        try:
            model = self._load_model()
            with self._score_lock:
                scores = model.predict(
                    [(query, self._passage(chunk)) for chunk in chunks],
                    batch_size=self.settings.reranker_batch_size,
                    show_progress_bar=False,
                )
        except Exception:
            # A shortlist in retrieval order is a worse answer than a reordered one, but it is
            # still an answer. Losing the model should cost quality, not availability.
            LOGGER.warning("Reranking failed; keeping retrieval order.", exc_info=True)
            return chunks[:limit]

        ordered = sorted(
            zip(chunks, (float(score) for score in scores), strict=True),
            key=lambda pair: pair[1],
            reverse=True,
        )
        # The reranker gets a vote, not a veto. It reads meaning better than retrieval does, but
        # retrieval knows things it never sees: which file this came from, how the question's
        # words are spread across title and path, how recent it is. Questions that name a document
        # are decided by exactly that, so replacing the order outright trades one kind of failure
        # for another. Fusing positions keeps both opinions, on the same footing as the signals
        # that were already fused to build this shortlist.
        rerank_position = {chunk.id: position for position, (chunk, _) in enumerate(ordered)}
        rerank_score = {chunk.id: score for chunk, score in ordered}
        # Retrieval's opinion is the best case any one signal made, not the blended order. A
        # passage can be the single strongest keyword match in the index and still land deep in
        # the blend; it reaches this shortlist on that signal's insistence, so judging it again
        # by the blend that buried it would just repeat the mistake that hid it.
        by_semantic = _positions(chunks, lambda chunk: chunk.semantic_score)
        by_lexical = _positions(chunks, lambda chunk: chunk.lexical_score)
        retrieval_position = {
            chunk.id: min(position, by_semantic[chunk.id], by_lexical[chunk.id])
            for position, chunk in enumerate(chunks)
        }
        k = self.settings.reranker_rrf_k
        fused = sorted(
            enumerate(chunks),
            key=lambda pair: (
                self.settings.reranker_weight / (k + rerank_position[pair[1].id])
                + (1 - self.settings.reranker_weight) / (k + retrieval_position[pair[1].id])
            ),
            reverse=True,
        )
        return [
            chunk.model_copy(update={"rerank_score": rerank_score[chunk.id]})
            for _, chunk in fused[:limit]
        ]
