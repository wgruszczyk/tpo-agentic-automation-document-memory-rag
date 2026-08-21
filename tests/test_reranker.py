from __future__ import annotations

from datetime import UTC, datetime

from product_memory.models import ChunkResult
from product_memory.retrieval.reranker import Reranker
from product_memory.settings import Settings


def _chunk(chunk_id: str, title: str = "doc", content: str = "text") -> ChunkResult:
    return ChunkResult(
        id=chunk_id,
        document_id="doc-1",
        document_title=title,
        source_path="/knowledge/doc.md",
        chunk_index=0,
        content=content,
        start_char=0,
        end_char=10,
        effective_at=datetime(2026, 1, 1, tzinfo=UTC),
        semantic_score=0.8,
        lexical_score=0.1,
        recency_score=0.9,
        score=0.5,
    )


class StubModel:
    def __init__(self, scores: dict[str, float]) -> None:
        self.scores = scores
        self.pairs: list[tuple[str, str]] = []

    def predict(self, pairs, batch_size, show_progress_bar):  # type: ignore[no-untyped-def]
        self.pairs = list(pairs)
        return [self.scores[passage.split("\n\n", 1)[1]] for _, passage in pairs]


def _reranker(scores: dict[str, float], **overrides: object) -> tuple[Reranker, StubModel]:
    reranker = Reranker(Settings(_env_file=None, **overrides))  # type: ignore[arg-type]
    model = StubModel(scores)
    reranker._model = model  # noqa: SLF001 - injecting the model is the point of the test
    return reranker, model


def test_rerank_shows_the_model_the_document_title_not_only_the_text() -> None:
    reranker, model = _reranker({"body one": 1.0, "body two": 0.0})

    reranker.rerank(
        "who owns this",
        [_chunk("a", title="Annex6_IT-Security", content="body one"), _chunk("b", content="body two")],
        limit=2,
    )

    # Questions that name a document are answered by the title, which the chunk text may never
    # repeat. Scoring the bare content throws that away at the last step.
    assert model.pairs[0] == ("who owns this", "Annex6_IT-Security\n\nbody one")


def test_rerank_lets_retrieval_keep_a_say_rather_than_being_overruled() -> None:
    reranker, _ = _reranker(
        {"first": 0.0, "second": 0.2, "third": 0.4, "fourth": 0.3},
        reranker_weight=0.5,
        reranker_rrf_k=20,
    )

    ranked = reranker.rerank(
        "a question",
        [
            _chunk("a", content="first"),
            _chunk("b", content="second"),
            _chunk("c", content="third"),
            _chunk("d", content="fourth"),
        ],
        limit=4,
    )

    # The model ranks them c, d, b, a - it likes "c" most and retrieval's own favourite "a" least.
    # "c" still wins, because the model has the better eye for meaning. But "a" comes second on
    # retrieval's word alone, ahead of the chunk the model placed directly below "c". Questions
    # that name a document are won exactly there. Handing the decision over would return c, d, b, a
    # and drop "a" to last.
    assert [chunk.id for chunk in ranked] == ["c", "a", "b", "d"]


def test_rerank_can_be_handed_the_whole_decision() -> None:
    reranker, _ = _reranker({"first": 0.0, "second": 0.1, "third": 0.2}, reranker_weight=1.0)

    ranked = reranker.rerank(
        "a question",
        [_chunk("a", content="first"), _chunk("b", content="second"), _chunk("c", content="third")],
        limit=3,
    )

    assert [chunk.id for chunk in ranked] == ["c", "b", "a"]


def test_rerank_records_the_score_it_gave() -> None:
    reranker, _ = _reranker({"first": 0.25, "second": 0.75})

    ranked = reranker.rerank(
        "a question", [_chunk("a", content="first"), _chunk("b", content="second")], limit=2
    )

    assert {chunk.id: chunk.rerank_score for chunk in ranked} == {"a": 0.25, "b": 0.75}


def test_rerank_falls_back_to_retrieval_order_when_the_model_fails() -> None:
    reranker = Reranker(Settings(_env_file=None))

    class Broken:
        def predict(self, *args, **kwargs):  # type: ignore[no-untyped-def]
            raise RuntimeError("no model here")

    reranker._model = Broken()  # noqa: SLF001

    ranked = reranker.rerank("a question", [_chunk("a"), _chunk("b")], limit=1)

    # Losing the model should cost quality, not availability.
    assert [chunk.id for chunk in ranked] == ["a"]
