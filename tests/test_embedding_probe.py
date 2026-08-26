from __future__ import annotations

from contextlib import contextmanager
from typing import Any

import numpy as np
import pytest

from product_memory.embedding_probe import (
    StaleIndexError,
    _score,
    build_pool,
    compare_embedding_models,
)
from product_memory.embeddings.base import passage_text
from product_memory.evaluation import EvalCase, Expectation

POOL = [
    {"id": "1", "title": "Pricing", "content": "fees", "source_path": "a/pricing.xlsx"},
    {"id": "2", "title": "Notes", "content": "misc", "source_path": "b/notes.md"},
]


class FakeCursor:
    def __init__(self, rows: list[dict]):
        self.rows = rows

    def fetchall(self) -> list[dict]:
        return self.rows


class FakeConnection:
    def __init__(self, wanted: list[dict], noise: list[dict]):
        self.wanted = wanted
        self.noise = noise
        self.queries: list[str] = []

    def execute(self, sql: str, params: dict | None = None) -> FakeCursor:
        self.queries.append(sql)
        return FakeCursor(self.noise if "AND NOT" in sql else self.wanted)


class FakeDatabase:
    def __init__(self, wanted: list[dict], noise: list[dict], status: str = "ready"):
        self.connection_instance = FakeConnection(wanted, noise)
        self.status = status

    @contextmanager
    def connection(self):
        yield self.connection_instance

    def get_state(self, _key: str) -> dict[str, Any]:
        return {"status": self.status}


class FakeProvider:
    """Ranks by how many query words a passage contains, so the test has a predictable order."""

    def __init__(self, model: str, seen: list[str] | None = None):
        self.model = model
        self.seen = seen if seen is not None else []

    def _vector(self, text: str) -> list[float]:
        lowered = text.lower()
        return [float("fee" in lowered), float("note" in lowered), 0.1]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        self.seen.extend(texts)
        return [self._vector(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vector(text)

    def profile(self) -> dict[str, Any]:
        return {"model": self.model, "dimension": 3}


def _rows_with_vectors(rows: list[dict], provider: FakeProvider) -> list[dict]:
    return [
        {**row, "embedding": provider._vector(passage_text(row["title"], row["content"]))}
        for row in rows
    ]


def test_the_pool_keeps_expected_documents_and_adds_distractors() -> None:
    db = FakeDatabase(wanted=[POOL[0]], noise=[POOL[1]])
    cases = [EvalCase(question="fees?", expect=[Expectation("pricing", 3)])]

    pool = build_pool(db, cases, distractors=1)  # type: ignore[arg-type]

    assert [row["source_path"] for row in pool] == ["a/pricing.xlsx", "b/notes.md"]
    # LIKE would read the underscores that fill source paths as single-character wildcards.
    assert all("ILIKE" not in query for query in db.connection_instance.queries)
    assert all("position(lower(fragment) in lower(d.source_path))" in query
               for query in db.connection_instance.queries)


def test_a_question_set_without_expectations_cannot_be_compared() -> None:
    db = FakeDatabase(wanted=[], noise=[])

    with pytest.raises(ValueError, match="no expected documents"):
        build_pool(db, [EvalCase(question="fees?")], distractors=1)  # type: ignore[arg-type]


def test_scoring_rewards_the_expected_document_appearing_first() -> None:
    provider = FakeProvider("stub")
    vectors = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    paths = ["a/pricing.xlsx", "b/notes.md"]
    cases = [EvalCase(question="fee", expect=[Expectation("pricing", 3)])]

    assert _score(vectors, paths, provider, cases, top_k=2) == {"hit_rate": 1.0, "mrr": 1.0}


def test_a_document_ranked_second_scores_half_the_reciprocal_rank() -> None:
    provider = FakeProvider("stub")
    # The distractor sits closer to the query than the expected document does.
    vectors = np.array([[0.9, 0.0, 0.0], [0.5, 0.0, 0.0]])
    paths = ["b/notes.md", "a/pricing.xlsx"]
    cases = [EvalCase(question="fee", expect=[Expectation("pricing", 3)])]

    assert _score(vectors, paths, provider, cases, top_k=2) == {"hit_rate": 1.0, "mrr": 0.5}


def test_the_candidate_is_given_the_same_text_the_index_stored() -> None:
    current = FakeProvider("current")
    seen: list[str] = []
    candidate = FakeProvider("candidate", seen=seen)
    db = FakeDatabase(
        wanted=_rows_with_vectors([POOL[0]], current),
        noise=_rows_with_vectors([POOL[1]], current),
    )
    cases = [EvalCase(question="fee", expect=[Expectation("pricing", 3)])]

    report = compare_embedding_models(db, current, candidate, cases, distractors=1)  # type: ignore[arg-type]

    # Embedding the bare content instead would quietly handicap the candidate.
    assert seen == ["Title: Pricing\n\nfees", "Title: Notes\n\nmisc"]
    assert report["current"]["model"] == "current"
    assert report["candidate"]["model"] == "candidate"
    assert report["current"]["hit_rate"] == report["candidate"]["hit_rate"] == 1.0
    assert report["pool"] == {
        "chunks": 2,
        "target_documents": 1,
        "distractor_documents": 1,
        "questions": 1,
        "top_k": 7,
    }


def test_an_index_that_is_not_ready_cannot_be_compared_against() -> None:
    db = FakeDatabase(wanted=[], noise=[], status="reindexing")
    cases = [EvalCase(question="fee", expect=[Expectation("pricing", 3)])]

    with pytest.raises(StaleIndexError, match="not ready"):
        compare_embedding_models(db, FakeProvider("a"), FakeProvider("b"), cases)  # type: ignore[arg-type]
