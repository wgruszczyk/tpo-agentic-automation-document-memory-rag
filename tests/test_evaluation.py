from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from product_memory.evaluation import (
    Expectation,
    grade_of,
    load_cases,
    ndcg,
    rank_of_expected,
    run_evaluation,
)


class StubRetriever:
    def __init__(self, answers: dict[str, list[str]]):
        self.answers = answers
        self.queries: list[str] = []

    def retrieve(self, query: str, **_: object) -> SimpleNamespace:
        self.queries.append(query)
        documents = [
            SimpleNamespace(source_path=path, score=1.0) for path in self.answers.get(query, [])
        ]
        return SimpleNamespace(documents=documents)


def test_load_cases_accepts_a_single_expected_path_as_a_string(tmp_path: Path) -> None:
    path = tmp_path / "questions.yaml"
    path.write_text("- question: What are the fees?\n  expect: pricing\n", encoding="utf-8")

    cases = load_cases(path)

    assert cases[0].question == "What are the fees?"
    assert cases[0].fragments == ["pricing"]
    assert cases[0].expect[0].grade == 1


def test_load_cases_rejects_a_question_without_text(tmp_path: Path) -> None:
    path = tmp_path / "questions.yaml"
    path.write_text("- expect: pricing\n", encoding="utf-8")

    with pytest.raises(ValueError, match="missing a question"):
        load_cases(path)


def test_rank_of_expected_matches_a_case_insensitive_path_fragment() -> None:
    assert rank_of_expected(["a/Notes.md", "b/Pricing-2026.xlsx"], ["pricing"]) == 2
    assert rank_of_expected(["a/Notes.md"], ["pricing"]) is None


def test_run_evaluation_reports_hit_rate_and_mrr(tmp_path: Path) -> None:
    path = tmp_path / "questions.yaml"
    path.write_text(
        "- question: fees?\n"
        "  expect: pricing\n"
        "- question: brands?\n"
        "  expect: countries\n",
        encoding="utf-8",
    )
    retriever = StubRetriever({"fees?": ["notes.md", "pricing.xlsx"], "brands?": ["notes.md"]})

    report = run_evaluation(retriever, load_cases(path), top_k=5)

    assert report["scored"] == 2
    assert report["hit_rate"] == 0.5
    assert report["mrr"] == 0.25
    assert report["misses"] == ["brands?"]


def test_load_cases_reads_graded_expectations(tmp_path: Path) -> None:
    path = tmp_path / "questions.yaml"
    path.write_text(
        "- question: fees?\n"
        "  expect:\n"
        "    - path: pricing\n"
        "      grade: 3\n"
        "    - path: notes\n"
        "      grade: 1\n",
        encoding="utf-8",
    )

    expect = load_cases(path)[0].expect

    assert [(item.fragment, item.grade) for item in expect] == [("pricing", 3), ("notes", 1)]


def test_a_grade_describes_the_strongest_expectation_a_document_matches() -> None:
    expect = [Expectation("pricing", 3), Expectation("2026", 1)]

    assert grade_of("a/Pricing-2026.xlsx", expect) == 3
    assert grade_of("a/plan-2026.md", expect) == 1
    assert grade_of("a/notes.md", expect) == 0


def test_ndcg_is_one_when_the_best_documents_come_first() -> None:
    expect = [Expectation("a", 3), Expectation("b", 2)]

    assert ndcg([3, 2], expect, top_k=5) == pytest.approx(1.0)
    # The same documents in the wrong order must score lower.
    assert ndcg([2, 3], expect, top_k=5) < 1.0
    assert ndcg([0, 0], expect, top_k=5) == 0.0


def test_run_evaluation_reports_recall_precision_ndcg_and_latency(tmp_path: Path) -> None:
    path = tmp_path / "questions.yaml"
    path.write_text(
        "- question: fees?\n"
        "  expect:\n"
        "    - path: pricing\n"
        "      grade: 3\n"
        "    - path: annex\n"
        "      grade: 2\n",
        encoding="utf-8",
    )
    retriever = StubRetriever({"fees?": ["pricing.xlsx", "notes.md"]})

    report = run_evaluation(retriever, load_cases(path), top_k=2)

    assert report["recall"] == 0.5
    assert report["precision"] == 0.5
    assert 0.0 < report["ndcg"] < 1.0
    assert report["latency_seconds"]["p95"] >= report["latency_seconds"]["p50"]
    assert report["results"][0]["seconds"] >= 0
