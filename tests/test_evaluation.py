from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from product_memory.evaluation import load_cases, rank_of_expected, run_evaluation


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
    assert cases[0].expect == ["pricing"]


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
