from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(slots=True)
class EvalCase:
    question: str
    expect: list[str]
    project: str | None = None


def load_cases(path: Path) -> list[EvalCase]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or []
    if not isinstance(raw, list):
        raise ValueError(f"{path} must contain a list of questions")

    cases: list[EvalCase] = []
    for index, entry in enumerate(raw, start=1):
        if not isinstance(entry, dict):
            raise ValueError(f"Question {index} in {path} must be a mapping")
        question = str(entry.get("question", "")).strip()
        if not question:
            raise ValueError(f"Question {index} in {path} is missing a question")
        expect = entry.get("expect") or []
        if isinstance(expect, str):
            expect = [expect]
        cases.append(
            EvalCase(
                question=question,
                expect=[str(item).strip() for item in expect if str(item).strip()],
                project=entry.get("project"),
            )
        )
    return cases


def rank_of_expected(source_paths: list[str], expect: list[str]) -> int | None:
    for rank, source_path in enumerate(source_paths, start=1):
        lowered = source_path.lower()
        if any(fragment.lower() in lowered for fragment in expect):
            return rank
    return None


def run_evaluation(retriever: Any, cases: list[EvalCase], top_k: int) -> dict[str, Any]:
    results: list[dict[str, Any]] = []
    reciprocal_ranks = 0.0
    hits = 0
    scored = 0

    for case in cases:
        response = retriever.retrieve(
            query=case.question,
            top_k_documents=top_k,
            project=case.project,
            include_full_documents=False,
        )
        source_paths = [document.source_path for document in response.documents]
        rank = rank_of_expected(source_paths, case.expect) if case.expect else None
        if case.expect:
            scored += 1
            if rank:
                hits += 1
                reciprocal_ranks += 1.0 / rank
        results.append(
            {
                "question": case.question,
                "expect": case.expect,
                "rank": rank,
                "hit": bool(rank),
                "top_documents": source_paths[:top_k],
                "top_score": response.documents[0].score if response.documents else None,
            }
        )

    return {
        "questions": len(cases),
        "scored": scored,
        "top_k": top_k,
        "hit_rate": round(hits / scored, 4) if scored else None,
        "mrr": round(reciprocal_ranks / scored, 4) if scored else None,
        "misses": [result["question"] for result in results if result["expect"] and not result["hit"]],
        "results": results,
    }
