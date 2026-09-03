from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

DEFAULT_GRADE = 1


@dataclass(slots=True)
class Expectation:
    fragment: str
    grade: int = DEFAULT_GRADE


@dataclass(slots=True)
class EvalCase:
    question: str
    expect: list[Expectation] = field(default_factory=list)
    project: str | None = None

    @property
    def fragments(self) -> list[str]:
        return [expectation.fragment for expectation in self.expect]


def _expectations(raw: Any, path: Path, index: int) -> list[Expectation]:
    if raw is None:
        return []
    if isinstance(raw, str):
        raw = [raw]
    if not isinstance(raw, list):
        raise ValueError(f"Question {index} in {path} must list what it expects")

    expectations: list[Expectation] = []
    for entry in raw:
        if isinstance(entry, dict):
            fragment = str(entry.get("path", entry.get("fragment", ""))).strip()
            grade = int(entry.get("grade", DEFAULT_GRADE))
        else:
            fragment, grade = str(entry).strip(), DEFAULT_GRADE
        if not fragment:
            continue
        if grade < 0:
            raise ValueError(f"Question {index} in {path} has a negative grade")
        expectations.append(Expectation(fragment=fragment, grade=grade))
    return expectations


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
        cases.append(
            EvalCase(
                question=question,
                expect=_expectations(entry.get("expect"), path, index),
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


def grade_of(source_path: str, expect: list[Expectation]) -> int:
    lowered = source_path.lower()
    # One document can satisfy several expectations; the strongest one describes it.
    grades = [item.grade for item in expect if item.fragment.lower() in lowered]
    return max(grades) if grades else 0


def _dcg(grades: list[int]) -> float:
    return sum(
        (2**grade - 1) / math.log2(position + 1) for position, grade in enumerate(grades, start=1)
    )


def ndcg(retrieved_grades: list[int], expect: list[Expectation], top_k: int) -> float:
    actual = _dcg(retrieved_grades[:top_k])
    ideal = _dcg(sorted((item.grade for item in expect), reverse=True)[:top_k])
    return actual / ideal if ideal else 0.0


def _percentile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = min(math.ceil(fraction * len(ordered)) - 1, len(ordered) - 1)
    return ordered[max(position, 0)]


JUDGE_PROMPT = """
You check whether an answer stayed inside its sources.

Read the SOURCES, then read the ANSWER. Reply with exactly one word:
SUPPORTED   - every factual claim in the answer is stated in the sources.
UNSUPPORTED - the answer asserts something the sources do not say, or contradicts them.

An answer that declines because the sources do not cover the question is SUPPORTED. Wording need
not match; meaning must. Reply with the single word and nothing else.
""".strip()


def _judge_verdict(provider: Any, options: Any, context: str, prose: str) -> bool | None:
    from product_memory.generation.base import ChatMessage, ChatProviderError

    try:
        result = provider.complete(
            [
                ChatMessage(role="system", content=JUDGE_PROMPT),
                ChatMessage(role="user", content=f"SOURCES:\n{context}\n\nANSWER:\n{prose}"),
            ],
            options,
        )
    except ChatProviderError:
        return None
    verdict = result.text.strip().upper()
    if "UNSUPPORTED" in verdict:
        return False
    if "SUPPORTED" in verdict:
        return True
    return None


def run_generation_evaluation(
    chat: Any, cases: list[EvalCase], judge_model: str | None = None
) -> dict[str, Any]:
    """Score the answers, not the retrieval underneath them.

    Two of the three measures are deterministic: whether the documents a question expected were
    among the sources the answer was built from, and how much of what it was given was expected.
    The third asks a model whether the answer stayed inside those sources, which is the only
    practical way to catch invention, and is worth exactly what a local judge is worth: a trend
    to watch across runs, not a number to quote. Judging with the same model family that wrote
    the answer flatters it.
    """
    from product_memory.generation.base import ChatOptions

    judge_options = (
        ChatOptions(
            model=judge_model,
            temperature=0.0,
            max_tokens=8,
            num_ctx=chat.settings.chat_num_ctx,
            thinking=False,
            keep_alive=chat.settings.chat_keep_alive,
            timeout_seconds=chat.settings.chat_timeout_seconds,
        )
        if judge_model
        else None
    )

    results: list[dict[str, Any]] = []
    durations: list[float] = []
    coverage_total = 0.0
    precision_total = 0.0
    scored = 0
    grounded = 0
    judged = 0
    supported = 0

    for case in cases:
        started = time.perf_counter()
        answer = chat.ask(case.question, project=case.project)
        durations.append(time.perf_counter() - started)
        if answer.grounded:
            grounded += 1

        cited = [citation.source_path for citation in answer.citations]
        coverage = precision = None
        if case.expect:
            scored += 1
            met = sum(
                1
                for item in case.expect
                if any(item.fragment.lower() in path.lower() for path in cited)
            )
            coverage = met / len(case.expect)
            relevant = sum(1 for path in cited if grade_of(path, case.expect) > 0)
            precision = relevant / len(cited) if cited else 0.0
            coverage_total += coverage
            precision_total += precision

        verdict = None
        if judge_options is not None and answer.context:
            verdict = _judge_verdict(chat.provider, judge_options, answer.context, answer.prose)
            if verdict is not None:
                judged += 1
                supported += int(verdict)

        results.append(
            {
                "question": case.question,
                "asked": answer.question,
                "grounded": answer.grounded,
                "citation_coverage": round(coverage, 4) if coverage is not None else None,
                "citation_precision": round(precision, 4) if precision is not None else None,
                "supported": verdict,
                "seconds": round(durations[-1], 3),
                "cited": cited,
                "answer": answer.prose,
            }
        )

    def averaged(total: float, over: int) -> float | None:
        return round(total / over, 4) if over else None

    return {
        "questions": len(cases),
        "scored": scored,
        "grounded_rate": averaged(float(grounded), len(cases)),
        "citation_coverage": averaged(coverage_total, scored),
        "citation_precision": averaged(precision_total, scored),
        "judged": judged,
        "groundedness": averaged(float(supported), judged),
        "judge_model": judge_model,
        "latency_seconds": {
            "mean": round(sum(durations) / len(durations), 3) if durations else None,
            "p50": round(_percentile(durations, 0.50), 3),
            "p95": round(_percentile(durations, 0.95), 3),
            "max": round(max(durations), 3) if durations else None,
        },
        "results": results,
    }


def run_evaluation(retriever: Any, cases: list[EvalCase], top_k: int) -> dict[str, Any]:
    results: list[dict[str, Any]] = []
    reciprocal_ranks = 0.0
    hits = 0
    scored = 0
    recall_total = 0.0
    precision_total = 0.0
    ndcg_total = 0.0
    durations: list[float] = []

    for case in cases:
        started = time.perf_counter()
        response = retriever.retrieve(
            query=case.question,
            top_k_documents=top_k,
            project=case.project,
            include_full_documents=False,
        )
        duration = time.perf_counter() - started
        durations.append(duration)

        source_paths = [document.source_path for document in response.documents]
        returned = source_paths[:top_k]
        rank = rank_of_expected(source_paths, case.fragments) if case.expect else None
        retrieved_grades = [grade_of(path, case.expect) for path in source_paths]

        case_ndcg = recall = precision = None
        if case.expect:
            scored += 1
            if rank:
                hits += 1
                reciprocal_ranks += 1.0 / rank
            # Several documents can match one fragment, so recall counts expectations met
            # rather than documents hit.
            met = sum(
                1
                for item in case.expect
                if any(item.fragment.lower() in path.lower() for path in returned)
            )
            recall = met / len(case.expect)
            relevant = sum(1 for grade in retrieved_grades[:top_k] if grade > 0)
            precision = relevant / len(returned) if returned else 0.0
            case_ndcg = ndcg(retrieved_grades, case.expect, top_k)
            recall_total += recall
            precision_total += precision
            ndcg_total += case_ndcg

        results.append(
            {
                "question": case.question,
                "expect": [{"path": item.fragment, "grade": item.grade} for item in case.expect],
                "rank": rank,
                "hit": bool(rank),
                "recall": round(recall, 4) if recall is not None else None,
                "precision": round(precision, 4) if precision is not None else None,
                "ndcg": round(case_ndcg, 4) if case_ndcg is not None else None,
                "seconds": round(duration, 3),
                "top_documents": returned,
                "top_score": response.documents[0].score if response.documents else None,
            }
        )

    def averaged(total: float) -> float | None:
        return round(total / scored, 4) if scored else None

    return {
        "questions": len(cases),
        "scored": scored,
        "top_k": top_k,
        "hit_rate": averaged(float(hits)),
        "mrr": averaged(reciprocal_ranks),
        "recall": averaged(recall_total),
        "precision": averaged(precision_total),
        "ndcg": averaged(ndcg_total),
        "latency_seconds": {
            "mean": round(sum(durations) / len(durations), 3) if durations else None,
            "p50": round(_percentile(durations, 0.50), 3),
            "p95": round(_percentile(durations, 0.95), 3),
            "max": round(max(durations), 3) if durations else None,
        },
        "misses": [
            result["question"] for result in results if result["expect"] and not result["hit"]
        ],
        "results": results,
    }
