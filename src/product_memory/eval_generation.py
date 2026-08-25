from __future__ import annotations

import hashlib
import math
import re
from dataclasses import dataclass
from typing import Any

from product_memory.db import Database
from product_memory.retrieval.service import _QUESTION_WORDS

_SENTENCE = re.compile(r"(?<=[.!?:;])\s+|\n+")
_WORD = re.compile(r"[^\W\d_][\w\-]*", re.UNICODE)
_MIN_TERM_LENGTH = 4
_ANSWER_GRADE = 3


@dataclass(slots=True)
class GeneratedCase:
    question: str
    source_path: str
    document_id: str
    chunk_id: str
    salience: float


def _tokens(text: str) -> list[str]:
    return [match.group(0).lower() for match in _WORD.finditer(text)]


def corpus_document_frequency(db: Database) -> dict[str, int]:
    """How many chunks each word appears in, read from the index rather than the files."""
    with db.connection() as conn:
        rows = conn.execute(
            "SELECT word, ndoc FROM ts_stat($$SELECT search_vector FROM chunks$$)"
        ).fetchall()
    return {row["word"]: int(row["ndoc"]) for row in rows}


@dataclass(slots=True)
class Vocabulary:
    """Corpus word statistics, restricted to words worth building a question from."""

    idf: dict[str, float]
    usable: frozenset[str]

    def weight(self, token: str) -> float:
        return self.idf.get(token, 0.0)


def build_vocabulary(
    frequencies: dict[str, int],
    total_chunks: int,
    min_frequency: int = 2,
    max_frequency_ratio: float = 0.05,
) -> Vocabulary:
    ceiling = max(min_frequency, int(total_chunks * max_frequency_ratio))
    idf = {word: math.log(1 + total_chunks / (1 + count)) for word, count in frequencies.items()}
    # A word appearing exactly once across the whole corpus is far more often a scanning error
    # or a broken ligature than a distinctive term, and asking for it tests nothing.
    usable = frozenset(
        word
        for word, count in frequencies.items()
        if min_frequency <= count <= ceiling
        and len(word) >= _MIN_TERM_LENGTH
        and word not in _QUESTION_WORDS
    )
    return Vocabulary(idf=idf, usable=usable)


def _salient_terms(text: str, vocabulary: Vocabulary, limit: int) -> list[str]:
    """The words that set this passage apart from the rest of the corpus, in written order."""
    seen: set[str] = set()
    scored: list[tuple[float, int, str]] = []
    for position, token in enumerate(_tokens(text)):
        if token in seen or token not in vocabulary.usable:
            continue
        seen.add(token)
        scored.append((vocabulary.weight(token), position, token))
    best = sorted(scored, key=lambda item: item[0], reverse=True)[:limit]
    return [token for _, _, token in sorted(best, key=lambda item: item[1])]


def _best_sentence(content: str, vocabulary: Vocabulary, min_terms: int) -> str | None:
    candidates = []
    for sentence in _SENTENCE.split(content):
        cleaned = " ".join(sentence.split())
        # Distinct words, because the question is built from distinct words too. A sentence that
        # repeats one term is not as informative as its length suggests.
        terms = {token for token in _tokens(cleaned) if token in vocabulary.usable}
        if len(terms) < min_terms:
            continue
        # Longer sentences accumulate weight simply by being longer, so score the average.
        weight = sum(vocabulary.weight(token) for token in terms) / len(terms)
        candidates.append((weight, cleaned))
    if not candidates:
        return None
    return max(candidates, key=lambda item: item[0])[1]


def _sample_documents(db: Database, count: int, seed: str, project: str | None) -> list[dict]:
    project_clause = "AND d.metadata->>'project' = %(project)s" if project else ""
    with db.connection() as conn:
        rows = conn.execute(
            f"""
            SELECT d.id, d.source_path, d.title
            FROM documents d
            WHERE d.is_active = TRUE
              AND EXISTS (SELECT 1 FROM chunks c WHERE c.document_id = d.id)
              {project_clause}
            ORDER BY md5(d.id::text || %(seed)s)
            LIMIT %(count)s
            """,
            {"seed": seed, "count": count, "project": project},
        ).fetchall()
    return [dict(row) for row in rows]


def _chunks_for(db: Database, document_id: Any, min_chars: int) -> list[dict]:
    with db.connection() as conn:
        rows = conn.execute(
            """
            SELECT id, content FROM chunks
            WHERE document_id = %(document_id)s AND length(content) >= %(min_chars)s
            ORDER BY chunk_index
            """,
            {"document_id": document_id, "min_chars": min_chars},
        ).fetchall()
    return [dict(row) for row in rows]


def generate_cases(
    db: Database,
    count: int = 50,
    seed: str = "product-memory",
    terms_per_question: int = 8,
    min_chunk_chars: int = 400,
) -> list[GeneratedCase]:
    """Build known-item probes: take a passage, ask for it in its own distinctive words.

    This measures whether the index can find the document a passage came from. That is a real
    property worth guarding against regressions, but it is not the same as answering a question
    a person would actually ask, and the wording comes from the document itself, which flatters
    any signal that matches on words. Treat the result as a regression net, not as a score.

    One bias is sharp enough to name. The probe is drawn from the richest sentence anywhere in a
    chunk, so these questions reward any change that lets the reranker read further into a chunk,
    whether or not it judges relevance better. Measured against a handwritten set, raising
    RERANKER_MAX_LENGTH moved these questions and the written ones in opposite directions. Do not
    tune truncation, chunk size, or anything else that changes how much text a stage sees, on a
    generated set alone.
    """
    with db.connection() as conn:
        total_chunks = int(conn.execute("SELECT count(*) AS n FROM chunks").fetchone()["n"])
    if not total_chunks:
        return []

    vocabulary = build_vocabulary(corpus_document_frequency(db), total_chunks)
    # Oversample, because a document can fail to yield a usable passage.
    documents = _sample_documents(db, count * 3, seed, project=None)

    cases: list[GeneratedCase] = []
    for document in documents:
        if len(cases) == count:
            break
        chunks = _chunks_for(db, document["id"], min_chunk_chars)
        if not chunks:
            continue
        chunk = max(
            chunks,
            key=lambda item: sum(
                vocabulary.weight(token)
                for token in set(_tokens(item["content"])) & vocabulary.usable
            ),
        )
        sentence = _best_sentence(chunk["content"], vocabulary, min_terms=terms_per_question)
        if sentence is None:
            continue
        terms = _salient_terms(sentence, vocabulary, terms_per_question)
        if len(terms) < max(3, terms_per_question // 2):
            continue
        cases.append(
            GeneratedCase(
                question=" ".join(terms),
                source_path=document["source_path"],
                document_id=str(document["id"]),
                chunk_id=str(chunk["id"]),
                salience=round(sum(vocabulary.weight(token) for token in terms) / len(terms), 3),
            )
        )
    return cases


def render_yaml(cases: list[GeneratedCase], seed: str) -> str:
    digest = hashlib.sha256(seed.encode("utf-8")).hexdigest()[:12]
    lines = [
        "# Generated by `product-memory generate-eval`. Do not commit: the questions and paths",
        "# below are drawn from your own documents.",
        "#",
        "# Each question is a known-item probe: the most distinctive words of one passage, asked",
        "# back. It checks that the passage's document can still be found, which catches",
        "# regressions well. It is not a question a person would ask, and its wording comes from",
        "# the document, which flatters signals that match on words. Replace these with real",
        "# questions as you write them, and keep the generated ones as a regression net.",
        "#",
        "# The probe sentence can sit anywhere in its chunk, so these questions reward any change",
        "# that lets a stage read further into a chunk, whether or not it judges relevance better.",
        "# Do not tune RERANKER_MAX_LENGTH or CHUNK_SIZE against this set alone.",
        "#",
        f"# seed: {seed} ({digest})",
        f"# questions: {len(cases)}",
        "",
    ]
    for case in cases:
        question = case.question.replace('"', '\\"')
        path = case.source_path.replace('"', '\\"')
        lines.append(f'- question: "{question}"')
        lines.append("  expect:")
        lines.append(f'    - path: "{path}"')
        lines.append(f"      grade: {_ANSWER_GRADE}")
        lines.append("")
    return "\n".join(lines)
