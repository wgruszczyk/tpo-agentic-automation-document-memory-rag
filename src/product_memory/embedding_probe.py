from __future__ import annotations

from typing import Any

import numpy as np

from product_memory.db import Database
from product_memory.embeddings.base import EmbeddingProvider, passage_text
from product_memory.evaluation import EvalCase
from product_memory.ingestion.service import INDEX_STATE_KEY


class StaleIndexError(RuntimeError):
    """Raised when the stored vectors were not produced by the provider being compared against."""


def _as_array(value: Any) -> np.ndarray:
    for attribute in ("to_numpy", "to_list"):
        if hasattr(value, attribute):
            return np.asarray(getattr(value, attribute)(), dtype=np.float32)
    return np.asarray(value, dtype=np.float32)


# Plain substring, case-insensitive: the same test the scoring uses. LIKE would read the
# underscores that fill source paths as single-character wildcards.
_MATCHES_FRAGMENT = """
    EXISTS (
        SELECT 1 FROM unnest(%(fragments)s::text[]) AS fragment
        WHERE position(lower(fragment) in lower(d.source_path)) > 0
    )
"""


def _normalize(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / np.maximum(norms, 1e-12)


def build_pool(
    db: Database, cases: list[EvalCase], distractors: int
) -> list[dict[str, Any]]:
    """Every chunk of the documents the questions expect, plus unrelated chunks to hide them among."""
    fragments = [expectation.fragment for case in cases for expectation in case.expect]
    if not fragments:
        raise ValueError("The question set names no expected documents to compare against.")

    with db.connection() as conn:
        wanted = conn.execute(
            f"""
            SELECT c.id, c.content, c.embedding, d.title, d.source_path
            FROM chunks c JOIN documents d ON d.id = c.document_id
            WHERE d.is_active = TRUE AND {_MATCHES_FRAGMENT}
            """,
            {"fragments": fragments},
        ).fetchall()
        noise = conn.execute(
            f"""
            SELECT c.id, c.content, c.embedding, d.title, d.source_path
            FROM chunks c JOIN documents d ON d.id = c.document_id
            WHERE d.is_active = TRUE AND NOT {_MATCHES_FRAGMENT}
            ORDER BY md5(c.id::text)
            LIMIT %(limit)s
            """,
            {"fragments": fragments, "limit": distractors},
        ).fetchall()
    return [dict(row) for row in wanted] + [dict(row) for row in noise]


def _score(
    vectors: np.ndarray,
    paths: list[str],
    provider: EmbeddingProvider,
    cases: list[EvalCase],
    top_k: int,
) -> dict[str, float]:
    hits = 0.0
    reciprocal = 0.0
    for case in cases:
        query = np.asarray(provider.embed_query(case.question), dtype=np.float32)
        query /= max(float(np.linalg.norm(query)), 1e-12)
        ranked: list[str] = []
        for index in np.argsort(-(vectors @ query)):
            path = paths[index]
            if path not in ranked:
                ranked.append(path)
            if len(ranked) == top_k:
                break
        for position, path in enumerate(ranked, start=1):
            if any(item.fragment.lower() in path.lower() for item in case.expect):
                hits += 1
                reciprocal += 1.0 / position
                break
    total = len(cases)
    return {
        "hit_rate": round(hits / total, 4),
        "mrr": round(reciprocal / total, 4),
    }


def compare_embedding_models(
    db: Database,
    current: EmbeddingProvider,
    candidate: EmbeddingProvider,
    cases: list[EvalCase],
    distractors: int = 1000,
    top_k: int = 7,
) -> dict[str, Any]:
    """Judge a candidate embedding model without re-embedding the whole index.

    Only the candidate has to embed anything: the current model's vectors are read back from the
    index. Scoring is cosine alone, so this isolates what changing the embedding model actually
    changes, and ignores the lexical signal, the recency boost and the reranker that follow it. A
    candidate that separates no better here will not earn a full reindex.
    """
    profile = db.get_state(INDEX_STATE_KEY) or {}
    if profile.get("status") != "ready":
        raise StaleIndexError(f"The index is not ready to compare against: {profile}")

    pool = build_pool(db, cases, distractors)
    paths = [row["source_path"] for row in pool]
    target_paths = {
        path
        for path in paths
        if any(
            item.fragment.lower() in path.lower() for case in cases for item in case.expect
        )
    }
    stored = _normalize(np.array([_as_array(row["embedding"]) for row in pool]))
    fresh = _normalize(
        np.asarray(
            candidate.embed_documents(
                [passage_text(row["title"], row["content"]) for row in pool]
            ),
            dtype=np.float32,
        )
    )

    return {
        "pool": {
            "chunks": len(pool),
            "target_documents": len(target_paths),
            "distractor_documents": len(set(paths) - target_paths),
            "questions": len(cases),
            "top_k": top_k,
        },
        "current": {
            "model": current.profile()["model"],
            **_score(stored, paths, current, cases, top_k),
        },
        "candidate": {
            "model": candidate.profile()["model"],
            **_score(fresh, paths, candidate, cases, top_k),
        },
    }
