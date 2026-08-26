from __future__ import annotations

import re
from collections import OrderedDict
from datetime import UTC, datetime
from typing import Any

import numpy as np
from dateutil import parser as date_parser

from product_memory.db import Database
from product_memory.embeddings.base import EmbeddingProvider
from product_memory.ingestion.service import (
    FAILED_STATE_KEY,
    INDEX_STATE_KEY,
    SKIPPED_STATE_KEY,
)
from product_memory.metrics import RESULT_COUNT, stage
from product_memory.models import (
    ChunkResult,
    DocumentResult,
    FetchResponse,
    RetrievalResponse,
    SearchItem,
    SearchResponse,
)
from product_memory.retrieval.compressor import ContextCompressor
from product_memory.retrieval.reranker import Reranker
from product_memory.settings import Settings

# Words that say how a question is phrased rather than what it is about. A question carries a
# handful of names and nouns worth matching on; everything else is grammar, and counting it
# drags the score of every document towards the same value.
_QUESTION_WORDS = frozenset(
    {
        # English
        "the", "and", "for", "are", "was", "were", "who", "what", "when", "where", "which",
        "why", "how", "does", "did", "has", "have", "had", "can", "could", "would", "should",
        "his", "her", "its", "our", "their", "this", "that", "these", "those", "there",
        "from", "with", "about", "into", "over", "any", "all", "you", "your", "not", "but",
        "please",
        # Polish
        "jest", "sie", "czy", "jak", "kto", "gdzie", "kiedy", "dlaczego", "ktory", "ktora",
        "ktore", "oraz", "dla", "nie", "tak", "przez", "jaki", "jaka", "jakie", "byl", "byla",
        # German
        "der", "die", "das", "und", "wer", "wie", "wann", "warum", "welche", "welcher",
        "ist", "sind", "war", "waren", "ein", "eine", "den", "dem", "des", "von", "mit", "fur",
    }
)

_MIN_TERM_LENGTH = 3
_MAX_QUERY_TERMS = 12


def query_terms(query: str) -> list[str]:
    """The parts of a query worth looking for literally, in the order they were written."""
    terms: list[str] = []
    for raw in re.split(r"[^\w]+", query.lower(), flags=re.UNICODE):
        term = raw.strip("_")
        if len(term) < _MIN_TERM_LENGTH or term in _QUESTION_WORDS or term in terms:
            continue
        terms.append(term)
        if len(terms) == _MAX_QUERY_TERMS:
            break
    return terms


def parse_boundary(value: str | datetime | None, label: str) -> datetime | None:
    if value is None or (isinstance(value, str) and not value.strip()):
        return None
    if isinstance(value, datetime):
        parsed = value
    else:
        try:
            parsed = date_parser.parse(value.strip())
        except (ValueError, OverflowError) as exc:
            raise ValueError(f"{label} is not a recognisable date: {value!r}") from exc
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)


class Retriever:
    def __init__(
        self,
        settings: Settings,
        db: Database,
        provider: EmbeddingProvider,
        compressor: ContextCompressor,
        reranker: Reranker | None = None,
    ):
        self.settings = settings
        self.db = db
        self.provider = provider
        self.compressor = compressor
        self.reranker = reranker

    def retrieve(
        self,
        query: str,
        top_k_chunks: int | None = None,
        top_k_documents: int | None = None,
        project: str | None = None,
        include_full_documents: bool = True,
        max_context_chars: int | None = None,
        since: str | datetime | None = None,
        until: str | datetime | None = None,
    ) -> RetrievalResponse:
        query = query.strip()
        if not query:
            raise ValueError("query cannot be empty")
        since_at = parse_boundary(since, "since")
        until_at = parse_boundary(until, "until")
        if since_at and until_at and since_at > until_at:
            raise ValueError("since must not be later than until")
        profile = self._ready_profile()
        chunk_limit = min(max(top_k_chunks or self.settings.default_top_k_chunks, 1), 50)
        document_limit = min(
            max(top_k_documents or self.settings.default_top_k_documents, 1),
            self.settings.max_returned_documents,
            25,
        )
        context_limit = min(
            max(max_context_chars or self.settings.default_context_chars, 2000),
            250000,
        )
        # Retrieval only has to get the right chunk into the room; the reranker decides where it
        # stands. So search deeper than we return, and let every signal put its own favourites in
        # the pool rather than only the ones the fused order already liked.
        reranking = self.reranker is not None
        candidates = self._search_chunks(
            query=query,
            profile_hash=profile["fingerprint"],
            limit=max(chunk_limit, self.settings.candidate_pool_chunks)
            if reranking
            else chunk_limit,
            project=project,
            since=since_at,
            until=until_at,
            per_signal=self.settings.candidate_pool_per_signal if reranking else 0,
        )
        if self.reranker is not None:
            with stage("rerank"):
                candidates = self.reranker.rerank(query, candidates, limit=len(candidates))
        chunks = candidates[:chunk_limit]
        # Documents are read off the whole pool, not just what is returned. Otherwise asking for
        # more documents than chunks silently returns fewer.
        with stage("documents"):
            documents = self._documents_for_chunks(
                candidates,
                limit=document_limit,
                include_content=include_full_documents,
            )
        with stage("compress"):
            context_pack = self.compressor.pack(
                chunks,
                max_chars=context_limit,
            )
        RESULT_COUNT.labels(kind="chunks").observe(len(chunks))
        RESULT_COUNT.labels(kind="documents").observe(len(documents))
        return RetrievalResponse(
            query=query,
            chunks=chunks,
            documents=documents,
            context_pack=context_pack,
            index_profile=profile,
        )

    def search(
        self,
        query: str,
        limit: int = 10,
        project: str | None = None,
        since: str | datetime | None = None,
        until: str | datetime | None = None,
    ) -> SearchResponse:
        limit = min(max(limit, 1), 50)
        response = self.retrieve(
            query=query,
            top_k_chunks=max(limit * 2, 10),
            top_k_documents=limit,
            project=project,
            include_full_documents=False,
            max_context_chars=4000,
            since=since,
            until=until,
        )
        results = [
            SearchItem(
                id=document.id,
                title=document.title,
                url=f"tpo-automation-document-rag://document/{document.id}",
                text=self._best_snippet(document.id, response.chunks),
                score=document.score,
                metadata={
                    **document.metadata,
                    "source_path": document.source_path,
                    "effective_at": document.effective_at.isoformat(),
                },
            )
            for document in response.documents
        ]
        return SearchResponse(results=results)

    def fetch(self, item_id: str) -> FetchResponse:
        clean_id = item_id.rsplit("/", 1)[-1]
        with self.db.connection() as conn:
            document = conn.execute(
                """
                SELECT id, title, source_path, content, effective_at, source_modified_at, metadata
                FROM documents WHERE id = %s AND is_active = TRUE
                """,
                (clean_id,),
            ).fetchone()
            if document:
                return FetchResponse(
                    id=str(document["id"]),
                    title=document["title"],
                    text=document["content"],
                    url=f"tpo-automation-document-rag://document/{document['id']}",
                    metadata={
                        **dict(document["metadata"]),
                        "source_path": document["source_path"],
                        "effective_at": document["effective_at"].isoformat(),
                        "source_modified_at": document["source_modified_at"].isoformat(),
                    },
                )

            chunk = conn.execute(
                """
                SELECT c.id, c.content, c.chunk_index, d.id AS document_id, d.title, d.source_path,
                       d.effective_at, d.metadata
                FROM chunks c JOIN documents d ON d.id = c.document_id
                WHERE c.id = %s AND d.is_active = TRUE
                """,
                (clean_id,),
            ).fetchone()
            if chunk:
                return FetchResponse(
                    id=str(chunk["id"]),
                    title=f"{chunk['title']} — chunk {chunk['chunk_index']}",
                    text=chunk["content"],
                    url=f"tpo-automation-document-rag://chunk/{chunk['id']}",
                    metadata={
                        **dict(chunk["metadata"]),
                        "document_id": str(chunk["document_id"]),
                        "source_path": chunk["source_path"],
                        "effective_at": chunk["effective_at"].isoformat(),
                    },
                )
        raise KeyError(f"No active document or chunk found for id: {item_id}")

    def list_documents(
        self,
        limit: int = 100,
        project: str | None = None,
        since: str | datetime | None = None,
        until: str | datetime | None = None,
    ) -> list[DocumentResult]:
        limit = min(max(limit, 1), 500)
        params: list[Any] = []
        project_clause = ""
        if project:
            project_clause = "AND metadata->>'project' = %s"
            params.append(project)
        date_clause = ""
        since_at = parse_boundary(since, "since")
        until_at = parse_boundary(until, "until")
        if since_at:
            date_clause += " AND effective_at >= %s"
            params.append(since_at)
        if until_at:
            date_clause += " AND effective_at <= %s"
            params.append(until_at)
        params.append(limit)
        with self.db.connection() as conn:
            rows = conn.execute(
                f"""
                SELECT id, title, source_path, effective_at, source_modified_at, metadata
                FROM documents
                WHERE is_active = TRUE {project_clause}{date_clause}
                ORDER BY effective_at DESC
                LIMIT %s
                """,
                params,
            ).fetchall()
        return [
            DocumentResult(
                id=str(row["id"]),
                title=row["title"],
                source_path=row["source_path"],
                effective_at=row["effective_at"],
                source_modified_at=row["source_modified_at"],
                metadata=dict(row["metadata"]),
            )
            for row in rows
        ]

    def status(self) -> dict[str, Any]:
        profile = self.db.get_state(INDEX_STATE_KEY) or {"status": "not_initialized"}
        skipped = self.db.get_state(SKIPPED_STATE_KEY) or {}
        failed = self.db.get_state(FAILED_STATE_KEY) or {}
        with self.db.connection() as conn:
            counts = conn.execute(
                """
                SELECT
                  (SELECT count(*) FROM documents WHERE is_active = TRUE) AS documents,
                  (SELECT count(*) FROM chunks) AS chunks,
                  (SELECT max(updated_at) FROM documents WHERE is_active = TRUE) AS latest_document_update
                """
            ).fetchone()
        return {
            "status": profile.get("status", "unknown"),
            "profile": profile,
            "documents": counts["documents"],
            "chunks": counts["chunks"],
            "skipped_documents": skipped.get("count", 0),
            "failed_documents": failed.get("count", 0),
            "latest_document_update": counts["latest_document_update"].isoformat()
            if counts["latest_document_update"]
            else None,
        }

    def _ready_profile(self) -> dict[str, Any]:
        profile = self.db.get_state(INDEX_STATE_KEY)
        if not profile or profile.get("status") != "ready":
            raise RuntimeError(f"Knowledge index is not ready. Current state: {profile}")
        return profile

    def _search_chunks(
        self,
        query: str,
        profile_hash: str,
        limit: int,
        project: str | None,
        since: datetime | None = None,
        until: datetime | None = None,
        per_signal: int = 0,
    ) -> list[ChunkResult]:
        with stage("embed_query"):
            query_embedding = np.asarray(self.provider.embed_query(query), dtype=np.float32)
        params: dict[str, Any] = {
            "embedding": query_embedding,
            "query": query,
            "profile_hash": profile_hash,
            "limit": limit,
            "per_signal": per_signal,
            "semantic_weight": self.settings.semantic_weight,
            "lexical_weight": self.settings.lexical_weight,
            "recency_weight": self.settings.recency_weight,
            "rrf_k": self.settings.rrf_k,
            "terms": query_terms(query),
            "half_life": self.settings.recency_half_life_days,
            "min_semantic_score": self.settings.min_semantic_score,
            "scoring_pool": self.settings.scoring_pool_chunks,
            "project": project,
            "since": since,
            "until": until,
        }
        project_clause = "AND (%(project)s::text IS NULL OR d.metadata->>'project' = %(project)s::text)"
        date_clause = (
            "AND (%(since)s::timestamptz IS NULL OR d.effective_at >= %(since)s::timestamptz) "
            "AND (%(until)s::timestamptz IS NULL OR d.effective_at <= %(until)s::timestamptz)"
        )
        sql = f"""
            WITH query_input AS (
                SELECT
                    websearch_to_tsquery('simple', %(query)s) AS text_query,
                    lower(%(query)s) AS raw_query,
                    %(terms)s::text[] AS terms
            ),
            -- MATERIALIZED is load-bearing. Inlined, every scoring expression below is textually
            -- substituted into the sort key of each ranking window and evaluated again per sort.
            candidate AS MATERIALIZED (
                SELECT
                    c.id,
                    c.document_id,
                    d.effective_at,
                    GREATEST(0.0, 1.0 - (c.embedding <=> %(embedding)s)) AS semantic_score,
                    ts_rank_cd(
                        setweight(to_tsvector('simple', coalesce(d.title, '')), 'A') ||
                        setweight(to_tsvector('simple', coalesce(d.source_path, '')), 'B') ||
                        setweight(to_tsvector('simple', coalesce(d.metadata::text, '')), 'B') ||
                        setweight(c.search_vector, 'C'),
                        q.text_query,
                        32
                    ) * 6.0 AS ts_score,
                    LEAST(
                        1.0,
                        coalesce((
                            SELECT
                                0.45 * avg((position(t in lower(coalesce(d.title, ''))) > 0)::int)
                              + 0.35 * avg((position(t in lower(coalesce(d.source_path, ''))) > 0)::int)
                              + 0.30 * avg((position(t in lower(coalesce(d.metadata::text,''))) > 0)::int)
                              + 0.55 * avg((position(t in lower(coalesce(c.content, ''))) > 0)::int)
                            FROM unnest(q.terms) AS t
                        ), 0.0)
                    ) AS term_score,
                    GREATEST(
                        similarity(coalesce(d.title, ''), %(query)s),
                        similarity(coalesce(d.source_path, ''), %(query)s),
                        word_similarity(%(query)s, coalesce(d.title, '')),
                        word_similarity(%(query)s, coalesce(d.source_path, ''))
                    ) AS name_similarity,
                    exp(
                        -ln(2.0) * GREATEST(
                            0.0,
                            EXTRACT(EPOCH FROM (now() - d.effective_at)) / 86400.0
                        ) / %(half_life)s
                    ) AS recency_score
                FROM chunks c
                JOIN documents d ON d.id = c.document_id
                CROSS JOIN query_input q
                WHERE d.is_active = TRUE
                  AND c.embedding_profile_hash = %(profile_hash)s
                  {project_clause}
                  {date_clause}
            ),
            gated AS (
                SELECT *,
                    LEAST(1.0, ts_score + term_score + name_similarity * 0.45) AS cheap_lexical_score
                FROM candidate
                WHERE semantic_score >= %(min_semantic_score)s
            ),
            cheaply_ranked AS (
                SELECT *,
                    rank() OVER (ORDER BY semantic_score DESC) AS cheap_semantic_rank,
                    rank() OVER (ORDER BY cheap_lexical_score DESC) AS cheap_lexical_rank,
                    rank() OVER (ORDER BY recency_score DESC) AS cheap_recency_rank
                FROM gated
            ),
            shortlist AS MATERIALIZED (
                -- Comparing the question against a whole chunk of prose costs more than every
                -- other signal in this query put together. A passage none of the cheap signals
                -- ranked anywhere near the top cannot win on that term alone, so only the ones
                -- already in contention are read in full.
                SELECT
                    g.id,
                    g.document_id,
                    g.effective_at,
                    g.semantic_score,
                    g.recency_score,
                    LEAST(
                        1.0,
                        g.ts_score + g.term_score + GREATEST(
                            g.name_similarity,
                            word_similarity(%(query)s, coalesce(d.metadata::text, '')) * 0.75,
                            word_similarity(%(query)s, coalesce(c.content, '')) * 0.65
                        ) * 0.45
                    ) AS lexical_score
                FROM cheaply_ranked g
                JOIN chunks c ON c.id = g.id
                JOIN documents d ON d.id = g.document_id
                WHERE g.cheap_semantic_rank <= %(scoring_pool)s
                   OR g.cheap_lexical_rank <= %(scoring_pool)s
                   OR g.cheap_recency_rank <= %(scoring_pool)s
            ),
            ranked AS (
                SELECT *,
                    rank() OVER (ORDER BY semantic_score DESC) AS semantic_rank,
                    rank() OVER (ORDER BY lexical_score DESC) AS lexical_rank,
                    rank() OVER (ORDER BY recency_score DESC) AS recency_rank
                FROM shortlist
            ),
            fused AS (
                SELECT *,
                    (%(semantic_weight)s / (%(rrf_k)s + semantic_rank)) +
                    (%(lexical_weight)s / (%(rrf_k)s + lexical_rank)) +
                    (%(recency_weight)s / (%(rrf_k)s + recency_rank)) AS score
                FROM ranked
            ),
            pooled AS (
                SELECT *, rank() OVER (ORDER BY score DESC, effective_at DESC) AS fused_rank
                FROM fused
            )
            -- The text is fetched only for the handful of rows that survived, so none of the
            -- ranking above has to carry it through a sort.
            SELECT
                p.id,
                p.document_id,
                d.title AS document_title,
                d.source_path,
                c.chunk_index,
                c.content,
                c.start_char,
                c.end_char,
                p.effective_at,
                d.metadata,
                p.semantic_score,
                p.lexical_score,
                p.recency_score,
                p.score
            FROM pooled p
            JOIN chunks c ON c.id = p.id
            JOIN documents d ON d.id = p.document_id
            WHERE p.fused_rank <= %(limit)s
               OR p.semantic_rank <= %(per_signal)s
               OR p.lexical_rank <= %(per_signal)s
            ORDER BY p.score DESC, p.effective_at DESC
        """
        with self.db.connection() as conn, stage("search_sql"):
            rows = conn.execute(sql, params).fetchall()
        return [
            ChunkResult(
                id=str(row["id"]),
                document_id=str(row["document_id"]),
                document_title=row["document_title"],
                source_path=row["source_path"],
                chunk_index=row["chunk_index"],
                content=row["content"],
                start_char=row["start_char"],
                end_char=row["end_char"],
                effective_at=row["effective_at"],
                metadata=dict(row["metadata"]),
                semantic_score=float(row["semantic_score"]),
                lexical_score=float(row["lexical_score"]),
                recency_score=float(row["recency_score"]),
                score=float(row["score"]),
            )
            for row in rows
        ]

    def _documents_for_chunks(
        self, chunks: list[ChunkResult], limit: int, include_content: bool
    ) -> list[DocumentResult]:
        grouped: OrderedDict[str, list[ChunkResult]] = OrderedDict()
        for chunk in chunks:
            grouped.setdefault(chunk.document_id, []).append(chunk)
        selected_ids = list(grouped.keys())[:limit]
        if not selected_ids:
            return []

        with self.db.connection() as conn:
            rows = conn.execute(
                """
                SELECT id, title, source_path, content, effective_at, source_modified_at, metadata
                FROM documents WHERE id::text = ANY(%s) AND is_active = TRUE
                """,
                (selected_ids,),
            ).fetchall()
        row_map = {str(row["id"]): row for row in rows}
        results: list[DocumentResult] = []
        for document_id in selected_ids:
            row = row_map[document_id]
            content = row["content"] if include_content else None
            truncated = False
            if content and len(content) > self.settings.max_full_document_chars:
                content = content[: self.settings.max_full_document_chars]
                truncated = True
            matches = grouped[document_id]
            results.append(
                DocumentResult(
                    id=document_id,
                    title=row["title"],
                    source_path=row["source_path"],
                    content=content,
                    effective_at=row["effective_at"],
                    source_modified_at=row["source_modified_at"],
                    metadata=dict(row["metadata"]),
                    score=max(chunk.score for chunk in matches),
                    matched_chunk_ids=[chunk.id for chunk in matches],
                    truncated=truncated,
                )
            )
        return results

    @staticmethod
    def _best_snippet(document_id: str, chunks: list[ChunkResult]) -> str:
        for chunk in chunks:
            if chunk.document_id == document_id:
                return chunk.content[:1000]
        return ""
