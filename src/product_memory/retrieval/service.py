from __future__ import annotations

from collections import OrderedDict
from typing import Any

import numpy as np

from product_memory.db import Database
from product_memory.embeddings.base import EmbeddingProvider
from product_memory.ingestion.service import INDEX_STATE_KEY
from product_memory.models import (
    ChunkResult,
    DocumentResult,
    FetchResponse,
    RetrievalResponse,
    SearchItem,
    SearchResponse,
)
from product_memory.retrieval.compressor import ContextCompressor
from product_memory.settings import Settings


class Retriever:
    def __init__(
        self,
        settings: Settings,
        db: Database,
        provider: EmbeddingProvider,
        compressor: ContextCompressor,
    ):
        self.settings = settings
        self.db = db
        self.provider = provider
        self.compressor = compressor

    def retrieve(
        self,
        query: str,
        top_k_chunks: int | None = None,
        top_k_documents: int | None = None,
        project: str | None = None,
        include_full_documents: bool = True,
        max_context_chars: int | None = None,
    ) -> RetrievalResponse:
        query = query.strip()
        if not query:
            raise ValueError("query cannot be empty")
        profile = self._ready_profile()
        chunk_limit = min(max(top_k_chunks or self.settings.default_top_k_chunks, 1), 50)
        document_limit = min(max(top_k_documents or self.settings.default_top_k_documents, 1), 20)
        context_limit = min(
            max(max_context_chars or self.settings.default_context_chars, 2000),
            250000,
        )
        chunks = self._search_chunks(
            query=query,
            profile_hash=profile["fingerprint"],
            limit=chunk_limit,
            project=project,
        )
        documents = self._documents_for_chunks(
            chunks,
            limit=document_limit,
            include_content=include_full_documents,
        )
        context_pack = self.compressor.pack(
            chunks,
            max_chars=context_limit,
        )
        return RetrievalResponse(
            query=query,
            chunks=chunks,
            documents=documents,
            context_pack=context_pack,
            index_profile=profile,
        )

    def search(self, query: str, limit: int = 10, project: str | None = None) -> SearchResponse:
        limit = min(max(limit, 1), 50)
        response = self.retrieve(
            query=query,
            top_k_chunks=max(limit * 2, 10),
            top_k_documents=limit,
            project=project,
            include_full_documents=False,
            max_context_chars=4000,
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

    def list_documents(self, limit: int = 100, project: str | None = None) -> list[DocumentResult]:
        limit = min(max(limit, 1), 500)
        params: list[Any] = []
        project_clause = ""
        if project:
            project_clause = "AND metadata->>'project' = %s"
            params.append(project)
        params.append(limit)
        with self.db.connection() as conn:
            rows = conn.execute(
                f"""
                SELECT id, title, source_path, effective_at, source_modified_at, metadata
                FROM documents
                WHERE is_active = TRUE {project_clause}
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
        self, query: str, profile_hash: str, limit: int, project: str | None
    ) -> list[ChunkResult]:
        query_embedding = np.asarray(self.provider.embed_query(query), dtype=np.float32)
        params: dict[str, Any] = {
            "embedding": query_embedding,
            "query": query,
            "profile_hash": profile_hash,
            "limit": limit,
            "semantic_weight": self.settings.semantic_weight,
            "lexical_weight": self.settings.lexical_weight,
            "recency_weight": self.settings.recency_weight,
            "half_life": self.settings.recency_half_life_days,
            "project": project,
        }
        project_clause = "AND (%(project)s IS NULL OR d.metadata->>'project' = %(project)s)"
        sql = f"""
            WITH scored AS (
                SELECT
                    c.id,
                    c.document_id,
                    d.title AS document_title,
                    d.source_path,
                    c.chunk_index,
                    c.content,
                    c.start_char,
                    c.end_char,
                    d.effective_at,
                    d.metadata,
                    GREATEST(0.0, 1.0 - (c.embedding <=> %(embedding)s)) AS semantic_score,
                    LEAST(1.0, ts_rank_cd(c.search_vector, plainto_tsquery('simple', %(query)s)) * 4.0)
                        AS lexical_score,
                    exp(
                        -ln(2.0) * GREATEST(
                            0.0,
                            EXTRACT(EPOCH FROM (now() - d.effective_at)) / 86400.0
                        ) / %(half_life)s
                    ) AS recency_score
                FROM chunks c
                JOIN documents d ON d.id = c.document_id
                WHERE d.is_active = TRUE
                  AND c.embedding_profile_hash = %(profile_hash)s
                  {project_clause}
            )
            SELECT *,
                (%(semantic_weight)s * semantic_score) +
                (%(lexical_weight)s * lexical_score) +
                (%(recency_weight)s * recency_score) AS score
            FROM scored
            ORDER BY score DESC, effective_at DESC
            LIMIT %(limit)s
        """
        with self.db.connection() as conn:
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
