from __future__ import annotations

import math
from collections import defaultdict
from typing import Any

from product_memory.db import Database
from product_memory.ingestion.service import INDEX_STATE_KEY


def inspect_documents(
    db: Database,
    *,
    active_only: bool = False,
    project: str | None = None,
    limit: int = 500,
    offset: int = 0,
    include_content: bool = False,
    include_chunks: bool = True,
    include_embeddings: bool = False,
    content_preview_chars: int = 500,
) -> dict[str, Any]:
    limit = min(max(limit, 1), 5000)
    offset = max(offset, 0)
    content_preview_chars = min(max(content_preview_chars, 0), 5000)

    where_sql, params = _document_filters(active_only=active_only, project=project)
    with db.connection() as conn:
        total = conn.execute(
            f"SELECT count(*) AS total FROM documents {where_sql}",
            params,
        ).fetchone()["total"]
        document_rows = conn.execute(
            f"""
            SELECT id, title, source_path, content, content_hash, source_modified_at,
                   effective_at, metadata, indexed_profile_hash, is_active, created_at, updated_at
            FROM documents
            {where_sql}
            ORDER BY is_active DESC, effective_at DESC, source_path
            LIMIT %s OFFSET %s
            """,
            [*params, limit, offset],
        ).fetchall()

        chunk_rows = []
        document_ids = [str(row["id"]) for row in document_rows]
        if include_chunks and document_ids:
            chunk_rows = conn.execute(
                """
                SELECT id, document_id, chunk_index, content, start_char, end_char, approx_tokens,
                       embedding, embedding_profile_hash, created_at
                FROM chunks
                WHERE document_id::text = ANY(%s)
                ORDER BY document_id, chunk_index
                """,
                (document_ids,),
            ).fetchall()

    chunks_by_document: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in chunk_rows:
        chunks_by_document[str(row["document_id"])].append(
            _serialize_chunk(
                row,
                include_content=include_content,
                include_embeddings=include_embeddings,
                content_preview_chars=content_preview_chars,
            )
        )

    documents = [
        _serialize_document(
            row,
            chunks=chunks_by_document.get(str(row["id"]), []),
            include_content=include_content,
            include_chunks=include_chunks,
            content_preview_chars=content_preview_chars,
        )
        for row in document_rows
    ]

    return {
        "total": total,
        "limit": limit,
        "offset": offset,
        "active_only": active_only,
        "project": project,
        "include_content": include_content,
        "include_chunks": include_chunks,
        "include_embeddings": include_embeddings,
        "content_preview_chars": content_preview_chars,
        "index_profile": db.get_state(INDEX_STATE_KEY),
        "documents": documents,
    }


def _document_filters(*, active_only: bool, project: str | None) -> tuple[str, list[Any]]:
    clauses = []
    params: list[Any] = []
    if active_only:
        clauses.append("is_active = TRUE")
    if project:
        clauses.append("metadata->>'project' = %s")
        params.append(project)
    if not clauses:
        return "", params
    return "WHERE " + " AND ".join(clauses), params


def _serialize_document(
    row: dict[str, Any],
    *,
    chunks: list[dict[str, Any]],
    include_content: bool,
    include_chunks: bool,
    content_preview_chars: int,
) -> dict[str, Any]:
    content = row["content"] or ""
    document = {
        "id": str(row["id"]),
        "title": row["title"],
        "source_path": row["source_path"],
        "content_hash": row["content_hash"],
        "content_length": len(content),
        "source_modified_at": _json_value(row["source_modified_at"]),
        "effective_at": _json_value(row["effective_at"]),
        "metadata": dict(row["metadata"]),
        "indexed_profile_hash": row["indexed_profile_hash"],
        "is_active": row["is_active"],
        "created_at": _json_value(row["created_at"]),
        "updated_at": _json_value(row["updated_at"]),
        "chunk_count": len(chunks),
        "embedded_chunk_count": sum(1 for chunk in chunks if chunk["embedding"]["present"]),
    }
    if include_content:
        document["content"] = content
    else:
        document["content_preview"] = content[:content_preview_chars]
    if include_chunks:
        document["chunks"] = chunks
    return document


def _serialize_chunk(
    row: dict[str, Any],
    *,
    include_content: bool,
    include_embeddings: bool,
    content_preview_chars: int,
) -> dict[str, Any]:
    content = row["content"] or ""
    chunk = {
        "id": str(row["id"]),
        "document_id": str(row["document_id"]),
        "chunk_index": row["chunk_index"],
        "start_char": row["start_char"],
        "end_char": row["end_char"],
        "approx_tokens": row["approx_tokens"],
        "content_length": len(content),
        "embedding_profile_hash": row["embedding_profile_hash"],
        "embedding": _embedding_summary(row["embedding"], include_embeddings=include_embeddings),
        "created_at": _json_value(row["created_at"]),
    }
    if include_content:
        chunk["content"] = content
    else:
        chunk["content_preview"] = content[:content_preview_chars]
    return chunk


def _embedding_summary(value: Any, *, include_embeddings: bool) -> dict[str, Any]:
    vector = _vector_values(value)
    summary: dict[str, Any] = {
        "present": bool(vector),
        "dimensions": len(vector),
        "norm": math.sqrt(sum(item * item for item in vector)) if vector else None,
        "preview": vector[:8],
    }
    if include_embeddings:
        summary["values"] = vector
    return summary


def _vector_values(value: Any) -> list[float]:
    if value is None:
        return []
    if hasattr(value, "to_list"):
        return [float(item) for item in value.to_list()]
    if hasattr(value, "tolist"):
        return [float(item) for item in value.tolist()]
    if isinstance(value, str):
        return [float(item) for item in value.strip("[]").split(",") if item]
    return [float(item) for item in value]


def _json_value(value: Any) -> Any:
    if hasattr(value, "isoformat"):
        return value.isoformat()
    return value
