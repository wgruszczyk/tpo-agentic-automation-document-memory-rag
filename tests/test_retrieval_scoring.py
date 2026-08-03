from __future__ import annotations

from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import UUID

from product_memory.embeddings.base import EmbeddingProvider
from product_memory.retrieval.service import Retriever
from product_memory.settings import Settings

ROOT = Path(__file__).resolve().parents[1]


class FakeProvider(EmbeddingProvider):
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [[1.0] for _ in texts]

    def embed_query(self, text: str) -> list[float]:
        return [1.0]

    def profile(self) -> dict[str, Any]:
        return {"provider": "fake", "model": "one", "dimension": 1}


class FakeResult:
    def __init__(self, rows: list[dict[str, Any]]):
        self.rows = rows

    def fetchall(self) -> list[dict[str, Any]]:
        return self.rows


class FakeConnection:
    def __init__(self) -> None:
        self.sql = ""
        self.params: dict[str, Any] = {}

    def execute(self, sql: str, params: dict[str, Any]) -> FakeResult:
        self.sql = sql
        self.params = params
        return FakeResult(
            [
                {
                    "id": UUID("11111111-1111-1111-1111-111111111111"),
                    "document_id": UUID("22222222-2222-2222-2222-222222222222"),
                    "document_title": "Payment Requirements",
                    "source_path": "requirements/payment.md",
                    "chunk_index": 0,
                    "content": "Payment retries stay in scope.",
                    "start_char": 0,
                    "end_char": 30,
                    "effective_at": datetime(2026, 8, 3, tzinfo=UTC),
                    "metadata": {"project": "checkout"},
                    "semantic_score": 1.0,
                    "lexical_score": 1.0,
                    "recency_score": 1.0,
                    "score": 1.0,
                }
            ]
        )


class FakeDatabase:
    def __init__(self) -> None:
        self.connection_instance = FakeConnection()

    @contextmanager
    def connection(self):
        yield self.connection_instance


def test_search_chunks_uses_stronger_lexical_signals() -> None:
    db = FakeDatabase()
    retriever = Retriever(
        settings=Settings(_env_file=None),
        db=db,  # type: ignore[arg-type]
        provider=FakeProvider(),
        compressor=None,  # type: ignore[arg-type]
    )

    chunks = retriever._search_chunks(  # noqa: SLF001
        query="payment retries",
        profile_hash="profile",
        limit=5,
        project="checkout",
    )

    sql = db.connection_instance.sql
    assert len(chunks) == 1
    assert db.connection_instance.params["query"] == "payment retries"
    assert "websearch_to_tsquery('simple', %(query)s)" in sql
    assert "setweight(to_tsvector('simple', coalesce(d.title, '')), 'A')" in sql
    assert "setweight(to_tsvector('simple', coalesce(d.source_path, '')), 'B')" in sql
    assert "setweight(to_tsvector('simple', coalesce(d.metadata::text, '')), 'B')" in sql
    assert "setweight(c.search_vector, 'C')" in sql
    assert "position(q.raw_query in lower(coalesce(c.content, '')))" in sql
    assert "similarity(coalesce(d.title, ''), %(query)s)" in sql
    assert "word_similarity(%(query)s, coalesce(c.content, ''))" in sql


def test_schema_enables_trigram_search_indexes() -> None:
    schema = (ROOT / "src" / "product_memory" / "schema.sql").read_text(encoding="utf-8")

    assert "CREATE EXTENSION IF NOT EXISTS pg_trgm;" in schema
    assert "documents_title_trgm_idx" in schema
    assert "documents_source_path_trgm_idx" in schema
    assert "chunks_content_trgm_idx" in schema
