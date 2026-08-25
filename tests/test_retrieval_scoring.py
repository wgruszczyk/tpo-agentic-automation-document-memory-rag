from __future__ import annotations

from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import UUID

import pytest

from product_memory.embeddings.base import EmbeddingProvider
from product_memory.models import ChunkResult
from product_memory.retrieval.service import Retriever, parse_boundary, query_terms
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
    assert "position(t in lower(coalesce(c.content, '')))" in sql
    assert "FROM unnest(q.terms) AS t" in sql
    assert db.connection_instance.params["terms"] == ["payment", "retries"]
    assert "similarity(coalesce(d.title, ''), %(query)s)" in sql
    assert "word_similarity(%(query)s, coalesce(c.content, ''))" in sql
    assert db.connection_instance.params["min_semantic_score"] == 0.60
    assert "WHERE semantic_score >= %(min_semantic_score)s" in sql


def test_reading_whole_chunks_is_confined_to_the_shortlist() -> None:
    db = FakeDatabase()
    retriever = Retriever(
        settings=Settings(_env_file=None),
        db=db,  # type: ignore[arg-type]
        provider=FakeProvider(),
        compressor=None,  # type: ignore[arg-type]
    )

    retriever._search_chunks(query="payment retries", profile_hash="profile", limit=5, project=None)  # noqa: SLF001

    sql = db.connection_instance.sql
    shortlist_start = sql.index("shortlist AS MATERIALIZED (")
    candidate_stage = sql[sql.index("candidate AS MATERIALIZED (") : sql.index("gated AS (")]
    shortlist_stage = sql[shortlist_start : sql.index("ranked AS (", shortlist_start)]

    # The scan over every chunk must not compare the question against chunk text or metadata.
    assert "word_similarity(%(query)s, coalesce(c.content" not in candidate_stage
    assert "coalesce(c.content, '')) * 0.65" not in candidate_stage
    assert "word_similarity(%(query)s, coalesce(d.metadata" not in candidate_stage
    assert "word_similarity(%(query)s, coalesce(c.content, ''))" in shortlist_stage
    assert "word_similarity(%(query)s, coalesce(d.metadata::text, ''))" in shortlist_stage
    assert db.connection_instance.params["scoring_pool"] == 400
    assert "cheap_semantic_rank <= %(scoring_pool)s" in sql
    assert "cheap_lexical_rank <= %(scoring_pool)s" in sql
    assert "cheap_recency_rank <= %(scoring_pool)s" in sql
    # Inlining these lets the planner re-evaluate every scoring expression once per ranking sort.
    assert "candidate AS MATERIALIZED (" in sql
    assert "shortlist AS MATERIALIZED (" in sql


def test_search_chunks_fuses_the_signals_by_rank_not_by_value() -> None:
    db = FakeDatabase()
    retriever = Retriever(
        settings=Settings(_env_file=None),
        db=db,  # type: ignore[arg-type]
        provider=FakeProvider(),
        compressor=None,  # type: ignore[arg-type]
    )

    retriever._search_chunks(  # noqa: SLF001
        query="payment retries", profile_hash="profile", limit=5, project=None
    )

    sql = db.connection_instance.sql
    assert "rank() OVER (ORDER BY semantic_score DESC) AS semantic_rank" in sql
    assert "rank() OVER (ORDER BY lexical_score DESC) AS lexical_rank" in sql
    assert "rank() OVER (ORDER BY recency_score DESC) AS recency_rank" in sql
    assert "%(semantic_weight)s / (%(rrf_k)s + semantic_rank)" in sql
    assert db.connection_instance.params["rrf_k"] == 150
    # Ranking the gated-out chunks would let them push the survivors down the list.
    assert sql.index("WHERE semantic_score >= %(min_semantic_score)s") < sql.index("AS score")
    assert "%(semantic_weight)s * semantic_score" not in sql


def test_query_terms_keeps_the_words_a_document_would_contain() -> None:
    assert query_terms("Who is Victor Aristondo, what is his role?") == ["victor", "aristondo", "role"]
    assert query_terms("Which vendor was chosen?") == ["vendor", "chosen"]
    assert query_terms("Aristondo") == ["aristondo"]


def test_query_terms_drops_duplicates_and_caps_the_list() -> None:
    assert query_terms("payment payment PAYMENT retries") == ["payment", "retries"]
    assert len(query_terms(" ".join(f"term{n}" for n in range(40)))) == 12


def test_query_terms_survives_a_query_made_only_of_grammar() -> None:
    assert query_terms("what is this?") == []
    assert query_terms("") == []


def test_search_chunks_filters_by_effective_date_window() -> None:
    db = FakeDatabase()
    retriever = Retriever(
        settings=Settings(_env_file=None),
        db=db,  # type: ignore[arg-type]
        provider=FakeProvider(),
        compressor=None,  # type: ignore[arg-type]
    )

    retriever._search_chunks(  # noqa: SLF001
        query="payment retries",
        profile_hash="profile",
        limit=5,
        project=None,
        since=datetime(2026, 1, 1, tzinfo=UTC),
        until=datetime(2026, 12, 31, tzinfo=UTC),
    )

    sql = db.connection_instance.sql
    assert "d.effective_at >= %(since)s::timestamptz" in sql
    assert "d.effective_at <= %(until)s::timestamptz" in sql
    assert db.connection_instance.params["since"] == datetime(2026, 1, 1, tzinfo=UTC)
    assert db.connection_instance.params["until"] == datetime(2026, 12, 31, tzinfo=UTC)


def test_parse_boundary_reads_iso_dates_and_assumes_utc() -> None:
    assert parse_boundary("2026-03-01", "since") == datetime(2026, 3, 1, tzinfo=UTC)
    assert parse_boundary("", "since") is None
    assert parse_boundary(None, "since") is None


def test_parse_boundary_rejects_nonsense() -> None:
    with pytest.raises(ValueError, match="not a recognisable date"):
        parse_boundary("whenever", "since")


def test_retrieve_rejects_an_inverted_date_window() -> None:
    retriever = CapturingRetriever(
        settings=Settings(_env_file=None),
        db=None,  # type: ignore[arg-type]
        provider=FakeProvider(),
        compressor=FakeCompressor(),  # type: ignore[arg-type]
    )

    with pytest.raises(ValueError, match="since must not be later than until"):
        retriever.retrieve("payment retries", since="2026-06-01", until="2026-01-01")


def test_retrieve_caps_requested_document_limit_to_configured_max() -> None:
    retriever = CapturingRetriever(
        settings=Settings(max_returned_documents=2, _env_file=None),
        db=None,  # type: ignore[arg-type]
        provider=FakeProvider(),
        compressor=FakeCompressor(),  # type: ignore[arg-type]
    )

    retriever.retrieve("payment retries", top_k_documents=10)

    assert retriever.document_limit == 2


def test_retrieve_uses_default_document_limit() -> None:
    retriever = CapturingRetriever(
        settings=Settings(_env_file=None),
        db=None,  # type: ignore[arg-type]
        provider=FakeProvider(),
        compressor=FakeCompressor(),  # type: ignore[arg-type]
    )

    retriever.retrieve("payment retries")

    assert retriever.document_limit == 7


def test_retrieve_allows_requested_document_limit_up_to_25() -> None:
    retriever = CapturingRetriever(
        settings=Settings(_env_file=None),
        db=None,  # type: ignore[arg-type]
        provider=FakeProvider(),
        compressor=FakeCompressor(),  # type: ignore[arg-type]
    )

    retriever.retrieve("payment retries", top_k_documents=25)

    assert retriever.document_limit == 25


def test_schema_enables_trigram_search_indexes() -> None:
    schema = (ROOT / "src" / "product_memory" / "schema.sql").read_text(encoding="utf-8")

    assert "CREATE EXTENSION IF NOT EXISTS pg_trgm;" in schema
    assert "documents_title_trgm_idx" in schema
    assert "documents_source_path_trgm_idx" in schema
    assert "chunks_content_trgm_idx" in schema


class CapturingRetriever(Retriever):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.document_limit: int | None = None
        self.chunk_limit: int | None = None
        self.per_signal: int | None = None

    def _ready_profile(self) -> dict[str, Any]:
        return {"status": "ready", "fingerprint": "profile"}

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
        self.chunk_limit = limit
        self.per_signal = per_signal
        return []

    def _documents_for_chunks(self, chunks: list[ChunkResult], limit: int, include_content: bool):
        self.document_limit = limit
        return []


class FakeCompressor:
    def pack(self, chunks: list[ChunkResult], max_chars: int) -> str:
        return ""


class StubReranker:
    """Orders by a caller-supplied key so a test can state the reranked order outright."""

    def __init__(self, order: list[str] | None = None) -> None:
        self.order = order or []
        self.seen: list[ChunkResult] = []
        self.query: str | None = None

    def rerank(self, query: str, chunks: list[ChunkResult], limit: int) -> list[ChunkResult]:
        self.query = query
        self.seen = list(chunks)
        ranked = sorted(
            chunks,
            key=lambda chunk: self.order.index(chunk.id) if chunk.id in self.order else len(self.order),
        )
        return ranked[:limit]


def _chunk(chunk_id: str, document_id: str) -> ChunkResult:
    return ChunkResult(
        id=chunk_id,
        document_id=document_id,
        document_title=f"doc {document_id}",
        source_path=f"/knowledge/{document_id}.md",
        chunk_index=0,
        content=f"content {chunk_id}",
        start_char=0,
        end_char=10,
        effective_at=datetime(2026, 1, 1, tzinfo=UTC),
        semantic_score=0.8,
        lexical_score=0.1,
        recency_score=0.9,
        score=0.5,
    )


class PoolRetriever(CapturingRetriever):
    """Returns a fixed candidate pool so the reranker's effect on the result is observable."""

    def __init__(self, pool: list[ChunkResult], *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.pool = pool
        self.documents_seen: list[ChunkResult] = []

    def _search_chunks(self, *args: Any, **kwargs: Any) -> list[ChunkResult]:
        super()._search_chunks(*args, **kwargs)
        return list(self.pool)

    def _documents_for_chunks(self, chunks: list[ChunkResult], limit: int, include_content: bool):
        self.documents_seen = list(chunks)
        return super()._documents_for_chunks(chunks, limit, include_content)


def test_retrieve_searches_deeper_than_it_returns_when_reranking() -> None:
    retriever = PoolRetriever(
        [],
        settings=Settings(candidate_pool_chunks=120, candidate_pool_per_signal=40, _env_file=None),
        db=None,  # type: ignore[arg-type]
        provider=FakeProvider(),
        compressor=FakeCompressor(),  # type: ignore[arg-type]
        reranker=StubReranker(),  # type: ignore[arg-type]
    )

    retriever.retrieve("payment retries", top_k_chunks=5)

    assert retriever.chunk_limit == 120
    assert retriever.per_signal == 40


def test_retrieve_keeps_the_pool_shallow_when_not_reranking() -> None:
    retriever = PoolRetriever(
        [],
        settings=Settings(candidate_pool_chunks=120, candidate_pool_per_signal=40, _env_file=None),
        db=None,  # type: ignore[arg-type]
        provider=FakeProvider(),
        compressor=FakeCompressor(),  # type: ignore[arg-type]
    )

    retriever.retrieve("payment retries", top_k_chunks=5)

    assert retriever.chunk_limit == 5
    # Widening the pool without reranking only buys noise, so the union stays switched off.
    assert retriever.per_signal == 0


def test_retrieve_returns_the_reranked_order_not_the_retrieved_one() -> None:
    pool = [_chunk("a", "doc-1"), _chunk("b", "doc-2"), _chunk("c", "doc-3")]
    retriever = PoolRetriever(
        pool,
        settings=Settings(_env_file=None),
        db=None,  # type: ignore[arg-type]
        provider=FakeProvider(),
        compressor=FakeCompressor(),  # type: ignore[arg-type]
        reranker=StubReranker(order=["c", "a", "b"]),  # type: ignore[arg-type]
    )

    response = retriever.retrieve("payment retries", top_k_chunks=2)

    assert [chunk.id for chunk in response.chunks] == ["c", "a"]


def test_retrieve_reads_documents_off_the_whole_pool_not_the_returned_chunks() -> None:
    pool = [_chunk("a", "doc-1"), _chunk("b", "doc-2"), _chunk("c", "doc-3")]
    retriever = PoolRetriever(
        pool,
        settings=Settings(_env_file=None),
        db=None,  # type: ignore[arg-type]
        provider=FakeProvider(),
        compressor=FakeCompressor(),  # type: ignore[arg-type]
        reranker=StubReranker(order=["a", "b", "c"]),  # type: ignore[arg-type]
    )

    retriever.retrieve("payment retries", top_k_chunks=1, top_k_documents=3)

    # Asking for more documents than chunks must not silently return fewer.
    assert [chunk.id for chunk in retriever.documents_seen] == ["a", "b", "c"]
