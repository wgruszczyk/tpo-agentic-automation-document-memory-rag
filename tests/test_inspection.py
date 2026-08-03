from __future__ import annotations

from contextlib import contextmanager
from datetime import UTC, datetime
from uuid import UUID

from pgvector import Vector

from product_memory.inspection import inspect_documents

DOCUMENT_ID = UUID("11111111-1111-1111-1111-111111111111")
CHUNK_ID = UUID("22222222-2222-2222-2222-222222222222")
NOW = datetime(2026, 8, 3, 12, 0, tzinfo=UTC)


class FakeResult:
    def __init__(self, rows: list[dict]):
        self.rows = rows

    def fetchone(self) -> dict:
        return self.rows[0]

    def fetchall(self) -> list[dict]:
        return self.rows


class FakeConnection:
    def execute(self, sql: str, params: object | None = None) -> FakeResult:
        if "count(*) AS total" in sql:
            return FakeResult([{"total": 1}])
        if "FROM documents" in sql:
            return FakeResult(
                [
                    {
                        "id": DOCUMENT_ID,
                        "title": "Checkout Requirements",
                        "source_path": "requirements/checkout.docx",
                        "content": "Checkout must support saved cards.",
                        "content_hash": "abc123",
                        "source_modified_at": NOW,
                        "effective_at": NOW,
                        "metadata": {"project": "checkout"},
                        "indexed_profile_hash": "profile",
                        "is_active": True,
                        "created_at": NOW,
                        "updated_at": NOW,
                    }
                ]
            )
        if "FROM chunks" in sql:
            return FakeResult(
                [
                    {
                        "id": CHUNK_ID,
                        "document_id": DOCUMENT_ID,
                        "chunk_index": 0,
                        "content": "Checkout must support saved cards.",
                        "start_char": 0,
                        "end_char": 34,
                        "approx_tokens": 6,
                        "embedding": [3.0, 4.0, 0.0],
                        "embedding_profile_hash": "profile",
                        "created_at": NOW,
                    }
                ]
            )
        raise AssertionError(f"Unexpected SQL: {sql}")


class FakeDatabase:
    @contextmanager
    def connection(self):
        yield FakeConnection()

    def get_state(self, key: str) -> dict:
        return {"status": "ready", "fingerprint": key}


def test_inspection_returns_embedding_summary_without_full_vector() -> None:
    result = inspect_documents(FakeDatabase())  # type: ignore[arg-type]

    document = result["documents"][0]
    chunk = document["chunks"][0]

    assert result["total"] == 1
    assert document["chunk_count"] == 1
    assert document["embedded_chunk_count"] == 1
    assert document["content_preview"] == "Checkout must support saved cards."
    assert "content" not in document
    assert chunk["embedding"] == {
        "present": True,
        "dimensions": 3,
        "norm": 5.0,
        "preview": [3.0, 4.0, 0.0],
    }


def test_inspection_can_include_full_content_and_embeddings() -> None:
    result = inspect_documents(
        FakeDatabase(),  # type: ignore[arg-type]
        include_content=True,
        include_embeddings=True,
    )

    document = result["documents"][0]
    chunk = document["chunks"][0]

    assert document["content"] == "Checkout must support saved cards."
    assert "content_preview" not in document
    assert chunk["content"] == "Checkout must support saved cards."
    assert chunk["embedding"]["values"] == [3.0, 4.0, 0.0]


def test_inspection_handles_pgvector_vector_values() -> None:
    result = inspect_documents(VectorDatabase())  # type: ignore[arg-type]

    chunk = result["documents"][0]["chunks"][0]

    assert chunk["embedding"]["dimensions"] == 3
    assert chunk["embedding"]["preview"] == [3.0, 4.0, 0.0]


class VectorConnection(FakeConnection):
    def execute(self, sql: str, params: object | None = None) -> FakeResult:
        result = super().execute(sql, params)
        if "FROM chunks" in sql:
            result.rows[0]["embedding"] = Vector([3.0, 4.0, 0.0])
        return result


class VectorDatabase(FakeDatabase):
    @contextmanager
    def connection(self):
        yield VectorConnection()
