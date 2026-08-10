from contextlib import contextmanager
from datetime import UTC, datetime
from types import SimpleNamespace
from uuid import uuid4

from product_memory.ingestion.chunker import TextChunk
from product_memory.ingestion.parser import ParsedDocument
from product_memory.ingestion.service import IngestionService
from product_memory.settings import Settings


class FakeCursor:
    def __init__(self) -> None:
        self.inserted_rows = []

    def __enter__(self) -> "FakeCursor":
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def executemany(self, _sql: str, rows: list[tuple]) -> None:
        self.inserted_rows.extend(rows)


class FakeConnection:
    def __init__(self) -> None:
        self.cursor_instance = FakeCursor()
        self.executed = []
        self.committed = False

    def execute(self, sql: str, params: tuple | None = None) -> None:
        self.executed.append((sql, params))

    def cursor(self) -> FakeCursor:
        return self.cursor_instance

    def commit(self) -> None:
        self.committed = True


class FakeDatabase:
    def __init__(self) -> None:
        self.connection_instance = FakeConnection()

    @contextmanager
    def connection(self):
        yield self.connection_instance


class FakeProvider:
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [[float(index), 0.0, 0.0] for index, _text in enumerate(texts)]


class FakeChunker:
    def split(self, _content: str) -> list[TextChunk]:
        return [
            TextChunk(index=0, content="first chunk", start_char=0, end_char=11, approx_tokens=3),
            TextChunk(index=1, content="second chunk", start_char=12, end_char=24, approx_tokens=3),
        ]


def test_index_document_uses_cursor_executemany() -> None:
    db = FakeDatabase()
    service = IngestionService(
        Settings(_env_file=None),
        db=db,  # type: ignore[arg-type]
        provider=FakeProvider(),  # type: ignore[arg-type]
        parser=None,  # type: ignore[arg-type]
        chunker=FakeChunker(),  # type: ignore[arg-type]
    )

    indexed = service._index_document(uuid4(), "Demo", "content", "profile")  # noqa: SLF001

    assert indexed == 2
    assert len(db.connection_instance.cursor_instance.inserted_rows) == 2
    assert db.connection_instance.committed


def test_deduplicate_documents_keeps_one_document_per_checksum() -> None:
    primary = parsed_document("a/source.md", "same")
    duplicate = parsed_document("b/source-copy.md", "same")
    unique = parsed_document("c/other.md", "different")

    documents, duplicates = IngestionService._deduplicate_documents([duplicate, unique, primary])  # noqa: SLF001

    assert duplicates == 1
    assert [document.source_path for document in documents] == ["a/source.md", "c/other.md"]
    assert documents[0].metadata["duplicate_source_paths"] == ["b/source-copy.md"]
    assert documents[0].metadata["duplicate_count"] == 1
    assert documents[0].metadata["all_source_paths"] == ["a/source.md", "b/source-copy.md"]


def test_deduplicate_documents_does_not_add_duplicate_metadata_for_unique_documents() -> None:
    document = parsed_document("source.md", "content")

    documents, duplicates = IngestionService._deduplicate_documents([document])  # noqa: SLF001

    assert duplicates == 0
    assert documents == [document]
    assert "duplicate_source_paths" not in documents[0].metadata


def parsed_document(source_path: str, content: str) -> ParsedDocument:
    now = datetime(2026, 8, 4, tzinfo=UTC)
    return ParsedDocument(
        source_path=source_path,
        title=source_path,
        content=content,
        content_hash=f"hash-{content}",
        source_modified_at=now,
        effective_at=now,
        metadata={"title": source_path, "extension": ".md"},
    )


class UpsertConnection(FakeConnection):
    def __init__(self, existing: dict) -> None:
        super().__init__()
        self.existing = existing

    def execute(self, sql: str, params: tuple | None = None):  # type: ignore[override]
        super().execute(sql, params)
        return SimpleNamespace(fetchone=lambda: self.existing)


class UpsertDatabase(FakeDatabase):
    def __init__(self, existing: dict) -> None:
        super().__init__()
        self.connection_instance = UpsertConnection(existing)


def upsert_service(existing: dict) -> tuple[IngestionService, list]:
    service = IngestionService(
        Settings(_env_file=None),
        db=UpsertDatabase(existing),  # type: ignore[arg-type]
        provider=FakeProvider(),  # type: ignore[arg-type]
        parser=None,  # type: ignore[arg-type]
        chunker=FakeChunker(),  # type: ignore[arg-type]
    )
    indexed: list = []
    service._index_document = lambda *args: indexed.append(args) or 0  # type: ignore[assignment] # noqa: SLF001
    return service, indexed


def stored_row(parsed: ParsedDocument, profile_hash: str) -> dict:
    return {
        "id": uuid4(),
        "title": parsed.title,
        "content_hash": parsed.content_hash,
        "effective_at": parsed.effective_at,
        "metadata": parsed.metadata,
        "indexed_profile_hash": profile_hash,
        "is_active": True,
    }


def test_upsert_document_leaves_untouched_sources_alone() -> None:
    parsed = parsed_document("source.md", "content")
    service, indexed = upsert_service(stored_row(parsed, "profile"))

    assert service._upsert_document(parsed, "profile") == "unchanged"  # noqa: SLF001
    assert indexed == []


def test_upsert_document_force_reruns_extraction_output_that_looks_unchanged() -> None:
    parsed = parsed_document("source.md", "content")
    service, indexed = upsert_service(stored_row(parsed, "profile"))

    assert service._upsert_document(parsed, "profile", force=True) == "updated"  # noqa: SLF001
    assert len(indexed) == 1
