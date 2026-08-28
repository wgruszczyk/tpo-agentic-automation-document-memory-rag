from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

from product_memory.ingestion.chunker import TextChunk
from product_memory.ingestion.parser import ParsedDocument
from product_memory.ingestion.service import (
    DuplicateIndex,
    IngestionService,
    _without_derived,
    reading_order,
)
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


def test_documents_are_read_before_recordings() -> None:
    # A scan that met files in name order would leave every document waiting behind hours of audio.
    paths = [
        Path("/k/a-recording.mp4"),
        Path("/k/b-notes.md"),
        Path("/k/c-deck.pptx"),
        Path("/k/d-call.m4a"),
    ]

    assert [path.name for path in reading_order(paths)] == [
        "b-notes.md",
        "c-deck.pptx",
        "a-recording.mp4",
        "d-call.m4a",
    ]


def test_the_first_path_holding_some_content_is_the_one_that_is_kept() -> None:
    seen = DuplicateIndex()

    assert seen.duplicate_of("hash-a", "a/source.md") is None
    assert seen.duplicate_of("hash-a", "b/source-copy.md") == "a/source.md"
    assert seen.duplicate_of("hash-b", "c/other.md") is None

    assert seen.count == 1
    assert seen.duplicates_by_canonical == {"a/source.md": ["b/source-copy.md"]}


def test_content_seen_once_records_no_duplicates() -> None:
    seen = DuplicateIndex()

    assert seen.duplicate_of("hash-a", "source.md") is None

    assert seen.count == 0
    assert seen.duplicates_by_canonical == {}


def test_scan_bookkeeping_does_not_count_as_the_document_changing() -> None:
    # These keys are written after the folder has been read, so a document whose only difference
    # is one of them must not be re-embedded on the next scan.
    stored = {"title": "Deck", "duplicate_count": 1, "all_source_paths": ["a", "b"]}
    parsed = {"title": "Deck"}

    assert _without_derived(stored) == _without_derived(parsed)


def test_a_real_metadata_difference_is_still_a_change() -> None:
    assert _without_derived({"title": "Deck"}) != _without_derived({"title": "Notes"})


class StoredDocumentConnection(FakeConnection):
    """Answers the 'has this document changed' lookup with a row already carrying duplicate keys."""

    def __init__(self, stored_metadata: dict) -> None:
        super().__init__()
        self.stored_metadata = stored_metadata

    def execute(self, sql: str, params: tuple | None = None):
        super().execute(sql, params)
        if "FROM documents WHERE source_path" in sql:
            return SimpleNamespace(
                fetchone=lambda: {
                    "id": uuid4(),
                    "title": "Deck",
                    "content_hash": "hash-a",
                    "effective_at": datetime(2026, 1, 1, tzinfo=UTC),
                    "metadata": self.stored_metadata,
                    "indexed_profile_hash": "profile",
                    "is_active": True,
                }
            )
        return SimpleNamespace(fetchone=lambda: None, fetchall=lambda: [])


def _service_with(connection: FakeConnection) -> IngestionService:
    database = FakeDatabase()
    database.connection_instance = connection
    return IngestionService(
        Settings(_env_file=None),
        db=database,  # type: ignore[arg-type]
        provider=FakeProvider(),  # type: ignore[arg-type]
        parser=None,  # type: ignore[arg-type]
        chunker=FakeChunker(),  # type: ignore[arg-type]
    )


def test_a_document_is_not_rewritten_just_because_it_has_duplicates() -> None:
    # The duplicate keys are added after the document is written, so comparing them against a
    # freshly parsed document would report a change on every single scan, re-embedding forever.
    connection = StoredDocumentConnection(
        {"title": "Deck", "duplicate_source_paths": ["b/copy.md"], "duplicate_count": 1}
    )
    parsed = ParsedDocument(
        source_path="a/source.md",
        title="Deck",
        content="content",
        content_hash="hash-a",
        source_modified_at=datetime(2026, 1, 1, tzinfo=UTC),
        effective_at=datetime(2026, 1, 1, tzinfo=UTC),
        metadata={"title": "Deck"},
    )

    outcome = _service_with(connection)._upsert_document(parsed, "profile")  # noqa: SLF001

    assert outcome == "unchanged"


def test_a_document_whose_content_really_changed_is_rewritten() -> None:
    connection = StoredDocumentConnection({"title": "Deck"})
    parsed = ParsedDocument(
        source_path="a/source.md",
        title="Deck",
        content="new content",
        content_hash="hash-b",
        source_modified_at=datetime(2026, 1, 1, tzinfo=UTC),
        effective_at=datetime(2026, 1, 1, tzinfo=UTC),
        metadata={"title": "Deck"},
    )

    outcome = _service_with(connection)._upsert_document(parsed, "profile")  # noqa: SLF001

    assert outcome == "updated"


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
