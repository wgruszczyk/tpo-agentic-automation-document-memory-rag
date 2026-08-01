from contextlib import contextmanager
from uuid import uuid4

from product_memory.ingestion.chunker import TextChunk
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
