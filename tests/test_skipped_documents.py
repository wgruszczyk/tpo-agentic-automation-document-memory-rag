import logging
from pathlib import Path
from typing import Any

from product_memory.ingestion.extractors import UnreadableDocumentError
from product_memory.ingestion.parser import EmptyDocumentError
from product_memory.ingestion.service import SKIPPED_STATE_KEY, IngestionService, _skip_reason
from product_memory.runtime import _SuppressHealthProbes
from product_memory.settings import Settings


class StateDatabase:
    def __init__(self) -> None:
        self.state: dict[str, dict[str, Any]] = {}
        self.writes = 0

    def get_state(self, key: str) -> dict[str, Any] | None:
        return self.state.get(key)

    def set_state(self, key: str, value: dict[str, Any]) -> None:
        self.state[key] = value
        self.writes += 1


def _service(db: StateDatabase) -> IngestionService:
    return IngestionService(
        Settings(_env_file=None),
        db=db,  # type: ignore[arg-type]
        provider=None,  # type: ignore[arg-type]
        parser=None,  # type: ignore[arg-type]
        chunker=None,  # type: ignore[arg-type]
    )


def test_skip_reason_drops_the_path_that_is_already_recorded() -> None:
    path = Path("/knowledge/pictures/screenshot.png")

    empty = EmptyDocumentError(f"Document has no extractable text: {path}")
    unreadable = UnreadableDocumentError(f"{path}: password protected")

    assert _skip_reason(empty, path) == "Document has no extractable text"
    assert _skip_reason(unreadable, path) == "password protected"


def test_skipped_documents_are_stored_instead_of_logged_per_file(
    caplog: Any,
) -> None:
    db = StateDatabase()
    service = _service(db)

    with caplog.at_level(logging.INFO):
        service._record_skipped(  # noqa: SLF001
            {"b/photo.png": "Document has no extractable text", "a/empty.docx": "Document is empty"}
        )

    stored = db.get_state(SKIPPED_STATE_KEY)
    assert stored is not None
    assert stored["count"] == 2
    assert [entry["source_path"] for entry in stored["documents"]] == ["a/empty.docx", "b/photo.png"]
    assert "photo.png" not in caplog.text
    assert len(caplog.records) == 1


def test_an_unchanged_skip_list_is_neither_rewritten_nor_logged_again(caplog: Any) -> None:
    db = StateDatabase()
    service = _service(db)
    skipped = {"a/photo.png": "Document has no extractable text"}
    service._record_skipped(skipped)  # noqa: SLF001

    with caplog.at_level(logging.INFO):
        service._record_skipped(skipped)  # noqa: SLF001

    assert db.writes == 1
    assert caplog.records == []


def test_skipped_documents_reports_an_empty_list_before_the_first_scan() -> None:
    assert _service(StateDatabase()).skipped_documents() == {"count": 0, "documents": []}


def test_health_probe_access_lines_are_dropped_and_others_are_kept() -> None:
    log_filter = _SuppressHealthProbes()

    def record(path: str) -> logging.LogRecord:
        return logging.LogRecord(
            name="uvicorn.access",
            level=logging.INFO,
            pathname=__file__,
            lineno=1,
            msg='%s - "%s %s HTTP/%s" %d',
            args=("127.0.0.1:50114", "GET", path, "1.1", 200),
            exc_info=None,
        )

    assert log_filter.filter(record("/health/live")) is False
    assert log_filter.filter(record("/health")) is False
    assert log_filter.filter(record("/mcp")) is True
