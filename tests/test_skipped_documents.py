import logging
from pathlib import Path
from typing import Any

from product_memory.ingestion.extractors import UnreadableDocumentError
from product_memory.ingestion.parser import EmptyDocumentError
from product_memory.ingestion.service import (
    SKIPPED_STATE_KEY,
    IngestionService,
    _format_report,
    _skip_reason,
)
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


def test_a_report_is_rendered_as_an_indented_yaml_block() -> None:
    rendered = _format_report({"scan": {"added": 1}, "index": {"documents": 2, "size": "9 MB"}})

    assert rendered == "\n".join(
        [
            "  scan:",
            "    added: 1",
            "  index:",
            "    documents: 2",
            "    size: 9 MB",
        ]
    )
