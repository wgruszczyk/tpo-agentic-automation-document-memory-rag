from datetime import UTC, datetime
from pathlib import Path

from product_memory.ingestion.parser import DocumentParser
from product_memory.settings import Settings


def test_frontmatter_and_effective_date(tmp_path: Path) -> None:
    path = tmp_path / "meeting.md"
    path.write_text(
        "---\ntitle: Payment meeting\nproject: checkout\neffective_at: 2026-07-31\n---\nBody text",
        encoding="utf-8",
    )
    parser = DocumentParser(Settings(knowledge_dir=tmp_path))
    parsed = parser.parse(path)
    assert parsed.title == "Payment meeting"
    assert parsed.metadata["project"] == "checkout"
    assert parsed.effective_at == datetime(2026, 7, 31, tzinfo=UTC)
    assert parsed.content == "Body text"


def test_date_from_filename(tmp_path: Path) -> None:
    path = tmp_path / "2026-06-15-refinement.txt"
    path.write_text("Useful knowledge", encoding="utf-8")
    parsed = DocumentParser(Settings(knowledge_dir=tmp_path)).parse(path)
    assert parsed.effective_at.date().isoformat() == "2026-06-15"
