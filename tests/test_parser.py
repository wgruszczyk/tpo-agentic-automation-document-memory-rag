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


def test_infers_reliable_metadata_from_teams_webvtt(tmp_path: Path) -> None:
    path = tmp_path / "teams-checkout-refinement.vtt"
    path.write_text(
        """WEBVTT

NOTE language:en-US
NOTE duration:"00:12:05.5000000"

Meeting title: Checkout refinement
Date: July 31, 2026 10:00 AM

00:00:00.000 --> 00:00:03.500
<v Alice Brown>We agreed to keep payment retries in the MVP.</v>

00:00:03.500 --> 00:00:05.000
<v Bob Smith>I will update the acceptance criteria.</v>
""",
        encoding="utf-8",
    )

    parsed = DocumentParser(Settings(knowledge_dir=tmp_path)).parse(path)

    assert parsed.title == "Checkout refinement"
    assert parsed.effective_at == datetime(2026, 7, 31, 10, 0, tzinfo=UTC)
    assert parsed.metadata["source_type"] == "meeting_transcript"
    assert parsed.metadata["source_app"] == "microsoft_teams"
    assert parsed.metadata["transcript_format"] == "webvtt"
    assert parsed.metadata["language"] == "en-US"
    assert parsed.metadata["duration_seconds"] == 725.5
    assert parsed.metadata["speakers"] == ["Alice Brown", "Bob Smith"]
    assert parsed.metadata["speaker_count"] == 2


def test_frontmatter_overrides_inferred_transcript_metadata(tmp_path: Path) -> None:
    path = tmp_path / "teams-checkout-refinement.vtt"
    path.write_text(
        """---
title: Explicit title
effective_at: 2026-08-01
project: checkout
---
WEBVTT

Meeting title: Inferred title
Date: July 31, 2026

00:00:00.000 --> 00:00:03.500
<v Alice Brown>We agreed to keep payment retries in the MVP.</v>
""",
        encoding="utf-8",
    )

    parsed = DocumentParser(Settings(knowledge_dir=tmp_path)).parse(path)

    assert parsed.title == "Explicit title"
    assert parsed.effective_at == datetime(2026, 8, 1, tzinfo=UTC)
    assert parsed.metadata["project"] == "checkout"
    assert parsed.metadata["speakers"] == ["Alice Brown"]


def test_ignores_labeled_metadata_inside_transcript_body(tmp_path: Path) -> None:
    path = tmp_path / "2026-06-15-teams-notes.vtt"
    path.write_text(
        """WEBVTT

00:00:00.000 --> 00:00:03.500
<v Alice Brown>Title: this is just something somebody said.</v>

00:00:03.500 --> 00:00:05.000
<v Bob Smith>Date: July 31, 2026 is not the meeting date.</v>
""",
        encoding="utf-8",
    )

    parsed = DocumentParser(Settings(knowledge_dir=tmp_path)).parse(path)

    assert parsed.title == "2026 06 15 teams notes"
    assert parsed.effective_at == datetime(2026, 6, 15, tzinfo=UTC)
    assert parsed.metadata["speakers"] == ["Alice Brown", "Bob Smith"]
