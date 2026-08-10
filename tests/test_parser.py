from datetime import UTC, datetime
from pathlib import Path

import extract_msg
from docx import Document as DocxDocument
from pptx import Presentation
from pptx.util import Inches

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


def test_source_path_stays_relative_for_symlinked_document(tmp_path: Path) -> None:
    knowledge_dir = tmp_path / "knowledge"
    external_dir = tmp_path / "teams-source"
    external_dir.mkdir(parents=True)
    knowledge_dir.mkdir()
    path = external_dir / "meeting.md"
    path.write_text("Useful knowledge", encoding="utf-8")
    (knowledge_dir / "teams").symlink_to(external_dir, target_is_directory=True)

    parsed = DocumentParser(Settings(knowledge_dir=knowledge_dir)).parse(
        knowledge_dir / "teams" / "meeting.md"
    )

    assert parsed.source_path == "teams/meeting.md"


def test_extracts_docx_content_and_properties(tmp_path: Path) -> None:
    path = tmp_path / "checkout-requirements.docx"
    document = DocxDocument()
    document.core_properties.title = "Checkout Requirements"
    document.core_properties.author = "Product Team"
    document.add_heading("Checkout Requirements", level=1)
    document.add_paragraph("The checkout flow must support saved cards.")
    table = document.add_table(rows=1, cols=2)
    table.rows[0].cells[0].text = "Priority"
    table.rows[0].cells[1].text = "High"
    document.save(path)

    parsed = DocumentParser(Settings(knowledge_dir=tmp_path)).parse(path)

    assert parsed.title == "Checkout Requirements"
    assert "saved cards" in parsed.content
    assert "Priority | High" in parsed.content
    assert parsed.metadata["author"] == "Product Team"
    assert parsed.metadata["source_format"] == "docx"
    assert parsed.metadata["extension"] == ".docx"


def test_extracts_pdf_content_and_properties(tmp_path: Path) -> None:
    path = tmp_path / "pricing-requirements.pdf"
    _write_simple_pdf(path, "Pricing Requirements", "Discount approval requires finance review.")

    parsed = DocumentParser(Settings(knowledge_dir=tmp_path)).parse(path)

    assert parsed.title == "Pricing Requirements"
    assert "finance review" in parsed.content
    assert parsed.metadata["source_format"] == "pdf"
    assert parsed.metadata["page_count"] == 1
    assert parsed.metadata["extension"] == ".pdf"


def test_extracts_pptx_content_notes_tables_and_properties(tmp_path: Path) -> None:
    path = tmp_path / "checkout-roadmap.pptx"
    presentation = Presentation()
    presentation.core_properties.title = "Checkout Roadmap"
    presentation.core_properties.author = "Product Team"
    slide = presentation.slides.add_slide(presentation.slide_layouts[5])
    slide.shapes.title.text = "Checkout Roadmap"
    slide.shapes.add_textbox(Inches(1), Inches(2), Inches(5), Inches(1)).text = (
        "Saved cards launch in Q4."
    )
    table = slide.shapes.add_table(1, 2, Inches(1), Inches(3), Inches(5), Inches(1)).table
    table.cell(0, 0).text = "Priority"
    table.cell(0, 1).text = "High"
    slide.notes_slide.notes_text_frame.text = "Confirm launch date with engineering."
    presentation.save(path)

    parsed = DocumentParser(Settings(knowledge_dir=tmp_path)).parse(path)

    assert parsed.title == "Checkout Roadmap"
    assert "Saved cards launch in Q4." in parsed.content
    assert "Priority | High" in parsed.content
    assert "Confirm launch date with engineering." in parsed.content
    assert parsed.metadata["author"] == "Product Team"
    assert parsed.metadata["source_format"] == "pptx"
    assert parsed.metadata["slide_count"] == 1
    assert parsed.metadata["extension"] == ".pptx"


class _FakeAttachment:
    def __init__(self, name: str) -> None:
        self.name = name


class _FakeMsgMessage:
    def __init__(self) -> None:
        self.subject = "Renewal terms for Acme"
        self.sender = "alice@example.com"
        self.to = "bob@example.com"
        self.cc = None
        self.date = datetime(2026, 3, 4, 12, 30, tzinfo=UTC)
        self.body = "We agreed to net-30 payment terms.\n"
        self.attachments = [_FakeAttachment("contract.pdf")]

    def __enter__(self) -> "_FakeMsgMessage":
        return self

    def __exit__(self, *_args: object) -> None:
        return None


def test_extracts_msg_content_and_properties(tmp_path: Path, monkeypatch) -> None:
    path = tmp_path / "renewal.msg"
    path.write_bytes(b"")
    monkeypatch.setattr(extract_msg, "openMsg", lambda *_args, **_kwargs: _FakeMsgMessage())

    parsed = DocumentParser(Settings(knowledge_dir=tmp_path)).parse(path)

    assert parsed.title == "Renewal terms for Acme"
    assert "net-30 payment terms" in parsed.content
    assert parsed.metadata["source_format"] == "msg"
    assert parsed.metadata["sender"] == "alice@example.com"
    assert parsed.metadata["to"] == "bob@example.com"
    assert parsed.metadata["attachment_names"] == ["contract.pdf"]
    assert parsed.metadata["extension"] == ".msg"
    assert parsed.effective_at == datetime(2026, 3, 4, 12, 30, tzinfo=UTC)


def _write_simple_pdf(path: Path, title: str, body: str) -> None:
    lines = [title, body]
    text_commands = ["BT", "/F1 12 Tf", "72 720 Td"]
    for index, line in enumerate(lines):
        if index:
            text_commands.append("0 -18 Td")
        text_commands.append(f"({_escape_pdf_text(line)}) Tj")
    text_commands.append("ET")
    stream = "\n".join(text_commands).encode("ascii")

    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
        b"/Resources << /Font << /F1 4 0 R >> >> /Contents 5 0 R >>",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
        b"<< /Length " + str(len(stream)).encode("ascii") + b" >>\nstream\n" + stream + b"\nendstream",
        b"<< /Title (" + _escape_pdf_text(title).encode("ascii") + b") >>",
    ]

    content = bytearray(b"%PDF-1.4\n")
    offsets = [0]
    for number, pdf_object in enumerate(objects, start=1):
        offsets.append(len(content))
        content.extend(f"{number} 0 obj\n".encode("ascii"))
        content.extend(pdf_object)
        content.extend(b"\nendobj\n")

    xref_offset = len(content)
    content.extend(f"xref\n0 {len(objects) + 1}\n".encode("ascii"))
    content.extend(b"0000000000 65535 f \n")
    for offset in offsets[1:]:
        content.extend(f"{offset:010d} 00000 n \n".encode("ascii"))
    content.extend(
        f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R /Info 6 0 R >>\n"
        f"startxref\n{xref_offset}\n%%EOF\n".encode("ascii")
    )
    path.write_bytes(bytes(content))


def _escape_pdf_text(value: str) -> str:
    return value.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
