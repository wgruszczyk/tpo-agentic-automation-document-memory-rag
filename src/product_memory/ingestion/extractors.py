from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from docx import Document as DocxDocument
from pptx import Presentation
from pypdf import PdfReader

# pypdf logs a WARNING for every recovered xref entry in malformed-but-readable PDFs;
# these are handled automatically and not actionable, so keep them out of app logs.
logging.getLogger("pypdf").setLevel(logging.ERROR)


@dataclass(slots=True)
class ExtractedDocument:
    content: str
    metadata: dict[str, Any]


def extract_document(path: Path) -> ExtractedDocument:
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        extracted = _extract_pdf(path)
    elif suffix == ".docx":
        extracted = _extract_docx(path)
    elif suffix == ".pptx":
        extracted = _extract_pptx(path)
    else:
        extracted = ExtractedDocument(content=_read_text(path), metadata={})
    return _sanitize_extracted(extracted)


def _sanitize_extracted(extracted: ExtractedDocument) -> ExtractedDocument:
    # Postgres text/jsonb columns reject NUL bytes; malformed PDF metadata can contain them.
    return ExtractedDocument(
        content=strip_null_bytes(extracted.content),
        metadata=strip_null_bytes(extracted.metadata),
    )


def strip_null_bytes(value: Any) -> Any:
    if isinstance(value, str):
        return value.replace("\x00", "")
    if isinstance(value, dict):
        return {key: strip_null_bytes(item) for key, item in value.items()}
    if isinstance(value, list):
        return [strip_null_bytes(item) for item in value]
    return value


def _extract_pdf(path: Path) -> ExtractedDocument:
    reader = PdfReader(str(path))
    if reader.is_encrypted:
        reader.decrypt("")

    pages = [(page.extract_text() or "").strip() for page in reader.pages]
    content = "\n\n".join(page for page in pages if page)
    metadata: dict[str, Any] = {
        "source_format": "pdf",
        "page_count": len(reader.pages),
    }

    info = reader.metadata
    if info:
        if info.title:
            metadata["title"] = str(info.title).strip()
        if info.author:
            metadata["author"] = str(info.author).strip()
        if info.subject:
            metadata["subject"] = str(info.subject).strip()
        if info.creator:
            metadata["creator"] = str(info.creator).strip()
        if info.producer:
            metadata["producer"] = str(info.producer).strip()
        if info.creation_date:
            metadata["created_at"] = _isoformat(info.creation_date)
        if info.modification_date:
            metadata["modified_at"] = _isoformat(info.modification_date)

    return ExtractedDocument(content=content, metadata=metadata)


def _extract_docx(path: Path) -> ExtractedDocument:
    document = DocxDocument(str(path))
    parts = [paragraph.text.strip() for paragraph in document.paragraphs if paragraph.text.strip()]

    for table in document.tables:
        for row in table.rows:
            cells = [cell.text.strip() for cell in row.cells if cell.text.strip()]
            if cells:
                parts.append(" | ".join(cells))

    metadata: dict[str, Any] = {"source_format": "docx"}
    props = document.core_properties
    if props.title:
        metadata["title"] = props.title.strip()
    if props.author:
        metadata["author"] = props.author.strip()
    if props.subject:
        metadata["subject"] = props.subject.strip()
    if props.keywords:
        metadata["keywords"] = props.keywords.strip()
    if props.created:
        metadata["created_at"] = _isoformat(props.created)
    if props.modified:
        metadata["modified_at"] = _isoformat(props.modified)

    return ExtractedDocument(content="\n\n".join(parts), metadata=metadata)


def _extract_pptx(path: Path) -> ExtractedDocument:
    presentation = Presentation(str(path))
    parts: list[str] = []

    for slide in presentation.slides:
        slide_parts: list[str] = []
        for shape in slide.shapes:
            if shape.has_text_frame:
                text = shape.text.strip()
                if text:
                    slide_parts.append(text)
            if shape.has_table:
                for row in shape.table.rows:
                    cells = [cell.text.strip() for cell in row.cells if cell.text.strip()]
                    if cells:
                        slide_parts.append(" | ".join(cells))

        notes = slide.notes_slide.notes_text_frame.text.strip()
        if notes:
            slide_parts.append(notes)
        if slide_parts:
            parts.append("\n".join(slide_parts))

    metadata: dict[str, Any] = {
        "source_format": "pptx",
        "slide_count": len(presentation.slides),
    }
    props = presentation.core_properties
    if props.title:
        metadata["title"] = props.title.strip()
    if props.author:
        metadata["author"] = props.author.strip()
    if props.subject:
        metadata["subject"] = props.subject.strip()
    if props.keywords:
        metadata["keywords"] = props.keywords.strip()
    if props.created:
        metadata["created_at"] = _isoformat(props.created)
    if props.modified:
        metadata["modified_at"] = _isoformat(props.modified)

    return ExtractedDocument(content="\n\n".join(parts), metadata=metadata)


def _read_text(path: Path) -> str:
    for encoding in ("utf-8", "utf-8-sig", "cp1250", "latin-1"):
        try:
            return path.read_text(encoding=encoding)
        except UnicodeDecodeError:
            continue
    raise UnicodeDecodeError("unknown", b"", 0, 1, f"Cannot decode {path}")


def _isoformat(value: datetime) -> str:
    return value.isoformat()
