from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml
from dateutil import parser as date_parser

from product_memory.ingestion.cache import ExtractionCache, file_signature
from product_memory.ingestion.extractors import ExtractedDocument, extract_document, strip_null_bytes
from product_memory.ingestion.metadata import infer_document_metadata
from product_memory.ingestion.ocr import OcrEngine
from product_memory.settings import Settings

_FRONTMATTER = re.compile(r"\A---\s*\n(.*?)\n---\s*\n?", re.DOTALL)
_FILENAME_DATE = re.compile(r"(?<!\d)(20\d{2})[-_.](\d{2})[-_.](\d{2})(?!\d)")
_MARKDOWN_TITLE = re.compile(r"^#\s+(.+?)\s*$", re.MULTILINE)


@dataclass(slots=True)
class ParsedDocument:
    source_path: str
    title: str
    content: str
    content_hash: str
    source_modified_at: datetime
    effective_at: datetime
    metadata: dict[str, Any]


class EmptyDocumentError(ValueError):
    """Raised when a file carries no extractable text, such as a picture without readable text."""


class DocumentParser:
    def __init__(self, settings: Settings, cache: ExtractionCache | None = None):
        self.settings = settings
        self.ocr = OcrEngine(settings)
        self.cache = cache

    def parse(self, path: Path, force: bool = False) -> ParsedDocument:
        extracted = self._extract(path, force=force)
        frontmatter, content = self._extract_frontmatter(extracted.content)
        content = content.strip()
        if not content and frontmatter:
            # A note that is nothing but front matter still carries its text in the field values.
            content = self._render_frontmatter(frontmatter)
        if not content:
            raise EmptyDocumentError(f"Document has no extractable text: {path}")

        metadata = {**extracted.metadata, **infer_document_metadata(path, content), **frontmatter}
        metadata = strip_null_bytes(metadata)
        stat = path.stat()
        modified_at = datetime.fromtimestamp(stat.st_mtime, tz=UTC)
        effective_at = self._effective_date(path, metadata, modified_at)
        title = self._title(path, metadata, content)
        relative_path = path.absolute().relative_to(self.settings.knowledge_dir.absolute()).as_posix()

        normalized_metadata = dict(metadata)
        normalized_metadata.setdefault("title", title)
        normalized_metadata.setdefault("effective_at", effective_at.isoformat())
        normalized_metadata.setdefault("extension", path.suffix.lower())
        normalized_metadata = json.loads(
            json.dumps(normalized_metadata, ensure_ascii=False, default=str)
        )

        return ParsedDocument(
            source_path=relative_path,
            title=title,
            content=content,
            content_hash=hashlib.sha256(content.encode("utf-8")).hexdigest(),
            source_modified_at=modified_at,
            effective_at=effective_at,
            metadata=normalized_metadata,
        )

    def _extract(self, path: Path, force: bool = False) -> ExtractedDocument:
        if self.cache is None:
            return extract_document(path, self.ocr)
        signature = file_signature(path)
        relative_path = path.absolute().relative_to(self.settings.knowledge_dir.absolute()).as_posix()
        if not force:
            cached = self.cache.get(relative_path, signature)
            if cached is not None:
                return cached
        extracted = extract_document(path, self.ocr)
        self.cache.set(relative_path, signature, extracted)
        return extracted

    @staticmethod
    def _extract_frontmatter(text: str) -> tuple[dict[str, Any], str]:
        match = _FRONTMATTER.match(text)
        if not match:
            return {}, text
        loaded = yaml.safe_load(match.group(1)) or {}
        if not isinstance(loaded, dict):
            raise ValueError("YAML front matter must be a mapping")
        return loaded, text[match.end() :]

    @staticmethod
    def _render_frontmatter(frontmatter: dict[str, Any]) -> str:
        lines = []
        for key, value in frontmatter.items():
            if value is None or value == "":
                continue
            if isinstance(value, list | tuple):
                value = ", ".join(str(item) for item in value)
            lines.append(f"{key}: {value}")
        return "\n".join(lines).strip()

    @staticmethod
    def _title(path: Path, metadata: dict[str, Any], content: str) -> str:
        if metadata.get("title"):
            return str(metadata["title"]).strip()
        match = _MARKDOWN_TITLE.search(content)
        if match:
            return match.group(1).strip()
        return path.stem.replace("_", " ").replace("-", " ").strip()

    @staticmethod
    def _effective_date(path: Path, metadata: dict[str, Any], fallback: datetime) -> datetime:
        candidate = metadata.get("effective_at") or metadata.get("date")
        if candidate:
            parsed = date_parser.parse(str(candidate))
            return parsed.replace(tzinfo=parsed.tzinfo or UTC).astimezone(UTC)

        match = _FILENAME_DATE.search(path.name)
        if match:
            return datetime(int(match.group(1)), int(match.group(2)), int(match.group(3)), tzinfo=UTC)
        return fallback
