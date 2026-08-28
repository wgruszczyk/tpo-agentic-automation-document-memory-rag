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
from product_memory.ingestion.extractors import (
    RECORDING_EXTENSIONS,
    EmptyDocumentError,
    ExtractedDocument,
    _OcrCollector,
    extract_document,
    read_screens,
    strip_null_bytes,
)
from product_memory.ingestion.frames import FrameSampler
from product_memory.ingestion.metadata import infer_document_metadata
from product_memory.ingestion.ocr import OcrEngine
from product_memory.ingestion.transcription import Transcriber
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


class DocumentParser:
    def __init__(self, settings: Settings, cache: ExtractionCache | None = None):
        self.settings = settings
        self.ocr = OcrEngine(settings)
        self.transcriber = Transcriber(settings)
        self.frames = FrameSampler(settings)
        self.cache = cache

    def parse(self, path: Path, force: bool = False) -> ParsedDocument:
        if path.stat().st_size == 0:
            # An empty file is nothing to report on; skip it rather than failing the format reader.
            raise EmptyDocumentError(f"Document is empty: {path}")
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

    def has_cached_extraction(self, path: Path, relative_path: str) -> bool:
        """Whether this file's text is already extracted, so re-reading it would cost nothing."""
        if self.cache is None:
            return False
        try:
            signature = self._signature(path)
        except OSError:
            # Asking about a file that is no longer there is answered, not raised: it has no
            # cached extraction, and reading it will fail in the ordinary way a moment later.
            return False
        return self.cache.get(relative_path, signature) is not None

    def _signature(self, path: Path) -> str:
        signature = file_signature(path)
        if path.suffix.lower() in RECORDING_EXTENSIONS:
            # A transcript depends on the model as much as on the file, and nothing else does.
            return f"{signature}:{self.settings.transcription_model}"
        return signature

    def _extract(self, path: Path, force: bool = False) -> ExtractedDocument:
        if self.cache is None:
            return extract_document(path, self.ocr, self.transcriber, self.frames)
        signature = self._signature(path)
        relative_path = path.absolute().relative_to(self.settings.knowledge_dir.absolute()).as_posix()
        # A rebuild re-reads files to pick up extraction changes, but re-hearing a recording costs
        # the length of the meeting again for a transcript the signature says cannot have changed.
        if not force or path.suffix.lower() in RECORDING_EXTENSIONS:
            cached = self.cache.get(relative_path, signature)
            if cached is not None:
                return self._with_screens(path, relative_path, signature, cached)
        extracted = extract_document(path, self.ocr, self.transcriber, self.frames)
        self.cache.set(relative_path, signature, extracted)
        return extracted

    def _with_screens(
        self, path: Path, relative_path: str, signature: str, cached: ExtractedDocument
    ) -> ExtractedDocument:
        """Add the shared screens to a transcript that was heard before they were being kept.

        Listening to the meeting again would cost its whole length for words already known, so the
        pictures are added on top of the transcript that is already in hand.
        """
        if path.suffix.lower() not in RECORDING_EXTENSIONS or not self.frames.enabled:
            return cached
        if self.cache is None or "screen_count" in cached.metadata:
            return cached
        collector = _OcrCollector(
            self.ocr,
            keep_images=self.settings.store_images,
            max_bytes=self.settings.max_stored_image_bytes,
        )
        content, screens = read_screens(path, cached.content, collector, self.frames)
        if not screens:
            return cached
        updated = ExtractedDocument(
            content=content,
            metadata={**cached.metadata, "screen_count": screens},
            images=collector.images,
        )
        self.cache.set(relative_path, signature, updated)
        return updated

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
