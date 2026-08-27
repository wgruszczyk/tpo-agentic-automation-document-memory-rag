from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import Any

from product_memory.db import Database
from product_memory.ingestion.extractors import ExtractedDocument


def file_signature(path: Path) -> str:
    stat = path.stat()
    return f"{stat.st_mtime_ns}:{stat.st_size}"


class ExtractionCache:
    """Stores extractor output so an unchanged file is never OCR'd twice."""

    def __init__(self, db: Database):
        self.db = db

    def get(self, source_path: str, signature: str) -> ExtractedDocument | None:
        with self.db.connection(register_vector_type=False) as conn:
            row = conn.execute(
                "SELECT content, metadata FROM extraction_cache WHERE source_path = %s AND signature = %s",
                (source_path, signature),
            ).fetchone()
        if row is None:
            return None
        return ExtractedDocument(content=row["content"], metadata=dict(row["metadata"]))

    def set(self, source_path: str, signature: str, extracted: ExtractedDocument) -> None:
        with self.db.connection(register_vector_type=False) as conn:
            conn.execute(
                """
                INSERT INTO extraction_cache (source_path, signature, content, metadata, updated_at)
                VALUES (%s, %s, %s, %s::jsonb, now())
                ON CONFLICT (source_path) DO UPDATE SET
                    signature = EXCLUDED.signature,
                    content = EXCLUDED.content,
                    metadata = EXCLUDED.metadata,
                    updated_at = now()
                """,
                (
                    source_path,
                    signature,
                    extracted.content,
                    json.dumps(extracted.metadata, ensure_ascii=False, default=str),
                ),
            )
            self._store_images(conn, source_path, signature, extracted)
            conn.commit()

    @staticmethod
    def _store_images(conn: Any, source_path: str, signature: str, doc: ExtractedDocument) -> None:
        # Images belong to the extraction, not to the document row, so they are replaced whenever
        # the file is read again and left alone when a cached read is reused.
        conn.execute("DELETE FROM images WHERE source_path = %s", (source_path,))
        for ordinal, image in enumerate(doc.images):
            conn.execute(
                """
                INSERT INTO images (
                    id, source_path, signature, label, ordinal,
                    media_type, width, height, byte_size, text, data
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """,
                (
                    uuid.uuid4(),
                    source_path,
                    signature,
                    image.label,
                    ordinal,
                    image.media_type,
                    image.width,
                    image.height,
                    len(image.data),
                    image.text,
                    image.data,
                ),
            )
