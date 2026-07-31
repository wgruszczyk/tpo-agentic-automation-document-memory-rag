from __future__ import annotations

import json
import logging
import os
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from product_memory.db import Database
from product_memory.embeddings.base import EmbeddingProvider
from product_memory.ingestion.chunker import DocumentChunker
from product_memory.ingestion.parser import DocumentParser, ParsedDocument
from product_memory.settings import Settings

LOGGER = logging.getLogger(__name__)
INDEX_STATE_KEY = "index_profile"
INDEX_LOCK = "product_memory_index_rebuild"
PIPELINE_VERSION = "1"
EXAMPLE_KNOWLEDGE_PATH = "example-knowledge.md"
KNOWLEDGE_README_PATHS = {"README.md", ".README.md"}


class IngestionService:
    def __init__(
        self,
        settings: Settings,
        db: Database,
        provider: EmbeddingProvider,
        parser: DocumentParser,
        chunker: DocumentChunker,
    ):
        self.settings = settings
        self.db = db
        self.provider = provider
        self.parser = parser
        self.chunker = chunker

    def index_profile(self) -> dict[str, Any]:
        provider_profile = self.provider.profile()
        fingerprint = self.provider.fingerprint(
            {
                "pipeline_version": PIPELINE_VERSION,
                "chunk_size": self.settings.chunk_size,
                "chunk_overlap": self.settings.chunk_overlap,
            }
        )
        return {
            "fingerprint": fingerprint,
            "provider": provider_profile,
            "chunk_size": self.settings.chunk_size,
            "chunk_overlap": self.settings.chunk_overlap,
            "pipeline_version": PIPELINE_VERSION,
        }

    def ensure_index_profile(self) -> dict[str, Any]:
        profile = self.index_profile()
        current = self.db.get_state(INDEX_STATE_KEY)
        if (
            current
            and current.get("fingerprint") == profile["fingerprint"]
            and current.get("status") == "ready"
        ):
            return current

        with self.db.advisory_lock(INDEX_LOCK):
            current = self.db.get_state(INDEX_STATE_KEY)
            if (
                current
                and current.get("fingerprint") == profile["fingerprint"]
                and current.get("status") == "ready"
            ):
                return current
            LOGGER.warning("Index profile changed or is incomplete. Rebuilding all embeddings.")
            self._set_profile_state(profile, status="reindexing")
            try:
                stats = self._reindex_all_locked(profile)
                self._set_profile_state(profile, status="ready", extra=stats)
            except Exception as exc:
                self._set_profile_state(profile, status="error", extra={"error": str(exc)})
                raise
        return self.db.get_state(INDEX_STATE_KEY) or profile

    def reindex_all(self) -> dict[str, Any]:
        profile = self.index_profile()
        with self.db.advisory_lock(INDEX_LOCK):
            self._set_profile_state(profile, status="reindexing")
            try:
                stats = self._reindex_all_locked(profile)
                self._set_profile_state(profile, status="ready", extra=stats)
                return stats
            except Exception as exc:
                self._set_profile_state(profile, status="error", extra={"error": str(exc)})
                raise

    def _reindex_all_locked(self, profile: dict[str, Any]) -> dict[str, Any]:
        with self.db.connection() as conn:
            documents = conn.execute(
                "SELECT id, title, content FROM documents WHERE is_active = TRUE ORDER BY effective_at"
            ).fetchall()
            conn.execute("DELETE FROM chunks")
            conn.execute("UPDATE documents SET indexed_profile_hash = NULL")
            conn.commit()

        chunk_count = 0
        for document in documents:
            chunk_count += self._index_document(
                document_id=document["id"],
                title=document["title"],
                content=document["content"],
                profile_hash=profile["fingerprint"],
            )
        return {
            "documents": len(documents),
            "chunks": chunk_count,
            "completed_at": datetime.now(UTC).isoformat(),
        }

    def scan_once(self) -> dict[str, int]:
        profile = self.ensure_index_profile()
        with self.db.advisory_lock(INDEX_LOCK):
            return self._scan_once_locked(profile)

    def _scan_once_locked(self, profile: dict[str, Any]) -> dict[str, int]:
        root = self.settings.knowledge_dir
        root.mkdir(parents=True, exist_ok=True)
        paths = self._discover_paths(root)
        found_paths: set[str] = set()
        added = updated = unchanged = failed = 0

        for path in paths:
            relative_path = self._relative_source_path(root, path)
            found_paths.add(relative_path)
            try:
                parsed = self.parser.parse(path)
                outcome = self._upsert_document(parsed, profile["fingerprint"])
                if outcome == "added":
                    added += 1
                elif outcome == "updated":
                    updated += 1
                else:
                    unchanged += 1
            except Exception:
                failed += 1
                LOGGER.exception("Failed to ingest %s", path)

        removed = self._deactivate_missing(found_paths)
        result = {
            "added": added,
            "updated": updated,
            "unchanged": unchanged,
            "removed": removed,
            "failed": failed,
        }
        if added or updated or removed or failed:
            LOGGER.info("Ingestion scan: %s", result)
        return result

    def _discover_paths(self, root: Path) -> list[Path]:
        paths = sorted(
            path
            for path in self._walk_files(root)
            if self._is_supported_visible_file(root, path)
        )
        relative_paths = {self._relative_source_path(root, path) for path in paths}
        if self._has_real_knowledge_document(relative_paths):
            paths = [
                path
                for path in paths
                if self._relative_source_path(root, path) != EXAMPLE_KNOWLEDGE_PATH
            ]
        return paths

    @staticmethod
    def _walk_files(root: Path) -> list[Path]:
        files: list[Path] = []
        seen_dirs: set[Path] = set()
        for dirpath, dirnames, filenames in os.walk(root.absolute(), followlinks=True):
            directory = Path(dirpath)
            try:
                real_directory = directory.resolve()
            except OSError:
                dirnames[:] = []
                continue

            if real_directory in seen_dirs:
                dirnames[:] = []
                continue
            seen_dirs.add(real_directory)

            dirnames[:] = [dirname for dirname in dirnames if not dirname.startswith(".")]
            files.extend(directory / filename for filename in filenames)
        return files

    def _is_supported_visible_file(self, root: Path, path: Path) -> bool:
        if not path.is_file() or path.suffix.lower() not in self.settings.extensions:
            return False
        relative_path = self._relative_source_path(root, path)
        if relative_path in KNOWLEDGE_README_PATHS:
            return False
        return not any(part.startswith(".") for part in Path(relative_path).parts)

    @staticmethod
    def _has_real_knowledge_document(relative_paths: set[str]) -> bool:
        return any(path != EXAMPLE_KNOWLEDGE_PATH for path in relative_paths)

    @staticmethod
    def _relative_source_path(root: Path, path: Path) -> str:
        return path.absolute().relative_to(root.absolute()).as_posix()

    def _upsert_document(self, parsed: ParsedDocument, profile_hash: str) -> str:
        with self.db.connection() as conn:
            existing = conn.execute(
                """
                SELECT id, title, content_hash, effective_at, metadata, indexed_profile_hash, is_active
                FROM documents WHERE source_path = %s
                """,
                (parsed.source_path,),
            ).fetchone()
            if (
                existing
                and existing["title"] == parsed.title
                and existing["content_hash"] == parsed.content_hash
                and existing["effective_at"] == parsed.effective_at
                and dict(existing["metadata"]) == parsed.metadata
                and existing["indexed_profile_hash"] == profile_hash
                and existing["is_active"]
            ):
                return "unchanged"

            document_id = existing["id"] if existing else uuid.uuid4()
            if existing:
                # Stop serving old chunks while a changed source is being re-embedded.
                conn.execute(
                    "UPDATE chunks SET embedding_profile_hash = 'stale' WHERE document_id = %s",
                    (document_id,),
                )
            conn.execute(
                """
                INSERT INTO documents (
                    id, source_path, title, content, content_hash, source_modified_at,
                    effective_at, metadata, indexed_profile_hash, is_active, updated_at
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s::jsonb, NULL, TRUE, now())
                ON CONFLICT (source_path) DO UPDATE SET
                    title = EXCLUDED.title,
                    content = EXCLUDED.content,
                    content_hash = EXCLUDED.content_hash,
                    source_modified_at = EXCLUDED.source_modified_at,
                    effective_at = EXCLUDED.effective_at,
                    metadata = EXCLUDED.metadata,
                    indexed_profile_hash = NULL,
                    is_active = TRUE,
                    updated_at = now()
                """,
                (
                    document_id,
                    parsed.source_path,
                    parsed.title,
                    parsed.content,
                    parsed.content_hash,
                    parsed.source_modified_at,
                    parsed.effective_at,
                    json.dumps(parsed.metadata, ensure_ascii=False, default=str),
                ),
            )
            conn.commit()

        self._index_document(document_id, parsed.title, parsed.content, profile_hash)
        return "updated" if existing else "added"

    def _index_document(
        self, document_id, title: str, content: str, profile_hash: str
    ) -> int:  # type: ignore[no-untyped-def]
        chunks = self.chunker.split(content)
        passage_texts = [f"Title: {title}\n\n{chunk.content}" for chunk in chunks]
        embeddings = self.provider.embed_documents(passage_texts)
        if len(embeddings) != len(chunks):
            raise RuntimeError("Embedding provider returned a different number of vectors than input chunks")

        with self.db.connection() as conn:
            conn.execute("DELETE FROM chunks WHERE document_id = %s", (document_id,))
            rows = []
            for chunk, embedding in zip(chunks, embeddings, strict=True):
                rows.append(
                    (
                        uuid.uuid4(),
                        document_id,
                        chunk.index,
                        chunk.content,
                        chunk.start_char,
                        chunk.end_char,
                        chunk.approx_tokens,
                        np.asarray(embedding, dtype=np.float32),
                        profile_hash,
                    )
                )
            if rows:
                conn.executemany(
                    """
                    INSERT INTO chunks (
                        id, document_id, chunk_index, content, start_char, end_char,
                        approx_tokens, embedding, embedding_profile_hash
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                    """,
                    rows,
                )
            conn.execute(
                "UPDATE documents SET indexed_profile_hash = %s, updated_at = now() WHERE id = %s",
                (profile_hash, document_id),
            )
            conn.commit()
        return len(chunks)

    def _deactivate_missing(self, found_paths: set[str]) -> int:
        with self.db.connection() as conn:
            active = conn.execute("SELECT id, source_path FROM documents WHERE is_active = TRUE").fetchall()
            missing = [row for row in active if row["source_path"] not in found_paths]
            for row in missing:
                conn.execute("DELETE FROM chunks WHERE document_id = %s", (row["id"],))
                conn.execute(
                    """
                    UPDATE documents
                    SET is_active = FALSE, indexed_profile_hash = NULL, updated_at = now()
                    WHERE id = %s
                    """,
                    (row["id"],),
                )
            conn.commit()
        return len(missing)

    def _set_profile_state(
        self, profile: dict[str, Any], status: str, extra: dict[str, Any] | None = None
    ) -> None:
        value = {
            **profile,
            "status": status,
            "updated_at": datetime.now(UTC).isoformat(),
            **(extra or {}),
        }
        self.db.set_state(INDEX_STATE_KEY, value)
