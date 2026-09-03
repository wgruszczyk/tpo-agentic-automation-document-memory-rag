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
from product_memory.embeddings.base import EmbeddingProvider, passage_text
from product_memory.ingestion.chunker import DocumentChunker
from product_memory.ingestion.extractors import RECORDING_EXTENSIONS, UnreadableDocumentError
from product_memory.ingestion.parser import DocumentParser, EmptyDocumentError, ParsedDocument
from product_memory.metrics import INGESTION_DOCUMENTS, INGESTION_SECONDS, observe
from product_memory.settings import Settings

LOGGER = logging.getLogger(__name__)
INDEX_STATE_KEY = "index_profile"
SKIPPED_STATE_KEY = "skipped_documents"
FAILED_STATE_KEY = "failed_documents"
INDEX_LOCK = "product_memory_index_rebuild"
PIPELINE_VERSION = "1"
EXAMPLE_KNOWLEDGE_PATH = "example-knowledge.md"
KNOWLEDGE_README_PATHS = {"README.md", ".README.md"}
# A dot hides a file; "~$" marks the stub Office writes beside a document while it is open, which
# carries the right extension but holds no readable document.
IGNORED_NAME_PREFIXES = (".", "~$")
# Written by the scan once the whole folder is known, not read from the file, so a document whose
# only difference is one of these has not actually changed.
_DERIVED_KEYS = ("duplicate_source_paths", "duplicate_count", "all_source_paths")


def _without_derived(metadata: Any) -> dict[str, Any]:
    return {key: value for key, value in dict(metadata).items() if key not in _DERIVED_KEYS}


def reading_order(paths: list[Path]) -> list[Path]:
    """Documents first, recordings last.

    One meeting can take longer to hear than every document in the folder takes to read, so a
    scan that met them in name order would leave ordinary files waiting behind hours of audio.
    """
    return sorted(paths, key=lambda path: (path.suffix.lower() in RECORDING_EXTENSIONS, path))


class DuplicateIndex:
    """Remembers which path first carried each checksum, so copies are indexed once.

    The scan streams rather than reading the whole folder first, so this answers from what it has
    seen so far. Paths arrive sorted, which is what makes the surviving copy predictable.
    """

    def __init__(self) -> None:
        self.canonical_by_hash: dict[str, str] = {}
        self.duplicates_by_canonical: dict[str, list[str]] = {}
        self.count = 0

    def duplicate_of(self, content_hash: str, source_path: str) -> str | None:
        """Return the path already holding this content, or None when this is the first copy."""
        canonical = self.canonical_by_hash.get(content_hash)
        if canonical is None:
            self.canonical_by_hash[content_hash] = source_path
            return None
        self.duplicates_by_canonical.setdefault(canonical, []).append(source_path)
        self.count += 1
        return canonical


def _skip_reason(error: Exception, path: Path) -> str:
    reason = str(error).replace(str(path), "").strip().strip(":").strip()
    return reason or type(error).__name__


def _format_report(report: dict[str, Any], indent: int = 2) -> str:
    lines = []
    for key, value in report.items():
        if isinstance(value, dict):
            lines.append(f"{' ' * indent}{key}:")
            lines.append(_format_report(value, indent + 2))
        else:
            lines.append(f"{' ' * indent}{key}: {value}")
    return "\n".join(lines)


class FailureReporter:
    """Decides how loudly a file that could not be read should be reported.

    A stack trace earns its space the first time a fault appears and not the hundredth. An
    unreadable file stays unreadable, so repeating it every scan says nothing new, and when a
    whole folder fails the same way the traces differ only in the path.
    """

    def __init__(self, already_known: dict[str, str]):
        self._already_known = already_known
        self._reported_reasons: set[str] = set()

    def report(self, path: Path, source_path: str, reason: str) -> None:
        if self._already_known.get(source_path) == reason:
            return
        if reason in self._reported_reasons:
            LOGGER.warning("Failed to ingest %s: %s", path, reason)
            return
        self._reported_reasons.add(reason)
        # Only meaningful while an exception is being handled, which is the only caller.
        LOGGER.exception("Failed to ingest %s", path)


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

    def scan_once(self) -> dict[str, Any]:
        profile = self.ensure_index_profile()
        with self.db.advisory_lock(INDEX_LOCK):
            return self._scan_once_locked(profile)

    def rebuild_all(self) -> dict[str, Any]:
        # reindex_all only re-embeds stored content, so extraction changes such as new OCR
        # output need every file read from disk again.
        profile = self.ensure_index_profile()
        with self.db.advisory_lock(INDEX_LOCK):
            return self._scan_once_locked(profile, force=True)

    def _scan_once_locked(self, profile: dict[str, Any], force: bool = False) -> dict[str, Any]:
        with observe(INGESTION_SECONDS):
            return self._scan(profile, force=force)

    def _scan(self, profile: dict[str, Any], force: bool = False) -> dict[str, Any]:
        root = self.settings.knowledge_dir
        root.mkdir(parents=True, exist_ok=True)
        # Recordings are read last. One meeting can take longer to hear than every document in the
        # folder takes to read, and ordinary files should not wait behind it.
        paths = reading_order(self._discover_paths(root))
        added = updated = unchanged = deferred = 0
        skipped_documents: dict[str, str] = {}
        failed_documents: dict[str, str] = {}
        known_failures = {
            entry["source_path"]: entry["reason"]
            for entry in self.failed_documents().get("documents", [])
        }
        failures = FailureReporter(known_failures)
        recordings_left = self.settings.transcription_per_scan_limit
        seen = DuplicateIndex()
        found_paths: set[str] = set()

        for path in paths:
            relative_path = self._relative_source_path(root, path)
            try:
                if self._is_unread_recording(path, relative_path):
                    if recordings_left <= 0:
                        deferred += 1
                        continue
                    recordings_left -= 1
                parsed = self.parser.parse(path, force=force)
            except FileNotFoundError:
                # A synced folder can withdraw a file between listing it and reading it. There is
                # nothing to report: it is gone, and whatever was indexed from it is retired below.
                continue
            except (EmptyDocumentError, UnreadableDocumentError) as error:
                # A knowledge folder holds pictures, empty files and locked workbooks that will
                # never be indexable. Naming each one on every scan buries everything else, so
                # they are kept as a list and only reported when the list itself changes.
                skipped_documents[relative_path] = _skip_reason(error, path)
                continue
            except Exception as error:
                reason = f"{type(error).__name__}: {error}"
                failed_documents[relative_path] = reason
                failures.report(path, relative_path, reason)
                continue

            if seen.duplicate_of(parsed.content_hash, parsed.source_path) is not None:
                continue
            found_paths.add(parsed.source_path)

            # Each document is written as it is read, rather than the whole folder being held in
            # memory and written at the end. A scan interrupted halfway keeps what it has done.
            outcome = self._upsert_document(parsed, profile["fingerprint"], force=force)
            if outcome == "added":
                added += 1
            elif outcome == "updated":
                updated += 1
            else:
                unchanged += 1

        self._record_skipped(skipped_documents)
        self._record_failures(failed_documents)
        self._apply_duplicate_metadata(seen.duplicates_by_canonical)
        failed = len(failed_documents)

        removed = self._deactivate_missing(found_paths)
        counters = {
            "added": added,
            "updated": updated,
            "unchanged": unchanged,
            "removed": removed,
            "failed": failed,
            "skipped": len(skipped_documents),
            "deferred": deferred,
            "duplicates": seen.count,
        }
        for outcome, count in counters.items():
            INGESTION_DOCUMENTS.labels(outcome=outcome).inc(count)
        result = {"scan": counters, "index": self._index_totals()}
        # Failures and skips are standing conditions reported by their own recorders when they
        # change; repeating the whole report every scan because one file is still unreadable is
        # what buried everything else last time.
        if added or updated or removed:
            LOGGER.info("Ingestion scan\n%s", _format_report(result))
        return result

    def _index_totals(self) -> dict[str, Any]:
        with self.db.connection() as conn:
            row = conn.execute(
                """
                SELECT
                  (SELECT count(*) FROM documents WHERE is_active = TRUE) AS documents,
                  (SELECT count(*) FROM chunks) AS chunks,
                  pg_total_relation_size('documents') + pg_total_relation_size('chunks') AS bytes,
                  pg_size_pretty(
                    pg_total_relation_size('documents') + pg_total_relation_size('chunks')
                  ) AS size
                """
            ).fetchone()
        return {
            "documents": row["documents"],
            "chunks": row["chunks"],
            "bytes": row["bytes"],
            "size": row["size"],
        }

    def index_totals(self) -> dict[str, Any]:
        return self._index_totals()

    def image_totals(self) -> list[dict[str, Any]]:
        """Count stored pictures by where they came from."""
        with self.db.connection() as conn:
            return [
                dict(row)
                for row in conn.execute(
                    """
                    SELECT
                      CASE
                        WHEN label LIKE 'screen at %' THEN 'screen'
                        WHEN source_path ~* '[.](png|jpe?g|tiff?|bmp|webp|gif)$' THEN 'standalone'
                        ELSE 'embedded'
                      END AS kind,
                      count(*) AS images,
                      count(DISTINCT source_path) AS sources,
                      coalesce(sum(byte_size), 0) AS bytes
                    FROM images
                    GROUP BY 1
                    """
                ).fetchall()
            ]

    def _is_unread_recording(self, path: Path, relative_path: str) -> bool:
        """A recording whose transcript is not already cached, so reading it costs real time."""
        if path.suffix.lower() not in RECORDING_EXTENSIONS:
            return False
        return not self.parser.has_cached_extraction(path, relative_path)

    def _record_document_reasons(
        self, key: str, reasons: dict[str, str]
    ) -> tuple[int, int, int] | None:
        """Store a path-to-reason map, reporting totals only when it differs from last time."""
        entries = [
            {"source_path": path, "reason": reason} for path, reason in sorted(reasons.items())
        ]
        previous_entries = (self.db.get_state(key) or {}).get("documents", [])
        if entries == previous_entries:
            return None

        self.db.set_state(
            key,
            {
                "count": len(entries),
                "documents": entries,
                "updated_at": datetime.now(UTC).isoformat(),
            },
        )
        previous_paths = {entry["source_path"] for entry in previous_entries}
        current_paths = set(reasons)
        return (
            len(entries),
            len(current_paths - previous_paths),
            len(previous_paths - current_paths),
        )

    def _record_skipped(self, skipped_documents: dict[str, str]) -> None:
        changed = self._record_document_reasons(SKIPPED_STATE_KEY, skipped_documents)
        if changed:
            LOGGER.info(
                "Excluded files with no indexable text: %s total, %s new, %s resolved. "
                "Run 'product-memory skipped' for the list.",
                *changed,
            )

    def _record_failures(self, failed_documents: dict[str, str]) -> None:
        changed = self._record_document_reasons(FAILED_STATE_KEY, failed_documents)
        if changed:
            LOGGER.warning(
                "Files that could not be read: %s total, %s new, %s resolved. "
                "Run 'product-memory failures' for the list.",
                *changed,
            )

    def skipped_documents(self) -> dict[str, Any]:
        return self.db.get_state(SKIPPED_STATE_KEY) or {"count": 0, "documents": []}

    def failed_documents(self) -> dict[str, Any]:
        return self.db.get_state(FAILED_STATE_KEY) or {"count": 0, "documents": []}

    def _apply_duplicate_metadata(self, duplicates_by_canonical: dict[str, list[str]]) -> None:
        """Tell each kept document which other paths hold the same content.

        The duplicates are only known once the whole folder has been read, while the documents
        themselves are written as they are read, so this is applied afterwards rather than being
        folded into the document. It is derived from the scan rather than from the file, which is
        why _upsert_document does not treat it as the document having changed.
        """
        with self.db.connection() as conn:
            stale = conn.execute(
                "SELECT source_path FROM documents WHERE metadata ? 'duplicate_source_paths'"
            ).fetchall()
            for row in stale:
                if row["source_path"] not in duplicates_by_canonical:
                    conn.execute(
                        f"""
                        UPDATE documents
                        SET metadata = metadata {' '.join(f"- '{key}'" for key in _DERIVED_KEYS)}
                        WHERE source_path = %s
                        """,
                        (row["source_path"],),
                    )
            for canonical, duplicate_paths in duplicates_by_canonical.items():
                conn.execute(
                    "UPDATE documents SET metadata = metadata || %s::jsonb WHERE source_path = %s",
                    (
                        json.dumps(
                            {
                                "duplicate_source_paths": duplicate_paths,
                                "duplicate_count": len(duplicate_paths),
                                "all_source_paths": [canonical, *duplicate_paths],
                            }
                        ),
                        canonical,
                    ),
                )
            conn.commit()

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
        return not any(
            part.startswith(IGNORED_NAME_PREFIXES) for part in Path(relative_path).parts
        )

    @staticmethod
    def _has_real_knowledge_document(relative_paths: set[str]) -> bool:
        return any(path != EXAMPLE_KNOWLEDGE_PATH for path in relative_paths)

    @staticmethod
    def _relative_source_path(root: Path, path: Path) -> str:
        return path.absolute().relative_to(root.absolute()).as_posix()

    def _upsert_document(self, parsed: ParsedDocument, profile_hash: str, force: bool = False) -> str:
        with self.db.connection() as conn:
            existing = conn.execute(
                """
                SELECT id, title, content_hash, effective_at, metadata, indexed_profile_hash, is_active
                FROM documents WHERE source_path = %s
                """,
                (parsed.source_path,),
            ).fetchone()
            if (
                not force
                and existing
                and existing["title"] == parsed.title
                and existing["content_hash"] == parsed.content_hash
                and existing["effective_at"] == parsed.effective_at
                and _without_derived(existing["metadata"]) == _without_derived(parsed.metadata)
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
        passage_texts = [passage_text(title, chunk.content) for chunk in chunks]
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
                with conn.cursor() as cursor:
                    cursor.executemany(
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
                conn.execute("DELETE FROM images WHERE source_path = %s", (row["source_path"],))
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
