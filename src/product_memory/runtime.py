from __future__ import annotations

import json
import logging
from datetime import UTC, datetime

from product_memory.db import Database
from product_memory.embeddings.factory import create_embedding_provider
from product_memory.ingestion.cache import ExtractionCache
from product_memory.ingestion.chunker import DocumentChunker
from product_memory.ingestion.parser import DocumentParser
from product_memory.ingestion.service import IngestionService
from product_memory.metrics import (
    INDEX_BYTES,
    INDEX_CHUNKS,
    INDEX_DOCUMENTS,
    INDEX_FAILED,
    INDEX_SKIPPED,
)
from product_memory.retrieval.compressor import ContextCompressor
from product_memory.retrieval.reranker import Reranker
from product_memory.retrieval.service import Retriever
from product_memory.settings import Settings, get_settings

QUIET_PATHS = ("/health", "/health/live", "/metrics")


class _SuppressProbes(logging.Filter):
    """Drops the access log lines the healthcheck and the metrics scrape write every few seconds."""

    def filter(self, record: logging.LogRecord) -> bool:
        args = record.args
        if not isinstance(args, tuple) or len(args) < 3:
            return True
        return str(args[2]) not in QUIET_PATHS


class JsonFormatter(logging.Formatter):
    """One JSON object per line, so Loki can index the fields instead of the rendered column."""

    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "timestamp": datetime.fromtimestamp(record.created, tz=UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False, default=str)


class Runtime:
    def __init__(self, settings: Settings | None = None):
        self.settings = settings or get_settings()
        self.db = Database(self.settings)
        self.provider = create_embedding_provider(self.settings)
        self.parser = DocumentParser(self.settings, ExtractionCache(self.db))
        self.chunker = DocumentChunker(self.settings)
        self.ingestion = IngestionService(
            self.settings, self.db, self.provider, self.parser, self.chunker
        )
        self.retriever = Retriever(
            self.settings,
            self.db,
            self.provider,
            ContextCompressor(),
            Reranker(self.settings) if self.settings.reranker_enabled else None,
        )

    def initialize(self) -> None:
        level = getattr(logging, self.settings.log_level.upper(), logging.INFO)
        if self.settings.log_format == "json":
            handler = logging.StreamHandler()
            handler.setFormatter(JsonFormatter())
            logging.basicConfig(level=level, handlers=[handler], force=True)
            # uvicorn installs its own handlers, so its lines would stay unformatted and Loki
            # would index them as opaque text. Send them to the root handler instead.
            for name in ("uvicorn", "uvicorn.error", "uvicorn.access"):
                uvicorn_logger = logging.getLogger(name)
                uvicorn_logger.handlers.clear()
                uvicorn_logger.propagate = True
        else:
            logging.basicConfig(
                level=level,
                format="%(asctime)s %(levelname)s %(name)s: %(message)s",
            )
        access_logger = logging.getLogger("uvicorn.access")
        if not any(isinstance(existing, _SuppressProbes) for existing in access_logger.filters):
            access_logger.addFilter(_SuppressProbes())
        self.db.wait_until_ready()
        self.db.initialize_schema()
        self.ensure_index_profile_only()

    def ensure_index_profile_only(self) -> None:
        # The first scan is left to the watcher. Transcribing a recording takes as long as the
        # meeting did, and a service that will not answer until the whole folder is read is worse
        # than one that answers from what it has while the rest arrives.
        self.ingestion.ensure_index_profile()

    def refresh_index_gauges(self) -> None:
        totals = self.ingestion.index_totals()
        INDEX_DOCUMENTS.set(totals["documents"])
        INDEX_CHUNKS.set(totals["chunks"])
        INDEX_BYTES.set(totals["bytes"])
        INDEX_SKIPPED.set(self.ingestion.skipped_documents().get("count", 0))
        INDEX_FAILED.set(self.ingestion.failed_documents().get("count", 0))
