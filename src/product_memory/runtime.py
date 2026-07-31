from __future__ import annotations

import logging

from product_memory.db import Database
from product_memory.embeddings.factory import create_embedding_provider
from product_memory.ingestion.chunker import DocumentChunker
from product_memory.ingestion.parser import DocumentParser
from product_memory.ingestion.service import IngestionService
from product_memory.retrieval.compressor import ContextCompressor
from product_memory.retrieval.service import Retriever
from product_memory.settings import Settings, get_settings


class Runtime:
    def __init__(self, settings: Settings | None = None):
        self.settings = settings or get_settings()
        self.db = Database(self.settings)
        self.provider = create_embedding_provider(self.settings)
        self.parser = DocumentParser(self.settings)
        self.chunker = DocumentChunker(self.settings)
        self.ingestion = IngestionService(
            self.settings, self.db, self.provider, self.parser, self.chunker
        )
        self.retriever = Retriever(
            self.settings, self.db, self.provider, ContextCompressor()
        )

    def initialize(self) -> None:
        logging.basicConfig(
            level=getattr(logging, self.settings.log_level.upper(), logging.INFO),
            format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        )
        self.db.wait_until_ready()
        self.db.initialize_schema()
        self.ingestion.ensure_index_profile()
        self.ingestion.scan_once()
