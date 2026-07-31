from __future__ import annotations

import json
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import psycopg
from pgvector.psycopg import register_vector
from psycopg.rows import dict_row

from product_memory.settings import Settings


class Database:
    def __init__(self, settings: Settings):
        self.settings = settings

    @contextmanager
    def connection(
        self, register_vector_type: bool = True
    ) -> Iterator[psycopg.Connection[Any]]:
        with psycopg.connect(self.settings.database_url, row_factory=dict_row) as conn:
            if register_vector_type:
                register_vector(conn)
            yield conn

    def wait_until_ready(self, timeout_seconds: int = 120) -> None:
        deadline = time.monotonic() + timeout_seconds
        last_error: Exception | None = None
        while time.monotonic() < deadline:
            try:
                with self.connection(register_vector_type=False) as conn:
                    conn.execute("SELECT 1")
                    return
            except Exception as exc:  # pragma: no cover - depends on external PostgreSQL
                last_error = exc
                time.sleep(1)
        raise RuntimeError(f"Database did not become ready: {last_error}")

    def initialize_schema(self) -> None:
        schema_path = Path(__file__).with_name("schema.sql")
        sql = schema_path.read_text(encoding="utf-8")
        with self.connection(register_vector_type=False) as conn:
            conn.execute(sql, prepare=False)
            conn.commit()

    @contextmanager
    def advisory_lock(self, name: str) -> Iterator[None]:
        with self.connection() as conn:
            conn.execute("SELECT pg_advisory_lock(hashtext(%s))", (name,))
            try:
                yield
            finally:
                conn.execute("SELECT pg_advisory_unlock(hashtext(%s))", (name,))
                conn.commit()

    def get_state(self, key: str) -> dict[str, Any] | None:
        with self.connection() as conn:
            row = conn.execute("SELECT value FROM system_state WHERE key = %s", (key,)).fetchone()
            return dict(row["value"]) if row else None

    def set_state(self, key: str, value: dict[str, Any]) -> None:
        with self.connection() as conn:
            conn.execute(
                """
                INSERT INTO system_state (key, value, updated_at)
                VALUES (%s, %s::jsonb, now())
                ON CONFLICT (key) DO UPDATE SET value = EXCLUDED.value, updated_at = now()
                """,
                (key, json.dumps(value, ensure_ascii=False)),
            )
            conn.commit()
