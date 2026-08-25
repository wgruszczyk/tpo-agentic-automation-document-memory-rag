from __future__ import annotations

import time
from collections.abc import Iterator
from contextlib import contextmanager

from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram

REGISTRY = CollectorRegistry(auto_describe=True)

# Buckets reach 30s because a cold model load happens inside the first query, and a bucket set
# that tops out earlier would report it only as "+Inf" and hide how bad a slow query really was.
_LATENCY_BUCKETS = (0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 3.0, 5.0, 10.0, 30.0)

STAGE_SECONDS = Histogram(
    "product_memory_stage_seconds",
    "Duration of one stage of a retrieval.",
    labelnames=("stage",),
    buckets=_LATENCY_BUCKETS,
    registry=REGISTRY,
)

TOOL_SECONDS = Histogram(
    "product_memory_tool_seconds",
    "Duration of one MCP tool call, end to end.",
    labelnames=("tool",),
    buckets=_LATENCY_BUCKETS,
    registry=REGISTRY,
)

TOOL_CALLS = Counter(
    "product_memory_tool_calls_total",
    "MCP tool calls, by outcome.",
    labelnames=("tool", "outcome"),
    registry=REGISTRY,
)

RESULT_COUNT = Histogram(
    "product_memory_result_count",
    "How many items a retrieval returned.",
    labelnames=("kind",),
    buckets=(0, 1, 2, 3, 5, 7, 10, 15, 25, 40),
    registry=REGISTRY,
)

INGESTION_SECONDS = Histogram(
    "product_memory_ingestion_seconds",
    "Duration of one knowledge folder scan.",
    buckets=(0.1, 0.5, 1, 2, 5, 10, 30, 60, 300, 900),
    registry=REGISTRY,
)

INGESTION_DOCUMENTS = Counter(
    "product_memory_ingestion_documents_total",
    "Documents seen by a scan, by outcome.",
    labelnames=("outcome",),
    registry=REGISTRY,
)

INDEX_DOCUMENTS = Gauge(
    "product_memory_index_documents",
    "Active documents in the index.",
    registry=REGISTRY,
)

INDEX_CHUNKS = Gauge(
    "product_memory_index_chunks",
    "Chunks in the index.",
    registry=REGISTRY,
)

INDEX_SKIPPED = Gauge(
    "product_memory_index_skipped_documents",
    "Knowledge files excluded because they hold no indexable text.",
    registry=REGISTRY,
)

INDEX_BYTES = Gauge(
    "product_memory_index_bytes",
    "Disk used by the documents and chunks relations.",
    registry=REGISTRY,
)


@contextmanager
def observe(metric: Histogram, **labels: str) -> Iterator[None]:
    started = time.perf_counter()
    try:
        yield
    finally:
        target = metric.labels(**labels) if labels else metric
        target.observe(time.perf_counter() - started)


@contextmanager
def stage(name: str) -> Iterator[None]:
    with observe(STAGE_SECONDS, stage=name):
        yield


def render() -> tuple[bytes, str]:
    from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

    return generate_latest(REGISTRY), CONTENT_TYPE_LATEST
