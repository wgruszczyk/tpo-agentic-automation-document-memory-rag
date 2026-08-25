import json
import logging

from product_memory.metrics import REGISTRY, STAGE_SECONDS, TOOL_CALLS, render, stage
from product_memory.runtime import QUIET_PATHS, JsonFormatter, _SuppressProbes


def _sample(name: str, **labels: str) -> float:
    value = REGISTRY.get_sample_value(name, labels)
    return 0.0 if value is None else value


def test_a_stage_records_one_observation() -> None:
    before = _sample("product_memory_stage_seconds_count", stage="unit-test")

    with stage("unit-test"):
        pass

    assert _sample("product_memory_stage_seconds_count", stage="unit-test") == before + 1


def test_a_failing_stage_is_still_timed() -> None:
    before = _sample("product_memory_stage_seconds_count", stage="unit-test-failure")

    try:
        with stage("unit-test-failure"):
            raise RuntimeError("boom")
    except RuntimeError:
        pass

    assert _sample("product_memory_stage_seconds_count", stage="unit-test-failure") == before + 1


def test_rendered_metrics_expose_the_stage_histogram() -> None:
    with stage("render-check"):
        pass
    TOOL_CALLS.labels(tool="render-check", outcome="ok").inc()

    payload, content_type = render()
    text = payload.decode("utf-8")

    assert "text/plain" in content_type
    assert "product_memory_stage_seconds_bucket" in text
    assert 'product_memory_tool_calls_total{outcome="ok",tool="render-check"}' in text


def test_stage_histogram_carries_a_bucket_above_the_latency_budget() -> None:
    # A query is allowed a few seconds, so the buckets have to straddle that to be readable.
    buckets = STAGE_SECONDS._upper_bounds  # noqa: SLF001
    assert 3.0 in buckets
    assert max(bound for bound in buckets if bound != float("inf")) >= 10.0


def test_probe_requests_are_kept_out_of_the_access_log() -> None:
    log_filter = _SuppressProbes()

    def record(path: str) -> logging.LogRecord:
        return logging.LogRecord(
            name="uvicorn.access",
            level=logging.INFO,
            pathname=__file__,
            lineno=1,
            msg='%s - "%s %s HTTP/%s" %d',
            args=("127.0.0.1:50114", "GET", path, "1.1", 200),
            exc_info=None,
        )

    assert all(log_filter.filter(record(path)) is False for path in QUIET_PATHS)
    assert log_filter.filter(record("/mcp")) is True


def test_json_logs_carry_the_fields_loki_indexes() -> None:
    record = logging.LogRecord(
        name="product_memory.ingestion",
        level=logging.WARNING,
        pathname=__file__,
        lineno=1,
        msg="Skipped %s files",
        args=(11,),
        exc_info=None,
    )

    payload = json.loads(JsonFormatter().format(record))

    assert payload["level"] == "WARNING"
    assert payload["logger"] == "product_memory.ingestion"
    assert payload["message"] == "Skipped 11 files"
    assert payload["timestamp"].endswith("+00:00")
