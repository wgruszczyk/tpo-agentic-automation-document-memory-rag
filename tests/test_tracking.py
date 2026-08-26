from __future__ import annotations

import logging

from product_memory.settings import Settings
from product_memory.tracking import TRACKED_SETTINGS, log_evaluation, run_metrics, run_parameters

REPORT = {
    "questions": 50,
    "scored": 50,
    "top_k": 7,
    "hit_rate": 0.68,
    "mrr": 0.4542,
    "recall": 0.68,
    "precision": 0.0971,
    "ndcg": 0.5094,
    "latency_seconds": {"mean": 5.6, "p50": 5.4, "p95": 7.6, "max": 9.1},
    "misses": ["a question"],
    "results": [{"question": "a question"}],
}

PROFILE = {
    "fingerprint": "abc123",
    "provider": {"dimension": 768, "revision": "d128750", "model": "e5-base"},
}


def test_every_setting_that_can_move_a_score_is_recorded() -> None:
    parameters = run_parameters(Settings(_env_file=None), PROFILE)

    # A run whose parameters were not captured cannot be compared against later.
    for name in ("embedding_model", "reranker_model", "chunk_size", "semantic_weight",
                 "scoring_pool_chunks", "reranker_max_length", "min_semantic_score"):
        assert name in parameters
    assert set(TRACKED_SETTINGS) <= set(parameters)


def test_the_resolved_dimension_and_revision_come_from_the_index_not_the_settings() -> None:
    # embedding_revision is empty by default, but the index knows what was actually used.
    parameters = run_parameters(Settings(_env_file=None), PROFILE)

    assert parameters["embedding_dimension"] == 768
    assert parameters["resolved_embedding_revision"] == "d128750"


def test_metrics_are_flattened_so_latency_is_comparable_across_runs() -> None:
    metrics = run_metrics(REPORT)

    assert metrics["hit_rate"] == 0.68
    assert metrics["ndcg"] == 0.5094
    assert metrics["latency_p95"] == 7.6
    assert metrics["questions"] == 50
    assert "latency_seconds" not in metrics


def test_metrics_omit_values_a_question_set_could_not_produce() -> None:
    metrics = run_metrics({"scored": 0, "hit_rate": None, "latency_seconds": {"p95": None}})

    assert "hit_rate" not in metrics
    assert "latency_p95" not in metrics


def test_tracking_is_skipped_when_no_server_is_configured() -> None:
    assert log_evaluation(REPORT, Settings(_env_file=None), PROFILE, experiment="x") is None


def test_a_failing_tracking_server_does_not_lose_the_evaluation(caplog, monkeypatch) -> None:
    import mlflow

    # Pointing at a dead address instead would make this test wait out the client's retries.
    def unreachable(*_args: object, **_kwargs: object) -> None:
        raise ConnectionError("tracking server is down")

    monkeypatch.setattr(mlflow, "set_experiment", unreachable)
    settings = Settings(_env_file=None, mlflow_tracking_uri="http://mlflow:5000")

    with caplog.at_level(logging.WARNING):
        result = log_evaluation(REPORT, settings, PROFILE, experiment="x")

    # The evaluation has already been paid for; a tracking failure must not throw it away.
    assert result is None
    assert "MLflow" in caplog.text
