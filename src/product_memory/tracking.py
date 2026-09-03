from __future__ import annotations

import json
import logging
import tempfile
from pathlib import Path
from typing import Any

from product_memory.settings import Settings

LOGGER = logging.getLogger(__name__)

# Everything that can change a score. A run whose parameters are not recorded cannot be compared
# against, so this is deliberately wider than the settings anyone expects to touch.
TRACKED_SETTINGS = (
    "embedding_provider",
    "embedding_model",
    "embedding_revision",
    "chunk_size",
    "chunk_overlap",
    "semantic_weight",
    "lexical_weight",
    "recency_weight",
    "recency_half_life_days",
    "rrf_k",
    "min_semantic_score",
    "candidate_pool_chunks",
    "candidate_pool_per_signal",
    "scoring_pool_chunks",
    "reranker_enabled",
    "reranker_model",
    "reranker_revision",
    "reranker_max_length",
    "reranker_rrf_k",
    "reranker_weight",
)

# Only recorded when a run actually generated answers; see run_metrics.
TRACKED_CHAT_SETTINGS = (
    "chat_model",
    "chat_condense_model",
    "chat_temperature",
    "chat_num_ctx",
    "chat_context_chars",
    "chat_thinking",
)

_METRICS = ("hit_rate", "mrr", "recall", "precision", "ndcg")
_GENERATION_METRICS = (
    "grounded_rate",
    "citation_coverage",
    "citation_precision",
    "groundedness",
)


def run_parameters(settings: Settings, index_profile: dict[str, Any]) -> dict[str, Any]:
    parameters = {name: getattr(settings, name) for name in TRACKED_SETTINGS}
    provider = index_profile.get("provider", {})
    parameters["embedding_dimension"] = provider.get("dimension")
    parameters["resolved_embedding_revision"] = provider.get("revision")
    return parameters


def run_metrics(report: dict[str, Any]) -> dict[str, float]:
    metrics = {name: report[name] for name in _METRICS if report.get(name) is not None}
    for name, value in (report.get("latency_seconds") or {}).items():
        if value is not None:
            metrics[f"latency_{name}"] = value
    metrics["questions"] = report.get("scored", 0)
    generation = report.get("generation") or {}
    for name in _GENERATION_METRICS:
        if generation.get(name) is not None:
            metrics[name] = generation[name]
    for name, value in (generation.get("latency_seconds") or {}).items():
        if value is not None:
            metrics[f"answer_latency_{name}"] = value
    return metrics


def log_evaluation(
    report: dict[str, Any],
    settings: Settings,
    index_profile: dict[str, Any],
    experiment: str,
    run_name: str | None = None,
    questions: str | None = None,
    corpus: dict[str, Any] | None = None,
) -> str | None:
    """Record one evaluation as an MLflow run. Returns the run id, or None when tracking is off.

    A failure here must not lose the evaluation that has already been paid for, so problems are
    logged and swallowed rather than raised.
    """
    if not settings.mlflow_tracking_uri:
        return None

    try:
        import mlflow
    except ImportError:
        LOGGER.warning("MLFLOW_TRACKING_URI is set but the mlflow client is not installed.")
        return None

    try:
        mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
        mlflow.set_experiment(experiment)
        with mlflow.start_run(run_name=run_name) as run:
            parameters = run_parameters(settings, index_profile)
            generation = report.get("generation") or {}
            if generation:
                # Only on runs that generated something. Otherwise every retrieval run carries a
                # chat model it never used, and comparing runs turns into reading footnotes.
                parameters.update(
                    {name: getattr(settings, name) for name in TRACKED_CHAT_SETTINGS}
                )
                parameters["judge_model"] = generation.get("judge_model")
            # The same questions against a larger index are harder questions, so a score means
            # nothing without the size of the haystack it was scored against.
            parameters.update({f"corpus_{name}": value for name, value in (corpus or {}).items()})
            mlflow.log_params(parameters)
            mlflow.log_metrics(run_metrics(report))
            mlflow.set_tags(
                {
                    "index_fingerprint": index_profile.get("fingerprint", "unknown"),
                    "questions": questions or "unknown",
                    "top_k": report.get("top_k"),
                }
            )
            # The per-question detail is what makes a regression diagnosable later.
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "evaluation.json"
                path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
                mlflow.log_artifact(str(path))
            return run.info.run_id
    except Exception:
        LOGGER.warning("Could not record this evaluation in MLflow.", exc_info=True)
        return None
