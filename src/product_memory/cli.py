from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import typer
from rich import print_json

from product_memory.runtime import Runtime

app = typer.Typer(no_args_is_help=True, help="Administration commands for Product Memory RAG.")


def tool_result_payload(result: Any) -> Any:
    if result.structured_content is not None:
        return result.structured_content
    for item in result.content or []:
        text = getattr(item, "text", None)
        if text is None:
            continue
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return {"text": text}
    return None


def ready_runtime() -> Runtime:
    runtime = Runtime()
    runtime.db.wait_until_ready()
    runtime.db.initialize_schema()
    return runtime


@app.command("wait-for-db")
def wait_for_db(timeout_seconds: int = 120) -> None:
    runtime = Runtime()
    runtime.db.wait_until_ready(timeout_seconds=timeout_seconds)
    typer.echo("database ready")


@app.command("ingest-once")
def ingest_once() -> None:
    runtime = ready_runtime()
    result = runtime.ingestion.scan_once()
    print_json(json.dumps(result))


@app.command("reindex")
def reindex() -> None:
    """Rebuild embeddings from the content already stored in the database."""
    runtime = ready_runtime()
    result = runtime.ingestion.reindex_all()
    print_json(json.dumps(result))


@app.command("rebuild")
def rebuild() -> None:
    """Re-read every file from disk, then rebuild content, metadata, and embeddings."""
    runtime = ready_runtime()
    result = runtime.ingestion.rebuild_all()
    print_json(json.dumps(result))


@app.command("status")
def status() -> None:
    runtime = ready_runtime()
    print_json(json.dumps(runtime.retriever.status(), default=str))


@app.command("skipped")
def skipped() -> None:
    """List the knowledge files that carry no indexable text, with the reason for each."""
    runtime = ready_runtime()
    print_json(json.dumps(runtime.ingestion.skipped_documents(), default=str))


@app.command("failures")
def failures() -> None:
    """List the knowledge files that could not be read, with the error for each."""
    runtime = ready_runtime()
    print_json(json.dumps(runtime.ingestion.failed_documents(), default=str))


@app.command("warmup")
def warmup() -> None:
    """Download the embedding and reranker models into the cache, and report their revisions.

    This is the only command that reaches the internet on purpose. Pin the revisions it prints
    to keep later deployments reproducible and offline.
    """
    from product_memory.embeddings.factory import create_embedding_provider
    from product_memory.ingestion.transcription import Transcriber
    from product_memory.retrieval.reranker import Reranker
    from product_memory.settings import get_settings

    settings = get_settings().model_copy(update={"allow_model_download": True})
    report = {"embedding": create_embedding_provider(settings).profile()}
    if settings.reranker_enabled:
        report["reranker"] = Reranker(settings).warmup()
    if settings.enable_transcription:
        transcriber = Transcriber(settings)
        transcriber._load_model()  # noqa: SLF001 - the download is the point
        report["transcription"] = {"model": transcriber.repository}
    print_json(json.dumps(report, default=str))


@app.command("generate-eval")
def generate_eval(
    count: int = typer.Option(50, min=1, max=500, help="How many questions to generate."),
    seed: str = typer.Option("product-memory", help="Changing this samples different documents."),
    terms: int = typer.Option(8, min=3, max=20, help="Words taken from each sampled passage."),
) -> None:
    """Build a question set from the indexed documents and write the YAML to stdout.

    Each question is the most distinctive wording of one passage, expecting the document it came
    from. Written questions are better evidence; these exist so there is something to measure
    before anyone has written any.
    """
    from product_memory.eval_generation import generate_cases, render_yaml

    runtime = ready_runtime()
    cases = generate_cases(runtime.db, count=count, seed=seed, terms_per_question=terms)
    if not cases:
        raise typer.BadParameter("No indexed documents were usable. Run an ingest first.")

    # The questions go to stdout so the caller decides where they land; the knowledge and eval
    # folders are mounted read-only on purpose.
    typer.echo(render_yaml(cases, seed))
    documents = {path for case in cases for path in case.source_paths}
    typer.echo(f"Generated {len(cases)} questions from {len(documents)} documents.", err=True)


@app.command("compare-embeddings")
def compare_embeddings(
    model: str = typer.Option(..., help="The candidate embedding model to judge."),
    revision: str = typer.Option("", help="Pin the candidate to a Hugging Face commit."),
    questions: str = typer.Option("/eval/questions.yaml", help="Question set to score against."),
    distractors: int = typer.Option(1000, min=0, max=100000, help="Unrelated chunks in the pool."),
    top_k: int = typer.Option(7, min=1, max=50, help="Documents each question may return."),
) -> None:
    """Judge a candidate embedding model without re-embedding the whole index.

    Only the candidate embeds anything; the current model's vectors are read from the index. Scores
    on cosine alone, so it isolates the embedding and ignores the lexical, recency and reranking
    stages that follow. Run it before paying for a full reindex.
    """
    from product_memory.embedding_probe import compare_embedding_models
    from product_memory.embeddings.factory import create_embedding_provider
    from product_memory.evaluation import load_cases
    from product_memory.settings import get_settings

    path = Path(questions)
    if not path.exists():
        raise typer.BadParameter(f"{path} does not exist. Run 'product-memory generate-eval' first.")

    runtime = ready_runtime()
    candidate_settings = get_settings().model_copy(
        update={
            "embedding_model": model,
            "embedding_revision": revision or None,
            "allow_model_download": True,
        }
    )
    report = compare_embedding_models(
        runtime.db,
        current=runtime.provider,
        candidate=create_embedding_provider(candidate_settings),
        cases=load_cases(path),
        distractors=distractors,
        top_k=top_k,
    )
    print_json(json.dumps(report, default=str))


@app.command("smoke-test")
def smoke_test(url: str = "http://127.0.0.1:2600/mcp") -> None:
    """Connect through MCP, list tools, and execute a real retrieval against the sample knowledge."""
    from mcp import Client

    async def run() -> dict:
        async with Client(url) as client:
            tools = await client.list_tools()
            status_result = await client.call_tool("knowledge_status", {})
            retrieval_result = await client.call_tool(
                "retrieve_knowledge",
                {"query": "What is outside the MVP scope?", "top_k_chunks": 3, "top_k_documents": 1},
            )
            return {
                "tools": [tool.name for tool in tools.tools],
                "status": tool_result_payload(status_result),
                "retrieval": tool_result_payload(retrieval_result),
            }

    print_json(json.dumps(asyncio.run(run()), default=str))


@app.command("query")
def query_mcp(
    query: str = typer.Argument(..., help="Question to ask through retrieve_knowledge."),
    url: str = typer.Option("http://127.0.0.1:2600/mcp", help="MCP Streamable HTTP endpoint."),
    top_k_chunks: int = typer.Option(10, min=1, max=50, help="Maximum ranked chunks to return."),
    top_k_documents: int = typer.Option(7, min=1, max=25, help="Maximum complete documents to return."),
    project: str | None = typer.Option(None, help="Optional metadata.project filter."),
    since: str | None = typer.Option(None, help="Only documents effective on or after this ISO date."),
    until: str | None = typer.Option(None, help="Only documents effective on or before this ISO date."),
    include_full_documents: bool = typer.Option(
        True,
        "--include-full-documents/--no-full-documents",
        help="Include complete top document content in the response.",
    ),
    max_context_chars: int | None = typer.Option(None, min=2000, help="Optional context_pack character cap."),
) -> None:
    """Run retrieve_knowledge through the MCP server and print formatted JSON."""
    from mcp import Client

    async def run() -> dict:
        payload = {
            "query": query,
            "top_k_chunks": top_k_chunks,
            "top_k_documents": top_k_documents,
            "project": project,
            "since": since,
            "until": until,
            "include_full_documents": include_full_documents,
            "max_context_chars": max_context_chars,
        }
        compact_payload = {key: value for key, value in payload.items() if value is not None}
        async with Client(url) as client:
            result = await client.call_tool("retrieve_knowledge", compact_payload)
            return tool_result_payload(result)

    print_json(json.dumps(asyncio.run(run()), default=str))


@app.command("eval")
def evaluate(
    questions: str = typer.Option(
        "/eval/questions.yaml", help="YAML file of questions and expected source paths."
    ),
    top_k: int = typer.Option(7, min=1, max=25, help="How many documents each question may return."),
    verbose: bool = typer.Option(False, help="Print the ranked documents for every question."),
    track: bool = typer.Option(False, help="Record the run in MLflow for later comparison."),
    experiment: str = typer.Option("product-memory", help="MLflow experiment to record under."),
    run_name: str = typer.Option("", help="Name this run, so it is recognisable in MLflow."),
) -> None:
    """Score retrieval against a question set and report hit rate and mean reciprocal rank."""
    from product_memory.evaluation import load_cases, run_evaluation

    path = Path(questions)
    if not path.exists():
        raise typer.BadParameter(
            f"{path} does not exist. Copy eval/questions.example.yaml and fill in real questions."
        )
    runtime = ready_runtime()
    report = run_evaluation(runtime.retriever, load_cases(path), top_k)

    if track:
        from product_memory.tracking import log_evaluation

        if not runtime.settings.mlflow_tracking_uri:
            raise typer.BadParameter(
                "--track needs MLFLOW_TRACKING_URI. Start MLflow with 'make observability'."
            )
        status = runtime.retriever.status()
        run_id = log_evaluation(
            report,
            settings=runtime.settings,
            index_profile=status.get("profile", {}),
            experiment=experiment,
            run_name=run_name or None,
            questions=path.name,
            corpus={
                "documents": status.get("documents"),
                "chunks": status.get("chunks"),
                "images": status.get("images"),
            },
        )
        report["mlflow_run_id"] = run_id

    if not verbose:
        report.pop("results")
    print_json(json.dumps(report, default=str))


if __name__ == "__main__":
    app()
