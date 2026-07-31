from __future__ import annotations

import asyncio
import json

import typer
from rich import print_json

from product_memory.runtime import Runtime

app = typer.Typer(no_args_is_help=True, help="Administration commands for Product Memory RAG.")


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
    runtime = ready_runtime()
    result = runtime.ingestion.reindex_all()
    print_json(json.dumps(result))


@app.command("status")
def status() -> None:
    runtime = ready_runtime()
    print_json(json.dumps(runtime.retriever.status(), default=str))


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
                "status": status_result.structured_content,
                "retrieval": retrieval_result.structured_content,
            }

    print_json(json.dumps(asyncio.run(run()), default=str))


if __name__ == "__main__":
    app()
