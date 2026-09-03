from __future__ import annotations

import asyncio
import json
from typing import Any

import httpx
import pytest

from product_memory import mcp_server
from product_memory.generation.chat import ChatService
from product_memory.settings import Settings
from tests.test_chat import SOURCES, StubProvider, StubRetriever


def _configure(monkeypatch: pytest.MonkeyPatch, **overrides: Any) -> None:
    settings = Settings(_env_file=None, chat_enabled=True, **overrides)
    monkeypatch.setattr(mcp_server.runtime, "settings", settings)
    monkeypatch.setattr(
        mcp_server.runtime,
        "_chat",
        ChatService(settings, StubRetriever(SOURCES), StubProvider()),
    )


def _request(method: str, path: str, **kwargs: Any) -> httpx.Response:
    async def run() -> httpx.Response:
        transport = httpx.ASGITransport(app=mcp_server.app)
        async with httpx.AsyncClient(transport=transport, base_url="http://localhost") as client:
            return await client.request(method, path, **kwargs)

    return asyncio.run(run())


def test_a_client_is_offered_one_model(monkeypatch: pytest.MonkeyPatch) -> None:
    _configure(monkeypatch)
    response = _request("GET", "/v1/models")

    assert response.status_code == 200
    assert [item["id"] for item in response.json()["data"]] == ["product-memory"]


def test_the_endpoint_is_absent_until_conversation_is_switched_on(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _configure(monkeypatch)
    monkeypatch.setattr(mcp_server.runtime.settings, "chat_enabled", False)

    assert _request("GET", "/v1/models").status_code == 503


def test_a_shared_secret_is_required_once_one_is_set(monkeypatch: pytest.MonkeyPatch) -> None:
    _configure(monkeypatch, chat_api_key="s3cret")

    assert _request("GET", "/v1/models").status_code == 401
    assert _request("GET", "/v1/models", headers={"Authorization": "Bearer wrong"}).status_code == 401
    assert _request("GET", "/v1/models", headers={"Authorization": "Bearer s3cret"}).status_code == 200


def test_an_answer_comes_back_in_the_shape_a_chat_client_expects(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _configure(monkeypatch)
    response = _request(
        "POST",
        "/v1/chat/completions",
        json={"messages": [{"role": "user", "content": "How do payment retries work?"}]},
    )

    body = response.json()
    assert response.status_code == 200
    assert body["object"] == "chat.completion"
    assert body["choices"][0]["finish_reason"] == "stop"
    assert "exponential backoff" in body["choices"][0]["message"]["content"]
    assert body["usage"]["prompt_tokens"] == 100
    # Sources travel as data too, so a client does not have to parse them back out of the prose.
    assert body["product_memory"]["citations"][0]["source_path"] == "tpo/decisions/payments.md"


def test_a_streamed_answer_ends_the_way_the_protocol_says_it_must(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _configure(monkeypatch)
    response = _request(
        "POST",
        "/v1/chat/completions",
        json={"messages": [{"role": "user", "content": "hi"}], "stream": True},
    )

    lines = [line for line in response.text.splitlines() if line.startswith("data: ")]
    assert lines[-1] == "data: [DONE]"
    deltas = [json.loads(line[6:]) for line in lines[:-1]]
    assert deltas[0]["choices"][0]["delta"]["role"] == "assistant"
    assert deltas[-1]["choices"][0]["finish_reason"] == "stop"
    text = "".join(delta["choices"][0]["delta"].get("content", "") for delta in deltas)
    assert "exponential backoff" in text
    assert "**Sources**" in text


def test_a_request_with_no_question_is_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    _configure(monkeypatch)

    assert _request("POST", "/v1/chat/completions", json={"messages": []}).status_code == 400
    assert (
        _request(
            "POST",
            "/v1/chat/completions",
            json={"messages": [{"role": "assistant", "content": "hi"}]},
        ).status_code
        == 400
    )


def test_content_sent_as_parts_is_still_read(monkeypatch: pytest.MonkeyPatch) -> None:
    _configure(monkeypatch)
    response = _request(
        "POST",
        "/v1/chat/completions",
        json={
            "messages": [
                {"role": "user", "content": [{"type": "text", "text": "How do retries work?"}]}
            ]
        },
    )

    assert response.status_code == 200
    assert response.json()["product_memory"]["question"] == "How do retries work?"


def test_a_citation_leads_to_a_readable_source(monkeypatch: pytest.MonkeyPatch) -> None:
    from product_memory.models import FetchResponse

    class Fetcher:
        def fetch(self, item_id: str) -> FetchResponse:
            if item_id != "doc-1":
                raise KeyError(f"No active document or chunk found for id: {item_id}")
            return FetchResponse(
                id="doc-1", title="Payments", text="Retries use backoff.", url="x"
            )

    monkeypatch.setattr(mcp_server.runtime, "retriever", Fetcher())

    found = _request("GET", "/documents/doc-1")
    assert found.status_code == 200
    assert found.text == "# Payments\n\nRetries use backoff."
    assert _request("GET", "/documents/nope").status_code == 404
