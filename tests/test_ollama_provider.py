from __future__ import annotations

import json
from typing import Any

import httpx
import pytest

from product_memory.generation.base import ChatMessage, ChatOptions, ChatProviderError
from product_memory.generation.ollama_client import OllamaChatProvider, ThinkFilter
from product_memory.settings import Settings

OPTIONS = ChatOptions(
    model="qwen3:8b",
    temperature=0.1,
    max_tokens=256,
    num_ctx=8192,
    thinking=False,
    keep_alive="5m",
    timeout_seconds=30.0,
)


def _ndjson(*events: dict[str, Any]) -> bytes:
    return "".join(json.dumps(event) + "\n" for event in events).encode("utf-8")


def _provider(handler) -> OllamaChatProvider:
    provider = OllamaChatProvider(Settings(_env_file=None))
    provider._client = httpx.Client(  # noqa: SLF001 - swapping the transport is the point
        transport=httpx.MockTransport(handler), base_url="http://host.docker.internal:11434"
    )
    return provider


def test_the_request_carries_the_settings_a_local_model_needs() -> None:
    seen: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen.update(json.loads(request.content))
        return httpx.Response(200, content=_ndjson({"done": True}))

    list(_provider(handler).stream([ChatMessage(role="user", content="hi")], OPTIONS))

    assert seen["model"] == "qwen3:8b"
    assert seen["stream"] is True
    assert seen["think"] is False
    assert seen["keep_alive"] == "5m"
    # Ollama defaults this to 4096 whatever the model can hold, so it must always be sent.
    assert seen["options"]["num_ctx"] == 8192
    assert seen["options"]["num_predict"] == 256
    assert seen["options"]["temperature"] == 0.1
    assert seen["messages"] == [{"role": "user", "content": "hi"}]


def test_a_streamed_answer_arrives_in_pieces_and_reports_its_tokens() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            content=_ndjson(
                {"message": {"content": "The decision "}},
                {"message": {"content": "was retries."}},
                {"done": True, "prompt_eval_count": 900, "eval_count": 12},
            ),
        )

    chunks = list(_provider(handler).stream([ChatMessage(role="user", content="hi")], OPTIONS))

    assert "".join(chunk.text for chunk in chunks) == "The decision was retries."
    assert chunks[-1].done
    assert chunks[-1].usage is not None
    assert chunks[-1].usage.prompt_tokens == 900
    assert chunks[-1].usage.completion_tokens == 12


def test_a_server_that_cannot_take_the_thinking_flag_is_asked_again_without_it() -> None:
    attempts: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        attempts.append(payload)
        if "think" in payload:
            return httpx.Response(400, json={"error": "model does not support think"})
        return httpx.Response(200, content=_ndjson({"message": {"content": "ok"}}, {"done": True}))

    result = _provider(handler).complete([ChatMessage(role="user", content="hi")], OPTIONS)

    assert result.text == "ok"
    assert len(attempts) == 2
    assert "think" not in attempts[1]


def test_a_model_that_was_never_pulled_says_how_to_pull_it() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(404, json={"error": "model not found"})

    with pytest.raises(ChatProviderError, match="ollama pull qwen3:8b"):
        _provider(handler).complete([ChatMessage(role="user", content="hi")], OPTIONS)


def test_an_error_reported_mid_stream_is_raised() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=_ndjson({"error": "out of memory"}))

    with pytest.raises(ChatProviderError, match="out of memory"):
        _provider(handler).complete([ChatMessage(role="user", content="hi")], OPTIONS)


def test_a_model_server_that_is_not_running_says_how_to_start_it() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused")

    with pytest.raises(ChatProviderError, match="ollama serve"):
        _provider(handler).complete([ChatMessage(role="user", content="hi")], OPTIONS)


def test_installed_models_are_listed() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"models": [{"model": "qwen3:8b"}, {"model": "qwen3:1.7b"}]})

    assert _provider(handler).available_models() == ["qwen3:1.7b", "qwen3:8b"]


def test_reasoning_is_not_shown_to_the_reader() -> None:
    filter_ = ThinkFilter()
    assert filter_.feed("<think>weighing it up</think>The answer.") == "The answer."


def test_reasoning_split_across_chunks_is_still_removed() -> None:
    filter_ = ThinkFilter()
    pieces = ["Before ", "<thi", "nk>hidden", " more hidden</thi", "nk>", "after."]
    assert "".join(filter_.feed(piece) for piece in pieces) + filter_.flush() == "Before after."


def test_an_answer_with_no_reasoning_passes_through_untouched() -> None:
    filter_ = ThinkFilter()
    assert filter_.feed("plain answer") + filter_.flush() == "plain answer"


def test_reasoning_that_never_closes_is_dropped_rather_than_leaked() -> None:
    filter_ = ThinkFilter()
    assert filter_.feed("<think>ran out of tokens") + filter_.flush() == ""
