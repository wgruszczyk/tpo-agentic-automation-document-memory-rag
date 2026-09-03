from __future__ import annotations

import json
import logging
from collections.abc import Iterator
from typing import Any

import httpx

from product_memory.generation.base import (
    ChatChunk,
    ChatMessage,
    ChatOptions,
    ChatProvider,
    ChatProviderError,
    ChatUsage,
)
from product_memory.settings import Settings

LOGGER = logging.getLogger(__name__)

_OPEN_THINK = "<think>"
_CLOSE_THINK = "</think>"


def _partial_tail(text: str, tag: str) -> int:
    """Length of the longest suffix of text that could still grow into tag."""
    for size in range(min(len(tag) - 1, len(text)), 0, -1):
        if text.endswith(tag[:size]):
            return size
    return 0


class ThinkFilter:
    """Removes inline <think> blocks from a stream without waiting for the end of it.

    Ollama reports reasoning in a separate field for models it knows think, but Qwen3 emits the
    tags inline whenever that negotiation does not happen, and half a reasoning trace arriving in
    a chat window reads as the model having lost its mind.
    """

    def __init__(self) -> None:
        self._buffer = ""
        self._inside = False

    def feed(self, text: str) -> str:
        self._buffer += text
        out: list[str] = []
        while True:
            if self._inside:
                index = self._buffer.find(_CLOSE_THINK)
                if index == -1:
                    self._buffer = self._buffer[-(len(_CLOSE_THINK) - 1) :]
                    break
                self._buffer = self._buffer[index + len(_CLOSE_THINK) :]
                self._inside = False
                continue
            index = self._buffer.find(_OPEN_THINK)
            if index == -1:
                held = _partial_tail(self._buffer, _OPEN_THINK)
                if held:
                    out.append(self._buffer[:-held])
                    self._buffer = self._buffer[-held:]
                else:
                    out.append(self._buffer)
                    self._buffer = ""
                break
            out.append(self._buffer[:index])
            self._buffer = self._buffer[index + len(_OPEN_THINK) :]
            self._inside = True
        return "".join(out)

    def flush(self) -> str:
        if self._inside:
            self._buffer = ""
            return ""
        text, self._buffer = self._buffer, ""
        return text


class _ThinkingUnsupported(Exception):
    """The server rejected the thinking flag; the same request without it will work."""


class OllamaChatProvider(ChatProvider):
    def __init__(self, settings: Settings):
        self.settings = settings
        self.base_url = settings.ollama_base_url.rstrip("/")
        self._client = httpx.Client(
            base_url=self.base_url,
            # A short connect timeout turns "Ollama is not running" into an immediate, obvious
            # failure; the read timeout has to cover a cold model load plus the whole answer.
            timeout=httpx.Timeout(
                connect=5.0, read=settings.chat_timeout_seconds, write=30.0, pool=5.0
            ),
        )

    def close(self) -> None:
        self._client.close()

    def profile(self) -> dict[str, Any]:
        return {"provider": "ollama", "model": self.settings.chat_model, "base_url": self.base_url}

    def available_models(self) -> list[str]:
        payload = self._get_json("/api/tags")
        return sorted(str(item.get("model", "")) for item in payload.get("models", []))

    def stream(self, messages: list[ChatMessage], options: ChatOptions) -> Iterator[ChatChunk]:
        payload = self._payload(messages, options)
        try:
            yield from self._stream(payload, options.timeout_seconds)
        except _ThinkingUnsupported:
            LOGGER.info("Ollama rejected the thinking flag for %s; retrying without it", options.model)
            payload.pop("think", None)
            yield from self._stream(payload, options.timeout_seconds)

    def _payload(self, messages: list[ChatMessage], options: ChatOptions) -> dict[str, Any]:
        return {
            "model": options.model,
            "messages": [message.as_payload() for message in messages],
            "stream": True,
            "think": options.thinking,
            "keep_alive": options.keep_alive,
            "options": {
                "temperature": options.temperature,
                "num_ctx": options.num_ctx,
                "num_predict": options.max_tokens,
            },
        }

    def _stream(self, payload: dict[str, Any], timeout: float) -> Iterator[ChatChunk]:
        model = str(payload.get("model", ""))
        filter_ = ThinkFilter()
        try:
            with self._client.stream("POST", "/api/chat", json=payload, timeout=timeout) as response:
                self._check(response, model, streaming=True)
                usage: ChatUsage | None = None
                for line in response.iter_lines():
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        event = json.loads(line)
                    except json.JSONDecodeError:
                        LOGGER.warning("Ignoring unparsable line from Ollama")
                        continue
                    if error := event.get("error"):
                        raise ChatProviderError(f"Ollama reported an error: {error}")
                    text = filter_.feed(str(event.get("message", {}).get("content", "")))
                    if text:
                        yield ChatChunk(text=text)
                    if event.get("done"):
                        usage = ChatUsage(
                            prompt_tokens=int(event.get("prompt_eval_count") or 0),
                            completion_tokens=int(event.get("eval_count") or 0),
                        )
                tail = filter_.flush()
                yield ChatChunk(text=tail, done=True, usage=usage or ChatUsage())
        except httpx.ConnectError as exc:
            raise ChatProviderError(
                f"Cannot reach Ollama at {self.base_url}. Start it with `ollama serve`, and if this "
                "service runs in a container make sure Ollama listens on all interfaces "
                "(OLLAMA_HOST=0.0.0.0)."
            ) from exc
        except httpx.TimeoutException as exc:
            raise ChatProviderError(
                f"Ollama did not answer within {timeout:.0f}s. A first call loads the model, which "
                "on a small machine can outlast this; raise CHAT_TIMEOUT_SECONDS or use a smaller "
                "model."
            ) from exc

    def _get_json(self, path: str) -> dict[str, Any]:
        try:
            response = self._client.get(path)
        except httpx.ConnectError as exc:
            raise ChatProviderError(
                f"Cannot reach Ollama at {self.base_url}. Start it with `ollama serve`."
            ) from exc
        self._check(response, "", streaming=False)
        return response.json()

    def _check(self, response: httpx.Response, model: str, *, streaming: bool) -> None:
        if response.is_success:
            return
        if streaming:
            response.read()
        body = response.text.strip()
        if response.status_code == 400 and "think" in body.lower():
            raise _ThinkingUnsupported(body)
        if response.status_code == 404 and model:
            raise ChatProviderError(
                f"Ollama has no model named {model!r}. Pull it first: `ollama pull {model}`."
            )
        raise ChatProviderError(f"Ollama returned HTTP {response.status_code}: {body[:500]}")
