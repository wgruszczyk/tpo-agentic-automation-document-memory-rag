from __future__ import annotations

import json
import time
import uuid
from collections.abc import Iterable, Iterator
from typing import Any

from product_memory.generation.base import ChatMessage
from product_memory.generation.chat import ChatAnswer, ChatEvent

# One model id, whatever is behind it. The client is choosing "answer from our documents", not a
# checkpoint; which local model serves that is an operator's decision, made in .env.
MODEL_ID = "product-memory"

_ROLES = {"system", "user", "assistant"}


class RequestError(ValueError):
    """A malformed request, safe to report back to the caller verbatim."""


def parse_messages(payload: dict[str, Any]) -> list[ChatMessage]:
    raw = payload.get("messages")
    if not isinstance(raw, list) or not raw:
        raise RequestError("messages must be a non-empty array")
    messages: list[ChatMessage] = []
    for item in raw:
        if not isinstance(item, dict):
            raise RequestError("each message must be an object")
        role = item.get("role")
        if role not in _ROLES:
            raise RequestError(f"unsupported message role: {role!r}")
        messages.append(ChatMessage(role=role, content=_content(item.get("content"))))
    if messages[-1].role != "user":
        raise RequestError("the last message must come from the user")
    return messages


def _content(value: Any) -> str:
    """Accepts both the plain string and the multipart array clients send for attachments."""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        parts = [
            str(part.get("text", ""))
            for part in value
            if isinstance(part, dict) and part.get("type") == "text"
        ]
        return "\n".join(part for part in parts if part)
    if value is None:
        return ""
    raise RequestError("message content must be a string or an array of text parts")


def models_payload() -> dict[str, Any]:
    return {
        "object": "list",
        "data": [
            {
                "id": MODEL_ID,
                "object": "model",
                "created": int(time.time()),
                "owned_by": "product-memory",
            }
        ],
    }


def completion_payload(answer: ChatAnswer, completion_id: str) -> dict[str, Any]:
    return {
        "id": completion_id,
        "object": "chat.completion",
        "created": int(time.time()),
        "model": MODEL_ID,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": answer.answer},
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": answer.usage.prompt_tokens,
            "completion_tokens": answer.usage.completion_tokens,
            "total_tokens": answer.usage.total_tokens,
        },
        # Not part of the OpenAI shape. Clients that do not know the field drop it; the ones that
        # do get the sources without having to parse them back out of the prose.
        "product_memory": {
            "question": answer.question,
            "grounded": answer.grounded,
            "citations": [citation.as_dict() for citation in answer.citations],
        },
    }


def new_completion_id() -> str:
    return f"chatcmpl-{uuid.uuid4().hex}"


def _sse(data: dict[str, Any]) -> str:
    return f"data: {json.dumps(data, ensure_ascii=False, default=str)}\n\n"


def _delta(completion_id: str, delta: dict[str, Any], finish_reason: str | None) -> str:
    return _sse(
        {
            "id": completion_id,
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": MODEL_ID,
            "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
        }
    )


def iter_sse(events: Iterable[ChatEvent], completion_id: str) -> Iterator[str]:
    yield _delta(completion_id, {"role": "assistant", "content": ""}, None)
    for event in events:
        if event.text:
            yield _delta(completion_id, {"content": event.text}, None)
    yield _delta(completion_id, {}, "stop")
    yield "data: [DONE]\n\n"


def error_sse(completion_id: str, message: str) -> Iterator[str]:
    """Streaming has already committed to a 200, so a failure has to arrive as text."""
    yield _delta(completion_id, {"content": f"\n\n**Error:** {message}\n"}, None)
    yield _delta(completion_id, {}, "stop")
    yield "data: [DONE]\n\n"

