from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any, Literal

Role = Literal["system", "user", "assistant"]


@dataclass(frozen=True)
class ChatMessage:
    role: Role
    content: str

    def as_payload(self) -> dict[str, str]:
        return {"role": self.role, "content": self.content}


@dataclass(frozen=True)
class ChatOptions:
    model: str
    temperature: float = 0.2
    max_tokens: int = 1024
    num_ctx: int = 16384
    thinking: bool = False
    keep_alive: str = "10m"
    timeout_seconds: float = 180.0


@dataclass(frozen=True)
class ChatUsage:
    prompt_tokens: int = 0
    completion_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


@dataclass(frozen=True)
class ChatChunk:
    """One step of a streamed answer. Usage only arrives on the final chunk."""

    text: str = ""
    done: bool = False
    usage: ChatUsage | None = None


@dataclass
class ChatResult:
    text: str = ""
    usage: ChatUsage = field(default_factory=ChatUsage)


class ChatProviderError(RuntimeError):
    """Raised with a message meant for whoever has to fix the configuration."""


class ChatProvider(ABC):
    @abstractmethod
    def stream(self, messages: list[ChatMessage], options: ChatOptions) -> Iterator[ChatChunk]:
        raise NotImplementedError

    @abstractmethod
    def profile(self) -> dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def available_models(self) -> list[str]:
        raise NotImplementedError

    def complete(self, messages: list[ChatMessage], options: ChatOptions) -> ChatResult:
        result = ChatResult()
        parts: list[str] = []
        for chunk in self.stream(messages, options):
            parts.append(chunk.text)
            if chunk.usage is not None:
                result.usage = chunk.usage
        result.text = "".join(parts)
        return result
