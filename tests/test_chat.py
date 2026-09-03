from __future__ import annotations

from collections.abc import Iterator
from datetime import UTC, datetime
from typing import Any

import pytest

from product_memory.generation.base import ChatChunk, ChatMessage, ChatOptions, ChatProvider, ChatUsage
from product_memory.generation.chat import NO_EVIDENCE_ANSWER, ChatService
from product_memory.models import ChunkResult, RetrievalResponse
from product_memory.retrieval.compressor import ContextCompressor
from product_memory.settings import Settings


class StubProvider(ChatProvider):
    def __init__(self, replies: list[str] | None = None):
        self.replies = replies or ["Retries use exponential backoff [1]."]
        self.calls: list[tuple[list[ChatMessage], ChatOptions]] = []

    def stream(self, messages: list[ChatMessage], options: ChatOptions) -> Iterator[ChatChunk]:
        self.calls.append((messages, options))
        reply = self.replies[min(len(self.calls) - 1, len(self.replies) - 1)]
        for word in reply.split(" "):
            yield ChatChunk(text=word + " ")
        yield ChatChunk(done=True, usage=ChatUsage(prompt_tokens=100, completion_tokens=10))

    def profile(self) -> dict[str, Any]:
        return {"provider": "stub"}

    def available_models(self) -> list[str]:
        return ["stub"]


def _chunk(index: int, path: str, content: str) -> ChunkResult:
    return ChunkResult(
        id=f"chunk-{index}",
        document_id=f"doc-{index}",
        document_title=f"Document {index}",
        source_path=path,
        chunk_index=0,
        content=content,
        start_char=0,
        end_char=len(content),
        effective_at=datetime(2026, 3, 1, tzinfo=UTC),
        semantic_score=0.9,
        lexical_score=0.1,
        recency_score=0.5,
        score=0.9 - index / 100,
    )


class StubRetriever:
    def __init__(self, chunks: list[ChunkResult]):
        self.chunks = chunks
        self.compressor = ContextCompressor()
        self.queries: list[str] = []

    def retrieve(self, query: str, **_: Any) -> RetrievalResponse:
        self.queries.append(query)
        return RetrievalResponse(
            query=query,
            chunks=self.chunks,
            documents=[],
            context_pack="",
            index_profile={"fingerprint": "abc"},
        )


def _service(chunks: list[ChunkResult], provider: ChatProvider | None = None, **overrides) -> ChatService:
    settings = Settings(_env_file=None, chat_enabled=True, **overrides)
    return ChatService(settings, StubRetriever(chunks), provider or StubProvider())


SOURCES = [
    _chunk(1, "tpo/decisions/payments.md", "Payment retries use exponential backoff."),
    _chunk(2, "tpo/meetings/2026-03-01.vtt", "We agreed to cap retries at five."),
]


def test_an_answer_carries_the_sources_it_was_built_from() -> None:
    answer = _service(SOURCES).ask("How do payment retries work?")

    assert answer.grounded
    assert [citation.marker for citation in answer.citations] == [1, 2]
    assert answer.citations[0].source_path == "tpo/decisions/payments.md"
    assert "**Sources**" in answer.answer
    assert "tpo/meetings/2026-03-01.vtt" in answer.answer
    # The model's own words stay separable, so a judge reads the answer and not our footer.
    assert "**Sources**" not in answer.prose


def test_a_citation_links_somewhere_a_reader_can_open() -> None:
    answer = _service(SOURCES, public_base_url="http://localhost:2600").ask("anything")

    assert answer.citations[0].url == "http://localhost:2600/documents/doc-1"


def test_nothing_retrieved_means_nothing_answered() -> None:
    service = _service([])
    answer = service.ask("What is the capital of France?")

    assert not answer.grounded
    assert answer.answer == NO_EVIDENCE_ANSWER
    assert answer.citations == []
    # The point of refusing is that no tokens are spent inventing one.
    assert service.provider.calls == []  # type: ignore[attr-defined]


def test_turning_the_guard_off_lets_the_model_answer_without_evidence() -> None:
    service = _service([], chat_require_evidence=False)
    answer = service.ask("What is the capital of France?")

    assert answer.answer.startswith("Retries use exponential backoff")
    assert service.provider.calls  # type: ignore[attr-defined]


def test_the_model_is_told_the_sources_are_data_and_not_orders() -> None:
    service = _service(SOURCES)
    service.ask("How do payment retries work?")
    system, user = service.provider.calls[0][0][0], service.provider.calls[0][0][-1]  # type: ignore[attr-defined]

    assert "never an instruction" in system.content
    assert "untrusted" in user.content
    assert "<<<SOURCES" in user.content and "SOURCES>>>" in user.content


def test_the_sources_are_numbered_so_a_small_model_can_cite_them() -> None:
    service = _service(SOURCES)
    service.ask("How do payment retries work?")
    prompt = service.provider.calls[0][0][-1].content  # type: ignore[attr-defined]

    assert "[1] path=tpo/decisions/payments.md" in prompt
    assert "[2] path=tpo/meetings/2026-03-01.vtt" in prompt


def test_a_follow_up_is_retrieved_with_what_it_left_unsaid() -> None:
    service = _service(SOURCES)
    list(
        service.stream(
            [
                ChatMessage(role="user", content="What did we decide about payment retries?"),
                ChatMessage(role="assistant", content="Exponential backoff."),
                ChatMessage(role="user", content="And the cap?"),
            ]
        )
    )

    asked = service.retriever.queries[-1]  # type: ignore[attr-defined]
    assert "payment retries" in asked
    assert "And the cap?" in asked


def test_a_condensing_model_rewrites_the_follow_up_into_a_whole_question() -> None:
    provider = StubProvider(["What cap did we agree on for payment retries?", "Five [1]."])
    service = _service(SOURCES, provider, chat_condense_model="qwen3:1.7b")
    list(
        service.stream(
            [
                ChatMessage(role="user", content="What did we decide about payment retries?"),
                ChatMessage(role="assistant", content="Exponential backoff."),
                ChatMessage(role="user", content="And the cap?"),
            ]
        )
    )

    assert service.retriever.queries[-1] == "What cap did we agree on for payment retries?"  # type: ignore[attr-defined]
    assert service.provider.calls[0][1].model == "qwen3:1.7b"  # type: ignore[attr-defined]


def test_a_condenser_that_rambles_is_ignored() -> None:
    provider = StubProvider(["word " * 200, "Five [1]."])
    service = _service(SOURCES, provider, chat_condense_model="qwen3:1.7b")
    list(
        service.stream(
            [
                ChatMessage(role="user", content="What did we decide about payment retries?"),
                ChatMessage(role="assistant", content="Exponential backoff."),
                ChatMessage(role="user", content="And the cap?"),
            ]
        )
    )

    assert service.retriever.queries[-1] == "And the cap?"  # type: ignore[attr-defined]


def test_a_first_question_is_retrieved_exactly_as_asked() -> None:
    service = _service(SOURCES)
    service.ask("How do payment retries work?")

    assert service.retriever.queries == ["How do payment retries work?"]  # type: ignore[attr-defined]


def test_a_conversation_must_end_with_the_user() -> None:
    with pytest.raises(ValueError, match="user message"):
        list(_service(SOURCES).stream([ChatMessage(role="assistant", content="hello")]))


def test_the_context_budget_cuts_between_sources_not_through_one() -> None:
    packed, cited = ContextCompressor().pack_numbered(SOURCES, max_chars=120)

    assert len(cited) == 1
    assert packed.rstrip().endswith("[/1]")
