from datetime import UTC, datetime

from product_memory.models import ChunkResult
from product_memory.retrieval.compressor import ContextCompressor


def chunk(identifier: str, document: str, score: float, content: str) -> ChunkResult:
    return ChunkResult(
        id=identifier,
        document_id=document,
        document_title=document,
        source_path=f"{document}.md",
        chunk_index=0,
        content=content,
        start_char=0,
        end_char=len(content),
        effective_at=datetime(2026, 1, 1, tzinfo=UTC),
        semantic_score=score,
        lexical_score=0,
        recency_score=1,
        score=score,
    )


def test_compressor_deduplicates_and_respects_budget() -> None:
    chunks = [
        chunk("1", "a", 0.9, "Same text"),
        chunk("2", "a", 0.8, "Same   text"),
        chunk("3", "b", 0.7, "Different text"),
        chunk(
            "4",
            "c",
            0.6,
            "Same text with a long repeated product context one two three four five six seven eight",
        ),
        chunk(
            "5",
            "c",
            0.5,
            "Same text with a long repeated product context one two three four five six seven nine",
        ),
    ]
    packed = ContextCompressor().pack(chunks, max_chars=1000)
    assert "Same text" in packed
    assert "Same   text" not in packed
    assert "Different text" in packed
    assert "six seven eight" in packed
    assert "six seven nine" not in packed
    assert len(packed) <= 1000
