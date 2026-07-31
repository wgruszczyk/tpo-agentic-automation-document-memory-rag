from product_memory.ingestion.chunker import DocumentChunker
from product_memory.settings import Settings


def test_chunker_preserves_content_and_offsets() -> None:
    text = ("Paragraph one. " * 40) + "\n\n" + ("Paragraph two. " * 40)
    chunks = DocumentChunker(Settings(chunk_size=300, chunk_overlap=50)).split(text)
    assert len(chunks) > 1
    assert all(chunk.content for chunk in chunks)
    assert all(chunk.end_char > chunk.start_char for chunk in chunks)
