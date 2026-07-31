from __future__ import annotations

from dataclasses import dataclass

from langchain_text_splitters import RecursiveCharacterTextSplitter

from product_memory.settings import Settings


@dataclass(slots=True)
class TextChunk:
    index: int
    content: str
    start_char: int
    end_char: int
    approx_tokens: int


class DocumentChunker:
    def __init__(self, settings: Settings):
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
            length_function=len,
            add_start_index=True,
            separators=["\n\n", "\n", ". ", "? ", "! ", "; ", ", ", " ", ""],
        )

    def split(self, text: str) -> list[TextChunk]:
        documents = self.splitter.create_documents([text])
        chunks: list[TextChunk] = []
        for index, document in enumerate(documents):
            content = document.page_content.strip()
            if not content:
                continue
            start = int(document.metadata.get("start_index", 0))
            chunks.append(
                TextChunk(
                    index=index,
                    content=content,
                    start_char=start,
                    end_char=start + len(content),
                    approx_tokens=max(1, round(len(content) / 4)),
                )
            )
        return chunks
