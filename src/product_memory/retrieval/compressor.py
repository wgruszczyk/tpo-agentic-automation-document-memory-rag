from __future__ import annotations

import hashlib
import re
from collections import defaultdict

from product_memory.models import ChunkResult

_WHITESPACE = re.compile(r"\s+")


class ContextCompressor:
    """Deterministic, source-preserving context packing for an external agent.

    Full chunks remain available in the structured response. The context pack removes
    near-duplicates, prefers document diversity, and respects a strict character budget.
    """

    def pack(self, chunks: list[ChunkResult], max_chars: int) -> str:
        selected = self._diverse_deduplicated(chunks)
        blocks: list[str] = []
        used = 0
        for chunk in selected:
            header = (
                f"[SOURCE document_id={chunk.document_id} chunk_id={chunk.id} "
                f"path={chunk.source_path} date={chunk.effective_at.date().isoformat()} "
                f"score={chunk.score:.4f}]\n"
            )
            block = header + chunk.content.strip() + "\n[/SOURCE]\n"
            if used + len(block) > max_chars:
                remaining = max_chars - used
                if remaining > len(header) + 200:
                    blocks.append(block[:remaining].rstrip() + "\n[TRUNCATED]\n")
                break
            blocks.append(block)
            used += len(block)
        return "\n".join(blocks)

    def pack_numbered(
        self, chunks: list[ChunkResult], max_chars: int
    ) -> tuple[str, list[ChunkResult]]:
        """The same selection as pack, but labelled [1..n] and paired with what each label means.

        A local model asked to cite document_id=8f3a... will get it wrong often enough to matter,
        and a wrong citation is worse than none. Small integers it can copy, and the caller keeps
        the mapping back to the real source.
        """
        selected = self._diverse_deduplicated(chunks)
        blocks: list[str] = []
        cited: list[ChunkResult] = []
        used = 0
        for chunk in selected:
            marker = len(cited) + 1
            header = (
                f"[{marker}] path={chunk.source_path} "
                f"date={chunk.effective_at.date().isoformat()}\n"
            )
            block = header + chunk.content.strip() + f"\n[/{marker}]\n"
            if used + len(block) > max_chars:
                # Half a source reads as a complete one and gets cited as if it were, so the
                # budget cuts between sources. Unless nothing fits at all, which beats no context.
                if cited:
                    break
                block = block[: max(max_chars, len(header) + 200)]
            blocks.append(block)
            cited.append(chunk)
            used += len(block)
        return "\n".join(blocks), cited

    @classmethod
    def _diverse_deduplicated(cls, chunks: list[ChunkResult]) -> list[ChunkResult]:
        unique: list[ChunkResult] = []
        unique_candidates: list[ChunkResult] = []
        seen_hashes: set[str] = set()
        seen_shingles: list[set[str]] = []
        by_document: dict[str, list[ChunkResult]] = defaultdict(list)

        for chunk in chunks:
            normalized = _WHITESPACE.sub(" ", chunk.content.strip().lower())
            digest = hashlib.sha1(normalized.encode("utf-8")).hexdigest()
            shingles = cls._shingles(normalized)
            if digest in seen_hashes or any(
                cls._jaccard(shingles, previous) >= 0.80 for previous in seen_shingles
            ):
                continue
            seen_hashes.add(digest)
            seen_shingles.append(shingles)
            unique_candidates.append(chunk)
            by_document[chunk.document_id].append(chunk)

        # First pass: best chunk from every document. Second pass: remaining score order.
        first_pass = [items[0] for items in by_document.values()]
        first_pass.sort(key=lambda item: item.score, reverse=True)
        unique.extend(first_pass)

        used_ids = {item.id for item in unique}
        remaining = [chunk for chunk in unique_candidates if chunk.id not in used_ids]
        remaining.sort(key=lambda item: item.score, reverse=True)
        unique.extend(remaining)
        return unique

    @staticmethod
    def _shingles(text: str, size: int = 5) -> set[str]:
        words = text.split()
        if len(words) < size:
            return {text}
        return {" ".join(words[index : index + size]) for index in range(len(words) - size + 1)}

    @staticmethod
    def _jaccard(left: set[str], right: set[str]) -> float:
        if not left or not right:
            return 0.0
        return len(left & right) / len(left | right)
