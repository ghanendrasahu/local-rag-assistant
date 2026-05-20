from __future__ import annotations

from local_rag_assistant.models import Chunk


def chunk_documents(
    docs: list[Chunk],
    chunk_size: int,
    chunk_overlap: int,
) -> list[Chunk]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be > 0")
    if chunk_overlap < 0 or chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be >= 0 and < chunk_size")

    output: list[Chunk] = []
    for doc in docs:
        parts = _chunk_text(doc.content, chunk_size, chunk_overlap)
        for idx, part in enumerate(parts):
            output.append(
                Chunk(
                    id=f"{doc.id}::chunk-{idx}",
                    source=doc.source,
                    content=part,
                )
            )
    return output


def _chunk_text(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    words = text.split()
    if not words:
        return []

    chunks: list[str] = []
    current: list[str] = []
    current_len = 0
    for word in words:
        extra = len(word) + (1 if current else 0)
        if current and current_len + extra > chunk_size:
            chunks.append(" ".join(current))
            overlap_words = _tail_words_for_overlap(current, chunk_overlap)
            current = overlap_words.copy()
            current_len = len(" ".join(current))
        current.append(word)
        current_len += extra

    if current:
        chunks.append(" ".join(current))
    return chunks


def _tail_words_for_overlap(words: list[str], max_chars: int) -> list[str]:
    if max_chars == 0:
        return []
    out: list[str] = []
    total = 0
    for word in reversed(words):
        add_len = len(word) + (1 if out else 0)
        if total + add_len > max_chars:
            break
        out.append(word)
        total += add_len
    out.reverse()
    return out
