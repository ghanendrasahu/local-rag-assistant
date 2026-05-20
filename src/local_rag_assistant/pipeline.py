from __future__ import annotations

from pathlib import Path

from local_rag_assistant.chunking import chunk_documents
from local_rag_assistant.config import RagConfig
from local_rag_assistant.index import TfidfIndex
from local_rag_assistant.ingest import load_documents


def build_index(config: RagConfig) -> Path:
    docs = load_documents(config.data_dir)
    if not docs:
        raise ValueError(f"No TXT/PDF files found under: {config.data_dir}")
    chunks = chunk_documents(docs, config.chunk_size, config.chunk_overlap)
    if not chunks:
        raise ValueError("No chunks created. Check your input documents.")

    idx = TfidfIndex.build(chunks)
    config.ensure_paths()
    index_file = config.index_dir / "tfidf_index.pkl"
    idx.save(index_file)
    return index_file


def ask_question(query: str, config: RagConfig) -> dict:
    index_file = config.index_dir / "tfidf_index.pkl"
    if not index_file.exists():
        raise FileNotFoundError(
            f"Index not found at {index_file}. Run indexing first."
        )

    idx = TfidfIndex.load(index_file)
    hits = idx.search(query=query, top_k=config.top_k)
    answer = _build_answer(query, hits)
    return {
        "query": query,
        "answer": answer,
        "sources": [
            {"source": h.chunk.source, "score": round(h.score, 4)}
            for h in hits
        ],
    }


def _build_answer(query: str, hits: list) -> str:
    if not hits:
        return (
            "I could not find relevant context in the indexed documents. "
            "Try rephrasing the question or indexing more sources."
        )
    best = hits[0].chunk.content
    return f"Best match for '{query}':\n\n{best}"
