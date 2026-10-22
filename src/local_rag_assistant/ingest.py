from __future__ import annotations

from pathlib import Path

from pypdf import PdfReader

from local_rag_assistant.models import Chunk


def load_documents(data_dir: Path) -> list[Chunk]:
    """Load raw documents from TXT and PDF files."""
    docs: list[Chunk] = []
    supported = [*data_dir.rglob("*.txt"), *data_dir.rglob("*.pdf")]
    for path in sorted(supported):
        text = _read_file(path)
        if not text.strip():
            continue
        docs.append(
            Chunk(
                id=str(path),
                source=str(path),
                content=_normalize_whitespace(text),
            )
        )
    return docs


def _read_file(path: Path) -> str:
    if path.suffix.lower() == ".txt":
        return path.read_text(encoding="utf-8", errors="ignore")
    if path.suffix.lower() == ".pdf":
        return _read_pdf(path)
    return ""


def _read_pdf(path: Path) -> str:
    reader = PdfReader(str(path))
    pages = [page.extract_text() or "" for page in reader.pages]
    return "\n".join(pages)


def _normalize_whitespace(text: str) -> str:
    return " ".join(text.split())
