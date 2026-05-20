from local_rag_assistant.chunking import chunk_documents
from local_rag_assistant.models import Chunk


def test_chunk_documents_produces_overlapping_chunks() -> None:
    docs = [Chunk(id="1", source="a.txt", content=" ".join(["python"] * 80))]
    chunks = chunk_documents(docs, chunk_size=80, chunk_overlap=20)
    assert len(chunks) > 1
    assert chunks[0].source == "a.txt"


def test_chunk_documents_validation() -> None:
    docs = [Chunk(id="1", source="a.txt", content="hello world")]
    try:
        chunk_documents(docs, chunk_size=10, chunk_overlap=10)
        assert False, "Expected ValueError"
    except ValueError:
        assert True
