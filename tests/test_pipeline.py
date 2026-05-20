from pathlib import Path

from local_rag_assistant.config import RagConfig
from local_rag_assistant.pipeline import ask_question, build_index


def test_build_and_query_index(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True)
    (data_dir / "doc1.txt").write_text(
        "Python is a readable programming language.", encoding="utf-8"
    )
    (data_dir / "doc2.txt").write_text(
        "Machine learning has supervised and unsupervised methods.",
        encoding="utf-8",
    )

    cfg = RagConfig(data_dir=data_dir, index_dir=tmp_path / "index")
    index_path = build_index(cfg)
    assert index_path.exists()

    result = ask_question("What is Python?", cfg)
    assert "Best match" in result["answer"]
    assert len(result["sources"]) >= 1
