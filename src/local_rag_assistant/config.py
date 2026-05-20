from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(slots=True)
class RagConfig:
    data_dir: Path = Path("data")
    index_dir: Path = Path("artifacts/index")
    chunk_size: int = 700
    chunk_overlap: int = 120
    top_k: int = 4

    def ensure_paths(self) -> None:
        self.index_dir.mkdir(parents=True, exist_ok=True)
