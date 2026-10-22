from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from local_rag_assistant.models import Chunk


@dataclass(slots=True)
class SearchHit:
    score: float
    chunk: Chunk


class TfidfIndex:
    def __init__(
        self,
        vectorizer: TfidfVectorizer,
        matrix: np.ndarray,
        chunks: list[Chunk],
    ) -> None:
        self.vectorizer = vectorizer
        self.matrix = matrix
        self.chunks = chunks

    @classmethod
    def build(cls, chunks: list[Chunk]) -> TfidfIndex:
        if not chunks:
            raise ValueError("Cannot build index with zero chunks.")
        vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words="english")
        matrix = vectorizer.fit_transform([c.content for c in chunks]).toarray()
        return cls(vectorizer=vectorizer, matrix=matrix, chunks=chunks)

    def search(self, query: str, top_k: int) -> list[SearchHit]:
        if top_k <= 0:
            raise ValueError("top_k must be > 0")
        q_vec = self.vectorizer.transform([query]).toarray()[0]
        q_norm = np.linalg.norm(q_vec)
        if q_norm == 0:
            return []
        m_norm = np.linalg.norm(self.matrix, axis=1)
        denom = np.maximum(m_norm * q_norm, 1e-12)
        scores = (self.matrix @ q_vec) / denom
        idx = np.argsort(scores)[::-1][:top_k]
        return [
            SearchHit(score=float(scores[i]), chunk=self.chunks[int(i)])
            for i in idx
            if scores[i] > 0
        ]

    def save(self, path: Path) -> None:
        payload = {
            "vectorizer": self.vectorizer,
            "matrix": self.matrix,
            "chunks": self.chunks,
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(pickle.dumps(payload))

    @classmethod
    def load(cls, path: Path) -> TfidfIndex:
        payload = pickle.loads(path.read_bytes())
        return cls(
            vectorizer=payload["vectorizer"],
            matrix=payload["matrix"],
            chunks=payload["chunks"],
        )
