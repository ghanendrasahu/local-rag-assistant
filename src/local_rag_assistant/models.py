from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class Chunk:
    id: str
    source: str
    content: str
