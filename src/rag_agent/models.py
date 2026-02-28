from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class SearchHit:
    path: Path
    line: int
    snippet: str

    @property
    def citation(self) -> str:
        return f"{self.path}:{self.line}"
