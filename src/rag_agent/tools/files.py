from __future__ import annotations

from pathlib import Path

from rag_agent.context import RunContext
from rag_agent.deps import RagDeps

TRUNCATION_MARKER = "\n\n...[truncated]"
ALLOWED_SUFFIXES = {".md", ".markdown"}


def _resolve_markdown_path(notes_path: Path, file_path: str) -> Path:
    base = notes_path.resolve()
    candidate = (base / file_path).resolve()

    if not candidate.is_relative_to(base):
        raise ValueError("File path must stay inside the notes repository.")
    if candidate.suffix.lower() not in ALLOWED_SUFFIXES:
        raise ValueError("Only markdown files can be read.")
    if not candidate.is_file():
        raise FileNotFoundError(candidate)

    return candidate


def read_file(ctx: RunContext[RagDeps], file_path: str, max_chars: int = 12_000) -> str:
    if max_chars < 1:
        raise ValueError("max_chars must be positive.")

    resolved = _resolve_markdown_path(ctx.deps.notes_path, file_path)
    contents = resolved.read_text(encoding="utf-8")
    if len(contents) <= max_chars:
        return contents
    return f"{contents[:max_chars]}{TRUNCATION_MARKER}"
