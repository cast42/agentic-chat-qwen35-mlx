from __future__ import annotations

import argparse
import asyncio
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from pydantic_ai.settings import ModelSettings

from rag_agent.agent import (
    DEFAULT_MODEL,
    build_agent,
    render_citations,
)
from rag_agent.context import RunContext
from rag_agent.deps import RagDeps
from rag_agent.models import SearchHit
from rag_agent.tools import (
    read_file,
    rg_search,
    semantic_search,
)

DEFAULT_MAX_TOKENS = 65_536
DEFAULT_MAX_RESULTS = 8
DEFAULT_MAX_FILES = 4
DEFAULT_FILE_CHARS = 3_000
_THINK_START_TAG = "<think>"
_THINK_END_TAG = "</think>"


@dataclass(frozen=True, slots=True)
class _ContextProxy:
    deps: RagDeps


@dataclass(slots=True)
class _StreamChunkSanitizer:
    inside_think: bool = False
    pending: str = ""

    def sanitize(self, chunk: str) -> str:
        self.pending += chunk
        visible_parts: list[str] = []

        while self.pending:
            if self.inside_think:
                end_idx = self.pending.find(_THINK_END_TAG)
                if end_idx < 0:
                    keep = len(_THINK_END_TAG) - 1
                    if len(self.pending) > keep:
                        self.pending = self.pending[-keep:]
                    return "".join(visible_parts)
                self.pending = self.pending[end_idx + len(_THINK_END_TAG) :]
                self.inside_think = False
                continue

            start_idx = self.pending.find(_THINK_START_TAG)
            if start_idx < 0:
                keep = len(_THINK_START_TAG) - 1
                if len(self.pending) <= keep:
                    return "".join(visible_parts)
                flush_len = len(self.pending) - keep
                visible_parts.append(self.pending[:flush_len])
                self.pending = self.pending[flush_len:]
                return "".join(visible_parts)

            visible_parts.append(self.pending[:start_idx])
            self.pending = self.pending[start_idx + len(_THINK_START_TAG) :]
            self.inside_think = True

        return "".join(visible_parts)

    def flush(self) -> str:
        if self.inside_think:
            self.pending = ""
            return ""
        remaining = self.pending
        self.pending = ""
        return remaining


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _dedupe_hits(hits: list[SearchHit]) -> list[SearchHit]:
    deduped: list[SearchHit] = []
    seen: set[str] = set()
    for hit in hits:
        citation = hit.citation
        if citation in seen:
            continue
        seen.add(citation)
        deduped.append(hit)
    return deduped


def _build_retrieval_context(question: str, deps: RagDeps) -> tuple[str, list[SearchHit]]:
    ctx = cast(RunContext[RagDeps], _ContextProxy(deps=deps))
    combined_hits: list[SearchHit] = []

    for search_fn in (semantic_search, rg_search):
        try:
            combined_hits.extend(search_fn(ctx, query=question, max_results=DEFAULT_MAX_RESULTS))
        except Exception:
            continue

    hits = _dedupe_hits(combined_hits)[:DEFAULT_MAX_RESULTS]
    if not hits:
        return "", []

    hit_lines = "\n".join(f"- {hit.citation}: {hit.snippet}" for hit in hits)

    file_sections: list[str] = []
    seen_paths: set[str] = set()
    for hit in hits:
        path_text = hit.path.as_posix()
        if path_text in seen_paths:
            continue
        seen_paths.add(path_text)
        try:
            excerpt = read_file(ctx, file_path=path_text, max_chars=DEFAULT_FILE_CHARS)
        except Exception:
            continue
        file_sections.append(f"### {path_text}\n{excerpt}")
        if len(file_sections) >= DEFAULT_MAX_FILES:
            break

    context_parts = [f"Repository search hits:\n{hit_lines}"]
    if file_sections:
        context_parts.append("Repository file excerpts:\n" + "\n\n".join(file_sections))
    return "\n\n".join(context_parts), hits


def _augment_question(question: str, retrieval_context: str) -> str:
    if not retrieval_context:
        return question
    return (
        "Use the retrieved repository context below to answer the user question.\n\n"
        f"Question:\n{question}\n\n"
        f"{retrieval_context}\n\n"
        "Cite evidence with `path:line` when possible.\n"
        "Return only the final answer and do not include reasoning steps."
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Local agentic RAG chatbot")
    parser.add_argument("question", help="Question to ask the chatbot")
    parser.add_argument(
        "--notes-path",
        default=os.environ.get("RAG_NOTES_PATH", "notes_repo"),
        help="Path to the notes repository (default: notes_repo or RAG_NOTES_PATH)",
    )
    parser.add_argument(
        "--model",
        default=os.environ.get("RAG_MODEL", DEFAULT_MODEL),
        help=f"MLX model id (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=_env_int("RAG_MAX_TOKENS", DEFAULT_MAX_TOKENS),
        help="Maximum generated tokens (default: 2048 or RAG_MAX_TOKENS)",
    )
    return parser


async def _run_stream(
    question: str,
    deps: RagDeps,
    model: str,
    max_tokens: int,
) -> int:
    agent = build_agent(model=model)
    retrieval_context, hits = _build_retrieval_context(question, deps)
    prompt = _augment_question(question, retrieval_context)
    sanitizer = _StreamChunkSanitizer()

    async with agent.run_stream(
        prompt,
        deps=deps,
        model_settings=ModelSettings(extra_body={"max_tokens": max_tokens}),
    ) as result:
        async for chunk in result.stream_text(delta=True):
            visible = sanitizer.sanitize(chunk)
            if visible:
                print(visible, end="", flush=True)

    tail = sanitizer.flush()
    if tail:
        print(tail, end="", flush=True)
    print()
    citations = render_citations(hits)
    if citations:
        print(citations, end="")
    return 0


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()

    notes_path = Path(args.notes_path).resolve()
    if not notes_path.exists():
        print(f"Notes path does not exist: {notes_path}", file=sys.stderr)
        return 2

    deps = RagDeps(notes_path=notes_path)
    if args.max_tokens < 1:
        print("--max-tokens must be positive.", file=sys.stderr)
        return 2

    try:
        return asyncio.run(
            _run_stream(
                args.question,
                deps=deps,
                model=args.model,
                max_tokens=args.max_tokens,
            )
        )
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
