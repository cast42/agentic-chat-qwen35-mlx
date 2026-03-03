from __future__ import annotations

import re
import subprocess
from pathlib import Path
from typing import Literal

from rag_agent.context import RunContext
from rag_agent.deps import RagDeps
from rag_agent.models import SearchHit

QmdCommand = Literal["query", "search", "get", "multi-get"]
_QMD_COLLECTION = "notes"
QMD_HELP = """qmd query "question" --collection notes              # Auto-expand + rerank
qmd query $'lex: X\\nvec: Y' --collection notes  --json     # Structured
qmd query $'expand: question' --collection notes --json     # Explicit expand
qmd search "keywords" --collection notes --json             # BM25 only (no LLM)
qmd get "#abc123" --collection notes --json                 # By docid
qmd multi-get "journals/2026-*.md" -l 40 --collection notes  # Batch pull snippets by glob
qmd multi-get notes/foo.md,notes/bar.md --collection notes   # Comma-separated list, preserves order"""


def _normalize_multi_get_token(token: str) -> str | None:
    cleaned = token.strip().strip("'\"")
    if not cleaned:
        return None

    cleaned = cleaned.lstrip("[(").rstrip("])")
    if cleaned.startswith("qmd://notes/"):
        cleaned = cleaned.removeprefix("qmd://notes/")
    else:
        cleaned = cleaned.removeprefix("qmd://")
    cleaned = cleaned.rstrip(".,;")

    if re.fullmatch(r".+\.md:\d+", cleaned):
        cleaned = cleaned.rsplit(":", maxsplit=1)[0]

    return cleaned or None


def _normalize_multi_get_argument(argument: str) -> str:
    tokens = [token for token in re.split(r"[\s,]+", argument.strip()) if token]

    if len(tokens) <= 1 and "qmd://" not in argument and ".md:" not in argument:
        return argument

    normalized_tokens = [
        normalized
        for token in tokens
        if (normalized := _normalize_multi_get_token(token)) is not None
    ]
    if not normalized_tokens:
        return argument

    return ",".join(normalized_tokens)


def _run_qmd_command(command: list[str], notes_path: Path) -> str:
    command_with_collection = [*command, "--collection", _QMD_COLLECTION]
    result = subprocess.run(
        ["qmd", *command_with_collection],
        cwd=notes_path,
        capture_output=True,
        text=True,
        check=False,
    )

    if result.returncode == 0:
        return result.stdout.strip()
    stderr = result.stderr.strip()
    raise RuntimeError(stderr or f"qmd command failed with exit code {result.returncode}.")


def run_qmd_tool(
    ctx: RunContext[RagDeps],
    command: QmdCommand,
    argument: str,
    line_limit: int | None = None,
    json_output: bool = False,
) -> str:
    cleaned_argument = argument.strip()
    if not cleaned_argument:
        raise ValueError("argument must not be empty.")
    if command == "multi-get":
        cleaned_argument = _normalize_multi_get_argument(cleaned_argument)

    command_parts: list[str] = [command, cleaned_argument]
    if line_limit is not None:
        if command != "multi-get":
            raise ValueError("line_limit is only supported for `multi-get`.")
        if line_limit < 1:
            raise ValueError("line_limit must be positive.")
        command_parts.extend(["-l", str(line_limit)])
    if json_output:
        if command != "get":
            raise ValueError("json_output is only supported for `get`.")
        command_parts.append("--json")

    return _run_qmd_command(command_parts, ctx.deps.notes_path)


def qmd_query(ctx: RunContext[RagDeps], question: str) -> str:
    return run_qmd_tool(ctx, command="query", argument=question)


def qmd_search(ctx: RunContext[RagDeps], keywords: str) -> str:
    return run_qmd_tool(ctx, command="search", argument=keywords)


def qmd_get(ctx: RunContext[RagDeps], docid: str) -> str:
    return run_qmd_tool(ctx, command="get", argument=docid)


def qmd_get_json(ctx: RunContext[RagDeps], docid: str) -> str:
    return run_qmd_tool(ctx, command="get", argument=docid, json_output=True)


def qmd_multi_get(ctx: RunContext[RagDeps], target: str, line_limit: int = 40) -> str:
    return run_qmd_tool(ctx, command="multi-get", argument=target, line_limit=line_limit)


def citations_for_hits(hits: list[SearchHit]) -> list[str]:
    citations: list[str] = []
    seen: set[str] = set()
    for hit in hits:
        citation = hit.citation
        if citation in seen:
            continue
        seen.add(citation)
        citations.append(citation)
    return citations
