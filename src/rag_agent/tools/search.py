from __future__ import annotations

from pathlib import Path
import re
import subprocess
from typing import Iterable

from rag_agent.context import RunContext
from rag_agent.deps import RagDeps
from rag_agent.models import SearchHit

_RG_PATTERN = re.compile(r"^(?P<path>.+?):(?P<line>\d+):(?P<snippet>.*)$")
_QMD_PATTERN = re.compile(
    r"^(?P<path>[^:\s]+\.m(?:d|arkdown))(?::(?P<line>\d+))?[:\-\s]*(?P<snippet>.*)$",
    re.IGNORECASE,
)


def _normalize_path(notes_path: Path, raw_path: str) -> Path:
    notes_root = notes_path.resolve()
    candidate = Path(raw_path)
    if not candidate.is_absolute():
        candidate = (notes_root / candidate).resolve()
    else:
        candidate = candidate.resolve()

    try:
        return candidate.relative_to(notes_root)
    except ValueError:
        return Path(raw_path)


def _parse_hits(
    lines: Iterable[str],
    notes_path: Path,
    pattern: re.Pattern[str],
    max_results: int,
) -> list[SearchHit]:
    hits: list[SearchHit] = []
    for raw_line in lines:
        match = pattern.match(raw_line.strip())
        if not match:
            continue

        line_text = match.groupdict().get("line") or "1"
        try:
            line = int(line_text)
        except ValueError:
            continue

        hit = SearchHit(
            path=_normalize_path(notes_path, match.group("path")),
            line=max(1, line),
            snippet=match.group("snippet").strip(),
        )
        hits.append(hit)
        if len(hits) >= max_results:
            break

    return hits


def _run_search_command(command: list[str], notes_path: Path) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(
        command,
        cwd=notes_path,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode in (0, 1):
        return result

    stderr = result.stderr.strip()
    raise RuntimeError(stderr or f"Command failed with exit code {result.returncode}.")


def rg_search(ctx: RunContext[RagDeps], query: str, max_results: int = 8) -> list[SearchHit]:
    if max_results < 1:
        raise ValueError("max_results must be positive.")
    if not query.strip():
        return []

    command = ["rg", "-n", "--no-heading", "--color", "never", query, "."]
    result = _run_search_command(command, ctx.deps.notes_path)
    return _parse_hits(result.stdout.splitlines(), ctx.deps.notes_path, _RG_PATTERN, max_results)


def semantic_search(ctx: RunContext[RagDeps], query: str, max_results: int = 8) -> list[SearchHit]:
    if max_results < 1:
        raise ValueError("max_results must be positive.")
    if not query.strip():
        return []

    command = ["qmd", "search", query]
    result = _run_search_command(command, ctx.deps.notes_path)

    # qmd output format can vary by version; parse best-effort and return available markdown hits.
    hits = _parse_hits(result.stdout.splitlines(), ctx.deps.notes_path, _QMD_PATTERN, max_results)
    if hits:
        return hits
    return _parse_hits(result.stdout.splitlines(), ctx.deps.notes_path, _RG_PATTERN, max_results)


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
