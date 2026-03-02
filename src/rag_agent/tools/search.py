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
_WORD_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]*")
_QUERY_STOPWORDS = {
    "a",
    "about",
    "an",
    "and",
    "are",
    "can",
    "did",
    "do",
    "does",
    "for",
    "how",
    "i",
    "in",
    "is",
    "it",
    "me",
    "of",
    "on",
    "or",
    "say",
    "tell",
    "the",
    "to",
    "us",
    "what",
    "with",
    "you",
}
_MAX_FALLBACK_TERMS = 4


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


def _rg_command(query: str) -> list[str]:
    return ["rg", "-n", "--no-heading", "--color", "never", "-i", query, "."]


def _run_rg_query(notes_path: Path, query: str, max_results: int) -> list[SearchHit]:
    result = _run_search_command(_rg_command(query), notes_path)
    return _parse_hits(result.stdout.splitlines(), notes_path, _RG_PATTERN, max_results)


def _keyword_fallback_queries(query: str) -> list[str]:
    queries: list[str] = []
    seen: set[str] = set()

    for token in _WORD_PATTERN.findall(query):
        keyword = token.lower()
        if len(keyword) < 3 or keyword in _QUERY_STOPWORDS or keyword in seen:
            continue
        seen.add(keyword)
        queries.append(keyword)
        if len(queries) >= _MAX_FALLBACK_TERMS:
            break

    return queries


def rg_search(ctx: RunContext[RagDeps], query: str, max_results: int = 8) -> list[SearchHit]:
    if max_results < 1:
        raise ValueError("max_results must be positive.")
    if not query.strip():
        return []

    hits = _run_rg_query(ctx.deps.notes_path, query, max_results)
    if hits:
        return hits

    fallback_terms = _keyword_fallback_queries(query)
    if not fallback_terms:
        return []

    deduped_hits: list[SearchHit] = []
    seen: set[str] = set()
    for term in fallback_terms:
        for hit in _run_rg_query(ctx.deps.notes_path, term, max_results):
            citation = hit.citation
            if citation in seen:
                continue
            seen.add(citation)
            deduped_hits.append(hit)
            if len(deduped_hits) >= max_results:
                return deduped_hits

    return deduped_hits


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
