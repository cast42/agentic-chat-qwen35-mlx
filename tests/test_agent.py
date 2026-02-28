from __future__ import annotations

from pathlib import Path

from rag_agent.agent import render_citations
from rag_agent.models import SearchHit


def test_render_citations_formats_sources_block() -> None:
    hits = [
        SearchHit(path=Path("a.md"), line=3, snippet="alpha"),
        SearchHit(path=Path("b.md"), line=9, snippet="beta"),
    ]
    rendered = render_citations(hits)
    assert "Sources:" in rendered
    assert "- a.md:3" in rendered
    assert "- b.md:9" in rendered
