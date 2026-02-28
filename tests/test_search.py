from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import subprocess
from typing import cast

from rag_agent.context import RunContext
from rag_agent.deps import RagDeps
from rag_agent.tools.search import citations_for_hits, rg_search, semantic_search


@dataclass
class FakeContext:
    deps: RagDeps


def test_rg_search_runs_command_and_parses_hits(monkeypatch, tmp_path: Path) -> None:
    notes = tmp_path / "notes_repo"
    notes.mkdir()

    calls: list[list[str]] = []

    def fake_run(command, **kwargs):  # type: ignore[no-untyped-def]
        calls.append(command)
        return subprocess.CompletedProcess(
            args=command,
            returncode=0,
            stdout="topic.md:3:agentic systems\n",
            stderr="",
        )

    monkeypatch.setattr("rag_agent.tools.search.subprocess.run", fake_run)
    ctx = cast(RunContext[RagDeps], FakeContext(deps=RagDeps(notes_path=notes)))
    hits = rg_search(ctx, query="agentic")

    assert calls[0][:2] == ["rg", "-n"]
    assert hits[0].citation == "topic.md:3"
    assert hits[0].snippet == "agentic systems"


def test_semantic_search_parses_qmd_output(monkeypatch, tmp_path: Path) -> None:
    notes = tmp_path / "notes_repo"
    notes.mkdir()

    def fake_run(command, **kwargs):  # type: ignore[no-untyped-def]
        return subprocess.CompletedProcess(
            args=command,
            returncode=0,
            stdout="concepts/agents.md:8:Tool-driven retrieval\n",
            stderr="",
        )

    monkeypatch.setattr("rag_agent.tools.search.subprocess.run", fake_run)
    ctx = cast(RunContext[RagDeps], FakeContext(deps=RagDeps(notes_path=notes)))
    hits = semantic_search(ctx, query="retrieval")

    assert hits[0].citation == "concepts/agents.md:8"
    assert hits[0].snippet == "Tool-driven retrieval"


def test_citations_for_hits_deduplicates(monkeypatch, tmp_path: Path) -> None:
    notes = tmp_path / "notes_repo"
    notes.mkdir()

    def fake_run(command, **kwargs):  # type: ignore[no-untyped-def]
        return subprocess.CompletedProcess(
            args=command,
            returncode=0,
            stdout="topic.md:2:alpha\ntopic.md:2:alpha\n",
            stderr="",
        )

    monkeypatch.setattr("rag_agent.tools.search.subprocess.run", fake_run)
    ctx = cast(RunContext[RagDeps], FakeContext(deps=RagDeps(notes_path=notes)))
    hits = rg_search(ctx, query="alpha")
    assert citations_for_hits(hits) == ["topic.md:2"]
