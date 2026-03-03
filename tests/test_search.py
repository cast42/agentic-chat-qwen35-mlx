from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import subprocess
from typing import cast

import pytest

from rag_agent.context import RunContext
from rag_agent.deps import RagDeps
from rag_agent.models import SearchHit
from rag_agent.tools.search import (
    citations_for_hits,
    qmd_get,
    qmd_get_json,
    qmd_multi_get,
    qmd_query,
    qmd_search,
    run_qmd_tool,
)


@dataclass
class FakeContext:
    deps: RagDeps


def test_qmd_query_runs_command(monkeypatch, tmp_path: Path) -> None:
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

    output = qmd_query(ctx, question="agentic systems")

    assert calls == [["qmd", "query", "agentic systems", "--collection", "notes"]]
    assert output == "topic.md:3:agentic systems"


def test_qmd_search_runs_command(monkeypatch, tmp_path: Path) -> None:
    notes = tmp_path / "notes_repo"
    notes.mkdir()

    calls: list[list[str]] = []

    def fake_run(command, **kwargs):  # type: ignore[no-untyped-def]
        calls.append(command)
        return subprocess.CompletedProcess(
            args=command,
            returncode=0,
            stdout="topic.md:11:bm25 match\n",
            stderr="",
        )

    monkeypatch.setattr("rag_agent.tools.search.subprocess.run", fake_run)
    ctx = cast(RunContext[RagDeps], FakeContext(deps=RagDeps(notes_path=notes)))

    output = qmd_search(ctx, keywords="agentic")

    assert calls == [["qmd", "search", "agentic", "--collection", "notes"]]
    assert output == "topic.md:11:bm25 match"


def test_qmd_multi_get_supports_line_limit(monkeypatch, tmp_path: Path) -> None:
    notes = tmp_path / "notes_repo"
    notes.mkdir()

    calls: list[list[str]] = []

    def fake_run(command, **kwargs):  # type: ignore[no-untyped-def]
        calls.append(command)
        return subprocess.CompletedProcess(args=command, returncode=0, stdout="ok", stderr="")

    monkeypatch.setattr("rag_agent.tools.search.subprocess.run", fake_run)
    ctx = cast(RunContext[RagDeps], FakeContext(deps=RagDeps(notes_path=notes)))

    output = qmd_multi_get(ctx, target="journals/2026-*.md", line_limit=40)

    assert output == "ok"
    assert calls == [
        ["qmd", "multi-get", "journals/2026-*.md", "-l", "40", "--collection", "notes"]
    ]


def test_run_qmd_tool_normalizes_qmd_uris_for_multi_get(monkeypatch, tmp_path: Path) -> None:
    notes = tmp_path / "notes_repo"
    notes.mkdir()

    calls: list[list[str]] = []

    def fake_run(command, **kwargs):  # type: ignore[no-untyped-def]
        calls.append(command)
        return subprocess.CompletedProcess(args=command, returncode=0, stdout="ok", stderr="")

    monkeypatch.setattr("rag_agent.tools.search.subprocess.run", fake_run)
    ctx = cast(RunContext[RagDeps], FakeContext(deps=RagDeps(notes_path=notes)))

    output = run_qmd_tool(
        ctx,
        command="multi-get",
        argument=(
            "qmd://notes/topics/design.md:15 "
            "qmd://notes/topics/ive.md "
            "[qmd://notes/topics/jobs.md:3]"
        ),
    )

    assert output == "ok"
    assert calls == [
        [
            "qmd",
            "multi-get",
            "topics/design.md,topics/ive.md,topics/jobs.md",
            "--collection",
            "notes",
        ]
    ]


def test_run_qmd_tool_blocks_line_limit_for_non_multi_get(tmp_path: Path) -> None:
    notes = tmp_path / "notes_repo"
    notes.mkdir()
    ctx = cast(RunContext[RagDeps], FakeContext(deps=RagDeps(notes_path=notes)))

    with pytest.raises(ValueError, match="line_limit"):
        run_qmd_tool(ctx, command="query", argument="topic", line_limit=10)


def test_qmd_get_raises_on_command_failure(monkeypatch, tmp_path: Path) -> None:
    notes = tmp_path / "notes_repo"
    notes.mkdir()

    def fake_run(command, **kwargs):  # type: ignore[no-untyped-def]
        return subprocess.CompletedProcess(args=command, returncode=2, stdout="", stderr="boom")

    monkeypatch.setattr("rag_agent.tools.search.subprocess.run", fake_run)
    ctx = cast(RunContext[RagDeps], FakeContext(deps=RagDeps(notes_path=notes)))

    with pytest.raises(RuntimeError, match="boom"):
        qmd_get(ctx, docid="#abc123")


def test_qmd_get_json_runs_get_with_json(monkeypatch, tmp_path: Path) -> None:
    notes = tmp_path / "notes_repo"
    notes.mkdir()

    calls: list[list[str]] = []

    def fake_run(command, **kwargs):  # type: ignore[no-untyped-def]
        calls.append(command)
        return subprocess.CompletedProcess(args=command, returncode=0, stdout="{}", stderr="")

    monkeypatch.setattr("rag_agent.tools.search.subprocess.run", fake_run)
    ctx = cast(RunContext[RagDeps], FakeContext(deps=RagDeps(notes_path=notes)))
    output = qmd_get_json(ctx, docid="#abc123")

    assert output == "{}"
    assert calls == [["qmd", "get", "#abc123", "--json", "--collection", "notes"]]


def test_citations_for_hits_deduplicates() -> None:
    hits = [
        SearchHit(path=Path("topic.md"), line=2, snippet="alpha"),
        SearchHit(path=Path("topic.md"), line=2, snippet="alpha again"),
    ]

    assert citations_for_hits(hits) == ["topic.md:2"]
