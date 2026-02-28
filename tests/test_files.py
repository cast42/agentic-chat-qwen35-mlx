from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import cast

import pytest

from rag_agent.context import RunContext
from rag_agent.deps import RagDeps
from rag_agent.tools.files import TRUNCATION_MARKER, read_file


@dataclass
class FakeContext:
    deps: RagDeps


def test_read_file_reads_markdown(tmp_path: Path) -> None:
    notes = tmp_path / "notes_repo"
    notes.mkdir()
    target = notes / "topic.md"
    target.write_text("hello world", encoding="utf-8")

    ctx = cast(RunContext[RagDeps], FakeContext(deps=RagDeps(notes_path=notes)))
    assert read_file(ctx, "topic.md") == "hello world"


def test_read_file_truncates_large_content(tmp_path: Path) -> None:
    notes = tmp_path / "notes_repo"
    notes.mkdir()
    target = notes / "topic.md"
    target.write_text("abcdef", encoding="utf-8")

    ctx = cast(RunContext[RagDeps], FakeContext(deps=RagDeps(notes_path=notes)))
    assert read_file(ctx, "topic.md", max_chars=3) == f"abc{TRUNCATION_MARKER}"


def test_read_file_blocks_directory_escape(tmp_path: Path) -> None:
    notes = tmp_path / "notes_repo"
    notes.mkdir()
    external = tmp_path / "external.md"
    external.write_text("hidden", encoding="utf-8")

    ctx = cast(RunContext[RagDeps], FakeContext(deps=RagDeps(notes_path=notes)))
    with pytest.raises(ValueError):
        read_file(ctx, "../external.md")


def test_read_file_requires_markdown(tmp_path: Path) -> None:
    notes = tmp_path / "notes_repo"
    notes.mkdir()
    target = notes / "topic.txt"
    target.write_text("not markdown", encoding="utf-8")

    ctx = cast(RunContext[RagDeps], FakeContext(deps=RagDeps(notes_path=notes)))
    with pytest.raises(ValueError):
        read_file(ctx, "topic.txt")
