from __future__ import annotations

import asyncio
from pathlib import Path
from typing import cast

import pytest

import rag_agent.agent as agent_module
from rag_agent.agent import DEFAULT_MODEL, SYSTEM_PROMPT, render_citations
from rag_agent.cli import (
    _QmdToolResult,
    _StreamChunkRouter,
    _build_final_prompt,
    _build_parser,
    _build_tool_planning_prompt,
    _extract_candidate_doc_ids,
    _extract_candidate_note_paths,
    _has_note_content_observations,
    _has_substantive_note_content_observations,
    _hydrate_note_content_with_fallback,
    _is_quit_command,
    _linkify_note_references,
    _run_chat_loop,
    run_qmd_action,
)
from rag_agent.context import RunContext
from rag_agent.deps import RagDeps
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


def test_default_model_matches_expected_qwen35_9b_variant() -> None:
    assert DEFAULT_MODEL == "mlx-community/Qwen3.5-9B-4bit"


def test_system_prompt_includes_qmd_help() -> None:
    assert "qmd query" in SYSTEM_PROMPT
    assert "qmd search" in SYSTEM_PROMPT
    assert "qmd get" in SYSTEM_PROMPT
    assert "--json" in SYSTEM_PROMPT
    assert "qmd multi-get" in SYSTEM_PROMPT


def test_cli_uses_default_model_when_env_not_set(monkeypatch) -> None:
    monkeypatch.delenv("RAG_MODEL", raising=False)
    args = _build_parser().parse_args(["hello"])
    assert args.model == DEFAULT_MODEL


def test_cli_accepts_missing_initial_question(monkeypatch) -> None:
    monkeypatch.delenv("RAG_MODEL", raising=False)
    args = _build_parser().parse_args([])
    assert args.question is None
    assert args.model == DEFAULT_MODEL


def test_is_quit_command_matches_expected_values() -> None:
    assert _is_quit_command("quit")
    assert _is_quit_command(" Exit ")
    assert not _is_quit_command("quit now")


def test_has_note_content_observations_detects_get_and_multi_get() -> None:
    assert not _has_note_content_observations(
        [_QmdToolResult(command="query", argument="topic", line_limit=None, output="...")]
    )
    assert _has_note_content_observations(
        [_QmdToolResult(command="get", argument="#abc123", line_limit=None, output="...")]
    )
    assert _has_note_content_observations(
        [_QmdToolResult(command="multi-get", argument="notes/foo.md", line_limit=40, output="...")]
    )


def test_has_substantive_note_content_observations_ignores_metadata_only_output() -> None:
    assert not _has_substantive_note_content_observations(
        [
            _QmdToolResult(
                command="multi-get",
                argument="notes/foo.md",
                line_limit=80,
                output="notes/foo.md\n@@ -4,4 @@ (3 before, 71 after)",
            )
        ]
    )
    assert _has_substantive_note_content_observations(
        [
            _QmdToolResult(
                command="multi-get",
                argument="notes/foo.md",
                line_limit=80,
                output="## Principles\nDesign starts with user needs and constraints.",
            )
        ]
    )


def test_build_tool_planning_prompt_requires_content_before_finish() -> None:
    prompt = _build_tool_planning_prompt("What about Jony Ive?", [])
    assert "Identify relevant notes" in prompt
    assert "Retrieve note content" in prompt
    assert "never `qmd://` URIs" in prompt
    assert "Do not finish before content retrieval" in prompt


def test_build_tool_planning_prompt_blocks_finish_until_content_loaded() -> None:
    observations = [
        _QmdToolResult(command="query", argument="Jony Ive", line_limit=None, output="...")
    ]
    prompt = _build_tool_planning_prompt("What about Jony Ive?", observations)
    assert "Do not call `finish_tooling` yet" in prompt
    assert "comma-separated plain paths" in prompt
    assert "`run_qmd_get_json`" in prompt


def test_build_tool_planning_prompt_requires_more_when_content_is_metadata_only() -> None:
    observations = [
        _QmdToolResult(
            command="multi-get",
            argument="notes/foo.md",
            line_limit=80,
            output="notes/foo.md\n@@ -4,4 @@ (3 before, 71 after)",
        )
    ]
    prompt = _build_tool_planning_prompt("What about Jony Ive?", observations)
    assert "metadata-only or too shallow" in prompt
    assert "Do not call `finish_tooling` yet" in prompt


def test_build_final_prompt_requires_answer_from_note_content() -> None:
    observations = [
        _QmdToolResult(command="get", argument="#abc123", line_limit=None, output="...")
    ]
    prompt = _build_final_prompt("Summarize this", observations)
    assert "Write the answer directly from note content" in prompt
    assert "Do not tell the user to read notes" in prompt


def test_extract_candidate_note_paths_normalizes_qmd_and_line_suffixes() -> None:
    observations = [
        _QmdToolResult(
            command="query",
            argument="qmd://notes/topics/foo.md:15",
            line_limit=None,
            output="see notes/topics/bar.md and https://github.com/cast42/notes/blob/main/topics/baz.md#L9",
        )
    ]
    assert _extract_candidate_note_paths(observations) == [
        "topics/foo.md",
        "topics/bar.md",
        "topics/baz.md",
    ]


def test_extract_candidate_doc_ids_deduplicates() -> None:
    observations = [
        _QmdToolResult(
            command="query",
            argument="about #abc123",
            line_limit=None,
            output="hit #abc123 and #def456",
        )
    ]
    assert _extract_candidate_doc_ids(observations) == ["#abc123", "#def456"]


def test_hydrate_note_content_with_fallback_uses_get_for_doc_ids(
    monkeypatch, tmp_path: Path
) -> None:
    calls: list[tuple[str, str, int | None, bool]] = []

    def fake_run_qmd_tool(
        _ctx: RunContext[RagDeps],
        *,
        command: str,
        argument: str,
        line_limit: int | None = None,
        json_output: bool = False,
    ) -> str:
        calls.append((command, argument, line_limit, json_output))
        return "## Content\nThis is substantive note content."

    monkeypatch.setattr("rag_agent.cli.run_qmd_tool", fake_run_qmd_tool)

    hydrated = _hydrate_note_content_with_fallback(
        observations=[
            _QmdToolResult(command="query", argument="#abc123", line_limit=None, output="...")
        ],
        deps=RagDeps(notes_path=tmp_path),
    )

    assert len(hydrated) == 2
    assert calls == [("get", "#abc123", None, True)]


def test_hydrate_note_content_with_fallback_fetches_candidate_paths(
    monkeypatch, tmp_path: Path
) -> None:
    calls: list[tuple[str, str, int | None, bool]] = []

    def fake_run_qmd_tool(
        _ctx: RunContext[RagDeps],
        *,
        command: str,
        argument: str,
        line_limit: int | None = None,
        json_output: bool = False,
    ) -> str:
        calls.append((command, argument, line_limit, json_output))
        return "## Snippet\nDesign starts with user needs."

    monkeypatch.setattr("rag_agent.cli.run_qmd_tool", fake_run_qmd_tool)

    hydrated = _hydrate_note_content_with_fallback(
        observations=[
            _QmdToolResult(
                command="query",
                argument="design",
                line_limit=None,
                output="notes/topics/designing-things-people-love/foo.md:15",
            )
        ],
        deps=RagDeps(notes_path=tmp_path),
    )

    assert len(hydrated) == 2
    assert calls == [("multi-get", "topics/designing-things-people-love/foo.md", 120, False)]


def test_run_chat_loop_appends_message_history_for_followups(monkeypatch, tmp_path: Path) -> None:
    calls: list[str] = []
    seen_history_lengths: list[int] = []

    async def fake_run_stream(  # type: ignore[no-untyped-def]
        question,
        deps,
        model,
        max_tokens,
        message_history=None,
    ):
        del deps, model, max_tokens
        calls.append(question)
        previous_history = list(message_history or [])
        seen_history_lengths.append(len(previous_history))
        return [*previous_history, object()]

    user_inputs = iter(["follow up", "quit"])

    def fake_input(_prompt: str) -> str:
        return next(user_inputs)

    monkeypatch.setattr("rag_agent.cli._run_stream", fake_run_stream)
    monkeypatch.setattr("builtins.input", fake_input)

    exit_code = asyncio.run(
        _run_chat_loop(
            "first question",
            deps=RagDeps(notes_path=tmp_path),
            model=DEFAULT_MODEL,
            max_tokens=64,
        )
    )

    assert exit_code == 0
    assert calls == ["first question", "follow up"]
    assert seen_history_lengths == [0, 1]


def test_stream_chunk_router_splits_thinking_and_answer() -> None:
    router = _StreamChunkRouter()
    chunks = ["Answer start ", "<th", "ink>", "internal", "</thi", "nk>", " answer end"]
    thinking = ""
    answer = ""
    for chunk in chunks:
        next_thinking, next_answer = router.consume(chunk)
        thinking += next_thinking
        answer += next_answer
    tail_thinking, tail_answer = router.flush()
    thinking += tail_thinking
    answer += tail_answer

    assert thinking == "internal"
    assert answer == "Answer start  answer end"


def test_stream_chunk_router_handles_close_tag_without_open_tag() -> None:
    router = _StreamChunkRouter()
    thinking, answer = router.consume("thought one\n</think>\nfinal answer")
    tail_thinking, tail_answer = router.flush()

    assert thinking + tail_thinking == "thought one\n"
    assert answer + tail_answer == "\nfinal answer"


def test_linkify_note_references_formats_qmd_uri() -> None:
    value = "ref qmd://notes/twil/2026-week-06-agentic-coding.md"
    linked = _linkify_note_references(value)
    assert (
        linked
        == "ref [2026-week-06-agentic-coding](https://github.com/cast42/notes/blob/main/twil/2026-week-06-agentic-coding.md)"
    )


def test_linkify_note_references_formats_plain_path() -> None:
    value = "ref notes/twil/2026-week-06-agentic-coding.md"
    linked = _linkify_note_references(value)
    assert (
        linked
        == "ref [notes/twil/2026-week-06-agentic-coding.md](https://github.com/cast42/notes/blob/main/twil/2026-week-06-agentic-coding.md)"
    )


def test_linkify_note_references_preserves_line_suffix_in_label() -> None:
    value = "ref notes/twil/2026-week-06-agentic-coding.md:26"
    linked = _linkify_note_references(value)
    assert (
        linked
        == "ref [notes/twil/2026-week-06-agentic-coding.md:26](https://github.com/cast42/notes/blob/main/twil/2026-week-06-agentic-coding.md#L26)"
    )


def test_linkify_note_references_formats_bracketed_topic_path_as_title() -> None:
    value = "[topics/designing-things-people-love/2026-02-02-x-jony-ive-designing-things-people-love.md:15]"
    linked = _linkify_note_references(value)
    assert (
        linked
        == "[Designing Things People Love](https://github.com/cast42/notes/blob/main/topics/designing-things-people-love/2026-02-02-x-jony-ive-designing-things-people-love.md#L15)"
    )


def test_build_local_mlx_model_falls_back_for_extra_weights(monkeypatch) -> None:
    calls: list[str] = []

    def fail_initial(_model: str) -> tuple[object, object]:
        raise ValueError("Received 333 parameters not in model")

    def relaxed_loader(model: str) -> tuple[object, object]:
        calls.append(model)
        return object(), object()

    def fake_outlines(_mlx_model: object, _tokenizer: object) -> object:
        return "outlines-model"

    monkeypatch.setattr(agent_module, "_load_mlx_components", fail_initial)
    monkeypatch.setattr(agent_module, "_load_mlx_components_relaxed", relaxed_loader)
    monkeypatch.setattr(agent_module, "_build_outlines_model", fake_outlines)

    built = agent_module._build_local_mlx_model(DEFAULT_MODEL)
    assert built == "outlines-model"
    assert calls == [DEFAULT_MODEL]


def test_build_local_mlx_model_raises_non_weight_errors(monkeypatch) -> None:
    def fail_with_unrelated_error(_model: str) -> tuple[object, object]:
        raise ValueError("boom")

    monkeypatch.setattr(agent_module, "_load_mlx_components", fail_with_unrelated_error)

    with pytest.raises(ValueError, match="boom"):
        agent_module._build_local_mlx_model(DEFAULT_MODEL)


def test_run_qmd_action_drops_line_limit_for_non_multi_get(monkeypatch) -> None:
    calls: list[tuple[str, str, int | None]] = []

    def fake_run_qmd_tool(
        _ctx: RunContext[RagDeps],
        *,
        command: str,
        argument: str,
        line_limit: int | None = None,
    ) -> str:
        calls.append((command, argument, line_limit))
        return "ok"

    monkeypatch.setattr("rag_agent.cli.run_qmd_tool", fake_run_qmd_tool)

    result = run_qmd_action(
        cast(RunContext[RagDeps], object()),
        command="get",
        argument="#abc123",
        line_limit=40,
    )

    assert calls == [("get", "#abc123", None)]
    assert result.line_limit is None


def test_run_qmd_action_defaults_line_limit_for_multi_get(monkeypatch) -> None:
    calls: list[tuple[str, str, int | None]] = []

    def fake_run_qmd_tool(
        _ctx: RunContext[RagDeps],
        *,
        command: str,
        argument: str,
        line_limit: int | None = None,
    ) -> str:
        calls.append((command, argument, line_limit))
        return "ok"

    monkeypatch.setattr("rag_agent.cli.run_qmd_tool", fake_run_qmd_tool)

    result = run_qmd_action(
        cast(RunContext[RagDeps], object()),
        command="multi-get",
        argument="notes/foo.md,notes/bar.md",
    )

    assert calls == [("multi-get", "notes/foo.md,notes/bar.md", 80)]
    assert result.line_limit == 80
