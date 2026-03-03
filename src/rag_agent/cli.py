from __future__ import annotations

import argparse
import asyncio
import os
import re
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from pydantic_ai import PromptedOutput
from pydantic_ai.messages import ModelMessage
from pydantic_ai.settings import ModelSettings
from rich.console import Console
from rich.markdown import Markdown

from rag_agent.agent import DEFAULT_MODEL, build_agent, build_planning_agent
from rag_agent.context import RunContext
from rag_agent.deps import RagDeps
from rag_agent.tools.search import QMD_HELP, QmdCommand, run_qmd_tool

DEFAULT_MAX_TOKENS = 65_536
_MAX_TOOL_STEPS = 4
_MAX_TOOL_OUTPUT_CHARS = 8_000
_MAX_TOOL_PLANNING_TOKENS = 2_048
_TOOL_OUTPUT_TRUNCATION_MARKER = "\n\n...[truncated]"
_DEFAULT_MULTI_GET_LINE_LIMIT = 80
_FALLBACK_MULTI_GET_LINE_LIMIT = 120
_MAX_FALLBACK_NOTE_PATHS = 3
_MAX_FALLBACK_DOC_IDS = 3
_THINK_START_TAG = "<think>"
_THINK_END_TAG = "</think>"
_GITHUB_NOTES_BASE_URL = "https://github.com/cast42/notes/blob/main"
_SOURCE_PREFIXES = {
    "x",
    "youtube",
    "blog",
    "article",
    "wsj",
    "tweet",
    "podcast",
    "video",
    "post",
    "readme",
}
_TITLE_ANCHORS = {
    "designing",
    "building",
    "using",
    "learning",
    "understanding",
    "lessons",
    "guide",
    "introduction",
    "overview",
    "principles",
    "ideas",
    "thinking",
}
_QMD_NOTE_REF_PATTERN = re.compile(
    r"qmd://(?P<path>notes/[A-Za-z0-9._\-/]+\.md)(?::(?P<line>\d+))?"
)
_BRACKET_NOTE_REF_PATTERN = re.compile(
    r"\[(?P<path>(?:notes/)?[A-Za-z0-9._-]+(?:/[A-Za-z0-9._-]+)+\.md)(?::(?P<line>\d+))?\](?!\()"
)
_BARE_NOTE_REF_PATTERN = re.compile(
    r"(?P<prefix>^|[\s(\"'])(?P<path>(?:notes/)?[A-Za-z0-9._-]+(?:/[A-Za-z0-9._-]+)+\.md)(?::(?P<line>\d+))?"
)
_DOC_ID_PATTERN = re.compile(r"#[A-Za-z0-9]{6,}")
_NOTE_PATH_CANDIDATE_PATTERN = re.compile(
    r"(?:qmd://notes/|notes/)?[A-Za-z0-9._-]+(?:/[A-Za-z0-9._-]+)+\.md(?::\d+|#L\d+)?"
)


@dataclass(frozen=True, slots=True)
class _QmdToolResult:
    command: QmdCommand
    argument: str
    line_limit: int | None
    output: str


@dataclass(frozen=True, slots=True)
class _ToolingComplete:
    reason: str


ToolingStep = _QmdToolResult | _ToolingComplete


@dataclass(frozen=True, slots=True)
class _DepsOnlyRunContext:
    deps: RagDeps


@dataclass(slots=True)
class _StreamChunkRouter:
    inside_think: bool = False
    seen_think_tag: bool = False
    pending: str = ""

    def consume(self, chunk: str) -> tuple[str, str]:
        self.pending += chunk
        thinking_parts: list[str] = []
        answer_parts: list[str] = []

        while self.pending:
            if self.inside_think:
                end_idx = self.pending.find(_THINK_END_TAG)
                if end_idx < 0:
                    keep = len(_THINK_END_TAG) - 1
                    if len(self.pending) <= keep:
                        return "".join(thinking_parts), "".join(answer_parts)
                    flush_len = len(self.pending) - keep
                    thinking_parts.append(self.pending[:flush_len])
                    self.pending = self.pending[flush_len:]
                    return "".join(thinking_parts), "".join(answer_parts)
                thinking_parts.append(self.pending[:end_idx])
                self.pending = self.pending[end_idx + len(_THINK_END_TAG) :]
                self.inside_think = False
                continue

            # Some Qwen runs may emit `</think>` without an opening `<think>`.
            # In that case, everything before `</think>` is treated as thinking.
            close_only_idx = self.pending.find(_THINK_END_TAG) if not self.seen_think_tag else -1
            start_idx = self.pending.find(_THINK_START_TAG)
            if close_only_idx >= 0 and (start_idx < 0 or close_only_idx < start_idx):
                thinking_parts.append(self.pending[:close_only_idx])
                self.pending = self.pending[close_only_idx + len(_THINK_END_TAG) :]
                self.seen_think_tag = True
                continue

            if start_idx < 0:
                keep = len(_THINK_START_TAG) - 1
                if len(self.pending) <= keep:
                    return "".join(thinking_parts), "".join(answer_parts)
                flush_len = len(self.pending) - keep
                answer_parts.append(self.pending[:flush_len])
                self.pending = self.pending[flush_len:]
                return "".join(thinking_parts), "".join(answer_parts)

            answer_parts.append(self.pending[:start_idx])
            self.pending = self.pending[start_idx + len(_THINK_START_TAG) :]
            self.inside_think = True
            self.seen_think_tag = True

        return "".join(thinking_parts), "".join(answer_parts)

    def flush(self) -> tuple[str, str]:
        if self.inside_think:
            remaining_thinking = self.pending
            self.pending = ""
            return remaining_thinking, ""
        remaining_answer = self.pending
        self.pending = ""
        return "", remaining_answer


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _shorten(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return f"{text[:max_chars]}{_TOOL_OUTPUT_TRUNCATION_MARKER}"


def _note_url(path: str, line: str | None) -> str:
    repo_path = path.removeprefix("notes/")
    base_url = f"{_GITHUB_NOTES_BASE_URL}/{repo_path}"
    if line is None:
        return base_url
    return f"{base_url}#L{line}"


def _title_from_note_path(path: str) -> str:
    tokens = Path(path).stem.split("-")
    if (
        len(tokens) >= 4
        and re.fullmatch(r"\d{4}", tokens[0])
        and re.fullmatch(r"\d{2}", tokens[1])
        and re.fullmatch(r"\d{2}", tokens[2])
    ):
        tokens = tokens[3:]
    if tokens and tokens[0] in _SOURCE_PREFIXES:
        tokens = tokens[1:]
    for index, token in enumerate(tokens):
        if token in _TITLE_ANCHORS:
            tokens = tokens[index:]
            break
    if not tokens:
        tokens = Path(path).stem.split("-")
    return " ".join(token.capitalize() for token in tokens)


def _linkify_note_references(markdown_text: str) -> str:
    def qmd_replacer(match: re.Match[str]) -> str:
        path = match.group("path")
        line = match.group("line")
        label = Path(path).stem
        return f"[{label}]({_note_url(path, line)})"

    result = _QMD_NOTE_REF_PATTERN.sub(qmd_replacer, markdown_text)

    def bracketed_replacer(match: re.Match[str]) -> str:
        path = match.group("path")
        line = match.group("line")
        if path.startswith("notes/"):
            label = path if line is None else f"{path}:{line}"
        else:
            label = _title_from_note_path(path)
        return f"[{label}]({_note_url(path, line)})"

    result = _BRACKET_NOTE_REF_PATTERN.sub(bracketed_replacer, result)

    def bare_replacer(match: re.Match[str]) -> str:
        prefix = match.group("prefix")
        path = match.group("path")
        line = match.group("line")
        label = path if line is None else f"{path}:{line}"
        return f"{prefix}[{label}]({_note_url(path, line)})"

    return _BARE_NOTE_REF_PATTERN.sub(bare_replacer, result)


def _is_quit_command(value: str) -> bool:
    normalized = value.strip().lower()
    return normalized in {"quit", "exit"}


def run_qmd_action(
    ctx: RunContext[RagDeps],
    command: QmdCommand,
    argument: str,
    line_limit: int | None = None,
) -> _QmdToolResult:
    """Run qmd against the notes collection and return the tool output."""
    if command == "multi-get":
        effective_line_limit = line_limit or _DEFAULT_MULTI_GET_LINE_LIMIT
    else:
        effective_line_limit = None
    output = run_qmd_tool(
        ctx,
        command=command,
        argument=argument,
        line_limit=effective_line_limit,
    )
    return _QmdToolResult(
        command=command,
        argument=argument,
        line_limit=effective_line_limit,
        output=_shorten(output, _MAX_TOOL_OUTPUT_CHARS),
    )


def run_qmd_get_json(ctx: RunContext[RagDeps], doc_id: str) -> _QmdToolResult:
    """Run `qmd get <doc_id> --json` against the notes collection."""
    output = run_qmd_tool(ctx, command="get", argument=doc_id, json_output=True)
    return _QmdToolResult(
        command="get",
        argument=doc_id,
        line_limit=None,
        output=_shorten(output, _MAX_TOOL_OUTPUT_CHARS),
    )


def finish_tooling(reason: str = "") -> _ToolingComplete:
    """Finish retrieval and continue with final answer generation."""
    return _ToolingComplete(reason=reason.strip())


def _has_note_content_observations(observations: Sequence[_QmdToolResult]) -> bool:
    return any(observation.command in {"get", "multi-get"} for observation in observations)


def _is_substantive_qmd_output(output: str) -> bool:
    lines = [line.strip() for line in output.splitlines() if line.strip()]
    if not lines:
        return False

    metadata_like_lines = [
        line
        for line in lines
        if (
            line.startswith("@@")
            or line.startswith("diff ")
            or line.startswith("index ")
            or line.startswith("---")
            or line.startswith("+++")
            or line.endswith(".md")
            or line.startswith("qmd ")
        )
    ]
    substantive_lines = [line for line in lines if line not in metadata_like_lines]
    if not substantive_lines:
        return False

    return len(" ".join(substantive_lines)) >= 40


def _has_substantive_note_content_observations(observations: Sequence[_QmdToolResult]) -> bool:
    return any(
        observation.command in {"get", "multi-get"}
        and _is_substantive_qmd_output(observation.output)
        for observation in observations
    )


def _normalize_note_path_candidate(candidate: str) -> str | None:
    normalized = candidate.strip().strip("'\"")
    normalized = normalized.lstrip("[(").rstrip("])")

    if "blob/main/" in normalized:
        normalized = normalized.split("blob/main/", maxsplit=1)[1]

    if normalized.startswith("qmd://notes/"):
        normalized = normalized.removeprefix("qmd://notes/")
    normalized = normalized.removeprefix("notes/")

    if re.fullmatch(r".+\.md:\d+", normalized):
        normalized = normalized.rsplit(":", maxsplit=1)[0]
    if ".md#L" in normalized:
        normalized = normalized.split("#L", maxsplit=1)[0]

    if not normalized.endswith(".md"):
        return None
    return normalized


def _extract_candidate_doc_ids(observations: Sequence[_QmdToolResult]) -> list[str]:
    seen: set[str] = set()
    doc_ids: list[str] = []

    for observation in observations:
        for source in (observation.argument, observation.output):
            for match in _DOC_ID_PATTERN.finditer(source):
                doc_id = match.group(0)
                if doc_id in seen:
                    continue
                seen.add(doc_id)
                doc_ids.append(doc_id)

    return doc_ids


def _extract_candidate_note_paths(observations: Sequence[_QmdToolResult]) -> list[str]:
    seen: set[str] = set()
    paths: list[str] = []

    for observation in observations:
        for source in (observation.argument, observation.output):
            for match in _NOTE_PATH_CANDIDATE_PATTERN.finditer(source):
                normalized = _normalize_note_path_candidate(match.group(0))
                if normalized is None or normalized in seen:
                    continue
                seen.add(normalized)
                paths.append(normalized)

    return paths


def _hydrate_note_content_with_fallback(
    observations: list[_QmdToolResult],
    deps: RagDeps,
) -> list[_QmdToolResult]:
    hydrated_observations = [*observations]
    run_ctx = cast(RunContext[RagDeps], _DepsOnlyRunContext(deps=deps))

    if not _has_substantive_note_content_observations(hydrated_observations):
        for doc_id in _extract_candidate_doc_ids(hydrated_observations)[:_MAX_FALLBACK_DOC_IDS]:
            try:
                output = run_qmd_tool(run_ctx, command="get", argument=doc_id, json_output=True)
            except RuntimeError:
                continue
            hydrated_observations.append(
                _QmdToolResult(
                    command="get",
                    argument=doc_id,
                    line_limit=None,
                    output=_shorten(output, _MAX_TOOL_OUTPUT_CHARS),
                )
            )

    candidate_paths = _extract_candidate_note_paths(hydrated_observations)
    if candidate_paths:
        target = ",".join(candidate_paths[:_MAX_FALLBACK_NOTE_PATHS])
        try:
            output = run_qmd_tool(
                run_ctx,
                command="multi-get",
                argument=target,
                line_limit=_FALLBACK_MULTI_GET_LINE_LIMIT,
            )
        except RuntimeError:
            return hydrated_observations
        hydrated_observations.append(
            _QmdToolResult(
                command="multi-get",
                argument=target,
                line_limit=_FALLBACK_MULTI_GET_LINE_LIMIT,
                output=_shorten(output, _MAX_TOOL_OUTPUT_CHARS),
            )
        )

    return hydrated_observations


def _render_command_line(observation: _QmdToolResult) -> str:
    command = f"qmd {observation.command} {observation.argument!r}"
    if observation.line_limit is not None:
        command += f" -l {observation.line_limit}"
    command += " --collection notes"
    return command


def _build_tool_planning_prompt(question: str, observations: list[_QmdToolResult]) -> str:
    has_note_content = _has_note_content_observations(observations)
    has_substantive_note_content = _has_substantive_note_content_observations(observations)

    if not observations:
        return (
            "Plan retrieval for the question below.\n"
            "Use exactly this sequence:\n"
            "1) Identify relevant notes with `run_qmd_action` (`query` or `search`).\n"
            "2) Retrieve note content with `run_qmd_get_json` or `run_qmd_action` with `multi-get`.\n"
            "   For `multi-get`, pass comma-separated `notes/...` paths or globs, never `qmd://` URIs.\n"
            f"   `multi-get` should include snippets (line_limit around {_DEFAULT_MULTI_GET_LINE_LIMIT} or higher).\n"
            "3) Only then call `finish_tooling`.\n"
            "Do not finish before content retrieval.\n\n"
            f"Question:\n{question}\n\n"
            "qmd help:\n"
            "```ascii\n"
            f"{QMD_HELP}\n"
            "```"
        )

    rendered_observations = "\n\n".join(
        (
            f"Observation {index}:\n"
            f"Command: {_render_command_line(observation)}\n"
            f"Output:\n{observation.output}"
        )
        for index, observation in enumerate(observations, start=1)
    )

    if not has_note_content:
        next_step_instruction = (
            "Continue retrieval planning for the question below.\n"
            "You already identified notes, but have not retrieved note content yet.\n"
            "Call exactly one content retrieval step now: `run_qmd_get_json` or "
            '`run_qmd_action` with `command="multi-get"`.\n'
            "For `multi-get`, use comma-separated plain paths (for example `notes/a.md,notes/b.md`).\n"
            "Do not call `finish_tooling` yet.\n\n"
        )
    elif not has_substantive_note_content:
        next_step_instruction = (
            "Continue retrieval planning for the question below.\n"
            "Content retrieval so far is metadata-only or too shallow.\n"
            "Run exactly one more content retrieval step with richer content.\n"
            "Prefer `run_qmd_get_json` for specific doc IDs, or `multi-get` with a higher `line_limit`.\n"
            "Do not call `finish_tooling` yet.\n\n"
        )
    else:
        next_step_instruction = (
            "Continue retrieval planning for the question below.\n"
            "You already have note content.\n"
            "If evidence is enough, call `finish_tooling`.\n"
            "Otherwise call exactly one more retrieval step.\n\n"
        )

    return (
        f"{next_step_instruction}"
        f"Question:\n{question}\n\n"
        f"Current observations:\n{rendered_observations}"
    )


def _build_final_prompt(question: str, observations: list[_QmdToolResult]) -> str:
    if not observations:
        return question

    rendered_observations = "\n\n".join(
        (f"Command: {_render_command_line(observation)}\nOutput:\n{observation.output}")
        for observation in observations
    )

    return (
        "Use the qmd retrieval context below to answer the user question.\n"
        "Base claims on that evidence and include citations in `path:line` form when present.\n\n"
        "Format:\n"
        "- Put reasoning in `<think>...</think>`.\n"
        "- Put the final answer after `</think>` using Markdown.\n\n"
        "Answer rules:\n"
        "- Write the answer directly from note content in the retrieval context.\n"
        "- Do not tell the user to read notes or retrieve documents themselves.\n"
        "- Do not include a 'limitations' or 'would you like me to retrieve' section.\n"
        "- If note content is still missing, say the evidence is insufficient.\n\n"
        "- Convert `qmd://notes/...` refs to `[name](https://github.com/cast42/notes/blob/main/...)` links.\n"
        "- Convert `notes/...` refs to `[notes/...](https://github.com/cast42/notes/blob/main/...)` links.\n"
        "- Convert `[topics/...:line]` style refs to human-title links to `cast42/notes`.\n\n"
        f"Question:\n{question}\n\n"
        f"qmd retrieval context:\n{rendered_observations}"
    )


async def _collect_tool_observations(
    question: str,
    deps: RagDeps,
    model: str,
    max_tokens: int,
) -> list[_QmdToolResult]:
    agent = build_planning_agent(model=model)
    observations: list[_QmdToolResult] = []
    planning_prompt = _build_tool_planning_prompt(question, observations)

    for _ in range(_MAX_TOOL_STEPS):
        planning_result = await agent.run(
            planning_prompt,
            deps=deps,
            output_type=PromptedOutput(
                [run_qmd_action, run_qmd_get_json, finish_tooling],
                name="retrieval_step",
                description="Run one qmd command (`get --json` supported) or finish retrieval.",
            ),
            instructions=(
                "This run is for retrieval planning only. "
                "Do not answer the user question directly. "
                "Choose either `run_qmd_action`, `run_qmd_get_json`, or `finish_tooling`."
            ),
            model_settings=ModelSettings(
                extra_body={"max_tokens": min(max_tokens, _MAX_TOOL_PLANNING_TOKENS)}
            ),
        )

        step_output = cast(ToolingStep, planning_result.output)

        if isinstance(step_output, _ToolingComplete):
            if not _has_substantive_note_content_observations(observations):
                planning_prompt = (
                    f"{_build_tool_planning_prompt(question, observations)}\n\n"
                    "You attempted to finish too early. Retrieve substantive note content first."
                )
                continue
            break

        observations.append(step_output)
        planning_prompt = _build_tool_planning_prompt(question, observations)

    return _hydrate_note_content_with_fallback(observations, deps)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Local agentic RAG chatbot")
    parser.add_argument("question", nargs="?", help="Initial question to ask the chatbot")
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
        help=f"Maximum generated tokens (default: {DEFAULT_MAX_TOKENS} or RAG_MAX_TOKENS)",
    )
    return parser


async def _run_stream(
    question: str,
    deps: RagDeps,
    model: str,
    max_tokens: int,
    message_history: Sequence[ModelMessage] | None = None,
) -> list[ModelMessage]:
    observations = await _collect_tool_observations(question, deps, model, max_tokens)

    agent = build_agent(model=model)
    prompt = _build_final_prompt(question, observations)
    router = _StreamChunkRouter()
    answer_parts: list[str] = []
    fallback_parts: list[str] = []
    thinking_started = False
    console = Console()

    async with agent.run_stream(
        prompt,
        deps=deps,
        message_history=message_history,
        model_settings=ModelSettings(extra_body={"max_tokens": max_tokens}),
    ) as result:
        async for chunk in result.stream_text(delta=True):
            thinking_chunk, answer_chunk = router.consume(chunk)
            if thinking_chunk:
                if not thinking_started:
                    print("Thinking:\n", end="", flush=True)
                    thinking_started = True
                print(thinking_chunk, end="", flush=True)
            if answer_chunk:
                if router.seen_think_tag:
                    answer_parts.append(answer_chunk)
                else:
                    if not thinking_started:
                        print("Thinking:\n", end="", flush=True)
                        thinking_started = True
                    print(answer_chunk, end="", flush=True)
                    fallback_parts.append(answer_chunk)

    thinking_tail, answer_tail = router.flush()
    if thinking_tail:
        if not thinking_started:
            print("Thinking:\n", end="", flush=True)
            thinking_started = True
        print(thinking_tail, end="", flush=True)
    if thinking_started:
        print("\n", flush=True)

    if router.seen_think_tag:
        if answer_tail:
            answer_parts.append(answer_tail)
        answer = "".join(answer_parts).replace("<|im_end|>", "").strip()
    else:
        if answer_tail:
            fallback_parts.append(answer_tail)
        answer = "".join(fallback_parts).replace("<|im_end|>", "").strip()

    if answer:
        answer = _linkify_note_references(answer)
        console.print("Answer:\n")
        console.print(Markdown(answer))
    return result.all_messages()


async def _run_chat_loop(
    initial_question: str | None,
    deps: RagDeps,
    model: str,
    max_tokens: int,
) -> int:
    message_history: list[ModelMessage] = []
    pending_question = initial_question

    while True:
        if pending_question is None:
            try:
                pending_question = input("\nYou: ")
            except EOFError, KeyboardInterrupt:
                print()
                return 0

        question = pending_question.strip()
        pending_question = None

        if not question:
            continue
        if _is_quit_command(question):
            return 0

        message_history = await _run_stream(
            question,
            deps=deps,
            model=model,
            max_tokens=max_tokens,
            message_history=message_history,
        )


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
            _run_chat_loop(
                args.question,
                deps=deps,
                model=args.model,
                max_tokens=args.max_tokens,
            )
        )
    except KeyboardInterrupt:
        print()
        return 0
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
