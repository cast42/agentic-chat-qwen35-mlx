from __future__ import annotations

import argparse
import asyncio
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from pydantic_ai import PromptedOutput
from pydantic_ai.settings import ModelSettings

from rag_agent.agent import DEFAULT_MODEL, build_agent, build_planning_agent
from rag_agent.context import RunContext
from rag_agent.deps import RagDeps
from rag_agent.tools.search import QMD_HELP, QmdCommand, run_qmd_tool

DEFAULT_MAX_TOKENS = 65_536
_MAX_TOOL_STEPS = 4
_MAX_TOOL_OUTPUT_CHARS = 8_000
_MAX_TOOL_PLANNING_TOKENS = 2_048
_TOOL_OUTPUT_TRUNCATION_MARKER = "\n\n...[truncated]"
_THINK_START_TAG = "<think>"
_THINK_END_TAG = "</think>"


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


def _shorten(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return f"{text[:max_chars]}{_TOOL_OUTPUT_TRUNCATION_MARKER}"


def run_qmd_action(
    ctx: RunContext[RagDeps],
    command: QmdCommand,
    argument: str,
    line_limit: int | None = None,
) -> _QmdToolResult:
    """Run qmd against the notes collection and return the tool output."""
    output = run_qmd_tool(ctx, command=command, argument=argument, line_limit=line_limit)
    return _QmdToolResult(
        command=command,
        argument=argument,
        line_limit=line_limit,
        output=_shorten(output, _MAX_TOOL_OUTPUT_CHARS),
    )


def finish_tooling(reason: str = "") -> _ToolingComplete:
    """Finish retrieval and continue with final answer generation."""
    return _ToolingComplete(reason=reason.strip())


def _render_command_line(observation: _QmdToolResult) -> str:
    command = f"qmd {observation.command} {observation.argument!r}"
    if observation.line_limit is not None:
        command += f" -l {observation.line_limit}"
    command += " --collection notes"
    return command


def _build_tool_planning_prompt(question: str, observations: list[_QmdToolResult]) -> str:
    if not observations:
        return (
            "Plan retrieval for the question below.\n"
            "Call `run_qmd_action` to gather evidence, or `finish_tooling` when evidence is sufficient.\n"
            "Prefer as few tool calls as needed.\n\n"
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

    return (
        "Continue retrieval planning for the question below.\n"
        "If current evidence is enough, call `finish_tooling`.\n"
        "Otherwise call `run_qmd_action` exactly once for the next best retrieval step.\n\n"
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
                [run_qmd_action, finish_tooling],
                name="retrieval_step",
                description="Run one qmd command or finish retrieval.",
            ),
            instructions=(
                "This run is for retrieval planning only. "
                "Do not answer the user question directly. "
                "Choose either `run_qmd_action` or `finish_tooling`."
            ),
            model_settings=ModelSettings(
                extra_body={"max_tokens": min(max_tokens, _MAX_TOOL_PLANNING_TOKENS)}
            ),
        )

        step_output = cast(ToolingStep, planning_result.output)

        if isinstance(step_output, _ToolingComplete):
            break

        observations.append(step_output)
        planning_prompt = _build_tool_planning_prompt(question, observations)

    return observations


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
        help=f"Maximum generated tokens (default: {DEFAULT_MAX_TOKENS} or RAG_MAX_TOKENS)",
    )
    return parser


async def _run_stream(
    question: str,
    deps: RagDeps,
    model: str,
    max_tokens: int,
) -> int:
    observations = await _collect_tool_observations(question, deps, model, max_tokens)

    agent = build_agent(model=model)
    prompt = _build_final_prompt(question, observations)
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
