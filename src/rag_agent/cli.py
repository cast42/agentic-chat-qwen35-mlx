from __future__ import annotations

import argparse
import asyncio
import os
from pathlib import Path
import sys

from rag_agent.agent import (
    DEFAULT_API_KEY,
    DEFAULT_BASE_URL,
    DEFAULT_MODEL,
    build_agent,
)
from rag_agent.deps import RagDeps


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
        help="Model name exposed by the local OpenAI-compatible server",
    )
    parser.add_argument(
        "--base-url",
        default=os.environ.get("RAG_OPENAI_BASE_URL", DEFAULT_BASE_URL),
        help="OpenAI-compatible base URL (default: http://127.0.0.1:8080/v1)",
    )
    parser.add_argument(
        "--api-key",
        default=os.environ.get("RAG_OPENAI_API_KEY", DEFAULT_API_KEY),
        help="API key for the OpenAI-compatible endpoint",
    )
    return parser


async def _run_stream(
    question: str,
    deps: RagDeps,
    model: str,
    base_url: str,
    api_key: str,
) -> int:
    agent = build_agent(model=model, base_url=base_url, api_key=api_key)
    async with agent.run_stream(question, deps=deps) as result:
        async for chunk in result.stream_text(delta=True):
            print(chunk, end="", flush=True)
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
    try:
        return asyncio.run(
            _run_stream(
                args.question,
                deps=deps,
                model=args.model,
                base_url=args.base_url,
                api_key=args.api_key,
            )
        )
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
