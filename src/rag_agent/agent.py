from __future__ import annotations

from pydantic_ai import Agent, RunContext
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider

from rag_agent.deps import RagDeps
from rag_agent.models import SearchHit
from rag_agent.tools import read_file, rg_search, semantic_search
from rag_agent.tools.search import citations_for_hits

DEFAULT_MODEL = "qwen3.5-35b-a3b"
DEFAULT_BASE_URL = "http://127.0.0.1:8080/v1"
DEFAULT_API_KEY = "mlx-local"

SYSTEM_PROMPT = """
You are a local agentic RAG assistant over a markdown notes repository.

Rules:
- Always run a search tool before answering.
- Prefer semantic_search for conceptual questions.
- Prefer rg_search for exact terms, names, and quotes.
- Read relevant files with read_file before writing the final answer.
- Include citations in the final answer using `path:line` format.
- If evidence is weak, perform another search before finalizing.
""".strip()


def _tool_payload(hits: list[SearchHit]) -> list[dict[str, str | int]]:
    return [
        {
            "path": str(hit.path),
            "line": hit.line,
            "snippet": hit.snippet,
            "citation": hit.citation,
        }
        for hit in hits
    ]


def render_citations(hits: list[SearchHit]) -> str:
    lines = citations_for_hits(hits)
    if not lines:
        return ""
    bullet_list = "\n".join(f"- {line}" for line in lines)
    return f"\n\nSources:\n{bullet_list}"


def build_agent(
    model: str = DEFAULT_MODEL,
    base_url: str = DEFAULT_BASE_URL,
    api_key: str = DEFAULT_API_KEY,
) -> Agent[RagDeps, str]:
    llm = OpenAIModel(
        model,
        provider=OpenAIProvider(
            base_url=base_url,
            api_key=api_key,
        ),
    )

    agent = Agent(
        model=llm,
        deps_type=RagDeps,
        system_prompt=SYSTEM_PROMPT,
    )

    @agent.tool
    def semantic_search_tool(
        ctx: RunContext[RagDeps],
        query: str,
        max_results: int = 8,
    ) -> list[dict[str, str | int]]:
        return _tool_payload(semantic_search(ctx, query=query, max_results=max_results))

    @agent.tool
    def rg_search_tool(
        ctx: RunContext[RagDeps],
        query: str,
        max_results: int = 8,
    ) -> list[dict[str, str | int]]:
        return _tool_payload(rg_search(ctx, query=query, max_results=max_results))

    @agent.tool
    def read_file_tool(
        ctx: RunContext[RagDeps],
        file_path: str,
        max_chars: int = 12_000,
    ) -> str:
        return read_file(ctx, file_path=file_path, max_chars=max_chars)

    return agent
