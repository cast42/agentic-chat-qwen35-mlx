# Plan --- Agentic RAG Chatbot (Pydantic-AI + MLX)

## Goal

Build a local agentic RAG chatbot using best-practice Python
development.

Knowledge source: https://github.com/cast42/notes

The system runs locally using a Qwen 3.5 MLX model and tool-driven
retrieval.

------------------------------------------------------------------------

## Base Project

Initialize from: https://github.com/cast42/minimal-python-boilerplate

Do NOT invent a new structure. Extend the boilerplate.

------------------------------------------------------------------------

## Core Principles

-   Pydantic-AI provides the agent loop.
-   Retrieval happens via tools.
-   No vector database.
-   No embeddings pipeline.
-   Keep architecture minimal and explicit.
-   Prefer small typed functions.

------------------------------------------------------------------------

## Model

Qwen3.5-35B-A3B (MLX 4bit)

------------------------------------------------------------------------

## Knowledge Source

Add as git submodule:

git submodule add https://github.com/cast42/notes notes_repo

Agent reads markdown directly from filesystem.

------------------------------------------------------------------------

## Required Features

The agent must: 1. Decide which search tool to use 2. Search notes 3.
Read markdown files 4. Synthesize answers 5. Cite sources

------------------------------------------------------------------------

## Project Structure

src/ rag_agent/ agent.py deps.py tools/ search.py files.py

------------------------------------------------------------------------

## Dependency Injection

@dataclass class RagDeps: notes_path: Path

All tools receive dependencies via RunContext\[RagDeps\]. No globals
allowed.

------------------------------------------------------------------------

## Tools

### ripgrep search

rg -n `<query>`{=html} notes_repo

### semantic search

qmd search `<query>`{=html}

### file reader

Reads markdown safely with truncation.

------------------------------------------------------------------------

## Agent Rules

-   Always search before answering.
-   Prefer semantic search for concepts.
-   Prefer ripgrep for exact queries.
-   Read files before summarizing.
-   Cite file paths.
-   Search again if uncertain.

------------------------------------------------------------------------

## CLI Interface

just run "question"

Must stream output and show citations.

------------------------------------------------------------------------

## Testing

Add pytest tests verifying: - tools execute - files read correctly -
citations appear

------------------------------------------------------------------------

## Non-Goals

Do NOT introduce: - LangChain - LlamaIndex - vector DBs - embeddings -
custom loops - async complexity

------------------------------------------------------------------------

## Success Criteria

just run "What did I write about agentic coding?"

Agent searches, reads, and answers with citations.

------------------------------------------------------------------------

## Definition of Done

-   just check passes
-   just test passes
-   minimal architecture preserved
