# Agentic RAG Chatbot (Pydantic-AI + MLX)

Minimal local RAG assistant that searches markdown notes with tools and cites source files.

## Setup

1. Install dependencies:

   ```bash
   just sync
   ```

2. Add notes content:

   ```bash
   git clone https://github.com/cast42/notes notes_repo
   ```

   If this repository is initialized as git, you can use a submodule instead:

   ```bash
   git submodule add https://github.com/cast42/notes notes_repo
   ```

## Check if the model runs

```bash
just run "Di una frase corta en espanol."
```

## Run

```bash
just run "What did I write about agentic coding?"
```

The agent can run tools against the notes collection via `qmd`:

```ascii
qmd query "question" --collection notes              # Auto-expand + rerank
qmd query $'lex: X\nvec: Y' --collection notes       # Structured
qmd query $'expand: question' --collection notes     # Explicit expand
qmd search "keywords" --collection notes             # BM25 only (no LLM)
qmd get "#abc123" --collection notes                 # By docid
qmd multi-get "journals/2026-*.md" -l 40 --collection notes  # Batch pull snippets by glob
qmd multi-get notes/foo.md,notes/bar.md --collection notes   # Comma-separated list, preserves order
```

Environment variables for local MLX inference:

- `RAG_MODEL` (default: `mlx-community/Qwen3.5-9B-4bit`)
- `RAG_MAX_TOKENS` (default: `2048`)

No OpenAI-compatible server is required. The first run may download model files.

## Quality

```bash
just check
just test
```
