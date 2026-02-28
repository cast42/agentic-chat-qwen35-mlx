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

## Run

```bash
just run "What did I write about agentic coding?"
```

Environment variables for local MLX/OpenAI-compatible serving:

- `RAG_OPENAI_BASE_URL` (default: `http://127.0.0.1:8080/v1`)
- `RAG_OPENAI_API_KEY` (default: `mlx-local`)
- `RAG_MODEL` (default: `qwen3.5-35b-a3b`)

## Quality

```bash
just check
just test
```
