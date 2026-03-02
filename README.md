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

Environment variables for local MLX inference:

- `RAG_MODEL` (default: `mlx-community/Qwen3-4B-Thinking-2507-4bit`)
- `RAG_MAX_TOKENS` (default: `2048`)

No OpenAI-compatible server is required. The first run may download model files.

## Quality

```bash
just check
just test
```
