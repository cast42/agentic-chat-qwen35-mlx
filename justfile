set shell := ["zsh", "-eu", "-o", "pipefail", "-c"]

default:
    @just --list

sync:
    uv sync

fmt:
    uv run ruff format src tests

check:
    uv run ruff check src tests
    uv run ruff format --check src tests
    uv run pyright src tests

test:
    uv run pytest -q

run question:
    uv run rag-agent "{{question}}"

