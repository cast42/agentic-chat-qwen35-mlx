# AGENTS.md

Instructions for AI coding agents working in this repository.

------------------------------------------------------------------------

# Project Identity

Local agentic RAG chatbot using:

-   Python ≥ 3.14
-   uv package management
-   just command runner
-   Pydantic-AI
-   MLX local models
-   markdown knowledge retrieval

Design goal: SIMPLE • LOCAL • TYPE-SAFE • MINIMAL

------------------------------------------------------------------------

# Command Execution Policy (CRITICAL)

The justfile is the single interface for workflows.

NEVER run tools directly: ruff, pytest, uv run ty, python ...

ALWAYS use: just check just test just run just fmt

If functionality is missing → extend the justfile.

Discover commands: just --list

------------------------------------------------------------------------

# Environment Setup

Install dependencies: uv sync

Add dependencies: uv add `<package>`{=html}

Never use pip or poetry.

------------------------------------------------------------------------

# Quality Gates

Repository must remain green:

just check just test

Agents must fix failures immediately.

------------------------------------------------------------------------

# Architecture Rules

## Pydantic-AI IS the agent loop

Do NOT implement planners, orchestration engines, or custom loops.

## Tool-Driven RAG

Retrieval uses: - semantic_search - rg_search - read_file

Do NOT add embeddings, vector DBs, LangChain, or LlamaIndex.

Filesystem is source of truth.

## Dependency Injection

All tools must use RunContext\[Deps\]. No globals.

------------------------------------------------------------------------

# Small Modules Rule

Prefer small files, typed functions, explicit logic. Target ≤300 LOC
core logic.

------------------------------------------------------------------------

# Agent Development Loop (MANDATORY)

Workflow: 1. Read relevant files 2. Make smallest change possible 3.
Run: just check just test 4. Fix issues immediately 5. Repeat

Repository must always return to green state.

Preferred change size: ≤3 files, ≤150 lines.

------------------------------------------------------------------------

# Coding Style

-   built-in generics (list\[str\])
-   strong typing
-   dataclasses for deps
-   pure functions preferred
-   explicit imports

Readable \> clever.

------------------------------------------------------------------------

# Testing Philosophy

Tests verify behavior: - tool usage - file reading - citation presence

Avoid heavy mocking.

------------------------------------------------------------------------

# CLI Expectations

just run "`<question>`{=html}"

Must stream output, show citations, reuse session context.

------------------------------------------------------------------------

# Anti-Patterns

Do NOT introduce: - new frameworks - async complexity - plugin systems -
configuration layers - premature abstractions

------------------------------------------------------------------------

# Decision Heuristic

When unsure: 1. Choose simpler solution 2. Extend existing code 3.
Prefer functions over classes 4. Prefer tools over systems

------------------------------------------------------------------------

# Definition of Done

A task is complete when: - just check passes - just test passes -
architecture rules respected - minimal design preserved
