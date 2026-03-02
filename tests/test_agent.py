from __future__ import annotations

from pathlib import Path

import pytest

import rag_agent.agent as agent_module
from rag_agent.agent import DEFAULT_MODEL, render_citations
from rag_agent.cli import _StreamChunkSanitizer, _build_parser
from rag_agent.models import SearchHit


def test_render_citations_formats_sources_block() -> None:
    hits = [
        SearchHit(path=Path("a.md"), line=3, snippet="alpha"),
        SearchHit(path=Path("b.md"), line=9, snippet="beta"),
    ]
    rendered = render_citations(hits)
    assert "Sources:" in rendered
    assert "- a.md:3" in rendered
    assert "- b.md:9" in rendered


def test_default_model_matches_expected_qwen3_4b_variant() -> None:
    assert DEFAULT_MODEL == "mlx-community/Qwen3-4B-Thinking-2507-4bit"


def test_cli_uses_default_model_when_env_not_set(monkeypatch) -> None:
    monkeypatch.delenv("RAG_MODEL", raising=False)
    args = _build_parser().parse_args(["hello"])
    assert args.model == DEFAULT_MODEL


def test_stream_chunk_sanitizer_strips_thinking_content() -> None:
    sanitizer = _StreamChunkSanitizer()
    chunks = ["Answer start ", "<th", "ink>", "internal", "</thi", "nk>", " answer end"]
    visible = "".join(sanitizer.sanitize(chunk) for chunk in chunks) + sanitizer.flush()
    assert visible == "Answer start  answer end"


def test_build_local_mlx_model_falls_back_for_extra_weights(monkeypatch) -> None:
    calls: list[str] = []

    def fail_initial(_model: str) -> tuple[object, object]:
        raise ValueError("Received 333 parameters not in model")

    def relaxed_loader(model: str) -> tuple[object, object]:
        calls.append(model)
        return object(), object()

    def fake_outlines(_mlx_model: object, _tokenizer: object) -> object:
        return "outlines-model"

    monkeypatch.setattr(agent_module, "_load_mlx_components", fail_initial)
    monkeypatch.setattr(agent_module, "_load_mlx_components_relaxed", relaxed_loader)
    monkeypatch.setattr(agent_module, "_build_outlines_model", fake_outlines)

    built = agent_module._build_local_mlx_model(DEFAULT_MODEL)
    assert built == "outlines-model"
    assert calls == [DEFAULT_MODEL]


def test_build_local_mlx_model_raises_non_weight_errors(monkeypatch) -> None:
    def fail_with_unrelated_error(_model: str) -> tuple[object, object]:
        raise ValueError("boom")

    monkeypatch.setattr(agent_module, "_load_mlx_components", fail_with_unrelated_error)

    with pytest.raises(ValueError, match="boom"):
        agent_module._build_local_mlx_model(DEFAULT_MODEL)
