from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, cast

from pydantic_ai import Agent

from rag_agent.deps import RagDeps
from rag_agent.models import SearchHit
from rag_agent.tools.search import QMD_HELP, citations_for_hits

if TYPE_CHECKING:
    from mlx.nn.layers.base import Module as MlxModule
    from pydantic_ai.models import Model
    from transformers import PreTrainedTokenizer

DEFAULT_MODEL = "mlx-community/Qwen3.5-9B-4bit"
_TOKENIZER_CONFIG = {"eos_token": "<|endoftext|>", "trust_remote_code": True}

SYSTEM_PROMPT = f"""
You are a local agentic RAG assistant over a markdown notes repository.

Tool retrieval is done through qmd planning steps, and tool outputs are provided in the conversation context.
Treat those qmd outputs as your primary evidence.

qmd help:
```ascii
{QMD_HELP}
```

Rules:
- Include citations in the final answer using `path:line` format whenever evidence exists.
- If the provided context is insufficient, say so explicitly.
- Respond with only the final answer, never your reasoning process or planning steps.
""".strip()


def _load_mlx_components(model: str) -> tuple[MlxModule, PreTrainedTokenizer]:
    from mlx_lm import load

    load_result = load(
        model,
        tokenizer_config=_TOKENIZER_CONFIG,
        return_config=True,
    )
    mlx_model, tokenizer, _config = cast(
        "tuple[MlxModule, PreTrainedTokenizer, dict[str, object]]",
        load_result,
    )
    return mlx_model, tokenizer


def _load_mlx_components_relaxed(model: str) -> tuple[MlxModule, PreTrainedTokenizer]:
    from huggingface_hub import snapshot_download
    from mlx_lm.utils import load_model, load_tokenizer

    model_path = Path(model)
    if not model_path.exists():
        model_path = Path(snapshot_download(model))

    mlx_model, config = load_model(model_path=model_path, strict=False)
    tokenizer = load_tokenizer(
        model_path,
        tokenizer_config_extra=_TOKENIZER_CONFIG,
        eos_token_ids=config.get("eos_token_id", None),
    )
    return cast("tuple[MlxModule, PreTrainedTokenizer]", (mlx_model, tokenizer))


def _build_outlines_model(mlx_model: MlxModule, tokenizer: PreTrainedTokenizer) -> Model:
    from pydantic_ai.models.outlines import OutlinesModel

    return OutlinesModel.from_mlxlm(mlx_model, tokenizer)


def _build_local_mlx_model(model: str) -> Model:
    try:
        mlx_model, tokenizer = _load_mlx_components(model)
    except ImportError as exc:
        raise RuntimeError(
            "Missing MLX dependencies. Run `just sync` to install `mlx-lm` and `outlines`."
        ) from exc
    except ValueError as exc:
        if "parameters not in model" not in str(exc):
            raise
        mlx_model, tokenizer = _load_mlx_components_relaxed(model)

    return _build_outlines_model(mlx_model, tokenizer)


def render_citations(hits: list[SearchHit]) -> str:
    lines = citations_for_hits(hits)
    if not lines:
        return ""
    bullet_list = "\n".join(f"- {line}" for line in lines)
    return f"\n\nSources:\n{bullet_list}"


def build_agent(model: str = DEFAULT_MODEL) -> Agent[RagDeps, str]:
    return Agent(
        model=_build_local_mlx_model(model),
        deps_type=RagDeps,
        system_prompt=SYSTEM_PROMPT,
    )


def build_planning_agent(model: str = DEFAULT_MODEL) -> Agent[RagDeps, str]:
    """Build a planning agent without a system prompt.

    This avoids multiple `system` role messages when using PromptedOutput with Outlines.
    """
    return Agent(
        model=_build_local_mlx_model(model),
        deps_type=RagDeps,
    )
