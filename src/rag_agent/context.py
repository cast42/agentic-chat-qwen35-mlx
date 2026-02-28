"""Compatibility layer for type-checking RunContext."""

from __future__ import annotations

from typing import Protocol, TypeVar

DepsT = TypeVar("DepsT")

try:
    from pydantic_ai import RunContext as RunContext
except ImportError:

    class RunContext(Protocol[DepsT]):
        """Small fallback so tool modules stay importable before dependency install."""

        deps: DepsT
