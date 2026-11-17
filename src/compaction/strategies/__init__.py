"""Compaction strategies. `make_strategy("truncate" | "summary" | "indexed", **kw)`."""

from __future__ import annotations

from .base import CompactionEvent, CompactionStrategy, split_tail
from .summary import SummaryStrategy
from .truncate import TruncateStrategy

__all__ = [
    "CompactionEvent",
    "CompactionStrategy",
    "SummaryStrategy",
    "TruncateStrategy",
    "make_strategy",
    "split_tail",
]


def make_strategy(name: str, **kw) -> CompactionStrategy:
    if name == "truncate":
        return TruncateStrategy(**kw)
    if name == "summary":
        return SummaryStrategy(**kw)
    if name == "indexed":
        from .indexed import IndexedRetentionStrategy  # lazy: built by a later agent

        return IndexedRetentionStrategy(**kw)
    raise ValueError(f"unknown strategy {name!r}")
