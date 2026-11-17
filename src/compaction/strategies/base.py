"""Compaction strategy interface and shared helpers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from ..context import LiveContext
from ..tools import ToolSpec
from ..transcript import Transcript

if TYPE_CHECKING:
    from ..agent import AgentConfig
    from ..llm import LLMClient


@dataclass
class CompactionEvent:
    step: int
    tokens_before: int  # strategies fill message-only estimates; Agent overwrites with full request estimates
    tokens_after: int
    compactor_input_tokens: int = 0
    compactor_output_tokens: int = 0
    detail: dict = field(default_factory=dict)


class CompactionStrategy(ABC):
    name: str = "base"
    # Set by Agent before bind(): where the run's transcript is saved (None if not
    # persisted). Strategies may write their own artifacts next to it.
    transcript_path: Path | None = None

    def bind(self, llm: LLMClient, transcript: Transcript, config: AgentConfig) -> None:
        self.llm = llm
        self.transcript = transcript
        self.config = config

    def tools(self) -> list[ToolSpec]:
        return []

    def system_addendum(self) -> str:
        return ""

    @abstractmethod
    def compact(
        self, ctx: LiveContext, session: int, step: int
    ) -> tuple[LiveContext, CompactionEvent]:
        """Return the NEW live context for session+1 (may append synthetic messages
        to the transcript, with session=session+1)."""


def split_tail(
    ctx: LiveContext, transcript: Transcript, keep_last: int
) -> tuple[list[str], list[str]]:
    """Split ctx.ids into (head_ids, tail_ids), tail ~ the last `keep_last` messages.

    The tail never starts with an orphan tool_result: if the natural cut falls on a
    user message holding tool_results, the cut moves one earlier so the tail begins
    with the assistant message that issued the tool_use. (Starting on an assistant
    message is fine because strategies prepend a user message.)"""
    ids = ctx.ids
    start = max(0, len(ids) - max(0, keep_last))
    # A turn's tool_results may span several user messages (fetch_messages results
    # are stored separately), so keep stepping back until the cut is clean.
    while 0 < start < len(ids):
        m = transcript.get(ids[start])
        if not (m.role == "user" and any(b.get("type") == "tool_result" for b in m.content)):
            break
        start -= 1
    return list(ids[:start]), list(ids[start:])
