"""Baseline: drop the oldest messages, keep the last K."""

from __future__ import annotations

from ..context import LiveContext
from ..tokens import estimate_tokens
from .base import CompactionEvent, CompactionStrategy, split_tail


class TruncateStrategy(CompactionStrategy):
    name = "truncate"

    def __init__(self, pin_first: bool = False) -> None:
        self.pin_first = (
            pin_first  # keep the very first (task) message in every session
        )

    def compact(
        self, ctx: LiveContext, session: int, step: int
    ) -> tuple[LiveContext, CompactionEvent]:
        before = estimate_tokens(ctx.to_api_messages(self.transcript))
        head, tail = split_tail(ctx, self.transcript, self.config.keep_last)
        pinned = head[:1] if (self.pin_first and head) else []
        dropped = len(head) - len(pinned)
        note = self.transcript.append(
            "user",
            f"[Context was truncated: {dropped} earlier messages were dropped and are no longer available.]",
            session=session + 1,
            synthetic=True,
        )
        new = LiveContext(pinned + [note.id] + tail)
        after = estimate_tokens(new.to_api_messages(self.transcript))
        return new, CompactionEvent(
            step, before, after, detail={"dropped": dropped, "kept": len(tail)}
        )
