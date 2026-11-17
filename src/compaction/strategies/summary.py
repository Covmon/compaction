"""Traditional compaction: a model writes a prose summary of the head."""

from __future__ import annotations

from ..context import LiveContext
from ..tokens import estimate_tokens
from .base import CompactionEvent, CompactionStrategy, split_tail

SUMMARY_SYSTEM = (
    "You compress the transcript of an AI agent's work so the agent can continue with a much "
    "smaller context. Write a detailed, structured summary preserving: the task and any "
    "instructions, what has been done, key findings, and open questions / next steps. Preserve "
    "EXACT identifiers, numbers, timestamps, file names, versions, hashes and names verbatim; "
    "never paraphrase them. If a previous summary is given, merge it with the new messages into "
    "one updated summary. Output only the summary."
)
HEADER = "[Context was compacted. Summary of earlier work:]\n"


class SummaryStrategy(CompactionStrategy):
    name = "summary"

    def __init__(
        self, summary_max_tokens: int = 2000, model: str | None = None
    ) -> None:
        self.summary_max_tokens = summary_max_tokens
        self.model = model
        self.summary: str | None = None

    def compact(
        self, ctx: LiveContext, session: int, step: int
    ) -> tuple[LiveContext, CompactionEvent]:
        before = estimate_tokens(ctx.to_api_messages(self.transcript))
        head, tail = split_tail(ctx, self.transcript, self.config.keep_last)
        # Synthetic head messages (earlier summary/notes) are represented by self.summary.
        to_summarize = [
            self.transcript.get(i) for i in head if not self.transcript.get(i).synthetic
        ]
        cin = cout = 0
        if to_summarize:
            prompt = ""
            if self.summary:
                prompt += f"<previous_summary>\n{self.summary}\n</previous_summary>\n\n"
            prompt += (
                "<new_messages>\n"
                + self.transcript.render(to_summarize, omit_retrieved=True)
                + "\n</new_messages>\n\nWrite the updated summary "
                f"(at most about {self.summary_max_tokens} tokens)."
            )
            resp = self.llm.create(
                model=self.model or self.config.model,
                system=SUMMARY_SYSTEM,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=self.summary_max_tokens,
                purpose="compactor",
            )
            cin, cout = resp.total_input_tokens, resp.output_tokens
            self.summary = resp.text.strip() or self.summary or "(summary unavailable)"
        if self.summary is None:
            return ctx, CompactionEvent(step, before, before, detail={"noop": True})
        note = self.transcript.append(
            "user", HEADER + self.summary, session=session + 1, synthetic=True
        )
        new = LiveContext([note.id] + tail)
        after = estimate_tokens(new.to_api_messages(self.transcript))
        return new, CompactionEvent(
            step,
            before,
            after,
            cin,
            cout,
            {
                "summarized": len(to_summarize),
                "summary_tokens": estimate_tokens(self.summary),
            },
        )
