"""LiveContext: the ordered ids currently visible to the model + API message building."""

from __future__ import annotations

import copy
from dataclasses import dataclass, field

from .tokens import estimate_tokens
from .transcript import Message, Transcript

TRUNCATION_NOTE = "\n...[truncated to fit context]"
_THINKING = ("thinking", "redacted_thinking")


def strip_thinking(messages: list[dict], *, keep_latest: bool = True) -> list[dict]:
    """THE single place that decides which thinking blocks are resent.

    The API only needs thinking blocks passed back for the latest assistant turn
    while a tool loop is in progress (assistant tool_use -> user tool_result). The
    harness rewrites history at compaction, so older thinking blocks (whose
    signatures would no longer match a coherent conversation) are always dropped.
    Kept only if `keep_latest` and the LAST assistant message contains a tool_use
    and is among the final two messages. If the API still rejects thinking,
    AnthropicLLM retries with keep_latest=False. Returns new dicts; input untouched.
    """
    last_asst = max(
        (i for i, m in enumerate(messages) if m["role"] == "assistant"), default=-1
    )
    out = []
    for i, m in enumerate(messages):
        c = m["content"]
        if isinstance(c, list):
            keep = (
                keep_latest
                and i == last_asst
                and i >= len(messages) - 2
                and any(b.get("type") == "tool_use" for b in c)
            )
            if not keep:
                c = [b for b in c if b.get("type") not in _THINKING]
                if not c:
                    c = [{"type": "text", "text": "(no output)"}]
        out.append({"role": m["role"], "content": c})
    return out


def _tool_use_ids(msg: dict) -> set[str]:
    return {b["id"] for b in msg["content"] if b.get("type") == "tool_use"}


def _result_str(b: dict) -> str:
    c = b.get("content", "")
    return (
        c
        if isinstance(c, str)
        else "".join(x.get("text", "") for x in c if isinstance(x, dict))
    )


@dataclass
class LiveContext:
    ids: list[str] = field(default_factory=list)
    # View-level truncation of tool_result bodies (chars); the transcript is untouched.
    # Used only by the hard-cap fallback in the agent.
    max_result_chars: int | None = None

    def copy(self) -> LiveContext:
        return LiveContext(list(self.ids), self.max_result_chars)

    def append(self, msg_id: str) -> None:
        self.ids.append(msg_id)

    def messages(self, transcript: Transcript) -> list[Message]:
        return [transcript.get(i) for i in self.ids]

    def to_api_messages(self, transcript: Transcript) -> list[dict]:
        """Build a valid Anthropic `messages` array: non-empty, alternating roles,
        first message `user`, last message `user`, tool_results first in their user
        message, and no orphan tool_result / unanswered tool_use (converted to text)."""
        msgs: list[dict] = []
        for m in self.messages(transcript):
            blocks = [
                copy.deepcopy(b)
                for b in m.content
                if not (b.get("type") == "text" and not b.get("text", "").strip())
            ]
            if blocks:
                msgs.append({"role": m.role, "content": blocks})
        msgs = self._merge(msgs)
        if self.max_result_chars is not None:
            for m in msgs:
                for b in m["content"]:
                    if b.get("type") == "tool_result":
                        s = _result_str(b)
                        if len(s) > self.max_result_chars:
                            b["content"] = s[: self.max_result_chars] + TRUNCATION_NOTE
        self._fix_tool_pairs(msgs)
        if not msgs or msgs[0]["role"] != "user":
            msgs.insert(
                0,
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "[Earlier context is not available.]"}
                    ],
                },
            )
        if msgs[-1]["role"] != "user":
            msgs.append(
                {"role": "user", "content": [{"type": "text", "text": "Continue."}]}
            )
        return strip_thinking(msgs)

    @staticmethod
    def _merge(msgs: list[dict]) -> list[dict]:
        merged: list[dict] = []
        for m in msgs:
            if merged and merged[-1]["role"] == m["role"]:
                merged[-1]["content"].extend(m["content"])
            else:
                merged.append(m)
        return merged

    @staticmethod
    def _fix_tool_pairs(msgs: list[dict]) -> None:
        """Convert orphan tool_results and unanswered tool_uses to text (in place)."""
        for i, m in enumerate(msgs):
            if m["role"] == "user":
                prev_ids = _tool_use_ids(msgs[i - 1]) if i > 0 else set()
                new = []
                for b in m["content"]:
                    if (
                        b.get("type") == "tool_result"
                        and b.get("tool_use_id") not in prev_ids
                    ):
                        b = {
                            "type": "text",
                            "text": f"[earlier tool result] {_result_str(b)}",
                        }
                    new.append(b)
                # API requires tool_result blocks to come first.
                new.sort(key=lambda b: b.get("type") != "tool_result")
                m["content"] = new
            else:
                answered = set()
                if i + 1 < len(msgs):
                    answered = {
                        b.get("tool_use_id")
                        for b in msgs[i + 1]["content"]
                        if b.get("type") == "tool_result"
                    }
                new = []
                for b in m["content"]:
                    if b.get("type") == "tool_use" and b["id"] not in answered:
                        args = str(b.get("input", {}))
                        b = {
                            "type": "text",
                            "text": f"[called {b['name']}({args}); result not available]",
                        }
                    new.append(b)
                if not any(b.get("type") == "tool_use" for b in new):
                    new = [b for b in new if b.get("type") not in _THINKING] or new
                m["content"] = new

    def tokens(self, transcript: Transcript, system: str, tools: list[dict]) -> int:
        return (
            estimate_tokens(system)
            + estimate_tokens(tools)
            + estimate_tokens(self.to_api_messages(transcript))
        )
