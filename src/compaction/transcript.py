"""Permanent, append-only transcript of every message ever produced in a run."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Literal

RETRIEVED_OMITTED = "[retrieved copy of earlier messages omitted]"


@dataclass
class Message:
    id: str  # "m0001", "m0002", ... monotonically increasing, never reused
    role: Literal["user", "assistant"]
    content: list[dict]  # Anthropic content blocks as plain dicts
    session: int  # which context-session produced it
    retrieved: bool = False  # carries the RESULT of a fetch_messages call
    synthetic: bool = False  # harness-injected (summary, index notice, nudge...)


@dataclass
class Transcript:
    messages: list[Message] = field(default_factory=list)
    _by_id: dict[str, Message] = field(default_factory=dict, repr=False)

    def append(
        self,
        role: str,
        content: list[dict] | str,
        *,
        session: int,
        retrieved: bool = False,
        synthetic: bool = False,
    ) -> Message:
        if isinstance(content, str):
            content = [{"type": "text", "text": content}]
        msg = Message(
            f"m{len(self.messages) + 1:04d}",
            role,
            content,
            session,
            retrieved,
            synthetic,
        )  # type: ignore[arg-type]
        self.messages.append(msg)
        self._by_id[msg.id] = msg
        return msg

    def get(self, msg_id: str) -> Message:
        return self._by_id[msg_id]

    def range(self, start_id: str, end_id: str) -> list[Message]:
        """Messages from start_id to end_id, inclusive (ids are sequential)."""
        i, j = (
            self.messages.index(self.get(start_id)),
            self.messages.index(self.get(end_id)),
        )
        return self.messages[i : j + 1]

    def all(self) -> list[Message]:
        return list(self.messages)

    def __len__(self) -> int:
        return len(self.messages)

    # ---- persistence -------------------------------------------------
    def save(self, path: str | Path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("w", encoding="utf-8") as f:
            for m in self.messages:
                f.write(json.dumps(asdict(m), ensure_ascii=False) + "\n")

    @classmethod
    def load(cls, path: str | Path) -> Transcript:
        t = cls()
        with Path(path).open(encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    m = Message(**json.loads(line))
                    t.messages.append(m)
                    t._by_id[m.id] = m
        return t

    # ---- rendering ---------------------------------------------------
    def render(self, msgs: list[Message], *, omit_retrieved: bool = False) -> str:
        """Human-readable flattening, e.g. `[m0007 assistant] text` followed by
        indented `tool_use name(args)` / `[tool_result id] body` lines. Thinking
        blocks are not rendered. With omit_retrieved, tool_result bodies of
        `retrieved` messages are replaced by a placeholder."""
        out: list[str] = []
        for m in msgs:
            lines: list[str] = []
            for b in m.content:
                t = b.get("type")
                if t == "text":
                    lines.append(b["text"])
                elif t == "tool_use":
                    lines.append(
                        f"tool_use {b['name']}({json.dumps(b.get('input', {}), ensure_ascii=False)})"
                    )
                elif t == "tool_result":
                    body = (
                        RETRIEVED_OMITTED
                        if (omit_retrieved and m.retrieved)
                        else _result_text(b)
                    )
                    lines.append(f"[tool_result {b.get('tool_use_id', '')}] {body}")
            out.append(f"[{m.id} {m.role}] " + "\n".join(lines))
        return "\n".join(out)


def _result_text(block: dict) -> str:
    c = block.get("content", "")
    if isinstance(c, str):
        return c
    return "".join(x.get("text", "") for x in c if isinstance(x, dict))
