"""The agent loop: model calls, tool execution, compaction and hard-cap enforcement."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

from .context import LiveContext
from .llm import LLMClient
from .strategies.base import CompactionEvent, CompactionStrategy
from .tools import ToolSpec
from .transcript import Transcript

__all__ = ["Agent", "AgentConfig", "Environment", "RunResult", "ToolSpec"]

NUDGE = "Continue. Call submit_answer when finished."
MAX_TOKENS_NOTE = (
    "Your previous response hit the output token limit and was cut off. "
    "Be more concise and continue."
)
MIN_RESULT_CHARS = 200
FETCH_TOOL = "fetch_messages"


class Environment(Protocol):
    def system_prompt(self) -> str: ...
    def initial_user_message(self) -> str: ...
    def tools(self) -> list[ToolSpec]: ...  # includes terminal `submit_answer`
    def done(self) -> bool: ...
    def answer(self) -> dict | None: ...

    # OPTIONAL (looked up with getattr, so envs may omit it): checked after each
    # step; returned text is appended as a new user message (e.g. phase-2 questions).
    def pending_user_message(self) -> str | None: ...


@dataclass
class AgentConfig:
    model: str = "claude-haiku-5-5"
    context_cap: int = 16_000
    compact_threshold: float = 0.80  # compact when effective tokens > cap * threshold
    keep_last: int = 4  # messages carried over into the new session
    max_steps: int = 120  # model calls
    max_tokens: int = 2048  # per agent response


@dataclass
class RunResult:
    answer: dict | None
    steps: int
    compactions: list[CompactionEvent]
    usage: dict
    transcript_path: str | None
    hit_cap_violation: bool
    error: str | None
    transcript: Transcript | None = field(default=None, repr=False)


class Agent:
    def __init__(
        self,
        llm: LLMClient,
        env: Environment,
        strategy: CompactionStrategy,
        config: AgentConfig,
        transcript_dir: str | Path | None = None,
        transcript_name: str = "transcript.jsonl",
    ) -> None:
        self.llm, self.env, self.strategy, self.config = llm, env, strategy, config
        self.transcript_path = (
            Path(transcript_dir) / transcript_name if transcript_dir else None
        )
        self.transcript = Transcript()
        self.ctx = LiveContext()
        self.session = 0
        self.compactions: list[CompactionEvent] = []
        self.hit_cap_violation = False
        self._steps = 0
        self._calib: tuple[int, int] | None = (
            None  # (reported prompt tokens, our estimate then)
        )
        strategy.transcript_path = self.transcript_path
        strategy.bind(llm, self.transcript, config)

    # ---- helpers ------------------------------------------------------
    def _add(self, role: str, content, **kw) -> None:
        self.ctx.append(
            self.transcript.append(role, content, session=self.session, **kw).id
        )

    def _request_parts(self) -> tuple[str, list[ToolSpec], list[dict]]:
        system = self.env.system_prompt() + self.strategy.system_addendum()
        specs = {t.name: t for t in self.env.tools()}
        specs.update({t.name: t for t in self.strategy.tools()})
        specs_l = list(specs.values())
        return system, specs_l, [t.to_api() for t in specs_l]

    def _effective(self, system: str, tools: list[dict]) -> tuple[int, int]:
        """(effective, raw estimate). effective = max(estimate, last reported + growth)."""
        est = self.ctx.tokens(self.transcript, system, tools)
        eff = est
        if self._calib:
            reported, est_then = self._calib
            eff = max(est, reported + max(0, est - est_then))
        return eff, est

    def _fit_to_cap(self, system: str, tools: list[dict]) -> bool:
        """Guarantee the request fits in context_cap. Drop oldest messages one at a
        time, then shrink tool_result bodies. Returns False if impossible."""
        cap = self.config.context_cap
        while self._effective(system, tools)[0] > cap:
            self.hit_cap_violation = True
            if len(self.ctx.ids) > 1:
                self.ctx.ids.pop(0)
                self._calib = None  # history changed; stop trusting the old reading
                continue
            cur = self.ctx.max_result_chars
            if cur is None:
                # Start from the biggest result currently visible.
                cur = max(
                    [
                        len(str(b.get("content", "")))
                        for m in self.ctx.messages(self.transcript)
                        for b in m.content
                        if b.get("type") == "tool_result"
                    ]
                    + [MIN_RESULT_CHARS * 2]
                )
            if cur <= MIN_RESULT_CHARS:
                return False
            self.ctx.max_result_chars = max(MIN_RESULT_CHARS, cur // 2)
        return True

    def _maybe_compact(
        self, system: str, tools: list[dict], step: int, since_compact: int
    ) -> None:
        cap = self.config.context_cap
        eff, _ = self._effective(system, tools)
        if (
            eff <= cap * self.config.compact_threshold
            or since_compact < 2
            or len(self.ctx.ids) < 2
        ):
            return
        new_ctx, ev = self.strategy.compact(self.ctx, self.session, step)
        if new_ctx.ids == self.ctx.ids:  # no-op compaction
            return
        ev.tokens_before = eff
        self.session += 1
        self.ctx = new_ctx
        self._calib = None
        ev.tokens_after = self.ctx.tokens(self.transcript, system, tools)
        self.compactions.append(ev)

    # ---- main loop ----------------------------------------------------
    def run(self) -> RunResult:
        error: str | None = None
        try:
            error = self._loop()
        except Exception as e:  # noqa: BLE001
            error = f"{type(e).__name__}: {e}"
        finally:
            if self.transcript_path:
                self.transcript.save(self.transcript_path)
        return RunResult(
            answer=self.env.answer(),
            steps=self._steps,
            compactions=self.compactions,
            usage=self.llm.usage.snapshot(),
            transcript_path=str(self.transcript_path) if self.transcript_path else None,
            hit_cap_violation=self.hit_cap_violation,
            error=error,
            transcript=self.transcript,
        )

    def _loop(self) -> str | None:
        cfg = self.config
        self._add("user", self.env.initial_user_message())
        steps = self._steps = 0
        last_compact_len = 0
        while not self.env.done():
            if steps >= cfg.max_steps:
                return "max_steps"
            system, specs, tools = self._request_parts()
            self._maybe_compact(
                system, tools, steps, len(self.ctx.ids) - last_compact_len
            )
            if self.compactions and self.compactions[-1].step == steps:
                last_compact_len = len(self.ctx.ids)
            if not self._fit_to_cap(system, tools):
                return "cap_unsatisfiable: request cannot fit in context_cap"
            messages = self.ctx.to_api_messages(self.transcript)
            est = self.ctx.tokens(self.transcript, system, tools)

            resp = self.llm.create(
                model=cfg.model,
                system=system,
                messages=messages,
                tools=tools or None,
                max_tokens=cfg.max_tokens,
                purpose="agent",
            )
            steps = self._steps = steps + 1
            self._calib = (
                (resp.total_input_tokens, est) if resp.total_input_tokens else None
            )

            if resp.stop_reason == "refusal":
                return "refusal"
            content = resp.content
            truncated = resp.stop_reason == "max_tokens"
            if truncated:  # a cut-off tool_use has incomplete input: never execute it
                content = [b for b in content if b.get("type") != "tool_use"]
            if not any(b.get("type") in ("text", "tool_use") for b in content):
                content = content + [{"type": "text", "text": "(no output)"}]
            self._add("assistant", content)

            uses = [b for b in content if b.get("type") == "tool_use"]
            if uses:
                handlers = {t.name: t.handler for t in specs}
                # Results of fetch_messages go in their own message flagged
                # `retrieved`, so a parallel read_file result in the same turn is
                # still indexed (only re-fetched copies are excluded).
                fresh, fetched = [], []
                for u in uses:
                    r = self._run_tool(handlers, u)
                    (fetched if u["name"] == FETCH_TOOL else fresh).append(r)
                if fresh:
                    self._add("user", fresh)
                if fetched:
                    self._add("user", fetched, retrieved=True)
            if self.env.done():
                break
            pending = getattr(self.env, "pending_user_message", lambda: None)()
            if pending:
                self._add(
                    "user", pending
                )  # merged with tool_results by to_api_messages
            elif truncated:
                self._add("user", MAX_TOKENS_NOTE, synthetic=True)
            elif not uses:
                self._add("user", NUDGE, synthetic=True)
        return None

    @staticmethod
    def _run_tool(handlers: dict, use: dict) -> dict:
        h = handlers.get(use["name"])
        block = {"type": "tool_result", "tool_use_id": use["id"]}
        if h is None:
            return {
                **block,
                "content": f"Error: unknown tool {use['name']!r}",
                "is_error": True,
            }
        try:
            return {**block, "content": str(h(use.get("input") or {}))}
        except Exception as e:  # noqa: BLE001
            return {
                **block,
                "content": f"Error: {type(e).__name__}: {e}",
                "is_error": True,
            }
