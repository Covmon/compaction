"""LLM clients (real + scripted test double) and usage/cost accounting."""

from __future__ import annotations

import copy
import dataclasses
import json
import threading
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Protocol

from .context import _result_str, strip_thinking
from .tokens import estimate_tokens

# (input $/MTok, output $/MTok). Cache reads bill at 10% of input, writes at 125%.
PRICES: dict[str, tuple[float, float]] = {
    "claude-haiku-5-5": (0.10, 0.50),
}
DEFAULT_PRICE = PRICES["claude-haiku-5-5"]
CACHE_READ_MULT = 0.10
CACHE_WRITE_MULT = 1.25


@dataclass
class LLMResponse:
    # Content blocks as plain dicts (text / tool_use / thinking / redacted_thinking).
    # Thinking blocks are kept (see context.strip_thinking for when they are resent).
    content: list[dict]
    stop_reason: str
    input_tokens: int  # uncached input tokens, as the API reports them
    output_tokens: int
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    # Cost as billed by the provider (e.g. OpenRouter's usage.cost). When None,
    # cost is computed from PRICES.
    billed_usd: float | None = None

    @property
    def total_input_tokens(self) -> int:
        """Full prompt size: uncached + cache reads + cache writes."""
        return self.input_tokens + self.cache_read_tokens + self.cache_write_tokens

    @property
    def text(self) -> str:
        return "".join(
            b.get("text", "") for b in self.content if b.get("type") == "text"
        )


def text_response(text: str, **kw: Any) -> LLMResponse:
    """Convenience: an end_turn text response."""
    return LLMResponse(
        [{"type": "text", "text": text}],
        kw.pop("stop_reason", "end_turn"),
        kw.pop("input_tokens", 0),
        kw.pop("output_tokens", 0),
        **kw,
    )


def tool_response(
    name: str,
    tool_input: dict,
    *,
    text: str = "",
    tool_id: str | None = None,
    **kw: Any,
) -> LLMResponse:
    """Convenience: a tool_use response (optionally preceded by text)."""
    blocks: list[dict] = [{"type": "text", "text": text}] if text else []
    blocks.append(
        {
            "type": "tool_use",
            "id": tool_id or f"toolu_{uuid.uuid4().hex[:12]}",
            "name": name,
            "input": tool_input,
        }
    )
    return LLMResponse(
        blocks, "tool_use", kw.pop("input_tokens", 0), kw.pop("output_tokens", 0), **kw
    )


def cost_usd(model: str, resp: LLMResponse) -> float:
    if resp.billed_usd is not None:
        return resp.billed_usd
    pin, pout = PRICES.get(model, DEFAULT_PRICE)
    return (
        resp.input_tokens * pin
        + resp.cache_read_tokens * pin * CACHE_READ_MULT
        + resp.cache_write_tokens * pin * CACHE_WRITE_MULT
        + resp.output_tokens * pout
    ) / 1e6


class UsageLedger:
    """Accumulates tokens and dollar cost per purpose ("agent", "compactor", ...)."""

    FIELDS = (
        "calls",
        "input_tokens",
        "output_tokens",
        "cache_read_tokens",
        "cache_write_tokens",
        "cost_usd",
    )

    def __init__(self) -> None:
        self._by_purpose: dict[str, dict[str, float]] = {}
        self._lock = threading.Lock()

    def record(self, purpose: str, model: str, resp: LLMResponse) -> None:
        with self._lock:
            row = self._by_purpose.setdefault(purpose, dict.fromkeys(self.FIELDS, 0))
            row["calls"] += 1
            row["input_tokens"] += resp.input_tokens
            row["output_tokens"] += resp.output_tokens
            row["cache_read_tokens"] += resp.cache_read_tokens
            row["cache_write_tokens"] += resp.cache_write_tokens
            row["cost_usd"] += cost_usd(model, resp)

    @property
    def total_cost(self) -> float:
        return sum(r["cost_usd"] for r in self._by_purpose.values())

    def snapshot(self) -> dict:
        with self._lock:
            by = {p: dict(r) for p, r in self._by_purpose.items()}
        total = {f: sum(r[f] for r in by.values()) for f in self.FIELDS}
        return {"by_purpose": by, "total": total}


class LLMClient(Protocol):
    usage: UsageLedger

    def create(
        self,
        *,
        model: str,
        system: str,
        messages: list[dict],
        tools: list[dict] | None = None,
        max_tokens: int = 4096,
        purpose: str = "agent",
        output_format: dict | None = None,
    ) -> LLMResponse: ...


class AnthropicLLM:
    """Anthropic SDK client: retries, prompt caching, adaptive-thinking handling."""

    def __init__(
        self,
        client: Any = None,
        *,
        effort: str | None = "low",
        max_retries: int = 3,
        rate_limit_retries: int = 6,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        import anthropic

        self._anthropic = anthropic
        # SDK-level retries cover connection errors / 5xx; we add our own 429 backoff.
        self.client = client or anthropic.Anthropic(max_retries=max_retries)
        self.effort = effort
        self.rate_limit_retries = rate_limit_retries
        self._sleep = sleep
        self.usage = UsageLedger()

    def _request(self, kw: dict) -> Any:
        a = self._anthropic
        delay = 2.0
        for attempt in range(self.rate_limit_retries + 1):
            try:
                return self.client.messages.create(**kw)
            except (a.RateLimitError, a.InternalServerError) as e:
                if attempt >= self.rate_limit_retries:
                    raise
                retry_after = None
                headers = getattr(getattr(e, "response", None), "headers", None)
                if headers is not None:
                    try:
                        retry_after = float(headers.get("retry-after"))
                    except (TypeError, ValueError):
                        retry_after = None
                self._sleep(min(60.0, retry_after if retry_after else delay))
                delay *= 2
        raise AssertionError("unreachable")

    def create(
        self,
        *,
        model: str,
        system: str,
        messages: list[dict],
        tools: list[dict] | None = None,
        max_tokens: int = 4096,
        purpose: str = "agent",
        output_format: dict | None = None,
    ) -> LLMResponse:
        kw: dict[str, Any] = {
            "model": model,
            "max_tokens": max_tokens,
            "system": system,
            "messages": messages,
            "cache_control": {
                "type": "ephemeral"
            },  # top-level: caches the last cacheable block
        }
        if tools:
            kw["tools"] = tools
            # One tool call per turn: a batch of parallel read_file calls lands in one
            # user message and can blow past context_cap before compaction can run.
            kw["tool_choice"] = {"type": "auto", "disable_parallel_tool_use": True}
        output_config: dict[str, Any] = {}
        if self.effort:
            output_config["effort"] = self.effort
        if output_format is not None:
            # Structured outputs: constrain the response to a JSON schema.
            output_config["format"] = {"type": "json_schema", "schema": output_format}
        if output_config:
            kw["output_config"] = output_config
        try:
            msg = self._request(kw)
        except self._anthropic.BadRequestError as e:
            # Fallback if the API objects to thinking blocks (e.g. signature mismatch
            # after history rewriting): drop all thinking and retry once.
            if "thinking" not in str(e).lower():
                raise
            kw["messages"] = strip_thinking(messages, keep_latest=False)
            msg = self._request(kw)
        resp = self._convert(msg)
        self.usage.record(purpose, model, resp)
        return resp

    @staticmethod
    def _convert(msg: Any) -> LLMResponse:
        blocks: list[dict] = []
        for b in msg.content:
            if b.type == "text":
                blocks.append({"type": "text", "text": b.text})
            elif b.type == "tool_use":
                blocks.append(
                    {"type": "tool_use", "id": b.id, "name": b.name, "input": b.input}
                )
            elif b.type == "thinking":
                blocks.append(
                    {
                        "type": "thinking",
                        "thinking": b.thinking,
                        "signature": b.signature,
                    }
                )
            elif b.type == "redacted_thinking":
                blocks.append({"type": "redacted_thinking", "data": b.data})
        u = msg.usage
        return LLMResponse(
            content=blocks,
            stop_reason=msg.stop_reason or "end_turn",
            input_tokens=u.input_tokens or 0,
            output_tokens=u.output_tokens or 0,
            cache_read_tokens=getattr(u, "cache_read_input_tokens", 0) or 0,
            cache_write_tokens=getattr(u, "cache_creation_input_tokens", 0) or 0,
        )


OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

_FINISH_TO_STOP = {
    "stop": "end_turn",
    "length": "max_tokens",
    "tool_calls": "tool_use",
    "content_filter": "refusal",
}


def to_openai_tools(tools: list[dict]) -> list[dict]:
    """Anthropic tool specs -> OpenAI function tools."""
    return [
        {
            "type": "function",
            "function": {
                "name": t["name"],
                "description": t.get("description", ""),
                "parameters": t.get("input_schema") or {"type": "object"},
            },
        }
        for t in tools
    ]


def to_openai_messages(system: str, messages: list[dict]) -> list[dict]:
    """Anthropic-style messages -> OpenAI chat messages. tool_use blocks become
    assistant tool_calls, tool_result blocks become `tool` messages (emitted before
    any text in the same user turn), and thinking blocks are dropped."""
    out: list[dict] = [{"role": "system", "content": system}] if system else []
    for m in messages:
        content = m["content"]
        if isinstance(content, str):
            out.append({"role": m["role"], "content": content})
            continue
        texts = [b.get("text", "") for b in content if b.get("type") == "text"]
        if m["role"] == "assistant":
            msg: dict[str, Any] = {"role": "assistant", "content": "\n".join(texts) or None}
            calls = [
                {
                    "id": b["id"],
                    "type": "function",
                    "function": {"name": b["name"], "arguments": json.dumps(b.get("input") or {})},
                }
                for b in content
                if b.get("type") == "tool_use"
            ]
            if calls:
                msg["tool_calls"] = calls
            if msg["content"] is not None or calls:
                out.append(msg)
            continue
        for b in content:
            if b.get("type") == "tool_result":
                body = _result_str(b)
                if b.get("is_error"):
                    body = f"[error] {body}"
                out.append({"role": "tool", "tool_call_id": b["tool_use_id"], "content": body})
        if texts:
            out.append({"role": "user", "content": "\n".join(texts)})
    return out


class OpenAICompatLLM:
    """OpenAI-compatible chat-completions client (OpenRouter by default), so open-weight
    models can stand in for Claude. Converts Anthropic-style messages and tools, keeps
    only the first tool call per response (not every provider honors
    parallel_tool_calls=false), and uses the provider's billed cost when it reports one.
    Qwen3-style reasoning is off unless `reasoning=True`: thinking tokens count against
    max_tokens and would truncate the agent's 2048-token responses."""

    def __init__(
        self,
        client: Any = None,
        *,
        base_url: str = OPENROUTER_BASE_URL,
        api_key: str | None = None,
        reasoning: bool = False,
        max_retries: int = 6,
    ) -> None:
        if client is None:
            import openai

            client = openai.OpenAI(base_url=base_url, api_key=api_key, max_retries=max_retries)
        self.client = client
        self.reasoning = reasoning
        self.openrouter = "openrouter.ai" in base_url
        self.usage = UsageLedger()

    def create(
        self,
        *,
        model: str,
        system: str,
        messages: list[dict],
        tools: list[dict] | None = None,
        max_tokens: int = 4096,
        purpose: str = "agent",
        output_format: dict | None = None,
    ) -> LLMResponse:
        kw: dict[str, Any] = {
            "model": model,
            "max_tokens": max_tokens,
            "messages": to_openai_messages(system, messages),
        }
        if tools:
            kw["tools"] = to_openai_tools(tools)
        if output_format is not None:
            kw["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": "output", "schema": output_format},
            }
        if self.openrouter:
            kw["extra_body"] = {
                # Route only to endpoints that support every parameter we send
                # (tools, response_format), rather than silently dropping them.
                "provider": {"require_parameters": True},
                "reasoning": {"enabled": self.reasoning, "exclude": True},
                "usage": {"include": True},
            }
        resp = self._convert(self.client.chat.completions.create(**kw))
        self.usage.record(purpose, model, resp)
        return resp

    @staticmethod
    def _convert(completion: Any) -> LLMResponse:
        choice = completion.choices[0]
        msg = choice.message
        blocks: list[dict] = []
        if msg.content:
            blocks.append({"type": "text", "text": msg.content})
        calls = msg.tool_calls or []
        if calls:
            c = calls[0]
            try:
                args = json.loads(c.function.arguments or "{}")
            except json.JSONDecodeError:
                args = {"_unparseable_arguments": c.function.arguments}
            blocks.append(
                {"type": "tool_use", "id": c.id, "name": c.function.name, "input": args}
            )
        stop = "tool_use" if calls else _FINISH_TO_STOP.get(choice.finish_reason, "end_turn")
        u = completion.usage
        prompt = (u.prompt_tokens or 0) if u else 0
        details = getattr(u, "prompt_tokens_details", None)
        cached = (getattr(details, "cached_tokens", 0) or 0) if details else 0
        billed = (getattr(u, "model_extra", None) or {}).get("cost") if u else None
        return LLMResponse(
            content=blocks,
            stop_reason=stop,
            input_tokens=prompt - cached,
            output_tokens=(u.completion_tokens or 0) if u else 0,
            cache_read_tokens=cached,
            billed_usd=float(billed) if billed is not None else None,
        )


class ScriptedLLM:
    """Test double. `script` is a list of LLMResponse (consumed in order) or a
    callable(messages, tools, purpose) -> LLMResponse. Every request is recorded in
    `self.requests` (dicts with model/system/messages/tools/max_tokens/purpose).
    If a response reports 0 tokens and `auto_usage` is set, usage is filled in from
    the estimator so ledgers are meaningful. `output_format` (a JSON schema) is
    recorded but not enforced."""

    def __init__(
        self,
        script: list[LLMResponse] | Callable[..., LLMResponse],
        *,
        auto_usage: bool = True,
    ) -> None:
        self._script = script
        self._i = 0
        self.auto_usage = auto_usage
        self.requests: list[dict] = []
        self.usage = UsageLedger()

    def create(
        self,
        *,
        model: str,
        system: str,
        messages: list[dict],
        tools: list[dict] | None = None,
        max_tokens: int = 4096,
        purpose: str = "agent",
        output_format: dict | None = None,
    ) -> LLMResponse:
        self.requests.append(
            {
                "model": model,
                "system": system,
                "messages": copy.deepcopy(messages),
                "tools": tools,
                "max_tokens": max_tokens,
                "purpose": purpose,
                "output_format": output_format,
            }
        )
        if callable(self._script):
            resp = self._script(messages, tools, purpose)
        else:
            if self._i >= len(self._script):
                raise RuntimeError("ScriptedLLM script exhausted")
            resp = self._script[self._i]
            self._i += 1
        if self.auto_usage and resp.input_tokens == 0 and resp.output_tokens == 0:
            resp = dataclasses.replace(
                resp,
                input_tokens=estimate_tokens(system)
                + estimate_tokens(tools)
                + estimate_tokens(messages),
                output_tokens=estimate_tokens(resp.content),
            )
        self.usage.record(purpose, model, resp)
        return resp
