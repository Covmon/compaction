"""AnthropicLLM with a fake SDK client (no network)."""

from types import SimpleNamespace as NS

import anthropic
import httpx2 as httpx
import pytest

from compaction.llm import AnthropicLLM


def fake_msg():
    return NS(
        content=[
            NS(type="thinking", thinking="t", signature="sig"),
            NS(type="text", text="hi"),
            NS(type="tool_use", id="tu1", name="f", input={"a": 1}),
        ],
        stop_reason="tool_use",
        usage=NS(
            input_tokens=10,
            output_tokens=5,
            cache_read_input_tokens=100,
            cache_creation_input_tokens=20,
        ),
    )


def status_err(cls, code, text="err"):
    req = httpx.Request("POST", "http://x")
    return cls(text, response=httpx.Response(code, request=req), body=None)


class FakeClient:
    def __init__(self, outcomes):
        self.outcomes, self.calls = list(outcomes), []
        self.messages = self

    def create(self, **kw):
        self.calls.append(kw)
        o = self.outcomes.pop(0)
        if isinstance(o, Exception):
            raise o
        return o


def test_convert_cache_and_effort():
    c = FakeClient([fake_msg()])
    llm = AnthropicLLM(c)
    r = llm.create(
        model="claude-haiku-5-5", system="s", messages=[], tools=[{"name": "f"}]
    )
    assert [b["type"] for b in r.content] == ["thinking", "text", "tool_use"]
    assert (
        r.input_tokens,
        r.cache_read_tokens,
        r.cache_write_tokens,
        r.total_input_tokens,
    ) == (10, 100, 20, 130)
    kw = c.calls[0]
    assert kw["output_config"] == {"effort": "low"} and kw["cache_control"] == {
        "type": "ephemeral"
    }
    assert "temperature" not in kw
    assert llm.usage.snapshot()["total"]["calls"] == 1


def test_429_backoff_then_success():
    sleeps = []
    c = FakeClient(
        [
            status_err(anthropic.RateLimitError, 429),
            status_err(anthropic.RateLimitError, 429),
            fake_msg(),
        ]
    )
    llm = AnthropicLLM(c, sleep=sleeps.append)
    llm.create(model="m", system="s", messages=[])
    assert len(c.calls) == 3 and sleeps == [2.0, 4.0]


def test_429_gives_up():
    c = FakeClient([status_err(anthropic.RateLimitError, 429)] * 3)
    llm = AnthropicLLM(c, rate_limit_retries=2, sleep=lambda s: None)
    with pytest.raises(anthropic.RateLimitError):
        llm.create(model="m", system="s", messages=[])


def test_thinking_rejection_falls_back():
    th = {"type": "thinking", "thinking": "t", "signature": "x"}
    msgs = [
        {"role": "user", "content": [{"type": "text", "text": "q"}]},
        {
            "role": "assistant",
            "content": [th, {"type": "tool_use", "id": "a", "name": "f", "input": {}}],
        },
        {
            "role": "user",
            "content": [{"type": "tool_result", "tool_use_id": "a", "content": "r"}],
        },
    ]
    c = FakeClient(
        [
            status_err(anthropic.BadRequestError, 400, "invalid thinking signature"),
            fake_msg(),
        ]
    )
    AnthropicLLM(c).create(model="m", system="s", messages=msgs)
    assert len(c.calls) == 2
    assert all(
        b["type"] != "thinking" for m in c.calls[1]["messages"] for b in m["content"]
    )
