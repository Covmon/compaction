"""OpenAICompatLLM with a fake OpenAI client (no network)."""

from types import SimpleNamespace as NS

from compaction.llm import OpenAICompatLLM, to_openai_messages, to_openai_tools


def completion(content=None, calls=(), finish="stop", cost=0.0012, cached=3):
    tool_calls = [
        NS(id=i, function=NS(name=n, arguments=args)) for i, n, args in calls
    ] or None
    return NS(
        choices=[NS(message=NS(content=content, tool_calls=tool_calls), finish_reason=finish)],
        usage=NS(
            prompt_tokens=100,
            completion_tokens=7,
            prompt_tokens_details=NS(cached_tokens=cached),
            model_extra={"cost": cost},
        ),
    )


class FakeClient:
    def __init__(self, outcomes):
        self.outcomes, self.calls = list(outcomes), []
        self.chat = NS(completions=self)

    def create(self, **kw):
        self.calls.append(kw)
        return self.outcomes.pop(0)


def test_message_conversion():
    msgs = [
        {"role": "user", "content": [{"type": "text", "text": "brief"}]},
        {
            "role": "assistant",
            "content": [
                {"type": "thinking", "thinking": "t", "signature": "s"},
                {"type": "text", "text": "reading"},
                {"type": "tool_use", "id": "c1", "name": "read_file", "input": {"path": "a"}},
            ],
        },
        {
            "role": "user",
            "content": [
                {"type": "tool_result", "tool_use_id": "c1", "content": "A", "is_error": True},
                {"type": "text", "text": "questions"},
            ],
        },
    ]
    out = to_openai_messages("sys", msgs)
    assert out == [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "brief"},
        {
            "role": "assistant",
            "content": "reading",
            "tool_calls": [
                {
                    "id": "c1",
                    "type": "function",
                    "function": {"name": "read_file", "arguments": '{"path": "a"}'},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "c1", "content": "[error] A"},
        {"role": "user", "content": "questions"},
    ]
    assert to_openai_messages("", [{"role": "user", "content": "plain"}]) == [
        {"role": "user", "content": "plain"}
    ]


def test_tool_conversion():
    t = {"name": "f", "description": "d", "input_schema": {"type": "object"}}
    assert to_openai_tools([t]) == [
        {"type": "function", "function": {"name": "f", "description": "d", "parameters": {"type": "object"}}}
    ]


def test_keeps_first_tool_call_and_billed_cost():
    c = FakeClient([
        completion(
            "ok",
            calls=[("c1", "read_file", '{"path": "a"}'), ("c2", "read_file", '{"path": "b"}')],
            finish="tool_calls",
        )
    ])
    llm = OpenAICompatLLM(c, base_url="http://localhost:11434/v1")
    r = llm.create(model="qwen/qwen3-14b", system="s", messages=[], tools=[{"name": "read_file"}])
    assert r.content == [
        {"type": "text", "text": "ok"},
        {"type": "tool_use", "id": "c1", "name": "read_file", "input": {"path": "a"}},
    ]
    assert r.stop_reason == "tool_use"
    assert (r.input_tokens, r.cache_read_tokens, r.total_input_tokens) == (97, 3, 100)
    snap = llm.usage.snapshot()["total"]
    assert snap["cost_usd"] == 0.0012 and snap["calls"] == 1
    assert "extra_body" not in c.calls[0]  # not OpenRouter: no OpenRouter-only fields


def test_openrouter_extras_and_structured_output():
    c = FakeClient([completion('{"ops": []}', finish="length")])
    llm = OpenAICompatLLM(c)
    r = llm.create(model="m", system="s", messages=[], output_format={"type": "object"})
    kw = c.calls[0]
    assert kw["response_format"]["json_schema"]["schema"] == {"type": "object"}
    assert kw["extra_body"]["provider"] == {"require_parameters": True}
    assert kw["extra_body"]["reasoning"]["enabled"] is False
    assert r.stop_reason == "max_tokens" and r.text == '{"ops": []}'


def test_unparseable_tool_arguments():
    c = FakeClient([completion(calls=[("c1", "read_file", "{bad")], finish="tool_calls")])
    r = OpenAICompatLLM(c).create(model="m", system="s", messages=[], tools=[{"name": "read_file"}])
    assert r.content == [
        {"type": "tool_use", "id": "c1", "name": "read_file", "input": {"_unparseable_arguments": "{bad"}}
    ]
