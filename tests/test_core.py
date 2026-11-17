"""Transcript, LiveContext, split_tail, ledger."""

import itertools

import pytest

from compaction.context import LiveContext, strip_thinking
from compaction.llm import (
    LLMResponse,
    ScriptedLLM,
    UsageLedger,
    cost_usd,
    text_response,
)
from compaction.strategies import make_strategy, split_tail
from compaction.tokens import estimate_tokens
from compaction.transcript import Transcript


def tu(i, name="read_file"):
    return {"type": "tool_use", "id": i, "name": name, "input": {"path": "a"}}


def tr(i, body="ok"):
    return {"type": "tool_result", "tool_use_id": i, "content": body}


def build(t, spec):
    """spec: list of (role, content). Returns LiveContext of all."""
    ctx = LiveContext()
    for role, c in spec:
        ctx.append(t.append(role, c, session=0).id)
    return ctx


def test_estimate_tokens():
    assert estimate_tokens("a" * 7) == 2
    assert estimate_tokens("") == 0
    assert estimate_tokens({"a": 1}) > 0


def test_transcript_save_load_render(tmp_path):
    t = Transcript()
    t.append("user", "hello", session=0)
    t.append("assistant", [{"type": "text", "text": "hi"}, tu("t1")], session=0)
    m3 = t.append(
        "user", [tr("t1", "BODY")], session=1, retrieved=True, synthetic=False
    )
    assert [m.id for m in t.all()] == ["m0001", "m0002", "m0003"]
    assert [m.id for m in t.range("m0002", "m0003")] == ["m0002", "m0003"]
    p = tmp_path / "x" / "t.jsonl"
    t.save(p)
    t2 = Transcript.load(p)
    assert t2.all() == t.all() and t2.get("m0003") == m3
    # ids keep increasing after load
    assert t2.append("user", "x", session=1).id == "m0004"
    out = t.render(t.all())
    assert "[m0001 user] hello" in out and "tool_use read_file" in out and "BODY" in out
    out2 = t.render(t.all(), omit_retrieved=True)
    assert "BODY" not in out2 and "[retrieved copy of earlier messages omitted]" in out2


def test_api_messages_merge_and_alternation():
    t = Transcript()
    ctx = build(
        t,
        [
            ("user", "task"),
            ("assistant", [tu("a")]),
            ("user", [tr("a")]),
            ("user", "question"),  # merges with tool_result message
        ],
    )
    msgs = ctx.to_api_messages(t)
    assert [m["role"] for m in msgs] == ["user", "assistant", "user"]
    assert [b["type"] for b in msgs[2]["content"]] == ["tool_result", "text"]


def test_api_messages_tool_results_sorted_first():
    t = Transcript()
    ctx = build(
        t,
        [
            ("user", "t"),
            ("assistant", [tu("a")]),
            ("user", "note"),
            ("user", [tr("a")]),
        ],
    )
    assert [b["type"] for b in ctx.to_api_messages(t)[2]["content"]] == [
        "tool_result",
        "text",
    ]


def test_api_messages_orphans_and_first_user():
    t = Transcript()
    ctx = build(
        t,
        [
            ("assistant", [{"type": "text", "text": "x"}, tu("a")]),
            ("user", [tr("a"), tr("zzz", "stray")]),
            ("assistant", [tu("b")]),
        ],
    )
    msgs = ctx.to_api_messages(t)
    assert msgs[0]["role"] == "user"
    roles = [m["role"] for m in msgs]
    assert all(r1 != r2 for r1, r2 in itertools.pairwise(roles))
    assert msgs[-1]["role"] == "user"
    for i, m in enumerate(msgs):
        for b in m["content"]:
            if b["type"] == "tool_result":
                assert b["tool_use_id"] in {
                    x["id"] for x in msgs[i - 1]["content"] if x["type"] == "tool_use"
                }
    # b was unanswered -> text, zzz orphan -> text
    flat = str(msgs)
    assert (
        "stray" in flat
        and "'tool_use_id': 'zzz'" not in flat
        and "'id': 'b'" not in flat
    )


def test_api_messages_drops_empty_and_truncates():
    t = Transcript()
    ctx = build(
        t,
        [
            ("user", [{"type": "text", "text": "  "}]),
            ("user", "q"),
            ("assistant", [tu("a")]),
            ("user", [tr("a", "x" * 1000)]),
        ],
    )
    ctx.max_result_chars = 50
    msgs = ctx.to_api_messages(t)
    assert msgs[0]["content"][0]["text"] == "q"
    assert len(msgs[2]["content"][0]["content"]) < 120


def test_thinking_only_kept_for_latest_tool_turn():
    th = {"type": "thinking", "thinking": "hm", "signature": "s"}
    t = Transcript()
    ctx = build(
        t,
        [
            ("user", "t"),
            ("assistant", [th, tu("a")]),
            ("user", [tr("a")]),
            ("assistant", [th, tu("b")]),
            ("user", [tr("b")]),
        ],
    )
    msgs = ctx.to_api_messages(t)
    assert msgs[1]["content"][0]["type"] == "tool_use"
    assert msgs[3]["content"][0]["type"] == "thinking"
    assert all(
        b["type"] != "thinking"
        for b in strip_thinking(msgs, keep_latest=False)[3]["content"]
    )
    # after a later user message the tool loop is "broken": nothing kept
    ctx.append(
        t.append("assistant", [th, {"type": "text", "text": "done"}], session=0).id
    )
    ctx.append(t.append("user", "more", session=0).id)
    assert not any(
        b["type"] == "thinking" for m in ctx.to_api_messages(t) for b in m["content"]
    )


def test_split_tail_boundaries():
    t = Transcript()
    spec = [("user", "task")]
    for i in range(4):
        spec += [("assistant", [tu(f"t{i}")]), ("user", [tr(f"t{i}")])]
    ctx = build(t, spec)  # 9 messages: u, (a,u)x4
    head, tail = split_tail(ctx, t, 3)  # cut at idx 6 = tool_result user -> move to 5
    assert t.get(tail[0]).role == "assistant" and len(tail) == 4
    assert head + tail == ctx.ids
    head, tail = split_tail(ctx, t, 2)  # idx 7 = assistant: fine
    assert len(tail) == 2
    head, tail = split_tail(ctx, t, 100)
    assert head == [] and tail == ctx.ids
    head, tail = split_tail(ctx, t, 0)
    assert tail == []
    # a real user message is a clean start
    t2 = Transcript()
    c2 = build(
        t2, [("user", "a"), ("assistant", "b"), ("user", "c"), ("assistant", "d")]
    )
    assert split_tail(c2, t2, 2)[1] == c2.ids[2:]


def test_ledger_cost_math():
    r = LLMResponse(
        [],
        "end_turn",
        1_000_000,
        1_000_000,
        cache_read_tokens=1_000_000,
        cache_write_tokens=1_000_000,
    )
    # 0.10 + 0.50 + 0.01 + 0.125
    assert cost_usd("claude-haiku-5-5", r) == pytest.approx(0.735)
    led = UsageLedger()
    led.record("agent", "claude-haiku-5-5", r)
    led.record(
        "compactor", "claude-haiku-5-5", LLMResponse([], "end_turn", 1_000_000, 0)
    )
    snap = led.snapshot()
    assert snap["by_purpose"]["agent"]["cost_usd"] == pytest.approx(0.735)
    assert snap["by_purpose"]["compactor"]["calls"] == 1
    assert (
        snap["total"]["cost_usd"]
        == pytest.approx(0.835)
        == pytest.approx(led.total_cost)
    )
    assert r.total_input_tokens == 3_000_000


def test_scripted_llm_modes():
    llm = ScriptedLLM([text_response("a")])
    assert (
        llm.create(
            model="m", system="s", messages=[{"role": "user", "content": "x"}]
        ).text
        == "a"
    )
    assert (
        llm.requests[0]["purpose"] == "agent"
        and llm.usage.snapshot()["total"]["calls"] == 1
    )
    with pytest.raises(RuntimeError):
        llm.create(model="m", system="s", messages=[])
    llm2 = ScriptedLLM(lambda m, tools, purpose: text_response(purpose))
    assert (
        llm2.create(model="m", system="", messages=[], purpose="compactor").text
        == "compactor"
    )


def test_make_strategy():
    assert make_strategy("truncate").name == "truncate"
    assert make_strategy("summary", summary_max_tokens=10).summary_max_tokens == 10
    with pytest.raises(ValueError):
        make_strategy("nope")
