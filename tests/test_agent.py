"""End-to-end agent loop with ScriptedLLM and a toy environment."""

import itertools

import pytest

from compaction.agent import Agent, AgentConfig, ToolSpec
from compaction.llm import LLMResponse, ScriptedLLM, text_response, tool_response
from compaction.strategies import make_strategy
from compaction.tokens import estimate_tokens


class Toy:
    """Reading `read(i)` returns a large blob; submit_answer finishes.
    Optionally reveals a second-phase message after `finish` is called."""

    def __init__(self, blob=1500, phase2=None):
        self.blob, self.phase2 = blob, phase2
        self._done, self._ans, self._finished, self._revealed = (
            False,
            None,
            False,
            False,
        )

    def system_prompt(self):
        return "You are a test agent."

    def initial_user_message(self):
        return "Read everything."

    def _read(self, a):
        return f"FILE {a['i']}: " + ("lorem ipsum " * (self.blob // 12))

    def _finish(self, a):
        self._finished = True
        return "finished"

    def _submit(self, a):
        self._done, self._ans = True, a
        return "submitted"

    def tools(self):
        obj = {"type": "object", "properties": {}}
        t = [
            ToolSpec("read", "read", obj, self._read),
            ToolSpec("submit_answer", "s", obj, self._submit),
        ]
        if self.phase2:
            t.append(ToolSpec("finish", "f", obj, self._finish))
        return t

    def done(self):
        return self._done

    def answer(self):
        return self._ans

    def pending_user_message(self):
        if self.phase2 and self._finished and not self._revealed:
            self._revealed = True
            return self.phase2
        return None


def reader_script(n_reads, final=("submit_answer", {"x": "1"}), phase2=False):
    """Callable script: do n_reads reads, [finish], then submit; summaries for compactor."""
    state = {"i": 0}

    def fn(messages, tools, purpose):
        if purpose == "compactor":
            return text_response("SUMMARY: read files with ids m1 etc.")
        i = state["i"]
        state["i"] += 1
        if i < n_reads:
            return tool_response("read", {"i": i})
        if phase2 and i == n_reads:
            return tool_response("finish", {})
        return tool_response(*final)

    return fn


@pytest.mark.parametrize("strategy", ["truncate", "summary"])
def test_forced_compactions_and_cap(strategy):
    cfg = AgentConfig(context_cap=3000, keep_last=4, max_steps=60)
    llm = ScriptedLLM(reader_script(14))
    res = Agent(llm, Toy(), make_strategy(strategy), cfg).run()
    assert res.error is None, res.error
    assert res.answer == {"x": "1"}
    assert len(res.compactions) >= 3
    assert not res.hit_cap_violation
    agent_reqs = [r for r in llm.requests if r["purpose"] == "agent"]
    assert len(agent_reqs) == res.steps
    for r in agent_reqs:
        est = (
            estimate_tokens(r["system"])
            + estimate_tokens(r["tools"])
            + estimate_tokens(r["messages"])
        )
        assert est <= cfg.context_cap
        msgs = r["messages"]
        assert msgs[0]["role"] == "user" and msgs[-1]["role"] == "user"
        assert all(a["role"] != b["role"] for a, b in itertools.pairwise(msgs))
    for ev in res.compactions:
        assert ev.tokens_after < ev.tokens_before
    if strategy == "summary":
        assert res.usage["by_purpose"]["compactor"]["calls"] == len(res.compactions)
        assert any("SUMMARY:" in str(r["messages"]) for r in agent_reqs)
    else:
        assert "compactor" not in res.usage["by_purpose"]
        assert any("truncated" in str(r["messages"]) for r in agent_reqs)


def test_hard_cap_enforced_when_tail_is_huge():
    # Each blob alone is ~ cap/2: kept tail can't fit; harness must drop / shrink.
    cfg = AgentConfig(context_cap=2500, keep_last=6, max_steps=30)
    llm = ScriptedLLM(reader_script(6))
    res = Agent(llm, Toy(blob=3000), make_strategy("truncate"), cfg).run()
    assert res.error is None, res.error
    assert res.hit_cap_violation
    for r in llm.requests:
        est = (
            estimate_tokens(r["system"])
            + estimate_tokens(r["tools"])
            + estimate_tokens(r["messages"])
        )
        assert est <= cfg.context_cap


def test_cap_unsatisfiable_reports_error_without_calling():
    cfg = AgentConfig(context_cap=50)
    llm = ScriptedLLM(reader_script(1))
    res = Agent(llm, Toy(), make_strategy("truncate"), cfg).run()
    assert res.error and res.error.startswith("cap_unsatisfiable")
    assert llm.requests == []


def test_pending_user_message_injection_and_transcript(tmp_path):
    cfg = AgentConfig(context_cap=4000, max_steps=40)
    llm = ScriptedLLM(reader_script(6, phase2=True))
    env = Toy(phase2="NOW ANSWER THE QUESTIONS")
    res = Agent(llm, env, make_strategy("summary"), cfg, transcript_dir=tmp_path).run()
    assert res.error is None and res.answer == {"x": "1"}
    reqs = [r for r in llm.requests if r["purpose"] == "agent"]
    after = [r for r in reqs if "NOW ANSWER" in str(r["messages"])]
    assert after, "pending message never reached the model"
    last = after[0]["messages"][-1]
    # injected text is merged into the same user message as the finish tool_result
    assert last["role"] == "user"
    assert [b["type"] for b in last["content"]] == ["tool_result", "text"]
    assert res.transcript_path and (tmp_path / "transcript.jsonl").exists()


def test_nudge_when_no_tool_call():
    script = [
        text_response("thinking out loud"),
        tool_response("submit_answer", {"x": "2"}),
    ]
    llm = ScriptedLLM(script)
    res = Agent(llm, Toy(), make_strategy("truncate"), AgentConfig()).run()
    assert res.answer == {"x": "2"} and res.steps == 2
    assert "Continue. Call submit_answer" in str(llm.requests[1]["messages"][-1])


def test_refusal_and_max_tokens_and_tool_errors():
    llm = ScriptedLLM([LLMResponse([], "refusal", 10, 0)])
    res = Agent(llm, Toy(), make_strategy("truncate"), AgentConfig()).run()
    assert res.error == "refusal" and res.answer is None

    cut = LLMResponse(
        [
            {"type": "text", "text": "partial"},
            {"type": "tool_use", "id": "t1", "name": "read", "input": {}},
        ],
        "max_tokens",
        5,
        5,
    )
    llm = ScriptedLLM(
        [
            cut,
            tool_response("nonexistent", {}),
            tool_response("submit_answer", {"x": "3"}),
        ]
    )
    res = Agent(llm, Toy(), make_strategy("truncate"), AgentConfig()).run()
    assert res.answer == {"x": "3"} and res.error is None
    assert "output token limit" in str(
        llm.requests[1]["messages"][-1]
    )  # truncated tool_use never executed
    assert "unknown tool" in str(llm.requests[2]["messages"][-1])


def test_max_steps_and_llm_exception():
    llm = ScriptedLLM(lambda m, t, p: tool_response("read", {"i": 0}))
    res = Agent(
        llm, Toy(blob=10), make_strategy("truncate"), AgentConfig(max_steps=3)
    ).run()
    assert res.error == "max_steps" and res.steps == 3

    def boom(m, t, p):
        raise ValueError("kaboom")

    res = Agent(
        ScriptedLLM(boom), Toy(), make_strategy("truncate"), AgentConfig()
    ).run()
    assert "kaboom" in res.error


def test_usage_in_result_matches_ledger():
    llm = ScriptedLLM(
        [
            tool_response(
                "submit_answer", {"x": "1"}, input_tokens=1000, output_tokens=100
            )
        ]
    )
    res = Agent(llm, Toy(), make_strategy("truncate"), AgentConfig()).run()
    a = res.usage["by_purpose"]["agent"]
    assert a["input_tokens"] == 1000 and a["cost_usd"] == pytest.approx(
        (1000 * 0.10 + 100 * 0.50) / 1e6
    )
