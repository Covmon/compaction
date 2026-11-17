"""IndexedRetentionStrategy end to end with ScriptedLLM (no network)."""

import itertools
import json
import re
from types import SimpleNamespace as NS

from compaction.agent import Agent, AgentConfig, ToolSpec
from compaction.context import LiveContext
from compaction.llm import AnthropicLLM, ScriptedLLM, text_response, tool_response
from compaction.strategies import make_strategy
from compaction.strategies.index import OPS_SCHEMA, Index
from compaction.strategies.indexed import IndexedRetentionStrategy
from compaction.tokens import estimate_tokens
from compaction.transcript import RETRIEVED_OMITTED, Transcript


def secret(i: int) -> str:
    return f"SECRET-{i}-{(i * 7919) % 10007:05d}"


class Files:
    """read(i) returns a large blob containing a unique exact value."""

    def __init__(self, blob=1500):
        self.blob, self._done, self._ans = blob, False, None

    def system_prompt(self):
        return "You are a test agent."

    def initial_user_message(self):
        return "Read files 0..N and report the secret code of file 0."

    def _read(self, a):
        return f"FILE {a['i']}: code {secret(a['i'])}. " + ("lorem ipsum " * (self.blob // 12))

    def _submit(self, a):
        self._done, self._ans = True, a
        return "submitted"

    def tools(self):
        obj = {"type": "object", "properties": {}}
        return [ToolSpec("read", "read", obj, self._read), ToolSpec("submit_answer", "s", obj, self._submit)]

    def done(self):
        return self._done

    def answer(self):
        return self._ans


HDR = re.compile(r"^\[(m\d{4}) (user|assistant)\] ", re.MULTILINE)


def smart_indexer(prompt: str) -> str:
    """Scripted indexer: one new section per call, one subsection per message pair,
    summarised from the tool call it contains ("Read file i")."""
    ids = re.search(r"^Ids to index: (.*)$", prompt, re.MULTILINE).group(1).split(", ")
    body = prompt.split("<new_messages>\n", 1)[1].split("\n</new_messages>", 1)[0]
    chunks = {m.group(1): body[m.start() : nxt.start() if nxt else len(body)]
              for m, nxt in itertools.zip_longest(HDR.finditer(body), list(HDR.finditer(body))[1:])}
    k = prompt.count("\n## ") + 1
    ops = [{"op": "add_section", "title": f"Batch {k}", "id": None, "section": None, "summary": None, "ranges": None}]
    groups: list[list[str]] = []
    for i in ids:  # a read call starts a new group; its result joins it
        if not groups or "tool_use read(" in chunks.get(i, "") or "tool_result" not in chunks.get(i, ""):
            groups.append([i])
        else:
            groups[-1].append(i)
    for g in groups:
        m = re.search(r'read\(\{"i": (\d+)\}\)', "".join(chunks.get(i, "") for i in g))
        summ = f"Read file {m.group(1)}" if m else "Task / other messages"
        ops.append({"op": "add_subsection", "section": f"Batch {k}", "summary": summ,
                    "ranges": [f"{g[0]}-{g[-1]}"], "id": None, "title": None})
    return json.dumps({"ops": ops})


def agent_script(n_reads, more_reads=8, indexer=smart_indexer):
    """Reads n_reads files; then finds file 0 in the index and fetches it; then reads
    `more_reads` more files (forcing compactions that include the retrieved message);
    then submits the secret it saw in the fetched content."""
    st = {"i": 0, "phase": "read", "after": 0, "found": None}

    def fn(messages, tools, purpose):
        if purpose == "compactor":
            return text_response(indexer(messages[0]["content"]))
        last = json.dumps(messages[-1]["content"])
        if st["phase"] == "read":
            if st["i"] < n_reads:
                st["i"] += 1
                return tool_response("read", {"i": st["i"] - 1})
            st["phase"] = "fetch"
            notice = json.dumps(messages)
            sub = re.search(r"### (\d+\.\d+) Read file 0 ", notice).group(1)
            return tool_response("fetch_messages", {"ids": [sub]})
        if st["phase"] == "fetch":
            m = re.search(r"SECRET-0-\d{5}", last)
            st["found"] = m.group(0) if m else None
            st["phase"] = "more"
        if st["after"] < more_reads:
            st["after"] += 1
            return tool_response("read", {"i": 100 + st["after"]})
        return tool_response("submit_answer", {"code": st["found"]})

    return fn


def check_requests(llm, cfg):
    reqs = [r for r in llm.requests if r["purpose"] == "agent"]
    for r in reqs:
        est = estimate_tokens(r["system"]) + estimate_tokens(r["tools"]) + estimate_tokens(r["messages"])
        assert est <= cfg.context_cap
        msgs = r["messages"]
        assert msgs[0]["role"] == "user" and msgs[-1]["role"] == "user"
        assert all(a["role"] != b["role"] for a, b in itertools.pairwise(msgs))
    return reqs


def run(tmp_path=None, indexer=smart_indexer, n_reads=14, more_reads=8, **kw):
    cfg = AgentConfig(context_cap=3000, keep_last=4, max_steps=80)
    llm = ScriptedLLM(agent_script(n_reads, more_reads, indexer))
    strat = make_strategy("indexed", **kw)
    res = Agent(llm, Files(), strat, cfg, transcript_dir=tmp_path).run()
    return cfg, llm, strat, res


def unindexable(strat, transcript, msg):
    names = strat._tool_names()
    return msg.synthetic or strat._only_retrieved(msg, names)


def test_full_run_fetch_recovers_early_exact_value(tmp_path):
    cfg, llm, strat, res = run(tmp_path)
    assert res.error is None, res.error
    assert len(res.compactions) >= 3
    assert not res.hit_cap_violation
    check_requests(llm, cfg)
    # the early exact value was recovered verbatim after it left the context
    assert res.answer == {"code": secret(0)}
    T = res.transcript
    fetch_results = [m for m in T.messages if m.retrieved]
    assert len(fetch_results) == 1 and secret(0) in json.dumps(fetch_results[0].content)
    # the fetch happened after file 0 had been compacted out of the context
    assert res.compactions[0].step < next(
        i for i, r in enumerate(llm.requests) if r["purpose"] == "agent" and "fetch_messages" in json.dumps(r["messages"][-2:])
    )
    # tools + addendum exposed to the agent
    req0 = next(r for r in llm.requests if r["purpose"] == "agent")
    assert {t["name"] for t in req0["tools"]} >= {"read_index", "fetch_messages"}
    assert "fetch_messages" in req0["system"]
    # structured outputs requested for the indexer
    idx_reqs = [r for r in llm.requests if r["purpose"] == "compactor"]
    assert idx_reqs and all(r["output_format"] == OPS_SCHEMA for r in idx_reqs)
    assert all(r["model"] == "claude-haiku-5-5" for r in idx_reqs)
    assert res.usage["by_purpose"]["compactor"]["calls"] == len(idx_reqs)
    # resume notice + full index shown right after compaction
    after = [r for r in llm.requests if r["purpose"] == "agent" and "Context was reset" in json.dumps(r["messages"])]
    assert after and "# Index" in json.dumps(after[0]["messages"][0])
    ev = res.compactions[0]
    assert ev.detail["subsections_added"] > 0 and ev.detail["auto_covered"] == 0
    assert ev.compactor_input_tokens > 0
    # index persisted next to the transcript
    assert (tmp_path / "index.md").read_text().startswith("# Index")
    assert Index.from_json(json.loads((tmp_path / "index.json").read_text())).render() == strat.index.render()


def test_nothing_leaves_context_unindexed():
    _, _, strat, res = run()
    T = res.transcript
    covered = strat.index.covered()
    # Every message that was ever compacted away (i.e. before the last kept tail) is
    # indexed unless it is synthetic or a pure fetch copy.
    assert len(res.compactions) >= 3
    last_tail_start = int(res.compactions[-1].detail["tail_start"][1:])
    for m in T.messages[: last_tail_start - 1]:
        if not unindexable(strat, T, m):
            assert int(m.id[1:]) in covered, m.id


def test_retrieved_messages_never_reindexed():
    _, llm, strat, res = run()
    T = res.transcript
    fetched = next(m for m in T.messages if m.retrieved)
    idx_prompts = [r["messages"][0]["content"] for r in llm.requests if r["purpose"] == "compactor"]
    # secret(0) is in the original read result: indexed exactly once
    assert sum(secret(0) in p for p in idx_prompts) == 1
    # the fetch result itself was compacted away later, but never sent to the indexer
    assert int(fetched.id[1:]) < int(T.messages[-1].id[1:])
    assert all(fetched.id not in re.search(r"^Ids to index: (.*)$", p, re.MULTILINE).group(1) for p in idx_prompts)
    assert sum(e.detail["skipped_retrieved"] for e in res.compactions) >= 1
    # fetching the original plus the retrieved copy returns the original only once
    out = strat._fetch({"ids": ["m0002-m0003", fetched.id]})
    assert out.count(secret(0)) == 1
    assert fetched.id not in re.findall(r"^\[(m\d{4}) ", out, re.MULTILINE)


def test_garbage_indexer_output_falls_back_to_auto_coverage():
    def script(messages, tools, purpose):
        if purpose == "compactor":
            return text_response("garbage {not json")
        n = sum(1 for r in llm2.requests if r["purpose"] == "agent") - 1
        return tool_response("read", {"i": n}) if n <= 12 else tool_response("submit_answer", {"ok": 1})

    cfg = AgentConfig(context_cap=3000, keep_last=4, max_steps=40)
    llm2 = ScriptedLLM(script)
    strat2 = make_strategy("indexed")
    res2 = Agent(llm2, Files(), strat2, cfg).run()
    assert res2.error is None and res2.answer == {"ok": 1}
    assert len(res2.compactions) >= 2
    check_requests(llm2, cfg)
    for ev in res2.compactions:
        assert ev.detail["auto_covered"] == ev.detail["indexed_messages"] > 0
        assert ev.detail["indexer_failures"]
    auto = strat2.index.render()
    assert "(auto)" in auto and 'read({"i": 0})' in auto
    # auto subsections are fetchable
    assert secret(0) in strat2._fetch({"ids": ["1.1", "1.2"]})


def test_indexer_exception_and_client_without_output_format():
    class OldClient(ScriptedLLM):
        def create(self, *, model, system, messages, tools=None, max_tokens=4096, purpose="agent"):
            return super().create(model=model, system=system, messages=messages, tools=tools,
                                  max_tokens=max_tokens, purpose=purpose)

    def script(messages, tools, purpose):
        if purpose == "compactor":
            return text_response(smart_indexer(messages[0]["content"]))
        n = sum(1 for r in llm.requests if r["purpose"] == "agent")
        return tool_response("read", {"i": n}) if n <= 10 else tool_response("submit_answer", {"ok": 1})

    llm = OldClient(script)
    res = Agent(llm, Files(), make_strategy("indexed"), AgentConfig(context_cap=3000)).run()
    assert res.error is None and res.compactions
    assert all(ev.detail["auto_covered"] == 0 for ev in res.compactions)

    def boom(messages, tools, purpose):
        if purpose == "compactor":
            raise RuntimeError("overloaded")
        n = sum(1 for r in llm2.requests if r["purpose"] == "agent")
        return tool_response("read", {"i": n}) if n <= 10 else tool_response("submit_answer", {"ok": 1})

    llm2 = ScriptedLLM(boom)
    res2 = Agent(llm2, Files(), make_strategy("indexed"), AgentConfig(context_cap=3000)).run()
    assert res2.error is None and res2.compactions
    assert "overloaded" in res2.compactions[0].detail["indexer_failures"][0]
    assert res2.compactions[0].detail["auto_covered"] > 0


# ---- strategy-level unit tests on a hand-built transcript ----------------------
def bound(cap=3000, **kw):
    T = Transcript()
    s = IndexedRetentionStrategy(**kw)
    s.bind(ScriptedLLM([]), T, AgentConfig(context_cap=cap))
    return T, s


def test_fetch_budget_truncation():
    T, s = bound(cap=2000)  # budget = 700 tokens
    T.append("user", "task", session=0)
    for i in range(10):
        T.append("assistant", [{"type": "tool_use", "id": f"t{i}", "name": "read", "input": {"i": i}}], session=0)
        T.append("user", [{"type": "tool_result", "tool_use_id": f"t{i}", "content": f"FILE {i} " + "x" * 900}], session=0)
    s.index.apply([{"op": "add_section", "title": "Reads"}] + [
        {"op": "add_subsection", "section": "1", "summary": f"Read file {i}", "ranges": [f"m{2 + 2 * i:04d}-m{3 + 2 * i:04d}"]}
        for i in range(10)
    ])
    out = s._fetch({"ids": ["1"]})
    assert "[TRUNCATED" in out and "Fetch narrower ids" in out
    assert estimate_tokens(out) <= s.fetch_budget + 80
    assert "FILE 0 " in out and "FILE 9 " not in out
    assert "=== 1.1 Read file 0 (Reads)" in out
    # narrow fetch fits; unknown ids give a helpful error; partial unknown is noted
    assert "[TRUNCATED" not in s._fetch({"ids": ["1.10"]})
    err = s._fetch({"ids": ["7.7"]})
    assert err.startswith("Error: unknown index id(s): 7.7") and "1.1" in err
    assert "unknown ids ignored: 9" in s._fetch({"ids": ["1.2", "9"]})
    assert s._fetch({"ids": "1.2"}).count("FILE 1 ") == 1  # a bare string is accepted
    assert "Error" in s._fetch({})


def test_parallel_fetch_and_read_in_one_retrieved_message():
    """A user message holding both a fetch_messages result and a read result is
    marked retrieved by the agent; only the fetch body is hidden from the indexer."""
    T, s = bound()
    T.append("user", "task", session=0)
    T.append("assistant", [{"type": "tool_use", "id": "a", "name": "read", "input": {"i": 0}}], session=0)
    T.append("user", [{"type": "tool_result", "tool_use_id": "a", "content": "ORIGINAL-0"}], session=0)
    T.append("assistant", [
        {"type": "tool_use", "id": "f", "name": "fetch_messages", "input": {"ids": ["1.1"]}},
        {"type": "tool_use", "id": "r", "name": "read", "input": {"i": 1}},
    ], session=0)
    mixed = T.append("user", [
        {"type": "tool_result", "tool_use_id": "f", "content": "COPY ORIGINAL-0"},
        {"type": "tool_result", "tool_use_id": "r", "content": "ORIGINAL-1"},
    ], session=0, retrieved=True)
    T.append("assistant", [{"type": "tool_use", "id": "f2", "name": "fetch_messages", "input": {"ids": ["1"]}}], session=0)
    pure = T.append("user", [{"type": "tool_result", "tool_use_id": "f2", "content": "COPY2 ORIGINAL-0"}],
                    session=0, retrieved=True)
    for _ in range(4):
        T.append("assistant", "thinking", session=0)
        T.append("user", "ok", session=0)
    seen = []

    def indexer(messages, tools, purpose):
        seen.append(messages[0]["content"])
        return text_response('{"ops": []}')

    s.llm = ScriptedLLM(indexer)
    ctx = LiveContext([m.id for m in T.messages])
    new, ev = s.compact(ctx, 0, 5)
    p = seen[0]
    assert "ORIGINAL-1" in p and "COPY" not in p and RETRIEVED_OMITTED in p
    assert mixed.id in p and pure.id not in re.search(r"^Ids to index: (.*)$", p, re.MULTILINE).group(1)
    assert ev.detail["skipped_retrieved"] == 1
    assert ev.detail["auto_covered"] == ev.detail["indexed_messages"]
    # fetch output: mixed message shown with the copy hidden, pure copy excluded
    out = s._fetch({"ids": ["m0001-m0007"]})
    assert "ORIGINAL-1" in out and "COPY" not in out and f"[{pure.id} " not in out
    assert new.ids[1:] == ctx.ids[-4:]
    assert T.get(new.ids[0]).synthetic and "Context was reset" in T.get(new.ids[0]).content[0]["text"]


def test_indexer_chunking_sees_updated_index():
    T, s = bound(max_indexer_input_tokens=4000)
    T.append("user", "task", session=0)
    for i in range(12):
        T.append("assistant", [{"type": "tool_use", "id": f"t{i}", "name": "read", "input": {"i": i}}], session=0)
        T.append("user", [{"type": "tool_result", "tool_use_id": f"t{i}", "content": "y" * 3000}], session=0)
    prompts = []

    def indexer(messages, tools, purpose):
        prompts.append(messages[0]["content"])
        return text_response(smart_indexer(messages[0]["content"]))

    s.llm = ScriptedLLM(indexer)
    new, ev = s.compact(LiveContext([m.id for m in T.messages]), 0, 1)
    assert ev.detail["indexer_calls"] == len(prompts) >= 3
    assert all(estimate_tokens(p) < 4000 for p in prompts)
    assert "## 1. Batch 1" in prompts[1]  # later chunks see the index built so far
    assert ev.detail["auto_covered"] == 0
    tail = set(new.ids[1:])
    assert s.index.coverage([m.id for m in T.messages if m.id not in tail and not m.synthetic])[1] == []


def test_noop_when_head_has_nothing_new():
    T, s = bound()
    T.append("user", "task", session=0)
    T.append("assistant", "hi", session=0)
    ctx = LiveContext([m.id for m in T.messages])
    new, ev = s.compact(ctx, 0, 1)
    assert new.ids == ctx.ids and ev.detail.get("noop")
    assert "empty" in s._read_index({})


def test_anthropic_output_format_merges_with_effort():
    class Fake:
        def __init__(self):
            self.calls, self.messages = [], self

        def create(self, **kw):
            self.calls.append(kw)
            return NS(content=[NS(type="text", text='{"ops": []}')], stop_reason="end_turn",
                      usage=NS(input_tokens=1, output_tokens=1, cache_read_input_tokens=0,
                               cache_creation_input_tokens=0))

    c = Fake()
    AnthropicLLM(c, effort="low").create(model="claude-haiku-5-5", system="s", messages=[], output_format=OPS_SCHEMA)
    assert c.calls[0]["output_config"] == {"effort": "low", "format": {"type": "json_schema", "schema": OPS_SCHEMA}}
    AnthropicLLM(c, effort=None).create(model="claude-haiku-5-5", system="s", messages=[])
    assert "output_config" not in c.calls[1]


def test_parallel_fetch_and_env_tool_through_agent():
    """A turn with parallel [fetch_messages, read]: the agent stores the fetch result in
    its own retrieved message; the read result is indexed, the fetch copy is not."""
    from compaction.llm import LLMResponse

    st = {"n": 0}

    def script(messages, tools, purpose):
        if purpose == "compactor":
            return text_response(smart_indexer(messages[0]["content"]))
        st["n"] += 1
        n = st["n"]
        if n <= 8:
            return tool_response("read", {"i": n - 1})
        if n == 9:
            return LLMResponse([
                {"type": "tool_use", "id": "tf", "name": "fetch_messages", "input": {"ids": ["m0002-m0003"]}},
                {"type": "tool_use", "id": "tr", "name": "read", "input": {"i": 50}},
            ], "tool_use", 0, 0)
        if n <= 18:
            return tool_response("read", {"i": 100 + n})
        return tool_response("submit_answer", {"ok": 1})

    cfg = AgentConfig(context_cap=3000, keep_last=4, max_steps=40)
    llm = ScriptedLLM(script)
    strat = make_strategy("indexed")
    res = Agent(llm, Files(), strat, cfg).run()
    assert res.error is None and res.answer == {"ok": 1}
    check_requests(llm, cfg)
    T = res.transcript
    fetched = next(m for m in T.messages if m.retrieved)
    fresh = T.messages[T.messages.index(fetched) - 1]
    assert not fresh.retrieved and secret(50) in json.dumps(fresh.content)
    assert [b["tool_use_id"] for b in fetched.content] == ["tf"]
    assert secret(0) in json.dumps(fetched.content)
    prompts = [r["messages"][0]["content"] for r in llm.requests if r["purpose"] == "compactor"]
    ids_lines = [re.search(r"^Ids to index: (.*)$", p, re.MULTILINE).group(1) for p in prompts]
    # both messages have left the context and been through compaction
    assert int(res.compactions[-1].detail["tail_start"][1:]) > int(fetched.id[1:])
    assert any(fresh.id in line for line in ids_lines)  # env result indexed...
    assert sum(secret(50) in p for p in prompts) == 1
    assert not any(fetched.id in line for line in ids_lines)  # ...fetch copy not
    assert sum(secret(0) in p for p in prompts) == 1  # only the original read of file 0
    assert not any("Original messages for" in p for p in prompts)
    assert int(fresh.id[1:]) in strat.index.covered()
