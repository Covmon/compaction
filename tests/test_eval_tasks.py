import re
from datetime import datetime

import pytest

from compaction.eval.tasks import (
    DISABLED_MSG,
    InvestigationEnvironment,
    MockAgentPolicy,
    Question,
    grade,
    make_task,
    make_tasks,
    normalize_answer,
    run_mock_episode,
)

SEEDS = list(range(20))


@pytest.fixture(scope="module")
def tasks():
    return make_tasks(len(SEEDS), base_seed=0)


def _tools(env):
    return {t.name: t.handler for t in env.tools()}


# ---------------------------------------------------------------- generation


def test_make_tasks_deterministic():
    a = make_tasks(3, base_seed=11)
    b = make_tasks(3, base_seed=11)
    assert [t.brief for t in a] == [t.brief for t in b]
    assert [[q.to_dict() for q in t.questions] for t in a] == [[q.to_dict() for q in t.questions] for t in b]
    assert [t.seed for t in a] == [11, 12, 13]


def test_question_mix(tasks):
    for t in tasks:
        qs = t.questions
        assert len(qs) == 10
        assert len({q.qid for q in qs}) == 10
        assert len({q.answer for q in qs}) == 10
        tags = [tag for q in qs for tag in q.tags]
        assert tags.count("early") >= 5 and tags.count("late") >= 2
        assert tags.count("multi_hop") >= 3 and tags.count("single_hop") >= 4
        for q in qs:
            assert q.kind in ("exact", "numeric")
            assert all(e in t.world.files for e in q.evidence)
    all_templates = {q.template for t in tasks for q in t.questions}
    assert len(all_templates) >= 15


def test_brief_has_checklist_but_not_questions(tasks):
    t = tasks[0]
    for i, p in enumerate(t.world.reading_order):
        assert f"{i + 1}. {p}" in t.brief
    for q in t.questions:
        assert q.text not in t.brief
        assert q.answer not in t.brief


# ---------------------------------------------------------------- derivability


def _files(world, prefix):
    return {p: t for p, t in world.files.items() if p.startswith(prefix)}


def _row(text, key):
    """Cells of the markdown table row whose first cell (or any cell) equals key."""
    for line in text.splitlines():
        if line.startswith("|"):
            cells = [c.strip() for c in line.strip().strip("|").split("|")]
            if key in cells:
                return cells
    raise AssertionError(f"no row with {key}")


def _ticket_field(text, field):
    return _row(text, field)[1]


def _solve_multi(world, q: Question) -> str:
    """Re-derive multi-hop answers from file TEXT only (no ground-truth facts)."""
    files = world.files
    m = re.search(r"(INC-\d+)", q.text)
    ticket = files[f"tickets/{m.group(1)}.md"] if m else None
    if q.template == "oncall_for_port":
        port = re.search(r"port (\d+)", q.text).group(1)
        hits = [t for t in _files(world, "tickets/").values() if f"port {port}" in t]
        assert len(hits) == 1
        start = datetime.strptime(_ticket_field(hits[0], "Impact start"), "%Y-%m-%dT%H:%M:%SZ")
        for line in files["oncall/roster.md"].splitlines():
            cells = [c.strip() for c in line.strip().strip("|").split("|")]
            if len(cells) == 5 and re.fullmatch(r"W\d+", cells[0]):
                a = datetime.strptime(cells[1] + " 09:00", "%Y-%m-%d %H:%M")
                b = datetime.strptime(cells[2] + " 09:00", "%Y-%m-%d %H:%M")
                if a <= start < b:
                    return cells[3]
        raise AssertionError("no roster row")
    if q.template == "trigger_commit":
        dep = re.search(r"deploy (D-\d+)", ticket).group(1)
        rows = [ln for t in _files(world, "deploys/").values() for ln in t.splitlines()
                if ln.startswith(dep + ",")]
        assert len(rows) == 1
        return rows[0].split(",")[4]
    if q.template == "resolver_employee_id":
        h = _ticket_field(ticket, "Resolved by")
        return _row(files["people/directory.md"], h)[2]
    if q.template == "pm_due_for_resolver":
        h = _ticket_field(ticket, "Resolved by")
        pm = files[f"postmortems/PM-{m.group(1)}.md"]
        return _row(pm, h)[3]
    if q.template == "log_error_company":
        path = re.search(r"(logs/\S+\.log)", q.text).group(1)
        first = next(ln for ln in files[path].splitlines() if " ERROR " in ln)
        acc = re.search(r"account=(ACC-\d+)", first).group(1)
        line = next(ln for ln in files["db/accounts_dump.sql"].splitlines() if f"'{acc}'" in ln)
        return re.findall(r"'([^']*)'", line)[1]
    if q.template == "config_change_approver":
        svc = re.search(r"port of (\S+) was changed", q.text).group(1)
        chg = re.search(r"until (CHG-\d+)", files[f"config/{svc}.yaml"]).group(1)
        return _row(files["changes/change-calendar.md"], chg)[-1]
    raise AssertionError(f"no solver for {q.template}")


def test_answers_derivable(tasks):
    n_multi = 0
    for t in tasks:
        for q in t.questions:
            # The answer literally appears in the corpus, in one of the evidence files.
            assert any(q.answer in t.world.files[e] for e in q.evidence), (t.id, q.qid, q.answer)
            if q.hop == "multi_hop":
                n_multi += 1
                assert _solve_multi(t.world, q) == q.answer, (t.id, q.template)
    assert n_multi >= 3 * len(tasks)


def test_single_hop_answers_unambiguous(tasks):
    # Identifier-keyed single-hop questions: the key appears on exactly one line of its evidence file,
    # and that line contains the answer.
    for t in tasks:
        for q in t.questions:
            if q.hop != "single_hop":
                continue
            m = re.search(r"\b((?:D|ACC|INV-\d{4}|E)-\d+|INC-\d+|AI-\d+)\b", q.text)
            if q.template in ("deploy_commit", "account_mrr", "invoice_amount"):
                text = t.world.files[q.evidence[0]]
                lines = [ln for ln in text.splitlines() if m.group(1) in ln]
                assert len(lines) == 1 and q.answer in lines[0]


# ---------------------------------------------------------------- grading


def test_normalize_answer():
    assert normalize_answer('  "Lars  Tanaka". ') == "lars tanaka"
    assert normalize_answer("`E12345`") == "e12345"
    assert normalize_answer("@ltanaka") == "ltanaka"
    assert normalize_answer("2026-03-25 03:42:24 UTC") == normalize_answer("2026-03-25T03:42:24Z")


def _q(answer, kind="exact", tol=None):
    return Question("q1", "?", answer, kind, "single_hop", "early", "t", ["x"], tol)


def test_grade_normalization(tasks):
    t = tasks[0]
    orig = t.questions
    try:
        t.questions = [_q("Lars Tanaka"), _q("2026-03-25T03:42:24Z")]
        t.questions[1].qid = "q2"
        g = grade(t, {"q1": " lars tanaka.", "q2": "2026-03-25 03:42:24"})
        assert g["score"] == 1.0
        g = grade(t, {"q1": "Lars", "q2": "2026-03-25T03:42Z"})
        assert g["score"] == 0.0
        t.questions = [_q("11916.50", "numeric", 0.01)]
        assert grade(t, {"q1": "$11,916.50 USD"})["score"] == 1.0
        assert grade(t, {"q1": "11916.5"})["score"] == 1.0
        assert grade(t, {"q1": "11916.6"})["score"] == 0.0
        assert grade(t, {"q1": "about eleven thousand"})["score"] == 0.0
        assert grade(t, {})["score"] == 0.0
        t.questions = [_q("170", "numeric", 0)]
        assert grade(t, {"q1": "170 seconds"})["score"] == 1.0
        assert grade(t, {"q1": "171"})["score"] == 0.0
    finally:
        t.questions = orig


def test_grade_breakdown(tasks):
    t = tasks[1]
    g = grade(t, t.truth())
    assert g["score"] == 1.0 and g["correct"] == g["total"] == 10
    assert sum(v["total"] for k, v in g["by_tag"].items() if k in ("early", "late")) == 10


# ---------------------------------------------------------------- environment


def test_phase_protocol_and_tool_disabling(tasks):
    t = tasks[2]
    env = InvestigationEnvironment(t)
    tools = _tools(env)
    assert set(tools) == {"list_files", "read_file", "search", "finish_investigation", "submit_answer"}
    assert env.initial_user_message() == t.brief and env.system_prompt()
    assert env.pending_user_message() is None
    assert "Error" in tools["submit_answer"]({"answers": {"q1": "x"}})
    assert not env.done()

    first = t.checklist[0]
    assert tools["read_file"]({"path": first}).endswith(t.world.files[first])
    assert tools["read_file"]({"path": "./" + first}).startswith("=== " + first)
    assert "no such file" in tools["read_file"]({"path": "nope.txt"})
    assert first in tools["list_files"]({})

    # finish refuses while files are unread (up to max_finish_refusals), then accepts.
    r = tools["finish_investigation"]({"notes": "n"})
    assert r.startswith("Not finished:") and t.checklist[1] in r
    assert env.pending_user_message() is None
    tools["finish_investigation"]({"notes": "n"})
    r = tools["finish_investigation"]({"notes": "n"})
    assert r.startswith("Investigation closed (1/")
    assert env.phase == 2

    msg = env.pending_user_message()
    assert msg is not None
    for q in t.questions:
        assert f"{q.qid}: {q.text}" in msg
    assert env.pending_user_message() is None  # exactly once
    assert env.pending_user_message() is None

    for name, inp in (("list_files", {}), ("read_file", {"path": first}), ("search", {"query": "port"})):
        assert tools[name](inp) == DISABLED_MSG
    assert "already over" in tools["finish_investigation"]({"notes": ""})

    assert tools["submit_answer"]({"answers": '{"q1": "a", "q2": 5}'}).startswith("Answers recorded")
    assert env.done() and env.answer() == {"q1": "a", "q2": "5"}
    assert tools["submit_answer"]({"answers": {"q1": "b"}}) == "Answers were already recorded."
    assert env.answer()["q1"] == "a"

    s = env.stats()
    assert s["files_read_count"] == 1 and s["finish_refusals"] == 2
    assert s["disabled_tool_calls"] == 3
    assert s["phase1_steps"] == 2 and s["phase2_steps"] == 3
    assert s["questions_delivered"]


def test_search(tasks):
    t = tasks[0]
    env = InvestigationEnvironment(t)
    search = _tools(env)["search"]
    out = search({"query": "INFO"})
    lines = out.splitlines()
    assert len(lines) == 21 and "more hits not shown" in lines[-1]
    assert re.match(r"^\S+:\d+: ", lines[0])
    inc = next(iter(t.world.facts["incidents"]))
    out = search({"query": inc.lower()})
    assert f"tickets/{inc}.md:1: # {inc}" in out
    assert "No matches" in search({"query": "zzzz-not-there"})
    assert "Error" in search({"query": "  "})


# ---------------------------------------------------------------- mock agent


@pytest.mark.parametrize("seed", SEEDS)
def test_mock_perfect_episode(seed):
    t = make_task(seed)
    env = InvestigationEnvironment(t)
    res = run_mock_episode(env, MockAgentPolicy(t))
    assert res["grade"]["score"] == 1.0
    s = res["stats"]
    assert s["coverage"] == 1.0 and s["finish_refusals"] == 0
    assert s["actual_read_order"] == t.checklist
    assert s["tool_calls_phase2"] == 1 and s["disabled_tool_calls"] == 0
    assert res["steps"] == len(t.checklist) + 3
    # Staleness: tokens read after the oldest needed evidence; early questions are staler on average.
    st = s["question_staleness"]
    early = [st[q.qid] for q in t.questions if q.position == "early"]
    late = [st[q.qid] for q in t.questions if q.position == "late"]
    assert all(v is not None for v in st.values())
    assert sum(early) / len(early) > sum(late) / len(late)


def test_mock_variants(tasks):
    t = tasks[3]
    res = run_mock_episode(InvestigationEnvironment(t), MockAgentPolicy(t, forget_early=True))
    for p in res["grade"]["per_question"]:
        assert p["correct"] == ("late" in p["tags"])
    # Reading only a few files: finish is refused twice, the mock re-reads, and finishes.
    pol = MockAgentPolicy(t, read_limit=5)
    res = run_mock_episode(InvestigationEnvironment(t), pol)
    assert res["grade"]["score"] == 1.0
    assert res["stats"]["coverage"] == 1.0 and res["stats"]["finish_refusals"] == 1
    assert MockAgentPolicy(t).next_action() == ("list_files", {})
