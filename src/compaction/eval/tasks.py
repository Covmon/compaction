"""Investigation tasks over a synthetic world: questions, environment, grading.

Protocol (see DESIGN.md "Eval"):

* **Phase 1** - the initial user message is an investigation *brief*: the
  topics that will be asked about (not the questions) and a numbered
  checklist of every file in a recommended reading order. The agent reads the
  files and calls ``finish_investigation(notes)``. If checklist files are
  still unread, ``finish_investigation`` refuses (listing them) up to
  ``max_finish_refusals`` times, then accepts anyway.
* **Phase 2** - after ``finish_investigation`` succeeds, the agent loop's next
  call to ``pending_user_message()`` returns the questions (exactly once).
  From then on ``list_files`` / ``read_file`` / ``search`` return
  :data:`DISABLED_MSG`, so answers measure *retention*. The agent answers
  with ``submit_answer({"answers": {qid: str}})``; ``done()`` becomes True.

Question tags: ``early|late`` (by checklist position of the earliest-read
evidence file: first half -> early), ``single_hop|multi_hop``,
``exact|numeric``. Each question lists its ``evidence`` files so a report
can re-tag by the order the agent actually read files in
(``env.stats()["question_staleness"]``).
"""

from __future__ import annotations

import json
import random
import re
from dataclasses import dataclass, field

from compaction.eval.world import World, estimate_tokens_text, generate_world
from compaction.tools import ToolSpec

__all__ = [
    "DISABLED_MSG",
    "InvestigationEnvironment",
    "MockAgentPolicy",
    "Question",
    "Task",
    "grade",
    "make_task",
    "make_tasks",
    "normalize_answer",
    "run_mock_episode",
]

DISABLED_MSG = "Investigation phase is over; answer from what you have retained"
SEARCH_MAX_HITS = 20


# --------------------------------------------------------------------------
# Data types
# --------------------------------------------------------------------------


@dataclass
class Question:
    qid: str
    text: str
    answer: str
    kind: str  # "exact" | "numeric"
    hop: str  # "single_hop" | "multi_hop"
    position: str  # "early" | "late"
    template: str
    evidence: list[str]
    tolerance: float | None = None  # absolute tolerance for numeric answers
    evidence_index: int = 0  # checklist index (0-based) of the earliest evidence file

    @property
    def tags(self) -> list[str]:
        return [self.position, self.hop, self.kind]

    def to_dict(self) -> dict:
        return {
            "qid": self.qid, "text": self.text, "answer": self.answer, "kind": self.kind,
            "hop": self.hop, "position": self.position, "template": self.template,
            "evidence": list(self.evidence), "tolerance": self.tolerance,
            "evidence_index": self.evidence_index, "tags": self.tags,
        }


@dataclass
class Task:
    id: str
    seed: int
    world: World
    brief: str
    questions: list[Question] = field(default_factory=list)

    @property
    def checklist(self) -> list[str]:
        return list(self.world.reading_order)

    def truth(self) -> dict[str, str]:
        return {q.qid: q.answer for q in self.questions}

    def questions_message(self) -> str:
        lines = [
            "The investigation phase is over. list_files, read_file and search are now disabled.",
            "Answer the following questions using only what you have retained from the investigation.",
            ("Give exact values: identifiers, timestamps (ISO 8601 UTC as written in the files), commit "
             "hashes in full, amounts as plain numbers, people as full names unless asked otherwise."),
            "If you are unsure, give your best guess rather than leaving an answer blank.",
            "",
        ]
        for q in self.questions:
            lines.append(f"{q.qid}: {q.text}")
        lines += ["", 'Call submit_answer exactly once with {"answers": {"q1": "...", "q2": "...", ...}}.']
        return "\n".join(lines)


# --------------------------------------------------------------------------
# Question generation
# --------------------------------------------------------------------------

TOPICS = [
    ("incident tickets: exact timestamps, the deploy that triggered each incident, who resolved it, "
     "who was on call when it started"),
    "deploy logs: deploy ids, versions and full commit hashes",
    ("service configuration: production listen ports, database timeouts, the change requests that "
     "changed them and who approved those changes"),
    "service logs: error codes and the customer accounts they affected",
    "billing: account monthly recurring revenue, vendor invoice amounts",
    "runbooks: the thresholds that trigger action (not warning levels), certificate details",
    "people: employee IDs; postmortem action items, owners and due dates",
    "architecture: message bus topics and partition counts",
]


def _cands(world: World) -> list[dict]:
    """All candidate questions derivable from ``world.facts``."""
    f = world.facts
    out: list[dict] = []

    def add(template, subject, text, answer, kind, evidence, tol=None):
        out.append({"template": template, "subject": subject, "text": text, "answer": str(answer),
                    "kind": kind, "evidence": evidence, "tolerance": tol,
                    "hop": "multi_hop" if len(evidence) > 1 else "single_hop"})

    by_handle = {p["handle"]: p for p in f["people"]}

    for svc, c in f["configs"].items():
        add("config_port", svc,
            f"According to {c['path']}, what is the current production listen port of {svc}?",
            c["port"], "numeric", [c["path"]], 0)
        add("config_timeout", svc,
            f"What is the production database timeout_ms configured in {c['path']}?",
            c["timeout_ms"], "numeric", [c["path"]], 0)
        chg = f["changes"][c["change_id"]]
        add("config_change_approver", svc,
            f"The production listen port of {svc} was changed by a change request referenced in its "
            "config file. Who approved that change request (full name)?",
            chg["approved_by"], "exact", [c["path"], "changes/change-calendar.md"])

    trigger_ids = {i["trigger_deploy"] for i in f["incidents"].values()}
    for d in f["deploys"]:
        if d["deploy_id"] in trigger_ids or d["status"] != "ok":
            continue
        add("deploy_commit", d["deploy_id"],
            f"What is the full commit hash of deploy {d['deploy_id']}?",
            d["commit"], "exact", [d["path"]])

    for inc, i in f["incidents"].items():
        add("incident_start", inc,
            f"What is the impact start timestamp of {inc}?",
            i["impact_start"], "exact", [i["path"]])
        add("oncall_for_port", inc,
            f"Exactly one incident ticket says its service was refusing connections on port {i['port']}. "
            "Who was the primary on-call engineer (full name) at that incident's impact start?",
            i["oncall"]["primary"], "exact", [i["path"], "oncall/roster.md"])
        add("trigger_commit", inc,
            f"What is the full commit hash of the deploy that the {inc} ticket identifies as the trigger?",
            i["trigger_commit"], "exact", [i["path"], i["trigger_path"]])
        add("resolver_employee_id", inc,
            f"What is the employee ID of the engineer who resolved {inc}?",
            by_handle[i["resolved_by"]]["employee_id"], "exact", [i["path"], "people/directory.md"])

    for inc, pm in f["postmortems"].items():
        i = f["incidents"][inc]
        it = next(x for x in pm["items"] if x["owner"] == i["resolved_by"])
        add("pm_due_for_resolver", inc,
            f"In the postmortem for {inc}, what is the due date of the action item owned by the "
            f"engineer who resolved {inc}?",
            it["due"], "exact", [i["path"], pm["path"]])
        other = next(x for x in pm["items"] if x["owner"] != i["resolved_by"])
        add("pm_item_due", inc + other["id"],
            f"In the postmortem for {inc}, what is the due date of action item {other['id']}?",
            other["due"], "exact", [pm["path"]])

    accounts = {a["account_id"]: a for a in f["accounts"]}
    for svc, lg in f["logs"].items():
        e = lg["first_error"]
        add("log_error_code", svc,
            f"What error code appears on the first ERROR line of {lg['path']}?",
            e["code"], "exact", [lg["path"]])
        add("log_error_company", svc,
            f"Which company (as named in the accounts table) owns the account referenced on the first "
            f"ERROR line of {lg['path']}?",
            accounts[e["account"]]["company"], "exact", [lg["path"], "db/accounts_dump.sql"])

    for a in f["accounts"]:
        add("account_mrr", a["account_id"],
            f"What is the mrr_usd of account {a['account_id']} in the accounts table?",
            a["mrr_usd"], "numeric", ["db/accounts_dump.sql"], 0.01)
    for inv in f["invoices"]:
        add("invoice_amount", inv["invoice_no"],
            f"What is the amount of vendor invoice {inv['invoice_no']}?",
            inv["amount"], "numeric", ["finance/vendor-invoices.csv"], 0.01)

    rb = f["runbooks"]
    add("runbook_lag", "db-failover",
        "Per the Postgres failover runbook, replication lag above how many seconds (sustained) triggers failover?",
        rb["db-failover"]["lag"], "numeric", [rb["db-failover"]["path"]], 0)
    add("runbook_cache_pct", "cache-eviction",
        "Per the cache memory runbook, above what memory usage percentage must a controlled eviction be run?",
        rb["cache-eviction"]["pct"], "numeric", [rb["cache-eviction"]["path"]], 0)
    add("runbook_cert_days", "cert-rotation",
        "Per the certificate rotation runbook, the certificate must be rotated when it expires within how many days?",
        rb["cert-rotation"]["days"], "numeric", [rb["cert-rotation"]["path"]], 0)
    add("runbook_cert_serial", "cert-rotation-serial",
        "What is the serial number of the current edge wildcard certificate?",
        rb["cert-rotation"]["serial"], "exact", [rb["cert-rotation"]["path"]])
    add("runbook_backlog", "queue-backlog",
        "Per the dispatch queue runbook, a backlog above how many messages pages the on-call primary?",
        rb["queue-backlog"]["backlog"], "numeric", [rb["queue-backlog"]["path"]], 0)

    arch = f["architecture"]
    for topic, n in arch["partitions"].items():
        add("arch_partitions", topic,
            f"How many partitions does the {topic} topic have on the message bus?",
            n, "numeric", [arch["path"]], 0)

    for p in f["people"]:
        add("employee_id", p["handle"],
            f"What is the employee ID of {p['name']}?",
            p["employee_id"], "exact", ["people/directory.md"])
    return out


def _select(world: World, rng: random.Random, n: int) -> list[Question]:
    order = {p: i for i, p in enumerate(world.reading_order)}
    half = len(order) / 2
    cands = _cands(world)
    for c in cands:
        c["evidence_index"] = min(order[e] for e in c["evidence"])
        c["position"] = "early" if c["evidence_index"] < half else "late"
    # Interleave templates so each template is equally likely to be picked,
    # regardless of how many instances it has.
    by_tpl: dict[str, list[dict]] = {}
    for c in cands:
        by_tpl.setdefault(c["template"], []).append(c)
    tpls = sorted(by_tpl)
    rng.shuffle(tpls)
    for t in tpls:
        rng.shuffle(by_tpl[t])
    depth = max(len(v) for v in by_tpl.values())
    cands = [by_tpl[t][k] for k in range(depth) for t in tpls if k < len(by_tpl[t])]

    n_early = round(n * 0.6)
    n_multi = round(n * 0.4)
    m_early = round(n_multi * 0.6)
    want = {
        ("multi_hop", "early"): m_early,
         ("multi_hop", "late"): n_multi - m_early,
         ("single_hop", "early"): n_early - m_early,
         ("single_hop", "late"): (n - n_early) - (n_multi - m_early),
    }
    chosen: list[dict] = []
    used_tpl: dict[str, int] = {}
    used_subj: set[str] = set()
    used_ans: set[str] = set()
    for strict in (2, 1, 0):
        for c in cands:
            key = (c["hop"], c["position"])
            if want[key] <= 0 or c in chosen or c["answer"] in used_ans:
                continue
            if strict >= 1 and used_tpl.get(c["template"], 0) > 0:
                continue
            if strict >= 2 and c["subject"] in used_subj:
                continue
            chosen.append(c)
            want[key] -= 1
            used_tpl[c["template"]] = used_tpl.get(c["template"], 0) + 1
            used_subj.add(c["subject"])
            used_ans.add(c["answer"])
    # If some quota could not be met (unlucky reading order), relax the
    # position constraint first (keep the hop mix), then anything.
    for relax_hop in (False, True):
        for c in cands:
            if len(chosen) >= n:
                break
            short = [k for k, v in want.items() if v > 0]
            if c in chosen or c["answer"] in used_ans:
                continue
            if not relax_hop and not any(k[0] == c["hop"] for k in short):
                continue
            chosen.append(c)
            used_ans.add(c["answer"])
            for k in short:
                if k[0] == c["hop"] or relax_hop:
                    want[k] -= 1
                    break
    # Present in a seeded mixed order.
    rng.shuffle(chosen)
    qs = []
    for k, c in enumerate(chosen[:n]):
        qs.append(Question(
            qid=f"q{k + 1}", text=c["text"], answer=c["answer"], kind=c["kind"], hop=c["hop"],
            position=c["position"], template=c["template"], evidence=c["evidence"],
            tolerance=c["tolerance"], evidence_index=c["evidence_index"],
        ))
    return qs


def _brief(world: World, n_questions: int) -> str:
    lines = [
        f"# Investigation brief: {world.company} production operations audit",
        "",
        (f"You are auditing the production operations records of {world.company}: service registry, "
         "configs, logs, deploy logs, incident tickets, postmortems, the on-call roster, the people "
         "directory, a database dump, vendor invoices, runbooks and architecture notes."),
        "",
        "## Protocol",
        "",
        ("Phase 1 (now): read EVERY file on the checklist below with read_file, in the listed order. "
         "Files are short; read each one in full. Do not skip files: questions can be about any of them, "
         "and finish_investigation will refuse while checklist files are unread."),
        "When you have read all of them, call finish_investigation(notes) with a brief summary.",
        "",
        (f"Phase 2: you will then receive about {n_questions} questions. At that point list_files, "
         "read_file and search will be DISABLED - you must answer from what you have retained. "
         "Answer with submit_answer."),
        "",
        "## What you will be asked about",
        "",
    ]
    lines += [f"- {t}" for t in TOPICS]
    lines += [
        "",
        ("Answers must be exact (identifiers, ISO timestamps, full commit hashes, amounts, full names). "
         "Several questions join facts across files (for example: a value in one file identifies the "
         "record in another file that holds the answer). Beware of superseded, staging and warning-level "
         "values, similarly named services, and people who share a first name."),
        "",
        f"## Checklist ({len(world.reading_order)} files, recommended reading order)",
        "",
    ]
    lines += [f"{i + 1}. {p}" for i, p in enumerate(world.reading_order)]
    return "\n".join(lines)


def make_task(seed: int, n_questions: int = 10) -> Task:
    world = generate_world(seed)
    rng = random.Random(f"questions-{seed}")
    questions = _select(world, rng, n_questions)
    return Task(id=f"task-{seed:04d}", seed=seed, world=world,
                brief=_brief(world, len(questions)), questions=questions)


def make_tasks(n: int, base_seed: int = 0, n_questions: int = 10) -> list[Task]:
    """``n`` tasks with seeds ``base_seed .. base_seed + n - 1``."""
    return [make_task(base_seed + i, n_questions) for i in range(n)]


# --------------------------------------------------------------------------
# Environment
# --------------------------------------------------------------------------

SYSTEM_PROMPT = (
    "You are a meticulous operations analyst with read-only access to a company's operations "
    "records through file tools. Your conversation history is limited: older parts of it may be "
    "compacted or removed while you work. The task has two phases: an investigation phase in which "
    "you read files, and a question phase in which file access is disabled and you must answer from "
    "what you have retained. Precise identifiers, timestamps, numbers and names matter. "
    "Use tools for every action; do not stop without calling the tool the current phase requires."
)


class InvestigationEnvironment:
    """Duck-typed implementation of the agent ``Environment`` protocol.

    Methods used by the agent loop: ``system_prompt``, ``initial_user_message``,
    ``tools``, ``done``, ``answer`` and ``pending_user_message`` (polled after
    every agent step; returns the phase-2 questions exactly once).

    Extra: ``phase`` (1 or 2), ``stats()``, ``begin_phase2()`` (lets a harness
    force the question phase, e.g. when a phase-1 step budget is exhausted).
    """

    def __init__(self, task: Task, max_finish_refusals: int = 2):
        self.task = task
        self.world = task.world
        self.phase = 1
        self.max_finish_refusals = max_finish_refusals
        self._answers: dict[str, str] | None = None
        self._questions_pending = False
        self._questions_delivered = False
        self.notes: str | None = None
        # stats
        self._seq = 0  # tool calls handled
        self._polls = 0  # agent steps observed via pending_user_message
        self.phase1_steps = 0
        self.phase2_steps = 0
        self.tool_calls = {1: 0, 2: 0}
        self.disabled_calls = 0
        self.finish_refusals = 0
        self.read_events: list[dict] = []
        self.search_events: list[dict] = []
        self._tokens_read = 0  # cumulative tokens of file content returned in phase 1
        self._specs = [
            ToolSpec("list_files", "List every file path with its size in characters.",
                     {"type": "object", "properties": {}}, self._list_files),
            ToolSpec("read_file", "Return the full contents of one file.",
                     {"type": "object", "properties": {"path": {"type": "string", "description": "File path as shown by list_files."}},
                      "required": ["path"]}, self._read_file),
            ToolSpec("search", "Case-insensitive substring search over all file lines. Returns up to "
                     f"{SEARCH_MAX_HITS} hits as 'path:line: text'.",
                     {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]},
                     self._search),
            ToolSpec("finish_investigation", "End phase 1 (investigation). Call after reading every "
                     "checklist file. The questions arrive in the next message; file tools are then disabled.",
                     {"type": "object", "properties": {"notes": {"type": "string", "description": "Brief summary of the investigation."}},
                      "required": ["notes"]}, self._finish),
            ToolSpec("submit_answer", "Submit final answers (phase 2 only). Call once.",
                     {"type": "object", "properties": {"answers": {
                         "type": "object", "description": "Mapping question id (e.g. 'q1') -> answer string.",
                         "additionalProperties": {"type": "string"}}},
                      "required": ["answers"]}, self._submit),
        ]

    # -- protocol ------------------------------------------------------------
    def system_prompt(self) -> str:
        return SYSTEM_PROMPT

    def initial_user_message(self) -> str:
        return self.task.brief

    def tools(self) -> list[ToolSpec]:
        return list(self._specs)

    def done(self) -> bool:
        return self._answers is not None

    def answer(self) -> dict | None:
        return None if self._answers is None else dict(self._answers)

    def pending_user_message(self) -> str | None:
        self._polls += 1
        if self.phase == 1:
            self.phase1_steps += 1
        else:
            self.phase2_steps += 1
        if self._questions_pending and not self._questions_delivered:
            self._questions_pending = False
            self._questions_delivered = True
            return self.task.questions_message()
        return None

    def begin_phase2(self) -> None:
        if self.phase == 1:
            self.phase = 2
            self._questions_pending = True

    # -- tools ---------------------------------------------------------------
    def _tick(self) -> None:
        self._seq += 1
        self.tool_calls[self.phase] += 1

    def _list_files(self, inp: dict) -> str:
        self._tick()
        if self.phase != 1:
            self.disabled_calls += 1
            return DISABLED_MSG
        return "\n".join(f"{p}  ({len(self.world.files[p])} chars)" for p in self.world.reading_order)

    @staticmethod
    def _clean_path(p: object) -> str:
        p = str(p or "").strip().strip("'\"`")
        while p.startswith(("./", "/")):
            p = p.removeprefix("./").removeprefix("/")
        return p

    def _read_file(self, inp: dict) -> str:
        self._tick()
        if self.phase != 1:
            self.disabled_calls += 1
            return DISABLED_MSG
        path = self._clean_path(inp.get("path"))
        if path not in self.world.files:
            return f"Error: no such file: {path!r}. Use list_files to see valid paths."
        text = self.world.files[path]
        self._tokens_read += estimate_tokens_text(text)
        self.read_events.append({"seq": self._seq, "step": self._polls, "path": path,
                                 "tokens_read_total": self._tokens_read})
        return f"=== {path} ===\n{text}"

    def _search(self, inp: dict) -> str:
        self._tick()
        if self.phase != 1:
            self.disabled_calls += 1
            return DISABLED_MSG
        q = str(inp.get("query") or "")
        if not q.strip():
            return "Error: empty query."
        ql = q.lower()
        hits = []
        for path in sorted(self.world.files):
            for n, line in enumerate(self.world.files[path].splitlines(), 1):
                if ql in line.lower():
                    hits.append(f"{path}:{n}: {line}")
        shown = hits[:SEARCH_MAX_HITS]
        out = "\n".join(shown) if shown else f"No matches for {q!r}."
        if len(hits) > SEARCH_MAX_HITS:
            out += f"\n... ({len(hits) - SEARCH_MAX_HITS} more hits not shown; refine the query)"
        self._tokens_read += estimate_tokens_text(out)
        self.search_events.append({"seq": self._seq, "step": self._polls, "query": q,
                                   "hits": len(hits), "paths": sorted({h.split(":", 1)[0] for h in shown})})
        return out

    def _unread(self) -> list[str]:
        read = {e["path"] for e in self.read_events}
        return [p for p in self.world.reading_order if p not in read]

    def _finish(self, inp: dict) -> str:
        self._tick()
        if self.phase != 1:
            return "The investigation phase is already over. Answer the questions with submit_answer."
        unread = self._unread()
        if unread and self.finish_refusals < self.max_finish_refusals:
            self.finish_refusals += 1
            return (f"Not finished: {len(unread)} checklist file(s) have not been read yet: "
                    + ", ".join(unread) + ". Read them, then call finish_investigation again.")
        self.notes = str(inp.get("notes") or "")
        self.begin_phase2()
        n = len(self.world.reading_order)
        return (f"Investigation closed ({n - len(unread)}/{n} checklist files read). The questions "
                "will arrive in the next message. File tools are now disabled.")

    def _submit(self, inp: dict) -> str:
        self._tick()
        if self.phase == 1:
            return "Error: the questions have not been revealed yet. Finish the investigation first (finish_investigation)."
        if self._answers is not None:
            return "Answers were already recorded."
        raw = inp.get("answers")
        if isinstance(raw, str):
            try:
                raw = json.loads(raw)
            except json.JSONDecodeError:
                raw = None
        if not isinstance(raw, dict):
            return 'Error: answers must be an object mapping question id to answer, e.g. {"q1": "..."}.'
        self._answers = {str(k): "" if v is None else str(v) for k, v in raw.items()}
        return f"Answers recorded ({len(self._answers)} of {len(self.task.questions)})."

    # -- stats ---------------------------------------------------------------
    def stats(self) -> dict:
        last_read: dict[str, dict] = {}
        first_read: dict[str, int] = {}
        for e in self.read_events:
            last_read[e["path"]] = e
            first_read.setdefault(e["path"], e["seq"])
        staleness: dict[str, int | None] = {}
        for q in self.task.questions:
            evs = [last_read.get(p) for p in q.evidence]
            if any(e is None for e in evs):
                staleness[q.qid] = None
            else:
                oldest = min(evs, key=lambda e: e["seq"])
                staleness[q.qid] = self._tokens_read - oldest["tokens_read_total"]
        n = len(self.world.reading_order)
        read = sorted(first_read)
        return {
            "phase": self.phase,
            "phase1_steps": self.phase1_steps,
            "phase2_steps": self.phase2_steps,
            "tool_calls_phase1": self.tool_calls[1],
            "tool_calls_phase2": self.tool_calls[2],
            "disabled_tool_calls": self.disabled_calls,
            "finish_refusals": self.finish_refusals,
            "files_total": n,
            "files_read": read,
            "files_read_count": len(read),
            "coverage": len(read) / n if n else 0.0,
            "file_reads_total": len(self.read_events),
            "actual_read_order": sorted(first_read, key=first_read.get),
            "read_events": [dict(e) for e in self.read_events],
            "search_events": [dict(e) for e in self.search_events],
            "tokens_read_phase1": self._tokens_read,
            # tokens of file content read after the oldest needed evidence was last read
            "question_staleness": staleness,
            "questions_delivered": self._questions_delivered,
            "notes_chars": len(self.notes or ""),
        }


# --------------------------------------------------------------------------
# Grading
# --------------------------------------------------------------------------

_STRIP = " \t\r\n\"'`.,;:!?()[]{}<>*_"
_TS_RE = re.compile(
    r"^(\d{4}-\d{2}-\d{2})[t ](\d{2}:\d{2}:\d{2})(?:\.(\d+))?\s*(?:z|utc|\+00:?00|gmt)?$")
_NUM_RE = re.compile(r"-?\d+(?:\.\d+)?")


def normalize_answer(s: object) -> str:
    """Lowercase, collapse whitespace, strip surrounding quotes/punctuation,
    canonicalize ISO timestamps (``T``/space, trailing ``Z``/``UTC``)."""
    t = " ".join(str(s if s is not None else "").split()).lower()
    prev = None
    while prev != t:
        prev = t
        t = t.strip(_STRIP)
        t = t.removeprefix("@")
    m = _TS_RE.match(t)
    if m:
        frac = f".{m.group(3)}" if m.group(3) else ""
        t = f"{m.group(1)}t{m.group(2)}{frac}z"
    return t


def _parse_number(s: object) -> float | None:
    t = str(s if s is not None else "").replace(",", "")
    m = _NUM_RE.search(t)
    return float(m.group()) if m else None


def _check(q: Question, given: object) -> bool:
    if given is None:
        return False
    if q.kind == "numeric":
        g = _parse_number(given)
        e = float(q.answer)
        return g is not None and abs(g - e) <= (q.tolerance or 0) + 1e-9
    return normalize_answer(given) == normalize_answer(q.answer)


def grade(task: Task, answers: dict | None) -> dict:
    """Per-question correctness, overall score and per-tag breakdown."""
    answers = answers or {}
    per = []
    by_tag: dict[str, list[int]] = {}
    by_tpl: dict[str, list[int]] = {}
    for q in task.questions:
        given = answers.get(q.qid)
        ok = _check(q, given)
        per.append({"qid": q.qid, "expected": q.answer, "given": given, "correct": ok,
                    "tags": q.tags, "template": q.template})
        for tag in q.tags:
            by_tag.setdefault(tag, [0, 0])
            by_tag[tag][0] += ok
            by_tag[tag][1] += 1
        by_tpl.setdefault(q.template, [0, 0])
        by_tpl[q.template][0] += ok
        by_tpl[q.template][1] += 1
    correct = sum(p["correct"] for p in per)
    total = len(per)
    return {
        "task_id": task.id,
        "score": correct / total if total else 0.0,
        "correct": correct,
        "total": total,
        "per_question": per,
        "by_tag": {k: {"correct": v[0], "total": v[1]} for k, v in sorted(by_tag.items())},
        "by_template": {k: {"correct": v[0], "total": v[1]} for k, v in sorted(by_tpl.items())},
    }


# --------------------------------------------------------------------------
# Mock agent
# --------------------------------------------------------------------------


class MockAgentPolicy:
    """Scripted stand-in for the LLM agent, for ``--mock`` / plumbing tests.

    Interface::

        policy = MockAgentPolicy(task)               # perfect agent
        action = policy.next_action(observation)     # -> (tool_name, tool_input) | None

    Call ``next_action`` once per agent step. ``observation`` is optional text
    describing what the agent last saw (the previous tool result or an
    injected user message); the policy is a fixed script and only uses it to
    recover from a ``finish_investigation`` refusal. The script is:

    1. ``("list_files", {})``
    2. ``("read_file", {"path": p})`` for every checklist file, in order
    3. ``("finish_investigation", {"notes": ...})``
    4. ``("submit_answer", {"answers": {qid: answer}})`` - answers come from
       the task's ground truth, so a perfect run scores 1.0
    5. ``None`` afterwards (nothing left to do)

    ``wrong_qids`` / ``error_rate`` (seeded by the task seed) make chosen answers
    wrong, e.g. to exercise reporting; ``forget_early=True`` gets every
    ``early`` question wrong (a crude "forgetful" baseline).
    ``read_limit`` stops reading after that many files.

    The agent loop drives it by turning each action into an assistant
    ``tool_use`` block (e.g. via ``ScriptedLLM`` with a callable). See also
    :func:`run_mock_episode`, which drives an environment directly.
    """

    def __init__(self, task: Task, *, wrong_qids: list[str] | None = None,
                 error_rate: float = 0.0, forget_early: bool = False,
                 read_limit: int | None = None):
        self.task = task
        rng = random.Random(f"mock-{task.seed}")
        wrong = set(wrong_qids or [])
        for q in task.questions:
            if (error_rate and rng.random() < error_rate) or (forget_early and q.position == "early"):
                wrong.add(q.qid)
        self.wrong_qids = sorted(wrong)
        files = task.checklist if read_limit is None else task.checklist[:read_limit]
        self._script: list[tuple[str, dict]] = [("list_files", {})]
        self._script += [("read_file", {"path": p}) for p in files]
        self._script.append(("finish_investigation", {"notes": f"Read {len(files)} files."}))
        self._i = 0
        self._submitted = False

    def answers(self) -> dict[str, str]:
        return {q.qid: ("unknown" if q.qid in self.wrong_qids else q.answer) for q in self.task.questions}

    def next_action(self, observation: str | None = None) -> tuple[str, dict] | None:
        if observation and observation.startswith("Not finished:"):
            # Re-read whatever the environment says is missing, then finish again.
            missing = observation.split("read yet:", 1)[1].split(". Read them", 1)[0]
            extra = [("read_file", {"path": p.strip()}) for p in missing.split(",") if p.strip()]
            self._script[self._i:self._i] = extra + [("finish_investigation", {"notes": "retry"})]
        if self._i < len(self._script):
            act = self._script[self._i]
            self._i += 1
            return act
        if not self._submitted:
            self._submitted = True
            return ("submit_answer", {"answers": self.answers()})
        return None


def run_mock_episode(env: InvestigationEnvironment, policy: MockAgentPolicy, max_steps: int = 500) -> dict:
    """Drive ``env`` with ``policy`` exactly as the agent loop would (one tool
    call per step, ``pending_user_message`` polled after each step), with no
    LLM and no context management. Returns ``{"answer", "steps", "grade", "stats"}``."""
    handlers = {t.name: t.handler for t in env.tools()}
    obs: str | None = env.initial_user_message()
    steps = 0
    while not env.done() and steps < max_steps:
        act = policy.next_action(obs)
        steps += 1
        if act is None:
            break
        name, inp = act
        obs = handlers[name](inp)
        msg = env.pending_user_message()
        if msg is not None:
            obs = msg
    return {"answer": env.answer(), "steps": steps, "grade": grade(env.task, env.answer()),
            "stats": env.stats()}


if __name__ == "__main__":  # pragma: no cover - quick look at a task
    t = make_task(0)
    print(t.brief)
    print()
    for q in t.questions:
        print(q.qid, q.tags, q.template, "->", q.answer)
        print("   ", q.text)
