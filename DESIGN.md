# Design: Indexed Retention vs. Summary Compaction

This is the single source of truth for module boundaries and interfaces.
Every module must conform to the signatures here. If you need to change
an interface, note it in the "Deviations" section at the bottom.

## Goal

Compare three context-management strategies for an agent with an
**artificially small context window** (default cap: 16,000 tokens):

| Strategy | Name | Behaviour at threshold |
|---|---|---|
| `truncate` | Baseline | Drop the oldest messages; keep the last K |
| `summary` | Traditional compaction | A small model writes a prose summary; new context = summary + last K messages |
| `indexed` | **Indexed Retention** | A small model updates a hierarchical index of the transcript; new context = index notice + last K messages; agent gets `read_index` / `fetch_messages` tools |

Models: agent = `claude-haiku-5-5`, compactor/indexer = `claude-haiku-5-5`
(both configurable).

## Package layout (`src/compaction/`)

```
llm.py              LLMClient protocol, AnthropicLLM, ScriptedLLM (test double), usage accounting
tokens.py           estimate_tokens(...) heuristic counter
transcript.py       Message, Transcript (append-only, permanent, JSONL-persisted)
context.py          LiveContext: ordered list of message ids + helpers to build API `messages`
tools.py            ToolSpec (already written; import from compaction.tools)
agent.py            Agent loop, Environment protocol, RunResult (re-exports ToolSpec)
strategies/
  base.py           CompactionStrategy ABC, CompactionEvent
  truncate.py       TruncateStrategy
  summary.py        SummaryStrategy
  indexed.py        IndexedRetentionStrategy (indexer call + tools)
  index.py          Index data structure (parse/render/apply edits/resolve ids)
  __init__.py       make_strategy(name, **kw)
eval/
  world.py          Synthetic world generator (seeded, deterministic)
  tasks.py          Task definitions, InvestigationEnvironment (tools), grading
  runner.py         CLI: run tasks x strategies x seeds -> results/*.jsonl
  report.py         CLI: aggregate results -> results/report.md + charts
```

## Core types

```python
# transcript.py
@dataclass
class Message:
    id: str                 # "m0001", "m0002", ... monotonically increasing, never reused
    role: Literal["user", "assistant"]
    content: list[dict]     # Anthropic content blocks as plain dicts (text / tool_use / tool_result)
    session: int            # which context-session (0, 1, 2 ...) produced it
    retrieved: bool = False # True if this message carries the RESULT of a fetch_messages call
    synthetic: bool = False # True for harness-injected messages (summary, index notice, resume note)

class Transcript:
    def append(self, role, content, *, session, retrieved=False, synthetic=False) -> Message
    def get(self, msg_id) -> Message
    def range(self, start_id, end_id) -> list[Message]   # inclusive
    def all(self) -> list[Message]
    def save(self, path) / @classmethod load(path)
    def render(self, msgs, *, omit_retrieved=False) -> str
        # Human-readable flattening: "[m0007 assistant] text... / tool_use name(args) / [tool_result] ..."
        # omit_retrieved=True replaces retrieved tool_result bodies with
        # "[retrieved copy of earlier messages omitted]"
```

The transcript is **permanent**: compaction never deletes from it, it only
changes what's in the live context.

```python
# context.py
class LiveContext:
    ids: list[str]                      # message ids currently in context, in order
    def to_api_messages(self, transcript) -> list[dict]   # valid Anthropic `messages` array
    def tokens(self, transcript, system: str, tools: list[dict]) -> int
```

`to_api_messages` must merge consecutive same-role messages and guarantee:
the first message is `user`, and every `tool_result` has its matching
`tool_use` in the preceding assistant message (orphans are converted to text).

```python
# tokens.py
def estimate_tokens(obj) -> int   # ~len(json/text)/3.5, rounded up. Deterministic, no API call.
```

Context-cap enforcement uses `estimate_tokens` before each call, and
calibrates against `usage.input_tokens` from the previous response:
`effective = max(estimate, last_reported_input + estimate(new_since_last))`.

```python
# llm.py
@dataclass
class LLMResponse:
    content: list[dict]          # content blocks as dicts (text / tool_use); thinking blocks dropped
    stop_reason: str
    input_tokens: int
    output_tokens: int
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0

class LLMClient(Protocol):
    def create(self, *, model: str, system: str, messages: list[dict],
               tools: list[dict] | None = None, max_tokens: int = 4096,
               purpose: str = "agent") -> LLMResponse
    usage: "UsageLedger"        # accumulates tokens + $ per purpose ("agent", "compactor")

class AnthropicLLM(LLMClient): ...   # anthropic SDK, retries, prompt caching on system+tools
class ScriptedLLM(LLMClient): ...    # replays a list of LLMResponse or a callable(messages)->LLMResponse
```

Pricing for cost accounting lives in `llm.py` (`PRICES = {"claude-haiku-5-5": (0.10, 0.50), ...}` per MTok).

## Agent loop (`agent.py`)

```python
@dataclass
class ToolSpec:
    name: str
    description: str
    input_schema: dict
    handler: Callable[[dict], str]

class Environment(Protocol):
    def system_prompt(self) -> str
    def initial_user_message(self) -> str
    def tools(self) -> list[ToolSpec]       # includes a terminal `submit_answer` tool
    def done(self) -> bool                  # True after submit_answer was called
    def answer(self) -> dict | None

@dataclass
class AgentConfig:
    model: str = "claude-haiku-5-5"
    context_cap: int = 16_000
    compact_threshold: float = 0.80        # compact when effective tokens > cap * threshold
    keep_last: int = 4                     # messages carried over into the new session
    max_steps: int = 120                   # model calls
    max_tokens: int = 2048                 # per agent response

@dataclass
class RunResult:
    answer: dict | None
    steps: int
    compactions: list[CompactionEvent]
    usage: dict                            # ledger snapshot
    transcript_path: str | None
    hit_cap_violation: bool                # True if a request would exceed the cap even after compaction
    error: str | None

class Agent:
    def __init__(self, llm, env, strategy, config, transcript_dir=None): ...
    def run(self) -> RunResult
```

Loop: build request (system = env.system_prompt() + strategy.system_addendum();
tools = env tools + strategy.tools()) -> if effective tokens > threshold:
`strategy.compact(...)` -> call model -> append assistant message -> execute
tool_uses (env or strategy handlers), append one user message with all
tool_results (marking `retrieved=True` if any was a `fetch_messages` result)
-> repeat until env.done() or max_steps. If the model ends its turn without
a tool call and isn't done, append a user nudge: "Continue. Call submit_answer when finished."

Hard-cap rule: if after compaction the request still exceeds `context_cap`,
the oldest kept messages are dropped one at a time; if still impossible,
record `hit_cap_violation` and truncate tool_result text. A request sent to
the API must never exceed `context_cap` (estimated).

## Strategies (`strategies/base.py`)

```python
@dataclass
class CompactionEvent:
    step: int
    tokens_before: int
    tokens_after: int
    compactor_input_tokens: int
    compactor_output_tokens: int
    detail: dict

class CompactionStrategy(ABC):
    name: str
    def bind(self, llm: LLMClient, transcript: Transcript, config: AgentConfig) -> None
    def tools(self) -> list[ToolSpec]                 # [] for truncate/summary
    def system_addendum(self) -> str                  # "" by default
    @abstractmethod
    def compact(self, ctx: LiveContext, session: int, step: int) -> tuple[LiveContext, CompactionEvent]
        # Returns the NEW live context for session+1. May append synthetic messages to the transcript.
```

Shared helper `split_tail(ctx, transcript, keep_last)` returns `(head_ids, tail_ids)`
with the tail adjusted so it begins at a clean boundary (never starts with
an orphan tool_result; prefer starting at a user message).

### Summary (`summary.py`)

Compactor gets `previous summary (if any) + rendered head messages`
(retrieved bodies are not relevant here; there are none) and is asked for a
detailed, structured summary preserving exact identifiers, numbers, file
names, and open questions, within `summary_max_tokens` (default 2000).
New context: synthetic user message "[Context was compacted. Summary of
earlier work:]\n<summary>" + tail.

### Indexed Retention (`index.py`, `indexed.py`)

**Index format** (markdown, rendered from a structured model):

```
# Index
## 1. Explored service inventory  [m0002–m0019]
### 1.1 Listed all 24 files under /logs; noted 3 regions  [m0002–m0004]
### 1.2 Read auth-service logs: token expiry errors start 03:14Z  [m0005–m0011]
## 2. ...
```

- Sections `N` and subsections `N.M` have stable ids. Each subsection
  covers one or more **message ranges** (`[(start,end), ...]`); a section's
  coverage is the union of its subsections'.
- `Index.apply(ops)` supports:
  - `{"op": "add_section", "title": str}` -> new id = next integer
  - `{"op": "add_subsection", "section": "N", "summary": str, "ranges": [["m0005","m0011"]]}`
  - `{"op": "edit", "id": "N" | "N.M", "title"/"summary": str}` (continuity edits; cannot change ranges of existing subsections)
  - `{"op": "extend", "id": "N.M", "ranges": [...]}` (a later topic continues an earlier subsection)
- `Index.resolve(ids: list[str]) -> list[str]` -> sorted, de-duplicated message ids.
- `Index.render() -> str`, `Index.to_json()/from_json()`.

**Indexer call** (`IndexedRetentionStrategy.compact`):
1. Unindexed = non-synthetic messages in the head that aren't yet covered.
2. Render them with `omit_retrieved=True`: results of earlier
   `fetch_messages` calls are **never** indexed (they're already covered by
   the original ranges). This is enforced mechanically; the prompt also says so.
3. Indexer model gets: current index (rendered with ids) + rendered unindexed
   messages, and returns JSON `{"ops": [...]}` (use structured outputs with a
   JSON schema; fall back to extracting JSON from text).
4. Harness validates ops; any unindexed message left uncovered is auto-added
   to a subsection `"(auto) unindexed messages"` so coverage is total.
5. New context: synthetic user message containing the resume notice +
   the **full rendered index** (so the agent doesn't need a call to see it
   right after compaction) + tail.

**Agent tools**:
- `read_index()` -> rendered index.
- `fetch_messages(ids: list[str])` -> rendered original messages
  (`Transcript.render`, retrieved bodies omitted) for the given section /
  subsection ids. Guarded by a budget: if the result would exceed
  `fetch_budget` tokens (default 35% of cap) it is truncated with a note
  telling the agent to fetch narrower subsections.

## Eval (`eval/`)

**Synthetic investigation tasks.** A seeded generator builds a fictional
system (e.g. a company's microservices): ~20–40 "files" (logs, configs,
runbooks, tickets, a DB table dump), each ~400–1200 tokens. Facts are
scattered so that answering requires reading many files (total corpus
reading >> 16k tokens => several compactions), and the final questions ask
for **exact** values seen early (ids, timestamps, port numbers, version
strings, hashes, names) plus multi-hop joins (value from file A identifies
which record in file B matters).

Environment tools: `list_files()`, `read_file(path)`, `search(query)` (grep
over files, capped output), `submit_answer(answers: {qid: str})`.

Prompting forces exploration: the questions are revealed only partially at the
start ("you will be asked about X") OR the agent is told to first produce a
complete audit, then gets the questions. The default **protocol**:
phase 1, the agent is given an investigation brief and must read all relevant
files and call `finish_investigation()`. Phase 2, the questions are
then revealed in a new user message (after more compactions have happened),
and the agent answers via `submit_answer`. This isolates *retention*: the
needed facts are guaranteed to be outside the live context when questions arrive.

Grading: per-question normalized exact match (case/whitespace-insensitive;
numeric tolerance where specified). Score = fraction correct.

`runner.py`:
`uv run python -m compaction.eval.runner --strategies truncate,summary,indexed --tasks 10 --seeds 2 --out results/`
parallelism via a thread pool (`--workers 4`), resumable (skips done rows),
`--dry-run` uses ScriptedLLM-free `--mock` mode that runs end-to-end with a
deterministic fake agent to test plumbing without API calls.

`report.py`: per-strategy mean accuracy ± stderr, accuracy by question type
(early vs late, single-hop vs multi-hop), tokens/cost per task, compactions,
fetch calls, retrieved tokens; charts as PNG in `results/`.

## Deviations
(append here)

### Harness deviations (core agent harness)
- `Environment.pending_user_message() -> str | None` is an OPTIONAL method (agent uses `getattr`).
  Checked after each step once tool_results are appended (and after `done()`); returned text becomes a
  separate user message that `LiveContext.to_api_messages` merges into the tool_result user message
  (tool_results first). It replaces the "Continue" nudge for that step.
- `LLMResponse.content` KEEPS `thinking`/`redacted_thinking` blocks (stored in the transcript). They are
  resent only for the latest assistant tool_use turn of an unbroken tool loop; the sole decision point is
  `context.strip_thinking`, applied inside `to_api_messages` (so token estimates match what is sent).
  `AnthropicLLM` retries once with all thinking stripped on a 400 that mentions "thinking".
- `LLMResponse.input_tokens` is the uncached count as reported by the API; use `.total_input_tokens`
  (input + cache read + cache write) for prompt size. `UsageLedger.snapshot()` returns
  `{"by_purpose": {purpose: {calls,input_tokens,output_tokens,cache_read_tokens,cache_write_tokens,cost_usd}}, "total": {...}}`.
- `ScriptedLLM` callable signature is `(messages, tools, purpose)`; it records requests in `.requests` and,
  if a response reports 0 tokens, fills usage from the estimator. Helpers `text_response` / `tool_response`.
- `AnthropicLLM(effort="low")` sends `output_config={"effort": ...}` and top-level `cache_control`; it has no
  structured-output/JSON-schema parameter yet (the indexed strategy may add one to `create`).
- `Transcript.append` also accepts a plain str as content. `LiveContext` gains `copy()`, `append()`,
  `messages()` and a view-only `max_result_chars` (hard-cap fallback; the transcript is never altered).
- `Agent(..., transcript_dir)` writes `<transcript_dir>/transcript.jsonl` (kwarg `transcript_name`); use a
  distinct dir per run. `RunResult` gains `transcript` (the in-memory `Transcript`). `usage` is the whole
  LLM client's ledger, so use one LLM client per run. `Agent.run` never raises: failures go in `error`
  (`max_steps`, `refusal`, `cap_unsatisfiable...`, or the exception text).
- Agent overwrites `CompactionEvent.tokens_before/after` with full-request estimates (system+tools+messages);
  strategies only fill message-only estimates. Compaction is skipped unless >= 2 messages were added since the
  previous one (prevents compaction thrash when the kept tail alone exceeds the threshold).
- Hard cap: effective tokens = max(estimate, last reported prompt + growth). Over cap after compaction ->
  oldest messages dropped one at a time, then tool_result bodies shrunk (view-only), `hit_cap_violation=True`;
  if still impossible the run stops with `cap_unsatisfiable` WITHOUT sending the request.
- `max_tokens` stop: tool_use blocks are discarded (never executed) and a synthetic "be more concise" user
  note is added. `refusal` ends the run with `error="refusal"`.
- `split_tail`: if the cut lands on a user message holding tool_results, the cut moves one message earlier
  (tail starts with the assistant tool_use); a tail may start with an assistant message since strategies
  always prepend a user message.
- `TruncateStrategy` inserts a synthetic "[Context was truncated: N earlier messages dropped]" user note
  (`pin_first=False` option to keep the first task message). `SummaryStrategy` skips synthetic head messages
  (its stored previous summary stands in for them) and uses `purpose="compactor"`.

### Eval tasks (`eval/world.py`, `eval/tasks.py`)
- **Environment protocol addition**: `pending_user_message() -> str | None`, polled by the agent
  after every step. `InvestigationEnvironment` returns the phase-2 questions exactly once, right
  after `finish_investigation` succeeds. It also counts polls to report phase-1/phase-2 steps.
- **Extra tool** `finish_investigation(notes)` ends phase 1. It refuses (listing the unread files)
  while checklist files are unread, at most `max_finish_refusals=2` times, and then accepts anyway.
  After that, `list_files` / `read_file` / `search` return
  `"Investigation phase is over; answer from what you have retained"`. `submit_answer` called before the
  questions are revealed returns an error, and only the first successful `submit_answer` counts.
  `begin_phase2()` lets a harness force the question phase, for example when a step budget runs out.
- World: 33 files (~28-30k tokens at len/3.5, 600-1160 tokens each). The brief gives the topics
  (not the questions) and a checklist in a seeded reading order. There are 10 questions per task:
  about 6 `early` and 4 `late`, plus 4 `multi_hop` and 6 `single_hop`. Questions are also tagged
  `exact` or `numeric`. A question is tagged `early` when the checklist position of its
  earliest-listed evidence file is in the first half. Every `Question` stores its `evidence`
  paths, and `env.stats()["question_staleness"]` gives, per question, the tokens of file content
  the agent read after it last read the oldest needed evidence file. The report can use this to
  tag questions by the order the agent actually read files in.
- Grading normalizes case, whitespace, surrounding quotes and punctuation, and a leading `@`. It
  treats ISO timestamps written with `T` or a space, and ending in `Z` or `UTC`, as equal. For
  numeric answers it compares the first number in the answer, after removing commas and currency
  text, within an absolute tolerance (0 for integers, 0.01 for money).
- `MockAgentPolicy(task).next_action(observation) -> (tool, input) | None` is a scripted perfect
  agent for `--mock`. Its options are `forget_early`, `error_rate`, `wrong_qids` and `read_limit`.
  `run_mock_episode(env, policy)` drives an environment directly, without an LLM.

### Indexed Retention deviations (`strategies/index.py`, `strategies/indexed.py`)
- `LLMClient.create` (and `AnthropicLLM` / `ScriptedLLM`) gain `output_format: dict | None = None` (a JSON
  schema). `AnthropicLLM` sends `output_config={"format": {"type": "json_schema", "schema": ...}}`, merged with
  `effort`; `ScriptedLLM` records it in `requests[i]["output_format"]`. The strategy retries without the kwarg on
  `TypeError`, so older/mocked clients still work.
- Ops schema (`index.OPS_SCHEMA`): `{"ops": [op]}`, each op ONE flat object with all fields required
  (`op` enum, `id`, `section`, `title`, `summary`, `ranges`; unused = null), as structured outputs require
  `additionalProperties: false`. Ranges are strings `"m0005-m0011"` / `"m0007"`; the parser also accepts
  `["m0005","m0011"]` pairs, `{"start","end"}`, en dashes and `..`. Fallback `parse_ops` extracts JSON from
  prose / code fences / bare op lists.
- `add_subsection.section` may be an existing id, the exact title of a section (e.g. one added earlier in the
  same op list), or null = the last section added in this batch. Invalid ops / ranges are skipped and
  reported, never raised; sections left with no subsections are dropped. Range endpoints must lie within the
  span of the messages given to that indexer call (they may cross skipped ids).
- Leftover uncovered messages go into `"(auto) Unindexed messages"` subsections (<= 6 nearby messages each)
  whose summaries list the tool calls they contain (e.g. `read_file({"path": ...})`), so they stay findable.
- "Unindexed" = EVERY transcript message before the new tail that is not covered, not synthetic and not a pure
  fetch copy -- not just the head -- so messages the agent's hard-cap fallback dropped are indexed too.
  Messages that carry only `fetch_messages` results are skipped entirely; in any other message (a hand-built
  mixed one) only fetch bodies are replaced by `RETRIEVED_OMITTED`. `read_index` result bodies are shown to the
  indexer as `[index listing omitted]`. Fetch output applies the same rules.
- Index messages longer than the indexer budget are cut per message; the unindexed messages are split into
  chunks so each indexer request stays under `max_indexer_input_tokens` (60k); chunks run sequentially, each
  seeing the updated index. Indexer calls use `purpose="compactor"` (same ledger bucket as summary).
- The indexer prompt ends with the lines `Messages to index (...)` and `Ids to index: m0002, m0003, ...`
  (scripted/mock indexers may parse the latter).
- `fetch_messages` also accepts raw message ids/ranges (`"m0012"`, `"m0012-m0015"`); a bare string is accepted
  in place of the list; unknown ids are listed with the valid ids. Overlapping requests are de-duplicated and
  each subsection is shown under a `=== N.M summary (section title)  [ranges] ===` header.
- Resume notice also names the first task message id (`m0001`) so the agent can re-fetch its instructions. If
  the rendered index exceeds `max_index_frac` (30%) of the cap, the notice shows it abbreviated (truncated
  summaries, then section-only) and points to `read_index`.
- Persistence hook: `CompactionStrategy.transcript_path` (class attr, default None) is set by `Agent.__init__`
  before `bind()`. The indexed strategy writes `index.md` + `index.json` next to the transcript after every
  compaction (`<stem>.index.md/json` when `transcript_name` is not `transcript.jsonl`).
- `split_tail` now steps back repeatedly while the cut lands on a user tool_result message (a turn's results
  can span two user messages since fetch results are stored separately).
- Event detail keys: `indexed_messages, skipped_retrieved, indexer_calls, sections_added, subsections_added,
  edits, extends, op_errors, op_error_examples, indexer_failures, auto_covered, indexer_input_tokens,
  indexer_output_tokens, index_sections, index_subsections, index_tokens, fetch_calls_so_far, tail_start`.
  The strategy also counts `fetch_calls` / `fetched_tokens` as attributes.

### Eval runner / report / mock (`eval/runner.py`, `eval/report.py`, `eval/mock_llm.py`)
- Layout: `<out>/runs/<run_name>/{config.json,results.jsonl,report.md,*.png,<task>__<strategy>__r<k>/transcript.jsonl}`.
  `--seeds K` means K replicates per (task, strategy); tasks use seeds `base_seed .. base_seed+tasks-1`.
  Resume key is `(task_id, strategy, rep)`. Work order is replicate > task > strategy (interleaved).
- Phase-2 forcing: `PhaseGuardEnv` wraps the environment and calls `begin_phase2()` once
  `max_steps - 15` steps have been polled (rows record `phase2_forced`).
- Budget: stop launching when running total (including cost of resumed rows) >= `--budget-usd`; in-flight runs finish.
- Rows add per-question `evidence`, `staleness`, `evidence_fetched` (all evidence paths appear in retrieved
  messages; best effort), `fetch_calls` / `retrieved_tokens` from the transcript, and `index_stats` from the strategy.
- Strategy kwargs: summary gets `model=compactor_model`, indexed gets `indexer_model=compactor_model`.
- `MockLLM(task, strategy)` subclasses `ScriptedLLM`. Per-strategy forgetfulness (`MOCK_PROFILES`: truncate forgets
  early + 10% errors, summary 30% errors, indexed perfect) is arbitrary: mock results are PLUMBING TESTS, not findings,
  and reports/charts are labelled so. The mock indexer returns valid ops (dropping some ids on even calls to exercise
  auto-coverage); the indexed mock agent calls `fetch_messages(["1"])` once before submitting.
- Report: stderr is over runs (accuracy/cost) or over questions (by-tag / staleness / fetched tables).
  Staleness buckets: 0-8k, 8-16k, 16-24k, 24k+ tokens; unread-evidence questions excluded.
