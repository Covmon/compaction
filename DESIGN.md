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
