# compaction

## What this is

An experiment comparing **Indexed Retention**, a context-compaction approach, against
traditional summary compaction and plain truncation. An agent runs with an artificially
small context window (16,000 tokens) and must answer questions about material it has
read earlier and no longer holds in context.

## The idea

Indexed Retention keeps a permanent transcript and a markdown index that points into it.

- The harness stores every message permanently under a stable id (`m0001`, `m0002`, ...).
- The index is a tree of sections and subsections. Each subsection names what happened
  and the message ids where it happened.
- The agent has two tools: `read_index()` returns the index, and `fetch_messages(ids)`
  returns the original messages for the given sections or subsections.
- When the context nears the cap, a small, fast model appends to or edits the index for
  the messages that are leaving context. The context is then cleared, and a new session
  resumes with the index and the last few messages.
- The agent is never forced to compress details into prose. Anything it needs exactly can
  be fetched verbatim.
- Fetched copies are never re-indexed, so the index never contains duplicate content.

Example index (rendered form):

```
# Index
## 1. Task brief and plan  [m0001–m0002]
### 1.1 Task instructions: audit the payments incident across four services  [m0001]
### 1.2 Reading order from the checklist; topics include auth tokens and retry policy  [m0002]
## 2. Auth service investigation  [m0003–m0010]
### 2.1 Read logs/auth-service.log: token expiry errors start 03:14Z  [m0003–m0004]
### 2.2 search('ERR-4471') -> 3 hits in auth logs and 2 tickets  [m0005–m0006]
```

## Strategies compared

| Strategy | Name | Behaviour at threshold | Extra agent tools |
|---|---|---|---|
| `truncate` | Baseline | Drop the oldest messages and keep the last K | none |
| `summary` | Traditional compaction | A small model writes a structured prose summary; new context = summary + last K messages | none |
| `indexed` | Indexed Retention | A small model updates the index; new context = index + last K messages | `read_index`, `fetch_messages` |

## The eval

- **Worlds.** Each task is a synthetic, seeded investigation into a fictional system. A
  task has 33 files (logs, configs, runbooks, tickets, a database dump) totalling roughly
  28–30k tokens, against a 16k context cap. Reading the corpus therefore forces several
  compactions.
- **Two phases.** In phase 1 the agent gets a brief with topics (not questions) and a
  checklist of files in a seeded reading order. It reads the files and calls
  `finish_investigation`. This refuses while checklist files are unread, at most twice,
  then accepts. In phase 2 the questions are revealed and `list_files`, `read_file` and
  `search` are disabled, so the agent must answer from what it retained.
- **Questions.** Ten per task: six tagged `early` and four `late` (by where their earliest
  evidence file falls in the checklist), six `single_hop` and four `multi_hop`. Each is
  also tagged `exact` or `numeric`. Answers ask for exact values seen early: ids,
  timestamps, ports, version strings, hashes, names.
- **Grading.** Per-question normalized exact match. Normalization ignores case,
  whitespace, surrounding quotes and punctuation, and a leading `@`. Numeric answers are
  compared within a tolerance. The score is the fraction of questions correct.

## Quickstart

Requires [uv](https://docs.astral.sh/uv/) and Python 3.12. `make` is optional; each target
is shown as the underlying command.

```sh
uv sync                       # make install
uv run pytest -q              # make test
uv run ruff check src tests   # make lint
```

Offline mock run. It exercises the whole pipeline with a scripted agent and makes no API
calls. Its results are plumbing checks, not findings.

```sh
uv run compaction-eval --mock --tasks 3 --seeds 1 --out results/ --run-name mock
uv run compaction-report --run results/runs/mock     # make mock; make report RUN=mock
```

Real run. Needs `ANTHROPIC_API_KEY` in the environment. This runs the default sweep: 10
tasks, 3 strategies, 2 replicates.

```sh
export ANTHROPIC_API_KEY=...
uv run compaction-eval --strategies truncate,summary,indexed --tasks 10 --seeds 2 \
  --out results/ --run-name v1 --budget-usd 12
uv run compaction-report --run results/runs/v1      # make eval RUN=v1; make report RUN=v1
```

Useful flags for `compaction-eval`: `--workers` (thread pool, default 4), `--max-steps`
(default 150), `--context-cap` (default 16000), `--threshold` (default 0.8),
`--keep-last` (default 4), `--base-seed`, `--model`, `--compactor-model`. The runner is
resumable: re-running the same command skips rows already in `results.jsonl`.

`make eval` and `make mock` write under `results/`, which is not committed.

## Cost

The agent and the compactor (summary and indexer) both default to `claude-haiku-5-5`,
priced at $0.10 per million input tokens and $0.50 per million output tokens. Cache reads
are billed at 0.1× the input price and cache writes at 1.25×.

The default sweep (10 tasks × 3 strategies × 2 replicates) is estimated at about $5–10.
This is an estimate, not a measured figure. Exact costs are recorded per row in
`results.jsonl` and summarized in `report.md`.

`--budget-usd` (default 12.0) is a guard. The runner stops launching new episodes once
the running total, including resumed rows, reaches the budget. Runs already in flight
finish. Re-run the same command with a higher budget to continue.

## Layout

```
src/compaction/
  llm.py              LLM client protocol, Anthropic client, scripted test double, pricing and usage
  tokens.py           Token estimator (heuristic, characters / 3.5)
  transcript.py       Permanent, append-only message store (JSONL)
  context.py          Live context: ordered message ids and API message construction
  tools.py            ToolSpec
  agent.py            Agent loop, environment protocol, run result
  strategies/
    base.py           CompactionStrategy ABC, CompactionEvent, split_tail
    truncate.py       Baseline
    summary.py        Traditional summary compaction
    indexed.py        Indexed Retention (indexer call, read_index, fetch_messages)
    index.py          Index data structure: parse, render, apply ops, resolve ids
    __init__.py       make_strategy(name, **kwargs)
  eval/
    world.py          Seeded synthetic world generator
    tasks.py          Tasks, questions, investigation environment, grading
    mock_llm.py       Scripted mock agent and indexer for --mock runs
    runner.py         compaction-eval CLI: tasks x strategies x replicates
    report.py         compaction-report CLI: aggregate a run into report.md and PNG charts
tests/                pytest suite
DESIGN.md             Interfaces, module boundaries, and the Deviations log
.github/workflows/ci.yml   Lint, test, and mock smoke run on push and pull request
```

A run writes to `results/runs/<run-name>/`: `config.json`, `results.jsonl`, `report.md`,
PNG charts, and one `<task>__<strategy>__r<k>/` directory per episode containing
`transcript.jsonl` and, for `indexed`, `index.md` and `index.json`.

`DESIGN.md` is the source of truth for interfaces. Deviations from it are recorded in its
Deviations section.

## Results

Results will be added after the first real run.

## Limitations and open questions

- **Index growth.** The index grows with the transcript. The resume notice abbreviates it
  once it passes 30% of the cap and points the agent to `read_index`, but it is unclear how
  well this scales to much longer investigations.
- **Task instructions leave context.** After the first compaction the original task
  message is no longer in the live context. The resume notice names its id (`m0001`) so the
  agent can re-fetch it, but agents may not do so.
- **Heuristic token counts.** Token counts use characters / 3.5 and are not tokenizer
  output. The cap is enforced against these estimates, calibrated with the API's reported
  input tokens where available.
- **Synthetic tasks.** The worlds are generated, not real incident data. Their difficulty
  and the question mix are design choices, and the results may not transfer.
- **Mock results are not findings.** The mock profiles that give each strategy its
  forgetfulness are arbitrary. They exist to test plumbing only.
- **Single model.** Agent and compactor both use `claude-haiku-5-5`. Results with a
  larger compactor or a different agent model are not covered.
