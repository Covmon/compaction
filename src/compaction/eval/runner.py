"""Eval runner CLI: tasks x strategies x replicates -> ``<out>/runs/<name>/results.jsonl``.

    uv run python -m compaction.eval.runner --strategies truncate,summary,indexed \
        --tasks 10 --seeds 1 --out results/ --run-name sweep1

``--mock`` runs the whole pipeline offline with a scripted agent (plumbing test only).
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import sys
import threading
import time
import traceback
from collections.abc import Iterator
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

from compaction.agent import Agent, AgentConfig
from compaction.eval.tasks import InvestigationEnvironment, Task, grade, make_tasks
from compaction.strategies import make_strategy
from compaction.tokens import estimate_tokens

PHASE2_MARGIN = 15  # force the question phase this many steps before max_steps


# ---------------------------------------------------------------------------
# Config / jobs
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class RunConfig:
    strategies: list[str]
    tasks: int = 10
    seeds: int = 1
    base_seed: int = 0
    model: str = "claude-haiku-5-5"
    compactor_model: str = "claude-haiku-5-5"
    context_cap: int = 16_000
    threshold: float = 0.8
    keep_last: int = 4
    max_steps: int = 150
    workers: int = 4
    budget_usd: float = 12.0
    mock: bool = False
    run_name: str = "run"


@dataclasses.dataclass(frozen=True)
class Job:
    task_id: str
    seed: int
    strategy: str
    rep: int

    @property
    def key(self) -> tuple[str, str, int]:
        return (self.task_id, self.strategy, self.rep)

    @property
    def dirname(self) -> str:
        return f"{self.task_id}__{self.strategy}__r{self.rep}"


def make_jobs(cfg: RunConfig, tasks: list[Task]) -> list[Job]:
    """Interleaved order (replicate > task > strategy) so partial sweeps stay balanced."""
    return [
        Job(t.id, t.seed, s, r)
        for r in range(cfg.seeds)
        for t in tasks
        for s in cfg.strategies
    ]


class PhaseGuardEnv:
    """Wraps an InvestigationEnvironment; forces phase 2 when steps run low so every
    run gets to answer. Works because the agent polls ``pending_user_message`` once
    per step."""

    def __init__(self, env: InvestigationEnvironment, force_at_step: int) -> None:
        self._env = env
        self._force_at = force_at_step
        self._polls = 0
        self.forced = False

    def __getattr__(self, name):
        return getattr(self._env, name)

    def pending_user_message(self) -> str | None:
        self._polls += 1
        if self._env.phase == 1 and self._polls >= self._force_at:
            self.forced = True
            self._env.begin_phase2()
        return self._env.pending_user_message()


# ---------------------------------------------------------------------------
# Building blocks (module-level so tests can patch them)
# ---------------------------------------------------------------------------


def build_llm(cfg: RunConfig, task: Task, strategy: str):
    if cfg.mock:
        from compaction.eval.mock_llm import MockLLM

        return MockLLM(task, strategy)
    from compaction.llm import AnthropicLLM

    return AnthropicLLM()


def build_strategy(cfg: RunConfig, name: str):
    kw: dict = {}
    if name == "summary":
        kw["model"] = cfg.compactor_model
    elif name == "indexed":
        kw["indexer_model"] = cfg.compactor_model
    return make_strategy(name, **kw)


def _transcript_stats(transcript) -> dict:
    fetch_calls = 0
    retrieved_tokens = 0
    retrieved_text: list[str] = []
    if transcript is not None:
        for m in transcript.all():
            if m.role == "assistant":
                fetch_calls += sum(
                    1
                    for b in m.content
                    if b.get("type") == "tool_use" and b.get("name") == "fetch_messages"
                )
            elif m.retrieved:
                retrieved_tokens += estimate_tokens(m.content)
                retrieved_text.append(json.dumps(m.content, ensure_ascii=False))
    return {
        "fetch_calls": fetch_calls,
        "retrieved_tokens": retrieved_tokens,
        "retrieved_text": "\n".join(retrieved_text),
    }


def run_one(job: Job, task: Task, cfg: RunConfig, run_dir: Path) -> dict:
    """Run one episode. Never raises: failures are captured in the row's `error`."""
    t0 = time.time()
    row: dict = {
        "task_id": job.task_id,
        "task_seed": job.seed,
        "strategy": job.strategy,
        "rep": job.rep,
        "mock": cfg.mock,
        "run_name": cfg.run_name,
        "config": {
            k: getattr(cfg, k)
            for k in ("model", "compactor_model", "context_cap", "threshold",
                      "keep_last", "max_steps")
        },
        "score": 0.0,
        "error": None,
    }
    try:
        env = InvestigationEnvironment(task)
        guard = PhaseGuardEnv(env, max(1, cfg.max_steps - PHASE2_MARGIN))
        llm = build_llm(cfg, task, job.strategy)
        strategy = build_strategy(cfg, job.strategy)
        agent = Agent(
            llm,
            guard,
            strategy,
            AgentConfig(
                model=cfg.model,
                context_cap=cfg.context_cap,
                compact_threshold=cfg.threshold,
                keep_last=cfg.keep_last,
                max_steps=cfg.max_steps,
            ),
            transcript_dir=run_dir / job.dirname,
        )
        try:
            res = agent.run()
        finally:
            usage = llm.usage.snapshot()
        g = grade(task, res.answer)
        ts = _transcript_stats(res.transcript)
        retrieved_text = ts.pop("retrieved_text")
        stats = env.stats()
        for q in g["per_question"]:
            ev = next(x for x in task.questions if x.qid == q["qid"]).evidence
            q["evidence"] = list(ev)
            q["evidence_fetched"] = bool(retrieved_text) and all(p in retrieved_text for p in ev)
            q["staleness"] = stats["question_staleness"].get(q["qid"])
        row.update(
            score=g["score"],
            correct=g["correct"],
            total=g["total"],
            per_question=g["per_question"],
            by_tag=g["by_tag"],
            env_stats={k: v for k, v in stats.items() if k not in ("read_events", "search_events")},
            steps=res.steps,
            compactions=[dataclasses.asdict(c) for c in res.compactions],
            usage=usage,
            cost_usd=usage["total"]["cost_usd"],
            hit_cap_violation=res.hit_cap_violation,
            error=res.error,
            answered=res.answer is not None,
            phase2_forced=guard.forced,
            transcript_path=res.transcript_path,
            **ts,
        )
        idx = getattr(strategy, "index", None)
        if idx is not None:
            row["index_stats"] = {
                "sections": len(getattr(idx, "sections", [])),
                "fetch_calls": getattr(strategy, "fetch_calls", None),
                "fetched_tokens": getattr(strategy, "fetched_tokens", None),
            }
    except Exception as e:  # noqa: BLE001
        row["error"] = f"{type(e).__name__}: {e}"
        row["traceback"] = traceback.format_exc()[-2000:]
        row.setdefault("cost_usd", 0.0)
    row["wall_time_s"] = round(time.time() - t0, 3)
    return row


# ---------------------------------------------------------------------------
# Sweep
# ---------------------------------------------------------------------------


def load_rows(path: Path) -> list[dict]:
    rows = []
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    pass  # partial line from an interrupted write
    return rows


def run_sweep(cfg: RunConfig, out: Path, *, log=print) -> Path:
    run_dir = out / "runs" / cfg.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    results = run_dir / "results.jsonl"
    (run_dir / "config.json").write_text(
        json.dumps({**dataclasses.asdict(cfg), "started": datetime.now().isoformat()}, indent=2)
    )
    tasks = make_tasks(cfg.tasks, cfg.base_seed)
    by_id = {t.id: t for t in tasks}
    existing = load_rows(results)
    done = {(r["task_id"], r["strategy"], r["rep"]) for r in existing}
    total_cost = sum(r.get("cost_usd") or 0.0 for r in existing)
    pending: Iterator[Job] = iter(j for j in make_jobs(cfg, tasks) if j.key not in done)
    n_total = cfg.tasks * cfg.seeds * len(cfg.strategies)
    n_done = len(done)
    log(f"{'MOCK ' if cfg.mock else ''}run {cfg.run_name!r}: {n_total - n_done} to do, "
        f"{n_done} already done, spent so far ${total_cost:.4f} / budget ${cfg.budget_usd:.2f}")
    lock = threading.Lock()
    stopped_for_budget = False

    def record(fut: Future, job: Job) -> None:
        nonlocal total_cost, n_done
        row = fut.result()
        with lock:
            with results.open("a", encoding="utf-8") as f:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
            total_cost += row.get("cost_usd") or 0.0
            n_done += 1
            log(f"[{n_done}/{n_total}] {job.dirname} score={row['score']:.2f} "
                f"steps={row.get('steps', '-')} compactions={len(row.get('compactions', []))} "
                f"cost=${row.get('cost_usd') or 0:.4f} total=${total_cost:.4f}"
                + (f" ERROR={row['error']}" if row.get("error") else ""))

    inflight: dict[Future, Job] = {}
    with ThreadPoolExecutor(max_workers=max(1, cfg.workers)) as ex:
        for job in pending:
            while len(inflight) >= max(1, cfg.workers):
                fin, _ = wait(inflight, return_when=FIRST_COMPLETED)
                for fu in fin:
                    record(fu, inflight.pop(fu))
            if total_cost >= cfg.budget_usd:
                stopped_for_budget = True
                break
            inflight[ex.submit(run_one, job, by_id[job.task_id], cfg, run_dir)] = job
        for fu in list(inflight):
            record(fu, inflight.pop(fu))
    if stopped_for_budget:
        log(f"Budget ${cfg.budget_usd:.2f} reached (spent ${total_cost:.4f}); stopped launching runs. "
            "Re-run the same command with a higher --budget-usd to resume.")
    print_summary(load_rows(results), log=log)
    return results


def print_summary(rows: list[dict], log=print) -> None:
    by: dict[str, list[dict]] = {}
    for r in rows:
        by.setdefault(r["strategy"], []).append(r)
    log(f"\n{'strategy':<10} {'runs':>5} {'acc':>6} {'err':>4} {'compact':>8} {'$/run':>8}")
    for s, rs in by.items():
        n = len(rs)
        acc = sum(r["score"] for r in rs) / n
        errs = sum(1 for r in rs if r.get("error"))
        comp = sum(len(r.get("compactions", [])) for r in rs) / n
        cost = sum(r.get("cost_usd") or 0 for r in rs) / n
        log(f"{s:<10} {n:>5} {acc:>6.3f} {errs:>4} {comp:>8.1f} {cost:>8.4f}")
    if rows and any(r.get("mock") for r in rows):
        log("(MOCK - plumbing test, not a result)")


def parse_args(argv: list[str] | None = None) -> tuple[RunConfig, Path]:
    p = argparse.ArgumentParser(prog="compaction-eval", description=__doc__)
    p.add_argument("--strategies", default="truncate,summary,indexed")
    p.add_argument("--tasks", type=int, default=10)
    p.add_argument("--seeds", type=int, default=1, help="replicates per (task, strategy)")
    p.add_argument("--base-seed", type=int, default=0)
    p.add_argument("--model", default="claude-haiku-5-5")
    p.add_argument("--compactor-model", default="claude-haiku-5-5")
    p.add_argument("--context-cap", type=int, default=16_000)
    p.add_argument("--threshold", type=float, default=0.8)
    p.add_argument("--keep-last", type=int, default=4)
    p.add_argument("--max-steps", type=int, default=150)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--out", default="results/")
    p.add_argument("--mock", action="store_true")
    p.add_argument("--run-name", default=None)
    p.add_argument("--budget-usd", type=float, default=12.0)
    a = p.parse_args(argv)
    name = a.run_name or ("mock" if a.mock else datetime.now().strftime("run-%Y%m%d-%H%M%S"))
    cfg = RunConfig(
        strategies=[s.strip() for s in a.strategies.split(",") if s.strip()],
        tasks=a.tasks, seeds=a.seeds, base_seed=a.base_seed, model=a.model,
        compactor_model=a.compactor_model, context_cap=a.context_cap,
        threshold=a.threshold, keep_last=a.keep_last, max_steps=a.max_steps,
        workers=a.workers, budget_usd=a.budget_usd, mock=a.mock, run_name=name,
    )
    return cfg, Path(a.out)


def main(argv: list[str] | None = None) -> int:
    # Load ANTHROPIC_API_KEY (and other settings) from a git-ignored .env file
    # in the current directory, without overriding variables already exported.
    load_dotenv(override=False)
    # An empty key (e.g. an unedited .env copied from .env.example) would shadow
    # other credentials such as an `ant auth login` profile, so drop it.
    for var in ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN"):
        if os.environ.get(var, None) == "":
            del os.environ[var]
    cfg, out = parse_args(argv)
    run_sweep(cfg, out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
