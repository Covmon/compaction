"""Aggregate a run directory into ``report.md`` + PNG charts.

    uv run python -m compaction.eval.report --run results/runs/<name>
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Okabe-Ito (colorblind-safe), fixed per strategy across all charts.
STRATEGY_COLORS = {"truncate": "#E69F00", "summary": "#0072B2", "indexed": "#009E73"}
_EXTRA = ["#CC79A7", "#56B4E9", "#D55E00", "#F0E442", "#000000"]
STALENESS_BINS = [(0, 8_000), (8_000, 16_000), (16_000, 24_000), (24_000, float("inf"))]
MOCK_BANNER = "MOCK — plumbing test, not a result"
INK = "#333333"


def load_rows(run_dir: Path) -> list[dict]:
    rows = []
    for line in (run_dir / "results.jsonl").read_text(encoding="utf-8").splitlines():
        if line.strip():
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return rows


def mean(xs) -> float:
    xs = list(xs)
    return sum(xs) / len(xs) if xs else float("nan")


def stderr(xs) -> float:
    xs = list(xs)
    n = len(xs)
    if n < 2:
        return 0.0
    m = mean(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (n - 1) / n)


def fmt(x: float, d: int = 3) -> str:
    return "n/a" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.{d}f}"


class Palette:
    def __init__(self, strategies: list[str]) -> None:
        extra = iter(_EXTRA)
        self.c = {s: STRATEGY_COLORS.get(s) or next(extra) for s in strategies}

    def __getitem__(self, s: str) -> str:
        return self.c[s]


def _style(ax, title: str, ylabel: str, mock: bool) -> None:
    ax.set_title(title + ("\n" + MOCK_BANNER if mock else ""), fontsize=10, color="#B00020" if mock else INK)
    ax.set_ylabel(ylabel, color=INK)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(axis="y", alpha=0.25)
    ax.set_axisbelow(True)


def _save(fig, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def _grouped_bars(ax, groups: list[str], series: dict[str, list[tuple[float, float]]], pal: Palette) -> None:
    n = max(1, len(series))
    w = 0.8 / n
    for i, (s, vals) in enumerate(series.items()):
        xs = [g + (i - (n - 1) / 2) * w for g in range(len(groups))]
        ys = [0 if math.isnan(v[0]) else v[0] for v in vals]
        es = [v[1] for v in vals]
        ax.bar(xs, ys, w * 0.92, yerr=es, capsize=2, color=pal[s], label=s, error_kw={"lw": 1, "ecolor": INK})
    ax.set_xticks(range(len(groups)), groups)
    ax.legend(frameon=False, fontsize=8)


# ---------------------------------------------------------------------------
# Computations
# ---------------------------------------------------------------------------


def staleness_bucket(v) -> str | None:
    if v is None:
        return None
    for lo, hi in STALENESS_BINS:
        if lo <= v < hi:
            return f"{lo // 1000}-{int(hi) // 1000}k" if hi != float("inf") else f"{lo // 1000}k+"
    return None


BUCKETS = [staleness_bucket(lo) for lo, _ in STALENESS_BINS]


def acc_by(rows: list[dict], keyfn) -> dict[str, list[int]]:
    """{key: [0/1 correctness per question]} over every question of every row."""
    out: dict[str, list[int]] = {}
    for r in rows:
        for q in r.get("per_question", []):
            for k in keyfn(q):
                if k is not None:
                    out.setdefault(k, []).append(int(q["correct"]))
    return out


def paired(rows: list[dict], a: str, b: str) -> dict | None:
    ra = {(r["task_id"], r["rep"]): r["score"] for r in rows if r["strategy"] == a}
    rb = {(r["task_id"], r["rep"]): r["score"] for r in rows if r["strategy"] == b}
    keys = sorted(set(ra) & set(rb))
    if not keys:
        return None
    d = [ra[k] - rb[k] for k in keys]
    return {
        "n": len(d),
        "win": sum(x > 1e-9 for x in d),
        "tie": sum(abs(x) <= 1e-9 for x in d),
        "loss": sum(x < -1e-9 for x in d),
        "mean_diff": mean(d),
        "stderr": stderr(d),
    }


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


def make_report(run_dir: Path) -> Path:
    rows = load_rows(run_dir)
    if not rows:
        raise SystemExit(f"no rows in {run_dir / 'results.jsonl'}")
    mock = any(r.get("mock") for r in rows)
    strategies = [s for s in ["truncate", "summary", "indexed"] if any(r["strategy"] == s for r in rows)]
    strategies += sorted({r["strategy"] for r in rows} - set(strategies))
    by_s = {s: [r for r in rows if r["strategy"] == s] for s in strategies}
    pal = Palette(strategies)
    md: list[str] = []
    a = md.append
    charts: list[tuple[str, str]] = []

    a(f"# Eval report: {run_dir.name}")
    if mock:
        a(f"\n> **{MOCK_BANNER}.** Mock agent and compactor; numbers only verify the pipeline.")
    cfg_path = run_dir / "config.json"
    if cfg_path.exists():
        c = json.loads(cfg_path.read_text())
        a(f"\nConfig: model `{c.get('model')}`, compactor `{c.get('compactor_model')}`, cap {c.get('context_cap')}, "
          f"threshold {c.get('threshold')}, keep_last {c.get('keep_last')}, max_steps {c.get('max_steps')}, "
          f"{c.get('tasks')} tasks x {c.get('seeds')} replicate(s).")
    n_err = sum(1 for r in rows if r.get("error"))
    a(f"\n{len(rows)} runs, {n_err} with errors, {sum(1 for r in rows if r.get('phase2_forced'))} with phase 2 forced by step budget, "
      f"{sum(1 for r in rows if r.get('hit_cap_violation'))} cap violations. Total cost ${sum(r.get('cost_usd') or 0 for r in rows):.4f}.")

    # 1. accuracy
    a("\n## Accuracy per strategy\n")
    a("Mean task score (fraction of questions correct) ± standard error over runs.\n")
    a("| strategy | runs | accuracy | ± stderr | errors |")
    a("|---|---|---|---|---|")
    for s in strategies:
        sc = [r["score"] for r in by_s[s]]
        a(f"| {s} | {len(sc)} | {fmt(mean(sc))} | {fmt(stderr(sc))} | {sum(1 for r in by_s[s] if r.get('error'))} |")
    fig, ax = plt.subplots(figsize=(5, 3.6))
    _grouped_bars(ax, [""], {s: [(mean(r["score"] for r in by_s[s]), stderr(r["score"] for r in by_s[s]))] for s in strategies}, pal)
    ax.set_ylim(0, 1.05)
    _style(ax, "Accuracy per strategy", "mean score ± stderr", mock)
    _save(fig, run_dir / "accuracy.png")
    charts.append(("accuracy.png", "Accuracy per strategy"))

    # 2. by tag
    a("\n## Accuracy by question tag\n")
    tags = ["early", "late", "single_hop", "multi_hop"]
    a("| tag | " + " | ".join(strategies) + " |")
    a("|---|" + "---|" * len(strategies))
    series: dict[str, list[tuple[float, float]]] = {s: [] for s in strategies}
    for t in tags:
        cells = []
        for s in strategies:
            v = acc_by(by_s[s], lambda q, t=t: [t if t in q["tags"] else None]).get(t, [])
            cells.append(f"{fmt(mean(v))} (n={len(v)})")
            series[s].append((mean(v), stderr(v)))
        a(f"| {t} | " + " | ".join(cells) + " |")
    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    _grouped_bars(ax, tags, series, pal)
    ax.set_ylim(0, 1.05)
    _style(ax, "Accuracy by question tag", "fraction correct ± stderr (per question)", mock)
    _save(fig, run_dir / "accuracy_by_tag.png")
    charts.append(("accuracy_by_tag.png", "Accuracy by question tag"))

    # 3. staleness
    a("\n## Accuracy vs question staleness\n")
    a("Staleness = tokens of file content the agent read after it last read the oldest evidence file "
      "(questions whose evidence was never fully read are excluded).\n")
    a("| staleness | " + " | ".join(strategies) + " |")
    a("|---|" + "---|" * len(strategies))
    series = {s: [] for s in strategies}
    for b in BUCKETS:
        cells = []
        for s in strategies:
            v = acc_by(by_s[s], lambda q: [staleness_bucket(q.get("staleness"))]).get(b, [])
            cells.append(f"{fmt(mean(v))} (n={len(v)})")
            series[s].append((mean(v), stderr(v)))
        a(f"| {b} tokens | " + " | ".join(cells) + " |")
    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    _grouped_bars(ax, BUCKETS, series, pal)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("question staleness (tokens read since evidence)", color=INK)
    _style(ax, "Accuracy vs staleness", "fraction correct ± stderr", mock)
    _save(fig, run_dir / "accuracy_by_staleness.png")
    charts.append(("accuracy_by_staleness.png", "Accuracy vs staleness"))

    # 4. cost / tokens
    a("\n## Cost and tokens per task\n")
    a("| strategy | $ agent | $ compactor | $ total | agent in-tok | agent out-tok | compactor in-tok | compactor out-tok |")
    a("|---|---|---|---|---|---|---|---|")

    def purpose(r, p, f):
        return (r.get("usage", {}).get("by_purpose", {}).get(p, {}) or {}).get(f, 0) or 0

    cost = {}
    for s in strategies:
        rs = by_s[s]
        v = {
            f"{p}_{f}": mean(purpose(r, p, f) for r in rs)
            for p in ("agent", "compactor")
            for f in ("cost_usd", "input_tokens", "output_tokens", "cache_read_tokens")
        }
        cost[s] = v
        tot = v["agent_cost_usd"] + v["compactor_cost_usd"]
        a(f"| {s} | {v['agent_cost_usd']:.4f} | {v['compactor_cost_usd']:.4f} | {tot:.4f} | "
          f"{v['agent_input_tokens']:.0f} | {v['agent_output_tokens']:.0f} | "
          f"{v['compactor_input_tokens']:.0f} | {v['compactor_output_tokens']:.0f} |")
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8))
    x = range(len(strategies))
    ag = [cost[s]["agent_cost_usd"] for s in strategies]
    co = [cost[s]["compactor_cost_usd"] for s in strategies]
    axes[0].bar(x, ag, color=[pal[s] for s in strategies], label="agent")
    axes[0].bar(x, co, bottom=ag, color=[pal[s] for s in strategies], hatch="//", edgecolor="white", label="compactor (hatched)")
    axes[0].set_xticks(list(x), strategies)
    axes[0].legend(frameon=False, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=2)
    _style(axes[0], "Cost per task", "USD", mock)
    ai = [cost[s]["agent_input_tokens"] for s in strategies]
    ci = [cost[s]["compactor_input_tokens"] for s in strategies]
    axes[1].bar(x, ai, color=[pal[s] for s in strategies])
    axes[1].bar(x, ci, bottom=ai, color=[pal[s] for s in strategies], hatch="//", edgecolor="white")
    axes[1].set_xticks(list(x), strategies)
    _style(axes[1], "Input tokens per task (compactor hatched)", "tokens", mock)
    _save(fig, run_dir / "cost_tokens.png")
    charts.append(("cost_tokens.png", "Cost and tokens per task"))

    # 5. compactions
    a("\n## Compactions per run\n")
    a("| strategy | mean | min | max | mean steps |")
    a("|---|---|---|---|---|")
    fig, ax = plt.subplots(figsize=(5, 3.6))
    ms, es = [], []
    for s in strategies:
        n = [len(r.get("compactions", [])) for r in by_s[s]]
        a(f"| {s} | {fmt(mean(n), 2)} | {min(n)} | {max(n)} | {fmt(mean(r.get('steps', 0) for r in by_s[s]), 1)} |")
        ms.append(mean(n))
        es.append(stderr(n))
    ax.bar(range(len(strategies)), ms, yerr=es, capsize=3, color=[pal[s] for s in strategies])
    ax.set_xticks(range(len(strategies)), strategies)
    _style(ax, "Compactions per run", "count ± stderr", mock)
    _save(fig, run_dir / "compactions.png")
    charts.append(("compactions.png", "Compactions per run"))

    # 6. indexed retrieval
    idx_rows = [r for r in rows if r["strategy"] == "indexed"]
    if idx_rows:
        a("\n## Indexed retention: retrieval\n")
        fc = [r.get("fetch_calls", 0) for r in idx_rows]
        rt = [r.get("retrieved_tokens", 0) for r in idx_rows]
        a(f"Per run: fetch_messages calls mean {fmt(mean(fc), 2)} (± {fmt(stderr(fc), 2)}), "
          f"retrieved tokens mean {fmt(mean(rt), 0)} (± {fmt(stderr(rt), 0)}).\n")
        fetched = acc_by(idx_rows, lambda q: ["fetched" if q.get("evidence_fetched") else "not fetched"])
        a("Accuracy by whether a fetch result contained the question's evidence file path(s) "
          "(best effort; the agent may also have fetched indirectly):\n")
        a("| evidence retrieved? | accuracy | questions |")
        a("|---|---|---|")
        for k in ("fetched", "not fetched"):
            a(f"| {k} | {fmt(mean(fetched.get(k, [])))} | {len(fetched.get(k, []))} |")
        fig, axes = plt.subplots(1, 3, figsize=(11, 3.6))
        col = pal["indexed"]
        axes[0].hist(fc, bins=range(max(fc + [1]) + 2), color=col, align="left")
        _style(axes[0], "fetch_messages calls per run", "runs", mock)
        axes[1].hist(rt, bins=min(10, max(1, len(rt))), color=col)
        _style(axes[1], "retrieved tokens per run", "runs", mock)
        ks = ["fetched", "not fetched"]
        axes[2].bar(range(2), [mean(fetched.get(k, [])) if fetched.get(k) else 0 for k in ks],
                    yerr=[stderr(fetched.get(k, [])) for k in ks], capsize=3, color=col)
        axes[2].set_xticks(range(2), ks)
        axes[2].set_ylim(0, 1.05)
        _style(axes[2], "Accuracy by evidence fetched", "fraction correct", mock)
        _save(fig, run_dir / "indexed_retrieval.png")
        charts.append(("indexed_retrieval.png", "Indexed retention retrieval"))

    # 7. paired
    if len(strategies) > 1:
        a("\n## Paired comparison (same task and replicate)\n")
        a("Difference = score(A) − score(B). Win = A scored higher.\n")
        a("| A | B | pairs | win | tie | loss | mean diff | ± stderr |")
        a("|---|---|---|---|---|---|---|---|")
        for x, y in itertools.combinations(strategies, 2):
            p = paired(rows, x, y)
            if p:
                a(f"| {x} | {y} | {p['n']} | {p['win']} | {p['tie']} | {p['loss']} | {p['mean_diff']:+.3f} | {p['stderr']:.3f} |")

    # errors
    errs = [r for r in rows if r.get("error")]
    if errs:
        a("\n## Run errors\n")
        by_err: dict[tuple[str, str], int] = {}
        for r in errs:
            k = (r["strategy"], str(r["error"]).split(":")[0][:60])
            by_err[k] = by_err.get(k, 0) + 1
        for (s, e), n in sorted(by_err.items()):
            a(f"- {s}: `{e}` x{n}")

    a("\n## Charts\n")
    for f, t in charts:
        a(f"![{t}]({f})\n")
    if mock:
        a(f"\n*{MOCK_BANNER}.*")
    out = run_dir / "report.md"
    out.write_text("\n".join(md) + "\n", encoding="utf-8")
    return out


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="compaction-report", description=__doc__)
    p.add_argument("--run", required=True, help="results/runs/<name> directory")
    a = p.parse_args(argv)
    out = make_report(Path(a.run))
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
