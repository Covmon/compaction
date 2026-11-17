import json

import pytest

from compaction.eval import runner
from compaction.strategies import make_strategy


def _strategies():
    out = ["truncate", "summary"]
    try:
        make_strategy("indexed")
        out.append("indexed")
    except ImportError:
        pass
    return out


def _rows(out, name="t"):
    p = out / "runs" / name / "results.jsonl"
    return [json.loads(line) for line in p.read_text().splitlines() if line.strip()]


def _run(out, *extra, strategies=None, tasks=2):
    argv = ["--mock", "--strategies", ",".join(strategies or _strategies()), "--tasks", str(tasks),
            "--out", str(out), "--run-name", "t", "--workers", "2", *extra]
    cfg, o = runner.parse_args(argv)
    msgs = []
    runner.run_sweep(cfg, o, log=msgs.append)
    return msgs


def test_mock_sweep_rows(tmp_path):
    strategies = _strategies()
    _run(tmp_path)
    rows = _rows(tmp_path)
    assert len(rows) == 2 * len(strategies)
    assert {r["strategy"] for r in rows} == set(strategies)
    cfg = json.loads((tmp_path / "runs" / "t" / "config.json").read_text())
    assert cfg["mock"] is True
    for r in rows:
        assert r["error"] is None, r.get("traceback")
        assert r["mock"] and r["answered"] and 0 <= r["score"] <= 1
        assert len(r["per_question"]) == 10 and "tags" in r["per_question"][0]
        assert r["steps"] > 0 and r["compactions"], "16k cap must force compactions"
        assert r["usage"]["by_purpose"]["agent"]["calls"] == r["steps"]
        assert r["cost_usd"] > 0 and "question_staleness" in r["env_stats"]
        assert "fetch_calls" in r and "retrieved_tokens" in r and r["wall_time_s"] >= 0
        assert not r["hit_cap_violation"]
        assert (tmp_path / "runs" / "t" / f"{r['task_id']}__{r['strategy']}__r0" / "transcript.jsonl").exists()
        if r["strategy"] == "indexed":
            assert r["fetch_calls"] >= 1 and r["retrieved_tokens"] > 0 and "index_stats" in r
    by = {s: [r["score"] for r in rows if r["strategy"] == s] for s in strategies}
    assert all(v for v in by.values())


def test_job_order_interleaved():
    cfg = runner.RunConfig(strategies=["a", "b", "c"], tasks=2, seeds=2)
    jobs = runner.make_jobs(cfg, runner.make_tasks(2))
    assert [j.strategy for j in jobs[:3]] == ["a", "b", "c"]
    assert {j.rep for j in jobs[:6]} == {0}


def test_resumable(tmp_path):
    _run(tmp_path, strategies=["truncate"])
    n = len(_rows(tmp_path))
    msgs = _run(tmp_path, strategies=["truncate"])
    assert len(_rows(tmp_path)) == n
    assert "0 to do" in msgs[0]
    # extend: one more task adds only the new rows
    _run(tmp_path, strategies=["truncate"], tasks=3)
    assert len(_rows(tmp_path)) == 3


def test_budget_cutoff(tmp_path):
    msgs = _run(tmp_path, "--budget-usd", "0.001", "--workers", "1", strategies=["truncate"], tasks=3)
    assert len(_rows(tmp_path)) == 1
    assert any("Budget" in m for m in msgs)
    # resuming with budget lifted finishes the rest
    _run(tmp_path, strategies=["truncate"], tasks=3)
    assert len(_rows(tmp_path)) == 3


def test_error_captured(tmp_path, monkeypatch):
    real = runner.build_strategy

    def boom(cfg, name):
        if name == "summary":
            raise RuntimeError("kaboom")
        return real(cfg, name)

    monkeypatch.setattr(runner, "build_strategy", boom)
    _run(tmp_path, strategies=["truncate", "summary"])
    rows = _rows(tmp_path)
    assert len(rows) == 4
    bad = [r for r in rows if r["strategy"] == "summary"]
    assert all("kaboom" in r["error"] and r["score"] == 0 for r in bad)
    assert all(r["error"] is None for r in rows if r["strategy"] == "truncate")


def test_phase2_forced(tmp_path):
    _run(tmp_path, "--max-steps", "20", strategies=["truncate"], tasks=1)
    (r,) = _rows(tmp_path)
    assert r["phase2_forced"] and r["answered"] and r["error"] is None


@pytest.mark.parametrize("strategy", ["truncate", "summary"])
def test_phase_guard_unit(strategy):
    from compaction.eval.tasks import InvestigationEnvironment, make_task

    env = InvestigationEnvironment(make_task(0))
    g = runner.PhaseGuardEnv(env, 3)
    assert g.pending_user_message() is None and g.pending_user_message() is None
    assert "questions" in (g.pending_user_message() or "").lower() or env.phase == 2
    assert g.forced and env.phase == 2
