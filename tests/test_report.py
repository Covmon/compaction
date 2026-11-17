import json

from compaction.eval import report, runner


def _make_run(tmp_path, strategies=("truncate", "summary")):
    cfg, out = runner.parse_args(["--mock", "--strategies", ",".join(strategies), "--tasks", "2",
                                  "--out", str(tmp_path), "--run-name", "r", "--workers", "2"])
    runner.run_sweep(cfg, out, log=lambda *_: None)
    return tmp_path / "runs" / "r"


def test_report_files(tmp_path):
    run = _make_run(tmp_path)
    md = report.make_report(run)
    text = md.read_text()
    assert "MOCK — plumbing test, not a result" in text
    for sec in ("Accuracy per strategy", "Accuracy by question tag", "staleness", "Compactions per run",
                "Paired comparison", "Cost and tokens"):
        assert sec in text
    for png in ("accuracy.png", "accuracy_by_tag.png", "accuracy_by_staleness.png", "cost_tokens.png", "compactions.png"):
        assert (run / png).stat().st_size > 1000


def test_report_with_indexed_and_cli(tmp_path):
    try:
        from compaction.strategies import make_strategy

        make_strategy("indexed")
    except ImportError:
        return
    run = _make_run(tmp_path, ("truncate", "summary", "indexed"))
    assert report.main(["--run", str(run)]) == 0
    assert "Indexed retention: retrieval" in (run / "report.md").read_text()
    assert (run / "indexed_retrieval.png").exists()


def test_paired_and_stats():
    rows = [
        {"task_id": "a", "rep": 0, "strategy": "x", "score": 1.0},
        {"task_id": "b", "rep": 0, "strategy": "x", "score": 0.5},
        {"task_id": "c", "rep": 0, "strategy": "x", "score": 0.2},
        {"task_id": "a", "rep": 0, "strategy": "y", "score": 0.5},
        {"task_id": "b", "rep": 0, "strategy": "y", "score": 0.5},
        {"task_id": "c", "rep": 0, "strategy": "y", "score": 0.9},
    ]
    p = report.paired(rows, "x", "y")
    assert (p["win"], p["tie"], p["loss"]) == (1, 1, 1)
    assert abs(p["mean_diff"] - (0.5 + 0 - 0.7) / 3) < 1e-9
    assert report.stderr([1.0]) == 0.0
    assert report.staleness_bucket(None) is None and report.staleness_bucket(9000) == "8-16k"


def test_report_handles_error_rows(tmp_path):
    run = tmp_path / "e"
    run.mkdir()
    row = {"task_id": "t", "strategy": "truncate", "rep": 0, "score": 0.0, "error": "RuntimeError: x", "mock": False}
    (run / "results.jsonl").write_text(json.dumps(row) + "\n")
    text = report.make_report(run).read_text()
    assert "Run errors" in text and "MOCK —" not in text.split("## Charts")[0].split("\n", 3)[2]
