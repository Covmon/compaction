import hashlib
import json
import os
import re
import subprocess
import sys

import pytest

from compaction.eval.world import estimate_tokens_text, generate_world

SEEDS = list(range(10))


def _digest(w) -> str:
    blob = json.dumps({"files": w.files, "order": w.reading_order, "facts": w.facts},
                      sort_keys=True, default=str)
    return hashlib.sha256(blob.encode()).hexdigest()


def test_deterministic_by_seed():
    assert _digest(generate_world(3)) == _digest(generate_world(3))
    assert _digest(generate_world(3)) != _digest(generate_world(4))


def test_deterministic_across_hash_seeds():
    code = ("import hashlib,json;from compaction.eval.world import generate_world;w=generate_world(7);"
            "print(hashlib.sha256(json.dumps({'files':w.files,'order':w.reading_order,'facts':w.facts},"
            "sort_keys=True,default=str).encode()).hexdigest())")
    outs = set()
    for hs in ("1", "2", "999"):
        env = dict(os.environ, PYTHONHASHSEED=hs)
        outs.add(subprocess.run([sys.executable, "-c", code], env=env, capture_output=True,
                                text=True, check=True).stdout.strip())
    assert len(outs) == 1
    assert outs == {_digest(generate_world(7))}


@pytest.mark.parametrize("seed", SEEDS)
def test_corpus_size_bounds(seed):
    w = generate_world(seed)
    assert 25 <= len(w.files) <= 40
    for path, text in w.files.items():
        assert 1400 <= len(text) <= 4300, (path, len(text))
    assert w.corpus_tokens() >= 25_000
    assert w.corpus_tokens() == sum(estimate_tokens_text(t) for t in w.files.values())
    assert sorted(w.reading_order) == sorted(w.files)


@pytest.mark.parametrize("seed", SEEDS)
def test_planted_facts_consistent(seed):
    w = generate_world(seed)
    f = w.facts
    tickets = {p: t for p, t in w.files.items() if p.startswith("tickets/")}
    for i in f["incidents"].values():
        # The service port is mentioned in exactly one ticket.
        mentions = [p for p, t in tickets.items() if f"port {i['port']}" in t]
        assert mentions == [i["path"]]
        assert f"| Impact start | {i['impact_start']} |" in w.files[i["path"]]
        assert i["trigger_deploy"] in w.files[i["path"]]
        d = f["deploy_by_id"][i["trigger_deploy"]]
        assert d["commit"] in w.files[d["path"]]
    # Deploy ids and commits are unique.
    assert len({d["deploy_id"] for d in f["deploys"]}) == len(f["deploys"])
    assert len({d["commit"] for d in f["deploys"]}) == len(f["deploys"])
    # The planted first error really is the first ERROR line of each log.
    for lg in f["logs"].values():
        errs = [ln for ln in w.files[lg["path"]].splitlines() if " ERROR " in ln]
        assert len(errs) == 2
        assert errs[0].startswith(lg["first_error"]["ts"]) and lg["first_error"]["code"] in errs[0]
    # Superseded and staging values are present as distractors but differ.
    for c in f["configs"].values():
        text = w.files[c["path"]]
        assert re.search(rf"^  port: {c['port']}\b", text, re.MULTILINE)
        assert str(c["old_port"]) in text and str(c["staging_port"]) in text
        assert c["old_port"] != c["port"]
    # Two current people share a first name (distractor).
    firsts = [p["name"].split()[0] for p in f["people"]]
    assert len(set(firsts)) < len(firsts)
