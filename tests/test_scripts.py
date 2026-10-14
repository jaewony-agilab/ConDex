"""Offline scripts end to end on synthetic records (no simulator, no model)."""

import json
import subprocess
import sys
from pathlib import Path

from conftest import make_episode
from test_labels import facts_fn

REPO = Path(__file__).resolve().parents[1]


def sh(*args):
    r = subprocess.run([sys.executable, *map(str, args)], cwd=REPO, capture_output=True, text=True)
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-4000:]
    return r.stdout


def test_offline_pipeline(tmp_path):
    raw = tmp_path / "raw"
    bounds = [{"step": 15, "hand": "right", "type": "grasp", "settled_step": 20, "subtask": "pick", "retry": 0},
              {"step": 55, "hand": "right", "type": "release", "settled_step": 60, "subtask": "place", "retry": 0}]
    for i in range(4):
        ok = i % 2 == 0
        make_episode(raw / "TaskA" / "alone_s42" / f"ep_{i:04d}", n_steps=100, boundaries=bounds,
                     forces_at=lambda s: {"thumb": [0, 2.0, 0], "index": [2.0, 0, 0]} if 20 <= s < 58 else {},
                     facts=facts_fn(grasp=range(20, 58) if ok else range(20, 30),
                                    z=lambda s: 0.8 + 0.004 * max(0, min(s, 50) - 22), inr=range(58, 100) if ok else ()),
                     meta={"task": "TaskA", "episode_id": i, "condition": "executor_alone",
                           "summary": {"success": ok}})
    out = tmp_path / "out"
    sh("scripts/build_labels.py", raw, "--out", out / "labels")
    labels = [json.loads(x) for x in (out / "labels" / "labels.jsonl").read_text().splitlines()]
    assert len(labels) == 8 and {r["outcome"] for r in labels} >= {"success", "fail"}
    sh("scripts/build_judge_dataset.py", out / "labels" / "labels.jsonl", "--out", out / "ds", "--conditions", "plain", "overlay")
    meta = json.loads((out / "ds" / "overlay" / "dataset.json").read_text())
    assert meta["style_hash"] and sum(sum(v.values()) for v in meta["counts"].values()) == 8
    sh("scripts/render_check.py", raw, "--out", out / "render", "--per-task", 3)
    rc = json.loads((out / "render" / "render_check.json").read_text())
    assert rc["frames"] == 3 and rc["survival_pass_rate"] == 1.0
    sh("scripts/dataset_stats.py", raw, "--out", out / "stats.json", "--labels", out / "labels" / "labels.jsonl")
    st = json.loads((out / "stats.json").read_text())
    assert st["episodes"] == 4 and st["suggested_force_clip_n"] > 0 and st["boundaries"]["right/grasp"] == 4
    sh("scripts/check_boundaries.py", raw, "--out", out / "bsheets", "--episodes", 2)
    assert len(list((out / "bsheets").glob("*.png"))) == 4
    sh("scripts/failure_review.py", raw, "--out", out / "fail")
    assert len((out / "fail" / "failure_labels.csv").read_text().splitlines()) == 1 + 2  # header + 2 failed episodes
    res = tmp_path / "results"
    for t, sr in (("TaskA", 0.5), ("TaskB", 0.9)):
        (res / t).mkdir(parents=True)
        (res / t / "summary.json").write_text(json.dumps({"task": t, "condition": "alone", "seed": 42, "episodes": 20, "success_rate": sr}))
    assert "TaskA" in sh("scripts/summarize.py", "select", res).split("selected_tasks:")[1]
    assert "| TaskA | alone | 42 | 0.50 | 20 |" in sh("scripts/summarize.py", "sr", res)
