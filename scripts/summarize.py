"""Local tables from results/ (PLAN.md §8: analysis is generated locally from results/).

  sr      success rate per task x condition (+ seed) from every summary.json under a results dir,
          with the Phase 0 reproduction gate (|ours - published| <= 10pp) when published SR is known
  select  Phase 0 step 5: tasks with executor-alone SR inside the config's band -> selected_tasks YAML
  judge   Phase 3: per-condition judge metrics from every metrics.json under a results dir
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from condex import config as C  # noqa: E402


def summaries(root):
    return [json.loads(p.read_text()) | {"_path": str(p)} for p in sorted(Path(root).rglob("summary.json"))]


def short(task):
    return task.replace("_GR1ArmsAndWaistFourierHands_Env", "").replace("PosttrainPnPNovel", "Novel")


def cmd_sr(args, cfg):
    rows = defaultdict(lambda: [0, 0])
    extra = defaultdict(lambda: defaultdict(float))
    for s in summaries(args.root):
        key = (s["task"], s.get("condition", "alone") + (f"_{s['judge']}" if s.get("condition") in ("plain", "text", "overlay") else ""), s.get("seed"))
        rows[key][0] += round(s["success_rate"] * s["episodes"])
        rows[key][1] += s["episodes"]
        for k in ("recoveries", "replans", "judge_calls", "aborted"):
            extra[key][k] += s.get(k, 0)
        extra[key]["cost_usd"] += s.get("cost", {}).get("run_spent_usd", 0.0)
    pub = cfg.get("published_sr") or {}
    print("| task | condition | seed | SR | n | recoveries/ep | replans/ep | $/ep | gate |")
    print("|---|---|---|---|---|---|---|---|---|")
    for (task, cond, seed), (k, n) in sorted(rows.items()):
        e = extra[(task, cond, seed)]
        gate = ""
        if cond == "alone" and task in pub:
            gate = "ok" if abs(k / n - pub[task]) <= 0.10 else f"FAIL (pub {pub[task]:.2f})"
        print(f"| {short(task)} | {cond} | {seed} | {k / n:.2f} | {n} | {e['recoveries'] / n:.2f} | {e['replans'] / n:.2f} | "
              f"{e['cost_usd'] / n:.4f} | {gate} |")
    alone = [k / n for (t, c, _), (k, n) in rows.items() if c == "alone"]
    if alone:
        mean = sum(alone) / len(alone)
        print(f"\nexecutor-alone mean SR over {len(alone)} task runs: {mean:.3f} (published mean {cfg.get('published_mean_sr')})")


def cmd_select(args, cfg):
    lo, hi = cfg.get("select_band", [0.3, 0.7])
    by_task = defaultdict(lambda: [0, 0])
    for s in summaries(args.root):
        if s.get("condition", "alone") == "alone":
            by_task[s["task"]][0] += round(s["success_rate"] * s["episodes"])
            by_task[s["task"]][1] += s["episodes"]
    sel = [t for t, (k, n) in sorted(by_task.items()) if lo <= k / n <= hi]
    for t, (k, n) in sorted(by_task.items(), key=lambda kv: kv[1][0] / kv[1][1]):
        print(f"{k / n:.2f}  {'*' if t in sel else ' '} {short(t)}  (n={n})")
    print(f"\n# {len(sel)} tasks in [{lo}, {hi}]; gate needs >= 3 with mostly slip/drop failures (scripts/failure_review.py)")
    print("selected_tasks:")
    for t in sel:
        print(f"  - {t}")


def cmd_judge(args, cfg):
    ms = [json.loads(p.read_text()) | {"_path": str(p)} for p in sorted(Path(args.root).rglob("metrics.json"))]
    print("| judge | condition | split | n | outcome acc | grasp | release | fail recall | fail prec | valid acc (fail) | parse fail |")
    print("|---|---|---|---|---|---|---|---|---|---|---|")
    for m in ms:
        s = m["metrics"]
        f = lambda k: f"{s[k]:.3f}" if k in s else "-"  # noqa: E731
        print(f"| {m.get('judge')} | {m.get('condition')} | {m.get('split')} | {s['n']} | {f('outcome_acc')} | {f('outcome_acc_grasp')} | "
              f"{f('outcome_acc_release')} | {f('fail_recall')} | {f('fail_precision')} | {f('valid_acc_on_fail')} | {f('parse_fail_rate')} |")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["sr", "select", "judge"])
    ap.add_argument("root")
    ap.add_argument("--config", default=None)
    args = ap.parse_args()
    cfg = C.load(args.config)
    {"sr": cmd_sr, "select": cmd_select, "judge": cmd_judge}[args.cmd](args, cfg)


if __name__ == "__main__":
    main()
