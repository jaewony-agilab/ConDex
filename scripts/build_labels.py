"""Phase 2: per-boundary labels from the recorder output, split by (task, episode seed).
Writes labels.jsonl (+ counts in labels_summary.json). Privileged: runs from scripts/ only."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from condex import config as C  # noqa: E402
from condex.episode import EpisodeReader, find_episodes  # noqa: E402
from condex.privileged.labels import LabelConfig, label_episode, split_of  # noqa: E402

SEED_STRIDE = 100000  # condex.sim.env: episode seed = env_seed * 100000 + episode_id


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("raw")
    ap.add_argument("--out", required=True, help="directory for labels.jsonl")
    ap.add_argument("--config", default=None)
    args = ap.parse_args()
    cfg = C.load(args.config)
    lc = LabelConfig(**cfg["labels"])
    fr = (cfg["split"]["train"], cfg["split"]["val"])
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for ep in find_episodes(args.raw):
        d = EpisodeReader(ep)
        if d.meta.get("condition", "executor_alone") != "executor_alone":
            continue  # labels come from executor-alone rollouts; closed-loop runs are scored, not trained on
        ep_seed = int(d.meta["seed"]) * SEED_STRIDE + int(d.meta["episode_id"])
        for r in label_episode(d, lc):
            rows.append({**r, "episode_seed": ep_seed, "split": split_of(r["task"], ep_seed, fr)})
    with (out / "labels.jsonl").open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    summary = {"run": C.run_info(cfg["_path"]), "label_config": cfg["labels"], "n": len(rows),
               "by_split": dict(Counter(r["split"] for r in rows)),
               "by_class": dict(Counter(f"{r['type']}/{r['outcome']}/{r['plan_still_valid']}" for r in rows)),
               "by_reason": dict(Counter(r["reason"] for r in rows).most_common(30))}
    (out / "labels_summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
