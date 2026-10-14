"""Phase 1 step 4: force distribution (sets the overlay clip), contact rates per finger, boundary
counts per type, label balance (if labels exist), disk per episode."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from condex.episode import EpisodeReader, disk_usage, find_episodes  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("raw")
    ap.add_argument("--out", required=True)
    ap.add_argument("--labels", default=None, help="labels.jsonl from build_labels.py")
    ap.add_argument("--contact-n", type=float, default=0.5)
    args = ap.parse_args()
    tip_f, palm_f = [], []
    contact_steps, steps = Counter(), 0
    btypes, per_task = Counter(), Counter()
    disk = defaultdict(list)
    eps = find_episodes(args.raw)
    for ep in eps:
        d = EpisodeReader(ep)
        per_task[d.meta.get("task")] += 1
        for b in d.boundaries:
            btypes[(b["hand"], b["type"])] += 1
        for s in d.steps():
            steps += 1
            tot = defaultdict(lambda: np.zeros(3))
            for r in d.taxel_rows(s):
                tot[(r["hand"], r["finger"])] += np.asarray(r["force_world"])
            for (h, f), v in tot.items():
                n = float(np.linalg.norm(v))
                if n >= args.contact_n:
                    contact_steps[f"{h}/{f}"] += 1
                    (palm_f if f == "palm" else tip_f).append(n)
        for k, v in disk_usage(ep).items():
            disk[k].append(v)
    pct = lambda x: {f"p{p}": round(float(np.percentile(x, p)), 3) for p in (50, 90, 95, 99)} if x else {}  # noqa: E731
    out = {"episodes": len(eps), "episodes_per_task": dict(per_task), "steps": steps,
           "fingertip_force_in_contact_n": {"n": len(tip_f), **pct(tip_f)},
           "palm_force_in_contact_n": {"n": len(palm_f), **pct(palm_f)},
           "suggested_force_clip_n": round(float(np.percentile(tip_f, 95)), 2) if tip_f else None,
           "contact_rate_per_link": {k: round(v / max(1, steps), 4) for k, v in sorted(contact_steps.items())},
           "boundaries": {f"{h}/{t}": n for (h, t), n in sorted(btypes.items())},
           "disk_mb_per_episode": {k: round(float(np.mean(v)) / 1e6, 2) for k, v in disk.items()}}
    if args.labels:
        rows = [json.loads(line) for line in Path(args.labels).read_text().splitlines() if line.strip()]
        out["labels"] = {"by_type_outcome_valid": dict(Counter(f"{r['type']}/{r['outcome']}/{r['plan_still_valid']}" for r in rows)),
                         "by_split": dict(Counter(r["split"] for r in rows))}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
