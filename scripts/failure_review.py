"""Phase 0 step 5: label 5 executor-alone failures per task from video.

Writes, per task, a GIF of the policy camera for up to N failed episodes and a CSV to fill in
(slip / drop / reach-miss / wrong-object / other). The `suggested` column comes from the privileged
events (drop -> drop; no grasp_ok -> reach-miss; grasp_ok without lift -> slip; ...) and is only a
hint for the human label.
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from condex.episode import EpisodeReader, find_episodes  # noqa: E402


def suggest(d) -> str:
    ev = {(e["name"], e.get("hand")) for e in d.events if e.get("kind") == "event"}
    names = {n for n, _ in ev}
    if "drop" in names:
        return "drop"
    if "grasp_ok" not in names:
        return "reach-miss"
    if "lift" not in names:
        return "slip"
    facts = [e for e in d.events if e.get("kind") == "fact" and e.get("name") == "grasped_other" and e["value"][0] > 0.5]
    if facts:
        return "wrong-object"
    return "other"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("raw")
    ap.add_argument("--out", required=True)
    ap.add_argument("--per-task", type=int, default=5)
    ap.add_argument("--camera", default="ego_view")
    ap.add_argument("--stride", type=int, default=4)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    picked = defaultdict(list)
    for ep in find_episodes(args.raw):
        d = EpisodeReader(ep)
        if d.meta.get("condition") != "executor_alone" or d.meta.get("summary", {}).get("success", True):
            continue
        t = d.meta["task"]
        if len(picked[t]) < args.per_task:
            picked[t].append(d)
    with (out / "failure_labels.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["task", "episode", "gif", "suggested", "label (slip|drop|reach-miss|wrong-object|other)", "notes"])
        for t, ds in sorted(picked.items()):
            for d in ds:
                steps = d.frame_steps(args.camera)[:: args.stride]
                frames = [Image.fromarray(d.frame(args.camera, s)) for s in steps]
                gif = f"{t[:40]}_{d.root.name}.gif"
                if frames:
                    frames[0].save(out / gif, save_all=True, append_images=frames[1:], duration=100, loop=0)
                w.writerow([t, str(d.root), gif, suggest(d), "", ""])
    print(f"{sum(len(v) for v in picked.values())} failures over {len(picked)} tasks -> {out}/failure_labels.csv")


if __name__ == "__main__":
    main()
