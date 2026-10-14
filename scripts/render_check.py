"""Phase 1 steps 2-3: render random boundaries per task with a style and check them.

For each sampled boundary and camera: plain | overlay | overlay after the judge's preprocessing,
side by side, plus the per-marker survival check (condex.survival). Writes the sheets and
render_check.json (style hash, pass rate, failures). Inspect the sheets: arrows only during
contact, colors stable, anchors on the fingertips, no drift.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from condex import render as R  # noqa: E402
from condex.episode import find_episodes, load  # noqa: E402
from condex.preprocess import to_model_input  # noqa: E402
from condex.survival import mark_survival  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("raw")
    ap.add_argument("--style", default="configs/style/default.yaml")
    ap.add_argument("--out", required=True)
    ap.add_argument("--per-task", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    style = R.load_style(args.style)
    out = Path(args.out)
    (out / "sheets").mkdir(parents=True, exist_ok=True)
    by_task = defaultdict(list)
    for ep in find_episodes(args.raw):
        d = load(str(ep))
        by_task[d.meta.get("task")] += [(str(ep), i) for i in range(len(d.boundaries))]
    rng = random.Random(args.seed)
    results, fails = [], []
    for task, items in sorted(by_task.items()):
        for ep, i in rng.sample(items, min(args.per_task, len(items))):
            d = load(ep)
            step = int(d.boundaries[i]["settled_step"])
            for cam in d.camera_names():
                if not d.frame_path(cam, step).exists():
                    continue
                plain = d.frame(cam, step)
                marks = R.make_marks(d, step, cam, style)
                over = R.draw_marks(plain, marks, style)
                model = to_model_input(over, style.judge)
                h = plain.shape[0]
                panel = Image.new("RGB", (plain.shape[1] * 2 + model.shape[1] * h // model.shape[0], h), (255, 255, 255))
                panel.paste(Image.fromarray(plain), (0, 0))
                panel.paste(Image.fromarray(over), (plain.shape[1], 0))
                panel.paste(Image.fromarray(model).resize((model.shape[1] * h // model.shape[0], h), Image.NEAREST), (2 * plain.shape[1], 0))
                name = f"{task[:40]}_{Path(ep).name}_b{i:02d}_{cam}.png"
                panel.save(out / "sheets" / name)
                surv = mark_survival(plain, marks, style)
                r = {"task": task, "episode": ep, "boundary": i, "camera": cam, "sheet": name, "marks": surv,
                     "n_contact": sum(m.contact for m in marks), "n_in_view": sum(m.in_view for m in marks)}
                results.append(r)
                fails += [{**s, "sheet": name} for s in surv if not s["ok"]]
    n_marks = sum(len(r["marks"]) for r in results)
    summary = {"style": args.style, "style_hash": style.hash(), "frames": len(results), "drawn_marks": n_marks,
               "survival_pass_rate": 1 - len(fails) / max(1, n_marks), "failures": fails[:200],
               "visible_frac_p5": float(np.percentile([s["visible_frac"] for r in results for s in r["marks"]], 5)) if n_marks else None}
    (out / "render_check.json").write_text(json.dumps({**summary, "frames_detail": results}, indent=1))
    print(json.dumps({k: v for k, v in summary.items() if k != "failures"}, indent=1))


if __name__ == "__main__":
    main()
