"""Phase 0 step 4: do boundaries land after the fingers settle? Local or cluster, from records.

For each boundary of the first N episodes: a strip of the policy-camera frames from a few steps
before the command crossing to a few steps after the settled step, and a trace of the commanded
closure and the finger-joint speed with the crossing (|) and settled (:) steps marked. Look at them.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from condex import config as C  # noqa: E402
from condex.episode import EpisodeReader, find_episodes  # noqa: E402


def trace(d, b, cl, lo, hi, w=600, h=120):
    img = Image.new("RGB", (w, h), (255, 255, 255))
    dr = ImageDraw.Draw(img)
    steps = [s for s in range(lo, hi + 1) if s in d.proprio]
    if len(steps) < 2:
        return img
    x = lambda s: (s - lo) / max(1, hi - lo) * (w - 1)  # noqa: E731
    clo = [cl.closure(d.proprio[s].get(f"hand_cmd.{b['hand']}", [0] * 6)) for s in steps]
    q = np.array([d.proprio[s].get(f"finger_pos.{b['hand']}", [0] * 6) for s in steps])
    vel = np.r_[0, np.abs(np.diff(q, axis=0)).max(1)]
    vmax = max(vel.max(), 1e-6)
    dr.line([(x(s), h - 10 - c * (h - 20)) for s, c in zip(steps, np.clip(clo, 0, 1))], fill=(0, 90, 200), width=2)
    dr.line([(x(s), h - 10 - v / vmax * (h - 20)) for s, v in zip(steps, vel)], fill=(220, 60, 0), width=1)
    dr.line([(x(b["step"]), 0), (x(b["step"]), h)], fill=(0, 0, 0), width=1)
    for yy in range(0, h, 6):
        dr.line([(x(b["settled_step"]), yy), (x(b["settled_step"]), yy + 3)], fill=(0, 0, 0))
    dr.text((4, 2), "blue: commanded closure  red: finger joint speed  |: crossing  : settled", fill=(0, 0, 0))
    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("raw")
    ap.add_argument("--out", required=True)
    ap.add_argument("--episodes", type=int, default=10)
    ap.add_argument("--camera", default="ego_view")
    ap.add_argument("--config", default=None)
    args = ap.parse_args()
    cfg = C.load(args.config)
    cls = C.closure_configs(cfg)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    report = []
    for ep in find_episodes(args.raw)[: args.episodes]:
        d = EpisodeReader(ep)
        for i, b in enumerate(d.boundaries):
            lo, hi = b["step"] - 4, b["settled_step"] + 4
            steps = [s for s in d.frame_steps(args.camera) if lo <= s <= hi]
            steps = steps[:: max(1, len(steps) // 8)][:8]
            frames = [Image.fromarray(d.frame(args.camera, s)).resize((160, 160)) for s in steps]
            sheet = Image.new("RGB", (max(600, 160 * len(frames)), 160 + 120), (255, 255, 255))
            for k, (s, f) in enumerate(zip(steps, frames)):
                sheet.paste(f, (160 * k, 0))
                ImageDraw.Draw(sheet).text((160 * k + 3, 3), str(s), fill=(255, 255, 0))
            sheet.paste(trace(d, b, cls[b["hand"]], lo, hi), (0, 160))
            name = f"{ep.parent.name}_{ep.name}_b{i:02d}_{b['hand']}_{b['type']}.png"
            sheet.save(out / name)
            report.append({"episode": str(ep), "boundary": i, "hand": b["hand"], "type": b["type"], "crossing": b["step"],
                           "settled": b["settled_step"], "settle_steps": b["settled_step"] - b["step"], "png": name})
    (out / "boundaries.json").write_text(json.dumps(report, indent=1))
    n = len(report)
    print(f"{n} boundaries; settle steps median {np.median([r['settle_steps'] for r in report]) if n else 'n/a'}; sheets in {out}")


if __name__ == "__main__":
    main()
