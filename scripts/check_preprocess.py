"""Verify that our pre-resize leaves nothing for the real Qwen3-VL processor to change, and save
what the model sees (PLAN.md §4: "verified by pushing frames through the judge's exact preprocessing").

  python scripts/check_preprocess.py <image.png> [...] --out <dir>   (judge venv)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from condex.preprocess import check_processor  # noqa: E402
from condex.render import load_style  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("images", nargs="+")
    ap.add_argument("--model", default="Qwen/Qwen3-VL-8B-Instruct")
    ap.add_argument("--style", default="configs/style/default.yaml")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    style = load_style(args.style)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rep = []
    for p in args.images:
        r = check_processor(args.model, np.asarray(Image.open(p).convert("RGB")), style.judge)
        Image.fromarray(r.pop("reconstructed")).save(out / f"model_view_{Path(p).stem}.png")
        rep.append({"image": p, **r})
    ok = all(r["same_size"] and r["max_abs_diff"] is not None and r["max_abs_diff"] <= 2.0 for r in rep)
    (out / "check_preprocess.json").write_text(json.dumps({"ok": ok, "style_hash": style.hash(), "images": rep}, indent=1))
    print(json.dumps({"ok": ok, "images": rep}, indent=1))


if __name__ == "__main__":
    main()
