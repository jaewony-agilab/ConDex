"""Phase 3 inputs: labels.jsonl -> per-condition JSONL + images (same boundaries, same splits).

  <out>/<condition>/{train,val,heldout}.jsonl, train_oversampled.jsonl, images/, dataset.json
dataset.json names the style config and its hash (PLAN.md §7).
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from condex import config as C  # noqa: E402
from condex.render import load_style  # noqa: E402
from condex.train.dataset import build_rows, oversample, read_jsonl, write_jsonl  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("labels", help="labels.jsonl")
    ap.add_argument("--out", required=True)
    ap.add_argument("--conditions", nargs="+", default=["plain", "text", "overlay"])
    ap.add_argument("--style", default=None)
    ap.add_argument("--config", default=None)
    ap.add_argument("--oversample-ratio", type=float, default=0.5)
    ap.add_argument("--oversample-max", type=int, default=5)
    args = ap.parse_args()
    cfg = C.load(args.config)
    style_path = args.style or str(C.REPO / cfg["judge"]["style"])
    style = load_style(style_path)
    labels = read_jsonl(Path(args.labels))
    for cond in args.conditions:
        out = Path(args.out) / cond
        rows = build_rows(labels, cond, style, cfg["judge"]["cameras"], out)
        for split in ("train", "val", "heldout"):
            write_jsonl(out / f"{split}.jsonl", [r for r in rows if r["split"] == split])
        tr = oversample([{**r, **r["label"]} for r in rows if r["split"] == "train"], args.oversample_ratio, args.oversample_max)
        write_jsonl(out / "train_oversampled.jsonl", [{k: v for k, v in r.items() if k not in ("outcome", "plan_still_valid", "reason")} for r in tr])
        meta = {"run": C.run_info(cfg["_path"]), "condition": cond, "style": style_path, "style_hash": style.hash(),
                "cameras": cfg["judge"]["cameras"], "labels": args.labels,
                "counts": {s: dict(Counter(f"{r['label']['outcome']}/{r['label']['plan_still_valid']}" for r in rows if r["split"] == s))
                           for s in ("train", "val", "heldout")},
                "train_oversampled": len(tr)}
        (out / "dataset.json").write_text(json.dumps(meta, indent=1))
        print(json.dumps(meta, indent=1))


if __name__ == "__main__":
    main()
