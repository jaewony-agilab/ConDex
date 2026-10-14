"""Phase 3 step 2: LoRA fine-tune of Qwen3-VL-8B on one condition's dataset (1 A100).

  python scripts/train_judge.py <dataset>/<condition> --seed 0 --out <adapter dir>
Trains on train_oversampled.jsonl, validates on val.jsonl; writes adapters (best/, epoch*/) and
train_summary.json (with job id, commit, dataset style hash).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from condex import config as C  # noqa: E402
from condex.train.dataset import read_jsonl  # noqa: E402
from condex.train.lora import TrainConfig, train  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset")
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--grad-accum", type=int, default=16)
    ap.add_argument("--val-limit", type=int, default=300)
    args = ap.parse_args()
    root = Path(args.dataset)
    cfg = TrainConfig(seed=args.seed, epochs=args.epochs, lr=args.lr, rank=args.rank, alpha=2 * args.rank, grad_accum=args.grad_accum)
    tr = read_jsonl(root / "train_oversampled.jsonl")
    va = read_jsonl(root / "val.jsonl")[: args.val_limit]
    out = Path(args.out)
    s = train(tr, va, root, out, cfg)
    meta = json.loads((root / "dataset.json").read_text())
    s.update({"run": C.run_info(None), "dataset": str(root), "condition": meta["condition"], "style_hash": meta["style_hash"]})
    (out / "train_summary.json").write_text(json.dumps(s, indent=1))
    print(json.dumps({k: v for k, v in s.items() if k != "history"}, indent=1))


if __name__ == "__main__":
    main()
