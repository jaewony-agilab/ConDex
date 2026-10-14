"""Phase 3: score a judge on a dataset split (frozen Qwen3-VL / LoRA via vLLM, or Claude Haiku 5.5).

  python scripts/eval_judge.py <dataset>/<condition> --split heldout --judge vlm --vlm-model <name> --out <dir>
Packets are the dataset's (previous verdict = previous label, teacher-forced). Writes
predictions.jsonl and metrics.json (accuracy per field, confusion matrices).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from condex import config as C  # noqa: E402
from condex.metrics import score  # noqa: E402
from condex.train.dataset import packet_from_row, read_jsonl  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset", help="<dataset>/<condition> directory")
    ap.add_argument("--split", default="heldout")
    ap.add_argument("--judge", choices=["vlm", "claude"], default="vlm")
    ap.add_argument("--vlm-model", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--config", default=None)
    args = ap.parse_args()
    cfg = C.load(args.config)
    root = Path(args.dataset)
    rows = read_jsonl(root / f"{args.split}.jsonl")[: args.limit]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if args.judge == "vlm":
        from condex.judge.backends import VLMJudge

        judge = VLMJudge(os.environ["CONDEX_VLM_URL"], args.vlm_model or cfg["judge"]["vlm_model"])
    else:
        from condex.cost import Ledger
        from condex.judge.backends import ClaudeJudge

        judge = ClaudeJudge(Ledger(calls_path=out / "calls.jsonl"), cfg["judge"]["claude_model"], cfg["judge"]["claude_effort"])

    def one(r):
        v = judge(packet_from_row(r, root))
        return {"id": r["id"], "type": r["type"], "task": r.get("task"), "label": r["label"], "pred": v.as_dict()}

    with ThreadPoolExecutor(args.workers) as ex:
        preds = list(ex.map(one, rows))
    with (out / "predictions.jsonl").open("w") as f:
        for p in preds:
            f.write(json.dumps(p) + "\n")
    meta = json.loads((root / "dataset.json").read_text()) if (root / "dataset.json").exists() else {}
    res = {"run": C.run_info(cfg["_path"]), "judge": judge.name, "condition": meta.get("condition"), "split": args.split,
           "style_hash": meta.get("style_hash"), "dataset": str(root), "metrics": score(preds)}
    if args.judge == "claude":
        res["cost"] = judge.ledger.summary()
    (out / "metrics.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
