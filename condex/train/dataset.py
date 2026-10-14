"""Judge datasets (PLAN.md Phase 2-3): labels + episode records -> rendered images + JSONL.

One JSONL row per boundary:
  {"id", "split", "condition", "style_hash", "episode", "boundary_index", "type",
   "system", "content": [{"type": "text", "text"} | {"type": "image", "path"}],
   "target": "<verdict JSON>", "label": {outcome, plan_still_valid, reason}}
The previous verdict shown in the packet is the previous boundary's label (teacher forcing).
The training split can be oversampled toward minority (outcome, plan_still_valid) classes.
"""

from __future__ import annotations

import json
import random
from collections import Counter
from pathlib import Path

from PIL import Image

from condex.episode import load
from condex.judge.prompt import build_packet
from condex.judge.schema import Verdict
from condex.render import Style


def class_key(r: dict) -> str:
    return f"{r['outcome']}/{r['plan_still_valid']}"


def oversample(rows: list[dict], target_ratio: float = 0.5, max_factor: int = 5, seed: int = 0) -> list[dict]:
    """Repeat rows of each class until it has >= target_ratio x the largest class (at most max_factor x)."""
    counts = Counter(class_key(r) for r in rows)
    if not counts:
        return rows
    top = max(counts.values())
    out = list(rows)
    rng = random.Random(seed)
    for k, n in counts.items():
        want = min(int(target_ratio * top), n * max_factor)
        pool = [r for r in rows if class_key(r) == k]
        out += [rng.choice(pool) for _ in range(max(0, want - n))]
    rng.shuffle(out)
    return out


def build_rows(labels: list[dict], condition: str, style: Style, cameras: list[str], out_dir: Path,
               use_prev_verdict: bool = True) -> list[dict]:
    """labels: rows from condex.privileged.labels.label_episode (with 'split')."""
    img_dir = out_dir / "images" / condition
    img_dir.mkdir(parents=True, exist_ok=True)
    by_ep: dict[str, list[dict]] = {}
    for r in labels:
        by_ep.setdefault(r["episode"], []).append(r)
    rows = []
    for ep, labs in by_ep.items():
        data = load(ep)
        labs = sorted(labs, key=lambda r: r["boundary_index"])
        lab_of = {r["boundary_index"]: r for r in labs}
        ep_tag = Path(ep).name
        for r in labs:
            i = r["boundary_index"]
            prev = lab_of.get(i - 1)
            pv = Verdict(prev["outcome"], prev["plan_still_valid"], prev["reason"], "label") if (prev and use_prev_verdict) else None

            def sink(name, img, _tag=f"{ep_tag}_b{i:03d}"):
                p = img_dir / f"{_tag}_{name}.png"
                if not p.exists():
                    Image.fromarray(img).save(p)
                return str(p.relative_to(out_dir))

            pk = build_packet(data, i, condition, style, cameras, pv, sink=sink)
            content = [{"type": "text", "text": it.text} if it.kind == "text" else {"type": "image", "path": it.path}
                       for it in pk.content]
            target = Verdict(r["outcome"], r["plan_still_valid"], r["reason"]).to_json()
            rows.append({"id": f"{ep_tag}/b{i:03d}", "split": r["split"], "condition": condition, "style_hash": style.hash(),
                         "episode": ep, "boundary_index": i, "type": r["type"], "task": r.get("task"),
                         "system": pk.system, "content": content, "target": target,
                         "label": {k: r[k] for k in ("outcome", "plan_still_valid", "reason")}})
    return rows


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def packet_from_row(row: dict, root: Path):
    """JSONL row -> Packet with images loaded (offline judge evaluation)."""
    import numpy as np

    from condex.judge.prompt import Item, Packet

    items = [Item("text", c["text"]) if c["type"] == "text"
             else Item("image", image=np.asarray(Image.open(Path(root) / c["path"]).convert("RGB")), path=c["path"])
             for c in row["content"]]
    meta = {k: row[k] for k in ("episode", "boundary_index", "condition", "style_hash", "type")}
    return Packet(row["system"], items, meta)
