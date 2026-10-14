"""Judge scoring: per-field accuracy and confusion matrices (PLAN.md Phase 3 step 3)."""

from __future__ import annotations

from collections import Counter

from condex.judge.schema import OUTCOMES, VALID


def confusion(truth: list[str], pred: list[str], classes) -> dict[str, dict[str, int]]:
    c = Counter(zip(truth, pred))
    return {t: {p: c.get((t, p), 0) for p in classes} for t in classes}


def score(rows: list[dict]) -> dict:
    """rows: {"label": {...}, "pred": {...}, "type": grasp|release}. plan_still_valid is scored on
    rows whose true outcome is fail (that is where the gate reads it)."""
    out: dict = {"n": len(rows)}
    if not rows:
        return out
    t_o = [r["label"]["outcome"] for r in rows]
    p_o = [r["pred"]["outcome"] for r in rows]
    out["outcome_acc"] = sum(a == b for a, b in zip(t_o, p_o)) / len(rows)
    out["outcome_confusion"] = confusion(t_o, p_o, OUTCOMES)
    fails = [r for r in rows if r["label"]["outcome"] == "fail"]
    if fails:
        t_v = [r["label"]["plan_still_valid"] for r in fails]
        p_v = [r["pred"]["plan_still_valid"] for r in fails]
        out["valid_acc_on_fail"] = sum(a == b for a, b in zip(t_v, p_v)) / len(fails)
        out["valid_confusion_on_fail"] = confusion(t_v, p_v, VALID)
    # what the gate acts on: success vs not, as a detector of failures
    tp = sum(1 for a, b in zip(t_o, p_o) if a == "fail" and b == "fail")
    out["fail_recall"] = tp / max(1, t_o.count("fail"))
    out["fail_precision"] = tp / max(1, p_o.count("fail"))
    out["parse_fail_rate"] = sum(1 for r in rows if not r["pred"].get("parse_ok", True)) / len(rows)
    for btype in ("grasp", "release"):
        sub = [r for r in rows if r.get("type") == btype]
        if sub:
            out[f"outcome_acc_{btype}"] = sum(r["label"]["outcome"] == r["pred"]["outcome"] for r in sub) / len(sub)
    return out
