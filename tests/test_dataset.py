"""Judge datasets: rows per condition, oversampling, label masking, scoring."""

from collections import Counter

import numpy as np

from condex import render as R
from condex.judge.schema import Verdict
from condex.metrics import score
from condex.train.dataset import build_rows, oversample, packet_from_row
from condex.train.lora import IGNORE, chat_messages, mask_labels


def labels_for(ep):
    base = {"episode": str(ep), "task": "SyntheticTask", "split": "train", "subtask": "", "retry": 0}
    return [{**base, "boundary_index": 0, "type": "grasp", "outcome": "success", "plan_still_valid": "yes", "reason": "held"},
            {**base, "boundary_index": 1, "type": "release", "outcome": "fail", "plan_still_valid": "no", "reason": "stuck"}]


def test_rows_per_condition(episode, tmp_path):
    out = tmp_path / "ds"
    rows = {c: build_rows(labels_for(episode), c, R.Style(), ["ego_view"], out) for c in ("plain", "text", "overlay")}
    for c, rs in rows.items():
        assert [r["id"].split("/")[1] for r in rs] == ["b000", "b001"]
        assert all((out / x["path"]).exists() for r in rs for x in r["content"] if x["type"] == "image")
        assert rs[1]["target"] == Verdict("fail", "no", "stuck").to_json()
    # teacher-forced previous verdict and same text skeleton apart from the condition extras
    assert "verdict outcome=success" in " ".join(x.get("text", "") for x in rows["plain"][1]["content"])
    assert any("Overlay legend" in x.get("text", "") for x in rows["overlay"][1]["content"])
    pk = packet_from_row(rows["overlay"][1], out)
    assert len(pk.images()) == 2 and pk.meta["condition"] == "overlay"
    msgs = chat_messages(rows["overlay"][1], with_target=True)
    assert msgs[-1]["role"] == "assistant" and sum(c["type"] == "image" for c in msgs[1]["content"]) == 2


def test_oversample_balances_minority():
    rows = [{"outcome": "success", "plan_still_valid": "yes"}] * 100 + [{"outcome": "fail", "plan_still_valid": "no"}] * 5
    c = Counter(r["outcome"] for r in oversample(rows, target_ratio=0.5, max_factor=5))
    assert c["success"] == 100 and c["fail"] == 25


def test_mask_labels():
    assert mask_labels([1, 2, 3, 4, 5], 3) == [IGNORE, IGNORE, IGNORE, 4, 5]


def test_score():
    lab = lambda o, v="yes": {"outcome": o, "plan_still_valid": v}  # noqa: E731
    rows = [{"label": lab("fail", "no"), "pred": lab("fail", "no"), "type": "grasp"},
            {"label": lab("fail", "yes"), "pred": lab("success"), "type": "grasp"},
            {"label": lab("success"), "pred": lab("success"), "type": "release"}]
    s = score(rows)
    assert np.isclose(s["outcome_acc"], 2 / 3) and s["fail_recall"] == 0.5 and s["valid_acc_on_fail"] == 1.0
    assert s["outcome_confusion"]["fail"]["success"] == 1
