"""Boundary detector, gate, recovery, cost caps, judge packets and parsing."""

import json

import numpy as np
import pytest

from condex import episode as E
from condex import render as R
from condex.boundary import ClosureConfig, HandBoundaryDetector
from condex.cost import BudgetExceeded, Ledger
from condex.gate import Action, Gate
from condex.judge.backends import ClaudeJudge, VLMJudge
from condex.judge.prompt import build_packet
from condex.judge.schema import Verdict, parse
from condex.recovery import RecoveryConfig, recovery_actions

CL = ClosureConfig(flexion_dims=(1, 2, 3, 4), open_value=0.0, closed_value=1.0, settle_steps=3, max_settle_steps=20)


def cmd(c):
    return np.array([0.0, c, c, c, c, 0.0])


def test_boundary_waits_for_settle_and_hysteresis():
    d = HandBoundaryDetector("right", CL)
    q = np.zeros(6)
    out = []
    for s in range(60):
        c = 1.0 if 10 <= s < 40 else 0.0
        if 10 <= s < 18 or 40 <= s < 45:
            q = q + 0.05 * (1 if s < 40 else -1)  # fingers moving
        b = d.update(s, cmd(c), q)
        if b:
            out.append(b)
    assert [b.type for b in out] == ["grasp", "release"]
    assert out[0].step == 10 and out[0].settled_step >= 18 + 2
    assert out[1].step == 40 and out[1].settled_step >= 45 + 2
    d2 = HandBoundaryDetector("right", CL)  # hovering between thresholds never re-triggers
    seq = [d2.update(s, cmd(v), np.zeros(6)) for s, v in enumerate([0.7, 0.5, 0.65, 0.5, 0.62] + [0.62] * 10)]
    assert sum(b is not None for b in seq) == 1


def test_gate_table():
    g = Gate(max_retries=2)
    F = lambda v="yes": Verdict("fail", v, "slip")  # noqa: E731
    assert g.decide("right", Verdict("success", "yes", ""))[0] == Action.CONTINUE
    assert g.decide("right", F())[0] == Action.RECOVER
    assert g.decide("right", F())[0] == Action.RECOVER
    assert g.decide("right", F())[0] == Action.REPLAN  # cap hit
    assert g.decide("left", F("no"))[0] == Action.REPLAN
    assert g.decide("left", Verdict("unclear", "yes", ""))[0] == Action.CONTINUE
    assert g.decide("left", Verdict("unclear", "yes", ""))[0] == Action.REPLAN
    assert g.decide("right", Verdict("success", "yes", ""))[0] == Action.CONTINUE
    assert g.decide("right", F())[0] == Action.RECOVER  # success reset the hand's retries
    assert Gate(recovery_enabled=False).decide("right", F())[0] == Action.ABORT


def test_recovery_opens_then_retraces():
    hist = [{"action.right_hand": cmd(1.0), "action.right_arm": np.full(7, i, float), "action.left_arm": np.zeros(7)} for i in range(30)]
    acts = recovery_actions(hist, "right", "action.right_hand", "action.right_arm", CL, RecoveryConfig(5, 10, 2))
    assert len(acts) == 5 + 10 + 2
    assert np.allclose(acts[4]["action.right_hand"], cmd(0.0)) and acts[2]["action.right_hand"][1] == pytest.approx(0.4)
    assert [a["action.right_arm"][0] for a in acts[5:15]] == list(range(29, 19, -1))
    assert all(np.allclose(a["action.left_arm"], 0) for a in acts)


def test_parse():
    v = parse('blah {"outcome": "fail", "plan_still_valid": "no", "reason": "dropped"} trailing')
    assert (v.outcome, v.plan_still_valid, v.parse_ok) == ("fail", "no", True)
    bad = parse('{"outcome": "maybe"}')
    assert bad.outcome == "unclear" and not bad.parse_ok
    assert json.loads(Verdict("success", "yes", "held").to_json()) == {"outcome": "success", "plan_still_valid": "yes", "reason": "held"}


def test_cost_caps(tmp_path):
    caps = tmp_path / "caps.yaml"
    caps.write_text(f"run_cap_usd: 0.001\nglobal_cap_usd: 1.0\nglobal_ledger: {tmp_path}/ledger.jsonl\n"
                    "prices: {claude-haiku-5-5: {input_per_mtok: 0.10, output_per_mtok: 0.50}}\n")
    led = Ledger(caps, tmp_path / "calls.jsonl")
    led.ensure()
    assert led.charge("judge", "claude-haiku-5-5", 5000, 1000) == pytest.approx(0.001)
    with pytest.raises(BudgetExceeded):
        led.ensure()
    assert led.global_spent() == pytest.approx(0.001) and led.summary()["by_purpose"]["judge"] == pytest.approx(0.001)
    with pytest.raises(FileNotFoundError):
        Ledger(tmp_path / "missing.yaml")


def test_packets_per_condition(episode):
    d = E.load(str(episode))
    style = R.Style()
    prev = Verdict("success", "yes", "held")
    plain = build_packet(d, 1, "plain", style, ["ego_view"], prev)
    over = build_packet(d, 1, "overlay", style, ["ego_view"], prev)
    text = build_packet(d, 1, "text", style, ["ego_view"], prev)
    assert len(plain.images()) == 2  # previous + current
    assert "Overlay legend" in over.text() and "Overlay legend" not in plain.text()
    assert "index 5.0 N pushing right" in text.text() and "thumb 3.0 N pushing up" in text.text()
    assert not np.array_equal(plain.images()[1], over.images()[1])
    assert "verdict outcome=success" in plain.text()
    first = build_packet(d, 0, "plain", style, ["ego_view"])
    assert len(first.images()) == 1 and "Previous check: none" in first.text()
    no_prev = build_packet(d, 1, "plain", style, ["ego_view"], prev, prev_frames=False)
    assert len(no_prev.images()) == 1
    assert all(im.shape[0] % 32 == 0 for im in over.images())
    # backend message conversion keeps order and image count
    m = VLMJudge.messages(over)
    assert sum(c["type"] == "image_url" for c in m[1]["content"]) == 2
    assert sum(c["type"] == "image" for c in ClaudeJudge.content(over)) == 2
