"""Events and Phase 2 labels on synthetic privileged facts."""

from conftest import make_episode

from condex import episode as E
from condex.privileged.events import derive_events
from condex.privileged.labels import LabelConfig, apply_behavioral, label_episode, split_of

CFG = LabelConfig(lift_horizon=20, hold_steps=10, release_horizon=15, still_steps=5)


def facts_fn(grasp=(), z=lambda s: 0.8, inr=(), success=None, other=(), base=0.5, touching=()):
    """Builds per-step fact rows. grasp/inr/other/touching: step ranges (sets of steps)."""
    def f(s):
        rows = [{"step": s, "kind": "object_pose", "name": "obj", "value": [0.1, 0.0, z(s), 1, 0, 0, 0]},
                {"step": s, "kind": "fact", "name": "grasped", "hand": "right", "value": [float(s in grasp)]},
                {"step": s, "kind": "fact", "name": "grasped_other", "hand": "right", "value": [float(s in other)]},
                {"step": s, "kind": "fact", "name": "in_receptacle", "value": [float(s in inr)]},
                {"step": s, "kind": "fact", "name": "success", "value": [float(success is not None and s >= success)]},
                {"step": s, "kind": "fact", "name": "base_dist", "value": [base]},
                {"step": s, "kind": "fact", "name": "reach_dist", "hand": "right", "value": [0.02 if s >= 18 else 0.3]}]
        if s in touching:
            rows.append({"step": s, "kind": "contact", "hand": "right", "a": "R_index_intermediate_link", "b": "obj"})
        return rows
    return f


def grasp_at(settled):
    return [{"step": settled - 5, "hand": "right", "type": "grasp", "settled_step": settled, "subtask": "", "retry": 0}]


def release_at(settled):
    return [{"step": settled - 5, "hand": "right", "type": "release", "settled_step": settled, "subtask": "", "retry": 0}]


def labels_for(tmp_path, n, facts, bounds, forces=None):
    ep = make_episode(tmp_path / "e", n_steps=n, facts=facts, boundaries=bounds, forces_at=forces)
    return label_episode(E.EpisodeReader(ep), CFG)


def test_grasp_success(tmp_path):
    lab = labels_for(tmp_path, 80, facts_fn(grasp=range(20, 80), z=lambda s: 0.8 + 0.005 * max(0, s - 22)), grasp_at(20))[0]
    assert (lab["outcome"], lab["plan_still_valid"]) == ("success", "yes")


def test_grasp_slip_names_fingers(tmp_path):
    lab = labels_for(tmp_path, 80, facts_fn(grasp=range(20, 32), z=lambda s: 0.8 + 0.005 * max(0, s - 22)), grasp_at(20),
                     forces=lambda s: {"thumb": [0, 2.0, 0], "index": [1.0, 0, 0]} if s < 32 else {})[0]
    assert lab["outcome"] == "fail" and "slipped" in lab["reason"] and "thumb and index" in lab["reason"]


def test_grasp_empty_and_on_object(tmp_path):
    lab = labels_for(tmp_path, 80, facts_fn(), grasp_at(20))[0]
    assert lab["outcome"] == "fail" and lab["reason"] == "hand closed with nothing between the fingertips"
    lab = labels_for(tmp_path / "b", 80, facts_fn(touching=range(15, 30)), grasp_at(20))[0]
    assert lab["outcome"] == "fail" and "did not secure" in lab["reason"]


def test_grasp_wrong_object_invalid(tmp_path):
    lab = labels_for(tmp_path, 80, facts_fn(other=range(20, 80)), grasp_at(20))[0]
    assert (lab["outcome"], lab["plan_still_valid"]) == ("fail", "no")


def test_grasp_unclear_when_truncated(tmp_path):
    lab = labels_for(tmp_path, 30, facts_fn(grasp=range(20, 30)), grasp_at(20))[0]
    assert lab["outcome"] == "unclear"


def test_release_success_drop_stuck(tmp_path):
    ok = labels_for(tmp_path / "a", 80, facts_fn(grasp=range(0, 20), inr=range(22, 80)), release_at(20))[0]
    assert ok["outcome"] == "success"
    drop = labels_for(tmp_path / "b", 80, facts_fn(grasp=range(0, 20)), release_at(20))[0]
    assert drop["outcome"] == "fail" and "seated" in drop["reason"]
    stuck = labels_for(tmp_path / "c", 80, facts_fn(grasp=range(0, 80)), release_at(20))[0]
    assert (stuck["outcome"], stuck["plan_still_valid"]) == ("fail", "no")


def test_release_success_when_episode_stops_at_task_success(tmp_path):
    lab = labels_for(tmp_path, 25, facts_fn(grasp=range(0, 20), inr=range(22, 25), success=24), release_at(20))[0]
    assert lab["outcome"] == "success"


def test_out_of_reach_invalidates(tmp_path):
    lab = labels_for(tmp_path, 80, facts_fn(base=1.2), grasp_at(20))[0]
    assert (lab["outcome"], lab["plan_still_valid"]) == ("fail", "no") and "out of reach" in lab["reason"]


def test_derive_events():
    rows = [r for s in range(80) for r in facts_fn(grasp=range(20, 40), z=lambda s: 0.8 + 0.01 * max(0, s - 22),
                                                    inr=range(60, 80), success=70)(s)]
    ev = {(e["name"], e.get("hand")): e["step"] for e in derive_events(rows, grasp_at(20))}
    assert ev[("reach", "right")] == 18 and ev[("first_close", "right")] == 15 and ev[("grasp_ok", "right")] == 20
    assert ev[("lift", "right")] == 25 and ev[("drop", "right")] == 40  # lost without a release boundary
    assert ev[("place", None)] == 60 and ev[("success", None)] == 70


def test_split_and_behavioral():
    assert split_of("T", 1) == split_of("T", 1)
    s = [split_of("T", i) for i in range(2000)]
    assert 0.6 < s.count("train") / 2000 < 0.8 and s.count("heldout") > 150
    rows = [{"episode": "e", "boundary_index": 0, "outcome": "fail", "plan_still_valid": "yes"}]
    assert apply_behavioral(rows, {("e", 0): False})[0]["plan_still_valid"] == "no"
