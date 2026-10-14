"""Boundary labels from the recorder output (PLAN.md Phase 2). Offline, pure.

outcome
  grasp:   success iff the hand holds the target, lifts it by `lift_m` within `lift_horizon` steps of
           the settled step, and keeps holding it for `hold_steps` after the lift (or until its own
           release boundary / task success, whichever comes first)
  release: success iff within `release_horizon` steps the hand no longer holds the target and the
           target is in the receptacle and stationary for `still_steps`
  unclear: the episode ends before the window needed to decide
plan_still_valid (state-based, v1)
  no  if the hand holds a different object, the target is out of reach (beyond `reach_limit_m` of
      the robot base, or fallen below the table), or the target is stuck to the hand after release
  yes otherwise
reason: one templated sentence from the facts (and the non-privileged taxels, to name fingers)
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np

from condex.episode import EpisodeData
from condex.privileged.events import Facts


@dataclass(frozen=True)
class LabelConfig:
    lift_m: float = 0.03
    lift_horizon: int = 60  # steps (20 Hz -> 3 s)
    hold_steps: int = 30
    release_horizon: int = 40
    still_steps: int = 10
    still_speed_mps: float = 0.02
    reach_limit_m: float = 0.75  # CALIBRATE: horizontal base -> object distance the arms can reach
    fall_drop_m: float = 0.10  # object this far below its start height = off the table
    contact_threshold_n: float = 0.5
    hz: float = 20.0


def _finger_list(fs: list[str]) -> str:
    fs = [f for f in fs if f != "palm"]
    if not fs:
        return "the hand"
    return fs[0] if len(fs) == 1 else ", ".join(fs[:-1]) + " and " + fs[-1]


def _fingers_in_contact(data: EpisodeData, hand: str, step: int, thr: float) -> list[str]:
    tot: dict[str, np.ndarray] = {}
    for r in data.taxel_rows(step):
        if r["hand"] == hand:
            tot[r["finger"]] = tot.get(r["finger"], 0) + np.asarray(r["force_world"], float)
    return [f for f, v in tot.items() if np.linalg.norm(v) >= thr]


def label_boundary(data: EpisodeData, b: dict, facts: Facts, cfg: LabelConfig = LabelConfig()) -> dict:
    h, s, n = b["hand"], int(b["settled_step"]), facts.n
    g = facts.flag("grasped", h)
    other = facts.flag("grasped_other", h)
    pos = facts.obj_pos()
    z = pos[:, 2]
    z_start = z[0]
    success_at = np.flatnonzero(facts.flag("success"))
    first_success = int(success_at[0]) if success_at.size else None
    base_dist = facts.get("base_dist")
    s = min(s, n - 1)

    out_of_reach = bool((not np.isnan(base_dist[s]) and base_dist[s] > cfg.reach_limit_m)
                        or (not np.isnan(z[s]) and z[s] < z_start - cfg.fall_drop_m))
    touching = facts.touching(h, s)
    fingers_now = _fingers_in_contact(data, h, s, cfg.contact_threshold_n)
    valid, outcome, reason = "yes", None, ""

    if b["type"] == "grasp":
        if other[s]:
            outcome, valid, reason = "fail", "no", "hand closed on the wrong object"
        else:
            end = min(s + cfg.lift_horizon, n - 1)
            lifted = [t for t in range(s, end + 1) if g[t] and not np.isnan(z[s]) and z[t] - z[s] >= cfg.lift_m]
            if lifted:
                t0 = lifted[0]
                stop = t0 + cfg.hold_steps
                own_release = [int(x["step"]) for x in data.boundaries if x["hand"] == h and x["type"] == "release" and int(x["step"]) > t0]
                stop = min([stop] + own_release + ([first_success] if first_success is not None and first_success > t0 else []))
                lost = [t for t in range(t0, min(stop, n)) if not g[t]]
                if lost:
                    before = _fingers_in_contact(data, h, max(lost[0] - 1, 0), cfg.contact_threshold_n)
                    outcome, reason = "fail", f"object slipped from the {_finger_list(before)} after lift"
                elif stop >= n:
                    outcome, reason = "unclear", "episode ended before the hold could be confirmed"
                else:
                    outcome, reason = "success", "object lifted and held"
            elif g[s] and end < s + cfg.lift_horizon:
                outcome, reason = "unclear", "episode ended before the outcome could be seen"
            elif g[s]:
                outcome, reason = "fail", "object held but not lifted"
            elif touching or fingers_now:
                outcome, reason = "fail", f"hand closed on the object with the {_finger_list(fingers_now)} but did not secure it"
            else:
                outcome, reason = "fail", "hand closed with nothing between the fingertips"
    else:
        end = s + cfg.release_horizon
        inr = facts.flag("in_receptacle")
        vel = np.r_[0.0, np.linalg.norm(np.diff(pos, axis=0), axis=1) * cfg.hz]
        seated = None
        for t in range(s, min(end, n - cfg.still_steps) + 1):  # only full stillness windows
            if not g[t] and inr[t] and np.nanmax(vel[t:t + cfg.still_steps]) < cfg.still_speed_mps:
                seated = t
                break
        if seated is not None or (first_success is not None and s <= first_success <= end):
            outcome, reason = "success", "object placed in the target and released"
        elif end + cfg.still_steps > n:
            outcome, reason = "unclear", "episode ended before the outcome could be seen"
        else:
            if g[end]:
                outcome, valid, reason = "fail", "no", "object stuck to the hand after opening"
            elif inr[end]:
                outcome, reason = "fail", "object released but did not come to rest in the target"
            else:
                outcome, reason = "fail", "released before the object was seated"
    if out_of_reach:
        valid = "no"
        if outcome == "fail":
            reason += "; the object is now out of reach"
    if outcome == "success":
        valid = "yes"
    return {"outcome": outcome, "plan_still_valid": valid, "reason": reason, "out_of_reach": out_of_reach,
            "grasped_at_boundary": bool(g[s]), "touching_links": touching}


def label_episode(data: EpisodeData, cfg: LabelConfig = LabelConfig()) -> list[dict]:
    facts = Facts(data.events, n_steps=(max(data.steps()) + 1) if data.proprio else None)
    rows = []
    for i, b in enumerate(data.boundaries):
        lab = label_boundary(data, b, facts, cfg)
        rows.append({"episode": str(data.root), "task": data.meta.get("task"), "seed": data.meta.get("seed"),
                     "episode_id": data.meta.get("episode_id"), "boundary_index": i, "hand": b["hand"], "type": b["type"],
                     "step": int(b["step"]), "settled_step": int(b["settled_step"]), "subtask": b.get("subtask", ""),
                     "retry": int(b.get("retry", 0)), **lab})
    return rows


def split_of(task: str, episode_seed: int, fractions=(0.7, 0.15)) -> str:
    """train / val / heldout by a hash of (task, episode seed): stable, condition-independent."""
    u = int(hashlib.sha256(f"{task}:{episode_seed}".encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
    return "train" if u < fractions[0] else ("val" if u < fractions[0] + fractions[1] else "heldout")


def apply_behavioral(labels: list[dict], conversions: dict[tuple[str, int], bool]) -> list[dict]:
    """v2 plan_still_valid: (episode, boundary_index) -> did scripted recovery convert this failure."""
    out = []
    for r in labels:
        k = (r["episode"], r["boundary_index"])
        r = dict(r)
        if r["outcome"] == "fail" and k in conversions:
            r["plan_still_valid_v1"] = r["plan_still_valid"]
            r["plan_still_valid"] = "yes" if conversions[k] else "no"
        out.append(r)
    return out
