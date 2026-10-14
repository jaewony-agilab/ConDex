"""Privileged oracle judge (condition X, the ceiling). Reads the recorder's facts at the boundary.

grasp:   success iff the hand holds the target now
release: success iff the hand no longer holds it and it is in the receptacle (or the task succeeded)
plan_still_valid: no if the hand holds another object or the target is out of reach
"""

from __future__ import annotations

from condex.judge.prompt import Packet
from condex.judge.schema import Verdict
from condex.privileged.recorder import Recorder


class OracleJudge:
    name = "oracle"

    def __init__(self, recorder: Recorder, reach_limit_m: float = 0.75):
        self.recorder, self.reach_limit_m = recorder, reach_limit_m

    def __call__(self, packet: Packet) -> Verdict:
        f = self.recorder.last
        hand, btype = packet.meta["hand"], packet.meta["type"]
        held = f.get(f"grasped.{hand}", 0.0) > 0.5
        other = f.get(f"grasped_other.{hand}", 0.0) > 0.5
        far = f.get("base_dist", 0.0) > self.reach_limit_m
        if btype == "grasp":
            ok = held
            reason = "oracle: target held" if ok else ("oracle: wrong object held" if other else "oracle: target not held")
        else:
            ok = (not held) and (f.get("in_receptacle", 0.0) > 0.5 or f.get("success", 0.0) > 0.5)
            reason = "oracle: target placed" if ok else ("oracle: target stuck to the hand" if held else "oracle: target not in the receptacle")
        valid = "no" if (other or far or (btype == "release" and held)) and not ok else "yes"
        return Verdict("success" if ok else "fail", valid, reason, self.name)
