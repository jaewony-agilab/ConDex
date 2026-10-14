"""Privileged recorder: per-step object state, grasp facts and hand-object contact pairs into the
episode's events (format: condex.privileged.events). Imported only by scripts/ and the oracle;
the executor, judge and renderer never see it (tests/test_isolation.py).

Reads robocasa/robosuite internals (cluster only): env.rs.objects, obj_body_id, _check_grasp,
_check_success, robocasa.utils.object_utils.check_obj_in_receptacle.
"""

from __future__ import annotations

import mujoco
import numpy as np

from condex.episode import EpisodeWriter
from condex.privileged.events import TARGET, EventConfig, derive_events
from condex.sim.taxels import TaxelSensor


class Recorder:
    def __init__(self, env, sensor: TaxelSensor, writer: EpisodeWriter, grippers: dict[str, str] | None = None,
                 event_cfg: EventConfig = EventConfig()):
        """grippers: hand -> robosuite gripper arm key (default: the hand name itself)."""
        self.env, self.sensor, self.writer, self.event_cfg = env, sensor, writer, event_cfg
        self.grippers = grippers or {}
        self.last: dict = {}
        self.bind()

    def bind(self) -> None:
        """After every reset: resolve object bodies and the hand subtrees."""
        rs, m = self.env.rs, self.env.model
        self.objects = dict(rs.objects)
        root_of_obj = {}
        for name, obj in self.objects.items():
            bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, obj.root_body)
            if bid >= 0:
                root_of_obj[bid] = name
        self.body_obj = {}
        for b in range(m.nbody):  # walk up to an object root
            x = b
            while x > 0 and x not in root_of_obj:
                x = int(m.body_parentid[x])
            if x in root_of_obj:
                self.body_obj[b] = root_of_obj[x]
        self.hand_bodies: dict[int, tuple[str, str]] = {}
        for (hand, finger), palm in ((k, b) for k, b in self.sensor.body_of.items() if k[1] == "palm"):
            for b in range(m.nbody):
                x = b
                while x > 0 and x != palm:
                    x = int(m.body_parentid[x])
                if x == palm:
                    self.hand_bodies[b] = (hand, self.sensor.body_names[b])
        try:
            self.base_body = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, rs.robots[0].robot_model.root_body)
        except Exception:  # noqa: BLE001 - optional fact
            self.base_body = -1

    def _grasped(self, hand: str, obj) -> bool:
        rs = self.env.rs
        gr = rs.robots[0].gripper
        g = gr[self.grippers.get(hand, hand)] if isinstance(gr, dict) else gr
        return bool(rs._check_grasp(gripper=g, object_geoms=obj))

    def record(self, step: int) -> None:
        rs, d = self.env.rs, self.env.data
        rows = []
        for name, obj in self.objects.items():
            bid = rs.obj_body_id.get(name) if hasattr(rs, "obj_body_id") else None
            if bid is None:
                continue
            rows.append({"step": step, "kind": "object_pose", "name": name,
                         "value": np.r_[d.xpos[bid], d.xquat[bid]].round(5).tolist()})
        target = self.objects.get(TARGET)
        obj_pos = np.asarray(d.xpos[rs.obj_body_id[TARGET]]) if target is not None else None
        for hand in self.sensor.links:
            if target is not None:
                g = self._grasped(hand, target)
                other = any(self._grasped(hand, o) for n, o in self.objects.items() if n != TARGET)
                tips = [np.asarray(d.xpos[b]) for (h, f), b in self.sensor.body_of.items() if h == hand and f != "palm"]
                rows += [{"step": step, "kind": "fact", "name": "grasped", "hand": hand, "value": [float(g)]},
                         {"step": step, "kind": "fact", "name": "grasped_other", "hand": hand, "value": [float(other)]},
                         {"step": step, "kind": "fact", "name": "reach_dist", "hand": hand,
                          "value": [float(np.linalg.norm(np.mean(tips, 0) - obj_pos))]}]
        if target is not None and "container" in self.objects:
            import robocasa.utils.object_utils as OU

            rows.append({"step": step, "kind": "fact", "name": "in_receptacle",
                         "value": [float(OU.check_obj_in_receptacle(rs, TARGET, "container"))]})
        rows.append({"step": step, "kind": "fact", "name": "success", "value": [float(rs._check_success())]})
        if target is not None and self.base_body >= 0:
            rows.append({"step": step, "kind": "fact", "name": "base_dist",
                         "value": [float(np.linalg.norm((obj_pos - d.xpos[self.base_body])[:2]))]})
        m = self.env.model
        seen = set()
        for i in range(d.ncon):
            c = d.contact[i]
            b1, b2 = int(m.geom_bodyid[c.geom1]), int(m.geom_bodyid[c.geom2])
            for hb, ob in ((b1, b2), (b2, b1)):
                if hb in self.hand_bodies and ob in self.body_obj:
                    hand, link = self.hand_bodies[hb]
                    key = (link, self.body_obj[ob])
                    if key not in seen:
                        seen.add(key)
                        rows.append({"step": step, "kind": "contact", "hand": hand, "a": link, "b": self.body_obj[ob]})
        self.writer.events.extend(rows)
        self.last = {r["name"] + (f".{r['hand']}" if r.get("hand") else ""): r["value"][0]
                     for r in rows if r["kind"] == "fact"}

    def finish(self) -> None:
        """Append the derived events (reach, first_close, grasp_ok, lift, drop, place, release, success)."""
        self.writer.events.extend(derive_events(self.writer.events, self.writer.boundaries,
                                                tuple(self.sensor.links), self.event_cfg))
