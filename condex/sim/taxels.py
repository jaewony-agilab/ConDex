"""Taxels on the fingertip and palm links, read from MuJoCo contacts (PLAN.md §2 "Sensor").

GR-1 Tabletop runs on MuJoCo (DECISIONS D1), so the plan's Isaac Lab contact sensors become this:
each fingertip link carries a 2x2 grid of taxels over its collision-geometry bounding box in the
link frame (along two grid axes; default = the two longest box axes), the palm one taxel. A contact
between a taxel link and another body adds its force (the force on the link) to the cell that
contains the contact point. Like a real tactile skin, contacts with other robot links count
(fingertips pressed together register); only contacts with the link's kinematic parent/children
(MuJoCo's parent-filter rule) are dropped, or all robot contacts with exclude_robot_contacts=True. Reading mjData is passive:
the benchmark, its dynamics and RLDX-1 are untouched. Contact partner identity is never exposed.
"""

from __future__ import annotations

import mujoco
import numpy as np

from condex.hands import CELLS, FINGERS, PALM, site_id, site_name


def resolve_body(model, suffix: str) -> int:
    names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i) or "" for i in range(model.nbody)]
    hits = [i for i, n in enumerate(names) if n == suffix or n.endswith("_" + suffix)]
    if len(hits) != 1:
        raise KeyError(f"taxel link {suffix!r}: expected 1 body, got {[names[i] for i in hits]}")
    return hits[0]


def body_box(model, body: int) -> tuple[np.ndarray, np.ndarray]:
    """Bounding box (center, half extents) of the body's collision geoms, in the body frame."""
    pts = []
    for g in range(model.ngeom):
        if model.geom_bodyid[g] != body or not (model.geom_contype[g] or model.geom_conaffinity[g]):
            continue
        c, h = model.geom_aabb[g, :3], model.geom_aabb[g, 3:]
        R = np.zeros(9)
        mujoco.mju_quat2Mat(R, model.geom_quat[g])
        corners = c + h * np.array([[i, j, k] for i in (-1, 1) for j in (-1, 1) for k in (-1, 1)])
        pts.append(model.geom_pos[g] + corners @ R.reshape(3, 3).T)
    if not pts:
        raise KeyError(f"body {mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body)!r} has no collision geom")
    pts = np.concatenate(pts)
    lo, hi = pts.min(0), pts.max(0)
    return (lo + hi) / 2, (hi - lo) / 2


class TaxelSensor:
    def __init__(self, model, data, links: dict[str, dict[str, str]], robot_prefixes: list[str],
                 grid_axes: tuple[int, int] | None = None, exclude_robot_contacts: bool = False):
        """links: hand -> {thumb, index, middle, ring, pinky, palm} -> body-name suffix."""
        self.links, self.robot_prefixes, self.grid_axes = links, list(robot_prefixes), grid_axes
        self.exclude_robot_contacts = exclude_robot_contacts
        self.bind(model, data)

    def bind(self, model, data) -> None:
        """(Re)resolve ids. Call after every env reset: robocasa rebuilds the MjModel per episode."""
        self.m, self.d = model, data
        self.key_of_body: dict[int, tuple[str, str]] = {}
        self.body_of: dict[tuple[str, str], int] = {}
        self.box: dict[tuple[str, str], tuple[np.ndarray, np.ndarray, tuple[int, int]]] = {}
        for hand, fingers in self.links.items():
            for finger, suffix in fingers.items():
                if finger not in (*FINGERS, PALM):
                    raise ValueError(f"unknown finger {finger!r}")
                b = resolve_body(model, suffix)
                self.key_of_body[b] = (hand, finger)
                self.body_of[(hand, finger)] = b
                c, h = body_box(model, b)
                axes = self.grid_axes or tuple(sorted(np.argsort(-h)[:2].tolist()))
                self.box[(hand, finger)] = (c, h, axes)
        names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i) or "" for i in range(model.nbody)]
        self.body_names = names
        self.robot_body = np.array([any(n.startswith(p) for p in self.robot_prefixes) for n in names])
        missing = [names[b] for b in self.key_of_body if not self.robot_body[b]]
        if missing:
            raise ValueError(f"taxel links not covered by robot_prefixes {self.robot_prefixes}: {missing}")
        self._f6 = np.zeros(6)

    def _ignored(self, body: int, other: int) -> bool:
        if not self.robot_body[other]:
            return False
        if self.exclude_robot_contacts:
            return True
        # kinematic neighbours through weld bodies, world excepted (MuJoCo's own filterparent rule)
        parent, weld = self.m.body_parentid, self.m.body_weldid
        wb, wo = int(weld[body]), int(weld[other])
        return bool((wo != 0 and int(weld[parent[body]]) == wo) or (wb != 0 and int(weld[parent[other]]) == wb))

    def _cell(self, key, p_local) -> int:
        if key[1] == PALM:
            return 0
        c, _, (a, b) = self.box[key]
        return 2 * int(p_local[a] >= c[a]) + int(p_local[b] >= c[b])

    def _cell_center(self, key, cell) -> np.ndarray:
        c, h, (a, b) = self.box[key]
        if key[1] == PALM:
            return c.copy()
        p = c.copy()
        p[a] += (h[a] / 2) * (1 if cell // 2 else -1)
        p[b] += (h[b] / 2) * (1 if cell % 2 else -1)
        return p

    def read(self) -> list[dict]:
        """One row per site (all sites, zeros included), forces in link and world frame [N].
        force_self_world: the part of force_world that comes from other robot links."""
        m, d = self.m, self.d
        acc = {(k, c): np.zeros(3) for k in self.body_of for c in range(1 if k[1] == PALM else CELLS)}
        acc_self = {k: np.zeros(3) for k in acc}  # the share from other robot links
        for i in range(d.ncon):
            con = d.contact[i]
            if con.efc_address < 0:
                continue
            b1, b2 = int(m.geom_bodyid[con.geom1]), int(m.geom_bodyid[con.geom2])
            k1, k2 = self.key_of_body.get(b1), self.key_of_body.get(b2)
            if k1 is None and k2 is None:
                continue
            mujoco.mj_contactForce(m, d, i, self._f6)
            fw = con.frame.reshape(3, 3).T @ self._f6[:3]  # world force of geom1 on geom2
            for key, body, other, f in ((k2, b2, b1, fw), (k1, b1, b2, -fw)):
                if key is None or self._ignored(body, other):
                    continue
                p_local = (con.pos - d.xpos[body]) @ d.xmat[body].reshape(3, 3)
                cell = (key, self._cell(key, p_local))
                acc[cell] += f
                if self.robot_body[other]:
                    acc_self[cell] += f
        rows = []
        for (key, cell), fw in acc.items():
            hand, finger = key
            body = self.body_of[key]
            R = d.xmat[body].reshape(3, 3)
            rows.append({"site_id": site_id(hand, finger, cell), "site": site_name(hand, finger, cell), "hand": hand,
                         "finger": finger, "link": self.body_names[body], "local_pos": self._cell_center(key, cell).tolist(),
                         "force_link": (fw @ R).tolist(), "force_world": fw.tolist(),
                         "force_self_world": acc_self[(key, cell)].tolist()})
        return sorted(rows, key=lambda r: r["site_id"])

    def link_poses(self) -> dict[str, list[float]]:
        """proprio columns: link.<name>.pos (world) and link.<name>.quat (world, wxyz)."""
        out = {}
        for key, b in self.body_of.items():
            n = self.body_names[b]
            out[f"link.{n}.pos"] = np.asarray(self.d.xpos[b]).tolist()
            out[f"link.{n}.quat"] = np.asarray(self.d.xquat[b]).tolist()
        return out
