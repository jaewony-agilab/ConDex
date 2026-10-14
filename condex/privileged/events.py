"""Privileged per-step facts -> derived events (pure; tested without a simulator).

The recorder writes, per step, long-format rows into events.parquet:
  kind=object_pose  name=<object>           value=[x, y, z, qw, qx, qy, qz]
  kind=fact         name=grasped,hand=h     value=[0|1]   target object held by hand h
  kind=fact         name=grasped_other,hand value=[0|1]   some other object held by hand h
  kind=fact         name=in_receptacle      value=[0|1]
  kind=fact         name=success            value=[0|1]
  kind=fact         name=reach_dist,hand    value=[m]     fingertip centroid -> target object
  kind=fact         name=base_dist          value=[m]     robot base -> target object (horizontal)
  kind=contact      a=<robot link>, b=<object>, hand=h
and, at the end of the episode, the derived events (kind=event):
  reach, first_close, grasp_ok, lift, drop, place, release, success  (each with step, hand where relevant)
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

import numpy as np

TARGET = "obj"  # robocasa's name for the task object


@dataclass(frozen=True)
class EventConfig:
    reach_m: float = 0.05  # fingertip centroid within this of the object = reach
    lift_m: float = 0.03  # object raised this far above its pre-grasp height = lifted
    drop_window: int = 10  # a release boundary within this many steps explains a lost grasp


class Facts:
    """Dense per-step arrays from the long-format rows. Missing steps carry the previous value."""

    def __init__(self, rows: list[dict], n_steps: int | None = None):
        steps = [int(r["step"]) for r in rows if r.get("kind") in ("fact", "object_pose")]
        self.n = (max(steps) + 1 if steps else 0) if n_steps is None else n_steps
        self.series: dict[tuple[str, str | None], np.ndarray] = {}
        self.poses: dict[str, np.ndarray] = {}
        self.contacts: dict[int, list[dict]] = defaultdict(list)
        raw: dict[tuple, dict[int, list]] = defaultdict(dict)
        for r in rows:
            k, s = r.get("kind"), int(r["step"])
            if k == "fact":
                raw[("f", r["name"], r.get("hand"))][s] = r["value"]
            elif k == "object_pose":
                raw[("p", r["name"], None)][s] = r["value"]
            elif k == "contact":
                self.contacts[s].append(r)
        for (t, name, hand), by_step in raw.items():
            width = len(next(iter(by_step.values())))
            arr = np.full((self.n, width), np.nan)
            for s, v in by_step.items():
                if s < self.n:
                    arr[s] = v
            for s in range(1, self.n):  # forward fill
                if np.isnan(arr[s]).all():
                    arr[s] = arr[s - 1]
            if t == "f":
                self.series[(name, hand)] = arr[:, 0] if width == 1 else arr
            else:
                self.poses[name] = arr

    def get(self, name: str, hand: str | None = None, default: float = np.nan) -> np.ndarray:
        return self.series.get((name, hand), np.full(self.n, default))

    def flag(self, name: str, hand: str | None = None) -> np.ndarray:
        return np.nan_to_num(self.get(name, hand, 0.0)) > 0.5

    def obj_pos(self, name: str = TARGET) -> np.ndarray:
        return self.poses.get(name, np.full((self.n, 7), np.nan))[:, :3]

    def touching(self, hand: str, step: int, obj: str = TARGET) -> list[str]:
        return sorted({r["a"] for r in self.contacts.get(step, []) if r.get("hand") == hand and r.get("b") == obj})


def _first(mask: np.ndarray, start: int = 0) -> int | None:
    idx = np.flatnonzero(mask[start:])
    return int(idx[0]) + start if idx.size else None


def derive_events(rows: list[dict], boundaries: list[dict], hands=("left", "right"),
                  cfg: EventConfig = EventConfig()) -> list[dict]:
    f = Facts(rows)
    ev: list[dict] = []

    def add(name, step, hand=None):
        if step is not None:
            ev.append({"step": int(step), "kind": "event", "name": name, "hand": hand})

    z = f.obj_pos()[:, 2]
    any_grasp = np.zeros(f.n, bool)
    for h in hands:
        add("reach", _first(np.nan_to_num(f.get("reach_dist", h), nan=np.inf) < cfg.reach_m), h)
        closes = [int(b["step"]) for b in boundaries if b["hand"] == h and b["type"] == "grasp"]
        add("first_close", min(closes) if closes else None, h)
        g = f.flag("grasped", h)
        any_grasp |= g
        g0 = _first(g)
        add("grasp_ok", g0, h)
        if g0 is not None and not np.isnan(z[g0]):
            add("lift", _first(g & (z - z[g0] >= cfg.lift_m), g0), h)
        releases = [int(b["step"]) for b in boundaries if b["hand"] == h and b["type"] == "release"]
        for b in boundaries:
            if b["hand"] == h and b["type"] == "release":
                add("release", int(b["step"]), h)
        # drops: grasp lost with no release boundary nearby and the object not in the receptacle
        inr = f.flag("in_receptacle")
        for s in np.flatnonzero(g[:-1] & ~g[1:]) + 1:
            if not any(abs(int(s) - r) <= cfg.drop_window for r in releases) and not inr[s]:
                add("drop", int(s), h)
    add("place", _first(f.flag("in_receptacle") & ~any_grasp))
    add("success", _first(f.flag("success")))
    return sorted(ev, key=lambda e: e["step"])
