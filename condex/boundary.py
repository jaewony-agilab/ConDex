"""Grasp/release boundaries from the executor's hand-closure command (PLAN.md Phase 0 step 4).

Per hand: the normalized closure of the commanded finger flexion crosses a threshold with
hysteresis (open -> closed = grasp, closed -> open = release); a settle window then runs until the
measured finger joints stop moving. The boundary is emitted at the settled step. Defined on the
action space, not on contacts, so the detector cannot see what the judge is asked about.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ClosureConfig:
    flexion_dims: tuple[int, ...]  # command dims that close the hand
    open_value: float  # command value of a fully open finger
    closed_value: float  # command value of a fully closed finger
    close_threshold: float = 0.6  # normalized closure to enter "closed"
    open_threshold: float = 0.3  # normalized closure to re-enter "open"
    settle_vel_eps: float = 1e-3  # max |dq| per step to count as still [rad]
    settle_steps: int = 3  # consecutive still steps
    max_settle_steps: int = 30  # emit anyway after this many steps

    def closure(self, command) -> float:
        c = np.atleast_1d(np.asarray(command, float))[list(self.flexion_dims)]
        return float(np.mean((c - self.open_value) / (self.closed_value - self.open_value)))


@dataclass
class Boundary:
    hand: str
    type: str  # grasp | release
    step: int  # command crossing
    settled_step: int


class HandBoundaryDetector:
    def __init__(self, hand: str, cfg: ClosureConfig):
        self.hand, self.cfg = hand, cfg
        self.reset()

    def reset(self, closed: bool = False) -> None:
        self.closed = closed
        self._pending: tuple[str, int] | None = None
        self._pending_steps = 0
        self._still = 0
        self._prev_q: np.ndarray | None = None
        self.last_closure = 0.0

    def update(self, step: int, command, joint_pos) -> Boundary | None:
        cfg = self.cfg
        closure = cfg.closure(command)
        self.last_closure = closure
        q = np.atleast_1d(np.asarray(joint_pos, float))
        vel = 0.0 if self._prev_q is None else float(np.max(np.abs(q - self._prev_q)))
        self._prev_q = q

        crossed = None
        if not self.closed and closure > cfg.close_threshold:
            self.closed, crossed = True, "grasp"
        elif self.closed and closure < cfg.open_threshold:
            self.closed, crossed = False, "release"
        if crossed is not None:  # a new crossing restarts the settle window
            self._pending, self._pending_steps, self._still = (crossed, step), 0, 0
            return None
        if self._pending is None:
            return None
        self._pending_steps += 1
        self._still = self._still + 1 if vel < cfg.settle_vel_eps else 0
        if self._still >= cfg.settle_steps or self._pending_steps >= cfg.max_settle_steps:
            (btype, crossing), self._pending = self._pending, None
            return Boundary(self.hand, btype, crossing, step)
        return None


class BoundaryDetector:
    """Both hands. update() returns the boundaries settled at this step (usually none)."""

    def __init__(self, cfgs: dict[str, ClosureConfig]):
        self.hands = {h: HandBoundaryDetector(h, c) for h, c in cfgs.items()}

    def reset(self) -> None:
        for d in self.hands.values():
            d.reset()

    def update(self, step: int, commands: dict, joints: dict) -> list[Boundary]:
        out = []
        for h, d in self.hands.items():
            b = d.update(step, commands[h], joints[h])
            if b is not None:
                out.append(b)
        return out
