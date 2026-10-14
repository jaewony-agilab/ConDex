"""Scripted recovery (PLAN.md §5): open the hand, lift back along the approach, then the harness
resumes the executor on the same subtask.

Joint space, no IK: the hand's flexion dims go to the open command over `open_steps`, the arm then
replays its last `retreat_steps` commanded targets in reverse (the approach path, backwards). Every
other key holds its last command. GR-1 arm actions are absolute joint targets.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from condex.boundary import ClosureConfig


@dataclass(frozen=True)
class RecoveryConfig:
    open_steps: int = 10
    retreat_steps: int = 20
    hold_steps: int = 5  # settle after the retreat


def recovery_actions(history: list[dict], hand: str, hand_key: str, arm_key: str, closure: ClosureConfig,
                     cfg: RecoveryConfig = RecoveryConfig()) -> list[dict]:
    """history: the executed per-step action dicts, oldest first."""
    if not history:
        return []
    last = {k: np.asarray(v, float).copy() for k, v in history[-1].items()}
    start = last[hand_key].copy()
    target = start.copy()
    target[list(closure.flexion_dims)] = closure.open_value
    out = []
    for i in range(1, cfg.open_steps + 1):
        a = {k: v.copy() for k, v in last.items()}
        a[hand_key] = start + (target - start) * i / cfg.open_steps
        out.append(a)
    arm_path = [np.asarray(h[arm_key], float) for h in history[-cfg.retreat_steps:]][::-1]
    for q in arm_path:
        a = {k: v.copy() for k, v in last.items()}
        a[hand_key], a[arm_key] = target.copy(), q
        out.append(a)
    out += [{k: v.copy() for k, v in out[-1].items()} for _ in range(cfg.hold_steps)]
    return out
