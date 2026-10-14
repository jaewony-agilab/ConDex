"""Synthetic episode records: a top-down camera over one hand whose fingertips sit in a row."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ.setdefault("MUJOCO_GL", "egl")

from condex.camera import Camera  # noqa: E402
from condex.episode import EpisodeWriter  # noqa: E402
from condex.hands import CELLS, FINGERS, site_id, site_name  # noqa: E402

LINKS = {"thumb": "R_thumb_distal_link", "index": "R_index_intermediate_link", "middle": "R_middle_intermediate_link",
         "ring": "R_ring_intermediate_link", "pinky": "R_pinky_intermediate_link", "palm": "r_palm"}
# fingertips along world x at height 0.5, palm behind them; camera 1 m above looking straight down
ANCHORS = {f: np.array([-0.08 + 0.04 * i, 0.03, 0.5]) for i, f in enumerate(FINGERS)} | {"palm": np.array([0.0, -0.05, 0.5])}
CAM = Camera.from_fovy(60.0, 256, 256, [0, 0, 1.5], np.eye(3))


def taxel_rows(forces: dict[str, np.ndarray], hand: str = "right") -> list[dict]:
    rows = []
    for f, link in LINKS.items():
        cells = 1 if f == "palm" else CELLS
        for c in range(cells):
            fw = np.asarray(forces.get(f, np.zeros(3)), float) / cells
            rows.append({"site_id": site_id(hand, f, c), "site": site_name(hand, f, c), "hand": hand, "finger": f, "link": link,
                         "local_pos": [0.0, 0.0, 0.0], "force_link": fw.tolist(), "force_world": fw.tolist(), "force_self_world": [0.0, 0.0, 0.0]})
    return rows


def make_episode(root: Path, n_steps: int = 120, forces_at=None, boundaries=None, facts=None, depth_value: float = 1.0,
                 meta=None) -> Path:
    """forces_at(step) -> {finger: world force}; facts(step) -> list of event rows (privileged)."""
    w = EpisodeWriter(root, {"task": "SyntheticTask", "seed": 42, "episode_id": 0, "instruction": "pick up the can and place it in the bowl",
                             **(meta or {})})
    rng = np.random.default_rng(0)
    base = (rng.integers(90, 140, (256, 256, 3))).astype(np.uint8)
    for s in range(n_steps):
        w.add_camera("ego_view", CAM, moving=True, step=s)
        if s % 10 == 0 or any(int(b["settled_step"]) == s for b in (boundaries or [])):
            w.add_frame("ego_view", s, base, np.full((256, 256), depth_value, np.float32))
        row = {"state.right_hand": [0.0] * 6, "hand_cmd.right": [0.0] * 6}
        for f, link in LINKS.items():
            row[f"link.{link}.pos"] = ANCHORS[f].tolist()
            row[f"link.{link}.quat"] = [1.0, 0.0, 0.0, 0.0]
        w.add_proprio(s, row)
        w.add_taxels(s, taxel_rows(forces_at(s) if forces_at else {}))
        if facts:
            w.events.extend(facts(s))
    for b in boundaries or []:
        w.add_boundary(b)
    return w.close()


@pytest.fixture
def episode(tmp_path):
    def forces(s):
        return {"index": [5.0, 0, 0], "thumb": [0, 3.0, 0], "palm": [0, 0, 4.0]} if s >= 50 else {}

    bounds = [{"step": 45, "hand": "right", "type": "grasp", "settled_step": 50, "subtask": "pick up the can", "retry": 0},
              {"step": 95, "hand": "right", "type": "release", "settled_step": 100, "subtask": "place the can", "retry": 0}]
    return make_episode(tmp_path / "ep0", forces_at=forces, boundaries=bounds)
