"""Episode record (PLAN.md §3). One directory per episode:

  frames/<camera>/<step>.png    raw RGB, row 0 = top
  depth/<camera>/<step>.npz     metric depth [m] (key "depth"), same pixels as the frame
  camera.json                   per camera: intrinsics, and extrinsics (static) or per-step extrinsics (moving)
  proprio.parquet               one row per step: step, joint_pos, hand_cmd.<hand>, finger_pos.<hand>,
                                link.<name>.pos (world), link.<name>.quat (world, wxyz)
  taxels.parquet                one row per (step, taxel): site_id, site, hand, finger, link,
                                local_pos (link frame), force_link, force_world
  events.parquet                privileged recorder output (condex.privileged); long format:
                                step, kind (object_pose | fact | contact | event), name, hand, a, b, value
  boundaries.parquet            step, hand, type (grasp | release), settled_step, subtask, retry
                                (+ judge/gate columns in closed-loop runs)
  meta.json                     task, seed, episode, executor checkpoint, commit, config path, job id, ...

Overlays are never stored here; they are rendered from these files (condex.render).
`EpisodeData` is the read interface shared by the on-disk reader and the live writer, so the
renderer and the judge see identical inputs offline and in the closed loop.
"""

from __future__ import annotations

import json
from collections import defaultdict
from functools import lru_cache
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image

from condex.camera import Camera

TABLES = ("proprio", "taxels", "events", "boundaries")


class EpisodeData:
    """Read interface. Subclasses fill: root, meta, cameras, proprio (step -> row),
    taxels (step -> rows), events (rows), boundaries (rows)."""

    root: Path
    meta: dict
    cameras: dict
    proprio: dict[int, dict]
    taxels: dict[int, list[dict]]
    events: list[dict]
    boundaries: list[dict]

    # --- frames ------------------------------------------------------------
    def frame_path(self, camera: str, step: int) -> Path:
        return self.root / "frames" / camera / f"{step:05d}.png"

    def frame(self, camera: str, step: int) -> np.ndarray:
        return np.asarray(Image.open(self.frame_path(camera, step)).convert("RGB"))

    def depth(self, camera: str, step: int) -> np.ndarray | None:
        p = self.root / "depth" / camera / f"{step:05d}.npz"
        return np.load(p)["depth"] if p.exists() else None

    def frame_steps(self, camera: str) -> list[int]:
        d = self.root / "frames" / camera
        return sorted(int(p.stem) for p in d.glob("*.png")) if d.exists() else []

    def camera_names(self) -> list[str]:
        return list(self.cameras)

    # --- geometry ----------------------------------------------------------
    def camera(self, name: str, step: int) -> Camera:
        c = self.cameras[name]
        if c.get("moving"):
            extr = c["per_step"][str(step)]
        else:
            extr = c["extrinsics"]
        return Camera.from_json(c["intrinsics"], extr)

    def link_pose(self, link: str, step: int) -> tuple[np.ndarray, np.ndarray]:
        row = self.proprio[step]
        return np.asarray(row[f"link.{link}.pos"], float), np.asarray(row[f"link.{link}.quat"], float)

    def taxel_rows(self, step: int) -> list[dict]:
        return self.taxels.get(step, [])

    def steps(self) -> list[int]:
        return sorted(self.proprio)


def _to_jsonable(v):
    if isinstance(v, np.ndarray):
        return v.tolist()
    if isinstance(v, (np.floating, np.integer, np.bool_)):
        return v.item()
    return v


def _write_table(path: Path, rows: list[dict]) -> None:
    rows = [{k: _to_jsonable(v) for k, v in r.items()} for r in rows]
    if not rows:
        pq.write_table(pa.table({"step": pa.array([], pa.int64())}), path)
        return
    keys = list(dict.fromkeys(k for r in rows for k in r))
    pq.write_table(pa.Table.from_pylist([{k: r.get(k) for k in keys} for r in rows]), path)


def _read_table(path: Path) -> list[dict]:
    return pq.read_table(path).to_pylist() if path.exists() else []


class EpisodeReader(EpisodeData):
    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.meta = json.loads((self.root / "meta.json").read_text())
        self.cameras = json.loads((self.root / "camera.json").read_text())
        self.proprio = {int(r["step"]): r for r in _read_table(self.root / "proprio.parquet")}
        self.taxels = defaultdict(list)
        for r in _read_table(self.root / "taxels.parquet"):
            self.taxels[int(r["step"])].append(r)
        self.events = _read_table(self.root / "events.parquet")
        self.boundaries = _read_table(self.root / "boundaries.parquet")


@lru_cache(maxsize=8)
def load(root: str) -> EpisodeReader:
    """Cached reader: batch rendering touches the same episode many times."""
    return EpisodeReader(root)


class EpisodeWriter(EpisodeData):
    """Live record. Frames go to disk immediately; tables are kept in memory and written by close().
    The privileged recorder appends to `events`; nothing here reads simulator state."""

    def __init__(self, root: str | Path, meta: dict):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.meta = dict(meta)
        self.cameras: dict = {}
        self.proprio: dict[int, dict] = {}
        self.taxels: dict[int, list[dict]] = defaultdict(list)
        self.events: list[dict] = []
        self.boundaries: list[dict] = []
        self.closed = False

    def add_camera(self, name: str, cam: Camera, moving: bool, step: int) -> None:
        """Register intrinsics once; for a moving camera call again every recorded step."""
        c = self.cameras.setdefault(name, {"intrinsics": cam.intrinsics_json(), "moving": bool(moving),
                                           "convention": "mujoco: looks -z, +y up; xmat = world_from_camera; pixels y down"})
        e = Camera.extrinsics_json(cam.pos, cam.xmat)
        if moving:
            c.setdefault("per_step", {})[str(step)] = e
        else:
            c.setdefault("extrinsics", e)

    def add_frame(self, camera: str, step: int, rgb: np.ndarray, depth: np.ndarray | None = None) -> None:
        p = self.frame_path(camera, step)
        p.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(np.ascontiguousarray(rgb, dtype=np.uint8)).save(p, compress_level=1)
        if depth is not None:
            d = self.root / "depth" / camera
            d.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(d / f"{step:05d}.npz", depth=np.asarray(depth, np.float16))

    def add_proprio(self, step: int, row: dict) -> None:
        self.proprio[step] = {"step": step, **{k: _to_jsonable(v) for k, v in row.items()}}

    def add_taxels(self, step: int, rows: list[dict]) -> None:
        self.taxels[step] = [{"step": step, **r} for r in rows]

    def add_boundary(self, row: dict) -> dict:
        """Returns the stored row; closed-loop runs add the judge/gate columns to it."""
        self.boundaries.append(dict(row))
        return self.boundaries[-1]

    def update_meta(self, **kv) -> None:
        self.meta.update(kv)

    def close(self) -> Path:
        if self.closed:
            return self.root
        (self.root / "meta.json").write_text(json.dumps(self.meta, indent=1, default=_to_jsonable))
        (self.root / "camera.json").write_text(json.dumps(self.cameras))
        _write_table(self.root / "proprio.parquet", [self.proprio[s] for s in sorted(self.proprio)])
        _write_table(self.root / "taxels.parquet", [r for s in sorted(self.taxels) for r in self.taxels[s]])
        _write_table(self.root / "events.parquet", self.events)
        _write_table(self.root / "boundaries.parquet", self.boundaries)
        self.closed = True
        return self.root


def disk_usage(root: str | Path) -> dict[str, int]:
    """Bytes per top-level entry (frames, depth, each table) - PLAN.md §9 disk question."""
    root = Path(root)
    out = {}
    for p in root.iterdir():
        out[p.name] = sum(f.stat().st_size for f in p.rglob("*") if f.is_file()) if p.is_dir() else p.stat().st_size
    return out


def find_episodes(root: str | Path) -> list[Path]:
    """Every episode directory (has meta.json) under root, sorted."""
    return sorted(p.parent for p in Path(root).rglob("meta.json"))
