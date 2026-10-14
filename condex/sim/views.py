"""Camera recording into the episode record (frames, depth, camera.json).

Config (`record.cameras` in configs/gr1.yaml), one entry per stored camera name:
  source: obs     the policy image from the observation (`obs_key`); its geometry is the MuJoCo
                  camera `camera` rendered at `raw` [w, h], padded to square and resized to the
                  image size (robocasa process_img). Depth is rendered at `depth_raw_scale` x raw
                  and mapped the same way.
  source: render  rendered by us at `size` [w, h] (wrist views: judge only, never fed to the policy)
  stride          record every n-th step; boundary steps are always recorded
All cameras are stored as moving (per-step extrinsics): the GR-1 head and wrists move.
"""

from __future__ import annotations

import numpy as np
from PIL import Image

from condex.episode import EpisodeWriter


def _pad_resize_depth(depth: np.ndarray, out: int) -> np.ndarray:
    h, w = depth.shape
    dim = max(h, w)
    canvas = np.full((dim, dim), np.inf, np.float32)
    t, l = (dim - h) // 2, (dim - w) // 2
    canvas[t:t + h, l:l + w] = depth
    return np.asarray(Image.fromarray(canvas).resize((out, out), Image.NEAREST))


class CameraRecorder:
    def __init__(self, env, cameras: dict[str, dict]):
        self.env, self.cameras = env, cameras

    def record(self, writer: EpisodeWriter, step: int, obs: dict, force: bool = False) -> None:
        for name, c in self.cameras.items():
            if not force and step % int(c.get("stride", 1)):
                continue
            if c["source"] == "obs":
                img = np.asarray(obs[c["obs_key"]])
                img = img[-1] if img.ndim == 4 else img
                raw_w, raw_h = c["raw"]
                cam = self.env.camera(c["camera"], raw_w, raw_h).padded_resized(img.shape[1])
                depth = None
                if c.get("depth", True):
                    k = float(c.get("depth_raw_scale", 0.5))
                    dw, dh = int(raw_w * k), int(raw_h * k)
                    _, d = self.env.render(c["camera"], dw, dh, depth=True)
                    depth = _pad_resize_depth(d, img.shape[1])
            else:
                w, h = c["size"]
                img, depth = self.env.render(c["camera"], w, h, depth=c.get("depth", True))
                cam = self.env.camera(c["camera"], w, h)
            writer.add_camera(name, cam, moving=True, step=step)
            writer.add_frame(name, step, img, depth)
