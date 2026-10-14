"""Pinhole cameras in MuJoCo convention, stored per episode in camera.json.

Conventions: the camera looks along its local -z with +y up; `xmat` is the world-from-camera
rotation (columns = camera axes) and `pos` the world position. Pixels are (x right, y down) in the
stored image (robosuite renders upside down; the stored frames are flipped to row 0 = top).
Intrinsics refer to the stored image, so any pad/resize the env applies before the policy sees
the frame is folded into fx, fy, cx, cy (see `padded_resized`).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Camera:
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float
    pos: np.ndarray  # (3,)
    xmat: np.ndarray  # (3, 3)

    @classmethod
    def from_fovy(cls, fovy_deg: float, width: int, height: int, pos, xmat) -> "Camera":
        """MuJoCo camera: square pixels, vertical field of view `fovy`, principal point at the center."""
        f = (height / 2.0) / np.tan(np.deg2rad(fovy_deg) / 2.0)
        return cls(width, height, f, f, width / 2.0, height / 2.0, np.asarray(pos, float), np.asarray(xmat, float).reshape(3, 3))

    def padded_resized(self, out_size: int) -> "Camera":
        """The camera of the image the env makes by centered pad-to-square and resize to `out_size`
        (robocasa GrootRoboCasaEnv.process_img)."""
        dim = max(self.width, self.height)
        s = out_size / dim
        pad_l, pad_t = (dim - self.width) // 2, (dim - self.height) // 2
        return Camera(out_size, out_size, self.fx * s, self.fy * s, (self.cx + pad_l) * s, (self.cy + pad_t) * s, self.pos, self.xmat)

    def with_pose(self, pos, xmat) -> "Camera":
        return Camera(self.width, self.height, self.fx, self.fy, self.cx, self.cy, np.asarray(pos, float), np.asarray(xmat, float).reshape(3, 3))

    def project(self, p_world) -> tuple[np.ndarray, np.ndarray]:
        """World points (..., 3) -> pixels (..., 2) and depth along the view axis (...)."""
        pc = (np.asarray(p_world, float) - self.pos) @ self.xmat  # R^T (p - c)
        depth = -pc[..., 2]
        safe = np.where(np.abs(depth) < 1e-9, 1e-9, depth)
        u = self.cx + self.fx * pc[..., 0] / safe
        v = self.cy - self.fy * pc[..., 1] / safe
        return np.stack([u, v], axis=-1), depth

    def intrinsics_json(self) -> dict:
        return {"width": self.width, "height": self.height, "fx": self.fx, "fy": self.fy, "cx": self.cx, "cy": self.cy}

    @staticmethod
    def extrinsics_json(pos, xmat) -> dict:
        return {"pos": np.asarray(pos, float).round(6).tolist(), "xmat": np.asarray(xmat, float).reshape(9).round(6).tolist()}

    @classmethod
    def from_json(cls, intr: dict, extr: dict) -> "Camera":
        return cls(int(intr["width"]), int(intr["height"]), float(intr["fx"]), float(intr["fy"]), float(intr["cx"]),
                   float(intr["cy"]), np.asarray(extr["pos"], float), np.asarray(extr["xmat"], float).reshape(3, 3))
