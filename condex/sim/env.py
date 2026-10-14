"""GR-1 Tabletop gym env, built the way RLDX-1's rollout_policy builds it
(gym.make(env_name, enable_render=True, seed=...) inside its MultiStepWrapper), except that the
wrapper executes one action per step so the harness sees every step; the harness replays each
16-step action chunk itself (DECISIONS D4).

Seeding mirrors RLDX-1's VideoRecordingWrapper: episode k with base seed s (env_idx 0) resets with
s * 100000 + k, so episode ids select the same initial states as the published eval.
Cluster only (needs the robocasa sim venv from scripts/install_rldx.sh).
"""

from __future__ import annotations

import os
import random

import numpy as np

from condex.camera import Camera

SEED_STRIDE = 100000


def batch_obs(obs: dict) -> dict:
    """Add the batch axis the way gymnasium's SyncVectorEnv(n=1) does."""
    return {k: (v,) if isinstance(v, str) else np.asarray(v)[None] for k, v in obs.items()}


class SimEnv:
    def __init__(self, env_name: str, video_delta_indices, state_delta_indices, seed: int = 42, env_idx: int = 0):
        os.environ.setdefault("MUJOCO_GL", "egl")
        os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
        import gymnasium as gym
        import robocasa  # noqa: F401  (registers envs)
        import robocasa.utils.gym_utils.gymnasium_groot  # noqa: F401
        from rldx.eval.sim.wrapper.multistep_wrapper import MultiStepWrapper

        self.env_name = env_name
        self.env_seed = seed + env_idx
        random.seed(self.env_seed)
        np.random.seed(self.env_seed)
        self.base = gym.make(env_name, enable_render=True, seed=self.env_seed)
        self.env = MultiStepWrapper(
            self.base,
            video_delta_indices=np.asarray(video_delta_indices),
            state_delta_indices=None if state_delta_indices is None else np.asarray(state_delta_indices),
            n_action_steps=1,
            max_episode_steps=None,
        )
        self.reset_hooks: list = []  # called after every reset (robocasa may rebuild the MjModel)

    @property
    def rs(self):
        """The robosuite env."""
        return self.base.unwrapped.env

    @property
    def model(self):
        return self.rs.sim.model._model

    @property
    def data(self):
        return self.rs.sim.data._data

    def reset(self, episode_id: int) -> dict:
        ep_seed = self.env_seed * SEED_STRIDE + int(episode_id)
        random.seed(ep_seed)
        np.random.seed(ep_seed)
        self.rs.seed = ep_seed
        self.rs.rng = np.random.default_rng(ep_seed)
        obs, _ = self.env.reset(seed=ep_seed)
        for hook in self.reset_hooks:
            hook(self)
        return obs

    def step(self, action: dict) -> tuple[dict, bool, dict]:
        obs, _, _, _, info = self.env.step({k: np.asarray(v)[None] for k, v in action.items()})
        return obs, bool(np.any(info.get("success", False))), info

    @staticmethod
    def instruction(obs: dict) -> str:
        for k, v in obs.items():
            if k.startswith("annotation."):
                return v if isinstance(v, str) else str(np.asarray(v).reshape(-1)[0])
        return ""

    # --- cameras -----------------------------------------------------------
    def camera_name(self, name: str) -> str:
        """Exact name, else the unique camera whose name ends with `_<name>` (robosuite prefixes)."""
        import mujoco

        m = self.model
        names = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_CAMERA, i) or "" for i in range(m.ncam)]
        if name in names:
            return name
        hits = [n for n in names if n.endswith("_" + name)]
        if len(hits) != 1:
            raise KeyError(f"camera {name!r}: matches {hits} among {names}")
        return hits[0]

    def camera(self, name: str, width: int, height: int) -> Camera:
        import mujoco

        cid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, self.camera_name(name))
        return Camera.from_fovy(float(self.model.cam_fovy[cid]), width, height, np.array(self.data.cam_xpos[cid]),
                                np.array(self.data.cam_xmat[cid]).reshape(3, 3))

    def render(self, name: str, width: int, height: int, depth: bool = True):
        """RGB and metric depth, flipped to row 0 = top (robosuite renders upside down)."""
        out = self.rs.sim.render(camera_name=self.camera_name(name), width=width, height=height, depth=depth)
        if not depth:
            return np.ascontiguousarray(out[::-1]), None
        rgb, d = out
        m = self.model
        near, far = m.vis.map.znear * m.stat.extent, m.vis.map.zfar * m.stat.extent
        metric = near / (1.0 - d * (1.0 - near / far))
        return np.ascontiguousarray(rgb[::-1]), np.ascontiguousarray(metric[::-1])
