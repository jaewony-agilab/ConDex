"""Build the cluster-side stack for one task from the config: executor client, env, taxels, cameras."""

from __future__ import annotations

from condex import config as C


def build(cfg: dict, task: str, seed: int, port: int):
    from condex.sim.env import SimEnv
    from condex.sim.executor import RLDXExecutor
    from condex.sim.taxels import TaxelSensor
    from condex.sim.views import CameraRecorder

    executor = RLDXExecutor("127.0.0.1", port, n_action_steps=cfg["n_action_steps"])
    video_idx, state_idx = executor.modality_delta_indices()
    env = SimEnv(C.env_name(cfg, task), video_idx, state_idx, seed=seed)
    t = cfg["taxels"]
    sensor = TaxelSensor(env.model, env.data, t["links"], t["robot_prefixes"],
                         tuple(t["grid_axes"]) if t.get("grid_axes") else None, t.get("exclude_robot_contacts", False))
    env.reset_hooks.append(lambda e: sensor.bind(e.model, e.data))
    return env, executor, sensor, CameraRecorder(env, cfg["record"]["cameras"])
