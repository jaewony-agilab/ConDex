"""Phase 0 step 2 probe + calibration inputs. Cluster: sim venv + policy server.

  1. open in air      hands commanded open, arms held: every taxel should read ~0
  2. fingertips together   flexion closed with nothing in the hand: thumb/finger taxels should read
  3. known-mass grasp  executor rollouts until the privileged recorder sees the target held and
                       lifted; the summed vertical taxel force is compared with m * g
  4. step-time overhead of the sensor read (and of recording) per env step
  5. calibration: hand-command and finger-joint percentiles from executor rollouts (closure
     open/closed values, flexion dims), fingertip |F| percentiles in contact (overlay clip),
     body / camera names, and one overlay frame at the most-closed step for a visual check
Writes probe.json and PNGs to --results.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from condex import config as C  # noqa: E402
from condex import render as R  # noqa: E402
from condex.episode import EpisodeWriter  # noqa: E402
from condex.harness import proprio_row  # noqa: E402
from condex.privileged.recorder import Recorder  # noqa: E402


def stats(rows):
    tot = {}
    for r in rows:
        k = f"{r['hand']}/{r['finger']}"
        tot[k] = tot.get(k, 0) + np.asarray(r["force_world"])
    return {k: round(float(np.linalg.norm(v)), 3) for k, v in tot.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="PnPCanToDrawerClose_GR1ArmsAndWaistFourierHands_Env")
    ap.add_argument("--episodes", type=int, default=3)
    ap.add_argument("--results", required=True)
    ap.add_argument("--raw", required=True)
    ap.add_argument("--config", default=None)
    ap.add_argument("--port", type=int, default=int(os.environ.get("RLDX_PORT", 20100)))
    args = ap.parse_args()
    cfg = C.load(args.config)
    res = Path(args.results)
    res.mkdir(parents=True, exist_ok=True)

    import mujoco

    from condex.sim.env import batch_obs
    from condex.sim.setup import build

    env, executor, sensor, cameras = build(cfg, args.task, cfg["seed"], args.port)
    out = {"run": C.run_info(cfg["_path"], task=args.task)}
    obs = env.reset(0)
    m = env.model
    out["bodies"] = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, i) for i in range(m.nbody)]
    out["cameras"] = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_CAMERA, i) for i in range(m.ncam)]
    out["obs_keys"] = {k: list(np.shape(v)) for k, v in obs.items()}
    out["taxel_links"] = {f"{h}/{f}": sensor.body_names[b] for (h, f), b in sensor.body_of.items()}
    out["taxel_boxes"] = {f"{h}/{f}": {"center": c.round(4).tolist(), "half": hh.round(4).tolist(), "grid_axes": list(ax)}
                          for (h, f), (c, hh, ax) in sensor.box.items()}

    # template action from one policy query: all keys, arms held where the policy wants them
    executor.reset("probe")
    template = executor.act(batch_obs(obs))[0]
    out["action_keys"] = {k: list(np.shape(v)) for k, v in template.items()}
    cl = C.closure_configs(cfg)
    hands = cfg["hands"]
    style = R.load_style(C.REPO / cfg["judge"]["style"])

    def scripted(closure: float, steps: int = 40):
        nonlocal obs
        env.reset(0)
        a = {k: np.asarray(v, float).copy() for k, v in template.items()}
        for h, hk in hands.items():
            c = a[hk["action"]]
            c[list(cl[h].flexion_dims)] = cl[h].open_value + closure * (cl[h].closed_value - cl[h].open_value)
        for _ in range(steps):
            obs, _, _ = env.step(a)
        return stats(sensor.read())

    out["probe_open_in_air"] = scripted(0.0)
    out["probe_fingertips_together"] = scripted(1.0)

    # overhead
    env.reset(0)
    a = {k: np.asarray(v, float) for k, v in template.items()}
    n = 50
    t0 = time.perf_counter()
    for _ in range(n):
        env.step(a)
    t_step = (time.perf_counter() - t0) / n
    t0 = time.perf_counter()
    for _ in range(n):
        env.step(a)
        sensor.read()
        sensor.link_poses()
    t_sensor = (time.perf_counter() - t0) / n - t_step
    out["step_time_ms"] = {"env_step": round(1e3 * t_step, 2), "sensor_read": round(1e3 * t_sensor, 3),
                           "overhead_frac": round(t_sensor / t_step, 4)}

    # executor rollouts: calibration percentiles and the known-mass grasp
    cmds = {h: [] for h in hands}
    joints = {h: [] for h in hands}
    tip_f, grasp_checks, best = [], [], (-1.0, None)
    for ep in range(args.episodes):
        w = EpisodeWriter(Path(args.raw) / f"probe_ep{ep}", {**out["run"], "episode_id": ep})
        rec = Recorder(env, sensor, w)
        obs = env.reset(ep)
        rec.bind()
        executor.reset(f"probe{ep}")
        obj_mass = None
        if "obj" in env.rs.objects:
            bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, env.rs.objects["obj"].root_body)
            obj_mass = float(env.model.body_subtreemass[bid])
        z0 = None
        queue = []
        for step in range(cfg["max_episode_steps"]):
            if not queue:
                queue = executor.act(batch_obs(obs))
            act = queue.pop(0)
            obs, success, _ = env.step(act)
            rows = sensor.read()
            rec.record(step)
            for h, hk in hands.items():
                cmds[h].append(np.asarray(act[hk["action"]], float))
                joints[h].append(np.asarray(obs[hk["state"]])[-1].astype(float))
            for r in rows:
                if r["finger"] != "palm":
                    f = np.linalg.norm(r["force_world"])
                    if f >= style.contact_threshold_n:
                        tip_f.append(f)
            closure = max(cl[h].closure(act[hk["action"]]) for h, hk in hands.items())
            if closure > best[0]:
                best = (closure, (ep, step))
                w.add_proprio(step, {**proprio_row(obs, act, {}), **sensor.link_poses()})
                w.add_taxels(step, rows)
                cameras.record(w, step, obs, force=True)
            zpos = env.data.xpos[env.rs.obj_body_id["obj"]][2] if "obj" in env.rs.objects else None
            if z0 is None:
                z0 = zpos
            for h in hands:
                if rec.last.get(f"grasped.{h}", 0) > 0.5 and zpos is not None and zpos - z0 > 0.03 and obj_mass:
                    fz = sum(np.asarray(r["force_world"])[2] for r in rows if r["hand"] == h)
                    grasp_checks.append({"episode": ep, "step": step, "hand": h, "sum_fz_n": round(float(fz), 3),
                                         "mg_n": round(obj_mass * 9.81, 3)})
            if success:
                break
        w.close()
    pct = lambda xs: {f"p{p}": np.percentile(np.stack(xs), p, axis=0).round(4).tolist() for p in (1, 5, 50, 95, 99)}  # noqa: E731
    out["hand_command_percentiles"] = {h: pct(v) for h, v in cmds.items() if v}
    out["finger_joint_percentiles"] = {h: pct(v) for h, v in joints.items() if v}
    out["fingertip_force_in_contact_n"] = ({"n": len(tip_f), **{f"p{p}": round(float(np.percentile(tip_f, p)), 3) for p in (50, 90, 95, 99)}}
                                           if tip_f else {"n": 0})
    out["known_mass_grasp"] = {"n": len(grasp_checks), "samples": grasp_checks[:20],
                               "median_ratio_upward_force_to_mg": (round(float(np.median([-g["sum_fz_n"] / g["mg_n"] for g in grasp_checks])), 3)
                                                                   if grasp_checks else None)}
    # one overlay at the most-closed step
    if best[1] is not None:
        ep, step = best[1]
        from condex.episode import EpisodeReader

        d = EpisodeReader(Path(args.raw) / f"probe_ep{ep}")
        for cam in d.camera_names():
            if d.frame_path(cam, step).exists():
                Image.fromarray(d.frame(cam, step)).save(res / f"mostclosed_{cam}_plain.png")
                Image.fromarray(R.render_frame(d, step, cam, style)).save(res / f"mostclosed_{cam}_overlay.png")
        out["most_closed"] = {"episode": ep, "step": step, "closure": round(best[0], 3)}
    (res / "probe.json").write_text(json.dumps(out, indent=1))
    print(json.dumps({k: v for k, v in out.items() if k not in ("bodies",)}, indent=1))


if __name__ == "__main__":
    main()
