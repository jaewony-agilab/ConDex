"""Episode loop around the frozen executor (PLAN.md Phase 0 step 3, Phase 4).

Every step: query the executor when its chunk is empty, step the env, then record proprio, taxels,
link poses and frames into the episode record (the recorder, if given, adds privileged facts).
The boundary detector runs on the commanded hand closure; at a settled boundary the frames are
recorded, and in closed-loop runs the judge sees a packet built from the live record (the same
code path as offline datasets), the gate decides, and recovery / replan / abort follow.

Sim-agnostic: env, executor, sensor, cameras, recorder and judge are injected (fakes in tests).
This module must not import condex.privileged: the recorder arrives as an object from scripts/.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field

import numpy as np

from condex.boundary import BoundaryDetector, ClosureConfig
from condex.cost import BudgetExceeded
from condex.episode import EpisodeWriter
from condex.gate import Action, Gate
from condex.judge.prompt import build_packet
from condex.judge.schema import Verdict
from condex.recovery import RecoveryConfig, recovery_actions
from condex.render import Style


@dataclass
class HandKeys:
    action: str  # e.g. action.right_hand (6-D closure command)
    state: str  # e.g. state.right_hand (measured finger joints)
    arm: str  # e.g. action.right_arm (replayed by recovery)


@dataclass
class LoopConfig:
    hands: dict[str, HandKeys]
    closure: dict[str, ClosureConfig]
    max_steps: int = 720
    stop_on_success: bool = True
    # closed loop
    condition: str | None = None  # None = executor alone; plain | text | overlay | oracle
    judge_cameras: list[str] = field(default_factory=list)
    style: Style = field(default_factory=Style)
    recovery: RecoveryConfig = field(default_factory=RecoveryConfig)
    reset_memory_on_resume: bool = True
    use_prev_frames: bool = True  # ablation: judge without the previous-boundary frames


def _state(obs: dict, key: str) -> np.ndarray:
    v = np.asarray(obs[key])
    return v[-1] if v.ndim > 1 else v


def proprio_row(obs: dict, action: dict | None, keys: dict[str, HandKeys]) -> dict:
    row = {}
    states = sorted(k for k in obs if k.startswith("state."))
    for k in states:
        row[k] = _state(obs, k).astype(float).tolist()
    if states:
        row["joint_pos"] = np.concatenate([np.atleast_1d(row[k]) for k in states]).tolist()
    for h, hk in keys.items():
        if hk.state in obs:
            row[f"finger_pos.{h}"] = _state(obs, hk.state).astype(float).tolist()
        if action is not None:
            row[f"hand_cmd.{h}"] = np.asarray(action[hk.action], float).tolist()
    if action is not None:
        for k, v in action.items():
            row[k] = np.asarray(v, float).reshape(-1).tolist()
    return row


def run_episode(env, executor, sensor, cameras, writer: EpisodeWriter, cfg: LoopConfig, episode_id: int,
                recorder=None, judge=None, gate: Gate | None = None, planner=None, batch_obs=lambda o: o) -> dict:
    t0 = time.time()
    obs = env.reset(episode_id)
    if recorder is not None:
        recorder.bind()
    executor.reset(f"{writer.meta.get('task', 'ep')}_{episode_id}_{uuid.uuid4().hex[:6]}")
    instruction = env.instruction(obs)
    writer.update_meta(instruction=instruction, episode_id=episode_id, condition=cfg.condition or "executor_alone")
    detector = BoundaryDetector(cfg.closure)
    if gate is not None:
        gate.reset()
    plan = None
    status, abort_reason = "running", ""
    try:
        plan = planner.plan(instruction) if planner is not None else None
    except BudgetExceeded as e:
        status, abort_reason = "aborted", f"budget: {e}"

    def record(step, obs, action, phase, force=False):
        writer.add_proprio(step, {**proprio_row(obs, action, cfg.hands), "phase": phase})
        writer.add_taxels(step, sensor.read())
        writer.proprio[step].update(sensor.link_poses())
        cameras.record(writer, step, obs, force=force)
        if recorder is not None:
            recorder.record(step)

    record(0, obs, None, "reset", force=True)
    queue: list[tuple[dict, str]] = []  # (action, phase)
    executed: list[dict] = []
    done_checks: list[dict] = []
    prev_verdict: Verdict | None = None
    retries = {h: 0 for h in cfg.hands}
    success, step, n_queries = False, 0, 0
    counts = {"judge_calls": 0, "recoveries": 0, "replans": 0}
    recovering = None

    def start_recovery(hand):
        hk = cfg.hands[hand]
        acts = recovery_actions(executed, hand, hk.action, hk.arm, cfg.closure[hand], cfg.recovery)
        queue[:] = [(a, "recovery") for a in acts]
        counts["recoveries"] += 1

    while status == "running" and step < cfg.max_steps:
        if not queue:
            queue[:] = [(a, "executor") for a in executor.act(batch_obs(obs))]
            n_queries += 1
        action, phase = queue.pop(0)
        obs, success, _ = env.step(action)
        step += 1
        executed.append(action)
        record(step, obs, action, phase)
        if success and cfg.stop_on_success:
            status = "success"
            break
        if phase == "recovery":
            if not queue:  # recovery finished: the hand is open again; resume the executor
                detector.hands[recovering].reset(closed=False)
                executor.interrupt(cfg.reset_memory_on_resume)
            continue
        commands = {h: action[hk.action] for h, hk in cfg.hands.items()}
        joints = {h: _state(obs, hk.state) for h, hk in cfg.hands.items()}
        for b in detector.update(step, commands, joints):
            cameras.record(writer, step, obs, force=True)
            subtask = plan.subtask_for(b.hand, b.type, done_checks) if plan is not None else ""
            row = {"step": b.step, "hand": b.hand, "type": b.type, "settled_step": b.settled_step,
                   "subtask": subtask, "retry": retries[b.hand]}
            row = writer.add_boundary(row)
            if judge is None:
                continue
            b_index = len(writer.boundaries) - 1
            oracle = cfg.condition == "oracle"  # the oracle reads the recorder, not images
            packet = build_packet(writer, b_index, "plain" if oracle else cfg.condition, cfg.style,
                                  [] if oracle else cfg.judge_cameras, prev_verdict, prev_frames=cfg.use_prev_frames)
            try:
                verdict = judge(packet)
                counts["judge_calls"] += 1
                decision, why = gate.decide(b.hand, verdict)
            except BudgetExceeded as e:
                status, abort_reason = "aborted", f"budget: {e}"
                break
            prev_verdict = verdict
            row.update({"judge": verdict.source, "outcome": verdict.outcome, "plan_still_valid": verdict.plan_still_valid,
                        "reason": verdict.reason, "parse_ok": verdict.parse_ok, "gate": decision.value})
            done_checks.append(row)
            if decision == Action.RECOVER:
                retries[b.hand] += 1
                recovering = b.hand
                start_recovery(b.hand)
            elif decision == Action.REPLAN:
                counts["replans"] += 1
                try:
                    d = planner.replan(instruction, why, done_checks) if planner is not None else {"decision": "resume"}
                except BudgetExceeded as e:
                    status, abort_reason = "aborted", f"budget: {e}"
                    break
                row["replan"] = d.get("decision")
                if d.get("decision") == "abort":
                    status, abort_reason = "aborted", f"planner: {d.get('why', '')}"
                elif d.get("decision") == "recover_then_resume" and d.get("hand") in cfg.hands:
                    recovering = d["hand"]
                    start_recovery(d["hand"])
                else:
                    queue.clear()
                    executor.interrupt(cfg.reset_memory_on_resume)
            elif decision == Action.ABORT:
                status, abort_reason = "aborted", why
            if decision == Action.CONTINUE and verdict.outcome == "success":
                retries[b.hand] = 0
    if status == "running":
        status = "timeout"
    if recorder is not None:
        recorder.finish()
    summary = {"episode_id": episode_id, "task": writer.meta.get("task"), "condition": cfg.condition or "executor_alone",
               "success": bool(success), "status": status, "abort_reason": abort_reason, "steps": step,
               "policy_queries": n_queries, "boundaries": len(writer.boundaries), **counts,
               "wall_s": round(time.time() - t0, 1)}
    writer.update_meta(summary=summary)
    writer.close()
    return summary
