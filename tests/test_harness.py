"""Episode loop with fake env / executor / sensor / cameras: record format, boundaries, closed loop."""

import numpy as np
from conftest import CAM, LINKS, taxel_rows

from condex import episode as E
from condex.boundary import ClosureConfig
from condex.gate import Gate
from condex.harness import HandKeys, LoopConfig, run_episode
from condex.judge.schema import Verdict
from condex.recovery import RecoveryConfig

CL = ClosureConfig(flexion_dims=(1, 2, 3, 4), open_value=0.0, closed_value=1.0, settle_steps=2, max_settle_steps=10)
HANDS = {"right": HandKeys("action.right_hand", "state.right_hand", "action.right_arm")}


class FakeEnv:
    """Fingers track the hand command with one step of lag. Success when told."""

    def __init__(self, success_at=None):
        self.success_at, self.t, self.q = success_at, 0, np.zeros(6)

    def reset(self, episode_id):
        self.t, self.q = 0, np.zeros(6)
        return self.obs()

    def obs(self):
        return {"state.right_hand": self.q[None].copy(), "state.right_arm": np.zeros((1, 7)),
                "video.ego_view": np.full((1, 256, 256, 3), 120, np.uint8), "annotation.task": "pick the can"}

    def step(self, action):
        self.t += 1
        self.q = 0.5 * self.q + 0.5 * np.asarray(action["action.right_hand"])
        return self.obs(), self.success_at is not None and self.t >= self.success_at, {}

    def instruction(self, obs):
        return obs["annotation.task"]


class FakeExecutor:
    """Closes the hand from step 10, opens it at step 40 (16-step chunks)."""

    def __init__(self, env):
        self.env, self.resets, self.interrupts = env, 0, 0

    def reset(self, sid):
        self.resets += 1

    def interrupt(self, reset_memory):
        self.interrupts += 1

    def act(self, obs):
        out = []
        for k in range(16):
            t = self.env.t + k + 1
            c = 1.0 if 10 <= t < 40 else 0.0
            out.append({"action.right_hand": np.array([0, c, c, c, c, 0.0]), "action.right_arm": np.full(7, float(t))})
        return out


class FakeSensor:
    def __init__(self, env):
        self.env = env

    def read(self):
        closed = self.env.q[1] > 0.5
        return taxel_rows({"index": [3.0, 0, 0], "thumb": [0, 3.0, 0]} if closed else {})

    def link_poses(self):
        out = {}
        for i, link in enumerate(LINKS.values()):
            out[f"link.{link}.pos"] = [-0.08 + 0.04 * i, 0.03, 0.5]
            out[f"link.{link}.quat"] = [1.0, 0, 0, 0]
        return out


class FakeCameras:
    def record(self, writer, step, obs, force=False):
        if force or step % 5 == 0:
            writer.add_camera("ego_view", CAM, moving=True, step=step)
            writer.add_frame("ego_view", step, obs["video.ego_view"][-1], np.full((256, 256), 1.0, np.float32))


def make_cfg(**kw):
    return LoopConfig(hands=HANDS, closure={"right": CL}, max_steps=80, judge_cameras=["ego_view"],
                      recovery=RecoveryConfig(3, 4, 1), **kw)


def run(tmp_path, judge=None, condition=None, success_at=None, **kw):
    env = FakeEnv(success_at)
    ex = FakeExecutor(env)
    w = E.EpisodeWriter(tmp_path / "ep", {"task": "Fake", "seed": 42})
    s = run_episode(env, ex, FakeSensor(env), FakeCameras(), w, make_cfg(condition=condition, **kw), 3,
                    judge=judge, gate=Gate(max_retries=1) if judge else None)
    return s, E.EpisodeReader(tmp_path / "ep"), ex


def test_executor_alone_records_everything(tmp_path):
    s, d, ex = run(tmp_path)
    assert s["status"] == "timeout" and s["steps"] == 80 and s["policy_queries"] == 5
    assert [(b["type"], b["step"]) for b in d.boundaries] == [("grasp", 10), ("release", 40)]
    for b in d.boundaries:
        assert d.frame_path("ego_view", b["settled_step"]).exists()
        assert b["settled_step"] > b["step"]
    assert d.meta["instruction"] == "pick the can" and d.meta["summary"]["boundaries"] == 2
    assert len(d.taxel_rows(20)) == 21 and "link.r_palm.pos" in d.proprio[20] and d.proprio[20]["hand_cmd.right"][1] == 1.0
    assert d.proprio[0]["phase"] == "reset" and d.camera("ego_view", 20).width == 256


def test_closed_loop_recovers_then_continues(tmp_path):
    calls = []

    def judge(packet):
        calls.append(packet)
        n = len(calls)
        return Verdict("fail" if n == 1 else "success", "yes", "slip" if n == 1 else "ok", "fake")

    s, d, ex = run(tmp_path, judge=judge, condition="overlay")
    assert s["recoveries"] == 1 and s["judge_calls"] >= 2
    first = d.boundaries[0]
    assert first["gate"] == "recover" and first["outcome"] == "fail" and first["judge"] == "fake"
    phases = [d.proprio[t]["phase"] for t in d.steps()]
    assert phases.count("recovery") == 3 + 4 + 1
    assert ex.interrupts >= 1  # executor resumed after the recovery
    assert d.boundaries[1]["retry"] == 1 and calls[1].meta["condition"] == "overlay"
    assert "Overlay legend" in calls[0].text()


def test_closed_loop_abort_and_stop_on_success(tmp_path):
    env = FakeEnv()
    w = E.EpisodeWriter(tmp_path / "b", {"task": "Fake"})
    s = run_episode(env, FakeExecutor(env), FakeSensor(env), FakeCameras(), w, make_cfg(condition="plain"), 0,
                    judge=lambda p: Verdict("fail", "yes", "x", "fake"), gate=Gate(recovery_enabled=False))
    assert s["status"] == "aborted" and s["boundaries"] == 1
    s2, _, _ = run(tmp_path / "c", success_at=30)
    assert s2["status"] == "success" and s2["steps"] == 30
