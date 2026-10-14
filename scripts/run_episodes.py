"""Run episodes of one task and record them (PLAN.md §3 format). Cluster: sim venv + policy server.

  condition alone    executor alone (Phase 0 sweep / repro, Phase 1 collection, Phase 4 A)
  condition plain|text|overlay  executor + judge + gate + recovery (+ planner)  (Phase 4 P/O, ablations)
  condition oracle   executor + privileged oracle judge (Phase 4 X)

Raw episode directories go to --raw (cluster storage); the small summary (summary.json,
episodes.jsonl, calls.jsonl) goes to --results.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from condex import config as C  # noqa: E402
from condex.episode import EpisodeWriter  # noqa: E402
from condex.gate import Gate  # noqa: E402
from condex.harness import HandKeys, LoopConfig, run_episode  # noqa: E402
from condex.privileged.events import EventConfig  # noqa: E402
from condex.privileged.oracle import OracleJudge  # noqa: E402
from condex.privileged.recorder import Recorder  # noqa: E402
from condex.recovery import RecoveryConfig  # noqa: E402
from condex.render import load_style  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True)
    ap.add_argument("--episodes", type=int, default=20)
    ap.add_argument("--start", type=int, default=0, help="first episode id")
    ap.add_argument("--seed", type=int, default=None, help="base env seed (default: config seed)")
    ap.add_argument("--condition", default="alone", choices=["alone", "plain", "text", "overlay", "oracle"])
    ap.add_argument("--judge", default="vlm", choices=["vlm", "claude"], help="for plain/text/overlay")
    ap.add_argument("--vlm-model", default=None, help="served model name (frozen id or LoRA adapter name)")
    ap.add_argument("--no-planner", action="store_true")
    ap.add_argument("--no-recovery", action="store_true", help="ablation: fail-valid aborts")
    ap.add_argument("--no-prev-frames", action="store_true", help="ablation: judge without previous-boundary frames")
    ap.add_argument("--raw", required=True, help="root for raw episode directories")
    ap.add_argument("--results", required=True)
    ap.add_argument("--config", default=None)
    ap.add_argument("--port", type=int, default=int(os.environ.get("RLDX_PORT", 20100)))
    args = ap.parse_args()

    cfg = C.load(args.config)
    seed = cfg["seed"] if args.seed is None else args.seed
    res = Path(args.results)
    res.mkdir(parents=True, exist_ok=True)
    info = C.run_info(cfg["_path"], task=args.task, condition=args.condition, seed=seed, judge=args.judge,
                      vlm_model=args.vlm_model, executor=cfg["model_path"])
    style = load_style(C.REPO / cfg["judge"]["style"])
    info["style_hash"] = style.hash()

    from condex.sim.env import batch_obs
    from condex.sim.setup import build

    env, executor, sensor, cameras = build(cfg, args.task, seed, args.port)

    closed = args.condition != "alone"
    ledger = judge = planner = None
    paid_judge = args.condition in ("plain", "text", "overlay") and args.judge == "claude"
    if paid_judge or (closed and not args.no_planner):
        from condex.cost import Ledger

        ledger = Ledger(calls_path=res / "calls.jsonl")
    if args.condition in ("plain", "text", "overlay"):
        from condex.judge.backends import ClaudeJudge, VLMJudge

        if args.judge == "claude":
            judge = ClaudeJudge(ledger, cfg["judge"]["claude_model"], cfg["judge"]["claude_effort"])
        else:
            judge = VLMJudge(os.environ["CONDEX_VLM_URL"], args.vlm_model or cfg["judge"]["vlm_model"])
    if closed and not args.no_planner:
        from condex.planner import Planner

        planner = Planner(ledger, cfg["planner"]["model"], cfg["planner"]["effort"])

    loop = LoopConfig(
        hands={h: HandKeys(**k) for h, k in cfg["hands"].items()}, closure=C.closure_configs(cfg),
        max_steps=cfg["max_episode_steps"], stop_on_success=cfg["stop_on_success"],
        condition=None if args.condition == "alone" else args.condition, judge_cameras=cfg["judge"]["cameras"],
        style=style, recovery=RecoveryConfig(**cfg["recovery"]), use_prev_frames=not args.no_prev_frames)
    tag = args.condition + ("" if args.condition in ("alone", "oracle") else f"_{args.judge}")
    tag += "_norec" if args.no_recovery else ""
    tag += "_noprev" if args.no_prev_frames else ""
    raw = Path(args.raw) / args.task / f"{tag}_s{seed}"
    out = []
    with (res / "episodes.jsonl").open("a") as f:
        for ep in range(args.start, args.start + args.episodes):
            writer = EpisodeWriter(raw / f"ep_{ep:04d}", {**info, "episode_id": ep})
            recorder = Recorder(env, sensor, writer, event_cfg=EventConfig(**cfg["events"]))
            j = OracleJudge(recorder, cfg["labels"]["reach_limit_m"]) if args.condition == "oracle" else judge
            gate = Gate(cfg["gate"]["max_retries"], recovery_enabled=not args.no_recovery) if closed else None
            s = run_episode(env, executor, sensor, cameras, writer, loop, ep, recorder=recorder, judge=j, gate=gate,
                            planner=planner, batch_obs=batch_obs)
            s["dir"] = str(writer.root)
            out.append(s)
            f.write(json.dumps(s) + "\n")
            f.flush()
            print(json.dumps(s), flush=True)
    n = len(out)
    summary = {**info, "episodes": n, "success_rate": sum(s["success"] for s in out) / max(1, n),
               "boundaries": sum(s["boundaries"] for s in out), "recoveries": sum(s["recoveries"] for s in out),
               "replans": sum(s["replans"] for s in out), "judge_calls": sum(s["judge_calls"] for s in out),
               "aborted": sum(s["status"] == "aborted" for s in out), "raw_dir": str(raw)}
    if ledger is not None:
        summary["cost"] = ledger.summary()
        summary["cost_per_episode_usd"] = ledger.run_spent / max(1, n)
    (res / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
